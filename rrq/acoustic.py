"""Native Torch Prompt-Singer, checkpoint-compatible without the Fairseq CLI."""
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

from .upstream import ROOT, load_state


def core_module():
    name='_rrq_acoustic_core'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'research/PromptSinger/model/acoustic_core.py')
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
    return sys.modules[name]


def extend_model(model,size,seed=1234):
    old=model.lm_head.weight.shape[0]
    if size<old: raise ValueError('vocabulary may only be appended')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    for parent,name,is_embedding in [(model.global_model.transformer,'wte',True),
                                     (model.local_model.transformer,'wte',True),(model,'lm_head',False)]:
        layer=getattr(parent,name); weight=layer.weight.detach()
        new=nn.Embedding(size,weight.shape[1]) if is_embedding else nn.Linear(weight.shape[1],size,bias=False)
        new=new.to(device=weight.device,dtype=weight.dtype)
        with torch.no_grad():
            new.weight[:old].copy_(weight)
            noise=torch.randn((size-old,weight.shape[1]),generator=generator)*.002
            new.weight[old:].copy_(weight.mean(0)+noise.to(weight))
        setattr(parent,name,new)
    return old


def load_acoustic(checkpoint,vocab_size,device='cpu'):
    data=load_state(Path(checkpoint))
    cfg=data.get('cfg')
    args=cfg['model'] if cfg is not None else vars(data['args'])
    keys=core_module().AudioboxMegabyteConfig.__dataclass_fields__
    values={k:args[k] for k in keys if k in args}
    state=data['model']
    values['n_vocab']=state['lm_head.weight'].shape[0]
    # Checkpoint tensors are the authority; defaults must not silently change architecture.
    model=core_module().AcousticLMT5Configured(core_module().AudioboxMegabyteConfig(**values))
    model.load_state_dict(state,strict=True)
    del data,state
    old=extend_model(model,vocab_size)
    return model.to(device),old,values


def next_patch_context(model,prefix,prompt):
    """Only the final global patch is needed; mathematically identical causal path."""
    if model.global_model.prefix_lm: raise ValueError('standalone generation requires causal checkpoint')
    idx=torch.cat([prefix.new_zeros((prefix.shape[0],3)),prefix],dim=1)
    if idx.shape[1]//3>model.n_ctx: raise ValueError('sequence exceeds checkpoint context')
    emb=model.global_model.transformer.wte(idx).reshape(idx.shape[0],-1,model.ln.out_features)
    emb[:,3:3+model.prompt_latent_length]=model.ln(prompt)
    positions=torch.arange(emb.shape[1],device=idx.device)[None]
    output=model.global_model(emb+model.global_model.transformer.wpe(positions))[:,-1:]
    return model.linear_map(output.reshape(-1,3,model.ln.out_features//3))


def patch_logits(model,context,current):
    local=torch.cat([current.new_zeros((current.shape[0],1)),current[:,:-1]],dim=1)
    return model.lm_head(model.local_model(model.local_model.get_embeding(local)+context))


@torch.inference_mode()
def generate_codes(model,prefix,prompt,vocab,frames):
    """Greedy, exact-duration, target-free acoustic rollout; no teacher forcing.

    Pitch bias comes from score, so generation starts after acoustic_start.
    Length is the provided phoneme clock. No learned stop/duration claim.
    """
    model.eval()
    starts=[vocab.index['aco_'+str(q*1024)] for q in range(3)]
    for q,start in enumerate(starts):
        if any(vocab.index['aco_'+str(q*1024+i)]!=start+i for i in range(1024)):
            raise ValueError('non-contiguous native codebook vocabulary')
    generated=[]
    for t in range(frames):
        context=next_patch_context(model,prefix,prompt)
        current=prefix.new_zeros((1,3))
        for q,start in enumerate(starts):
            logits=patch_logits(model,context,current)[:,q,start:start+1024]
            current[:,q]=logits.argmax(-1)+start
        generated.append(current[0]-current.new_tensor(starts))
        prefix=torch.cat([prefix,current],dim=1)
        if (t+1)%50==0: print(f'generated {t+1}/{frames} frames',flush=True)
    return torch.stack(generated,dim=1).cpu().numpy()
