"""Exact causal KV cache for the audited Prompt-Singer architecture."""
import numpy as np
import torch
from torch.nn import functional as F

from .acoustic import patch_logits


class GlobalCache:
    def __init__(self,model):
        if model.training or model.global_model.prefix_lm: raise ValueError('requires causal eval model')
        self.model=model; self.kv=None; self.position=0

    def _forward(self,x):
        fresh=self.kv is None
        if self.position+x.shape[1]>self.model.n_ctx: raise ValueError('context exhausted')
        positions=torch.arange(self.position,self.position+x.shape[1],device=x.device)[None]
        x=x+self.model.global_model.transformer.wpe(positions)
        new=[]
        for i,block in enumerate(self.model.global_model.transformer.h):
            att=block.attn; normalized=block.ln_1(x)
            q,k,v=att.c_attn(normalized).chunk(3,dim=-1)
            shape=(x.shape[0],x.shape[1],att.n_head,att.n_embd//att.n_head)
            q,k,v=[a.reshape(shape).transpose(1,2) for a in (q,k,v)]
            if not fresh:
                k=torch.cat([self.kv[i][0],k],dim=-2); v=torch.cat([self.kv[i][1],v],dim=-2)
            y=F.scaled_dot_product_attention(q,k,v,is_causal=fresh,dropout_p=0)
            y=y.transpose(1,2).reshape_as(x)
            x=x+att.c_proj(y); x=x+block.mlp(block.ln_2(x)); new.append((k,v))
        self.kv=new; self.position+=x.shape[1]
        x=self.model.global_model.transformer.ln_f(x)[:,-1]
        return self.model.linear_map(x.reshape(-1,3,x.shape[-1]//3))

    def prefill(self,prefix,prompt):
        idx=torch.cat([prefix.new_zeros((1,3)),prefix],dim=1)
        x=self.model.global_model.transformer.wte(idx).reshape(1,-1,self.model.ln.out_features)
        x[:,3:3+self.model.prompt_latent_length]=self.model.ln(prompt)
        return self._forward(x)

    def append(self,patch):
        if patch.shape!=(1,3) or self.kv is None: raise ValueError('append requires one patch after prefill')
        return self._forward(self.model.global_model.transformer.wte(patch).reshape(1,1,-1))


@torch.inference_mode()
def generate(model,prefix,prompt,vocab,frames,coarse=None):
    if frames<1: raise ValueError('positive frames required')
    if coarse is not None:
        coarse=np.asarray(coarse)
        if coarse.shape!=(frames,) or coarse.dtype.kind not in 'iu' or (coarse<0).any() or (coarse>=1024).any():
            raise ValueError('coarse prefix must be integer [frames] in [0,1024)')
    starts=[vocab.index[f'aco_{q*1024}'] for q in range(3)]
    for q,start in enumerate(starts):
        if any(vocab.index[f'aco_{q*1024+i}']!=start+i for i in range(1024)):
            raise ValueError('non-contiguous native vocabulary')
    model.eval(); cache=GlobalCache(model); context=cache.prefill(prefix,prompt)
    result=[]
    for t in range(frames):
        current=prefix.new_zeros((1,3))
        for q,start in enumerate(starts):
            if q==0 and coarse is not None: current[0,0]=int(coarse[t])+start
            else: current[:,q]=patch_logits(model,context,current)[:,q,start:start+1024].argmax(-1)+start
        result.append(current[0]-current.new_tensor(starts))
        if t+1<frames: context=cache.append(current)
    return torch.stack(result,1).cpu().numpy()
