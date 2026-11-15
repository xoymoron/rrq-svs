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


class BatchCache:
    """Left padding is masked out; real position IDs remain identical to serial AR."""
    def __init__(self,model,prefixes,prompt):
        self.model=model; self.kv=None
        lengths=[p.shape[1]//3+1 for p in prefixes]; width=max(lengths); batch=len(prefixes)
        device=prefixes[0].device
        ids=torch.zeros(batch,width*3,dtype=torch.long,device=device)
        self.valid=torch.zeros(batch,width,dtype=torch.bool,device=device)
        positions=torch.zeros(batch,width,dtype=torch.long,device=device)
        for i,(prefix,length) in enumerate(zip(prefixes,lengths)):
            ids[i,(width-length+1)*3:]=prefix[0]
            self.valid[i,width-length:]=True; positions[i,width-length:]=torch.arange(length,device=device)
        x=model.global_model.transformer.wte(ids).reshape(batch,width,-1)
        encoded=model.ln(prompt)
        for i,length in enumerate(lengths):
            offset=width-length+3; x[i,offset:offset+model.prompt_latent_length]=encoded[0]
        self.position=torch.tensor(lengths,device=device)
        self.context=self._forward(x,positions,True)

    def _forward(self,x,positions,fresh):
        x=x+self.model.global_model.transformer.wpe(positions)
        mask=self.valid[:,None,None,:]
        if fresh: mask=mask & torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).tril()[None,None]
        new=[]
        for i,block in enumerate(self.model.global_model.transformer.h):
            att=block.attn
            q,k,v=att.c_attn(block.ln_1(x)).chunk(3,-1)
            shape=(x.shape[0],x.shape[1],att.n_head,att.n_embd//att.n_head)
            q,k,v=[a.reshape(shape).transpose(1,2) for a in (q,k,v)]
            if not fresh:
                k=torch.cat([self.kv[i][0],k],-2); v=torch.cat([self.kv[i][1],v],-2)
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0)
            x=x+att.c_proj(y.transpose(1,2).reshape_as(x)); x=x+block.mlp(block.ln_2(x)); new.append((k,v))
        self.kv=new; x=self.model.global_model.transformer.ln_f(x)[:,-1]
        return self.model.linear_map(x.reshape(-1,3,x.shape[-1]//3))

    def append(self,patch):
        x=self.model.global_model.transformer.wte(patch).reshape(patch.shape[0],1,-1)
        self.valid=torch.cat([self.valid,torch.ones(patch.shape[0],1,dtype=torch.bool,device=patch.device)],1)
        context=self._forward(x,self.position[:,None],False); self.position+=1
        return context


@torch.inference_mode()
def generate_batch(model,prefixes,prompt,vocab,lengths,coarse=None):
    if not lengths or len(prefixes)!=len(lengths) or any(t<1 for t in lengths): raise ValueError('invalid batch lengths')
    if any(p.shape[0]!=1 or p.shape[1]%3 or p.shape[1]//3+1+max(lengths)>model.n_ctx for p in prefixes):
        raise ValueError('invalid prefix/context length')
    if coarse is not None:
        if len(coarse)!=len(lengths) or any(np.asarray(c).shape!=(t,) or np.asarray(c).dtype.kind not in 'iu'
            or (np.asarray(c)<0).any() or (np.asarray(c)>=1024).any() for c,t in zip(coarse,lengths)):
            raise ValueError('invalid fixed coarse batch')
    starts=[vocab.index[f'aco_{q*1024}'] for q in range(3)]
    if any(vocab.index[f'aco_{i}']!=starts[i//1024]+i%1024 for i in range(3072)): raise ValueError('noncontiguous vocabulary')
    model.eval(); cache=BatchCache(model,prefixes,prompt); context=cache.context
    generated=[]
    for t in range(max(lengths)):
        current=prefixes[0].new_zeros((len(prefixes),3))
        for q,start in enumerate(starts):
            if q==0 and coarse is not None:
                current[:,q]=current.new_tensor([int(c[t]) if t<len(c) else 0 for c in coarse])+start
            else: current[:,q]=patch_logits(model,context,current)[:,q,start:start+1024].argmax(-1)+start
        generated.append(current-current.new_tensor(starts))
        if t+1<max(lengths): context=cache.append(current)
    array=torch.stack(generated,-1).cpu().numpy()
    return [array[i,:,:t] for i,t in enumerate(lengths)]
