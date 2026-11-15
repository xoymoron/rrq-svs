"""Matched, fixed-budget multilingual SVS experiments, separate from smoke tools."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .cached_ar import generate_batch
from .core import requantize
from .data_download import free_space
from .io import sha256, write_json, safe_id
from .multilingual import Vocabulary, read_rows, pack
from .multilingual_run import setup, load_codes, ACOUSTIC
from .pipeline import prepare_audio
from .upstream import load_codec, load_vocoder


ARMS=('baseline','exposure','rrq')
SEEDS=(17,29,43)


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); write_json(tmp,value); tmp.replace(path)


def epoch_rows(rows,seed,epoch):
    result=sorted(rows,key=lambda r:r['id']); random.Random(f'{seed}:{epoch}').shuffle(result)
    return result


def check_splits(rows):
    seen_ids=set(); groups={}; audio={}
    for row in rows:
        if row['id'] in seen_ids: raise ValueError('duplicate sample ID')
        seen_ids.add(row['id'])
        if row['split'] not in ('train','valid','test'): raise ValueError('invalid split')
        for mapping,key in [(groups,row['song_group']),(audio,row['audio_sha256'])]:
            if key in mapping and mapping[key]!=row['split']: raise ValueError('split leakage')
            mapping[key]=row['split']


def arm_codes(clean,predicted,corrected,arm,replace):
    if arm not in ARMS: raise ValueError('invalid arm')
    if clean.shape!=predicted.shape or clean.shape!=corrected.shape: raise ValueError('map shape mismatch')
    if not np.array_equal(predicted[0],corrected[0]): raise ValueError('RRQ changed prefix')
    if arm=='baseline' or not replace: return clean.copy()
    result=corrected.copy() if arm=='rrq' else clean.copy()
    result[0]=predicted[0]; return result


def mixture(seed,epoch,item,rho=.5):
    draw=int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:{item}'.encode()).digest()[:8],'big')/2**64
    return draw<rho


def trainable_parameters(model,stage,old):
    model.eval().requires_grad_(False)
    if stage not in ('warmup','branch'): raise ValueError('unknown training stage')
    prefixes=['linear_map.','lm_head.']
    prefixes += [f'local_model.transformer.h.{i}.' for i in range(len(model.local_model.transformer.h)-2,len(model.local_model.transformer.h))]
    if stage=='warmup':
        prefixes+=['global_model.transformer.wte.']
        prefixes += [f'global_model.transformer.h.{i}.' for i in range(max(0,len(model.global_model.transformer.h)-2),len(model.global_model.transformer.h))]
    names=[]
    for name,param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes): param.requires_grad_(True); names.append(name)
    if stage=='warmup':
        def mask_old(gradient):
            g=gradient.clone(); g[:old]=0; return g
        model.global_model.transformer.wte.weight.register_hook(mask_old)
    return names


def configuration(root):
    root=Path(root); cfg=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    for key in ('bundle','assets','prompt','cache'): cfg[key]=Path(cfg[key])
    return cfg


def init(args):
    if args.root.exists(): raise ValueError('experiment directory already exists')
    rows=read_rows([args.bundle/'primary.jsonl']); check_splits(rows)
    if not all(any(r['split']==s for r in rows) for s in ('train','valid','test')): raise ValueError('empty split')
    cfg=dict(schema=1,bundle=str(args.bundle.resolve()),assets=str(args.assets.resolve()),prompt=str(args.prompt.resolve()),
        cache=str(args.cache.resolve()),seed=17,seeds=list(SEEDS),warmup_epochs=8,branch_epochs=6,
        backbone_lr=2e-5,embedding_lr=1e-3,weight_decay=.01,grad_clip=1.,rho=.5,
        primary_sha256=sha256(args.bundle/'primary.jsonl'),vocab_sha256=sha256(args.bundle/'vocabulary.json'),
        prompt_sha256=sha256(args.prompt),base_sha256=sha256(args.assets/ACOUSTIC),
        codec_sha256=sha256(args.assets/'codec/ckpt_01135000.pth'),
        split_ids={s:sorted(r['id'] for r in rows if r['split']==s) for s in ('train','valid','test')},
        conditioning='score_pitch_gt_phone_duration',common_selection='minimum language-macro validation clean CE',
        branch_selection='fixed final epoch 6, no test-based selection',
        temperature=.8,sampling_seed=20261115,
        generation='upstream temperature 0.8; no top-k; item-keyed common random numbers; identical frozen coarse for all arms',
        asr_repo='openai/whisper-large-v3-turbo',asr_revision='41f01f3fe87f28c78e2fbf8b568835947dd65ed9',
        caveats=['Bounded GTSinger pilot; not full-corpus training','Provided phone alignment; not autonomous score-only or zero-shot',
                 'Corpus pretraining overlap and translated song-title aliases unverified','No human listening or MOS'])
    args.root.mkdir(parents=True); atomic_json(args.root/'protocol.json',cfg)
    print('PROTOCOL_LOCKED',json.dumps({s:len(ids) for s,ids in cfg['split_ids'].items()}),flush=True)


def revise_sampling(args):
    """Preserve failed-greedy pilot; import identical common weights into a new protocol."""
    import shutil
    if args.root.exists(): raise ValueError('revised experiment root already exists')
    source=args.from_root
    cfg=json.loads((source/'protocol.json').read_text())
    if not (source/'warmup/complete.json').exists(): raise ValueError('source common training incomplete')
    cfg.update(temperature=.8,sampling_seed=20261115,
        generation='upstream temperature 0.8; no top-k; item-keyed common random numbers; identical frozen coarse for all arms',
        parent_protocol_sha256=sha256(source/'protocol.json'),parent_common_sha256=sha256(source/'warmup/best.pt'),
        revision_reason='Validation greedy code collapse; restore original Prompt-Singer temperature before any test evaluation')
    args.root.mkdir(parents=True); atomic_json(args.root/'protocol.json',cfg)
    out=args.root/'warmup'; out.mkdir()
    state=torch.load(source/'warmup/best.pt',map_location='cpu',weights_only=True)
    if state['protocol_sha256']!=cfg['parent_protocol_sha256']: raise ValueError('source warmup provenance mismatch')
    state['protocol_sha256']=sha256(args.root/'protocol.json')
    state['imported_source_sha256']=cfg['parent_common_sha256']; torch.save(state,out/'best.pt')
    for name in ('best.json','initial-validation.json','selected-validation.json','status.json'):
        shutil.copyfile(source/'warmup'/name,out/name)
    complete=json.loads((source/'warmup/complete.json').read_text())
    complete.update(protocol_sha256=state['protocol_sha256'],imported_from=str(source.resolve()),
        note='Identical trained parameter tensors, no additional warmup or test-based selection')
    atomic_json(out/'complete.json',complete)
    print('REVISED_SAMPLING_PROTOCOL',str(args.root),flush=True)


def sampling_options(ex,batch):
    return dict(temperature=ex.cfg.get('temperature',0.),sampling_seed=ex.cfg.get('sampling_seed',0),
                sample_ids=[r['id'] for r in batch])


class Experiment:
    def __init__(self,root,device):
        self.root=Path(root); self.cfg=configuration(root); self.device=device
        c=self.cfg
        if sha256(c['bundle']/'primary.jsonl')!=c['primary_sha256'] or sha256(c['bundle']/'vocabulary.json')!=c['vocab_sha256'] or sha256(c['prompt'])!=c['prompt_sha256']:
            raise ValueError('experiment inputs changed after protocol lock')
        if sha256(c['assets']/ACOUSTIC)!=c['base_sha256'] or sha256(c['assets']/'codec/ckpt_01135000.pth')!=c['codec_sha256']:
            raise ValueError('base or codec checkpoint changed after protocol lock')
        self.rows=read_rows([c['bundle']/'primary.jsonl']); check_splits(self.rows)
        if {s:sorted(r['id'] for r in self.rows if r['split']==s) for s in ('train','valid','test')}!=c['split_ids']:
            raise ValueError('split changed')
        self.model,self.vocab,self.old,self.model_cfg,self.prompt=setup(SimpleNamespace(**c,device=device))
        self.model.eval()
        self.codes={}; self.packed={}
        for row in self.rows:
            codes=load_codes(c['cache']/(safe_id(row['id'])+'.npz'),row,c['codec_sha256'])
            self.codes[row['id']]=codes; self.packed[row['id']]=pack(row,self.vocab,codes)
        self.protocol_hash=sha256(self.root/'protocol.json')

    def split(self,name): return [r for r in self.rows if r['split']==name]

    def load(self,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['protocol_sha256']!=self.protocol_hash: raise ValueError('checkpoint protocol mismatch')
        current=dict(self.model.named_parameters())
        with torch.no_grad():
            for name,value in data['parameters'].items():
                if name not in current or current[name].shape!=value.shape: raise ValueError('checkpoint shape/key mismatch')
                current[name].copy_(value)
        return data

    def save(self,path,names,epoch,metrics,optimizer=None):
        parameters={n:p.detach().cpu() for n,p in self.model.named_parameters() if n in names}
        count=sum(p.numel()*p.element_size() for p in parameters.values())
        free_space(self.root,max(count*4,1024**2))
        data=dict(protocol_sha256=self.protocol_hash,parameters=parameters,epoch=epoch,metrics=metrics)
        if optimizer is not None: data['optimizer']=optimizer.state_dict()
        path=Path(path); tmp=path.with_suffix('.tmp'); torch.save(data,tmp); tmp.replace(path)

    def loss(self,row,codes=None,suffix=False,shuffle=False):
        if codes is None and not shuffle: tokens,mask,start=self.packed[row['id']]
        else:
            altered=dict(row)
            if shuffle: altered['phones']=list(reversed(row['phones']))
            tokens,mask,start=pack(altered,self.vocab,self.codes[row['id']] if codes is None else codes)
        mask=mask.clone()
        if suffix: mask[start::3]=False
        tokens=tokens[None].to(self.device); mask=mask[None].to(self.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=self.model(tokens,target_acoustic_mask=mask,text_feature=self.prompt)
            loss=F.cross_entropy(logits[mask].float(),tokens[mask])
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        return loss

    @torch.no_grad()
    def validate(self,shuffle=False,suffix=False):
        values=[]
        for row in self.split('valid'):
            values.append(dict(id=row['id'],language=row['language'],singer=row['singer'],ce=float(self.loss(row,suffix=suffix))))
            if shuffle: values[-1]['reversed_phone_ce']=float(self.loss(row,suffix=suffix,shuffle=True))
        grouped=defaultdict(list)
        for v in values: grouped[v['language']].append(v['ce'])
        means={k:float(np.mean(v)) for k,v in grouped.items()}
        return dict(language_macro_ce=float(np.mean(list(means.values()))),per_language=means,samples=values)


def train(args):
    ex=Experiment(args.root,args.device); cfg=ex.cfg
    warm=args.stage=='warmup'; seed=17 if warm else args.seed
    torch.manual_seed(seed)
    out=args.root/('warmup' if warm else f'{args.arm}-{seed}'); out.mkdir(exist_ok=True)
    if (out/'complete.json').exists(): print('ALREADY_COMPLETE',str(out),flush=True); return
    if not warm:
        ex.load(args.root/'warmup/best.pt')
        index=json.loads((args.root/'targets/index.json').read_text())
        if index['common_sha256']!=sha256(args.root/'warmup/best.pt') or index['protocol_sha256']!=ex.protocol_hash:
            raise ValueError('RRQ cache common checkpoint mismatch')
    names=trainable_parameters(ex.model,args.stage,ex.old)
    common_names=trainable_names_from_checkpoint(args.root/'warmup/best.pt') if not warm else []
    save_names=sorted(set(names)|set(common_names))
    groups=[]
    for n,p in ex.model.named_parameters():
        if p.requires_grad:
            is_emb=n=='global_model.transformer.wte.weight'
            groups.append(dict(params=[p],lr=cfg['embedding_lr'] if is_emb else cfg['backbone_lr'],weight_decay=0. if is_emb else cfg['weight_decay']))
    optimizer=torch.optim.AdamW(groups)
    original_rows=ex.model.global_model.transformer.wte.weight[:ex.old].detach().clone()
    start_epoch=1; best=float('inf')
    if (out/'last.pt').exists():
        last=ex.load(out/'last.pt'); optimizer.load_state_dict(last['optimizer']); start_epoch=last['epoch']+1; metrics=last['metrics']
        if warm: best=recover_best(out,ex.protocol_hash,last)
    elif warm:
        atomic_json(out/'initial-validation.json',ex.validate(shuffle=True))
    epochs=cfg['warmup_epochs'] if warm else cfg['branch_epochs']
    cache={}
    if not warm:
        for row in ex.split('train'):
            with np.load(args.root/'targets'/(safe_id(row['id'])+'.npz'),allow_pickle=False) as f:
                meta=json.loads(str(f['metadata'].item()))
                if meta['audio_sha256']!=row['audio_sha256'] or meta['common_sha256']!=index['common_sha256'] or meta['split']!='train':
                    raise ValueError('training target provenance mismatch')
                maps=[f[k].copy() for k in ('clean','predicted','rrq')]
                if not np.array_equal(maps[0],ex.codes[row['id']]): raise ValueError('clean cache differs')
                cache[row['id']]=maps
    total=time.monotonic()
    for epoch in range(start_epoch,epochs+1):
        schedule=epoch_rows(ex.split('train'),seed,epoch)
        losses=[]; replacements=0; epoch_time=time.monotonic()
        log=out/f'epoch-{epoch:02d}.jsonl'
        with log.open('w',encoding='utf-8') as stream:
            for step,row in enumerate(schedule,1):
                replacement=mixture(seed,epoch,row['id'],cfg['rho'])
                codes=None if warm else arm_codes(*cache[row['id']],args.arm,replacement)
                optimizer.zero_grad(set_to_none=True)
                loss=ex.loss(row,codes,suffix=not warm); loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in ex.model.parameters() if p.requires_grad],cfg['grad_clip'])
                if not torch.isfinite(norm): raise ValueError('nonfinite gradient')
                optimizer.step()
                if not torch.equal(original_rows,ex.model.global_model.transformer.wte.weight[:ex.old]): raise RuntimeError('original input embedding rows changed')
                item=dict(epoch=epoch,step=step,id=row['id'],language=row['language'],loss=float(loss.detach()),grad_norm=float(norm),mixture_draw=replacement)
                stream.write(json.dumps(item)+'\n'); stream.flush(); losses.append(item['loss']); replacements+=replacement
                if step%20==0 or step==len(schedule):
                    status=dict(stage=args.stage,arm=args.arm if not warm else 'common',seed=seed,epoch=epoch,epochs=epochs,
                        step=step,steps=len(schedule),mean_train_ce=float(np.mean(losses)),elapsed_seconds=time.monotonic()-epoch_time)
                    atomic_json(out/'status.json',status); print(json.dumps(status),flush=True)
        metrics=ex.validate(suffix=not warm)
        metrics.update(epoch=epoch,train_ce=float(np.mean(losses)),schedule_sha256=hashlib.sha256('\n'.join(r['id'] for r in schedule).encode()).hexdigest(),
                       mixture_draws=replacements,seconds=time.monotonic()-epoch_time)
        atomic_json(out/f'validation-{epoch:02d}.json',metrics)
        ex.save(out/'last.pt',save_names,epoch,metrics,optimizer)
        if warm and metrics['language_macro_ce']<best:
            best=metrics['language_macro_ce']; ex.save(out/'best.pt',save_names,epoch,metrics)
            atomic_json(out/'best.json',dict(epoch=epoch,value=best))
        print('EPOCH_DONE',json.dumps({k:v for k,v in metrics.items() if k not in ('samples','per_language')}),flush=True)
    if warm:
        ex.load(out/'best.pt'); atomic_json(out/'selected-validation.json',ex.validate(shuffle=True))
    else: ex.save(out/'final.pt',save_names,epochs,metrics)
    atomic_json(out/'complete.json',dict(stage=args.stage,epochs=epochs,seed=seed,trainable_names=names,protocol_sha256=ex.protocol_hash,
        common_sha256=None if warm else index['common_sha256'],seconds=time.monotonic()-total))
    print('TRAIN_COMPLETE',str(out),flush=True)


def trainable_names_from_checkpoint(path):
    return list(torch.load(path,map_location='cpu',weights_only=True)['parameters'])


def recover_best(out,protocol_hash,last=None):
    """Recover a best checkpoint if interruption occurred after last.pt was saved."""
    out=Path(out)
    last=torch.load(out/'last.pt',map_location='cpu',weights_only=True) if last is None else last
    candidates=[last]
    if (out/'best.pt').exists(): candidates.append(torch.load(out/'best.pt',map_location='cpu',weights_only=True))
    if any(c['protocol_sha256']!=protocol_hash for c in candidates): raise ValueError('best checkpoint protocol mismatch')
    chosen=min(candidates,key=lambda c:c['metrics']['language_macro_ce'])
    if chosen is last:
        free_space(out,sum(v.numel()*v.element_size() for v in chosen['parameters'].values()))
        temp=out/'best.tmp'; torch.save({k:v for k,v in chosen.items() if k!='optimizer'},temp); temp.replace(out/'best.pt')
    value=chosen['metrics']['language_macro_ce']
    atomic_json(out/'best.json',dict(epoch=chosen['epoch'],value=value))
    return value


def render_pair_valid(path,frames,expected):
    import soundfile as sf
    if not path.exists() or not path.with_suffix('.npz').exists(): return False
    try:
        info=sf.info(path)
        if (info.frames,info.samplerate,info.channels)!=(frames*480,24000,1): return False
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:
            meta=json.loads(str(f['metadata'].item())); codes=f['codes']
            if codes.shape!=(3,frames) or codes.dtype.kind not in 'iu' or (codes<0).any() or (codes>=1024).any(): return False
        return all(meta.get(k)==v for k,v in expected.items()) and meta.get('waveform_sha256')==sha256(path)
    except (ValueError,KeyError,OSError): return False


def batches_of_rows(rows,size=8):
    ordered=sorted(rows,key=lambda r:(r['n_frames'],r['id']))
    return [ordered[i:i+size] for i in range(0,len(ordered),size)]


def make_targets(args):
    ex=Experiment(args.root,args.device); ex.load(args.root/'warmup/best.pt'); ex.model.requires_grad_(False)
    out=args.root/'targets'; out.mkdir(exist_ok=True); common_hash=sha256(args.root/'warmup/best.pt')
    codec=load_codec(ex.cfg['assets']/'codec/ckpt_01135000.pth',args.device)
    pending=[]
    for row in ex.rows:
        path=out/(safe_id(row['id'])+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as f: meta=json.loads(str(f['metadata'].item()))
            if meta['common_sha256']!=common_hash or meta['audio_sha256']!=row['audio_sha256'] or meta['protocol_sha256']!=ex.protocol_hash:
                raise ValueError('stale target cache')
        else: pending.append(row)
    done=len(ex.rows)-len(pending)
    with torch.inference_mode():
      for batch in batches_of_rows(pending):
        prefixes=[pack(r,ex.vocab)[0][None].to(args.device) for r in batch]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predictions=generate_batch(ex.model,prefixes,ex.prompt,ex.vocab,[r['n_frames'] for r in batch],**sampling_options(ex,batch))
        for row,predicted in zip(batch,predictions):
            path=out/(safe_id(row['id'])+'.npz')
            pcm,_,_=prepare_audio(row['audio'],1.)
            z=codec.encoder(torch.from_numpy(pcm)[None,None].to(args.device))
            clean=codec.quantizer.vq.encode(z,n_q=3)
            if not np.array_equal(clean[:,0].cpu().numpy(),ex.codes[row['id']]): raise ValueError('native codec cache mismatch')
            corrected=requantize(z,torch.from_numpy(predicted[:1])[:,None].to(args.device),codec.quantizer.vq.layers[:3])[:,0].cpu().numpy()
            meta=dict(id=row['id'],split=row['split'],audio_sha256=row['audio_sha256'],common_sha256=common_hash,
                codec_sha256=ex.cfg['codec_sha256'],protocol_sha256=ex.protocol_hash,target_audio_used_for_prediction=False,
                target_audio_used_for_rrq_training_label=True)
            free_space(out,2*1024**2)
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,clean=clean[:,0].cpu().numpy(),predicted=predicted,rrq=corrected,
                latent=z[0].float().cpu().numpy(),metadata=np.array(json.dumps(meta)))
            tmp.replace(path)
            done+=1
            print('TARGETS',done,len(ex.rows),row['language'],row['id'],flush=True)
            atomic_json(out/'status.json',dict(done=done,total=len(ex.rows),last_id=row['id']))
    atomic_json(out/'index.json',dict(common_sha256=common_hash,protocol_sha256=ex.protocol_hash,
        split_ids=ex.cfg['split_ids'],count=len(ex.rows)))
    print('TARGETS_COMPLETE',flush=True)


def render(args):
    import soundfile as sf
    ex=Experiment(args.root,args.device)
    if args.variant=='common': ex.load(args.root/'warmup/best.pt')
    elif args.variant!='initial': ex.load(args.root/f'{args.variant}-{args.seed}/final.pt')
    ex.model.requires_grad_(False)
    label=args.variant if args.variant in ('common','initial') else f'{args.variant}-{args.seed}'
    out=args.root/'renders'/args.split/label; out.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=sha256(args.root/'warmup/best.pt') if args.variant=='common' else (
        sha256(args.root/f'{args.variant}-{args.seed}/final.pt') if args.variant in ARMS else ex.cfg['base_sha256'])
    provenance=dict(protocol_sha256=ex.protocol_hash,checkpoint_sha256=checkpoint_hash,variant=label,
        generation='batched-kv-v2',temperature=ex.cfg.get('temperature',0.))
    vocoder=load_vocoder(ex.cfg['assets']/'vocoder_24k/g_00885000',args.device)
    rows=ex.split(args.split)
    pending=[r for r in rows if not render_pair_valid(out/(r['id']+'.wav'),r['n_frames'],provenance)]
    done=len(rows)-len(pending)
    for batch in batches_of_rows(pending):
        if args.variant=='common':
            generated=[]
            for row in batch:
                with np.load(args.root/'targets'/(row['id']+'.npz'),allow_pickle=False) as f: generated.append(f['predicted'].copy())
        else:
            coarse=None
            if args.variant in ARMS:
                coarse=[]
                for row in batch:
                    with np.load(args.root/'targets'/(row['id']+'.npz'),allow_pickle=False) as f: coarse.append(f['predicted'][0].copy())
            prefixes=[pack(r,ex.vocab)[0][None].to(args.device) for r in batch]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                generated=generate_batch(ex.model,prefixes,ex.prompt,ex.vocab,[r['n_frames'] for r in batch],coarse,**sampling_options(ex,batch))
        for row,codes in zip(batch,generated):
            path=out/(row['id']+'.wav')
            with torch.inference_mode(): wave=vocoder(torch.from_numpy(codes)[None].to(args.device))[0,0].float().cpu().numpy()
            if len(wave)!=row['n_frames']*480 or not np.isfinite(wave).all(): raise ValueError('invalid decoder output')
            temp=path.with_suffix('.tmp.wav'); sf.write(temp,wave,24000,subtype='FLOAT')
            tmp_codes=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp_codes,codes=codes,metadata=np.array(json.dumps(dict(provenance,waveform_sha256=sha256(temp)))))
            tmp_codes.replace(path.with_suffix('.npz')); temp.replace(path)
            done+=1
            print('RENDER',label,args.split,done,len(rows),row['id'],flush=True)
    atomic_json(out/'complete.json',dict(count=len(ex.split(args.split)),variant=label,protocol_sha256=ex.protocol_hash,
        checkpoint_sha256=checkpoint_hash,
        fixed_common_coarse=args.variant in ARMS))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init')
    for name in ('bundle','assets','prompt','cache'): q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('revise-sampling'); q.add_argument('--from-root',type=Path,required=True)
    q=sub.add_parser('train'); q.add_argument('--stage',choices=['warmup','branch'],required=True)
    q.add_argument('--arm',choices=ARMS,default='baseline'); q.add_argument('--seed',type=int,choices=SEEDS,default=17)
    sub.add_parser('targets')
    q=sub.add_parser('render'); q.add_argument('--variant',choices=['initial','common',*ARMS],required=True)
    q.add_argument('--split',choices=['valid','test'],required=True); q.add_argument('--seed',type=int,choices=SEEDS,default=17)
    args=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(17)
    {'init':init,'revise-sampling':revise_sampling,'train':train,'targets':make_targets,'render':render}[args.command](args)


if __name__=='__main__': main()
