"""Visible, reproducible multilingual adaptation and actual acoustic rollout.

Default training updates new global input embedding rows only. It is a warmup,
not a completed multilingual SVS or a zero-shot singer cloning implementation.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .acoustic import load_acoustic, generate_codes
from .io import sha256, write_json, save_prediction, safe_id, validate_codes
from .multilingual import Vocabulary, read_rows, pack
from .pipeline import prepare_audio
from .upstream import load_codec, load_vocoder
from .data_download import free_space


ACOUSTIC='prompt-singer-flant5-large-finetuned/checkpoint_last.pt'
T5='flant5-large-finetuned/checkpoint-40000'


def balanced_rows(rows,split,smoke=False,seed=1234):
    groups=defaultdict(list)
    for row in rows:
        if row['split']==split: groups[row['language']].append(row)
    rng=random.Random(seed)
    for group in groups.values():
        if smoke: group.sort(key=lambda r:(r['n_frames'],r['id']))
        else: rng.shuffle(group)
    if smoke: return [groups[k][0] for k in sorted(groups)]
    result=[]
    while any(groups.values()):
        for k in sorted(groups):
            if groups[k]: result.append(groups[k].pop())
    return result


def prompt_cache(args):
    from transformers import T5Tokenizer, T5EncoderModel
    root=args.assets/T5
    out=args.output
    if out.exists(): raise ValueError('prompt cache exists; do not overwrite provenance')
    tokenizer=T5Tokenizer.from_pretrained(root,local_files_only=True)
    model=T5EncoderModel.from_pretrained(root,local_files_only=True).eval().requires_grad_(False).to(args.device)
    tokens=tokenizer('A singer sings.',truncation=True,max_length=77,padding='max_length',return_tensors='pt')['input_ids'].to(args.device)
    # Match upstream: padding tokens are included, no attention_mask is passed.
    with torch.inference_mode(): feature=model(input_ids=tokens).last_hidden_state.cpu()
    if feature.shape!=(1,77,1024) or not torch.isfinite(feature).all(): raise ValueError('invalid T5 feature')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(feature=feature,prompt='A singer sings.',
                    checkpoint_sha256=sha256(root/'pytorch_model.bin'),upstream_padding_policy=True),out)
    print('PROMPT_CACHE_OK',list(feature.shape),flush=True)


def codec_cache(args):
    rows=read_rows([args.manifest]); root=args.output
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=args.assets/'codec/ckpt_01135000.pth'
    codec=load_codec(checkpoint,args.device); digest=sha256(checkpoint)
    for i,row in enumerate(rows):
        dest=root/(safe_id(row['id'])+'.npz')
        audio_hash=sha256(row['audio'])
        if dest.exists():
            load_codes(dest,row,codec_sha256=digest)
        else:
            pcm,_,_=prepare_audio(row['audio'],1.)
            with torch.inference_mode():
                z=codec.encoder(torch.from_numpy(pcm)[None,None].to(args.device))
                codes=codec.quantizer.vq.encode(z,n_q=3)[:,0].cpu().numpy()
            if codes.shape!=(3,row['n_frames']): raise ValueError('codec frame count mismatch')
            free_space(root,1024**2)
            np.savez_compressed(dest,codes=codes,metadata=np.array(json.dumps(dict(id=row['id'],
                target_audio_sha256=audio_hash,codec_sha256=digest,frame_rate=50))))
        if (i+1)%25==0 or i+1==len(rows): print(f'CODEC_CACHE {i+1}/{len(rows)}',flush=True)
    write_json(root/'report.json',dict(items=len(rows),manifest_sha256=sha256(args.manifest),codec_sha256=digest))


def load_codes(path,row,codec_sha256=None):
    with np.load(path,allow_pickle=False) as f:
        codes=validate_codes(f['codes']).copy(); meta=json.loads(str(f['metadata'].item()))
    if meta['id']!=row['id'] or meta['target_audio_sha256']!=sha256(row['audio']) or meta['frame_rate']!=50:
        raise ValueError('stale codec cache')
    if codec_sha256 is not None and meta['codec_sha256']!=codec_sha256: raise ValueError('codec checkpoint changed')
    if codes.shape!=(3,row['n_frames']): raise ValueError('codec cache frame mismatch')
    return codes


def setup(args):
    vocab=Vocabulary.load(args.bundle/'vocabulary.json')
    info=json.loads((args.bundle/'vocabulary.json').read_text())
    model,old,cfg=load_acoustic(args.assets/ACOUSTIC,len(vocab.tokens),args.device)
    if old!=info['base_size']: raise ValueError('checkpoint/dictionary base vocabulary mismatch')
    if model.global_model.prefix_lm: raise ValueError('expected published causal checkpoint')
    prompt_data=torch.load(args.prompt,map_location='cpu',weights_only=True)
    if prompt_data['checkpoint_sha256']!=sha256(args.assets/T5/'pytorch_model.bin') or prompt_data['prompt']!='A singer sings.' or not prompt_data.get('upstream_padding_policy'):
        raise ValueError('prompt feature provenance differs from official fixed prompt')
    prompt=prompt_data['feature'].to(args.device)
    if prompt.shape!=(1,77,cfg['text_enc_dim']) or not torch.isfinite(prompt).all(): raise ValueError('invalid prompt feature')
    return model,vocab,old,cfg,prompt


def loss_for(model,row,vocab,prompt,cache,device,codec_sha256,rrq_cache=None):
    if rrq_cache is None: codes=load_codes(cache/(safe_id(row['id'])+'.npz'),row,codec_sha256)
    else: codes=rrq_cache.get(row['id'],row['audio'],row['n_frames'])
    tokens,mask,_=pack(row,vocab,codes)
    if rrq_cache is not None:  # matched suffix-only objective for all three RRQ arms
        suffix_ids=torch.tensor([vocab.index['aco_'+str(k)] for k in range(1024,3072)])
        mask &= torch.isin(tokens,suffix_ids)
    if len(tokens)//3>model.n_ctx: raise ValueError('training example exceeds context')
    tokens=tokens[None].to(device); mask=mask[None].to(device)
    with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
        logits,_=model(tokens,target_acoustic_mask=mask,text_feature=prompt)
        loss=F.cross_entropy(logits[mask].float(),tokens[mask])
    if not torch.isfinite(loss): raise ValueError('nonfinite acoustic loss')
    return loss


def train(args):
    if args.output.exists(): raise ValueError('run output already exists')
    rows=read_rows([args.bundle/'primary.jsonl'])
    schedule=balanced_rows(rows,'train',args.smoke,args.seed)
    validation=balanced_rows(rows,'valid',True,args.seed)
    if not schedule: raise ValueError('no primary training examples')
    model,vocab,old,cfg,prompt=setup(args)
    codec_hash=sha256(args.assets/'codec/ckpt_01135000.pth')
    model.eval().requires_grad_(False)
    weight=model.global_model.transformer.wte.weight
    weight.requires_grad_(True)
    old_rows=weight[:old].detach().clone()
    def appended_only(grad):
        grad=grad.clone(); grad[:old]=0; return grad
    hook=weight.register_hook(appended_only)
    opt=torch.optim.AdamW([weight],lr=args.lr,weight_decay=0)
    rrq_cache=None
    if args.rrq_cache:
        from .training import CachedTargets
        rrq_cache=CachedTargets(dict(cache_dir=str(args.rrq_cache),arm=args.arm,rho=args.rho,seed=args.seed,
                                    expected_codec_sha256=codec_hash))
    args.output.mkdir(parents=True)
    records=[]
    print('TRAIN_START',json.dumps(dict(steps=args.steps,languages=sorted({r['language'] for r in schedule}),
        examples=len(schedule),smoke=args.smoke,trainable='new global embedding rows',old_rows=old)),flush=True)
    start=time.monotonic()
    with (args.output/'steps.jsonl').open('w',encoding='utf-8') as log:
        for step in range(args.steps):
            row=schedule[step%len(schedule)]; opt.zero_grad(set_to_none=True)
            loss=loss_for(model,row,vocab,prompt,args.cache,args.device,codec_hash,rrq_cache)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([weight],1.)
            if not torch.isfinite(norm): raise ValueError('nonfinite gradient')
            opt.step()
            if not torch.equal(old_rows,weight[:old]): raise RuntimeError('pretrained rows changed')
            item=dict(step=step+1,id=row['id'],language=row['language'],loss=float(loss.detach()),grad_norm=float(norm),
                      seconds=time.monotonic()-start)
            log.write(json.dumps(item)+'\n'); log.flush(); print(json.dumps(item),flush=True); records.append(item)
    hook.remove()
    metrics=[]
    with torch.no_grad():
        for row in validation:
            # Plain clean CE is only a runtime check, not an RRQ comparison.
            value=float(loss_for(model,row,vocab,prompt,args.cache,args.device,codec_hash))
            metrics.append(dict(id=row['id'],language=row['language'],clean_ce=value))
            print('VALID_SMOKE',json.dumps(metrics[-1]),flush=True)
    base_hash=sha256(args.assets/ACOUSTIC)
    torch.save(dict(new_global_rows=weight[old:].detach().cpu(),base_size=old,
        vocabulary_sha256=sha256(args.bundle/'vocabulary.json'),base_checkpoint_sha256=base_hash,
        prompt_sha256=sha256(args.prompt),codec_sha256=codec_hash,
        steps=args.steps,config=cfg,smoke=args.smoke),args.output/'adapter.pt')
    write_json(args.output/'report.json',dict(steps=args.steps,train_examples=len(schedule),
        languages=sorted({r['language'] for r in schedule}),validation_smoke=metrics,
        old_embedding_rows_unchanged=True,only_new_global_rows_trained=True,
        is_quality_result=False,smoke=args.smoke,wall_seconds=time.monotonic()-start,
        objective='suffix_ce_'+args.arm if args.rrq_cache else 'all_codebook_ce',
        caveat='Runtime adaptation check; no before/after quality or zero-shot claim'))
    print('ADAPTATION_COMPLETE',str(args.output),flush=True)


def validate_adapter(state,old,vocabulary_sha256,base_checkpoint_sha256,prompt_sha256):
    if state['vocabulary_sha256']!=vocabulary_sha256 or state['base_size']!=old or state['base_checkpoint_sha256']!=base_checkpoint_sha256:
        raise ValueError('adapter provenance mismatch')
    if state.get('prompt_sha256')!=prompt_sha256:
        raise ValueError('adapter prompt provenance mismatch')


def rollout(args):
    if args.output.exists(): raise ValueError('rollout output already exists')
    rows=balanced_rows(read_rows([args.bundle/'primary.jsonl']),args.split,True,args.seed)
    if not rows: raise ValueError('no rollout examples')
    model,vocab,old,cfg,prompt=setup(args)
    base_hash=sha256(args.assets/ACOUSTIC)
    if args.adapter:
        state=torch.load(args.adapter,map_location='cpu',weights_only=True)
        validate_adapter(state,old,sha256(args.bundle/'vocabulary.json'),base_hash,sha256(args.prompt))
        with torch.no_grad(): model.global_model.transformer.wte.weight[old:].copy_(state['new_global_rows'])
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True)
    manifest=[]; generated=[]
    for row in rows:
        prefix,_,_=pack(row,vocab)
        frames=row['n_frames'] if args.max_frames is None else min(args.max_frames,row['n_frames'])
        with torch.autocast(device_type=torch.device(args.device).type,dtype=torch.bfloat16,enabled=str(args.device).startswith('cuda')):
            codes=generate_codes(model,prefix[None].to(args.device),prompt,vocab,frames)
        dest=args.output/(safe_id(row['id'])+'.npz')
        save_prediction(dest,codes,dict(id=row['id'],origin='model_rollout',frame_rate=50,
            generator='Prompt-Singer multilingual append-only adapter',base_checkpoint_sha256=base_hash,
            prompt_sha256=sha256(args.prompt),vocabulary_sha256=sha256(args.bundle/'vocabulary.json'),
            adapter_sha256=sha256(args.adapter) if args.adapter else None,
            target_audio_used_for_prediction=False,conditioning='score_pitch_gt_phone_duration',
            sampling='greedy exact annotated duration',full_phrase=frames==row['n_frames']))
        if frames==row['n_frames']:
            manifest.append(dict(id=row['id'],singer=row['singer'],audio=row['audio'],prediction=str(dest.resolve()),
                gain=1.,origin='model_rollout',alignment='same_50hz_grid',conditioning='annotation_conditioned'))
        generated.append(dict(id=row['id'],language=row['language'],frames=frames,full_phrase=frames==row['n_frames']))
        print('ROLLOUT',json.dumps(generated[-1]),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    vocoder=load_vocoder(args.assets/'vocoder_24k/g_00885000',args.device)
    with torch.inference_mode():
        for row in generated:
            with np.load(args.output/(row['id']+'.npz'),allow_pickle=False) as f: codes=f['codes'].copy()
            wave=vocoder(torch.from_numpy(codes)[None].to(args.device))[0,0].float().cpu().numpy()
            if len(wave)!=row['frames']*480 or not np.isfinite(wave).all(): raise ValueError('invalid decoder output')
            sf.write(args.output/(row['id']+'.wav'),wave,24000,subtype='FLOAT')
    (args.output/'diagnostic-manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest),encoding='utf-8')
    write_json(args.output/'report.json',dict(samples=generated,target_audio_used_for_prediction=False,
        is_quality_result=False,decoder='native CodeBigVGAN 24kHz',
        caveat='Annotation-conditioned runtime check; new phones need adequate training'))
    print('ROLLOUT_AND_DECODE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--assets',type=Path,default=Path('artifacts/pretrained'))
    sub=p.add_subparsers(dest='command',required=True)
    prompt=sub.add_parser('prompt'); prompt.add_argument('--output',type=Path,required=True)
    cache=sub.add_parser('cache'); cache.add_argument('--manifest',type=Path,required=True); cache.add_argument('--output',type=Path,required=True)
    for name in ('train','rollout'):
        parser=sub.add_parser(name)
        parser.add_argument('--bundle',type=Path,required=True); parser.add_argument('--prompt',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True); parser.add_argument('--seed',type=int,default=1234)
        if name=='train':
            parser.add_argument('--cache',type=Path,required=True); parser.add_argument('--steps',type=int,default=32)
            parser.add_argument('--lr',type=float,default=.001); parser.add_argument('--smoke',action='store_true')
            parser.add_argument('--rrq-cache',type=Path); parser.add_argument('--arm',choices=['clean','exposure','rrq'],default='clean')
            parser.add_argument('--rho',type=float,default=.5)
        else:
            parser.add_argument('--adapter',type=Path); parser.add_argument('--max-frames',type=int)
            parser.add_argument('--split',choices=['train','valid','test'],default='valid')
    args=p.parse_args()
    if getattr(args,'steps',1)<1 or (getattr(args,'max_frames',None) is not None and args.max_frames<1): p.error('positive steps/frames required')
    torch.manual_seed(getattr(args,'seed',1234)); torch.set_num_threads(2)
    dict(prompt=prompt_cache,cache=codec_cache,train=train,rollout=rollout)[args.command](args)


if __name__=='__main__': main()
