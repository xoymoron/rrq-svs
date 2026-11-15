"""Two-GPU TokSing-DAC diagnostics: cache, reconstruction, warmup, oracle.

All report samples are validation only; no test-set model selection. Native
RRQ labels may access target latents, the generator may not.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .core import make_arms
from .dac_codec import FrozenDAC
from .distributed import Distributed
from .io import sha256, write_json
from .metrics import audio_metrics, mean_maps
from .toksing_data import read_rows, inventory, load_audio, frame_condition, score_batch, annotation_hash
from .toksing_model import TokSingDAC
from .tracking import token_metrics


def shard(rows, rank, world):
    return rows[rank::world]


def atomic_save(value, path):
    path=Path(path); temp=path.with_suffix('.tmp')
    torch.save(value,temp); temp.replace(path)


def read_cache(root,row,device='cpu'):
    value=torch.load(Path(root)/'cache'/(row['id']+'.pt'),map_location=device,weights_only=True)
    if value['id']!=row['id']:
        raise ValueError('cache ID mismatch')
    if value['annotation_sha256']!=annotation_hash(row):
        raise ValueError('cache annotation mismatch')
    return value


def summary(samples):
    if not samples or len({r['id'] for r in samples})!=len(samples):
        raise ValueError('empty or duplicate evaluation')
    grouped=defaultdict(list)
    for r in samples: grouped[r['language']].append(r)
    languages={lang:{arm:mean_maps([r['arms'][arm] for r in rows])
                     for arm in rows[0]['arms']} for lang,rows in grouped.items()}
    macro={arm:mean_maps([v[arm] for v in languages.values()]) for arm in samples[0]['arms']}
    paired=[]
    if 'rrq_oracle' in samples[0]['arms']:
        for r in samples:
            a,b=r['arms']['rrq_oracle'],r['arms']['exposure']
            paired.append(dict(id=r['id'],language=r['language'],delta={
                k:a[k]-b[k] if a[k] is not None and b.get(k) is not None else None
                for k in a if isinstance(a[k],(int,float)) or a[k] is None}))
    return dict(n=len(samples),per_language_counts={k:len(v) for k,v in grouped.items()},
                per_language=languages,language_macro=macro,paired_rrq_minus_exposure=paired,
                samples=sorted(samples,key=lambda r:r['id']))


def tracker(root,stage,config):
    if not os.environ.get('WANDB_PROJECT'): return None
    import wandb
    run=wandb.init(project=os.environ['WANDB_PROJECT'],entity=os.environ.get('WANDB_ENTITY'),
                   name=root.name+'/'+stage,group=root.name,job_type=stage,config=config,
                   dir=str(root),save_code=False)
    write_json(root/(stage+'-wandb.json'),dict(url=run.url,id=run.id))
    return run


def publish(run,report,root,stage):
    if run is None:return
    import wandb
    for arm,metrics in report['language_macro'].items():
        run.log({stage+'/'+arm+'/'+k:v for k,v in metrics.items() if v is not None})
    cols=['id','language','arm','latent_mse','mrstft_spectral_convergence','f0_rmse_cents','voicing_error']
    table=wandb.Table(columns=cols)
    for r in report['samples']:
        for arm,m in r['arms'].items():table.add_data(r['id'],r['language'],arm,*[m.get(k) for k in cols[3:]])
    run.log({stage+'/samples':table})
    # Fixed first ID per language, sorted before looking at metrics.
    for lang in sorted(report['per_language']):
        r=next(r for r in report['samples'] if r['language']==lang)
        for arm in ['reference',*r['arms']]:
            path=root/'audio'/stage/(r['id']+'-'+arm+'.wav')
            if path.exists(): run.log({stage+'/audio/'+lang+'/'+arm:wandb.Audio(str(path),caption=r['id'])})
    run.finish()


def cache_data(args,d,rows,device):
    root=Path(args.output); (root/'cache').mkdir(exist_ok=True)
    codec=FrozenDAC(args.codec,device)
    errors=[]
    for row in tqdm(shard(rows,d.rank,d.world),desc=f'codec cache rank{d.rank}'):
        path=root/'cache'/(row['id']+'.pt')
        source=Path(row['audio']); source_hash=sha256(source)
        if path.exists():
            cached=read_cache(root,row)
            if cached['codec_sha256']!=codec.fingerprint or cached['source_sha256']!=source_hash:
                raise ValueError('stale cache: '+row['id'])
            continue
        try:
            x=load_audio(source,codec.sample_rate)
            z,codes=codec.encode(torch.from_numpy(x)[None,None].to(device))
            cond=frame_condition(row,z.shape[-1],len(x),codec.sample_rate,codec.hop)
            atomic_save(dict(id=row['id'],z=z.cpu() if row['split']=='valid' else torch.empty(0),codes=codes.cpu(),
                frame_index=torch.from_numpy(cond['phone_index']),samples=len(x),
                sample_rate=codec.sample_rate,hop=codec.hop,codec_sha256=codec.fingerprint,
                source_sha256=source_hash,annotation_sha256=annotation_hash(row)),path)
        except ValueError as exc:
            errors.append(dict(id=row['id'],reason=str(exc)))
    all_errors=d.gather(errors)
    if d.primary: write_json(root/'cache-rejections.json',all_errors)
    if all_errors: raise ValueError(f'{len(all_errors)} invalid records; explicit manifest review required')


def evaluate(args,d,rows,device,oracle=False):
    root=Path(args.output); stage='oracle' if oracle else 'reconstruction'
    codec=FrozenDAC(args.codec,device)
    vocab=json.loads((root/'vocabulary.json').read_text())
    model=None; checkpoint_hash=None
    if oracle:
        checkpoint=root/'best.pt'; value=torch.load(checkpoint,map_location=device,weights_only=True)
        if value['manifest_sha256']!=sha256(args.manifest):raise ValueError('model/manifest mismatch')
        if value['codec_sha256']!=codec.fingerprint:raise ValueError('model/codec mismatch')
        checkpoint_hash=sha256(checkpoint)
        model=TokSingDAC(**value['model_config']).to(device)
        model.load_state_dict(value['model']);model.eval()
    selected=[r for r in rows if r['split']=='valid']
    folder=root/'audio'/stage;folder.mkdir(parents=True,exist_ok=True)
    samples=[]
    for row in tqdm(shard(selected,d.rank,d.world),desc=stage+f' rank{d.rank}'):
        c=read_cache(root,row,device)
        if c['codec_sha256']!=codec.fingerprint:raise ValueError('wrong codec checkpoint')
        x=load_audio(row['audio'],codec.sample_rate)
        arms={'codec':c['codes']}
        if oracle:
            with torch.inference_mode():
                prediction=model.generate(**score_batch(row,c,vocab,device)).permute(1,0,2)
                arms=make_arms(c['z'],c['codes'],prediction,codec.layers)
            atomic_save(dict(id=row['id'],codes=prediction.cpu(),origin='model_rollout',
                model_sha256=checkpoint_hash,codec_sha256=codec.fingerprint,
                conditioning='score_pitch_gt_phone_duration',decoding='greedy'),
                root/'predictions'/(row['id']+'.pt'))
        result={}
        sf.write(folder/(row['id']+'-reference.wav'),x,codec.sample_rate,subtype='FLOAT')
        for arm,tokens in arms.items():
            y=codec.decode(tokens,c['samples'])[0,0].cpu().numpy()
            zhat=codec.codec.quantizer.from_codes(tokens.permute(1,0,2))[0]
            result[arm]=audio_metrics(x,y,codec.sample_rate)
            result[arm]['latent_mse']=float(F.mse_loss(zhat,c['z']))
            sf.write(folder/(row['id']+'-'+arm+'.wav'),y,codec.sample_rate,subtype='FLOAT')
        item=dict(id=row['id'],singer=row['singer'],language=row['language'],arms=result)
        if oracle:
            item.update(coarse_match=float((prediction[0]==c['codes'][0]).float().mean()),
                suffix_changed=float((arms['rrq_oracle'][1:]!=c['codes'][1:]).float().mean()),
                token_metrics=token_metrics(prediction[:,0].cpu().numpy()))
        samples.append(item)
    samples=d.gather(samples)
    if {r['id'] for r in samples}!={r['id'] for r in selected}:raise ValueError('inexact validation coverage')
    if d.primary:
        report=summary(samples)
        report.update(stage=stage,codec_sha256=codec.fingerprint,model_sha256=checkpoint_hash,
            manifest_sha256=sha256(args.manifest),conditioning='score_pitch_gt_phone_duration',
            note='Oracle accesses target audio; not learned RRQ performance nor a global upper bound.')
        write_json(root/(stage+'.json'),report)
        print(json.dumps(report['language_macro']),flush=True)
        publish(tracker(root,stage,dict(codec_sha256=codec.fingerprint)),report,root,stage)


def train(args,d,rows,device):
    root=Path(args.output);vocab=json.loads((root/'vocabulary.json').read_text())
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    config=dict(phones=len(vocab['phones']),singers=len(vocab['singers']),languages=len(vocab['languages']),
                dim=args.dim,layers=args.layers,codebooks=9,entries=1024)
    model=TokSingDAC(**config).to(device)
    wrapped=d.wrap(model);optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    train_rows=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='valid']
    if not train_rows or not valid:raise ValueError('require train and validation records')
    groups=defaultdict(list)
    for r in train_rows:groups[r['language']].append(r)
    first=1;best=float('inf');step=0
    last=root/'last.pt'
    if last.exists():
        value=torch.load(last,map_location=device,weights_only=True)
        if (value['manifest_sha256']!=sha256(args.manifest) or value['model_config']!=config
                or value['codec_sha256']!=sha256(args.codec)):
            raise ValueError('resume protocol mismatch')
        model.load_state_dict(value['model']);optim.load_state_dict(value['optim'])
        first=value['epoch']+1;best=value['best'];step=value['step']
    run=tracker(root,'warmup',dict(model=config,parameters=sum(p.numel() for p in model.parameters()),
        manifest_sha256=sha256(args.manifest),world=d.world,seed=args.seed,
        language_sampling='uniform with replacement',conditioning='score_pitch_gt_phone_duration')) if d.primary else None
    epochs=tqdm(range(first,args.epochs+1),desc='epochs',disable=not d.primary)
    for epoch in epochs:
        start=time.time();model.train();rng=random.Random(args.seed+epoch)
        languages=sorted(groups);count=int(np.ceil(len(train_rows)/d.world))*d.world
        draw=[rng.choice(groups[rng.choice(languages)]) for _ in range(count)]
        local=shard(draw,d.rank,d.world);losses=[]
        for row in tqdm(local,desc=f'epoch {epoch}',disable=not d.primary,leave=False):
            c=read_cache(root,row,device);batch=score_batch(row,c,vocab,device)
            target=c['codes'].permute(1,0,2)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=wrapped(**batch,codes=target)
                loss=F.cross_entropy(logits.flatten(0,2),target.flatten())
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step();losses.append(float(loss));step+=1
            if run and step%20==0:run.log({'train/ce':float(loss),'train/step':step,'train/grad_norm':float(norm)})
        model.eval();metrics=[]
        with torch.inference_mode():
            for row in shard(valid,d.rank,d.world):
                c=read_cache(root,row,device);batch=score_batch(row,c,vocab,device);target=c['codes'].permute(1,0,2)
                logits=model(**batch,codes=target)
                ce=float(F.cross_entropy(logits.flatten(0,2),target.flatten()))
                coarse=float((logits[:,0].argmax(-1)==target[:,0]).float().mean())
                reverse=dict(batch,phones=batch['phones'].flip(1))
                rev=model(**reverse,codes=target)
                gap=float(F.cross_entropy(rev.flatten(0,2),target.flatten()))-ce
                metrics.append(dict(id=row['id'],language=row['language'],ce=ce,coarse_accuracy=coarse,phone_reverse_gap=gap))
        metrics=d.gather(metrics);train_mean=float(np.mean(d.gather(losses)))
        if len(metrics)!=len(valid) or len({r['id'] for r in metrics})!=len(valid):raise ValueError('validation coverage')
        lang={k:mean_maps([{p:r[p] for p in ('ce','coarse_accuracy','phone_reverse_gap')} for r in metrics if r['language']==k]) for k in sorted({r['language'] for r in metrics})}
        score=float(np.mean([v['ce'] for v in lang.values()]));improved=score<best;best=min(best,score)
        synced=d.assert_synced(model)
        if d.primary:
            state=dict(model=model.state_dict(),optim=optim.state_dict(),model_config=config,
                epoch=epoch,best=best,step=step,manifest_sha256=sha256(args.manifest),
                codec_sha256=sha256(args.codec))
            atomic_save(state,last)
            if improved:atomic_save(state,root/'best.pt')
            record=dict(epoch=epoch,train_ce=train_mean,valid_ce=score,per_language=lang,
                        seconds=time.time()-start,parameter_sha256=synced,samples=metrics)
            write_json(root/f'epoch-{epoch:03d}.json',record)
            print(json.dumps({k:v for k,v in record.items() if k not in ('samples','per_language')}),flush=True)
            if run:
                values={'epoch':epoch,'train/epoch_ce':train_mean,'valid/language_macro_ce':score}
                values.update({f'valid/{l}/{k}':v for l,m in lang.items() for k,v in m.items()})
                run.log(values)
        d.barrier()
    if run:run.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['cache','reconstruction','train','oracle'])
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--codec',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--dim',type=int,default=384);p.add_argument('--layers',type=int,default=6)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=17)
    args=p.parse_args();torch.set_num_threads(2)
    d=Distributed();d.start();device=torch.device(f'cuda:{d.local_rank}' if torch.cuda.is_available() else 'cpu')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rows=read_rows(args.manifest)
    # Verify immutable cache inputs once per stage, not at every training step.
    if args.stage!='cache':
        codec_hash=sha256(args.codec)
        for row in shard(rows,d.rank,d.world):
            cached=read_cache(root,row)
            if cached['codec_sha256']!=codec_hash or cached['source_sha256']!=sha256(row['audio']):
                raise ValueError('stale codec/audio cache: '+row['id'])
    if d.primary:
        vocab=inventory(rows);path=root/'vocabulary.json'
        if path.exists() and json.loads(path.read_text())!=vocab:raise ValueError('inventory mismatch')
        write_json(path,vocab);(root/'predictions').mkdir(exist_ok=True)
    d.barrier()
    try:
        if args.stage=='cache':cache_data(args,d,rows,device)
        elif args.stage=='train':train(args,d,rows,device)
        else:evaluate(args,d,rows,device,args.stage=='oracle')
    finally:d.close()


if __name__=='__main__':main()
