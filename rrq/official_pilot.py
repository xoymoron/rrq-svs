"""Bounded original-code baseline audit. Never launches the full experiment.

CLI stages are explicit: prepare, profile, train, targets. The first train is a
memorization/held-out diagnostic on a declared small panel, not a paper result.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .core import requantize
from .distributed import Distributed
from .encodec_codec import FrozenEnCodec
from .io import sha256, write_json
from .metrics import audio_metrics, mean_maps
from .official_data import duration_counts, score_inputs, world_f0, continuous_lf0, pitch_metrics, synthesis_gate
from .official_toksing import OfficialTokSing, upstream_config, source_provenance
from .targets import training_pair, preserve_clean_frames
from .toksing_data import inventory, load_audio, read_rows


def atomic_torch(path, data):
    path = Path(path); temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(data, temporary); temporary.replace(path)


def select_panel(rows, split, per_language):
    selected=[]
    for language in sorted({r['language'] for r in rows}):
        candidates=[r for r in rows if r['split']==split and r['language']==language and 2<=r['audio_seconds']<=6]
        by_singer=defaultdict(list)
        for row in sorted(candidates,key=lambda r:r['id']):
            by_singer[row['singer']].append(row)
        ordered=[]
        while any(by_singer.values()):
            for singer in sorted(by_singer):
                if by_singer[singer]: ordered.append(by_singer[singer].pop(0))
        if len(ordered)<per_language: raise ValueError('insufficient short phrases for '+language)
        selected.extend(ordered[:per_language])
    return selected


def prepare(args):
    root=Path(args.output)
    if (root/'pilot.json').exists(): raise RuntimeError('pilot panel already frozen; do not overwrite')
    root.mkdir(parents=True,exist_ok=True)
    rows=read_rows(Path(args.prepared)/'manifest.jsonl')
    train=select_panel(rows,'train',4); valid=select_panel(rows,'valid',2)
    vocab=inventory([r for r in rows if r['split']=='train'])
    codec=FrozenEnCodec('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    selected=[]; audit=[]
    for row in tqdm(train+valid,desc='EnCodec pilot preparation'):
        if sha256(row['audio'])!=row['source_sha256']: raise ValueError('source changed')
        identity=dict(source=row['source_sha256'],codec=codec.fingerprint,sr=24000,bandwidth=6.,
                      f0='pyworld-0.3.5-dio-stonemask-60-1500-hop320-continuous-lf0-v2',schema=2)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=cache/(key+'.pt')
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=False)
            if value['identity']!=identity: raise ValueError('acoustic cache identity mismatch')
        else:
            x=load_audio(row['audio'],codec.sample_rate)
            z,codes=codec.encode(torch.from_numpy(x)[None,None].to(next(codec.parameters()).device))
            native=codec.codec.encode(torch.from_numpy(x)[None,None].to(z.device))
            if len(native)!=1 or native[0][1] is not None or not torch.equal(native[0][0],codes.permute(1,0,2)):
                raise ValueError('wrapper tokens differ from native EnCodec.encode')
            recoded=requantize(z,codes[:1],codec.layers)
            if not torch.equal(recoded,codes): raise ValueError('clean-prefix re-quantization is not native-token exact')
            decoded=codec.decode(codes,len(x))[0,0].cpu().numpy()
            upstream=codec.codec.decode(native)[0,0,:len(x)].cpu().numpy()
            if not np.array_equal(decoded,upstream): raise ValueError('native decoder mismatch')
            f0=world_f0(x,codec.sample_rate)
            value=dict(identity=identity,z=z.cpu(),codes=codes.cpu(),audio=x,f0=f0[:codes.shape[-1]],
                       lf0=continuous_lf0(f0,codes.shape[-1]),
                       codec_audio=decoded,samples=len(x),frames=codes.shape[-1])
            atomic_torch(path,value)
        ds=duration_counts(row['phone_durations'],value['frames'],value['samples'],24000,320)
        if ds.max()>=512: raise ValueError('official duration embedding range exceeded')
        selected.append(dict(row=row,cache=str(path.resolve()),cache_sha256=sha256(path),counts=ds.tolist()))
        generated_f0=world_f0(value['codec_audio'],24000,frames=value['frames'])
        metrics=audio_metrics(value['audio'],value['codec_audio'],24000)
        metrics.update(pitch_metrics(value['f0'],generated_f0))
        audit.append(dict(id=row['id'],language=row['language'],split=row['split'],metrics=metrics))
    config=upstream_config(phones=len(vocab['phones'])+1,singers=len(vocab['singers']),languages=len(vocab['languages']))
    pilot=dict(schema=1,source=source_provenance(),manifest_sha256=sha256(Path(args.prepared)/'manifest.jsonl'),
               codec_sha256=codec.fingerprint,model_config=config,vocab=vocab,items=selected,
               seed=17,epochs=64,per_gpu_batch=1,world_size=2,learning_rate=.001,
               scope='36 short train phrases + 18 song-held-out valid phrases; feasibility only',
               full_training_authorized=False)
    write_json(root/'pilot.json',pilot)
    per_language={lang:mean_maps([r['metrics'] for r in audit if r['language']==lang])
                  for lang in sorted({r['language'] for r in audit})}
    write_json(root/'codec-reconstruction.json',dict(samples=audit,macro=mean_maps([r['metrics'] for r in audit]),
                                                   per_language=per_language,
                                                   gate=synthesis_gate(list(per_language.values())),
                                                   native_roundtrip_verified=True))
    print(json.dumps({k:v for k,v in pilot.items() if k not in ['items','vocab','model_config']},indent=2))


def read_pilot(root,device):
    root=Path(root);p=json.loads((root/'pilot.json').read_text())
    if p['source']['commit']!=source_provenance()['commit']: raise ValueError('source revision differs')
    for item in p['items']:
        if sha256(item['cache'])!=item['cache_sha256']: raise ValueError('modified acoustic cache')
        c=torch.load(item['cache'],map_location='cpu',weights_only=False)
        if c['identity']['codec']!=p['codec_sha256']: raise ValueError('wrong codec')
        item['data']=c
        item['score']=score_inputs(item['row'],torch.tensor(item['counts']),p['vocab'],device)
        item['tokens']=c['codes'].permute(1,0,2).to(device).long()
        item['lf0']=torch.from_numpy(c['lf0']).float()[None,:,None].to(device)
    return p


def evaluate(model,pilot,root,tag,codec,run=None,step=0):
    model.eval(); rows=[]
    audio_dir=root/'audio'/tag;audio_dir.mkdir(parents=True,exist_ok=True)
    # All valid items plus predetermined first train item in every language.
    seen=set(); items=[]
    for item in pilot['items']:
        r=item['row']
        if r['split']=='valid' or r['language'] not in seen:
            items.append(item)
        if r['split']=='train':seen.add(r['language'])
    for item in tqdm(items,desc='baseline generation '+tag,leave=False):
        r,c=item['row'],item['data'];out=model.generate(**item['score'])
        if out['codes'].shape!=item['tokens'].shape: raise ValueError('generated duration differs from fixed score')
        wav=codec.decode(out['codes'].permute(1,0,2),c['samples'])[0,0].cpu().numpy()
        m=audio_metrics(c['audio'],wav,24000)
        m['acf_f0_rmse_cents']=m.pop('f0_rmse_cents')
        m.update(pitch_metrics(c['f0'],world_f0(wav,24000,frames=c['frames'])))
        pred=out['lf0'].cpu().numpy();pred=np.where(pred>0,np.exp(pred),0.)
        head=pitch_metrics(c['f0'],pred)
        row=dict(id=r['id'],language=r['language'],split=r['split'],waveform=m,predicted_f0=head)
        rows.append(row)
        for name,audio in [('reference',c['audio']),('codec',c['codec_audio']),('generated',wav)]:
            sf.write(audio_dir/(r['id']+'-'+name+'.wav'),audio,24000,subtype='FLOAT')
    grouped=defaultdict(list)
    for r in rows: grouped[(r['split'],r['language'])].append(r)
    aggregates={split:{lang:mean_maps([r['waveform'] for r in values]) for (s,lang),values in grouped.items() if s==split}
                for split in ('train','valid')}
    gates={split:synthesis_gate(list(values.values())) for split,values in aggregates.items()}
    report=dict(tag=tag,step=step,samples=rows,per_language=aggregates,gates=gates,
                waveform_passed=all(g['passed'] for g in gates.values()),
                full_training_unlocked=False,
                reason='Pilot only; held-out lyric intelligibility and baseline review are still required')
    write_json(root/(tag+'.json'),report)
    if run is not None:
        import wandb
        log={'pilot/step':step}
        for split,langs in aggregates.items():
            for lang,values in langs.items():
                for key,value in values.items():
                    if value is not None:log[f'{split}/{lang}/{key}']=value
        log['audition']=wandb.Table(columns=['id','language','split','reference','codec','generated'],data=[
            [r['id'],r['language'],r['split']]+[wandb.Audio(str(audio_dir/(r['id']+'-'+name+'.wav')),caption=f'{tag} {r["id"]} {name}')
                                              for name in ('reference','codec','generated')] for r in rows])
        run.log(log)
    return report


def profile(args):
    root=Path(args.output);device='cuda:0';p=read_pilot(root,device)
    from .toksing_model import TokSingDAC
    old=TokSingDAC(len(p['vocab']['phones']),len(p['vocab']['singers']),len(p['vocab']['languages']))
    report={'old_custom_predictor_parameters':sum(x.numel() for x in old.parameters()),'models':{}}
    del old
    sample=max(p['items'],key=lambda r:r['data']['frames'])
    for mode in ['parallel','prefix']:
        torch.manual_seed(p['seed']);m=OfficialTokSing(p['model_config'],mode=='prefix').to(device)
        opt=torch.optim.Adam(m.parameters(),lr=p['learning_rate'],eps=1e-6)
        torch.cuda.reset_peak_memory_stats();times=[]
        for index in range(8):
            torch.cuda.synchronize();start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            out=m(score=sample['score'],codes=sample['tokens'],lf0=sample['lf0'])
            out['loss'].backward()
            pitch_gradient=sum(float(p.grad.abs().sum()) for p in m.base.f0_predictor.parameters() if p.grad is not None)
            if pitch_gradient<=0: raise RuntimeError('official LF0 clamp has killed the pitch gradient')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            torch.cuda.synchronize()
            if index>=2:times.append(time.perf_counter()-start)
        report['models'][mode]={**m.parameter_report(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                                'seconds_per_phrase_update':float(np.mean(times)),
                                'final_pitch_gradient_l1':pitch_gradient,
                                'phrase_seconds':sample['row']['audio_seconds'],'frames':sample['data']['frames']}
        del opt,m;torch.cuda.empty_cache()
    write_json(root/'resource-profile.json',report);print(json.dumps(report,indent=2))


def train(args):
    d=Distributed();d.start();device=f'cuda:{d.local_rank}'
    root=Path(args.output);p=read_pilot(root,device)
    reconstruction=json.loads((root/'codec-reconstruction.json').read_text())
    if not reconstruction['native_roundtrip_verified'] or not reconstruction['gate']['passed']:
        raise RuntimeError('codec reconstruction gate failed; predictor training blocked')
    mode=args.mode;arm=args.arm
    output=(root/'continuations' if args.bank else root)/(mode+'-'+arm)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'started.json').exists(): raise RuntimeError('bounded run already started; no silent restart/extension')
    if d.world!=p['world_size']: raise ValueError('pilot requires the declared two GPUs')
    if mode=='parallel' and arm!='baseline': raise ValueError('RRQ comparison requires prefix-conditioned head')
    torch.manual_seed(p['seed']);m=OfficialTokSing(p['model_config'],mode=='prefix').to(device)
    bank=None
    if arm!='baseline' and not args.bank:
        raise ValueError('matched continuation requires a frozen common-prefix bank')
    if args.bank:
        if mode!='prefix': raise ValueError('common-prefix bank requires the same prefix architecture')
        bank_path=Path(args.bank);meta=json.loads((bank_path/'bank.json').read_text())
        if meta['pilot_sha256']!=sha256(root/'pilot.json'): raise ValueError('bank from another pilot')
        if not meta['quality_passed']: raise ValueError('common baseline quality gate failed')
        if sha256(meta['common_checkpoint'])!=meta['common_sha256']: raise ValueError('changed common checkpoint')
        state=torch.load(meta['common_checkpoint'],map_location='cpu',weights_only=False)
        if state['mode']!='prefix' or state['pilot_sha256']!=sha256(root/'pilot.json'):
            raise ValueError('checkpoint protocol mismatch')
        m.load_state_dict(state['model']);bank=bank_path
    opt=torch.optim.Adam(m.parameters(),lr=p['learning_rate'],eps=1e-6)
    wrapped=d.wrap(m); train_items=[r for r in p['items'] if r['row']['split']=='train']
    if len(train_items)%d.world: raise ValueError('unmatched DDP draw count')
    run=None;codec=None
    if d.primary:
        write_json(output/'started.json',dict(pilot_sha256=sha256(root/'pilot.json'),mode=mode,arm=arm,
                                             parameter_report=m.parameter_report(),start=time.time()))
        import wandb
        run=wandb.init(project='rrq-svs',entity='oxymoron_music',group='official-toksing-feasibility-v1',
                       name=mode+'-'+arm,job_type='bounded-baseline-audit',dir=str(output),
                       config={k:v for k,v in p.items() if k not in ['items','vocab']})
        write_json(output/'wandb.json',dict(url=run.url,id=run.id))
        # Frozen codec stays on CPU during training; only temporary validation decode on GPU.
        codec=FrozenEnCodec('cpu')
    d.barrier();step=0
    for epoch in range(1,p['epochs']+1):
        torch.manual_seed(p['seed']+epoch*1000+d.rank);m.train()
        order=list(range(len(train_items)));random.Random(p['seed']+epoch).shuffle(order)
        losses=[];start=time.perf_counter()
        bar=tqdm(range(0,len(order),d.world),desc=f'{mode}/{arm} epoch {epoch}/{p["epochs"]}',disable=not d.primary)
        for i in bar:
            item=train_items[order[i+d.rank]];labels=item['tokens'];context=None
            if bank is not None:
                target_path=bank/(item['row']['id']+'.pt')
                if sha256(target_path)!=meta['target_sha256'][item['row']['id']]:
                    raise ValueError('modified frozen target bank')
                target=torch.load(target_path,map_location=device,weights_only=False)
                coin=torch.tensor([random.Random(p['seed']+epoch*100000+i+d.rank).random()<.5],device=device)
                context,labels=training_pair(item['tokens'].permute(1,0,2),target['predicted'],target['corrected'],arm,coin)
                context,labels=context.permute(1,0,2),labels.permute(1,0,2)
            opt.zero_grad(set_to_none=True)
            out=wrapped(score=item['score'],codes=labels,lf0=item['lf0'],context=context)
            if not torch.isfinite(out['loss']): raise ValueError('nonfinite baseline loss')
            out['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            step+=1;losses.append(float(out['loss'].detach()))
            if d.primary:
                bar.set_postfix(loss=f'{losses[-1]:.3f}',pitch=f'{out["stats"]["pitch_loss"]:.3f}')
                if step%20==0:run.log({'pilot/step':step,'epoch':epoch,'train/loss':losses[-1],
                                      'train/grad_norm':float(norm),**{'train/'+k:float(v) for k,v in out['stats'].items()}})
        values=d.gather(losses)
        if d.primary:
            run.log({'epoch':epoch,'train/epoch_loss':float(np.mean(values)),'train/epoch_seconds':time.perf_counter()-start})
        if epoch in (1,16,32,64):
            d.barrier()
            if d.primary:
                codec.to(device)
                report=evaluate(m,p,output,f'epoch-{epoch:03d}',codec,run,step)
                codec.cpu();torch.cuda.empty_cache()
                atomic_torch(output/'last.pt',dict(model=m.state_dict(),config=m.config,mode=mode,arm=arm,
                                                 epoch=epoch,step=step,pilot_sha256=sha256(root/'pilot.json')))
            d.barrier()
    digest=d.assert_synced(m)
    if d.primary:
        write_json(output/'complete.json',dict(epoch=p['epochs'],steps=step,parameters_sha256=digest,
                                              pilot_sha256=sha256(root/'pilot.json'),full_training_unlocked=False))
        run.finish()
    d.close()


def targets(args):
    root=Path(args.output);p=read_pilot(root,'cuda:0');common=root/'prefix-baseline'
    quality=json.loads((common/'epoch-064.json').read_text())
    if not quality['waveform_passed']: raise RuntimeError('baseline failed pitch/spectral feasibility; RRQ bank blocked')
    checkpoint=common/'last.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['pilot_sha256']!=sha256(root/'pilot.json'): raise ValueError('checkpoint protocol mismatch')
    m=OfficialTokSing(p['model_config'],True).cuda().eval();m.load_state_dict(state['model'])
    codec=FrozenEnCodec('cuda:0');bank=root/'prefix-bank';bank.mkdir(exist_ok=False)
    target_sha256={}
    for item in tqdm(p['items'],desc='matched native RRQ targets'):
        native=item['tokens'].permute(1,0,2)
        predicted=m.generate(**item['score'])['codes'].permute(1,0,2)
        corrected=preserve_clean_frames(native,predicted,requantize(item['data']['z'].cuda(),predicted[:1],codec.layers))
        target_path=bank/(item['row']['id']+'.pt')
        atomic_torch(target_path,dict(predicted=predicted.cpu(),corrected=corrected.cpu()))
        target_sha256[item['row']['id']]=sha256(target_path)
    write_json(bank/'bank.json',dict(pilot_sha256=sha256(root/'pilot.json'),common_checkpoint=str(checkpoint.resolve()),
                                    common_sha256=sha256(checkpoint),target_sha256=target_sha256,quality_passed=True))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','profile','train','targets'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepared',default='../data/toksing/gtsinger-prepared')
    parser.add_argument('--cache',default='../data/toksing/encodec-pilot-cache')
    parser.add_argument('--mode',choices=['parallel','prefix'],default='parallel')
    parser.add_argument('--arm',choices=['baseline','exposure','rrq'],default='baseline')
    parser.add_argument('--bank')
    args=parser.parse_args();globals()[args.stage](args)


if __name__=='__main__':main()
