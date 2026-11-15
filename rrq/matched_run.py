"""GTSinger-only common warmup, matched continuation, and score-only evaluation.

Nothing trains by importing this module or running readiness. Experiment stages
are explicit; target banks are created only after common training completes.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .core import requantize
from .dac_codec import FrozenDAC, clean_prefix_audit
from .distributed import Distributed
from .experiment import Experiment, epoch_schedule, mixture_choice, model_config, fingerprint, common_quality_gate
from .io import sha256, write_json
from .metrics import audio_metrics, mean_maps
from .targets import training_pair, preserve_clean_frames
from .toksing_data import load_audio, frame_condition, score_batch, annotation_hash
from .toksing_model import TokSingDAC
from .toksing_run import atomic_save, shard, summary, tracker
from .tracking import token_metrics


def seed_epoch(seed,epoch,rank):
    torch.manual_seed(seed+100003*epoch+rank)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+100003*epoch+rank)


def cache_stage(e,d,device):
    codec=FrozenDAC(e.codec,device);records=[];reconstruction=[]
    for row in tqdm(shard(e.rows,d.rank,d.world),desc=f'codec cache GPU{d.rank}',mininterval=2):
        if sha256(row['audio'])!=row['source_sha256']:raise ValueError('source changed after preparation')
        encoded=not e.acoustic.path(row['source_sha256']).exists()
        if encoded:
            x=load_audio(row['audio'],codec.sample_rate)
            z,codes=codec.encode(torch.from_numpy(x)[None,None].to(device))
            audit=clean_prefix_audit(z,codes,codec.layers) if row['id'] in e.panels['valid'] else None
            e.acoustic.put(row['source_sha256'],dict(codes=codes.cpu().short(),samples=len(x),
                sample_rate=codec.sample_rate,hop=codec.hop,clean_prefix_audit=audit,
                clean_prefix_audit_recipe=e.acoustic.audit_recipe))
        c=e.read_cache(row)
        records.append(dict(id=row['id'],frames=c['codes'].shape[-1],encoded=encoded,
                            alignment_rebuilt=c['alignment_rebuilt']))
        if row['id'] in e.panels['valid']:
            x=load_audio(row['audio'],codec.sample_rate)
            y=codec.decode(c['codes'].to(device).long(),c['samples'])[0,0].cpu().numpy()
            reconstruction.append(dict(id=row['id'],language=row['language'],singer=row['singer'],
                song_group=row['song_group'],arms={'codec':audio_metrics(x,y,codec.sample_rate)},
                clean_prefix_audit=c.get('clean_prefix_audit')))
    records=d.gather(records)
    reconstruction=d.gather(reconstruction)
    if len(records)!=len(e.rows) or {r['id'] for r in records}!=set(e.by_id):raise ValueError('cache coverage mismatch')
    if {r['id'] for r in reconstruction}!=set(e.panels['valid']):raise ValueError('reconstruction panel mismatch')
    if d.primary:
        report=summary(reconstruction);report['protocol_sha256']=e.hash
        report['clean_prefix_audited_samples']=sum(r['clean_prefix_audit'] is not None for r in reconstruction)
        write_json(e.root/'codec-reconstruction.json',report)
        write_json(e.root/'cache-complete.json',dict(protocol_sha256=e.hash,n=len(records),
            acoustic_encoded=sum(r['encoded'] for r in records),acoustic_reused=sum(not r['encoded'] for r in records),
            alignments_rebuilt=sum(r['alignment_rebuilt'] for r in records),acoustic_cache=str(e.acoustic.root),
            total_frames=sum(r['frames'] for r in records),max_frames=max(r['frames'] for r in records),
            latent_storage='none; exact FP32 target latent recomputed for targets/evaluation'))


def targets_stage(e,d,device):
    e.require_cache();checkpoint=e.root/'common/best.pt';state=e.checkpoint(checkpoint)
    gate=json.loads((e.root/'common/quality-gate.json').read_text())
    if gate['protocol_sha256']!=e.hash or gate['common_sha256']!=sha256(checkpoint):raise ValueError('stale common quality gate')
    if not gate['passed']:raise ValueError('Common baseline failed the declared feasibility gate; review before running RRQ branches')
    if state['arm']!='common':raise ValueError('prefix predictor must be the common model')
    common_hash=sha256(checkpoint);model=TokSingDAC(**state['model_config']).to(device)
    model.load_state_dict(state['model']);model.eval().requires_grad_(False)
    del state
    codec=FrozenDAC(e.codec,device)
    selected=e.train+[e.by_id[i] for i in e.panels['valid']]
    records=[];oracle_samples=[]
    for row in tqdm(shard(selected,d.rank,d.world),desc=f'frozen prefix targets GPU{d.rank}',mininterval=2):
        path=e.bank/(row['id']+'.pt');c=e.read_cache(row,device)
        if path.exists():
            b=e.read_bank(row,common_hash,device)
        else:
            with torch.inference_mode():
                predicted=model.generate(**score_batch(row,c,e.vocab,device)).permute(1,0,2)
                x=load_audio(row['audio'],codec.sample_rate)
                z,native=codec.encode(torch.from_numpy(x)[None,None].to(device))
                if not torch.equal(native,c['codes']):raise ValueError('native codec cache changed during target generation')
                corrected=requantize(z,predicted[:1],codec.layers)
                corrected=preserve_clean_frames(c['codes'],predicted,corrected)
            b=dict(id=row['id'],predicted=predicted.cpu().short(),corrected=corrected.cpu().short(),
                   common_sha256=common_hash,protocol_sha256=e.hash,source_sha256=row['source_sha256'],
                   annotation_sha256=annotation_hash(row))
            atomic_save(b,path)
        prediction=b['predicted'].to(device).long();corrected=b['corrected'].to(device).long()
        records.append(dict(id=row['id'],language=row['language'],split=row['split'],
            coarse_match=float((prediction[0]==c['codes'][0]).float().mean()),
            suffix_changed=float((corrected[1:]!=c['codes'][1:]).float().mean())))
        if row['split']=='valid':
            x=load_audio(row['audio'],codec.sample_rate)
            z,_=codec.encode(torch.from_numpy(x)[None,None].to(device))
            exposure=c['codes'].clone();exposure[0]=prediction[0]
            metrics={}
            for arm,tokens in dict(codec=c['codes'],exposure=exposure,rrq_oracle=corrected).items():
                y=codec.decode(tokens,c['samples'])[0,0].cpu().numpy()
                m=audio_metrics(x,y,codec.sample_rate)
                m['latent_mse']=float(F.mse_loss(codec.codec.quantizer.from_codes(tokens.permute(1,0,2))[0],z))
                metrics[arm]=m
            oracle_samples.append(dict(id=row['id'],language=row['language'],singer=row['singer'],arms=metrics))
    records=d.gather(records)
    oracle_samples=d.gather(oracle_samples)
    if len(records)!=len(selected) or {r['id'] for r in records}!={r['id'] for r in selected}:raise ValueError('target bank coverage mismatch')
    if d.primary:
        oracle=summary(oracle_samples)
        a,b=oracle['language_macro']['rrq_oracle'],oracle['language_macro']['exposure']
        viable=a['latent_mse']<b['latent_mse']-1e-6 and a['mrstft_spectral_convergence']<b['mrstft_spectral_convergence']-1e-6
        oracle.update(protocol_sha256=e.hash,common_sha256=common_hash,
                      oracle_room_detected=viable,note='Target audio is accessed here. This is not learned generation.')
        write_json(e.root/'target-bank-oracle.json',oracle)
        write_json(e.root/'target-bank-complete.json',dict(protocol_sha256=e.hash,common_sha256=common_hash,n=len(records),
            oracle_room_detected=viable,
            note='Training in-sample prefixes; validation out-of-sample prefixes. No out-of-fold claim.',samples=records))


@torch.inference_mode()
def evaluate_panel(e,d,device,model,codec,ids,common_hash=None,folder=None,ablate=False):
    model.eval();samples=[]
    for item_id in tqdm(shard(ids,d.rank,d.world),desc=f'validation GPU{d.rank}',disable=not d.primary,leave=False):
        row=e.by_id[item_id];c=e.read_cache(row,device);batch=score_batch(row,c,e.vocab,device)
        clean=c['codes'].permute(1,0,2)
        logits=model(**batch,codes=clean)
        ce=float(F.cross_entropy(logits.flatten(0,2),clean.flatten()))
        del logits
        generated=model.generate(**batch).permute(1,0,2)
        x=load_audio(row['audio'],codec.sample_rate)
        z=codec.codec.encoder(codec.codec.preprocess(torch.from_numpy(x)[None,None].to(device),codec.sample_rate))
        tokens={'generated':generated,'codec':c['codes']}
        diagnostic=dict(clean_context_ce=ce,coarse_accuracy=float((generated[0]==c['codes'][0]).float().mean()))
        if common_hash is not None:
            bank=e.read_bank(row,common_hash,device);prefix=bank['predicted'][:1]
            tokens['fixed_prefix']=model.generate_with_prefix(prefix.permute(1,0,2),**batch).permute(1,0,2)
            diagnostic['prefix_bank_agreement']=float((generated[0]==prefix[0]).float().mean())
            for arm in ('exposure','rrq'):
                context,labels=training_pair(c['codes'],bank['predicted'],bank['corrected'],arm,torch.ones(1,dtype=torch.bool,device=device))
                out=model(**batch,codes=context.permute(1,0,2))
                diagnostic[arm+'_suffix_ce']=float(F.cross_entropy(out[:,1:].flatten(0,2),labels[1:].permute(1,0,2).flatten()))
                del out
        if ablate:
            altered=dict(batch,phones=batch['phones'].flip(1))
            out=model(**altered,codes=clean)
            diagnostic['phone_reverse_ce_gap']=float(F.cross_entropy(out.flatten(0,2),clean.flatten()))-ce
            del out
            altered=dict(batch,midi=torch.where(batch['midi']>0,torch.clamp(batch['midi']+2,max=127),batch['midi']))
            out=model(**altered,codes=clean)
            diagnostic['note_shift_ce_gap']=float(F.cross_entropy(out.flatten(0,2),clean.flatten()))-ce
            del out
        metrics={}
        if folder:
            folder.mkdir(parents=True,exist_ok=True)
            sf.write(folder/(item_id+'-reference.wav'),x,codec.sample_rate,subtype='FLOAT')
        for arm,codes in tokens.items():
            y=codec.decode(codes,c['samples'])[0,0].cpu().numpy()
            m=audio_metrics(x,y,codec.sample_rate)
            zhat=codec.codec.quantizer.from_codes(codes.permute(1,0,2))[0]
            m['latent_mse']=float(F.mse_loss(zhat,z))
            m['nonfinite_samples']=int((~np.isfinite(y)).sum())
            metrics[arm]=m
            if folder:sf.write(folder/(item_id+'-'+arm+'.wav'),y,codec.sample_rate,subtype='FLOAT')
        samples.append(dict(id=item_id,language=row['language'],singer=row['singer'],song_group=row['song_group'],
                            arms=metrics,diagnostics=diagnostic,tokens=token_metrics(generated[:,0].cpu().numpy()),
                            clean_prefix_audit=c.get('clean_prefix_audit')))
    samples=d.gather(samples)
    if len(samples)!=len(ids) or {r['id'] for r in samples}!=set(ids):raise ValueError('evaluation coverage mismatch')
    report=summary(samples)
    report['diagnostics']={lang:mean_maps([r['diagnostics'] for r in samples if r['language']==lang]) for lang in report['per_language']}
    report['protocol_sha256']=e.hash
    return report


def log_validation(run,report,epoch,folder=None):
    if run is None:return
    import wandb
    values={'epoch':epoch}
    for arm,metrics in report['language_macro'].items():
        values.update({f'valid/{arm}/{k}':v for k,v in metrics.items() if v is not None})
    for lang,arms in report['per_language'].items():
        for arm,metrics in arms.items():values.update({f'valid/{lang}/{arm}/{k}':v for k,v in metrics.items() if v is not None})
    for lang,metrics in report['diagnostics'].items():values.update({f'valid/{lang}/{k}':v for k,v in metrics.items() if v is not None})
    for lang in report['per_language']:
        values.update({f'valid/{lang}/tokens/{k}':v for k,v in mean_maps([r['tokens'] for r in report['samples'] if r['language']==lang]).items()})
    values['codec/clean_prefix_rounding_changes']=sum((r.get('clean_prefix_audit') or {}).get('changed_tokens',0) for r in report['samples'])
    values['codec/clean_prefix_audited_samples']=sum(r.get('clean_prefix_audit') is not None for r in report['samples'])
    run.log(values)
    if folder:
        for lang in sorted(report['per_language']):
            sample=next(r for r in report['samples'] if r['language']==lang)
            for arm in ('reference',*sample['arms']):
                path=folder/(sample['id']+'-'+arm+'.wav')
                run.log({f'audio/{lang}/{arm}':wandb.Audio(str(path),caption=sample['id'])})


def train_stage(e,d,device,arm):
    e.require_cache();cfg=e.config
    if d.world!=cfg['world_size']:raise ValueError('matched training requires the locked world size')
    if arm not in ('common','baseline','exposure','rrq'):raise ValueError('unknown arm')
    common_hash=e.require_bank() if arm!='common' else None
    out=e.root/arm;out.mkdir(exist_ok=True)
    torch.manual_seed(cfg['seed']);mc=model_config(e.vocab,cfg)
    model=TokSingDAC(**mc).to(device)
    init_hash=None
    if arm!='common':
        common=e.checkpoint(e.root/'common/best.pt')
        model.load_state_dict(common['model']);init_hash=common_hash
        del common
    wrapped=d.wrap(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    epochs=cfg['common_epochs'] if arm=='common' else cfg['branch_epochs']
    first=1;step=0;best=float('inf');best_state=None
    if (out/'last.pt').exists():
        value=e.checkpoint(out/'last.pt')
        if value['arm']!=arm or value['initial_checkpoint_sha256']!=init_hash:raise ValueError('resume arm/initialization mismatch')
        model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer'])
        first=value['epoch']+1;step=value['step'];best=value['best']
        best_state=value['best_state']
        if d.primary and (not (out/'best.pt').exists() or sha256(out/'best.pt')!=value['best_checkpoint_sha256']):
            # last.pt is the epoch commit record; recover an interrupted best-file update.
            atomic_save(best_state,out/'best.pt')
        del value
    codec=FrozenDAC(e.codec,device)
    run=tracker(out,'training',dict(e.protocol,arm=arm,experiment_name=e.root.name,
                                  initial_checkpoint_sha256=init_hash)) if d.primary else None
    if run and first>1:run.log({'resume/from_epoch':first-1,'resume/optimizer_step':step})
    if first==1:
        # Same untouched initial weights, same validation panel, no optimization.
        initial=evaluate_panel(e,d,device,model,codec,e.panels['valid'],common_hash,ablate=True)
        if d.primary:
            write_json(out/'epoch-000.json',initial);log_validation(run,initial,0)
    accumulation=cfg['gradient_accumulation']
    epochbar=tqdm(range(first,epochs+1),desc=arm+' epochs',disable=not d.primary)
    for epoch in epochbar:
        started=time.time();seed_epoch(cfg['seed'],epoch,d.rank);model.train()
        draws=epoch_schedule(e.train,cfg['seed'],epoch,d.world,accumulation,cfg['chinese_sampling_weight'])
        local=list(enumerate(draws))[d.rank::d.world]
        losses=[];coarse_losses=[];suffix_losses=[];mixed=0;optimizer.zero_grad(set_to_none=True)
        for local_index,(global_index,item_id) in enumerate(tqdm(local,desc=f'{arm} epoch {epoch}',disable=not d.primary,leave=False)):
            row=e.by_id[item_id];c=e.read_cache(row,device);batch=score_batch(row,c,e.vocab,device)
            if arm=='common':context=labels=c['codes'];use=False
            else:
                bank=e.read_bank(row,common_hash,device)
                use=mixture_choice(cfg['seed'],epoch,global_index,cfg['predicted_prefix_probability'])
                context,labels=training_pair(c['codes'],bank['predicted'],bank['corrected'],arm,
                                            torch.tensor([use],dtype=torch.bool,device=device))
            update=(local_index+1)%accumulation==0
            sync=nullcontext() if update or d.world==1 else wrapped.no_sync()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    logits=wrapped(**batch,codes=context.permute(1,0,2))
                    target=labels.permute(1,0,2)
                    coarse=F.cross_entropy(logits[:,0].flatten(0,1),target[:,0].flatten())
                    suffix=F.cross_entropy(logits[:,1:].flatten(0,2),target[:,1:].flatten())
                    loss=(coarse+8*suffix)/9
                if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
                (loss/accumulation).backward()
            losses.append(float(loss.detach()));coarse_losses.append(float(coarse.detach()));suffix_losses.append(float(suffix.detach()))
            mixed+=int(use and arm!='baseline')
            if update:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                if run and step%20==0:run.log(dict(train_step=step,epoch=epoch,**{
                    'train/objective':losses[-1],'train/coarse_ce':coarse_losses[-1],
                    'train/suffix_ce':suffix_losses[-1],'train/grad_norm':float(norm),
                    'system/peak_gpu_GiB':torch.cuda.max_memory_allocated(device)/1024**3}))
        report=evaluate_panel(e,d,device,model,codec,e.panels['valid'],common_hash,ablate=True)
        score=report['language_macro']['generated'][cfg['selection_metric']]
        if score is None or not math.isfinite(score):raise ValueError('invalid selection metric')
        improved=score<best;best=min(best,score)
        audio_folder=None
        if improved or epoch==epochs:
            audio_folder=out/'audio'/'latest'
            # Save fixed, preselected samples without reranking by quality.
            chosen=[next(i for i in sorted(e.panels['valid']) if e.by_id[i]['language']==lang) for lang in sorted({r['language'] for r in e.valid})]
            evaluate_panel(e,d,device,model,codec,chosen,common_hash,folder=audio_folder)
        train_stats=d.gather([dict(loss=float(np.mean(losses)),coarse=float(np.mean(coarse_losses)),suffix=float(np.mean(suffix_losses)),mixed=mixed,n=len(local))])
        synced=d.assert_synced(model)
        if d.primary:
            state=dict(protocol_sha256=e.hash,arm=arm,initial_checkpoint_sha256=init_hash,
                model_config=mc,model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                step=step,best=best,parameter_sha256=synced,schedule_sha256=fingerprint(draws))
            if improved:
                best_state={k:v for k,v in state.items() if k not in ('optimizer','model')}
                best_state['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                atomic_save(best_state,out/'best.pt')
            report.update(epoch=epoch,step=step,selection_score=score,train=mean_maps(train_stats),
                draws=len(draws),predicted_context_fraction=sum(r['mixed'] for r in train_stats)/sum(r['n'] for r in train_stats),
                seconds=time.time()-started,parameter_sha256=synced,schedule_sha256=fingerprint(draws),
                shared_predicted_mixture_sha256=fingerprint([mixture_choice(cfg['seed'],epoch,i,cfg['predicted_prefix_probability']) for i in range(len(draws))]))
            write_json(out/f'epoch-{epoch:03d}.json',report)
            # All artifacts needed by this committed epoch exist before publishing last.
            state.update(best_state=best_state,best_checkpoint_sha256=sha256(out/'best.pt'))
            atomic_save(state,out/'last.pt')
            print(json.dumps(dict(arm=arm,epoch=epoch,step=step,selection_score=score,seconds=report['seconds'])),flush=True)
            log_validation(run,report,epoch,audio_folder)
        d.barrier()
    if run:run.finish()
    if arm=='common' and d.primary:
        state=e.checkpoint(out/'best.pt')
        initial=json.loads((out/'epoch-000.json').read_text())
        selected=json.loads((out/f"epoch-{state['epoch']:03d}.json").read_text())
        gate=common_quality_gate(initial,selected,cfg)
        gate.update(protocol_sha256=e.hash,common_sha256=sha256(out/'best.pt'))
        write_json(out/'quality-gate.json',gate)


def evaluate_stage(e,d,device,arm,split,unlock_test):
    if split=='test' and not unlock_test:raise ValueError('test evaluation requires explicit --unlock-test after protocol freeze')
    e.require_cache();common_hash=e.require_bank() if arm!='common' else None
    checkpoint=e.root/arm/'best.pt';state=e.checkpoint(checkpoint)
    if state['arm']!=arm:raise ValueError('checkpoint arm mismatch')
    model=TokSingDAC(**state['model_config']).to(device);model.load_state_dict(state['model']);model.eval()
    del state
    codec=FrozenDAC(e.codec,device)
    ids=sorted(r['id'] for r in e.rows if r['split']==split)
    # Full held-out evaluation is primary free-running only. Banks never contain test targets.
    report=evaluate_panel(e,d,device,model,codec,ids,None,folder=None,ablate=True)
    report.update(model_sha256=sha256(checkpoint),arm=arm,split=split)
    if d.primary:write_json(e.root/arm/f'evaluation-{split}.json',report)


def readiness(e,d,device):
    e.require_cache();checks=[]
    codec_report=json.loads((e.root/'codec-reconstruction.json').read_text())
    if codec_report['protocol_sha256']!=e.hash or {r['id'] for r in codec_report['samples']}!=set(e.panels['valid']):
        raise ValueError('codec reconstruction report mismatch')
    runtime=json.loads((e.root/'runtime-verification.json').read_text())
    if (not runtime['passed'] or runtime['config_sha256']!=fingerprint(e.config) or
        runtime['source_manifest_sha256']!=e.protocol['preprocess_config']['source_sha256'] or
        any(e.protocol['implementation'][k]!=v for k,v in runtime['implementation'].items())):
        raise ValueError('runtime verification does not match this configuration/implementation')
    for row in shard(e.rows,d.rank,d.world):
        c=e.read_cache(row)
        if sha256(row['audio'])!=row['source_sha256']:raise ValueError('source changed after cache')
        if c['codes'].shape[0]!=9 or c['sample_rate']!=44100 or c['hop']!=512:raise ValueError('wrong DAC format')
        checks.append(row['id'])
    checks=d.gather(checks)
    if len(checks)!=len(e.rows):raise ValueError('readiness coverage mismatch')
    if d.primary:
        report=dict(protocol_sha256=e.hash,n=len(checks),data_cache_verified=True,
                    codec_reconstruction=codec_report['language_macro']['codec'],
                    training_started=any((e.root/arm/'last.pt').exists() for arm in ('common','baseline','exposure','rrq')),
                    gpu_world=d.world,required_world=e.config['world_size'],ready_for_common=d.world==e.config['world_size'],
                    branches_require='completed common warmup and frozen target bank',test_locked=True)
        write_json(e.root/'readiness.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['cache','common','targets','branch','evaluate','readiness'])
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);p.add_argument('--codec',required=True)
    p.add_argument('--acoustic-cache',help='Shared native token store; default: PREPARED/../dac-cache')
    p.add_argument('--config',default='config/gtsinger-rrq-v1.json');p.add_argument('--arm',choices=['baseline','exposure','rrq'],default='baseline')
    p.add_argument('--split',choices=['valid','test'],default='valid');p.add_argument('--unlock-test',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=Distributed();d.start();device=torch.device(f'cuda:{d.local_rank}' if torch.cuda.is_available() else 'cpu')
    try:
        e=Experiment(a.prepared,a.output,a.codec,a.config,a.acoustic_cache);d.barrier()
        if a.stage=='cache':cache_stage(e,d,device)
        elif a.stage=='targets':targets_stage(e,d,device)
        elif a.stage in ('common','branch'):train_stage(e,d,device,'common' if a.stage=='common' else a.arm)
        elif a.stage=='evaluate':evaluate_stage(e,d,device,a.arm,a.split,a.unlock_test)
        else:readiness(e,d,device)
    finally:d.close()


if __name__=='__main__':main()
