"""Fixed validation-panel diagnostics; generation never receives target audio.

Pitch and condition-CE gates are operational checks, not intelligibility/MOS
evidence. Oracle suffixes use target latents ONLY after free-running synthesis.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import uniaudio as ua
from .core import requantize
from .io import sha256, safe_id, write_json
from .metrics import audio_metrics, mean_maps
from .official_data import world_f0, pitch_metrics, paired_pitch_metrics
from .tracking import token_metrics


def condition_variant(sequence,tokens,kind):
    if kind not in ('phone','midi'):raise ValueError('unknown condition perturbation')
    changed=sequence.clone()
    first=tokens.index(f'<sing_{kind}_start>');last=tokens.index(f'<sing_{kind}_end>')
    start=int(torch.nonzero(sequence[0]==first)[-1])+1
    stop=int(torch.nonzero(sequence[0]==last)[0])
    body=changed[:,start:stop].reshape(1,-1,3)
    if kind=='phone':body.copy_(body.flip(1))
    else:
        zero=tokens.index('<sing_midi_0>')
        body.add_(((body>zero)&(body<zero+137)).long())
    return changed


def pad_pair(reference,output):
    length=max(len(reference),len(output))
    return tuple(np.pad(x,(0,length-len(x))) for x in (reference,output))


def measure(reference,output):
    a,b=pad_pair(reference,output)
    metrics=audio_metrics(a,b,16000)
    metrics.update(pitch_metrics(world_f0(a,16000),world_f0(b,16000)))
    metrics['duration_error_seconds']=(len(output)-len(reference))/16000
    return metrics


def operational_gate(per_language,thresholds,languages):
    if set(per_language)!=set(languages):return False
    for values in per_language.values():
        pairs=[('within_50_cents','within_50_cents_min',True),
               ('voiced_recall','voiced_recall_min',True),
               ('voicing_error','voicing_error_max',False),
               ('f0_rmse_cents','f0_rmse_cents_max',False),
               ('mrstft_spectral_convergence','spectral_convergence_max',False),
               ('phone_ce_gap','phone_ce_gap_min',True),('midi_ce_gap','midi_ce_gap_min',True)]
        for key,threshold,minimum in pairs:
            value=values.get(key)
            if value is None or not math.isfinite(value):return False
            if minimum and value<=thresholds[threshold]:return False
            if not minimum and value>=thresholds[threshold]:return False
    return bool(per_language)


@torch.inference_mode()
def clean_ce(data,row_id,model,runtime):
    totals=np.zeros((3,3),dtype=np.float64);count=0
    for segment in (s for s in data.segments if s['row_id']==row_id):
        ex=data.example(segment,runtime.device);seq=ex['sequence']
        start=ex['prefix_length'];stop=start+3*ex['frames']
        for k,kind in enumerate((None,'phone','midi')):
            condition=seq if kind is None else condition_variant(seq,data.vocab['tokens'],kind)
            with runtime.amp():
                logits=model(condition,torch.zeros_like(seq,dtype=torch.bool))[:,start:stop]
            losses=F.cross_entropy(logits.float().transpose(1,2),seq[:,start:stop],reduction='none')
            totals[k]+=losses.reshape(-1,3).sum(0).double().cpu().numpy()
        count+=ex['frames']
    if not count:raise ValueError('validation row has no segments')
    values=totals/count
    return dict(clean_audio_ce=float(values[0].mean()),
                **{f'clean_q{q}_ce':float(values[0,q]) for q in range(3)},
                phone_ce_gap=float((values[1]-values[0]).mean()),
                midi_ce_gap=float((values[2]-values[0]).mean()))


@torch.inference_mode()
def synthesize(cfg,data,row_id,model,runtime):
    # This boundary deliberately has no target-cache, latent or waveform access.
    prefix,frames=data.inference_prefix(row_id,runtime.device)
    with runtime.amp():
        return ua.generate(model,prefix,frames,data.vocab['tokens'].index('<audio_0>'),
                           topk=cfg.get('topk',30),temperature=cfg.get('temperature',.8),
                           seed=int(hashlib.sha256(f"{cfg.get('seed',17)}:{row_id}".encode()).hexdigest()[:15],16))


@torch.inference_mode()
def evaluate_loaded(cfg,data,model,checkpoint_path,out,runtime,languages=None):
    import torchaudio
    from .uniaudio_data import _verify_artifact, score_conditions
    from .uniaudio_train import protocol
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    panels=json.loads(_verify_artifact(Path(cfg['prepared_path']),data.recipe,'panels.json').read_text())
    languages=sorted(languages or {r['language'] for r in data.rows.values()})
    selected=[i for i in panels['valid'] if data.rows[i]['language'] in languages]
    if not selected:raise ValueError('empty validation panel')
    was_training=model.training;model.eval()
    codec=ua.load_codec(cfg['upstream_path'],cfg['codec_path'],runtime.device)
    examples=[]
    try:
        for row_id in tqdm(selected[runtime.rank::runtime.world],desc=f'validation GPU{runtime.rank}'):
            row=data.rows[row_id];folder=out/safe_id(row_id);folder.mkdir(exist_ok=True)
            start=time.monotonic()
            predicted=synthesize(cfg,data,row_id,model,runtime)
            # Ground truth is first opened AFTER synthesis, only for evaluation.
            if sha256(row['audio'])!=row['source_sha256']:raise ValueError('evaluation audio changed')
            raw,rate=sf.read(row['audio'],dtype='float32',always_2d=True)
            raw=torch.from_numpy(raw.mean(1))
            if rate!=16000:raw=torchaudio.functional.resample(raw,rate,16000)
            raw=raw.numpy();record=data._cached(row)
            clean=record['codes'].to(runtime.device).long();z=record['latent'].to(runtime.device)
            score_samples=round(math.fsum(row['phone_durations'])*16000)
            generated=codec.decode(predicted)[0,0,:score_samples].cpu().numpy()
            reconstructed=codec.decode(clean)[0,0,:len(raw)].cpu().numpy()
            waves=dict(reference=raw,codec=reconstructed,generated=generated)
            stats=measure(raw,generated)
            stats.update(clean_ce(data,row_id,model,runtime))
            stats.update(token_metrics(predicted.cpu().numpy()))
            _,midi=score_conditions(row,data.vocab)
            score_f0=np.where(midi.numpy()>0,440*2**((midi.numpy()-69)/12),0.)
            score_metrics=pitch_metrics(score_f0,world_f0(generated,16000,len(midi)))
            stats.update({'score_'+k:v for k,v in score_metrics.items()})
            oracle=dict(available=False,reason='annotation and waveform codec frame counts differ')
            if predicted.shape==clean.shape:
                exposure=clean.clone();exposure[0]=predicted[0]
                relabeled=requantize(z,predicted[:1],codec.quantizer.vq.layers[:3])
                oracle=dict(available=True,arms={})
                for name,codes in (('exposure_oracle',exposure),('rrq_oracle',relabeled)):
                    wave=codec.decode(codes)[0,0,:len(raw)].cpu().numpy();waves[name]=wave
                    oracle['arms'][name]=dict(measure(raw,wave),latent_mse=(codec.quantizer.decode(codes)-z).square().mean().item())
                common_support=paired_pitch_metrics(world_f0(raw,16000),world_f0(waves['exposure_oracle'],16000),world_f0(waves['rrq_oracle'],16000))
                for name,metrics in zip(('exposure_oracle','rrq_oracle'),common_support):oracle['arms'][name].update(metrics)
                oracle['rrq_minus_exposure_latent_mse']=oracle['arms']['rrq_oracle']['latent_mse']-oracle['arms']['exposure_oracle']['latent_mse']
            paths={}
            for name,wave in waves.items():
                path=folder/f'{name}.wav';sf.write(path,wave,16000,subtype='FLOAT');paths[name]=str(path.resolve())
            torch.save(dict(predicted=predicted.cpu(),clean=clean.cpu()),folder/'tokens.pt')
            example=dict(id=row_id,language=row['language'],singer=row['singer'],**stats,
                         audio=paths,codec=measure(raw,reconstructed),oracle=oracle,seconds=time.monotonic()-start)
            write_json(folder/'metrics.json',example);examples.append(example)
    finally:model.train(was_training)
    examples=sum(runtime.gather(examples),[])
    if {r['id'] for r in examples}!=set(selected):raise ValueError('incomplete validation panel')
    grouped=defaultdict(list)
    for row in examples:
        grouped[row['language']].append({k:v for k,v in row.items() if isinstance(v,(int,float)) or v is None})
    per_language={language:mean_maps(values) for language,values in grouped.items()}
    report=dict(protocol_sha256=protocol(data),checkpoint_sha256=sha256(checkpoint_path),languages=languages,
                examples=sorted(examples,key=lambda x:x['id']),per_language=per_language,macro=mean_maps(list(per_language.values())),
                thresholds=cfg['gate'],operational_gate_passed=operational_gate(per_language,cfg['gate'],languages),
                scope='Fixed validation panel, two utterances per language; not the untouched test set.',
                quality_limit='Pitch/voicing and condition-CE diagnostic only; no lyric intelligibility or MOS claim.',
                oracle_limit='Target-latent suffix oracle after free generation; not deployable inference or learnability proof.',
                comparison='Same clock; zero-pad missing tails; no DTW, pitch shift, rescaling or time stretch.',
                sampling=dict(topk=cfg.get('topk',30),temperature=cfg.get('temperature',.8),seed=cfg.get('seed',17)))
    if runtime.rank==0:write_json(out/'evaluation.json',report)
    return report


def main():
    from .uniaudio_data import load_recipe,PreparedData
    from .uniaudio_train import Runtime,load_model
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--workspace',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--checkpoint')
    parser.add_argument('--languages',nargs='+')
    args=parser.parse_args();cfg=load_recipe(args.config,args.workspace)
    data=PreparedData(cfg['prepared_path'],cfg['cache_path']);runtime=Runtime()
    try:
        model,_=load_model(cfg,data,runtime.device,args.checkpoint)
        evaluate_loaded(cfg,data,model,args.checkpoint or cfg['checkpoint_path'],args.output,runtime,args.languages)
    finally:runtime.close()


if __name__=='__main__':main()
