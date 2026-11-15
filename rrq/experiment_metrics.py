"""Objective held-out evaluation; ASR and acoustic metrics are not human ratings."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import unicodedata

import numpy as np
from scipy.signal import stft, resample_poly
import soundfile as sf
import torch

from .experiment import configuration, atomic_json
from .io import sha256
from .multilingual import read_rows
from .pipeline import prepare_audio
from .upstream import load_codec, load_vocoder

ASR_SHA256='542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1'


def text_normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def character_error(reference,hypothesis):
    a,b=text_normalize(reference),text_normalize(hypothesis)
    if not a: raise ValueError('empty normalized reference lyrics')
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return dict(cer=previous[-1]/len(a),character_edits=previous[-1],reference_characters=len(a))


def lyrics(row):
    value=row['lyrics']; value=value if isinstance(value,list) else value.split()
    return ' '.join(x for x in value if not (x.startswith('<') and x.endswith('>')))


def waveform_metrics(job):
    import pyworld
    row,path=job
    _,ref,_=prepare_audio(row['audio'],1.)
    wave,sr=sf.read(path,dtype='float64')
    pad=row['n_frames']*480-len(ref)
    if not 0<=pad<480 or sr!=24000 or wave.shape!=(row['n_frames']*480,) or not np.isfinite(wave).all():
        raise ValueError('evaluation waveform/grid mismatch')
    ref=np.pad(ref,(0,pad)).astype(np.float64)
    sc=[]; logmag=[]
    for n in (512,1024,2048):
        a,b=[np.abs(stft(x,fs=sr,nperseg=n,noverlap=n-n//4,boundary=None,padded=False)[2]) for x in (ref,wave)]
        sc.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-8)))
        logmag.append(float(np.mean(np.abs(np.log(np.maximum(a,1e-5))-np.log(np.maximum(b,1e-5))))))
    tracks=[]
    for x in (ref,wave):
        f,t=pyworld.dio(x,sr,f0_floor=50,f0_ceil=1500,frame_period=20)
        f=pyworld.stonemask(x,f,t,sr)[:row['n_frames']]; tracks.append(f)
    a,b=tracks; av=a>0; bv=b>0; joint=av&bv
    cents=1200*np.log2(b[joint]/a[joint])
    return dict(mrstft_sc=float(np.mean(sc)),mrstft_logmag=float(np.mean(logmag)),
        f0_rmse_cents=float(np.sqrt(np.mean(cents**2))) if joint.any() else None,
        voicing_error=float(np.mean(av!=bv)),joint_voiced_fraction=float(np.mean(joint)),
        reference_voiced_fraction=float(np.mean(av)),generated_voiced_fraction=float(np.mean(bv)),
        peak_amplitude=float(np.max(np.abs(wave))))


def evaluate(args):
    cfg=configuration(args.root); rows=[r for r in read_rows([cfg['bundle']/'primary.jsonl']) if r['split']==args.split]
    render_root=args.root/'renders'/args.split; render_root.mkdir(parents=True,exist_ok=True)
    codec=load_codec(cfg['assets']/'codec/ckpt_01135000.pth',args.device)
    vocoder=load_vocoder(cfg['assets']/'vocoder_24k/g_00885000',args.device)
    with torch.inference_mode():
        for label in ('reference','codec'):
            folder=render_root/label; folder.mkdir(exist_ok=True)
            for row in rows:
                path=folder/(row['id']+'.wav')
                if path.exists(): continue
                if label=='reference':
                    _,wave,_=prepare_audio(row['audio'],1.); wave=np.pad(wave,(0,row['n_frames']*480-len(wave)))
                else:
                    with np.load(cfg['cache']/(row['id']+'.npz'),allow_pickle=False) as f: codes=f['codes'].copy()
                    wave=vocoder(torch.from_numpy(codes)[None].to(args.device))[0,0].float().cpu().numpy()
                sf.write(path,wave,24000,subtype='FLOAT')
    del vocoder
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    asr_hash=sha256(args.asr/'model.safetensors')
    if asr_hash!=ASR_SHA256: raise ValueError('ASR weights differ from pinned official revision')
    processor=AutoProcessor.from_pretrained(args.asr,local_files_only=True)
    asr=AutoModelForSpeechSeq2Seq.from_pretrained(args.asr,local_files_only=True,torch_dtype=torch.float16,
        attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
    labels=['reference','codec']+args.labels
    result=[]; destination=args.root/'evaluation'/args.split; destination.mkdir(parents=True,exist_ok=True)
    for label in labels:
        if label not in ('reference','codec') and not (render_root/label/'complete.json').exists():
            raise ValueError('render stage incomplete: '+label)
        output=destination/(label+'.json')
        if output.exists():
            saved=json.loads(output.read_text())
            if saved['asr_sha256']!=asr_hash or saved['protocol_sha256']!=sha256(args.root/'protocol.json'):
                raise ValueError('evaluation provenance mismatch')
            indexed={r['id']:r for r in saved['samples']}
            if set(indexed)=={r['id'] for r in rows} and all(
                indexed[r['id']]['waveform_sha256']==sha256(render_root/label/(r['id']+'.wav')) for r in rows):
                result.extend(saved['samples']); continue
        jobs=[(row,render_root/label/(row['id']+'.wav')) for row in rows]
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            metrics=list(pool.map(waveform_metrics,jobs))
        collected=[]
        for row,metric in zip(rows,metrics):
            path=render_root/label/(row['id']+'.wav')
            wave,sr=sf.read(path,dtype='float32'); audio=resample_poly(wave,2,3)
            inputs=processor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            features=inputs.input_features.to(args.device,dtype=torch.float16)
            with torch.inference_mode():
                tokens=asr.generate(input_features=features,attention_mask=inputs.attention_mask.to(args.device),
                    language=row['language'].lower(),task='transcribe',max_new_tokens=192,num_beams=1,do_sample=False)
            hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0]
            metric.update(character_error(lyrics(row),hypothesis))
            if label not in ('reference','codec'):
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as f: codes=f['codes'].copy()
                with np.load(args.root/'targets'/(row['id']+'.npz'),allow_pickle=False) as f:
                    latent=f['latent'].copy(); coarse=f['predicted'][0].copy()
                if any(label.startswith(arm+'-') for arm in ('baseline','exposure','rrq')) and not np.array_equal(codes[0],coarse):
                    raise ValueError('unmatched coarse prefix in comparison')
                with torch.inference_mode(): recon=codec.quantizer.vq.decode(torch.from_numpy(codes)[:,None].to(args.device))[0].float().cpu().numpy()
                metric['latent_mse']=float(np.mean((latent-recon)**2))
            item=dict(id=row['id'],language=row['language'],singer=row['singer'],song_group=row['song_group'],label=label,
                metrics=metric,reference_text=lyrics(row),asr_text=hypothesis,waveform_sha256=sha256(path),
                asr_hit_length_cap=bool(tokens.shape[-1]>=192 and tokens[0,-1].item()!=asr.config.eos_token_id))
            collected.append(item); print('EVAL',args.split,label,len(collected),len(rows),json.dumps(metric),flush=True)
        atomic_json(output,dict(samples=collected,asr_sha256=asr_hash,protocol_sha256=sha256(args.root/'protocol.json'),
            pitch_estimator='WORLD DIO+StoneMask 50..1500Hz 20ms',alignment='none, only reported codec endpoint zero padding',
            asr_language='provided corpus language',asr_prompt='none',character_normalization='NFKC casefold alphanumeric'))
        result.extend(collected)
    print('EVALUATION_COMPLETE',args.split,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--asr',type=Path,default=Path('artifacts/whisper-turbo'))
    p.add_argument('--split',choices=['valid','test'],required=True); p.add_argument('--labels',nargs='+',required=True)
    args=p.parse_args(); torch.set_num_threads(2); evaluate(args)


if __name__=='__main__': main()
