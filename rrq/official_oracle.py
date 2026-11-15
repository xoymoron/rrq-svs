"""Paired native-codec repair diagnostic using a real model's predicted q0.

Oracle labels use reference latent information. This is not deployable inference
and is not evidence that a score-conditioned suffix predictor can learn them.
No training, checkpoint selection, or automatic experiment launch happens here.
"""
import argparse
import json
from pathlib import Path

import torch
import soundfile as sf
from tqdm import tqdm

from .core import make_arms
from .encodec_codec import FrozenEnCodec
from .io import sha256, write_json
from .metrics import audio_metrics, mean_maps
from .official_data import pitch_metrics, world_f0, paired_pitch_metrics
from .official_pilot import read_pilot
from .official_toksing import OfficialTokSing


@torch.inference_mode()
def run(root, mode):
    root=Path(root)
    output=root/(mode+'-baseline')
    destination=output/'oracle.json'
    if destination.exists(): raise RuntimeError('oracle report already exists; no silent overwrite')
    audio_dir=output/'oracle-audio'
    audio_dir.mkdir(exist_ok=False)
    pilot=read_pilot(root,'cuda:0')
    checkpoint=output/'last.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    complete=json.loads((output/'complete.json').read_text())
    if (state['pilot_sha256']!=sha256(root/'pilot.json') or state['epoch']!=pilot['epochs']
            or state['mode']!=mode or complete['steps']!=state['step']):
        raise ValueError('require the completed declared baseline; no intermediate checkpoint selection')
    model=OfficialTokSing(pilot['model_config'],mode=='prefix').cuda().eval()
    model.load_state_dict(state['model'])
    codec=FrozenEnCodec('cuda:0')
    if codec.fingerprint!=pilot['codec_sha256']: raise ValueError('changed codec checkpoint')
    records=[]
    for item in tqdm(pilot['items'],desc=mode+' paired oracle'):
        row,cached=item['row'],item['data']
        sf.write(audio_dir/(row['id']+'-reference.wav'),cached['audio'],24000,subtype='FLOAT')
        generated=model.generate(**item['score'])['codes'].permute(1,0,2)
        native=item['tokens'].permute(1,0,2)
        z=cached['z'].cuda()
        arms=make_arms(z,native,generated,codec.layers)
        measures={};tracks={}
        for name,codes in arms.items():
            quantized=codec.codec.quantizer.decode(codes)
            audio=codec.decode(codes,cached['samples'])[0,0].cpu().numpy()
            sf.write(audio_dir/(row['id']+'-'+name+'.wav'),audio,24000,subtype='FLOAT')
            values=audio_metrics(cached['audio'],audio,24000)
            values['acf_f0_rmse_cents']=values.pop('f0_rmse_cents')
            tracks[name]=world_f0(audio,24000,frames=cached['frames'])
            values.update(pitch_metrics(cached['f0'],tracks[name]))
            values['latent_mse']=float((quantized-z).square().mean())
            measures[name]=values
        paired=paired_pitch_metrics(cached['f0'],tracks['exposure'],tracks['rrq_oracle'])
        for name,values in zip(('exposure','rrq_oracle'),paired): measures[name].update(values)
        delta={k:measures['rrq_oracle'][k]-v if v is not None and measures['rrq_oracle'][k] is not None else None
               for k,v in measures['exposure'].items()}
        records.append(dict(id=row['id'],split=row['split'],language=row['language'],arms=measures,
                            q0_agreement=float((generated[0]==native[0]).float().mean()),
                            rrq_minus_exposure=delta))
    groups={}
    for split in ('train','valid'):
        groups[split]={}
        for language in sorted(pilot['vocab']['languages']):
            rows=[r for r in records if r['split']==split and r['language']==language]
            groups[split][language]=dict(n=len(rows),q0_agreement=sum(r['q0_agreement'] for r in rows)/len(rows),
                arms={arm:mean_maps([r['arms'][arm] for r in rows]) for arm in records[0]['arms']},
                rrq_minus_exposure=mean_maps([r['rrq_minus_exposure'] for r in rows]),
                fraction_latent_improved=sum(r['rrq_minus_exposure']['latent_mse']<0 for r in rows)/len(rows),
                fraction_spectral_improved=sum(r['rrq_minus_exposure']['mrstft_spectral_convergence']<0 for r in rows)/len(rows))
    write_json(destination,dict(pilot_sha256=sha256(root/'pilot.json'),checkpoint_sha256=sha256(checkpoint),
                               codec_sha256=codec.fingerprint,samples=records,per_language=groups,
                               oracle_uses_reference_latent=True,learned_rrq_tested=False,
                               full_training_unlocked=False))
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['parallel','prefix'],required=True)
    args=parser.parse_args()
    run(args.output,args.mode)
