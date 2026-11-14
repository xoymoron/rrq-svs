"""Run `python -m rrq --help`. All generated artifacts stay in the requested output."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch

from .io import parse_generation, save_prediction, sha256, write_json
from .pipeline import run_diagnostic
from .upstream import ROOT, load_codec, load_vocoder
from .assets import fetch_assets


def models(args):
    codec_path = args.assets/'codec/ckpt_01135000.pth'
    vocoder_path = args.assets/'vocoder_24k/g_00885000'
    provenance = {'upstream_base':'14816cda5aae948cf0f84bd9f5b19b7189154666',
                  'codec_sha256':sha256(codec_path),'vocoder_sha256':sha256(vocoder_path),
                  'codec_config_sha256':sha256(codec_path.with_name('config.yaml')),
                  'vocoder_config_sha256':sha256(vocoder_path.with_name('config.json')),
                  'torch_version':str(torch.__version__), 'device':args.device}
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    provenance['implementation_commit'] = revision.stdout.strip() if revision.returncode == 0 else None
    provenance['rrq_source_hashes'] = {p.name:sha256(p) for p in sorted((ROOT/'rrq').glob('*.py'))}
    return load_codec(codec_path,args.device), load_vocoder(vocoder_path,args.device), provenance


def import_codes(args):
    predictions = parse_generation(args.input.read_text(encoding='utf-8-sig'))
    if args.output.exists():
        raise ValueError('output already exists')
    args.output.mkdir(parents=True)
    for item, codes in predictions.items():
        save_prediction(args.output/(item+'.npz'),codes,
                        {'id':item, 'origin':'model_rollout','frame_rate':50,
                         'generator':args.generator, 'seed':args.seed,
                         'source_sha256':sha256(args.input),
                         'target_audio_used_for_prediction':False,
                         'provenance_is_user_declared':True})
    print(f'Imported {len(predictions)} samples. Match D-IDs to targets explicitly in the manifest.')


def smoke(args, codec, vocoder, provenance):
    """Synthetic I/O check only: perturbed GT-derived tokens are NOT model predictions."""
    inputs = args.output.with_name(args.output.name+'-inputs')
    if args.output.exists() or inputs.exists():
        raise ValueError('smoke output/input directory already exists')
    inputs.mkdir(parents=True)
    torch.manual_seed(0)
    t = np.arange(5120,dtype=np.float32)/16000
    wave = (.12*np.sin(2*np.pi*220*t)+.035*np.sin(2*np.pi*440*t)).astype('float32')
    sf.write(inputs/'synthetic.wav',wave,16000,subtype='FLOAT')
    with torch.inference_mode():
        z = codec.encoder(torch.from_numpy(wave)[None,None].to(args.device))
        codes = codec.quantizer.vq.encode(z,n_q=3)[:,0].cpu().numpy()
    codes[0,::5] = (codes[0,::5]+1)%1024
    save_prediction(inputs/'synthetic.npz',codes,
                    {'id':'synthetic','origin':'synthetic_smoke','frame_rate':50,
                     'generator':'GT-derived synthetic perturbation; NOT a generator rollout',
                     'target_audio_used_for_prediction':True})
    manifest = inputs/'manifest.jsonl'
    manifest.write_text(json.dumps({'id':'synthetic','singer':'synthetic_oscillator',
                                   'audio':'synthetic.wav','prediction':'synthetic.npz',
                                   'origin':'synthetic_smoke','conditioning':'synthetic',
                                   'alignment':'same_50hz_grid','gain':1.})+'\n',encoding='utf-8')
    return run_diagnostic(manifest,args.output,codec,vocoder,provenance,args.device)


def main():
    parser = argparse.ArgumentParser(description='RRQ frozen mechanism diagnostic (no trained SVS claims)')
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('fetch-assets',help='Fetch pinned codec/vocoder only (~794 MB); no generator weights')
    p.add_argument('--output',type=Path,default=Path('artifacts/pretrained'))
    p = commands.add_parser('import-codes',help='Import ORIGINAL Prompt-Singer D- lines as [3,T] token caches')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generator',required=True,help='checkpoint/revision or checkpoint hash')
    p.add_argument('--seed',type=int,required=True)
    for command in ('diagnostic','smoke'):
        p = commands.add_parser(command,help='Four-arm oracle diagnostic' if command=='diagnostic' else 'Synthetic integration check, NOT research evidence')
        p.add_argument('--assets',type=Path,default=Path('artifacts/pretrained'))
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--device',default='cpu')
        p.add_argument('--threads',type=int,default=4)
        if command=='diagnostic': p.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.command == 'fetch-assets':
            fetch_assets(args.output); return
        if args.command == 'import-codes':
            import_codes(args); return
        if args.threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(args.threads)
        codec,vocoder,provenance = models(args)
        report = smoke(args,codec,vocoder,provenance) if args.command=='smoke' else run_diagnostic(args.manifest,args.output,codec,vocoder,provenance,args.device)
        print(json.dumps({'report':str(args.output/'report.json'),'origin':report['origin'],
                          'codec_state_unchanged':report['codec_state_unchanged'],
                          'n_phrases':report['summary']['n_phrases']},ensure_ascii=False))
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2, f'RRQ error: {exc}\n')
