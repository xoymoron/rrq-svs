"""Bounded frozen-checkpoint baseline/oracle audit; no training is launched."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import uniaudio as ua
from .core import make_arms, requantize
from .metrics import audio_metrics, mean_maps
from .official_data import world_f0, pitch_metrics, paired_pitch_metrics


def select_panel(rows, per_singer=2):
    selected = []
    eligible = [r for r in rows if r['language'] == 'Chinese'
                and r['split'] == 'valid' and 3 <= sum(r['phone_durations']) <= 8]
    for singer in sorted({r['singer'] for r in eligible}):
        candidates = sorted((r for r in eligible if r['singer'] == singer), key=lambda r: r['id'])
        songs = set()
        for row in candidates:
            if row['song_group'] in songs:
                continue
            references = sorted((r for r in rows if r['split'] == 'train'
                                  and r['singer'] == singer and r['song_group'] != row['song_group']
                                  and sum(r['phone_durations']) >= 3), key=lambda r: r['id'])
            if references:
                selected.append(dict(target=row, prompt=references[0], seed=17+len(selected)))
                songs.add(row['song_group'])
            if len(songs) == per_singer:
                break
    return selected


def prepare(args):
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    panel = select_panel(rows)
    if not panel:
        raise ValueError('no eligible native-vocabulary validation panel')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(dict(upstream_commit=ua.UPSTREAM_COMMIT,
                              scope='Mandarin native-vocabulary diagnostic; not multilingual validation',
                              selection='two distinct valid songs per singer, id order, 3-8 seconds',
                              n_input_rows=len(rows), items=panel), ensure_ascii=False, indent=2)
    path = out / 'panel.json'
    if path.exists() and path.read_text(encoding='utf8') != contents:
        raise ValueError('refusing to replace a frozen panel')
    path.write_text(contents, encoding='utf8')
    print(json.dumps({'panel_size': len(panel), 'singers': sorted({p['target']['singer'] for p in panel})}))


def waveform(path, device):
    import soundfile as sf
    import torchaudio
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = torch.from_numpy(x.mean(axis=1)).to(device)[None, None]
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    return x


@torch.inference_mode()
def run(args):
    import soundfile as sf
    torch.set_num_threads(2)
    torch.cuda.set_device(args.rank)
    device = f'cuda:{args.rank}'
    model, config = ua.load_generator(args.upstream, args.config, args.checkpoint, device)
    codec = ua.load_codec(args.upstream, args.codec, device)
    layers = codec.quantizer.vq.layers[:3]
    vocab = {line.split()[0]: i for i, line in enumerate(
        (Path(args.upstream) / 'UniAudio/tools/tokenizer/Sing/dict_phone.txt').read_text().splitlines())}
    tokens = config['token_list']; offset = tokens.index('<audio_0>')
    out = Path(args.output)
    panel = json.loads((out/'panel.json').read_text(encoding='utf8'))['items']
    metadata = dict(parameters=sum(p.numel() for p in model.parameters()),
                    codec_parameters=sum(p.numel() for p in codec.parameters()),
                    checkpoint=str(Path(args.checkpoint).resolve()), upstream_commit=ua.UPSTREAM_COMMIT,
                    dtype=str(next(model.parameters()).dtype), rank=args.rank, world_size=args.world_size,
                    sampling=dict(mode='sampling', topk=30, temperature=.8),
                    frames='performed phoneme duration at 50 Hz, EOS suppressed',
                    train_steps=0, strict_load=True)
    (out/f'runtime-{args.rank}.json').write_text(json.dumps(metadata, indent=2))
    for index, item in tqdm(list(enumerate(panel))[args.rank::args.world_size], desc=f'GPU {args.rank} phrases'):
        row, ref = item['target'], item['prompt']
        folder = out/row['id']; folder.mkdir(exist_ok=True)
        result = dict(id=row['id'], singer=row['singer'], language=row['language'],
                      prompt_id=ref['id'], seed=item['seed'], rank=args.rank)
        try:
            phones, midi = ua.score_frames(row, vocab, tokens)
            x = waveform(row['audio'], device)
            prompt_audio = waveform(ref['audio'], device)[..., :48000]
            z = codec.encoder(x)
            clean = codec.quantizer.encode(z, codec.frame_rate, 1.5).long()
            if len(midi) != clean.shape[-1]:
                raise ValueError(f'score/audio frame mismatch {len(midi)} != {clean.shape[-1]}')
            identity = requantize(z, clean[:1], layers)
            if not torch.equal(identity, clean):
                raise ValueError('clean-prefix re-quantization differs from native codec')
            prompt_codes = codec.encode(prompt_audio, target_bw=1.5).long()
            prefix = ua.singing_prefix(tokens, phones, midi, prompt_codes)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            predicted = ua.generate(model, prefix, len(midi), offset, seed=item['seed'],
                                    progress=lambda it: tqdm(it, desc='score frames', leave=False))
            torch.cuda.synchronize()
            result.update(generation_seconds=time.monotonic()-start,
                          peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                          frames=len(midi), samples=x.numel(), clean_prefix_identity=True,
                          q0_agreement=predicted[0].eq(clean[0]).float().mean().item())
            # Teacher-forced CE is a separate diagnostic, never a free generation metric.
            sequence = torch.cat([prefix, ua.flatten_codes(clean, offset)], dim=1)
            logits = model(sequence, torch.zeros_like(sequence, dtype=torch.bool))[:, prefix.shape[1]:]
            labels = sequence[:, prefix.shape[1]:]
            losses = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none')
            result['teacher_forced_ce_per_book'] = losses.reshape(-1,3).mean(0).tolist()
            del logits, sequence, losses
            arms = make_arms(z, clean, predicted, layers)
            raw = x[0, 0].cpu().numpy(); waves = {}; statistics = {}
            sf.write(folder/'reference.wav', raw, 16000, subtype='FLOAT')
            raw_f0 = world_f0(raw, 16000, len(midi))
            for name, codes in arms.items():
                wav = codec.decode(codes)[0, 0]
                # Remove only codec's documented final padding, not a time alignment.
                if not 0 <= wav.numel()-len(raw) < codec.hop_length:
                    raise ValueError('unexpected decoder length')
                wav = wav[:len(raw)].cpu().numpy(); waves[name] = wav
                sf.write(folder/f'{name}.wav', wav, 16000, subtype='FLOAT')
                f0 = world_f0(wav, 16000, len(midi))
                metric = audio_metrics(raw, wav, 16000)
                metric.update(pitch_metrics(raw_f0, f0))
                metric['latent_mse'] = (codec.quantizer.decode(codes)-z).square().mean().item()
                metric['score_pitch'] = pitch_metrics(
                    np.where(midi.numpy()>0, 440*2**((midi.numpy()-69)/12), 0), f0)
                statistics[name] = metric
            a, b = paired_pitch_metrics(raw_f0,
                world_f0(waves['exposure'],16000,len(midi)), world_f0(waves['rrq_oracle'],16000,len(midi)))
            statistics['exposure'].update(a); statistics['rrq_oracle'].update(b)
            for name, wave in waves.items():
                statistics[name]['vs_codec_spectral_convergence'] = audio_metrics(waves['codec'], wave,16000)['mrstft_spectral_convergence']
            result.update(arms=statistics, status='ok')
            torch.save(dict(clean=clean.cpu(), predicted=predicted.cpu(),
                            rrq=arms['rrq_oracle'].cpu()), folder/'tokens.pt')
        except Exception as exc:
            import traceback
            result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({k: result.get(k) for k in ['id','status','q0_agreement','generation_seconds','error']}), flush=True)


def aggregate(args):
    out = Path(args.output)
    panel = json.loads((out/'panel.json').read_text(encoding='utf8'))['items']
    rows = []
    for p in panel:
        path = out/p['target']['id']/'result.json'
        if not path.exists():
            raise ValueError(f'incomplete pilot: {path}')
        rows.append(json.loads(path.read_text()))
    ok = [r for r in rows if r['status']=='ok']
    summary = dict(total=len(rows), successful=len(ok), failures=[r for r in rows if r['status']!='ok'],
                   conclusion='diagnostic only; not learned RRQ effectiveness or multilingual validation')
    if ok:
        summary['phrase_mean'] = {arm: mean_maps([{k:v for k,v in r['arms'][arm].items() if not isinstance(v,dict)} for r in ok]) for arm in ok[0]['arms']}
        summary['paired_improvement_counts'] = {
            metric: sum(r['arms']['rrq_oracle'][metric] < r['arms']['exposure'][metric] for r in ok)
            for metric in ['latent_mse','mrstft_spectral_convergence']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if args.wandb:
        import wandb
        runtime = json.loads((out/'runtime-0.json').read_text())
        run = wandb.init(project='rrq-svs', name=f"uniaudio-{runtime['parameters']//1_000_000}m-native-oracle",
                         config=runtime, job_type='feasibility')
        table = wandb.Table(columns=['id','singer','arm','audio','latent_mse','spectral_convergence','f0_rmse_cents'])
        for r in ok:
            table.add_data(r['id'],r['singer'],'reference',wandb.Audio(str(out/r['id']/'reference.wav')),None,None,None)
            for arm,m in r['arms'].items():
                table.add_data(r['id'],r['singer'],arm,wandb.Audio(str(out/r['id']/f'{arm}.wav')),
                               m['latent_mse'],m['mrstft_spectral_convergence'],m['f0_rmse_cents'])
        run.log({'audition':table, 'n_success':len(ok), 'n_failed':len(rows)-len(ok)})
        for arm, metrics in summary.get('phrase_mean',{}).items():
            run.summary[arm] = metrics
        run.finish(exit_code=0 if len(ok)==len(rows) else 1)
    if len(ok) != len(rows):
        raise RuntimeError(f'{len(rows)-len(ok)} pilot items failed; see summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run','aggregate'])
    parser.add_argument('--manifest'); parser.add_argument('--upstream')
    parser.add_argument('--config'); parser.add_argument('--checkpoint'); parser.add_argument('--codec')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rank',type=int,default=0); parser.add_argument('--world-size',type=int,default=1)
    parser.add_argument('--wandb', action='store_true')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'aggregate':aggregate}[args.stage](args)


if __name__=='__main__':
    main()
