"""Inference-only MIDI intervention and decoding diagnosis of a saved checkpoint.

Uses the exact saved validation panel, no optimizer or target-conditioned generation.
The unchanged sampling/zero-shift tokens must reproduce the saved evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .io import sha256, safe_id, write_json
from .metrics import mean_maps


def validate_output_directory(output, source_files):
    output = Path(output).resolve()
    for source in source_files:
        parent = Path(source).resolve().parent
        if output == parent or output in parent.parents or parent in output.parents:
            raise ValueError('diagnostic output must not overlap a source run directory')


def score_clock_audio(audio, frames):
    return np.pad(audio, (0, max(0, frames * 320 - len(audio))))


def transpose_prefix(prefix, tokens, semitones):
    if not isinstance(semitones, int):
        raise ValueError('integer semitone shift required')
    start_id, end_id = (tokens.index(f'<sing_midi_{s}>') for s in ('start', 'end'))
    starts, ends = (torch.nonzero(prefix[0] == i).flatten() for i in (start_id, end_id))
    if len(starts) != 3 or len(ends) != 3:
        raise ValueError('expected exactly one native MIDI segment')
    start, stop = int(starts[-1]) + 1, int(ends[0])
    if stop <= start or (stop - start) % 3:
        raise ValueError('invalid MIDI patch boundaries')
    result = prefix.clone()
    body = result[:, start:stop]
    zero = tokens.index('<sing_midi_0>')
    notes = body - zero
    active = notes > 0
    shifted = notes[active] + semitones
    if (notes < 0).any() or (notes > 137).any() or ((shifted < 1) | (shifted > 137)).any():
        raise ValueError('transposition outside native MIDI range')
    body[active] += semitones
    return result


def shift_response(base, shifted, score_midi, semitones):
    base, shifted, score_midi = (np.asarray(x) for x in (base, shifted, score_midi))
    if base.shape != shifted.shape or base.shape != score_midi.shape:
        raise ValueError('counterfactual pitch clock mismatch')
    active = score_midi > 0
    joint = active & (base > 0) & (shifted > 0)
    denominator = max(1, int(active.sum()))
    delta = 12 * np.log2(shifted[joint] / base[joint])
    error = 100 * (delta - semitones)
    return dict(requested_semitones=semitones, joint_frames=int(joint.sum()),
                joint_voiced_score_coverage=float(joint.sum() / denominator),
                base_voiced_score_coverage=float((active & (base > 0)).sum() / denominator),
                shifted_voiced_score_coverage=float((active & (shifted > 0)).sum() / denominator),
                median_shift_semitones=float(np.median(delta)) if len(delta) else None,
                mean_shift_semitones=float(np.mean(delta)) if len(delta) else None,
                shift_error_mae_cents=float(np.mean(np.abs(error))) if len(delta) else None,
                within_50_cents_of_requested_shift=float(np.mean(np.abs(error) <= 50)) if len(delta) else None)


def summarize_oracles(report):
    groups = defaultdict(list)
    for row in report['examples']:
        groups[row['singer']].append(row)
    summary = {}
    for singer, rows in groups.items():
        available = [r for r in rows if r.get('oracle', {}).get('available')]
        arms = {arm: mean_maps([r['oracle']['arms'][arm] for r in available])
                for arm in ('exposure_oracle', 'rrq_oracle')}
        deltas = []
        for row in available:
            a, b = (row['oracle']['arms'][arm] for arm in ('exposure_oracle', 'rrq_oracle'))
            deltas.append({key: b[key] - a[key] if a.get(key) is not None and b.get(key) is not None else None
                           for key in a.keys() & b.keys()})
        summary[singer] = dict(total_phrases=len(rows), available_phrases=len(available), arms=arms,
                               rrq_minus_exposure=mean_maps(deltas),
                               unavailable=[dict(id=r['id'], reason=r.get('oracle', {}).get('reason', 'missing'))
                                            for r in rows if not r.get('oracle', {}).get('available')])
    return summary


def summarize_variants(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['singer']].append(row)
    result = {}
    for singer, items in groups.items():
        variants = {}
        for name in items[0]['variants']:
            values = [r['variants'][name] for r in items]
            metrics = mean_maps([v['metrics'] for v in values])
            if all('asr' in v for v in values):
                metrics['cer'] = sum(v['asr']['errors'] for v in values) / sum(v['asr']['characters'] for v in values)
            variants[name] = metrics
        response = {name: mean_maps([r['shift_response'][name] for r in items]) for name in items[0]['shift_response']}
        result[singer] = dict(n_phrases=len(items), variants=variants, shift_response=response,
                              greedy_vs_sampling_common_support=mean_maps([r['greedy_vs_sampling_common_support'] for r in items]),
                              reference_score_calibration=mean_maps([r['reference_score_calibration'] for r in items]))
    return result


def pitch_details(reference, generated):
    from .official_data import pitch_metrics
    result = pitch_metrics(reference, generated)
    mask = (reference > 0) & (generated > 0)
    cents = 1200 * np.log2(generated[mask] / reference[mask])
    result['signed_median_cents'] = float(np.median(cents)) if len(cents) else None
    # Diagnostic only; the primary score is NEVER octave-corrected.
    result['octave_wrapped_within50_diagnostic'] = float(np.mean(np.abs((cents + 600) % 1200 - 600) <= 50)) if len(cents) else None
    return result


@torch.inference_mode()
def diagnose(cfg, data, checkpoint, evaluation, out, runtime):
    import soundfile as sf
    from tqdm import tqdm
    from . import uniaudio as ua, pronunciation
    from .uniaudio_train import load_model, protocol
    from .uniaudio_data import score_conditions
    from .official_data import world_f0
    saved = json.loads(Path(evaluation).read_text())
    checkpoint_hash = sha256(checkpoint)
    if saved['checkpoint_sha256'] != checkpoint_hash or saved['protocol_sha256'] != protocol(data):
        raise ValueError('evaluation does not belong to this checkpoint/data')
    if saved['sampling'] != dict(topk=cfg['topk'], temperature=cfg['temperature'], seed=cfg['seed']):
        raise ValueError('sampling contract differs from saved evaluation')
    out = Path(out)
    validate_output_directory(out, [checkpoint, evaluation] +
                              [r['audio']['generated'] for r in saved['examples']])
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(checkpoint_sha256=checkpoint_hash, source_evaluation_sha256=sha256(evaluation),
                    diagnostic_sha256=sha256(__file__), generator_sha256=sha256(ua.__file__),
                    protocol_sha256=protocol(data), sampling=saved['sampling'], panel_ids=saved['panel_ids'],
                    shifts=[-2, 0, 2], modes=['sampling', 'greedy'], optimizer_steps=0)
    lock = out / 'contract.json'
    if lock.exists() and json.loads(lock.read_text()) != contract:
        raise ValueError('diagnostic output belongs to a different contract')
    if runtime.rank == 0:
        write_json(lock, contract)
        write_json(out / 'saved-oracle-summary.json', dict(per_singer=summarize_oracles(saved),
                   source_evaluation_sha256=contract['source_evaluation_sha256'],
                   limit='Target-assisted oracle, not deployable or a learnability result.'))
    model, _ = load_model(cfg, data, runtime.device, checkpoint)
    model.eval().requires_grad_(False)
    codec = ua.load_codec(cfg['upstream_path'], cfg['codec_path'], runtime.device)
    source = {r['id']: r for r in saved['examples']}
    rows = []
    for row_id in tqdm(saved['panel_ids'][runtime.rank::runtime.world], desc=f'MIDI diagnosis GPU{runtime.rank}'):
        row = data.rows[row_id]
        if row['split'] != 'valid':
            raise ValueError('diagnosis must not touch train/test selection')
        folder = out / safe_id(row_id)
        folder.mkdir(exist_ok=True)
        prefix, frames = data.inference_prefix(row_id, runtime.device)
        _, midi = score_conditions(row, data.vocab)
        seed = int(hashlib.sha256(f"{cfg['seed']}:{row_id}".encode()).hexdigest()[:15], 16)
        predictions = {}
        # Generate all six variants before opening any target/reference artifact.
        for mode in ('sampling', 'greedy'):
            for shift in (0, -2, 2):
                name = f'{mode}_{shift:+d}'
                condition = transpose_prefix(prefix, data.vocab['tokens'], shift)
                with runtime.amp():
                    predictions[name] = ua.generate(model, condition, frames, data.vocab['tokens'].index('<audio_0>'),
                        mode=mode, topk=cfg['topk'], temperature=cfg['temperature'], seed=seed)
        original = source[row_id]
        token_file = Path(original['audio']['generated']).parent / 'tokens.pt'
        saved_tokens = torch.load(token_file, map_location='cpu', weights_only=True)['predicted']
        if not torch.equal(predictions['sampling_+0'].cpu(), saved_tokens):
            raise ValueError(f'zero-shift sampling failed saved-token reproduction: {row_id}')
        raw, rate = sf.read(original['audio']['reference'], dtype='float32')
        if rate != 16000 or raw.ndim != 1:
            raise ValueError('saved reference audio must be mono 16 kHz')
        samples = round(math.fsum(row['phone_durations']) * 16000)
        reference_f0 = world_f0(score_clock_audio(raw, frames), 16000, frames)
        score = midi.numpy()
        score_f0 = np.where(score > 0, 440 * 2 ** ((score - 69) / 12), 0.)
        item = dict(id=row_id, singer=row['singer'], language=row['language'], source_token_sha256=sha256(token_file),
                    sampling_zero_tokens_identical=True, variants={}, shift_response={},
                    reference_score_calibration=pitch_details(score_f0, reference_f0))
        tracks = {}
        for name, codes in predictions.items():
            mode, shift_text = name.split('_')
            shift = int(shift_text)
            wave = codec.decode(codes)[0, 0, :samples].cpu().numpy()
            path = folder / f'{name}.wav'
            sf.write(path, wave, 16000, subtype='FLOAT')
            f0 = world_f0(wave, 16000, frames)
            tracks[name] = f0
            shifted_score = score_f0 * 2 ** (shift / 12)
            # Compare shifted audio to the shifted score, not to unchanged reference audio.
            metrics = {'score_' + k: v for k, v in pitch_details(shifted_score, f0).items()}
            if shift == 0:
                metrics.update({'reference_' + k: v for k, v in pitch_details(reference_f0, f0).items()})
            variant = dict(metrics=metrics, audio=str(path.resolve()))
            if cfg.get('pronunciation', {}).get('enabled'):
                transcript, _ = pronunciation.transcribe(path, dict(model_path=cfg['asr_model_path'], cache_path=cfg['asr_cache_path']), runtime.local_rank)
                lyrics = pronunciation.normalize(row['lyrics'])
                errors = pronunciation.edit_errors(lyrics, transcript['normalized'])
                variant['asr'] = dict(transcript=transcript, characters=len(lyrics), errors=errors, cer=errors / len(lyrics))
            item['variants'][name] = variant
        for mode in ('sampling', 'greedy'):
            for shift in (-2, 2):
                name = f'{mode}_{shift:+d}'
                item['shift_response'][name] = shift_response(tracks[f'{mode}_+0'], tracks[name], score, shift)
        mask = (reference_f0 > 0) & (tracks['sampling_+0'] > 0) & (tracks['greedy_+0'] > 0)
        common = {'reference_voiced_coverage': float(mask.sum() / max(1, (reference_f0 > 0).sum()))}
        for mode in ('sampling', 'greedy'):
            cents = 1200 * np.log2(tracks[f'{mode}_+0'][mask] / reference_f0[mask])
            common[f'{mode}_within50'] = float(np.mean(np.abs(cents) <= 50)) if len(cents) else None
            common[f'{mode}_rmse_cents'] = float(np.sqrt(np.mean(cents ** 2))) if len(cents) else None
        item['greedy_vs_sampling_common_support'] = common
        torch.save({k: v.cpu() for k, v in predictions.items()}, folder / 'tokens.pt')
        np.savez(folder / 'f0.npz', reference=reference_f0, score_midi=score, **tracks)
        write_json(folder / 'diagnosis.json', item)
        rows.append(item)
    rows = sum(runtime.gather(rows), [])
    if sorted(r['id'] for r in rows) != sorted(saved['panel_ids']):
        raise ValueError('incomplete diagnostic panel')
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError('checkpoint changed during diagnosis')
    if runtime.rank == 0:
        report = dict(contract=contract, per_singer=summarize_variants(rows),
                      saved_oracles=summarize_oracles(saved), examples=sorted(rows, key=lambda r: r['id']),
                      limit='Fixed validation panel, one matched sampling seed per phrase; no test or optimizer use.',
                      pitch_clock='Annotation 50 Hz frames; zero-pad short reference tails, no shift/stretch. Extra reference endpoint is outside score clock.')
        write_json(out / 'diagnosis.json', report)
        print(json.dumps(report['per_singer'], ensure_ascii=False), flush=True)


def main():
    from .uniaudio_data import load_recipe, PreparedData
    from .uniaudio_train import Runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--evaluation', required=True)
    args = parser.parse_args()
    cfg = load_recipe(args.config, args.workspace)
    data = PreparedData(cfg['prepared_path'], cfg['cache_path'])
    runtime = Runtime()
    try:
        diagnose(cfg, data, args.checkpoint, args.evaluation, args.output, runtime)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
