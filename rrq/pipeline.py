"""Frozen four-arm diagnostic. Target audio is used for ORACLE targets, not synthesis."""
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from .core import make_arms, requantize, validate_frozen
from .io import load_manifest, load_prediction, sha256, write_json
from .metrics import audio_metrics, aggregate


def prepare_audio(path, gain):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('target must be nonempty, finite, mono audio; downmix explicitly')
    audio = audio[:, 0]*gain
    # No peak/RMS normalization or clipping: the declared gain is the entire policy.
    def resample(rate):
        divisor = math.gcd(sr, rate)
        return resample_poly(audio, rate//divisor, sr//divisor).astype('float32') if sr != rate else audio.copy()
    return resample(16000), resample(24000), {'source_sr':sr, 'source_samples':len(audio), 'gain':gain}


def frozen(model):
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('codec and vocoder must be frozen and in eval mode')


def state_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def run_diagnostic(manifest, output, codec, vocoder, provenance, device='cpu'):
    rows = load_manifest(manifest)
    output = Path(output)
    if output.exists():
        raise ValueError('output directory already exists; choose a new run directory')
    frozen(codec); frozen(vocoder)
    layers = codec.quantizer.vq.layers[:3]
    validate_frozen(layers)
    before = state_digest(codec)
    results = []
    start = time.perf_counter()
    for row in rows:
        pcm16, target24, preprocessing = prepare_audio(row['audio'], row['gain'])
        if not 0.2 <= len(pcm16)/16000 <= 13:
            raise ValueError('use phrase durations between 0.2 and 13 seconds')
        prediction, meta = load_prediction(row['prediction'], row)
        latent = codec.encoder(torch.from_numpy(pcm16)[None, None].to(device))
        clean = codec.quantizer.vq.encode(latent, n_q=3)
        if clean.shape[-1] != prediction.shape[-1]:
            raise ValueError(f"{row['id']}: target/prediction frame mismatch; no implicit crop")
        predicted = torch.from_numpy(prediction.copy()).unsqueeze(1).to(device)
        identity = torch.equal(requantize(latent, clean[:1], layers), clean)
        if not identity:
            raise RuntimeError('clean-prefix identity failed against native quantizer')
        arms = make_arms(latent, clean, predicted, layers)
        expected_samples = clean.shape[-1]*480
        pad = expected_samples-len(target24)
        if not 0 <= pad < 480:
            raise ValueError('target length disagrees with 50Hz codec grid')
        reference = np.pad(target24, (0, pad))
        sample = {'id':row['id'],'singer':row['singer'], 'conditioning':row['conditioning'],
                  'clean_prefix_identity':identity, 'frames':clean.shape[-1],
                  'target_sha256':sha256(row['audio']), 'prediction_sha256':sha256(row['prediction']),
                  'prediction_provenance':meta, 'preprocessing':preprocessing,
                  'right_zero_padding_24k_samples':pad, 'arms':{}}
        decoded = {}
        for name, codes in arms.items():
            # Actual Prompt-Singer unit vocoder expects [B,Q,T], NOT the RVQ sum.
            waveform = vocoder(codes.permute(1, 0, 2).contiguous())
            if waveform.shape != (1, 1, expected_samples):
                raise ValueError('unexpected vocoder shape/length; refusing to crop')
            wave = waveform[0, 0].detach().float().cpu().numpy()
            metrics = audio_metrics(reference, wave, 24000)
            reconstruction = codec.quantizer.vq.decode(codes)
            metrics['latent_mse'] = float(torch.mean((latent-reconstruction)**2))
            sample['arms'][name] = metrics
            decoded[name] = wave
        itemdir = output/row['id']
        itemdir.mkdir(parents=True)
        cache_meta = {'schema_version':1, 'id':row['id'], 'singer':row['singer'],
                      'origin':row['origin'], 'conditioning':row['conditioning'],
                      'target_sha256':sample['target_sha256'], 'gain':row['gain'],
                      'prediction_sha256':sample['prediction_sha256'],
                      'provenance':provenance, 'frame_rate':50,
                      'oracle_targets':True, 'not_inference_output':True}
        np.savez_compressed(itemdir/'targets.npz', clean=clean[:, 0].cpu().numpy(),
                            predicted=prediction, rrq=arms['rrq_oracle'][:, 0].cpu().numpy(),
                            metadata=np.array(json.dumps(cache_meta, allow_nan=False)))
        for name, wave in decoded.items():
            sf.write(itemdir/(name+'.wav'), wave, 24000, subtype='FLOAT')
        results.append(sample)
    if before != state_digest(codec):
        raise RuntimeError('codec parameters/buffers changed during frozen diagnostic')
    report = {'schema_version':1, 'origin':rows[0]['origin'],
              'is_research_result':False,
              'interpretation':'Frozen oracle diagnostic only; no trained RRQ or zero-shot claim.',
              'manifest_sha256':sha256(manifest), 'provenance':provenance,
              'metric_settings':{'f0_estimator':'normalized autocorrelation diagnostic',
                                 'f0_range_hz':[60,1500], 'f0_window_ms':80,'f0_hop_ms':10,
                                 'mrstft_fft_sizes':[512,1024,2048], 'alignment':'none',
                                 'amplitude_normalization':'none'},
              'codec_state_unchanged':True, 'wall_seconds':time.perf_counter()-start,
              'samples':results,'summary':aggregate(results)}
    write_json(output/'report.json',report)
    return report
