"""Explicit codec clock and target-free score input for the official adapter."""
import math
import sys
import numpy as np
import torch


def duration_counts(durations, frames, samples, sr, hop):
    ds = np.asarray(durations, dtype=np.float64)
    if not len(ds) or (ds < 0).any() or not np.isfinite(ds).all() or abs(ds.sum()-samples/sr) > .04:
        raise ValueError('invalid annotation duration; refusing time stretch')
    if frames != math.ceil(samples/hop):
        raise ValueError('native codec frame mismatch')
    index = np.minimum(np.searchsorted(np.cumsum(ds),(np.arange(frames)+.5)*hop/sr,side='right'),len(ds)-1)
    return torch.from_numpy(np.bincount(index,minlength=len(ds))).long()


def score_inputs(row, counts, vocab, device):
    maps = {key:{v:i for i,v in enumerate(values)} for key,values in vocab.items()}
    phones = [maps['phones'].get(row['language']+'::'+p,maps['phones']['<UNK>'])+1 for p in row['phones']]
    notes = row['score'].get('ep_pitches', row['score'].get('notes'))
    return dict(phones=torch.tensor([phones],device=device), midi=torch.tensor([notes],device=device),
                durations=counts.to(device).long()[None],
                singer=torch.tensor([maps['singers'][row['singer']]],device=device),
                language=torch.tensor([maps['languages'][row['language']]],device=device))


def world_f0(audio, sr, frames=None, hop=320):
    import pyworld
    x = np.ascontiguousarray(audio,dtype=np.float64)
    f0,t = pyworld.dio(x,sr,f0_floor=60.,f0_ceil=1500.,frame_period=1000.*hop/sr)
    f0 = pyworld.stonemask(x,f0,t,sr)
    if frames is not None:
        # Match official Dio: truncate the WORLD endpoint, keep its native clock.
        if len(f0) < frames: raise ValueError('WORLD track shorter than native codec frames')
        f0 = f0[:frames]
    return f0


def continuous_lf0(raw_f0, frames):
    """Original Dio interpolation/logging, BEFORE truncating its endpoint."""
    from .official_toksing import upstream_root
    root = str(upstream_root())
    if root not in sys.path: sys.path.insert(0, root)
    from espnet2.tts.feats_extract.dio import Dio
    f0 = Dio._convert_to_continuous_f0(np.asarray(raw_f0,dtype=np.float64).copy())
    nonzero = f0 != 0
    f0[nonzero] = np.log(f0[nonzero])
    return Dio._adjust_num_frames(torch.from_numpy(f0).float(), frames).numpy()


def pitch_metrics(reference, generated):
    a,b = np.asarray(reference), np.asarray(generated)
    if a.shape != b.shape:
        raise ValueError('pitch trajectories must use the same clock')
    av,bv = a>0,b>0; joint=av & bv
    cents = 1200*np.log2(b[joint]/a[joint])
    return dict(f0_rmse_cents=float(np.sqrt(np.mean(cents*cents))) if joint.any() else None,
                within_50_cents=float(np.mean(np.abs(cents)<=50)) if joint.any() else 0.,
                voiced_recall=float(joint.sum()/max(1,av.sum())),
                voicing_error=float(np.mean(av!=bv)),joint_voiced_fraction=float(joint.mean()))


def paired_pitch_metrics(reference, first, second):
    """Compare errors on the SAME voiced frames; report lost coverage separately."""
    a,b,c=(np.asarray(x) for x in (reference,first,second))
    if a.shape!=b.shape or a.shape!=c.shape: raise ValueError('pitch clock mismatch')
    mask=(a>0)&(b>0)&(c>0)
    coverage=float(mask.sum()/max(1,(a>0).sum()))
    def metrics(x):
        cents=1200*np.log2(x[mask]/a[mask])
        return dict(common_support_f0_rmse_cents=float(np.sqrt(np.mean(cents*cents))) if mask.any() else None,
                    common_support_reference_voiced_coverage=coverage)
    return metrics(b),metrics(c)


def synthesis_gate(metrics):
    # Frozen operational pilot thresholds; passing is NOT a MOS/intelligibility claim.
    thresholds=dict(sc_max=.9,f0_rmse_cents_max=300.,within_50_cents_min=.5,
                    voiced_recall_min=.7,voicing_error_max=.2)
    checks=[]
    for m in metrics:
        checks.append(bool(m.get('f0_rmse_cents') is not None and
                           m['f0_rmse_cents']<thresholds['f0_rmse_cents_max'] and
                           m['mrstft_spectral_convergence']<thresholds['sc_max'] and
                           m['within_50_cents']>=thresholds['within_50_cents_min'] and
                           m['voiced_recall']>=thresholds['voiced_recall_min'] and
                           m['voicing_error']<=thresholds['voicing_error_max']))
    return dict(passed=bool(checks) and all(checks), checks=checks, thresholds=thresholds,
                scope='operational pitch/spectral pilot only; lyric intelligibility still requires evaluation')
