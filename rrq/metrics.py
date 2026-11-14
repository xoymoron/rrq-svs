"""Fixed, transparent diagnostic metrics. Autocorrelation F0 is not a MOS estimator."""
from collections import defaultdict

import numpy as np
from scipy.signal import stft, correlate, find_peaks


def pitch_track(x, sr):
    window, hop = round(sr * .08), round(sr * .01)
    low, high = max(1, int(sr / 1500)), int(sr / 60)
    f0, voiced = [], []
    for start in range(0, max(1, len(x)-window+1), hop):
        frame = x[start:start+window].astype(np.float64)
        frame -= frame.mean()
        rms = np.sqrt(np.mean(frame**2))
        ac = correlate(frame, frame, mode='full', method='fft')[len(frame)-1:]
        ac /= np.maximum(1, np.arange(len(frame), 0, -1))
        ac /= max(ac[0], 1e-12)
        segment = ac[low:high+1]
        peaks, _ = find_peaks(segment)
        if rms < 1e-4 or not len(peaks):
            f0.append(0.); voiced.append(False); continue
        strength = segment[peaks]
        eligible = peaks[strength >= max(.6, .9 * strength.max())]
        if not len(eligible):
            f0.append(0.); voiced.append(False); continue
        lag = float(eligible[0] + low)
        j = int(lag)
        denom = ac[j-1] - 2*ac[j] + ac[j+1]
        if abs(denom) > 1e-12:
            lag += float(np.clip(.5*(ac[j-1]-ac[j+1])/denom, -.5, .5))
        f0.append(sr/lag); voiced.append(True)
    return np.array(f0), np.array(voiced)


def audio_metrics(reference, output, sr):
    reference, output = np.asarray(reference), np.asarray(output)
    if reference.ndim != 1 or output.shape != reference.shape or len(reference) < 2048:
        raise ValueError('metrics require equal-length mono waveforms >=2048 samples; no auto-alignment')
    if not np.isfinite(reference).all() or not np.isfinite(output).all():
        raise ValueError('nonfinite audio')
    convergence, log_distance = [], []
    for n in (512, 1024, 2048):
        opts = dict(fs=sr, nperseg=n, noverlap=n-n//4, boundary=None, padded=False)
        a, b = (np.abs(stft(x, **opts)[2]) for x in (reference, output))
        convergence.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-8)))
        log_distance.append(float(np.mean(np.abs(np.log(np.maximum(a, 1e-5))-np.log(np.maximum(b, 1e-5))))))
    a, av = pitch_track(reference, sr)
    b, bv = pitch_track(output, sr)
    shared = av & bv
    cents = 1200*np.log2(b[shared]/a[shared])
    return {'mrstft_spectral_convergence': float(np.mean(convergence)),
            'mrstft_log_magnitude': float(np.mean(log_distance)),
            'voicing_error': float(np.mean(av != bv)),
            'f0_rmse_cents': float(np.sqrt(np.mean(cents**2))) if len(cents) else None,
            'joint_voiced_fraction': float(np.mean(shared)),
            'peak_amplitude': float(np.max(np.abs(output)))}


def mean_maps(maps):
    keys = set().union(*(m.keys() for m in maps))
    return {key: float(np.mean([m[key] for m in maps if m.get(key) is not None]))
            if any(m.get(key) is not None for m in maps) else None for key in sorted(keys)}


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['singer']].append(row)
    singers = {}
    for singer, samples in grouped.items():
        arms = {arm: mean_maps([r['arms'][arm] for r in samples]) for arm in samples[0]['arms']}
        # Pair per phrase first; never subtract means with different F0 eligibility.
        for baseline in ('exposure', 'model'):
            if baseline not in arms:
                continue
            deltas = []
            for r in samples:
                a, b = r['arms']['rrq_oracle'], r['arms'][baseline]
                deltas.append({k: a[k]-b[k] if a[k] is not None and b.get(k) is not None else None for k in a})
            arms['rrq_minus_'+baseline] = mean_maps(deltas)
        singers[singer] = {'n_phrases': len(samples), 'metrics': arms}
    macro = {arm: mean_maps([s['metrics'][arm] for s in singers.values()])
             for arm in next(iter(singers.values()))['metrics']}
    return {'n_singers': len(singers), 'n_phrases': len(rows), 'per_singer': singers,
            'singer_macro': macro}
