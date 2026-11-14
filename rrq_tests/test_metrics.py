import numpy as np
import pytest

from rrq.metrics import audio_metrics, aggregate, pitch_track


def test_identical_audio_and_silence_are_finite():
    for x in (np.zeros(24000), np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)):
        m = audio_metrics(x, x, 24000)
        assert m['mrstft_spectral_convergence'] == pytest.approx(0, abs=1e-6)
        assert m['mrstft_log_magnitude'] == pytest.approx(0, abs=1e-6)
        assert m['voicing_error'] == 0
        assert m['f0_rmse_cents'] is None or m['f0_rmse_cents'] < 1e-5


def test_pitch_shift_is_detected_and_length_mismatch_rejected():
    t = np.arange(24000) / 24000
    a, b = np.sin(2*np.pi*220*t), np.sin(2*np.pi*440*t)
    assert audio_metrics(a, b, 24000)['f0_rmse_cents'] > 1000
    with pytest.raises(ValueError): audio_metrics(a, b[:-1], 24000)


def test_aggregation_weights_singers_equally():
    rows = [{'singer':'a','arms':{'exposure':{'metric':10.0},'rrq_oracle':{'metric':0.0}}}]
    rows += [{'singer':'b','arms':{'exposure':{'metric':0.0},'rrq_oracle':{'metric':2.0}}}]*3
    result = aggregate(rows)
    assert result['singer_macro']['rrq_minus_exposure']['metric'] == -4.0


@pytest.mark.parametrize('frequency',[60,1490,1500])
def test_pitch_search_includes_declared_frequency_boundaries(frequency):
    wave=np.sin(2*np.pi*frequency*np.arange(24000)/24000)
    pitch,voiced=pitch_track(wave,24000)
    assert voiced.all()
    assert np.median(pitch)==pytest.approx(frequency,rel=.015)
