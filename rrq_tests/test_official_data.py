import numpy as np
import pytest
import torch
from rrq.official_data import duration_counts, score_inputs, synthesis_gate, continuous_lf0


def test_counts_preserve_codec_clock_with_zero_length_phone_and_endpoint_padding():
    d = duration_counts([.02,0,.03], frames=4, samples=1200, sr=24000, hop=320)
    assert d.tolist() == [1,0,3]
    assert d.sum() == 4
    with pytest.raises(ValueError, match='duration'):
        duration_counts([1.],4,1200,24000,320)
    with pytest.raises(ValueError, match='frame'):
        duration_counts([.05],5,1200,24000,320)


def test_score_has_no_audio_f0_or_token_dependency_and_padding_is_reserved():
    row=dict(phones=['a','absent'],language='Korean',singer='s1',score={'notes':[60,0]})
    vocab=dict(phones=['<UNK>','Korean::a'],singers=['s1'],languages=['Korean'])
    score=score_inputs(row,torch.tensor([2,3]),vocab,'cpu')
    assert set(score)=={'phones','midi','durations','singer','language'}
    assert score['phones'].tolist()==[[2,1]]


def test_gate_rejects_spectral_only_improvement_and_missing_voiced_pitch():
    good=dict(mrstft_spectral_convergence=.5, f0_rmse_cents=100.,
              within_50_cents=.8, voiced_recall=.9, voicing_error=.1)
    assert synthesis_gate([good])['passed']
    assert not synthesis_gate([{**good,'f0_rmse_cents':1800.}])['passed']
    assert not synthesis_gate([{**good,'f0_rmse_cents':None,'voiced_recall':0.}])['passed']
    assert not synthesis_gate([{**good,'voiced_recall':.1}])['passed']


def test_lf0_keeps_official_continuous_interpolation_without_changing_raw_voicing():
    raw = np.array([0.,100.,0.,200.,0.])
    result = continuous_lf0(raw, 4)
    np.testing.assert_allclose(result, np.log([100.,100.,150.,200.]), rtol=1e-6)
    assert raw.tolist() == [0.,100.,0.,200.,0.]
    np.testing.assert_array_equal(continuous_lf0(np.zeros(5),4), np.zeros(4))
