import importlib.util

import numpy as np
import pytest
import torch


def module():
    assert importlib.util.find_spec('rrq.uniaudio_diagnose'), 'inference-only diagnosis missing'
    from rrq import uniaudio_diagnose
    return uniaudio_diagnose


def test_transposition_keeps_rests_prompt_and_phones_and_rejects_overflow():
    mod = module()
    tokens = ['pad', '<sing_midi_start>', '<sing_midi_end>'] + [f'<sing_midi_{i}>' for i in range(138)]
    seq = torch.tensor([[99, 99, 99, 1, 1, 1, 3, 3, 3, 63, 63, 63, 2, 2, 2, 22, 22, 22]])
    shifted = mod.transpose_prefix(seq, tokens, -2)
    assert shifted.tolist() == [[99, 99, 99, 1, 1, 1, 3, 3, 3, 61, 61, 61, 2, 2, 2, 22, 22, 22]]
    assert seq[0, 9].item() == 63
    with pytest.raises(ValueError, match='range'):
        mod.transpose_prefix(seq, tokens, -60)


def test_shift_response_measures_requested_shift_on_same_voiced_frames():
    mod = module()
    a = np.array([100., 0., 200., 300.])
    b = np.array([100. * 2 ** (2 / 12), 110., 0., 300. * 2 ** (2 / 12)])
    result = mod.shift_response(a, b, np.array([60, 60, 60, 0]), 2)
    assert result['median_shift_semitones'] == pytest.approx(2.)
    assert result['within_50_cents_of_requested_shift'] == 1.
    assert result['joint_voiced_score_coverage'] == pytest.approx(1 / 3)
    assert result['base_voiced_score_coverage'] == pytest.approx(2 / 3)
    ignored = mod.shift_response(a, a, np.array([60, 60, 60, 0]), 2)
    assert ignored['median_shift_semitones'] == 0.
    assert ignored['within_50_cents_of_requested_shift'] == 0.


def test_unvoiced_outputs_never_count_as_successful_transposition():
    result = module().shift_response(np.zeros(3), np.zeros(3), np.ones(3), -2)
    assert result['median_shift_semitones'] is None
    assert result['within_50_cents_of_requested_shift'] is None
    assert result['joint_voiced_score_coverage'] == 0.


def test_oracle_summary_retains_missing_cases_and_paired_delta():
    report = {'examples': [
        {'id': 'a', 'singer': 'alto', 'oracle': {'available': True, 'arms': {
            'exposure_oracle': {'latent_mse': 4., 'voicing_error': .1, 'common_support_f0_rmse_cents': 100.},
            'rrq_oracle': {'latent_mse': 2., 'voicing_error': .3, 'common_support_f0_rmse_cents': 150.}}}},
        {'id': 'b', 'singer': 'alto', 'oracle': {'available': False, 'reason': 'frame mismatch'}}]}
    result = module().summarize_oracles(report)['alto']
    assert result['total_phrases'] == 2 and result['available_phrases'] == 1
    assert result['unavailable'][0]['id'] == 'b'
    assert result['rrq_minus_exposure']['latent_mse'] == -2.
    assert result['rrq_minus_exposure']['voicing_error'] == pytest.approx(.2)
    assert result['rrq_minus_exposure']['common_support_f0_rmse_cents'] == 50.


def test_diagnostic_output_cannot_overwrite_source_run(tmp_path):
    mod = module()
    source = tmp_path / 'baseline' / 'step-final' / 'evaluation.json'
    for output in (tmp_path, source.parent, source.parent / 'diagnosis'):
        with pytest.raises(ValueError, match='overlap'):
            mod.validate_output_directory(output, [source])
    mod.validate_output_directory(tmp_path / 'new-diagnosis', [source])


def test_short_reference_tail_is_padded_without_stretch_or_crop():
    mod = module()
    reference = np.ones(16000)
    aligned = mod.score_clock_audio(reference, 52)
    assert len(aligned) == 16640
    assert np.array_equal(aligned[:16000], reference)
    assert np.all(aligned[16000:] == 0)
    long = np.ones(17000)
    assert np.array_equal(mod.score_clock_audio(long, 52), long)
