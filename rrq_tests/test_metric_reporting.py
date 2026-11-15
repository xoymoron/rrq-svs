import pytest


def test_asr_reporting_separates_language_macro_character_weighting_and_caps():
    from rrq.experiment_summary import summarize_rows
    rows = [dict(id='a', language='A', asr_hit_length_cap=False,
                 metrics=dict(cer=0.1, character_edits=10, reference_characters=100)),
            dict(id='b', language='A', asr_hit_length_cap=False,
                 metrics=dict(cer=0.1, character_edits=10, reference_characters=100)),
            dict(id='c', language='B', asr_hit_length_cap=True,
                 metrics=dict(cer=10., character_edits=10, reference_characters=1))]
    result = summarize_rows(rows)
    # Existing headline must remain unchanged. A short failed phrase is visible,
    # rather than silently dropped or allowed to masquerade as a typical sample.
    assert result['language_macro']['cer'] == pytest.approx(5.05)
    assert result['asr_diagnostics']['character_weighted_cer'] == pytest.approx(30/201)
    assert result['asr_diagnostics']['median_phrase_cer'] == 0.1
    assert result['asr_diagnostics']['cap_count'] == 1
    assert result['per_language_counts'] == {'A': 2, 'B': 1}
    assert result['metric_support']['f0_rmse_cents'] == 0


def test_metric_reporting_refuses_duplicates_and_nonfinite_metrics():
    from rrq.experiment_summary import summarize_rows
    row = dict(id='x', language='A', metrics={'cer': 1.})
    with pytest.raises(ValueError, match='duplicate'):
        summarize_rows([row, row])
    with pytest.raises(ValueError, match='nonfinite'):
        summarize_rows([dict(row, metrics={'cer': float('nan')})])


def test_disk_error_reports_actual_space_budget(monkeypatch, tmp_path):
    from collections import namedtuple
    from rrq.data_download import free_space
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr('rrq.data_download.shutil.disk_usage', lambda _: usage(100, 80, 20))
    with pytest.raises(ValueError, match=r'free=20.*required=10.*reserve=12'):
        free_space(tmp_path, required=10, reserve=12)
