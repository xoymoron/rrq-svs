import numpy as np


def test_audition_selection_is_paired_order_independent_and_not_metric_selected():
    from rrq.tracking import audition_ids
    rows = [dict(id=f'{lang}-{i}', language=lang, metrics={'cer': i})
            for lang in ('Korean', 'English') for i in range(5)]
    chosen = audition_ids(rows, per_language=2)
    assert chosen == audition_ids(list(reversed(rows)), per_language=2)
    assert len(chosen) == 4
    for r in rows:
        r['metrics']['cer'] = 999
    assert chosen == audition_ids(rows, per_language=2)


def test_run_identity_separates_restarts_arms_and_protocols():
    from rrq.tracking import run_id
    assert run_id('v3', 'abc', 'rrq-17') == run_id('v3', 'abc', 'rrq-17')
    assert len({run_id(root, protocol, label) for root, protocol, label in
                [('v3', 'abc', 'rrq-17'), ('v2', 'abc', 'rrq-17'),
                 ('v3', 'def', 'rrq-17'), ('v3', 'abc', 'baseline-17')]}) == 4


def test_token_diagnostics_expose_collapse_per_book():
    from rrq.tracking import token_metrics
    values = token_metrics(np.array([[1, 1, 1, 1], [0, 1, 2, 3], [3, 3, 4, 4]]))
    assert values['codebook_1_unique'] == 1
    assert values['codebook_1_top_fraction'] == 1
    assert values['codebook_1_entropy_bits'] == 0
    assert values['codebook_2_unique'] == 4
    assert values['codebook_2_entropy_bits'] == 2


def test_optional_tracking_does_not_require_sdk(monkeypatch, tmp_path):
    from rrq.tracking import Tracker
    monkeypatch.delenv('WANDB_PROJECT', raising=False)
    with Tracker(tmp_path, 'noop') as tracker:
        tracker.log({'train/loss': 1.})
        assert tracker.run is None


def test_validation_phone_sensitivity_is_language_macro():
    from rrq.tracking import validation_metrics
    values = validation_metrics(dict(language_macro_ce=2., per_language={'A': 1., 'B': 3.},
        samples=[dict(language='A', ce=1., reversed_phone_ce=3.),
                 dict(language='A', ce=1., reversed_phone_ce=3.),
                 dict(language='B', ce=3., reversed_phone_ce=3.)]))
    assert values['validation/phone_reversal_ce_gap'] == 1.
    assert values['validation/language/B/phone_reversal_ce_gap'] == 0.
