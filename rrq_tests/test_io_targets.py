import numpy as np
import pytest
import torch

from rrq.io import parse_generation, validate_codes, load_manifest, safe_id
from rrq.targets import training_map


def test_parse_upstream_frame_order_and_offsets():
    text = 'D-abc\t<phone_start> foo <acoustic_start> <acoustic_start> <acoustic_start> aco_7 aco_1032 aco_2057 aco_3 aco_1028 aco_2053 <acoustic_end> <acoustic_end> <acoustic_end> </s>'
    result = parse_generation(text)
    assert np.array_equal(result['abc'], [[7, 3], [8, 4], [9, 5]])


@pytest.mark.parametrize('tokens', ['aco_0 aco_1024', 'aco_0 aco_0 aco_2048',
                                   'aco_0 unknown aco_2048', 'aco_0 aco_1024 aco_3072'])
def test_malformed_generation_is_not_silently_truncated(tokens):
    with pytest.raises(ValueError):
        parse_generation('D-x\t<acoustic_start> ' + tokens + ' <acoustic_end>')


def test_duplicate_ids_rejected():
    line = 'D-x\t<acoustic_start> aco_0 aco_1024 aco_2048 <acoustic_end>'
    with pytest.raises(ValueError): parse_generation(line + '\n' + line)


@pytest.mark.parametrize('bad', [np.zeros((2, 8), dtype=int), np.zeros((3, 0), dtype=int),
                                np.zeros((3, 8)), np.full((3, 8), 1024)])
def test_invalid_code_arrays_rejected(bad):
    with pytest.raises(ValueError): validate_codes(bad)


def test_training_maps_share_mixture_and_keep_context_consistent():
    clean = torch.zeros(3, 2, 5, dtype=torch.long)
    pred = torch.full_like(clean, 1)
    rrq = torch.full_like(clean, 2); rrq[0] = pred[0]
    choose = torch.tensor([True, False])
    exposure = training_map(clean, pred, rrq, 'exposure', choose)
    corrected = training_map(clean, pred, rrq, 'rrq', choose)
    assert torch.equal(exposure[0], corrected[0])
    assert torch.equal(exposure[1:], clean[1:])
    assert torch.equal(corrected[:, 0], rrq[:, 0])
    assert torch.equal(corrected[:, 1], clean[:, 1])
    assert torch.equal(training_map(clean, pred, rrq, 'clean', choose), clean)


def test_manifest_requires_origin_and_alignment_attestation(tmp_path):
    p = tmp_path / 'manifest.jsonl'
    p.write_text('{"id":"x","singer":"s","audio":"x.wav","prediction":"x.npz"}\n')
    with pytest.raises(ValueError): load_manifest(p)


def test_windows_filename_collisions_are_rejected_before_export():
    with pytest.raises(ValueError): safe_id('sample.')
    line='\t<acoustic_start> aco_0 aco_1024 aco_2048 <acoustic_end>'
    with pytest.raises(ValueError): parse_generation('D-clip'+line+'\nD-CLIP'+line)
