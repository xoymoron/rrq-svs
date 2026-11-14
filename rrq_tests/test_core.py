import pytest
import torch

from rrq.core import requantize, make_arms
from rrq.upstream import quantizer_class


def quantizer(projected=False):
    torch.manual_seed(4)
    q = quantizer_class()(num_quantizers=3, dim=4, codebook_size=16,
                          codebook_dim=2 if projected else 4, kmeans_init=False)
    return q.eval().requires_grad_(False)


@pytest.mark.parametrize('projected', [False, True])
def test_clean_prefix_is_exact_native_encoding_and_no_state_mutation(projected):
    q = quantizer(projected)
    z = torch.randn(2, 4, 9)
    before = {k: v.clone() for k, v in q.state_dict().items()}
    native = q.encode(z, n_q=3)
    original_z = z.clone()
    actual = requantize(z, native[:1], q.layers)
    assert torch.equal(actual, native)
    assert torch.equal(z, original_z)
    for k, v in before.items():
        assert torch.equal(v, q.state_dict()[k])


def test_forced_prefix_uses_native_projected_residual_not_token_embedding():
    q = quantizer(projected=True)
    z = torch.randn(1, 4, 7)
    prefix = torch.full((1, 1, 7), 5, dtype=torch.long)
    got = requantize(z, prefix, q.layers)
    assert torch.equal(got[:1], prefix)
    residual = z - q.layers[0].decode(prefix[0])
    for k in (1, 2):
        expected = q.layers[k].encode(residual)
        assert torch.equal(got[k], expected)
        residual -= q.layers[k].decode(expected)


def test_four_arms_preserve_actual_prediction_and_target():
    q = quantizer()
    z = torch.randn(1, 4, 8)
    truth = q.encode(z, n_q=3)
    predicted = (truth + 1) % 16
    backup = predicted.clone()
    arms = make_arms(z, truth, predicted, q.layers)
    assert set(arms) == {'codec', 'model', 'exposure', 'rrq_oracle'}
    assert torch.equal(arms['model'], predicted)
    assert torch.equal(arms['exposure'][1:], truth[1:])
    assert torch.equal(arms['rrq_oracle'][:1], predicted[:1])
    assert torch.equal(predicted, backup)


@pytest.mark.parametrize('case', ['frames', 'float', 'negative', 'range', 'nan', 'train', 'full_prefix'])
def test_invalid_inputs_fail_loudly(case):
    q = quantizer()
    z = torch.randn(1, 4, 7)
    p = torch.zeros((1, 1, 7), dtype=torch.long)
    if case == 'frames': p = p[..., :-1]
    if case == 'float': p = p.float()
    if case == 'negative': p.fill_(-1)
    if case == 'range': p.fill_(16)
    if case == 'nan': z[0, 0, 0] = float('nan')
    if case == 'train': q.train()
    if case == 'full_prefix': p = p.repeat(3, 1, 1)
    with pytest.raises(ValueError):
        requantize(z, p, q.layers)
