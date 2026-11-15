"""Parity with the imported official model, rather than a second implementation."""
import copy
import torch
import pytest

from rrq.official_toksing import OfficialTokSing, upstream_config, source_provenance

torch.set_num_threads(1)


def tiny():
    c = upstream_config(phones=12, singers=2, languages=2, codebooks=3, entries=8)
    c.update(adim=16, aheads=2, elayers=1, dlayers=1, eunits=32, dunits=32,
             text_encoder_blocks=1, text_encoder_ffn_expand=2,
             duration_predictor_chans=16)
    return c


def sample():
    return dict(phones=torch.tensor([[1, 2, 3]]), midi=torch.tensor([[60, 62, 64]]),
                durations=torch.tensor([[2, 3, 2]]), singer=torch.tensor([0]),
                language=torch.tensor([0]))


def test_disabled_adapter_matches_actual_upstream_logits_loss_and_gradients():
    # Original TokSing clamps LF0 at zero; make the synthetic gradient fixture
    # deterministic so an all-negative random initialization cannot kill it.
    torch.manual_seed(17)
    m = OfficialTokSing(tiny(), prefix_conditioning=False).eval()
    from espnet2.svs.discrete.toksing import TokSing
    assert type(m.base) is TokSing
    original = copy.deepcopy(m.base)
    codes = torch.randint(0, 8, (1, 3, 7))
    lf0 = torch.full((1, 7, 1), 5.5)
    args = m.training_arguments(sample(), codes, lf0)
    a = original(**args, joint_training=True)
    b = m(score=sample(), codes=codes, lf0=lf0)
    torch.testing.assert_close(a[0], b['loss'], rtol=0, atol=0)
    torch.testing.assert_close(a[2].view(1, 7, 3, 8).permute(0, 2, 1, 3), b['logits'], rtol=0, atol=0)
    b['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.base.f0_predictor.parameters())
    assert m.base.lf0_mapping.weight.grad.abs().sum() > 0
    assert m.base.duration_predictor.linear.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_prefix_adapter_initially_preserves_projection_then_uses_prefix_only_for_suffix():
    torch.manual_seed(17)
    plain = OfficialTokSing(tiny(), prefix_conditioning=False).eval()
    torch.manual_seed(17)
    m = OfficialTokSing(tiny(), prefix_conditioning=True).eval()
    codes = torch.randint(0, 8, (1, 3, 7)); lf0 = torch.full((1, 7, 1), 5.5)
    a = plain(score=sample(), codes=codes, lf0=lf0)['logits']
    b = m(score=sample(), codes=codes, lf0=lf0)['logits']
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for emb in m.base.linear_projection.prefix_embeddings:
        torch.nn.init.normal_(emb.weight, std=.3)
    altered = codes.clone(); altered[:, 0] = (altered[:, 0] + 1) % 8
    a = m(score=sample(), codes=codes, lf0=lf0)['logits']
    b = m(score=sample(), codes=codes, lf0=lf0, context=altered)['logits']
    torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
    assert not torch.allclose(a[:, 1:], b[:, 1:])


def test_generation_calls_official_inference_uses_predicted_pitch_and_no_target_tokens():
    m = OfficialTokSing(tiny(), prefix_conditioning=True).eval()
    seen = []
    hook = m.base.lf0_mapping.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    output = m.generate(**sample())
    hook.remove()
    assert output['codes'].shape == (1, 3, 7)
    torch.testing.assert_close(seen[0][0, :, 0], output['lf0'])
    assert torch.isfinite(output['lf0']).all()
    with pytest.raises(TypeError):
        m.generate(**sample(), lf0=torch.zeros(7))
    assert not m.base.length_regulator._forward_pre_hooks
    assert m.base.linear_projection.context is None


def test_original_import_is_pinned_and_source_tree_clean():
    p = source_provenance()
    assert p['commit'] == '5f142ef90bf87ee0f4c2d6907f7a9462b3854316'
    assert p['modified_tracked_files'] == []
    assert p['toksing_sha256']
