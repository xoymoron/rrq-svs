import importlib.util
import os
from pathlib import Path

import pytest
import torch


def module():
    assert importlib.util.find_spec('rrq.uniaudio_initialization'), 'scratch initialization missing'
    from rrq import uniaudio_initialization
    return uniaudio_initialization


def test_scratch_uses_original_architecture_and_independent_new_rows(monkeypatch):
    from rrq import uniaudio as ua
    root = os.environ['UNIAUDIO_ROOT']
    tokens = ['<mega_local>', '<mega_global>'] + [str(i) for i in range(100)]
    config = dict(token_list=tokens[:96], non_acoustic_repeat=3, prefix_lm=False,
                  block_size=80, n_embd=24, n_head=4, n_layer=2)
    def forbidden(*args, **kwargs):
        raise AssertionError('scratch must not read pretrained weights')
    monkeypatch.setattr(ua, 'load_release', forbidden)
    first = module().random_generator(root, config, tokens, seed=17)
    second = module().random_generator(root, config, tokens, seed=17)
    third = module().random_generator(root, config, tokens, seed=18)
    assert type(first) is ua.upstream_model(root).MegaByteModel
    assert first.g_emb.num_embeddings == 102
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    assert not torch.equal(first.g_emb.weight, third.g_emb.weight)
    assert not torch.equal(first.g_emb.weight[-1], first.g_emb.weight[-2])
    seq = torch.randint(2, 102, (1, 24))
    assert torch.isfinite(first(seq, torch.zeros_like(seq, dtype=torch.bool))).all()


def test_warmup_inverse_sqrt_is_step_based_and_resume_independent():
    mod = module()
    cfg = dict(learning_rate=.0001, lr_schedule='warmup_inverse_sqrt', warmup_steps=100)
    assert mod.learning_rate_at(cfg, 1) == pytest.approx(.000001)
    assert mod.learning_rate_at(cfg, 100) == pytest.approx(.0001)
    assert mod.learning_rate_at(cfg, 400) == pytest.approx(.00005)
    assert mod.learning_rate_at({'learning_rate': .00001}, 10000) == .00001
    with pytest.raises(ValueError):
        mod.learning_rate_at(dict(cfg, warmup_steps=0), 1)


def test_saved_initialization_cannot_silently_switch_to_scratch():
    mod = module()
    mod.validate_initialization({'initialization': 'release'}, {'run_contract': {}})
    mod.validate_initialization({'initialization': 'scratch'}, {'run_contract': {'initialization': 'scratch'}})
    with pytest.raises(ValueError, match='initialization'):
        mod.validate_initialization({'initialization': 'scratch'}, {'run_contract': {}})
