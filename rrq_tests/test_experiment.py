import numpy as np
import pytest
import torch

from rrq.acoustic import core_module, next_patch_context, patch_logits


def tiny(vocab=32):
    torch.set_num_threads(1); torch.manual_seed(1)
    core=core_module()
    cfg=core.AudioboxMegabyteConfig(n_vocab=vocab,n_ctx=200,n_state=48,n_head=4,n_layer=2,
        prefix_lm=False,dropout=0,prompt_latent_length=3,text_enc_dim=8)
    return core.AcousticLMT5Configured(cfg).eval()


def test_cached_global_matches_full_ar_for_multiple_steps():
    from rrq.cached_ar import GlobalCache
    model=tiny(); prefix=torch.randint(0,32,(1,30)); prompt=torch.randn(1,3,8)
    cache=GlobalCache(model)
    context=cache.prefill(prefix,prompt)
    for _ in range(5):
        torch.testing.assert_close(context,next_patch_context(model,prefix,prompt),rtol=1e-5,atol=2e-6)
        patch=torch.randint(0,32,(1,3)); prefix=torch.cat([prefix,patch],1)
        context=cache.append(patch)


def test_fixed_coarse_codes_are_used_at_every_step():
    from rrq.cached_ar import generate
    from rrq.multilingual import Vocabulary
    vocab=Vocabulary(['x']+[f'aco_{i}' for i in range(3072)])
    model=tiny(len(vocab.tokens)); prefix=torch.ones(1,30,dtype=torch.long); prompt=torch.zeros(1,3,8)
    coarse=np.array([5,47,1023])
    result=generate(model,prefix,prompt,vocab,3,coarse)
    assert np.array_equal(result[0],coarse)
    assert result.shape==(3,3) and result.min()>=0 and result.max()<1024
    with pytest.raises(ValueError): generate(model,prefix,prompt,vocab,3,np.array([5]))


def test_matched_maps_share_mixture_and_preserve_only_coarse_difference():
    from rrq.experiment import arm_codes, epoch_rows, check_splits
    clean=np.zeros((3,6),dtype=np.int64); predicted=np.full_like(clean,3)
    corrected=np.full_like(clean,5); corrected[0]=3
    rows=[dict(id=str(i),split='train',song_group=str(i),audio_sha256=str(i)) for i in range(9)]
    assert epoch_rows(rows,17,2)==epoch_rows(rows,17,2)
    assert epoch_rows(rows,17,2)!=epoch_rows(rows,17,3)
    e=arm_codes(clean,predicted,corrected,'exposure',True)
    r=arm_codes(clean,predicted,corrected,'rrq',True)
    assert np.array_equal(e[0],r[0]) and (e[1:]==0).all() and (r[1:]==5).all()
    assert np.array_equal(arm_codes(clean,predicted,corrected,'baseline',True),clean)
    with pytest.raises(ValueError,match='split'):
        check_splits(rows+[dict(rows[0],id='leak',split='test')])


def test_branch_training_freezes_global_and_embedding_parameters():
    from rrq.experiment import trainable_parameters
    model=tiny(); names=trainable_parameters(model,'branch',32)
    assert names and all(not n.startswith('global_model') for n in names)
    assert 'lm_head.weight' in names and 'linear_map.weight' in names
    assert not model.global_model.transformer.wte.weight.requires_grad


def test_character_error_keeps_language_characters_and_reports_insertions():
    from rrq.experiment_metrics import text_normalize, character_error
    assert text_normalize('안녕, Hello! É')=='안녕helloé'
    assert character_error('abc','axc')['cer']==pytest.approx(1/3)
    assert character_error('a','aaaa')['cer']==3
