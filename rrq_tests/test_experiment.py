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


def test_paired_summary_refuses_mismatched_items():
    from rrq.experiment_summary import paired_song_ci
    a={'x':dict(song_group='song',metrics={'cer':.2})}
    b={'x':dict(song_group='song',metrics={'cer':.4})}
    assert paired_song_ci(a,b,'cer')['song_macro_difference']==pytest.approx(-.2)
    with pytest.raises(ValueError,match='IDs'): paired_song_ci(a,{},'cer')


def test_batched_generation_matches_serial_with_unequal_prefix_lengths():
    from rrq.cached_ar import generate, generate_batch
    from rrq.multilingual import Vocabulary
    vocab=Vocabulary(['x']+[f'aco_{i}' for i in range(3072)])
    model=tiny(len(vocab.tokens)); prompt=torch.randn(1,3,8)
    prefixes=[torch.ones(1,30,dtype=torch.long),torch.ones(1,42,dtype=torch.long)]
    coarse=[np.array([1,2,3]),np.array([15,25])]
    actual=generate_batch(model,prefixes,prompt,vocab,[3,2],coarse)
    for i in range(2):
        expected=generate(model,prefixes[i],prompt,vocab,len(coarse[i]),coarse[i])
        assert np.array_equal(actual[i],expected)


def test_resume_recovers_unpublished_best_from_last(tmp_path):
    import json
    from rrq.experiment import recover_best
    data=dict(protocol_sha256='p',parameters={'x':torch.ones(1)},epoch=2,metrics={'language_macro_ce':1.0},optimizer={})
    torch.save(data,tmp_path/'last.pt')
    torch.save(dict(data,epoch=1,metrics={'language_macro_ce':2.0}),tmp_path/'best.pt')
    (tmp_path/'best.json').write_text(json.dumps({'epoch':1,'value':2.0}))
    assert recover_best(tmp_path,'p')==1.0
    assert json.loads((tmp_path/'best.json').read_text())['epoch']==2


def test_render_pair_requires_codes_and_matching_provenance(tmp_path):
    import json
    import soundfile as sf
    from rrq.experiment import render_pair_valid
    from rrq.io import sha256
    wave=tmp_path/'x.wav'; sf.write(wave,np.zeros(480*3),24000,subtype='FLOAT')
    assert not render_pair_valid(wave,3,{'protocol_sha256':'p'})
    np.savez(wave.with_suffix('.npz'),codes=np.zeros((3,3),dtype=np.int64),
             metadata=np.array(json.dumps(dict(protocol_sha256='p',waveform_sha256=sha256(wave)))))
    assert render_pair_valid(wave,3,{'protocol_sha256':'p'})
    assert not render_pair_valid(wave,3,{'protocol_sha256':'wrong'})
    assert not render_pair_valid(wave,4,{'protocol_sha256':'p'})


def test_sampling_reproducible_per_item_across_batch_composition():
    from rrq.cached_ar import generate_batch
    from rrq.multilingual import Vocabulary
    vocab=Vocabulary(['x']+[f'aco_{i}' for i in range(3072)])
    model=tiny(len(vocab.tokens)); prompt=torch.zeros(1,3,8)
    p=[torch.ones(1,30,dtype=torch.long),torch.ones(1,42,dtype=torch.long)]
    both=generate_batch(model,p,prompt,vocab,[5,4],temperature=.8,sample_ids=['a','b'],sampling_seed=7)
    alone=generate_batch(model,[p[1]],prompt,vocab,[4],temperature=.8,sample_ids=['b'],sampling_seed=7)
    assert np.array_equal(both[1],alone[0])
    changed=generate_batch(model,[p[1]],prompt,vocab,[4],temperature=.8,sample_ids=['b'],sampling_seed=8)
    assert not np.array_equal(alone[0],changed[0])
