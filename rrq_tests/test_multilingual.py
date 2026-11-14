from pathlib import Path
import numpy as np
import pytest
import torch

from rrq.multilingual import Vocabulary, conditioning, pack, export_bundle
from rrq.acoustic import core_module, extend_model

ROOT = Path(__file__).resolve().parents[1]


def row(tmp_path, language='English'):
    audio=tmp_path/'a.wav'; audio.write_bytes(b'fixture')
    return dict(id='test', corpus='gtsinger', language=language, singer='s',
                split='train', audio=str(audio), lyrics=['hello'],
                phones=['AH0_en','SP'], phone_durations=[.2,.2],
                score={'ep_pitches':[69,0]}, n_frames=20, audio_seconds=.4)


def test_append_only_vocab_and_score_not_target_f0(tmp_path):
    base=Vocabulary.original(ROOT/'data/dict_uni.txt')
    r=row(tmp_path)
    phones,pitch,bias=conditioning(r)
    vocab=base.extended(phones, list(pitch)+[bias])
    assert vocab.tokens[:len(base.tokens)]==base.tokens
    assert vocab.index['aco_3071']==base.index['aco_3071']
    assert phones[:10]==['AH0_en']*10
    assert list(pitch)==[222]*10+[0]*10 and bias==440
    assert vocab.index['AH0_en']!=vocab.index['<unk>']
    with pytest.raises(ValueError,match='unknown'): vocab.encode(['absent'])
    codes=np.zeros((3,20),dtype=np.int64)
    tokens,mask,start=pack(r,vocab,codes)
    assert len(tokens)%3==0 and mask.sum()==60
    assert tokens[start:start+3].tolist()==[vocab.index['aco_0'],vocab.index['aco_1024'],vocab.index['aco_2048']]
    prefix,_,_=pack(r,vocab)
    assert torch.equal(prefix,tokens[:start])


def test_dataset_no_oov_filter_and_aux_kept(tmp_path):
    import json
    a=row(tmp_path); b=dict(a,id='zh',language='Chinese',phones=['a','SP'])
    source=tmp_path/'manifest.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in [a,b]),encoding='utf-8')
    report=export_bundle([source],ROOT/'data/dict_uni.txt',tmp_path/'bundle')
    assert report['total']==2 and report['primary']==1 and report['auxiliary']==1
    assert report['rejected']==0
    a['score']={}
    source.write_text(json.dumps(a),encoding='utf-8')
    with pytest.raises(ValueError,match='score'): export_bundle([source],ROOT/'data/dict_uni.txt',tmp_path/'bad')


def test_real_core_extension_preserves_weights_and_forward():
    torch.set_num_threads(1)
    core=core_module()
    cfg=core.AudioboxMegabyteConfig(n_vocab=32,n_ctx=100,n_state=48,n_head=4,n_layer=1,
                                 prefix_lm=False,dropout=0,prompt_latent_length=3,text_enc_dim=8)
    model=core.AcousticLMT5Configured(cfg).eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    extend_model(model,37,seed=17)
    for k,v in original.items():
        assert torch.equal(model.state_dict()[k][:len(v)] if model.state_dict()[k].shape!=v.shape else model.state_dict()[k],v)
    x=torch.tensor([[5]*24+[35]*3+[12,13,14]])
    logits,_=model(x,target_acoustic_mask=torch.zeros_like(x,dtype=torch.bool),text_feature=torch.zeros(1,3,8))
    loss=torch.nn.functional.cross_entropy(logits[:,-3:].reshape(-1,37),x[:,-3:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.global_model.transformer.wte.weight.grad[35].abs().sum()>0


def test_fast_ar_step_matches_teacher_forced_causal_logits():
    from rrq.acoustic import next_patch_context, patch_logits
    torch.manual_seed(3)
    core=core_module()
    cfg=core.AudioboxMegabyteConfig(n_vocab=32,n_ctx=100,n_state=48,n_head=4,n_layer=1,
                                 prefix_lm=False,dropout=0,prompt_latent_length=3,text_enc_dim=8)
    model=core.AcousticLMT5Configured(cfg).eval()
    prefix=torch.randint(0,32,(1,27)); current=torch.tensor([[12,13,14]])
    prompt=torch.randn(1,3,8)
    all_tokens=torch.cat([prefix,current],dim=1)
    expected,_=model(all_tokens,target_acoustic_mask=torch.zeros_like(all_tokens,dtype=torch.bool),text_feature=prompt)
    context=next_patch_context(model,prefix,prompt)
    actual=patch_logits(model,context,current)
    torch.testing.assert_close(actual,expected[:,-3:],rtol=1e-5,atol=1e-6)


def test_runtime_schedule_and_cache_reject_stale_audio(tmp_path):
    import json
    from rrq.multilingual_run import balanced_rows, load_codes
    from rrq.io import sha256
    a=row(tmp_path); a['n_frames']=20
    b=dict(a,id='valid',split='valid',n_frames=10)
    c=dict(a,id='ja',language='Japanese')
    selected=balanced_rows([a,b,c],'train',smoke=True)
    assert {r['id'] for r in selected}=={'test','ja'}
    cached=tmp_path/'codes.npz'
    np.savez(cached,codes=np.zeros((3,20),dtype=np.int64),metadata=np.array(json.dumps(dict(
        id=a['id'],target_audio_sha256=sha256(a['audio']),frame_rate=50,codec_sha256='known'))))
    assert load_codes(cached,a,'known').shape==(3,20)
    Path(a['audio']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='stale'): load_codes(cached,a,'known')


def test_training_rejects_wrong_codec_before_forward(tmp_path):
    import json
    from rrq.multilingual_run import loss_for
    from rrq.io import sha256
    r=row(tmp_path)
    np.savez(tmp_path/'test.npz',codes=np.zeros((3,20),dtype=np.int64),metadata=np.array(json.dumps(dict(
        id=r['id'],target_audio_sha256=sha256(r['audio']),frame_rate=50,codec_sha256='wrong'))))
    with pytest.raises(ValueError,match='codec checkpoint changed'):
        loss_for(None,r,None,None,tmp_path,'cpu',codec_sha256='expected')


def test_adapter_rejects_changed_prompt():
    from rrq.multilingual_run import validate_adapter
    state=dict(base_size=3960,vocabulary_sha256='v',base_checkpoint_sha256='b',prompt_sha256='old')
    with pytest.raises(ValueError,match='prompt'):
        validate_adapter(state,3960,'v','b','new')
