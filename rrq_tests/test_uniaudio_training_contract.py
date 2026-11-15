import importlib.util
import os
from pathlib import Path

import pytest
import torch


def module():
    assert importlib.util.find_spec('rrq.uniaudio_train'), 'official SVS trainer missing'
    from rrq import uniaudio_train
    return uniaudio_train


def original_model():
    from rrq import uniaudio as ua
    root=os.environ.get('UNIAUDIO_ROOT',str(Path(__file__).resolve().parents[2]/'references/UniAudio'))
    return ua.upstream_model(root).MegaByteModel(96,80,24,4,2,False,3,5,4)


def test_vocabulary_extension_keeps_every_pretrained_weight_row():
    mod=module();model=original_model()
    before={k:v.clone() for k,v in model.state_dict().items()}
    mod.extend_vocabulary(model,103,seed=17)
    for key,value in model.state_dict().items():
        if key in ('g_emb.weight','l_emb.weight','lm_head.weight'):
            assert value.shape[0]==103 and torch.equal(value[:96],before[key])
        else:assert torch.equal(value,before[key])


def test_baseline_objective_equals_official_forward_and_rrq_backpropagates():
    mod=module();model=original_model().eval()
    seq=torch.randint(6,95,(1,24));mask=torch.zeros_like(seq,dtype=torch.bool)
    wrapped=mod.SVSModel(model)
    assert torch.allclose(wrapped(seq,seq,mask),model(seq,mask),atol=1e-6)
    local=seq.clone();local[:,12]=7
    logits=wrapped(seq,local,mask)
    loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),seq)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.g_emb.weight.grad.abs().sum()>0
    assert all(not p.requires_grad for p in model.continuous_mappings.parameters())


def test_stale_bank_is_rejected_before_training(tmp_path):
    mod=module()
    expected={'common_sha256':'a','protocol_sha256':'b','seed':17}
    with pytest.raises(ValueError,match='bank'):
        mod.validate_bank({'common_sha256':'old','protocol_sha256':'b','seed':17},expected)
    mod.validate_bank(expected,expected)


def test_global_token_weighting_matches_concatenated_loss():
    mod=module()
    a=torch.tensor(6.,requires_grad=True);b=torch.tensor(20.,requires_grad=True)
    # DDP averages rank gradients. Counts 2 and 5 must use 7 globally.
    loss=(mod.ddp_loss(a,7,2)+mod.ddp_loss(b,7,2))/2
    assert loss.item()==pytest.approx(26/7)


def test_protocol_binds_prompt_chunk_and_alignment_recipe():
    from types import SimpleNamespace
    mod=module()
    keys=('manifest_sha256','codec_sha256','codec_config_sha256','encoding_contract',
          'model_config_sha256','vocabulary_sha256','checkpoint_sha256','prompts_sha256',
          'segments_sha256','panels_sha256','implementation_sha256','upstream_commit')
    recipe={key:'same' for key in keys};recipe['config']={'max_frames':400,'prompt_frames':150}
    original=mod.protocol(SimpleNamespace(recipe=recipe))
    for key in ('prompts_sha256','segments_sha256','implementation_sha256'):
        assert mod.protocol(SimpleNamespace(recipe=dict(recipe,**{key:'changed'})))!=original
    assert mod.protocol(SimpleNamespace(recipe=dict(recipe,config={'max_frames':400,'prompt_frames':100})))!=original


def test_bank_contract_binds_sampling_and_file_contents(tmp_path):
    mod=module()
    assert hasattr(mod,'bank_contract'), 'prefix sampling contract missing'
    path=tmp_path/'common.pt';path.write_bytes(b'common')
    from types import SimpleNamespace
    data=SimpleNamespace(recipe={key:'same' for key in ('manifest_sha256','codec_sha256','codec_config_sha256',
        'encoding_contract','model_config_sha256','vocabulary_sha256','checkpoint_sha256','prompts_sha256',
        'segments_sha256','panels_sha256','implementation_sha256','upstream_commit')})
    data.recipe['config']={'max_frames':400,'prompt_frames':150}
    a=mod.bank_contract({'seed':17,'topk':30,'temperature':.8},data,path)
    b=mod.bank_contract({'seed':17,'topk':30,'temperature':1.},data,path)
    assert a!=b


def test_chinese_or_stale_gate_cannot_unlock_multilingual_branches():
    mod=module()
    assert hasattr(mod,'validate_gate'), 'gate identity validation missing'
    expected=dict(checkpoint_sha256='common',protocol_sha256='current',languages=['Chinese','English'],
                  thresholds={'minimum':1},sampling={'topk':30})
    for changed in ({'languages':['Chinese']},{'protocol_sha256':'stale'},{'thresholds':{'minimum':0}}):
        with pytest.raises(ValueError,match='gate'):
            mod.validate_gate(dict(expected,operational_gate_passed=True,**changed),expected)
    mod.validate_gate(dict(expected,operational_gate_passed=True),expected)
