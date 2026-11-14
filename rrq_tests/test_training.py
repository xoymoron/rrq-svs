import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rrq.io import sha256
from rrq.training import CachedTargets, suffix_mask


def cache(tmp_path, origin='model_rollout'):
    audio=tmp_path/'audio.wav'; audio.write_bytes(b'target source hash fixture')
    folder=tmp_path/'cache'/'x'; folder.mkdir(parents=True)
    clean=np.zeros((3,8),dtype=np.int64)
    predicted=np.ones_like(clean)
    corrected=np.full_like(clean,2); corrected[0]=predicted[0]
    meta={'schema_version':1,'id':'x','origin':origin,'frame_rate':50,'gain':1.,
          'target_sha256':sha256(audio),'oracle_targets':True}
    np.savez_compressed(folder/'targets.npz',clean=clean,predicted=predicted,rrq=corrected,
                        metadata=np.array(json.dumps(meta)))
    return audio


def test_cached_training_map_and_context_use_same_recomputed_suffix(tmp_path):
    audio=cache(tmp_path)
    cfg={'cache_dir':str(tmp_path/'cache'),'arm':'rrq','rho':1.,'seed':5}
    rrq=CachedTargets(cfg).get('x',audio,8)
    exposure=CachedTargets({**cfg,'arm':'exposure'}).get('x',audio,8)
    assert np.array_equal(rrq[0],exposure[0])
    assert (rrq[1:]==2).all()
    assert (exposure[1:]==0).all()
    assert (CachedTargets({**cfg,'rho':0}).get('x',audio,8)==0).all()


def test_cached_training_rejects_stale_audio_wrong_frames_and_synthetic(tmp_path):
    audio=cache(tmp_path)
    targets=CachedTargets({'cache_dir':str(tmp_path/'cache'),'arm':'rrq'})
    with pytest.raises(ValueError,match='frame'): targets.get('x',audio,7)
    audio.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'): targets.get('x',audio,8)
    other=tmp_path/'synthetic'; other.mkdir()
    audio2=cache(other,origin='synthetic_smoke')
    synthetic=CachedTargets({'cache_dir':str(other/'cache'),'arm':'rrq'})
    with pytest.raises(ValueError,match='synthetic'): synthetic.get('x',audio2,8)


def test_cached_training_checks_expected_codec_provenance(tmp_path):
    audio=cache(tmp_path)
    targets=CachedTargets({'cache_dir':str(tmp_path/'cache'),'expected_codec_sha256':'official'})
    with pytest.raises(ValueError,match='codec checkpoint'): targets.get('x',audio,8)


def test_loss_mask_only_trains_suffix_tokens_in_acoustic_region():
    # A dictionary need not assign consecutive IDs; classify by explicit token names.
    class Dictionary:
        indices={f'aco_{i}':4000+i for i in range(3072)}
        def index(self,s): return self.indices[s]
    target=torch.tensor([[0,4000,5024,6048,4001,5025,6049,9]])
    acoustic=torch.tensor([[False,True,True,True,True,True,True,False]])
    mask=suffix_mask(target,acoustic,Dictionary())
    assert torch.equal(mask,torch.tensor([[False,False,True,True,False,True,True,False]]))
    logits=torch.randn(1,8,7080,requires_grad=True)
    torch.nn.functional.cross_entropy(logits[mask],target[mask]).backward()
    assert logits.grad[~mask].abs().sum()==0
    assert logits.grad[mask].abs().sum()>0
