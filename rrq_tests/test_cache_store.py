from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from rrq.cache_store import AcousticCache, AlignmentCache
from rrq.io import sha256


def record():
    return dict(codes=torch.arange(36,dtype=torch.int16).reshape(9,1,4),
                samples=2048,sample_rate=44100,hop=512,clean_prefix_audit=None)


def row():
    return dict(id='phrase',source_sha256='audio-sha',split='train',song_group='song',phones=['a','b'],
                phone_durations=[.02,2048/44100-.02],score={'ep_pitches':[60,62]})


def test_split_and_annotations_do_not_invalidate_audio_tokens(tmp_path):
    store=AcousticCache(tmp_path/'shared','codec-sha',contract={'test':1})
    store.put('audio-sha',record());path=store.path('audio-sha');original=sha256(path)
    align=AlignmentCache(tmp_path/'experiment-a');a=row();audio=store.read(a['source_sha256'])
    first,rebuilt=align.read(a,audio);assert rebuilt
    a['split']='valid'
    second,rebuilt=align.read(a,audio)
    assert not rebuilt and torch.equal(first,second)
    a['phone_durations']=[.01,2048/44100-.01]
    third,rebuilt=align.read(a,audio)
    assert rebuilt and not torch.equal(first,third)
    assert sha256(path)==original and torch.equal(store.read('audio-sha')['codes'],record()['codes'])
    a['phone_durations']=[1.,1.]
    with pytest.raises(ValueError,match='duration'):align.read(a,audio)
    assert sha256(path)==original
    assert not {'split','phones','frame_index','annotation_sha256','id'} & set(audio)


def test_audio_codec_or_encoding_change_requires_a_new_object(tmp_path):
    store=AcousticCache(tmp_path,'codec-a',contract={'test':1});store.put('audio-a',record())
    for changed,source in [(store,'audio-b'),(AcousticCache(tmp_path,'codec-b',contract={'test':1}),'audio-a'),
                           (AcousticCache(tmp_path,'codec-a',contract={'test':2}),'audio-a')]:
        with pytest.raises(FileNotFoundError):changed.read(source)
    assert store.path('audio-a').exists()


def test_corrupted_token_object_is_rejected_without_overwriting(tmp_path):
    store=AcousticCache(tmp_path,'codec',contract={'test':1});store.put('audio',record())
    path=store.path('audio');value=torch.load(path,weights_only=True);value['codes'][0,0,0]=77
    torch.save(value,path);before=sha256(path)
    with pytest.raises(ValueError,match='checksum'):store.read('audio')
    assert sha256(path)==before


def test_concurrent_publish_never_replaces_an_existing_object(tmp_path,monkeypatch):
    import rrq.cache_store as module
    store=AcousticCache(tmp_path,'codec',contract={'test':1})
    original=module.atomic_save
    competitor=deepcopy(record());competitor['codes'][0,0,0]=99
    def raced(value,path,exclusive=False):
        monkeypatch.setattr(module,'atomic_save',original)
        store.put('audio',competitor)
        return original(value,path,exclusive=exclusive)
    monkeypatch.setattr(module,'atomic_save',raced)
    with pytest.raises(ValueError,match='immutable'):store.put('audio',record())
    assert store.read('audio')['codes'][0,0,0]==99


def test_changed_audit_recipe_does_not_invalidate_tokens(tmp_path):
    store=AcousticCache(tmp_path,'codec',contract={'test':1})
    r=dict(record(),clean_prefix_audit={'changed_tokens':0},clean_prefix_audit_recipe=store.audit_recipe)
    store.put('audio',r);original=sha256(store.path('audio'))
    assert store.read('audio')['clean_prefix_audit']=={'changed_tokens':0}
    store.audit_recipe='changed-diagnostic-only'
    result=store.read('audio')
    assert result['clean_prefix_audit'] is None and torch.equal(result['codes'],r['codes'])
    assert sha256(store.path('audio'))==original


def test_legacy_migration_preserves_files_and_never_loads_codec(tmp_path,monkeypatch):
    from rrq.migrate_codec_cache import migrate
    from rrq.cache_store import encoding_contract,fingerprint
    import rrq.migrate_codec_cache as module
    import dac
    legacy=tmp_path/'legacy';(legacy/'cache').mkdir(parents=True)
    codec=tmp_path/'codec.pth';codec.write_bytes(b'fake-codec-never-loaded')
    a=row();manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(a)+'\n')
    contract=encoding_contract()
    protocol=dict(manifest_sha256=sha256(manifest),codec_sha256=sha256(codec),
        implementation={n:sha256(Path(module.__file__).parent/n) for n in ('dac_codec.py','toksing_data.py')},
        environment={k:contract[k] for k in ('cuda','packages')})
    protocol['sha256']=fingerprint(protocol)
    (legacy/'protocol.json').write_text(json.dumps(protocol))
    old=dict(record(),id=a['id'],source_sha256=a['source_sha256'],codec_sha256=sha256(codec),
             annotation_sha256='obsolete-split-and-annotation',frame_index=torch.zeros(4))
    path=legacy/'cache'/'phrase.pt';torch.save(old,path);before=sha256(path)
    def forbidden(*args,**kwargs):raise AssertionError('migration must never load or run the codec')
    monkeypatch.setattr(dac.DAC,'load',forbidden)
    for created in (1,0):
        result=migrate(legacy,manifest,codec,tmp_path/'shared',tmp_path/'report.json')
        assert result['encoder_calls']==0 and result['created']==created
        assert result['exact_tokens_verified']==result['legacy_files_unchanged']==1
        assert sha256(path)==before
    old['source_sha256']='different-audio';torch.save(old,path)
    with pytest.raises(ValueError,match='provenance'):migrate(legacy,manifest,codec,tmp_path/'shared',tmp_path/'report.json')
