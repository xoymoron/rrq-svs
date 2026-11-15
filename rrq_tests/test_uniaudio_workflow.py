"""Scientific contracts for the official UniAudio SVS-only workflow."""
import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch


def api():
    assert importlib.util.find_spec('rrq.uniaudio_data'), 'multilingual SVS preparation missing'
    return importlib.import_module('rrq.uniaudio_data')


def vocabulary():
    tokens = ['<pad>', '<start>', '<end>', '<sing_task>', '<sing_phone_start>',
              '<sing_phone_end>', '<sing_midi_start>', '<sing_midi_end>',
              '<audio_prompt_start>', '<audio_prompt_end>', '<audio_start>', '<audio_end>',
              '<mega_local>', '<mega_global>']
    tokens += ['reserved'] * (128-len(tokens))
    tokens += [f'<audio_{i}>' for i in range(3073)]
    tokens += [f'<sing_phone_{i}>' for i in range(3)]
    tokens += [f'<sing_midi_{i}>' for i in range(138)]
    return tokens


def row(uid, split='train', song='A', language='Korean', phone='a_ko'):
    return dict(id=uid, split=split, song_group=song, singer='s1', language=language,
                phones=[phone], phone_durations=[4.], score={'ep_pitches':[60]},
                source_sha256=uid*64, audio_seconds=4.)


def test_vocab_preserves_native_ids_and_never_learns_validation_phones():
    mod = api(); old = vocabulary()
    train = row('a'); held = row('b', 'valid', 'B', phone='only_in_valid')
    v = mod.build_vocabulary([train, held], old, {'a':0,'AP':1,'SP':2})
    assert v['tokens'][:len(old)] == old
    assert 'Korean::only_in_valid' not in v['phone_ids']
    assert mod.phone_id(held, 'only_in_valid', v) == v['unknown_id']
    assert mod.phone_id(row('c', language='Chinese'), 'a_zh', v) == old.index('<sing_phone_0>')


def test_prompt_comes_from_other_train_song_and_chunks_cover_every_frame():
    mod = api()
    rows = [row('a'), row('b',song='B'), row('c','valid','C'), row('d','test','D')]
    prompts = mod.assign_prompts(rows, 17)
    by_id = {r['id']:r for r in rows}
    for r in rows:
        p = by_id[prompts[r['id']]]
        assert p['split']=='train' and p['song_group']!=r['song_group']
    assert mod.chunk_ranges(901,400)==[(0,400),(400,800),(800,901)]
    with pytest.raises(ValueError,match='prompt'):
        mod.assign_prompts([row('a')],17)


def test_acoustic_cache_reuses_split_and_annotation_changes_but_rejects_corruption(tmp_path):
    mod=api(); cache=mod.NativeCache(tmp_path,'codec',{'clock':50})
    record=cache.put('source',torch.ones(1,8,4),torch.zeros(3,1,4,dtype=torch.long),1280)
    assert cache.read('source')['key']==record['key']
    # Neither split nor annotation is an argument to cache identity.
    assert mod.NativeCache(tmp_path,'codec',{'clock':50}).path('source')==cache.path('source')
    assert mod.NativeCache(tmp_path,'other-codec',{'clock':50}).path('source')!=cache.path('source')
    with pytest.raises(ValueError,match='immutable'):
        cache.put('source',torch.zeros(1,8,4),torch.zeros(3,1,4,dtype=torch.long),1280)
    bad=torch.load(cache.path('source'),weights_only=True);bad['codes'][0,0,0]=1024
    torch.save(bad,cache.path('source'))
    with pytest.raises(ValueError):cache.read('source')


def test_duplicate_audit_ignores_unused_technique_but_keeps_score_conflicts():
    mod=api(); a=row('a'); b=row('b'); a['pcm_sha256']=b['pcm_sha256']='same'
    a['techniques']={'vibrato':[0]};b['techniques']={'vibrato':[1]}
    kept,duplicates,rejected=mod.deduplicate_svs([a,b])
    assert len(kept)==1 and len(duplicates)==1 and not rejected
    b['score']['ep_pitches']=[61]
    kept,duplicates,rejected=mod.deduplicate_svs([a,b])
    assert not kept and len(rejected)==2


def test_depth_rrq_changes_suffix_labels_not_q0_or_condition():
    assert importlib.util.find_spec('rrq.uniaudio_train'), 'SVS objective missing'
    mod=importlib.import_module('rrq.uniaudio_train')
    seq=torch.tensor([[1,1,1,128,1153,2178,131,1156,2181]])
    bank={'predicted':torch.tensor([7,8]),'rrq':torch.tensor([[7,8],[9,10],[11,12]])}
    exposed,ey=mod.arm_sequences(seq,3,2,bank,'exposure',torch.tensor([True,False]),128)
    rrq,ry=mod.arm_sequences(seq,3,2,bank,'rrq',torch.tensor([True,False]),128)
    assert torch.equal(ey,seq)
    assert exposed[0,3].item()==135 and rrq[0,3].item()==135
    assert ry[0,3].item()==128 and ry[0,4].item()==1161 and ry[0,5].item()==2187
    assert torch.equal(ry[:,:3],seq[:,:3]) and torch.equal(ry[:,6:],seq[:,6:])
    assert rrq[0,4].item()==1161


def test_epoch_batches_never_drop_or_repeat_real_examples():
    assert importlib.util.find_spec('rrq.uniaudio_train'), 'SVS trainer missing'
    mod=importlib.import_module('rrq.uniaudio_train')
    batches=mod.epoch_batches(list(range(5)),2,17,0)
    flat=[x for b in batches for x in b if x is not None]
    assert sorted(flat)==list(range(5)) and len(batches)==3 and len(batches[-1])==2
    assert batches==mod.epoch_batches(list(range(5)),2,17,0)
