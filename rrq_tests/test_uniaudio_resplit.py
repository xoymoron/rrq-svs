import json
from types import SimpleNamespace

import pytest


def corpus():
    return [dict(id=f'{s}-{p}-{g}-{i}', singer=s, language=lang, split=p,
                 song_group=f'{lang}-{p}-{g}', frames=200, samples=64000,
                 source_sha256=f'{s}-{p}-{g}-{i}', pcm_sha256=f'{s}-{p}-{g}-{i}',
                 audio_seconds=4.)
            for lang, singers in [('English', ['e1', 'e2', 'e3']), ('Russian', ['r1'])]
            for s in singers for p in ('train', 'valid', 'test')
            for g in range(3) for i in range(2)]


def test_holdout_keeps_languages_and_song_split_excludes_identity_and_reference(tmp_path):
    from rrq.uniaudio_resplit import singer_holdout
    from rrq.uniaudio_data import assign_prompts, read_rows, prompt_allowed
    original = corpus()
    rows = singer_holdout(original, ['e3'], 17, 150)
    assert rows == singer_holdout(original[::-1], ['e3'], 17, 150)
    assert {r['language'] for r in rows if r['split'] == 'train'} == {'English', 'Russian'}
    assert {r['singer'] for r in rows if r['split'] in ('train', 'valid')} == {'e1', 'e2', 'r1'}
    assert all(r['split'] == old['split'] for r in rows for old in original
               if r['id'] == old['id'] and r['singer'] != 'e3')
    assert {r['singer_cohort'] for r in rows if r['split'] == 'test'} == {'seen', 'unseen'}
    refs = [r for r in rows if r['split'] == 'reference']
    assert refs and {r['singer'] for r in refs} == {'e3'}
    assert not {r['song_group'] for r in refs} & {r['song_group'] for r in rows if r['split'] in ('valid', 'test')}
    prompts = assign_prompts(rows, 17)
    by_id = {r['id']: r for r in rows}
    assert set(prompts) == {r['id'] for r in rows if r['split'] in ('train', 'valid', 'test')}
    for uid, refid in prompts.items():
        row, ref = by_id[uid], by_id[refid]
        assert prompt_allowed(row, ref)
        assert ref['split'] == ('reference' if row['singer_cohort'] == 'unseen' else 'train')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert read_rows(manifest) == rows
    refs[0]['song_group'] = next(r['song_group'] for r in rows if r['split'] == 'test')
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='reference'):
        read_rows(manifest)


def test_holdout_rejects_missing_language_and_unknown_singer():
    from rrq.uniaudio_resplit import singer_holdout
    for singers in (['r1'], ['missing']):
        with pytest.raises(ValueError):
            singer_holdout(corpus(), singers, 17, 150)


def test_legacy_prepare_cannot_silently_ignore_holdout_config():
    from rrq import uniaudio_data as ud
    with pytest.raises(ValueError, match='resplit'):
        ud.prepare({'heldout_singers':['e3']})


def test_runtime_rejects_holdout_singer_in_optimizer_or_reference_rows():
    from rrq.uniaudio_resplit import singer_holdout, validate_holdout
    rows=singer_holdout(corpus(), ['e3'],17,150)
    cfg={'heldout_singers':['e3']}
    data=SimpleNamespace(rows={r['id']:r for r in rows},segments=[dict(row_id=r['id'],split=r['split']) for r in rows if r['split'] in ('train','valid','test')])
    validate_holdout(cfg,data)
    with pytest.raises(ValueError, match='holdout'):
        validate_holdout({'heldout_singers':['e2']},data)
    data.segments.append(dict(row_id=next(r['id'] for r in rows if r['split']=='reference'),split='train'))
    with pytest.raises(ValueError, match='segment'):
        validate_holdout(cfg,data)


def test_unseen_never_accepts_seen_train_reference_or_same_song():
    from rrq.uniaudio_data import prompt_allowed
    target = dict(id='t', singer='u', singer_cohort='unseen', split='test', song_group='target')
    ref = dict(id='p', singer='u', singer_cohort='unseen', split='reference', song_group='ref')
    assert prompt_allowed(target, ref)
    assert not prompt_allowed(target, dict(ref, split='train'))
    assert not prompt_allowed(target, dict(ref, song_group='target'))
    assert not prompt_allowed(dict(target, split='train'), ref)


def test_evaluation_selects_all_test_by_default_and_separates_cohorts():
    from rrq.uniaudio_evaluate import evaluation_ids, cohort_summary
    from rrq.uniaudio_resplit import singer_holdout
    data = SimpleNamespace(rows={r['id']: r for r in singer_holdout(corpus(), ['e3'], 17, 150)})
    test = evaluation_ids(data, {}, {'validation_per_language': 2}, ['English', 'Russian'], 'test')
    assert set(test) == {r['id'] for r in data.rows.values() if r['split'] == 'test'}
    valid = evaluation_ids(data, {}, {'validation_per_language': 2}, ['English', 'Russian'], 'valid')
    assert all(data.rows[i]['singer_cohort'] == 'seen' and data.rows[i]['split'] == 'valid' for i in valid)
    examples = [dict(data.rows[i], f0_rmse_cents=10. if data.rows[i]['singer_cohort']=='seen' else 100.) for i in test]
    summary = cohort_summary(examples)
    assert summary['seen']['macro']['f0_rmse_cents'] == 10.
    assert summary['unseen']['macro']['f0_rmse_cents'] == 100.
    assert set(summary['unseen']['per_language']) == {'English'}


def test_resplit_publishes_new_manifest_without_loading_codec_or_mutating_cache(tmp_path, monkeypatch):
    import torch
    import yaml
    from rrq import uniaudio_data as ud, uniaudio_resplit as mod
    from test_uniaudio_workflow import vocabulary
    original = tmp_path/'original'; original.mkdir()
    cfg = dict(prepared_path=str(tmp_path/'new'), cache_path=str(tmp_path/'cache'),
               model_config_path=str(tmp_path/'model.yaml'), upstream_path=str(tmp_path/'upstream'),
               heldout_singers=['e3'], seed=17, prompt_frames=150, max_frames=400, workers=2)
    phone = tmp_path/'upstream/UniAudio/tools/tokenizer/Sing/dict_phone.txt'
    phone.parent.mkdir(parents=True);phone.write_text('a 0\n')
    (tmp_path/'model.yaml').write_text(yaml.safe_dump(dict(token_list=vocabulary(),block_size=2048)))
    identity = {k:k for k in ('codec_sha256','codec_config_sha256','source_manifest_sha256','aliases_sha256',
                              'model_config_sha256','checkpoint_sha256','native_phone_dict_sha256')}
    identity.update(config=cfg, encoding_contract={'test':1})
    recipe = dict(identity)
    rows = [dict(r, phones=['a'], phone_durations=[4.], score={'ep_pitches':[60]}) for r in corpus()]
    recipe['manifest_sha256']=ud._freeze(original/'manifest.jsonl',''.join(ud._json(r)+'\n' for r in rows).encode())
    for name,value in dict(segments=[],prompts={},vocabulary={},audit={'original':True}).items():
        recipe[name+'_sha256']=ud._freeze(original/f'{name}.json',ud._json(value).encode())
    (original/'recipe.json').write_text(ud._json(recipe))
    cache=ud.NativeCache(cfg['cache_path'],identity['codec_sha256'],identity['encoding_contract'])
    for r in rows:cache.put(r['source_sha256'],torch.zeros(1,2,200),torch.zeros(3,1,200,dtype=torch.long),64000)
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in (tmp_path/'cache').rglob('*.pt')}
    monkeypatch.setattr(ud,'_identity',lambda cfg:identity)
    def no_encoder(*args,**kwargs):raise AssertionError('resplit must not load codec')
    monkeypatch.setattr(ud.ua,'load_codec',no_encoder)
    report=mod.resplit(cfg,original)
    data=ud.PreparedData(cfg['prepared_path'],cfg['cache_path'])
    assert report['cache_files_written']==0 and report['cache_records_verified']==len(rows)
    assert all(data.rows[s['row_id']]['split'] in ud.SPLITS for s in data.segments)
    assert all(data.rows[s['row_id']]['singer']!='e3' for s in data.segments if s['split']=='train')
    assert before=={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in (tmp_path/'cache').rglob('*.pt')}
    assert json.loads((original/'recipe.json').read_text())==recipe
