"""Derive singer holdouts from verified song splits; never load a codec encoder."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

from . import uniaudio_data as ud
from .gtsinger_prepare import select_panel
from .io import sha256


def validate_holdout(cfg, data):
    held = set(cfg.get('heldout_singers', []))
    if not held:
        if any(r.get('singer_cohort') == 'unseen' for r in data.rows.values()):
            raise ValueError('holdout manifest requires matching holdout configuration')
        return
    rows = list(data.rows.values())
    actual = {r['singer'] for r in rows if r.get('singer_cohort') == 'unseen'}
    if held != actual or any(r['singer'] in held for r in rows if r['split'] in ('train','valid')):
        raise ValueError('configured holdout does not match manifest identities')
    if {r['language'] for r in rows if r['split']=='train'} != {r['language'] for r in rows}:
        raise ValueError('holdout lost training language coverage')
    for singer in held:
        if not all(any(r['singer']==singer and r['split']==split for r in rows) for split in ('reference','test')):
            raise ValueError('holdout lacks reference or test data')
    for segment in data.segments:
        row = data.rows[segment['row_id']]
        if segment['split'] != row['split'] or segment['split'] not in ud.SPLITS:
            raise ValueError('inadmissible segment split')


def singer_holdout(source, singers, seed, prompt_frames):
    held = set(singers)
    known = {r['singer'] for r in source}
    if not held or not held < known:
        raise ValueError('holdout singers must be a nonempty proper subset')
    languages = {r['language'] for r in source}
    if {r['language'] for r in source if r['split'] == 'train' and r['singer'] not in held} != languages:
        raise ValueError('holdout would remove a training language')
    references = {}
    for singer in sorted(held):
        candidates = [r for r in source if r['singer'] == singer and r['split'] == 'train'
                      and r['frames'] >= prompt_frames]
        if not candidates or not any(r['singer'] == singer and r['split'] == 'test' for r in source):
            raise ValueError('held-out singer needs separate reference and test songs')
        references[singer] = min(candidates, key=lambda r: ud._digest([seed, 'enrollment', r['song_group'], r['id']]))['id']
    rows = []
    for source_row in sorted(source, key=lambda r: r['id']):
        row = dict(source_row, source_split=source_row['split'], split_policy='song-family-singer-holdout-v1',
                   singer_cohort='unseen' if source_row['singer'] in held else 'seen')
        if row['singer'] in held and row['split'] != 'test':
            row['split'] = 'reference' if row['id'] == references[row['singer']] else 'excluded'
        row['conditioning'] = 'phones, performed durations, note MIDI, same-singer other-song audio prompt'
        rows.append(row)
    return rows


def split_summary(rows):
    summary = {}
    for split in (*ud.SPLITS, 'reference', 'excluded'):
        selected = [r for r in rows if r['split'] == split]
        per_language = {}
        for language in sorted({r['language'] for r in selected}):
            items = [r for r in selected if r['language'] == language]
            per_language[language] = dict(rows=len(items), hours=sum(r['audio_seconds'] for r in items)/3600,
                                          singers=sorted({r['singer'] for r in items}))
        summary[split] = dict(rows=len(selected), hours=sum(r['audio_seconds'] for r in selected)/3600,
                              per_language=per_language)
    summary['test_cohorts'] = {cohort: dict(rows=len(rs), hours=sum(r['audio_seconds'] for r in rs)/3600,
                                         singers=sorted({r['singer'] for r in rs}),
                                         languages=sorted({r['language'] for r in rs}))
                              for cohort in ('seen', 'unseen')
                              for rs in [[r for r in rows if r['split']=='test' and r['singer_cohort']==cohort]]}
    return summary


def resplit(cfg, source_path):
    source_path, out = Path(source_path), Path(cfg['prepared_path'])
    if source_path.resolve() == out.resolve():
        raise ValueError('resplit must use a new prepared directory')
    original = ud.PreparedData(source_path, cfg['cache_path'])
    identity = ud._identity(cfg)
    for key in ('codec_sha256', 'codec_config_sha256', 'encoding_contract', 'source_manifest_sha256',
                'aliases_sha256', 'model_config_sha256', 'checkpoint_sha256', 'native_phone_dict_sha256'):
        if identity[key] != original.recipe[key]:
            raise ValueError(f'cannot reuse audited source: {key} changed')
    if any(r['split'] not in ud.SPLITS for r in original.rows.values()):
        raise ValueError('resplit source must be the original seen-singer song partition')
    rows = singer_holdout(list(original.rows.values()), cfg['heldout_singers'], cfg['seed'], cfg['prompt_frames'])
    # All source caches, including excluded rows, stay in place and are verified read-only.
    def verify(row):
        path = original.cache.path(row['source_sha256'])
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        original._cached(row)
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError('acoustic cache changed during read-only resplit')
        return str(path), before
    with ThreadPoolExecutor(max_workers=cfg.get('workers', 4)) as pool:
        caches = list(tqdm(pool.map(verify, rows), total=len(rows), desc='Verify existing codec cache (no encoding)'))
    model = yaml.safe_load(Path(cfg['model_config_path']).read_text(encoding='utf8'))
    if 3*cfg['max_frames'] + cfg['prompt_frames'] + 11 >= model['block_size']:
        raise ValueError('configured context overflow')
    phones = dict.fromkeys(line.split()[0] for line in
        (Path(cfg['upstream_path'])/'UniAudio/tools/tokenizer/Sing/dict_phone.txt').read_text(encoding='utf8').splitlines() if line.strip())
    vocab = ud.build_vocabulary(rows, model['token_list'], {p:i for i,p in enumerate(phones)})
    segments = [dict(id=f"{r['id']}-{a}", row_id=r['id'], start=a, stop=b, split=r['split'], language=r['language'])
                for r in rows if r['split'] in ud.SPLITS for a,b in ud.chunk_ranges(r['frames'], cfg['max_frames'])]
    prompts = ud.assign_prompts(rows, cfg['seed'])
    panels = {s:select_panel(rows,s,n) for s,n in [('valid',2),('train',1)]}
    panels.update({f'test_{c}': sorted(r['id'] for r in rows if r['split']=='test' and r['singer_cohort']==c)
                   for c in ('seen','unseen')})
    audit = dict(policy='song-family-singer-holdout-v1', heldout_singers=sorted(cfg['heldout_singers']),
                 selection='English/Italian/Korean each retain two training singers; frozen before training',
                 source_recipe_sha256=sha256(source_path/'recipe.json'), source_manifest_sha256=original.recipe['manifest_sha256'],
                 source_audit_sha256=original.recipe['audit_sha256'], summary=split_summary(rows),
                 cache_records_verified=len(caches), cache_files_written=0,
                 cache_stat_signature=ud._digest(sorted(caches)),
                 source_caches_preserved=True, training_started=False, optimizer_steps=0,
                 reference_policy='One fixed >=3s phrase per unseen singer from an original train song; excluded from loss and evaluation. Seen prompts from other training songs.',
                 unseen_scope='Generator-training identities only; original codec pretraining identity coverage is unknown.',
                 selection_policy='Validation is seen-singer only. Test is never used for model selection.',
                 limitation='Three held-out singers across three languages; no all-language zero-shot claim.')
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(identity, block_size=model['block_size'], resplit_implementation_sha256=sha256(__file__))
    fingerprint['manifest_sha256'] = ud._freeze(out/'manifest.jsonl', ''.join(ud._json(r)+'\n' for r in rows).encode('utf8'))
    ud.read_rows(out/'manifest.jsonl')
    for name, value in dict(vocabulary=vocab, segments=segments, prompts=prompts, panels=panels, audit=audit).items():
        fingerprint[f'{name}_sha256'] = ud._freeze(out/f'{name}.json', ud._json(value).encode('utf8'))
    ud._freeze(out/'recipe.json', ud._json(fingerprint).encode('utf8'))
    validate_holdout(cfg, ud.PreparedData(out, cfg['cache_path']))
    print(ud._json(audit), flush=True)
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--workspace',required=True)
    p.add_argument('--source-prepared',required=True)
    args=p.parse_args()
    resplit(ud.load_recipe(args.config,args.workspace),args.source_prepared)


if __name__=='__main__':main()
