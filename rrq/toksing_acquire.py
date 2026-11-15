"""Acquire all original GTSinger singing assets at a pinned revision, no sampling."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

from tqdm import tqdm

from .corpora import gtsinger_row, validate_annotations
from .data_download import download, safe_relative, free_space
from .io import write_json, sha256

GT_REV = '4426c862beed558b7e1cb8a4dce7e8c0c83bb208'
BASE = f'https://huggingface.co/datasets/GTSinger/GTSinger/resolve/{GT_REV}/'
LANGUAGES = ['Chinese', 'English', 'French', 'German', 'Italian', 'Japanese',
             'Korean', 'Russian', 'Spanish']


def singing_files(entries):
    result = []
    for entry in entries:
        if entry['type'] != 'file':
            continue
        parts = safe_relative(entry['path']).parts
        if len(parts) >= 6 and parts[0] in LANGUAGES and 'speech' not in parts[4].lower():
            result.append(entry)
    return result


def source_inventory(root, run):
    path = root / 'source-inventory.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['revision'] != GT_REV:
            raise ValueError('inventory revision mismatch')
        return data['files']
    url = f'https://huggingface.co/api/datasets/GTSinger/GTSinger/tree/{GT_REV}?recursive=true&limit=1000'
    entries = []
    with tqdm(desc='Listing original source files', unit='entries') as bar:
        while url:
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response:
                        page = json.load(response)
                        link = response.headers.get('Link', '')
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            entries.extend(page)
            bar.update(len(page))
            if run:
                run.log({'inventory/entries_seen': len(entries)})
            match = re.search(r'<([^>]+)>; rel="next"', link)
            url = match.group(1) if match else None
    files = singing_files(entries)
    if not files or len({x['path'] for x in files}) != len(files):
        raise ValueError('empty or duplicate source inventory')
    write_json(path, dict(revision=GT_REV, files=files,
                         scope='All original singing assets; exclude Speech take directories only.'))
    return files


def annotation_manifest(root, files, output):
    """Record singing metadata without making acquisition depend on training eligibility."""
    paths = {x['path'] for x in files}
    rows, issues = [], []
    hours = defaultdict(float)
    for language in LANGUAGES:
        path = download(BASE + f'processed/{language}/metadata.json',
                        root / 'metadata' / language / 'metadata.json')
        for item in json.loads(path.read_text(encoding='utf-8')):
            parts = item['item_name'].split('#')
            if 'speech' in parts[4].lower():
                continue
            rel = '/'.join(parts[:-1] + [parts[-1] + '.wav'])
            try:
                row = gtsinger_row(item)
                validate_annotations(row)
                if rel not in paths:
                    raise ValueError('metadata waveform not in original source inventory')
            except (ValueError, KeyError, TypeError) as exc:
                issues.append(dict(source_id=item['item_name'], reason=str(exc)))
                continue
            row.update(source_audio=rel, audio=str((root / 'raw' / rel).resolve()),
                       source_revision=GT_REV)
            rows.append(row)
            hours[language] += sum(row['phone_durations']) / 3600
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    covered = {r['source_audio'] for r in rows}
    raw_wavs = {x['path'] for x in files if x['path'].lower().endswith('.wav')}
    report = dict(revision=GT_REV, source_files=len(files), source_wavs=len(raw_wavs),
                  source_bytes=sum(x['size'] for x in files), metadata_rows=len(rows),
                  metadata_hours_by_language=dict(hours), annotation_issues=issues,
                  raw_wavs_without_valid_metadata=sorted(raw_wavs - covered),
                  note='Acquisition inventory, not a finalized training split. All raw files retained even if annotation is unusable.')
    write_json(output.with_suffix('.report.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('workers must be 1..16')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project='rrq-svs', entity='oxymoron_music',
                         job_type='data-acquisition', name='gtsinger-full-singing',
                         config=dict(revision=GT_REV, scope='all singing, nine languages'))
        write_json(root / 'wandb-run.json', dict(url=run.url, id=run.id))
    try:
        files = source_inventory(root, run)
        download(BASE + 'dataset_license.md', root / 'dataset_license.md', 100000)
        report = annotation_manifest(root, files, Path(args.output))
        print(json.dumps({k: v for k, v in report.items() if isinstance(v, (int, float, dict))}, ensure_ascii=False), flush=True)
        remaining = sum(x['size'] for x in files if not (root / 'raw' / x['path']).exists())
        free_space(root, remaining)
        total = sum(x['size'] for x in files)
        failures = []
        complete = verified_bytes = 0
        started = last = time.monotonic()

        def get(entry):
            path = download(BASE + urllib.parse.quote(entry['path'], safe='/'),
                            root / 'raw' / entry['path'], max(1, entry['size']))
            if path.stat().st_size != entry['size']:
                raise ValueError('source file size mismatch')
            expected = entry.get('lfs', {}).get('oid')
            if expected and sha256(path) != expected:
                raise ValueError('official LFS SHA256 mismatch')
            return entry

        with tqdm(total=len(files), desc='GTSinger singing', unit='file', mininterval=2) as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(get, entry): entry for entry in files}
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(dict(path=entry['path'], error=str(exc)))
                        if len(failures) <= 5:
                            tqdm.write('Download failure: ' + json.dumps(failures[-1], ensure_ascii=False))
                    else:
                        complete += 1
                        verified_bytes += entry['size']
                    bar.update(1)
                    now = time.monotonic()
                    if now - last >= 10 or bar.n == len(files):
                        status = dict(files_verified=complete, files_total=len(files),
                                      bytes_verified=verified_bytes, bytes_total=total,
                                      fraction=verified_bytes / total, failures=len(failures),
                                      elapsed_seconds=now-started, complete=complete == len(files))
                        bar.set_postfix(verified_GiB=f'{verified_bytes / 1024**3:.2f}', errors=len(failures))
                        write_json(root / 'download-status.json', status)
                        write_json(root / 'download-failures.json', failures)
                        if run:
                            run.log({'download/' + k: v for k, v in status.items()})
                        last = now
        write_json(root / 'download-failures.json', failures)
        if failures:
            raise RuntimeError(f'{len(failures)} files failed; rerun safely verifies and reuses completed files')
        if run:
            run.summary['complete'] = True
    except BaseException:
        if run:
            run.finish(exit_code=1)
        raise
    else:
        if run:
            run.finish()


if __name__ == '__main__':
    main()
