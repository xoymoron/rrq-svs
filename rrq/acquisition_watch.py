"""Observe official AIHub transfers; publish tqdm/W&B and verify all four archives."""
import argparse
import json
from pathlib import Path
import time
import zipfile

from tqdm import tqdm
from .io import write_json

NAMES = ('TL1.zip', 'TS1.zip', 'VL1.zip', 'VS1.zip')


def live_size(path):
    # The official merger deletes part files while this observer scans them.
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def archive_progress(roots):
    archives = {}
    received = 0
    for root in roots:
        root = Path(root)
        extracted_bytes = 0
        for name in NAMES:
            candidates = list(root.rglob(name))
            parts = list(root.rglob(name + '.part*'))
            sizes = [sum(live_size(p) for p in parts)]
            for path in candidates:
                if path.is_file():
                    sizes.append(live_size(path))
                    if name in archives and path.resolve() != archives[name].resolve():
                        raise ValueError(f'duplicate final archive: {name}')
                    archives[name] = path
            extracted_bytes += max(sizes)
        # Official shell first writes a TAR, then extracts parts and merges ZIPs.
        # During extraction all three representations can coexist.
        received += max(extracted_bytes, live_size(root / 'download.tar'))
    return dict(received_bytes=received, archives_present=len(archives), archives_expected=4), archives


def verify(archives):
    labels, waves, counts = set(), set(), {}
    for name in NAMES:
        path = archives[name]
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            counts[name] = len(infos)
            with tqdm(total=sum(i.file_size for i in infos), unit='B', unit_scale=True,
                      desc='CRC ' + name, mininterval=2) as bar:
                for info in infos:
                    if info.is_dir():
                        continue
                    with z.open(info) as source:
                        if info.filename.endswith('.json'):
                            raw = source.read()
                            item = json.loads(raw)
                            filename = item['data_info']['wav_filename']
                            if filename in labels:
                                raise ValueError('duplicate label waveform name')
                            labels.add(filename)
                            bar.update(len(raw))
                        else:
                            while block := source.read(1024**2):
                                bar.update(len(block))
                    if info.filename.lower().endswith('.wav'):
                        filename = Path(info.filename).name
                        if filename in waves:
                            raise ValueError('duplicate waveform name')
                        waves.add(filename)
    report = dict(crc_passed=True, member_counts=counts, labels=len(labels), waveforms=len(waves),
                  labels_without_waveform=sorted(labels-waves), waveforms_without_label=sorted(waves-labels))
    if labels != waves:
        raise ValueError('archive audio/label coverage mismatch: ' + json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pid', type=int, required=True, help='PID of already running official downloader')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    roots = [Path(p).resolve() for p in args.roots]
    if len(set(roots)) != len(roots):
        raise ValueError('duplicate roots')
    output = Path(args.output)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project='rrq-svs', entity='oxymoron_music', name='aihub473-full',
                         job_type='data-acquisition', config=dict(dataset=473, file_keys=[34217,34218,34219,34220]))
        write_json(output.with_suffix('.wandb.json'), dict(url=run.url, id=run.id))
    started = last_changed = time.monotonic()
    previous = -1
    # The portal lists rounded GB sizes. Never present this as an exact denominator.
    estimate = 31 * 1024**3 + 25 * 1024**2
    try:
        with tqdm(total=estimate, unit='B', unit_scale=True, desc='AIHub473 (size estimate)') as bar:
            while True:
                status, archives = archive_progress(roots)
                alive = Path(f'/proc/{args.pid}').exists()
                now = time.monotonic()
                if status['received_bytes'] != previous:
                    last_changed = now
                    previous = status['received_bytes']
                status.update(downloader_running=alive, elapsed_seconds=now-started,
                              seconds_since_growth=now-last_changed, total_bytes_estimate=estimate,
                              fraction_estimate=status['received_bytes']/estimate, verified=False)
                bar.n = status['received_bytes']
                bar.set_postfix(archives=f'{len(archives)}/4', stalled_seconds=int(now-last_changed))
                bar.refresh()
                write_json(output, status)
                if run:
                    run.log({'download/' + k: v for k, v in status.items()})
                if not alive:
                    if len(archives) != 4:
                        raise RuntimeError('Official downloader exited before all four archives were present')
                    report = verify(archives)
                    write_json(output.with_suffix('.verification.json'), report)
                    status.update(verified=True, complete=True)
                    write_json(output, status)
                    if run:
                        run.log({'download/verified': True, 'download/complete': True})
                        run.summary.update(report)
                    break
                time.sleep(15)
    except BaseException:
        if run:
            run.finish(exit_code=1)
        raise
    else:
        if run:
            run.finish()


if __name__ == '__main__':
    main()
