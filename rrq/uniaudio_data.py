"""Audited GTSinger timing and frozen UniAudio SoundStream cache preparation."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from . import uniaudio as ua
from .core import requantize
from .gtsinger_prepare import assign_splits, build_families, select_panel, title_key
from .io import safe_id, sha256, write_json

SAMPLE_RATE = 16000
FRAME_RATE = 50
HOP = 320
BOOKS = 3
BOOK_SIZE = 1024
TARGET_BW = 1.5
ENDPOINT_TOLERANCE = 0.04
SPLITS = ('train', 'valid', 'test')


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf8')).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _publish(path, write):
    """Publish a complete file without replacing a concurrent writer's file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write(temporary)
        try:
            if os.name == 'nt':
                temporary.rename(path)  # Windows rename fails if the destination exists.
            else:
                os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _freeze(path, content):
    if not path.exists():
        _publish(path, lambda p: p.write_bytes(content))
    if path.read_bytes() != content:
        raise ValueError(f'immutable prepared artifact differs: {path}')
    return hashlib.sha256(content).hexdigest()


def load_recipe(config_path, workspace):
    cfg = dict(seed=17, max_frames=400, prompt_frames=150, workers=4)
    cfg.update(_load(config_path))
    return {key: str((Path(workspace) / value).resolve())
            if key.endswith('_path') and isinstance(value, str) else value
            for key, value in cfg.items()}


def _read_jsonl(path):
    with Path(path).open(encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({safe_id(row['id']) for row in rows}) != len(rows):
        raise ValueError('empty manifest or duplicate IDs')
    return rows


def read_rows(path):
    rows = _read_jsonl(path)
    groups = {}
    for row in rows:
        split, song = row['split'], row['song_group']
        if split not in SPLITS or not song:
            raise ValueError('invalid manifest split or song family')
        if groups.setdefault(song, split) != split:
            raise ValueError('song leakage across splits')
    return rows


def _phone_key(language, phone):
    if language == 'Chinese':
        phone = ua.native_phone(phone, language)
    return f'{language}::{phone}'


def build_vocabulary(rows, original_tokens, native_phone_dict):
    tokens = list(original_tokens)
    native_ids = {token: i for i, token in enumerate(tokens)}
    ids = {_phone_key('Chinese', phone): native_ids[f'<sing_phone_{index}>']
           for phone, index in native_phone_dict.items()}
    base_size = len(tokens)
    counts = Counter(_phone_key(row['language'], phone) for row in rows
                     if row['split'] == 'train' for phone in row['phones'])
    new_keys = sorted(set(counts) - ids.keys())
    for key in new_keys:
        ids[key] = len(tokens)
        tokens.append(f'<sing_phone_gtsinger_{key}>')
    unknown_id = len(tokens)
    tokens.append('<sing_phone_gtsinger_UNK>')
    unknowns = {split: Counter(_phone_key(row['language'], phone) for row in rows
                              if row['split'] == split for phone in row['phones']
                              if _phone_key(row['language'], phone) not in ids)
                for split in ('valid', 'test')}
    return dict(tokens=tokens, phone_ids=ids, unknown_id=unknown_id,
                original_size=base_size, train_added=new_keys, heldout_unknowns=unknowns)


def phone_id(row, phone, vocab):
    return vocab['phone_ids'].get(_phone_key(row['language'], phone), vocab['unknown_id'])


def assign_prompts(rows, seed):
    train = defaultdict(list)
    for row in sorted(rows, key=lambda row: row['id']):
        if row['split'] == 'train':
            train[row['singer']].append((row['id'], row['song_group']))
    prompts = {}
    for row in rows:
        candidates = train[row['singer']]
        if candidates:
            start = int(_digest([seed, row['id']])[:16], 16) % len(candidates)
            for offset in range(len(candidates)):
                prompt, song = candidates[(start + offset) % len(candidates)]
                if song != row['song_group']:
                    prompts[row['id']] = prompt
                    break
        if row['id'] not in prompts:
            raise ValueError(f"no other train song prompt for {row['id']} ({row['singer']})")
    return prompts


def chunk_ranges(nframes, limit):
    if type(nframes) is not int or type(limit) is not int or min(nframes, limit) <= 0:
        raise ValueError('positive frame count and chunk limit required')
    return [(start, min(start + limit, nframes)) for start in range(0, nframes, limit)]


def deduplicate_svs(rows):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: row['id']):
        groups[row['pcm_sha256']].append(row)
    kept, duplicates, rejected = [], [], []
    for group in groups.values():
        signatures = {_digest([r['singer'], r['language'],
                               [_phone_key(r['language'], p) for p in r['phones']],
                               list(map(float, r['phone_durations'])), list(map(int, r['score']['ep_pitches']))])
                      for r in group}
        if len(signatures) != 1:
            ids = [row['id'] for row in group]
            rejected.extend(dict(id=i, reason='identical PCM has conflicting SVS conditioning',
                                 duplicate_ids=ids) for i in ids)
            continue
        kept.append(group[0])
        duplicates.extend(dict(id=r['id'], kept=group[0]['id']) for r in group[1:])
    return kept, duplicates, rejected


def _annotations(row):
    phones, notes, durations = row['phones'], row['score']['ep_pitches'], row['phone_durations']
    if not isinstance(phones, list) or not isinstance(notes, list) or not isinstance(durations, list):
        raise ValueError('phones, MIDI and durations must be lists')
    if not phones or len(phones) != len(notes) or len(phones) != len(durations):
        raise ValueError('annotation lengths differ or are empty')
    if any(not isinstance(p, str) or not p.strip() for p in phones):
        raise ValueError('invalid phone label')
    if any(type(note) not in (int, float) or not math.isfinite(note) or note % 1
           or not 0 <= note <= 137 for note in notes):
        raise ValueError('MIDI outside released vocabulary or not integer')
    if any(type(d) not in (int, float) or not math.isfinite(d) or d < 0 for d in durations):
        raise ValueError('invalid phone duration')
    if not any(durations):
        raise ValueError('all phone durations are zero')
    return phones, notes, durations


def _frames(samples):
    return (samples + HOP - 1) // HOP


def _check_duration(durations, samples):
    if abs(math.fsum(durations) - samples / SAMPLE_RATE) > ENDPOINT_TOLERANCE + 1e-9:
        raise ValueError('phone duration and waveform endpoint differ by more than 40 ms')


def _conditions(row, frames, vocab, annotations):
    phones, notes, durations = annotations
    positive = np.flatnonzero(durations)
    ends = np.cumsum(np.asarray(durations, dtype=np.float64)[positive])
    centers = (np.arange(frames, dtype=np.float64) + 0.5) / FRAME_RATE
    indices = positive[np.searchsorted(ends, centers, side='right').clip(max=len(ends) - 1)]
    return (torch.tensor([phone_id(row, phone, vocab) for phone in phones], dtype=torch.long)[indices],
            torch.tensor(notes, dtype=torch.long)[indices])


def frame_conditions(row, frames, samples, vocab):
    """Select the phone active at each native acoustic frame center."""
    if type(samples) is not int or samples < 1 or frames != _frames(samples):
        raise ValueError('invalid acoustic frame clock')
    annotations = _annotations(row)
    _check_duration(annotations[2], samples)
    return _conditions(row, frames, vocab, annotations)


def score_conditions(row, vocab):
    """Inference clock uses only the supplied annotation durations."""
    annotations = _annotations(row)
    samples = round(math.fsum(annotations[2]) * SAMPLE_RATE)
    if samples < 1:
        raise ValueError('annotation duration rounds to zero samples')
    return _conditions(row, _frames(samples), vocab, annotations)


class NativeCache:
    """Immutable whole-waveform latents and native labels, shared across splits."""

    def __init__(self, root, codec_hash, contract):
        self.root = Path(root)
        self.codec_hash = codec_hash
        self.contract = dict(contract)
        for key, expected in dict(sample_rate=SAMPLE_RATE, frame_rate=FRAME_RATE, hop_length=HOP,
                                  books=BOOKS, book_size=BOOK_SIZE).items():
            if self.contract.get(key, expected) != expected:
                raise ValueError(f'invalid native codec contract: {key}')

    def _key(self, source_hash):
        return _digest([source_hash, self.codec_hash, self.contract])

    def path(self, source_hash):
        key = self._key(source_hash)
        return self.root / key[:2] / f'{key}.pt'

    def _validate(self, latent, codes, samples):
        if type(samples) is not int or samples <= 0:
            raise ValueError('invalid native cache sample count')
        frames = _frames(samples)
        if (latent.dtype != torch.float32 or latent.ndim != 3 or latent.shape[0] != 1
                or latent.shape[2] != frames or latent.shape[1] <= 0
                or latent.shape[1] != self.contract.get('latent_dim', latent.shape[1])
                or not np.isfinite(latent.numpy()).all()):
            raise ValueError('invalid native cache latent')
        if (codes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or codes.shape != (BOOKS, 1, frames) or codes.min().item() < 0
                or codes.max().item() >= BOOK_SIZE):
            raise ValueError('invalid native cache codes')

    @staticmethod
    def _checksums(latent, codes, samples):
        return dict(latent=hashlib.sha256(latent.contiguous().numpy()).hexdigest(),
                    codes=hashlib.sha256(codes.contiguous().numpy()).hexdigest(),
                    metadata=_digest([samples, list(latent.shape), list(codes.shape), str(codes.dtype)]))

    def read(self, source_hash):
        path = self.path(source_hash)
        raw = torch.load(path, map_location='cpu', weights_only=True)
        try:
            if raw['key'] != self._key(source_hash):
                raise ValueError('native cache key mismatch')
            latent, codes, samples = raw['latent'], raw['codes'], raw['samples']
            self._validate(latent, codes, samples)
            if raw['checksums'] != self._checksums(latent, codes, samples):
                raise ValueError('native cache checksum mismatch')
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f'invalid native cache {path}') from exc
        return raw

    def put(self, source_hash, z, codes, samples):
        z, codes = z.detach().cpu().contiguous(), codes.detach().cpu().contiguous()
        self._validate(z, codes, samples)
        codes = codes.to(torch.int16)
        record = dict(key=self._key(source_hash), latent=z, codes=codes, samples=samples,
                      checksums=self._checksums(z, codes, samples))
        path = self.path(source_hash)
        if path.exists() or not _publish(path, lambda p: torch.save(record, p)):
            existing = self.read(source_hash)
            if existing['checksums'] != record['checksums']:
                raise ValueError(f'immutable native cache differs: {path}')
            return existing
        return record


def _decode(blob):
    audio, rate = sf.read(BytesIO(blob), dtype='float32', always_2d=True)
    if not audio.size or rate <= 0 or not np.isfinite(audio).all():
        raise ValueError('empty, nonfinite, or invalid waveform')
    mono = audio.mean(axis=1)
    if not np.isfinite(mono).all() or not mono.any():
        raise ValueError('silent or nonfinite mono waveform')
    return audio, rate, mono


def _inspect(row, source_directory):
    try:
        for key in ('language', 'singer'):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f'invalid {key}')
        if not title_key(row['song_title']):
            raise ValueError('empty song title')
        path = (source_directory / row['audio']).resolve()
        blob = path.read_bytes()
        audio, rate, _ = _decode(blob)
        # Torchaudio's sinc resampler emits ceil(input_samples * output_sr / input_sr).
        samples = (audio.shape[0] * SAMPLE_RATE + rate - 1) // rate
        durations = _annotations(row)[2]
        _check_duration(durations, samples)
        pcm = hashlib.sha256(f'{rate}:{audio.shape[1]}:'.encode())
        pcm.update(audio.astype('<f4', copy=False))
        return (dict(row, audio=str(path), audio_seconds=samples / SAMPLE_RATE, samples=samples,
                     frames=_frames(samples), source_sample_rate=rate, source_samples=audio.shape[0],
                     pcm_sha256=pcm.hexdigest(), source_sha256=hashlib.sha256(blob).hexdigest()), None)
    except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError, OverflowError) as exc:
        return None, dict(id=row['id'], reason=str(exc))


def _identity(cfg):
    upstream = cfg['upstream_path']
    head = subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '--no-optional-locks', '-C', upstream, 'status',
                                     '--porcelain', '--untracked-files=no'],
                                    text=True).strip()
    if head != ua.UPSTREAM_COMMIT or dirty:
        raise ValueError('UniAudio source must match the clean pinned upstream revision')
    codec = Path(cfg['codec_path'])
    codec_config = yaml.safe_load((codec / 'config.yaml').read_text(encoding='utf8'))['generator']
    settings = codec_config['config']
    if (codec_config['name'] != 'SoundStream' or settings['sample_rate'] != SAMPLE_RATE
            or math.prod(settings['ratios']) != HOP or settings['bins'] != BOOK_SIZE):
        raise ValueError('unexpected native SoundStream configuration')
    codec_config_hash = sha256(codec / 'config.yaml')
    encoding = dict(sample_rate=SAMPLE_RATE, frame_rate=FRAME_RATE, hop_length=HOP,
                    books=BOOKS, book_size=BOOK_SIZE, target_bw=TARGET_BW, latent_dim=settings['D'],
                    graph=ua.UPSTREAM_COMMIT, codec_config_sha256=codec_config_hash,
                    waveform='libsndfile float32; mean channels; CPU torchaudio sinc defaults; full waveform',
                    latent_dtype='float32', version=1)
    phone_dict = Path(cfg['upstream_path']) / 'UniAudio/tools/tokenizer/Sing/dict_phone.txt'
    source_report = Path(cfg['source_path']).with_suffix('.report.json')
    return dict(config=cfg, codec_sha256=sha256(codec / 'model.pth'),
                codec_config_sha256=codec_config_hash, checkpoint_sha256=sha256(cfg['checkpoint_path']),
                model_config_sha256=sha256(cfg['model_config_path']), native_phone_dict_sha256=sha256(phone_dict),
                source_manifest_sha256=sha256(cfg['source_path']), aliases_sha256=sha256(cfg['aliases_path']),
                source_report_sha256=sha256(source_report) if source_report.is_file() else None,
                encoding_contract=encoding, upstream_commit=ua.UPSTREAM_COMMIT,
                implementation_sha256=sha256(__file__), audio_tolerance_seconds=ENDPOINT_TOLERANCE)


def prepare(cfg):
    out = Path(cfg['prepared_path'])
    identity = _identity(cfg)
    lock = out / 'recipe.json'
    if lock.exists():
        existing = _load(lock)
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError('immutable prepared recipe inputs changed')
    cfg = identity['config']
    model = yaml.safe_load(Path(cfg['model_config_path']).read_text(encoding='utf8'))
    if model['non_acoustic_repeat'] != BOOKS or model['prefix_lm']:
        raise ValueError('expected released causal three-book UniAudio model config')
    context = model['block_size']
    if (min(cfg['max_frames'], cfg['prompt_frames']) <= 0
            or 3 * cfg['max_frames'] + cfg['prompt_frames'] + 11 >= context):
        raise ValueError('configured chunk and prompt can overflow UniAudio context')
    source = Path(cfg['source_path'])
    raw = _read_jsonl(source)
    inspected, rejected = [], []
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        results = pool.map(_inspect, raw, [source.parent] * len(raw))
        for row, issue in tqdm(results, total=len(raw), desc='Audit singing PCM and timing'):
            (rejected if issue else inspected).append(issue or row)
    families = build_families(inspected, _load(cfg['aliases_path'])['groups'])
    rows, duplicates, conflicts = deduplicate_svs(inspected)
    rejected.extend(conflicts)
    source_report = source.with_suffix('.report.json')
    audit = dict(input_rows=len(raw), included=len(rows), rejected=rejected, exact_duplicate_pcm=duplicates,
                 source_acquisition=_load(source_report) if source_report.is_file() else None)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if not rows:
            raise ValueError('no usable singing records')
        splits = assign_splits(rows, families, cfg['seed'])
        for row in rows:
            row.update(song_group=families[row['id']], split=splits[row['id']],
                       conditioning='GTSinger phones, performed phone durations, note MIDI, same-singer train prompt')
        prompts = assign_prompts(rows, cfg['seed'])
    except ValueError as exc:
        write_json(out / 'prepare-failure.json', dict(audit, failure=str(exc)))
        raise
    phone_dict = Path(cfg['upstream_path']) / 'UniAudio/tools/tokenizer/Sing/dict_phone.txt'
    phones = dict.fromkeys(line.split()[0] for line in phone_dict.read_text(encoding='utf8').splitlines() if line.strip())
    vocabulary = build_vocabulary(rows, model['token_list'], {phone: i for i, phone in enumerate(phones)})
    segments = [dict(id=f"{row['id']}-{start}", row_id=row['id'], start=start, stop=stop,
                     split=row['split'], language=row['language']) for row in rows
                for start, stop in chunk_ranges(row['frames'], cfg['max_frames'])]
    panels = {split: select_panel(rows, split, per_language=count)
              for split, count in [('valid', 2), ('train', 1)]}
    for split in SPLITS:
        per_language = defaultdict(lambda: dict(rows=0, seconds=0.0))
        for row in rows:
            if row['split'] == split:
                item = per_language[row['language']]
                item['rows'] += 1
                item['seconds'] += row['audio_seconds']
        audit.setdefault('splits', {})[split] = dict(per_language)
    fingerprint = dict(identity, block_size=context)
    manifest = ''.join(_json(row) + '\n' for row in rows).encode('utf8')
    fingerprint['manifest_sha256'] = _freeze(out / 'manifest.jsonl', manifest)
    for name, data in dict(vocabulary=vocabulary, segments=segments, panels=panels, prompts=prompts, audit=audit).items():
        fingerprint[f'{name}_sha256'] = _freeze(out / f'{name}.json', _json(data).encode('utf8'))
    _freeze(lock, _json(fingerprint).encode('utf8'))
    print(_json(dict(included=len(rows), rejected=len(rejected), duplicates=len(duplicates), splits=audit['splits'])),
          flush=True)
    return audit


def _verify_artifact(root, recipe, file):
    path = root / file
    if sha256(path) != recipe[f'{path.stem}_sha256']:
        raise ValueError(f'prepared artifact checksum mismatch: {path}')
    return path


class PreparedData:
    def __init__(self, prepared_path, cache_path):
        root = Path(prepared_path)
        self.recipe = _load(root / 'recipe.json')
        self.rows = {row['id']: row for row in read_rows(_verify_artifact(root, self.recipe, 'manifest.jsonl'))}
        self.segments = _load(_verify_artifact(root, self.recipe, 'segments.json'))
        self.vocab = _load(_verify_artifact(root, self.recipe, 'vocabulary.json'))
        self.prompts = _load(_verify_artifact(root, self.recipe, 'prompts.json'))
        self.cache = NativeCache(cache_path, self.recipe['codec_sha256'], self.recipe['encoding_contract'])

    def _cached(self, row):
        record = self.cache.read(row['source_sha256'])
        if record['samples'] != row['samples']:
            raise ValueError(f"cache and manifest clock differ for {row['id']}")
        return record

    def _prefix(self, row, phones, midi, device, ending):
        ref = self.rows[self.prompts[row['id']]]
        if not (ref['split'] == 'train' and ref['singer'] == row['singer'] and ref['song_group'] != row['song_group']):
            raise ValueError(f"inadmissible training prompt for {row['id']}")
        limit = self.recipe['config']['prompt_frames']
        if 3 * len(midi) + min(limit, ref['frames']) + 9 + ending >= self.recipe['block_size']:
            raise ValueError(f"UniAudio context overflow for {row['id']}")
        prompt = self._cached(ref)['codes'][..., :limit].to(device=device, dtype=torch.long)
        return ua.singing_prefix(self.vocab['tokens'], phones, midi, prompt)

    def inference_prefix(self, row_id, device='cpu'):
        row = self.rows[row_id]
        phones, midi = score_conditions(row, self.vocab)
        return self._prefix(row, phones, midi, device, ending=0), len(midi)

    def example(self, segment, device='cpu'):
        row = self.rows[segment['row_id']]
        start, stop = segment['start'], segment['stop']
        if not 0 <= start < stop <= row['frames']:
            raise ValueError(f"invalid segment bounds: {segment['id']}")
        phones, midi = frame_conditions(row, row['frames'], row['samples'], self.vocab)
        prefix = self._prefix(row, phones[start:stop], midi[start:stop], device, ending=2)
        record = self._cached(row)
        latent = record['latent'][..., start:stop].to(device)
        codes = record['codes'][..., start:stop].to(device=device, dtype=torch.long)
        tokens = self.vocab['tokens']
        audio = ua.flatten_codes(codes, tokens.index('<audio_0>'))
        suffix = torch.tensor([[tokens.index('<audio_end>')] * BOOKS + [tokens.index('<end>')] * BOOKS],
                              dtype=torch.long, device=device)
        sequence = torch.cat([prefix, audio, suffix], dim=1)
        return dict(id=segment['id'], row_id=row['id'], language=row['language'], frames=stop-start,
                    sequence=sequence, prefix_length=prefix.shape[1], codes=codes, latent=latent)


@torch.inference_mode()
def cache_all(cfg, rank=0, world=1):
    if not 0 <= rank < world:
        raise ValueError('invalid cache rank/world size')
    data = PreparedData(cfg['prepared_path'], cfg['cache_path'])
    identity = _identity(cfg)
    if any(data.recipe.get(key) != value for key, value in identity.items()):
        raise ValueError('prepared recipe inputs changed before cache extraction')
    import torchaudio

    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.set_num_threads(2)
    codec = ua.load_codec(cfg['upstream_path'], cfg['codec_path'], device)
    layers = codec.quantizer.vq.layers[:BOOKS]
    if codec.hop_length != HOP or len(layers) != BOOKS or any(layer.codebook_size != BOOK_SIZE for layer in layers):
        raise ValueError('loaded codec does not conform to native encoding contract')
    rows = list(data.rows.values())[rank::world]
    progress_path = Path(cfg['prepared_path']) / f'cache-progress-{rank}.json'
    progress = dict(rank=rank, world_size=world, total=len(rows), completed=0, reused=0, written=0,
                    failures=[], complete=False, manifest_sha256=data.recipe['manifest_sha256'])
    write_json(progress_path, progress)
    last = time.monotonic()
    for row in tqdm(rows, desc=f'UniAudio native cache GPU {rank}', mininterval=5):
        try:
            blob = Path(row['audio']).read_bytes()
            source_hash = hashlib.sha256(blob).hexdigest()
            if source_hash != row['source_sha256']:
                raise ValueError('waveform source checksum changed')
            if data.cache.path(source_hash).exists():
                data._cached(row)
                progress['reused'] += 1
            else:
                _, rate, mono = _decode(blob)
                audio = torch.from_numpy(mono)[None, None]
                if rate != SAMPLE_RATE:
                    audio = torchaudio.functional.resample(audio, rate, SAMPLE_RATE)
                if audio.numel() != row['samples']:
                    raise ValueError('prepared and actual resampled sample counts differ')
                latent = codec.encoder(audio.to(device))
                clean = codec.quantizer.encode(latent, codec.frame_rate, TARGET_BW).long()
                if not torch.equal(requantize(latent, clean[:1], layers), clean):
                    raise ValueError('clean prefix does not reproduce native codec labels')
                data.cache.put(source_hash, latent, clean, audio.numel())
                progress['written'] += 1
            progress['completed'] += 1
        except (ValueError, OSError, RuntimeError) as exc:
            progress['failures'].append(dict(id=row['id'], reason=str(exc)))
            write_json(progress_path, progress)
        if time.monotonic() - last >= 10:
            write_json(progress_path, progress)
            last = time.monotonic()
    progress['complete'] = progress['completed'] == progress['total']
    write_json(progress_path, progress)
    if not progress['complete']:
        raise ValueError(f"{len(progress['failures'])} native cache failures. See {progress_path}")
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'cache'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()
    cfg = load_recipe(args.config, args.workspace)
    if args.stage == 'prepare':
        prepare(cfg)
    else:
        cache_all(cfg, args.rank, args.world_size)


if __name__ == '__main__':
    main()
