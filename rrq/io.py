"""Strict data boundaries: explicit provenance, no inferred alignment or token cropping."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def validate_codes(codes):
    if codes.ndim != 2 or codes.shape[0] != 3 or codes.shape[1] == 0:
        raise ValueError('expected nonempty codes [3,T]')
    if not np.issubdtype(codes.dtype, np.integer) or np.any((codes < 0) | (codes >= 1024)):
        raise ValueError('codes must be integers in [0,1023]')
    return codes.astype(np.int64, copy=False)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('sample id must use ASCII letters, digits, dot, hyphen or underscore')
    # Windows reserved filenames must not become paths.
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}:
        raise ValueError('reserved sample id')
    return value


def parse_generation(text):
    """Read upstream D-ID<TAB>sequence; reject partial triplets and wrong offsets."""
    result = {}
    for line in text.splitlines():
        if not line.startswith('D-'):
            continue
        fields = line.split('\t')
        if len(fields) != 2:
            raise ValueError('expected upstream two-column D- generation format')
        item = safe_id(fields[0][2:])
        if item in result:
            raise ValueError('duplicate generated sample id: ' + item)
        tokens = fields[1].split()
        starts = [i for i, t in enumerate(tokens) if t == '<acoustic_start>']
        if not starts:
            raise ValueError('missing acoustic start boundary')
        acoustic = tokens[starts[-1]+1:]
        codes = []
        ended = False
        for token in acoustic:
            if token in {'<acoustic_end>', '</s>', '<pad>'}:
                ended = True
                continue
            if ended or not re.fullmatch(r'aco_\d+', token):
                raise ValueError('unexpected token in acoustic sequence: ' + token)
            k = len(codes) % 3
            index = int(token[4:]) - 1024 * k
            if not 0 <= index < 1024:
                raise ValueError('wrong codebook offset/order')
            codes.append(index)
        if not codes or len(codes) % 3:
            raise ValueError('empty or incomplete acoustic frame; refusing to truncate')
        result[item] = validate_codes(np.array(codes).reshape(-1, 3).T)
    if not result:
        raise ValueError('no D- records found')
    return result


def load_manifest(path):
    path = Path(path).resolve()
    rows, ids = [], set()
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required = {'id', 'singer', 'audio', 'prediction', 'origin', 'alignment', 'gain', 'conditioning'}
        if not required <= row.keys():
            raise ValueError('manifest requires ' + ', '.join(sorted(required)))
        item = safe_id(row['id'])
        if item in ids:
            raise ValueError('duplicate manifest id: ' + item)
        ids.add(item)
        if not isinstance(row['singer'], str) or not row['singer'].strip():
            raise ValueError('singer id must be nonempty')
        if row['origin'] not in {'model_rollout', 'synthetic_smoke'}:
            raise ValueError('origin must be model_rollout or synthetic_smoke')
        if row['alignment'] != 'same_50hz_grid':
            raise ValueError('explicit same_50hz_grid attestation is required')
        if row['conditioning'] not in {'synthetic', 'annotation_conditioned', 'score_lyrics'}:
            raise ValueError('conditioning must disclose annotation assistance')
        if (row['conditioning'] == 'synthetic') != (row['origin'] == 'synthetic_smoke'):
            raise ValueError('conditioning and origin are inconsistent')
        if not isinstance(row['gain'], (float, int)) or not np.isfinite(row['gain']) or row['gain'] <= 0:
            raise ValueError('gain must be finite and positive; no implicit normalization')
        for key in ('audio', 'prediction'):
            row[key] = (path.parent / row[key]).resolve()
            if not row[key].is_file():
                raise ValueError('missing ' + key + ': ' + str(row[key]))
        rows.append(row)
    if not rows:
        raise ValueError('empty manifest')
    if len({r['origin'] for r in rows}) != 1:
        raise ValueError('never combine synthetic smoke and model rollouts in one report')
    return rows


def save_prediction(path, codes, metadata):
    np.savez_compressed(path, codes=validate_codes(codes), metadata=np.array(json.dumps(metadata, allow_nan=False)))


def load_prediction(path, row):
    with np.load(path, allow_pickle=False) as archive:
        codes = validate_codes(archive['codes'])
        meta = json.loads(str(archive['metadata'].item()))
    if meta.get('id') != row['id'] or meta.get('origin') != row['origin']:
        raise ValueError('prediction provenance does not match manifest')
    if meta.get('frame_rate') != 50 or not meta.get('generator'):
        raise ValueError('missing generator provenance or wrong frame rate')
    if meta.get('target_audio_used_for_prediction') is not (row['origin'] == 'synthetic_smoke'):
        raise ValueError('prediction target leakage flag is inconsistent')
    return codes, meta
