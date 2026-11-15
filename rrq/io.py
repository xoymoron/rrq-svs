"""Strict data boundaries: explicit provenance, no inferred alignment or token cropping."""
import hashlib
import json
from pathlib import Path
import re
import os

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('sample id must use ASCII letters, digits, dot, hyphen or underscore')
    if value.endswith('.'):
        raise ValueError('trailing dots are not portable Windows filenames')
    # Windows reserved filenames must not become paths.
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}:
        raise ValueError('reserved sample id')
    return value
