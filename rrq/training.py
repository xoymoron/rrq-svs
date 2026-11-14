"""Opt-in adapter for the original Prompt-Singer loader and CE criterion.

Shared generator parameters may change first-code predictions. This does NOT
implement an independently frozen G1; report it as shared-model adaptation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .io import safe_id, sha256, validate_codes
from .targets import training_map


class CachedTargets:
    def __init__(self, config):
        self.root = Path(config['cache_dir']).resolve()
        self.arm = config.get('arm','rrq')
        self.rho = float(config.get('rho',.5))
        self.seed = int(config.get('seed',0))
        self.expected_codec_sha256 = config.get('expected_codec_sha256')
        if self.arm not in {'clean','exposure','rrq'} or not 0 <= self.rho <= 1:
            raise ValueError('invalid RRQ arm or mixture rate')

    def get(self, item, audio, frames):
        path = self.root/safe_id(item)/'targets.npz'
        with np.load(path,allow_pickle=False) as cache:
            maps = [validate_codes(cache[k]).copy() for k in ('clean','predicted','rrq')]
            meta = json.loads(str(cache['metadata'].item()))
        if meta.get('origin') != 'model_rollout':
            raise ValueError('synthetic/origin-unknown cache is forbidden in SVS training')
        if meta.get('id') != item or meta.get('frame_rate') != 50 or meta.get('schema_version') != 1:
            raise ValueError('cache identity/schema mismatch')
        if meta.get('gain') != 1 or not meta.get('oracle_targets'):
            raise ValueError('cached training requires gain=1 and explicit oracle targets')
        if meta.get('target_sha256') != sha256(audio):
            raise ValueError('target audio hash mismatch: rebuild cache')
        if self.expected_codec_sha256 is not None and meta.get('provenance',{}).get('codec_sha256') != self.expected_codec_sha256:
            raise ValueError('codec checkpoint mismatch in RRQ cache')
        if any(m.shape != (3,frames) for m in maps):
            raise ValueError('cached training frame mismatch')
        # A stable sequence-level choice independent of arm, worker order and RNG.
        draw = int.from_bytes(hashlib.sha256(f'{self.seed}:{item}'.encode()).digest()[:8],'big')/2**64
        choose = torch.tensor([draw < self.rho])
        tensors = [torch.from_numpy(m).unsqueeze(1) for m in maps]
        return training_map(*tensors,self.arm,choose)[:,0].numpy()


def suffix_mask(target, acoustic_mask, dictionary):
    """Separate loss mask; do not change the transformer's attention/prefix mask."""
    if acoustic_mask.shape != target.shape or acoustic_mask.dtype != torch.bool:
        raise ValueError('loss/target shape mismatch')
    names = [f'aco_{i}' for i in range(1024,3072)]
    if any(name not in dictionary.indices for name in names):
        raise ValueError('dictionary must contain every suffix token')
    suffix_ids = torch.tensor([dictionary.index(name) for name in names],device=target.device)
    return acoustic_mask & torch.isin(target,suffix_ids)
