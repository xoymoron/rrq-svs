"""Small, explicit DDP helpers for unpadded, equally weighted singing phrases."""
from collections import defaultdict
from datetime import timedelta
import hashlib
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def global_batches(rows, world):
    if world < 1 or len(rows) % world:
        raise ValueError('training sample count must be divisible by world size; never pad or drop')
    return [rows[i:i + world] for i in range(0, len(rows), world)]


def validation_result(values, expected_ids):
    ids = [v['id'] for v in values]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError('validation sample coverage is not exact')
    grouped = defaultdict(list)
    for value in values:
        grouped[value['language']].append(value['ce'])
    means = {key: float(np.mean(items)) for key, items in grouped.items()}
    return dict(language_macro_ce=float(np.mean(list(means.values()))), per_language=means,
                samples=sorted(values, key=lambda row: row['id']))


class Distributed:
    def __init__(self):
        self.rank = int(os.environ.get('RANK', 0))
        self.world = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    @property
    def primary(self):
        return self.rank == 0

    def start(self):
        if self.world > 1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group('nccl', timeout=timedelta(minutes=15))

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def gather(self, values):
        if self.world == 1:
            return values
        groups = [None] * self.world
        dist.all_gather_object(groups, values)
        return [item for group in groups for item in group]

    def wrap(self, model):
        if self.world == 1:
            return model
        device_ids=[self.local_rank] if next(model.parameters()).is_cuda else None
        return DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=False,
                                       gradient_as_bucket_view=True)

    def assert_synced(self, model):
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            if value.requires_grad:
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        hashes = self.gather([digest.hexdigest()])
        if len(set(hashes)) != 1:
            raise RuntimeError('DDP trainable parameters diverged between ranks')
        return hashes[0]

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
