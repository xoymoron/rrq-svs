"""Explicit generator initialization and optimizer-step LR; codec is unchanged."""
import math

import torch

from . import uniaudio as ua


def random_generator(root, config, tokens, seed, device='cpu'):
    # Instantiate the actual upstream class with its own initialization. New
    # language rows are independently random, not copies of a pretrained mean.
    config = dict(config, token_list=list(tokens))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = ua.model_from_config(root, config)
    return generator.to(device=device, dtype=torch.float32)


def validate_initialization(cfg, saved=None):
    mode = cfg.get('initialization', 'release')
    if mode not in ('release', 'scratch'):
        raise ValueError('unknown initialization mode')
    if saved is not None and saved.get('run_contract', {}).get('initialization', 'release') != mode:
        raise ValueError('checkpoint initialization differs from requested run')
    return mode


def learning_rate_at(cfg, update):
    peak = cfg.get('learning_rate', 1e-5)
    schedule = cfg.get('lr_schedule', 'constant')
    if update < 1 or not math.isfinite(peak) or peak <= 0:
        raise ValueError('positive learning rate and optimizer update required')
    if schedule == 'constant':
        return peak
    warmup = cfg.get('warmup_steps', 0)
    if schedule != 'warmup_inverse_sqrt' or warmup < 1:
        raise ValueError('valid schedule and warmup_steps required')
    # Same formula as official UniAudio WarmupLR, indexed by optimizer update.
    return peak * min(update / warmup, (warmup / update) ** .5)
