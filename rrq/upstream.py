"""Load upstream codec without importing the unrelated Fairseq training stack."""
import importlib
import json
from pathlib import Path
import sys
import types

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_state(path):
    """Allow only known OmegaConf metadata classes; never fall back to arbitrary pickle."""
    from collections import defaultdict
    from typing import Any
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    allowed = [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
               Any, defaultdict, dict, list, int]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def codec_module(suffix):
    name = '_rrq_soundstream'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / 'research/PromptSinger/dataset/tokenizer/soundstream')]
        sys.modules[name] = package
    return importlib.import_module(name + '.' + suffix)


def quantizer_class():
    # vq.py is the authority: upstream currently uses core_vq_lsx_version.
    return codec_module('quantization.vq').ResidualVectorQuantization


def load_codec(checkpoint: Path, device='cpu'):
    checkpoint = Path(checkpoint)
    config = yaml.safe_load(checkpoint.with_name('config.yaml').read_text())['generator']
    if config['name'] != 'SoundStream':
        raise ValueError('only the audited SoundStream architecture is supported')
    model = codec_module('models.soundstream').SoundStream(**config['config'])
    state = load_state(checkpoint)
    model.load_state_dict(state['codec_model'], strict=True)
    if model.sample_rate != 16000 or model.hop_length != 320:
        raise ValueError('expected Prompt-Singer 16kHz/50Hz codec')
    return model.eval().requires_grad_(False).to(device)


def load_vocoder(checkpoint: Path, device='cpu'):
    from wave_generation.models import CodeBigVGAN
    checkpoint = Path(checkpoint)
    config = json.loads(checkpoint.with_name('config.json').read_text())
    if (config['codebook_num'], config['code_num'], config['sampling_rate'],
            config['unit_hop_size']) != (3, 1024, 24000, 480):
        raise ValueError('expected Prompt-Singer 3x1024 24kHz/50Hz unit vocoder')
    cfg = types.SimpleNamespace(**config)
    model = CodeBigVGAN(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['generator'], strict=True)
    model.remove_weight_norm()
    # remove_weight_norm creates new Parameter objects; freeze AFTER conversion.
    return model.eval().requires_grad_(False).to(device)
