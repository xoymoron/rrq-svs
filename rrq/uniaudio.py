"""Thin adapters for pinned, unmodified UniAudio. No replacement generator.

The generation API receives condition tokens and score length only. Its cache
operations follow upstream infer.py; fixed-length termination is deliberately
score-derived, with EOS suppressed until all score frames have been produced.
"""
import importlib.util
from importlib import import_module
import math
from pathlib import Path
import sys
import datetime
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import yaml

UPSTREAM_COMMIT = 'c1146a8cecf84f1c2e877170d84998393b7e1e78'


def add_upstream_path(root):
    path = str(Path(root).resolve() / 'UniAudio')
    if path not in sys.path:
        sys.path.insert(0, path)


def upstream_model(root):
    path = Path(root).resolve() / 'UniAudio' / 'model.py'
    name = '_rrq_official_uniaudio_model'
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise RuntimeError('a different UniAudio source is already loaded')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_from_config(root, config):
    tokens = config['token_list']
    if config['non_acoustic_repeat'] != 3 or config['prefix_lm']:
        raise ValueError('this adapter requires the released causal three-book model')
    return upstream_model(root).MegaByteModel(
        n_vocab=len(tokens), n_ctx=config['block_size'], n_state=config['n_embd'],
        n_head=config['n_head'], n_layer=config['n_layer'], prefix_lm=False,
        na_repeat=3, local_sos=tokens.index('<mega_local>'),
        global_sos=tokens.index('<mega_global>'))


def inference_precision(model, device, dtype):
    model = model.to(device=device, dtype=dtype)
    # Upstream LayerNorm explicitly computes x.float(); its affine parameters
    # must remain float32, as in mixed-precision Whisper inference.
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()
    return model


def load_release(path, codec_metadata=False):
    """Allow only the scalar/config metadata found in the official archives.

    No unrestricted pickle fallback. A changed release with other globals fails.
    mmap avoids loading the saved optimizer tensors into RAM unnecessarily.
    """
    try:
        scalar = import_module('numpy._core.multiarray').scalar
    except ModuleNotFoundError:
        scalar = import_module('numpy.core.multiarray').scalar
    safe = [(scalar, 'numpy.core.multiarray.scalar'),
            (scalar, 'numpy._core.multiarray.scalar'), np.dtype, datetime.timedelta]
    safe += [type(np.dtype(t)) for t in (np.float32, np.float64, np.int32, np.int64, np.bool_)]
    if codec_metadata:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.nodes import AnyNode
        from omegaconf.base import Metadata, ContainerMetadata
        safe += [DictConfig, ListConfig, AnyNode, Metadata, ContainerMetadata,
                 Any, list, dict, int, defaultdict]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location='cpu', mmap=True, weights_only=True)


def load_generator(root, config_path, checkpoint, device='cuda:0', dtype=torch.float16):
    config = yaml.safe_load(Path(config_path).read_text())
    with torch.device('meta'):
        model = model_from_config(root, config)
    saved = load_release(checkpoint)
    state = {k.removeprefix('module.'): v for k, v in saved['model'].items()}
    model.load_state_dict(state, strict=True, assign=True)
    # This nonpersistent mask is not present in the checkpoint.
    model.causal_mask = torch.full((1, 1, model.n_ctx, model.n_ctx),
                                   -torch.inf).triu_(1)
    model = inference_precision(model, device, dtype).eval().requires_grad_(False)
    return model, config


def load_codec(root, directory, device='cuda:0'):
    add_upstream_path(root)
    from tools.tokenizer.soundstream.models.soundstream import SoundStream
    directory = Path(directory)
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    if config['generator']['name'] != 'SoundStream':
        raise ValueError('unexpected released codec architecture')
    model = SoundStream(**config['generator']['config'])
    saved = load_release(directory / 'model.pth', codec_metadata=True)
    model.load_state_dict(saved['codec_model'], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    if model.sample_rate != 16000 or model.frame_rate != 50:
        raise ValueError('unexpected codec clock')
    return model


def flatten_codes(codes, audio_offset, book_size=1024):
    if codes.ndim != 3 or codes.shape[0] != 3 or codes.dtype != torch.long:
        raise ValueError('expected three-book int64 [Q,B,T] codes')
    if torch.any((codes < 0) | (codes >= book_size)):
        raise ValueError('code out of range')
    offset = torch.arange(3, device=codes.device)[:, None, None] * book_size + audio_offset
    return (codes + offset).permute(1, 2, 0).reshape(codes.shape[1], -1)


def native_phone(phone, language):
    if language != 'Chinese':
        raise ValueError('native singing vocabulary is Mandarin; adaptation is required')
    return phone.removesuffix('_zh').replace('<AP>', 'AP').replace('<SP>', 'SP')


def singing_prefix(tokens, frame_phones, frame_midi, prompt_codes):
    """Exact upstream sing ordering, through the audio-start patch inclusive."""
    if frame_phones.shape != frame_midi.shape or frame_phones.ndim != 1:
        raise ValueError('phone and MIDI must share the score frame clock')
    device = prompt_codes.device
    def special(name):
        return torch.full((1, 3), tokens.index(name), dtype=torch.long, device=device)
    def typed(name, body):
        return [special(f'<{name}_start>'), body, special(f'<{name}_end>')]
    phones = frame_phones.to(device)[None].repeat_interleave(3, dim=1)
    midi = (frame_midi.to(device)[None] + tokens.index('<sing_midi_0>')).repeat_interleave(3, dim=1)
    parts = [special('<start>'), special('<sing_task>')]
    parts += typed('sing_phone', phones)
    parts += typed('sing_midi', midi)
    parts += typed('audio_prompt', flatten_codes(prompt_codes, tokens.index('<audio_0>')))
    parts += [special('<audio_start>')]
    return torch.cat(parts, dim=1)


def depth_forward(model, global_sequence, local_sequence, mask):
    """Use original modules with separately specified temporal/local histories.

    Clean=clean reproduces upstream forward exactly. For a depth-only RRQ
    experiment, keeping global_sequence clean isolates within-frame exposure;
    this is an explicit training protocol, not an inference architecture change.
    """
    if global_sequence.shape != local_sequence.shape or mask.shape != global_sequence.shape:
        raise ValueError('history shapes differ')
    b, length = global_sequence.shape
    q = model.na_repeat
    gx = torch.cat([torch.full_like(global_sequence[:, :q], model.global_sos),
                    global_sequence[:, :-q]], dim=1)
    gm = torch.cat([torch.zeros_like(mask[:, :q]), mask[:, :-q]], dim=1)
    context = model.global_forward(gx, gm)
    local = local_sequence.reshape(-1, q)
    lx = torch.cat([torch.full_like(local[:, :1], model.local_sos), local[:, :-1]], dim=1)
    return model.local_forward(lx, context).reshape(b, length, -1)


def _cache_hooks(layers):
    # Same key/value append semantics as upstream install_kv_cache_hook.
    cache, hooks = {}, []
    def save(module, inputs, output):
        cache[module] = output if module not in cache else torch.cat([cache[module], output], dim=1).detach()
        return cache[module]
    for layer in layers:
        hooks.extend([layer.attn.key.register_forward_hook(save),
                      layer.attn.value.register_forward_hook(save)])
    return cache, hooks


@torch.inference_mode()
def generate(model, prefix, frames, audio_offset, book_size=1024,
             mode='sampling', topk=30, temperature=.8, seed=17, progress=None):
    """Free-running temporal AND residual-depth generation; no target argument.

    Prefix ends with audio_start. A known score duration constrains length.
    It is not unconstrained EOS generation, and never uses target token length.
    """
    if not isinstance(frames, int) or frames < 1:
        raise ValueError('frame count must be a positive integer')
    if model.training or prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError('evaluation mode and one condition sequence required')
    q = model.na_repeat
    if prefix.shape[1] % q or prefix.shape[1] // q + frames >= model.n_ctx:
        raise ValueError('prefix/frame count exceeds context')
    if mode not in ('sampling', 'greedy') or temperature <= 0 or topk < 1:
        raise ValueError('invalid sampling settings')
    device = prefix.device
    rng = torch.Generator(device=device).manual_seed(seed)
    cache, hooks = _cache_hooks(model.g_layers)
    codes = torch.empty((q, 1, frames), dtype=torch.long, device=device)
    try:
        # Warm global cache only, exactly as full-forward's Q-token shift.
        warm = torch.cat([torch.full_like(prefix[:, :q], model.global_sos), prefix[:, :-q]], dim=1)
        model.global_forward(warm, torch.zeros_like(warm, dtype=torch.bool), cache)
        previous = prefix[:, -q:]
        iterator = range(frames) if progress is None else progress(range(frames))
        for t in iterator:
            context = model.global_forward(previous, None, cache)
            local_cache, local_hooks = _cache_hooks(model.l_layers)
            token = torch.full((1, 1), model.local_sos, dtype=torch.long, device=device)
            predicted = []
            try:
                for k in range(q):
                    logits = model.local_forward(token, context[:, k:k+1], local_cache)[:, 0]
                    start = audio_offset + k * book_size
                    scores = logits[:, start:start+book_size].float()
                    if mode == 'greedy':
                        code = scores.argmax(dim=-1, keepdim=True)
                    else:
                        values, indices = scores.topk(min(topk, book_size), dim=-1)
                        draw = torch.multinomial((values/temperature).softmax(-1), 1, generator=rng)
                        code = indices.gather(1, draw)
                    codes[k, 0, t] = code[0, 0]
                    token = code + start
                    predicted.append(token)
            finally:
                for hook in local_hooks:
                    hook.remove()
            previous = torch.cat(predicted, dim=1)
    finally:
        for hook in hooks:
            hook.remove()
    return codes


def score_frames(row, phone_dictionary, tokens):
    """50-Hz annotation-conditioned input, independent of waveform/token length.

    These are performed phoneme durations, not a new duration predictor.
    GTSinger's note pitches are used; target extracted F0 is never an input.
    """
    durations = torch.tensor(row['phone_durations'], dtype=torch.float64)
    notes = row['score']['ep_pitches']
    if len(notes) != len(durations) or len(row['phones']) != len(durations):
        raise ValueError('annotation lengths differ')
    if not torch.isfinite(durations).all() or (durations < 0).any():
        raise ValueError('invalid score duration')
    frames = math.ceil(round(durations.sum().item() * 16000) / 320)
    index = torch.searchsorted(durations.cumsum(0), (torch.arange(frames)+.5)/50, right=True)
    index = index.clamp(max=len(durations)-1)
    phones = [phone_dictionary[native_phone(p, row['language'])] for p in row['phones']]
    phones = torch.tensor(phones, dtype=torch.long)[index] + tokens.index('<sing_phone_0>')
    midi = torch.tensor(notes, dtype=torch.long)[index]
    if (midi < 0).any() or (midi > 137).any():
        raise ValueError('MIDI outside released vocabulary')
    return phones, midi
