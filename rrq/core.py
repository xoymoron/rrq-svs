"""Native residual re-quantization. Latents: [B,D,T]; codes: [Q,B,T]."""
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def validate_tokens(codes: Tensor, layers: Sequence[nn.Module], batch: int, frames: int):
    if codes.dtype != torch.long or codes.ndim != 3:
        raise ValueError('codes must be int64 [Q,B,T]')
    if codes.shape[1:] != (batch, frames) or not 1 <= len(codes) <= len(layers):
        raise ValueError('codebook, batch or frame count mismatch; cropping is forbidden')
    for code, layer in zip(codes, layers):
        if torch.any((code < 0) | (code >= layer.codebook_size)):
            raise ValueError('code index outside native codebook')


def validate_frozen(layers):
    if any(m.training for layer in layers for m in layer.modules()):
        raise ValueError('quantizer must be in eval mode')
    if any(p.requires_grad for layer in layers for p in layer.parameters()):
        raise ValueError('quantizer parameters must be frozen')
    for layer in layers:
        book = getattr(layer, '_codebook', None)
        if book is not None and hasattr(book, 'inited') and not bool(book.inited.all()):
            raise ValueError('codebook is uninitialized; load a trained checkpoint')


@torch.no_grad()
def requantize(z: Tensor, prefix: Tensor, layers: Sequence[nn.Module]) -> Tensor:
    """Clamp a supplied prefix and re-encode every remaining residual natively."""
    if z.ndim != 3 or not z.is_floating_point() or min(z.shape) <= 0:
        raise ValueError('latent must be nonempty floating [B,D,T]')
    if not torch.isfinite(z).all():
        raise ValueError('nonfinite latent')
    validate_frozen(layers)
    validate_tokens(prefix, layers, z.shape[0], z.shape[-1])
    if len(prefix) >= len(layers):
        raise ValueError('at least one residual codebook is required')
    if prefix.device != z.device:
        raise ValueError('prefix and latent devices must agree')
    residual = z.clone()
    result = []
    for k, layer in enumerate(layers):
        code = prefix[k] if k < len(prefix) else layer.encode(residual)
        residual = residual - layer.decode(code)
        result.append(code.clone())
    return torch.stack(result)


@torch.no_grad()
def make_arms(z: Tensor, clean: Tensor, predicted: Tensor, layers) -> dict[str, Tensor]:
    for codes in (clean, predicted):
        validate_tokens(codes, layers, z.shape[0], z.shape[-1])
        if len(codes) != len(layers):
            raise ValueError('full codebook sequence required for four-arm comparison')
    exposure = clean.clone()
    exposure[0] = predicted[0]
    return {'codec': clean.clone(), 'model': predicted.clone(), 'exposure': exposure,
            'rrq_oracle': requantize(z, predicted[:1], layers)}
