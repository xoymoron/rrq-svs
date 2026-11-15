"""Frozen DAC with the existing rrq.core native-quantizer interface."""
from pathlib import Path

import torch
from torch import nn

from .core import requantize
from .io import sha256


class DACLayer(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.codebook_size = native.codebook_size

    def encode(self, residual):
        return self.native.decode_latents(self.native.in_proj(residual))[1]

    def decode(self, codes):
        return self.native.out_proj(self.native.decode_code(codes))


class FrozenDAC(nn.Module):
    def __init__(self, checkpoint, device='cpu'):
        super().__init__()
        import dac
        self.codec = dac.DAC.load(str(checkpoint)).to(device).eval().requires_grad_(False)
        self.layers = nn.ModuleList([DACLayer(q) for q in self.codec.quantizer.quantizers])
        self.eval().requires_grad_(False)
        self.sample_rate = int(self.codec.sample_rate)
        self.hop = int(self.codec.hop_length)
        self.fingerprint = sha256(checkpoint)
        self.checkpoint = str(Path(checkpoint).resolve())

    @torch.inference_mode()
    def encode(self, audio):
        if audio.ndim != 3 or audio.shape[1] != 1 or not torch.isfinite(audio).all():
            raise ValueError('DAC requires finite [B,1,T] audio')
        padded = self.codec.preprocess(audio, self.sample_rate)
        z = self.codec.encoder(padded)
        _, codes, *_ = self.codec.quantizer(z)
        native = codes.permute(1, 0, 2).contiguous()
        # Test on real pretrained weights too, not only synthetic quantizers.
        if not torch.equal(requantize(z, native[:1], self.layers), native):
            raise RuntimeError('DAC adapter failed native clean-prefix identity')
        return z, native

    @torch.inference_mode()
    def decode(self, codes, samples):
        # DAC sums full-dimensional projected codes, never raw 8-D embeddings.
        z = self.codec.quantizer.from_codes(codes.permute(1, 0, 2))[0]
        audio = self.codec.decode(z)
        if not 0 <= audio.shape[-1] - samples < self.hop:
            raise ValueError('unexpected codec padding; refusing arbitrary crop')
        return audio[..., :samples]
