"""Frozen native EnCodec 24 kHz mono RVQ, with exact native quantizer calls."""
import hashlib
import math
import torch
from torch import nn


class EnCodecLayer(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.codebook_size = native._codebook.embed.shape[0]

    def encode(self, residual):
        return self.native.encode(residual)

    def decode(self, tokens):
        return self.native.decode(tokens)


class FrozenEnCodec(nn.Module):
    sample_rate, hop, codebooks = 24000, 320, 8

    def __init__(self, device='cpu'):
        super().__init__()
        from encodec import EncodecModel
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(6.)
        if self.codec.normalize or self.codec.segment_length is not None:
            raise ValueError('this adapter requires unsegmented, unnormalized 24 kHz EnCodec')
        digest = hashlib.sha256()
        for name, value in self.codec.state_dict().items():
            digest.update(name.encode()); digest.update(value.cpu().numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        self.layers = nn.ModuleList([EnCodecLayer(q) for q in self.codec.quantizer.vq.layers[:self.codebooks]])
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, audio):
        if audio.ndim != 3 or audio.shape[1] != 1 or not torch.isfinite(audio).all():
            raise ValueError('EnCodec requires finite mono [B,1,T]')
        z = self.codec.encoder(audio)
        codes = self.codec.quantizer.encode(z,self.codec.frame_rate,self.codec.bandwidth)
        if codes.shape[0] != self.codebooks or codes.shape[-1] != math.ceil(audio.shape[-1]/self.hop):
            raise ValueError('unexpected native EnCodec clock or codebook count')
        return z, codes

    @torch.inference_mode()
    def decode(self, codes, samples):
        if codes.ndim != 3 or codes.shape[0] != self.codebooks or codes.dtype != torch.long:
            raise ValueError('expected full EnCodec tokens [Q,B,T]')
        if (codes < 0).any() or (codes >= 1024).any():
            raise ValueError('invalid EnCodec code')
        z = self.codec.quantizer.decode(codes)
        audio = self.codec.decoder(z)
        if not 0 <= audio.shape[-1]-samples < self.hop:
            raise ValueError('unexpected endpoint padding; no time stretching')
        return audio[..., :samples]
