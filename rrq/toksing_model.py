"""TokSing-derived score encoder with a shared depth-conditional RVQ head.

Architecture reference: ESPnet espnet2/svs/discrete/toksing.py (Apache-2.0),
Wu et al., Interspeech 2024. This is an explicit PyTorch reimplementation of
the score/length-regulation path, NOT a reproduction of TokSing's SSL+LF0
vocoder. Initial diagnostic uses provided phoneme timing, not predicted timing.
"""
import math
import torch
from torch import nn


def positional(x):
    t = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device) *
                      (-math.log(10000.) / x.shape[2]))
    pe = torch.zeros(x.shape[1:], device=x.device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(t*scale), torch.cos(t*scale)
    return x + pe.to(x.dtype)


class TokSingDAC(nn.Module):
    def __init__(self, phones, singers, languages, codebooks=9, entries=1024,
                 dim=384, layers=6, heads=4, dropout=.1):
        super().__init__()
        self.config = dict(phones=phones, singers=singers, languages=languages,
                           codebooks=codebooks, entries=entries, dim=dim,
                           layers=layers, heads=heads, dropout=dropout)
        self.phone = nn.Embedding(phones,dim)
        self.note = nn.Embedding(128,dim)
        self.duration = nn.Linear(1,dim)
        self.singer = nn.Embedding(singers,dim)
        self.language = nn.Embedding(languages,dim)
        self.technique = nn.Linear(6,dim,bias=False)
        def stack():
            layer = nn.TransformerEncoderLayer(dim,heads,dim*4,dropout,
                                                batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(dim),
                                         enable_nested_tensor=False)
        self.encoder, self.decoder = stack(), stack()
        self.code_embeddings = nn.ModuleList([nn.Embedding(entries,dim)
                                             for _ in range(codebooks-1)])
        self.depth = nn.Embedding(codebooks,dim)
        self.head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),
                                  nn.Linear(dim,entries))

    def encode_score(self, phones, midi, durations, frame_index, singer, language, techniques=None):
        if phones.ndim != 2 or frame_index.ndim != 2:
            raise ValueError('expected batched score and frame index')
        if torch.any(durations < 0) or not torch.isfinite(durations).all():
            raise ValueError('invalid score durations')
        x = (self.phone(phones)+self.note(midi)+self.duration(durations[...,None])+
             self.singer(singer)[:,None]+self.language(language)[:,None])
        if techniques is None:
            techniques = x.new_zeros(*phones.shape,6)
        if techniques.shape != (*phones.shape,6):
            raise ValueError('expected six supplied technique flags per phoneme')
        x = x + self.technique(techniques.to(x.dtype))
        x = self.encoder(positional(x))
        x = torch.gather(x,1,frame_index[...,None].expand(-1,-1,x.shape[-1]))
        return self.decoder(positional(x))

    def logits_from_hidden(self, hidden, codes):
        prefix = torch.zeros_like(hidden)
        logits = []
        for q in range(self.config['codebooks']):
            logits.append(self.head(hidden+prefix+self.depth.weight[q]))
            if q < len(self.code_embeddings):
                prefix = prefix + self.code_embeddings[q](codes[:,q])
        return torch.stack(logits,1)

    def forward(self, *, codes, **score):
        return self.logits_from_hidden(self.encode_score(**score),codes)

    @torch.no_grad()
    def generate(self, **score):
        return self.generate_with_prefix(None,**score)

    @torch.no_grad()
    def generate_with_prefix(self, forced_prefix, **score):
        """Analysis-only prefix clamp; no target suffix or audio is accepted."""
        hidden = self.encode_score(**score)
        if forced_prefix is not None:
            if (forced_prefix.ndim!=3 or forced_prefix.shape[0]!=hidden.shape[0] or
                forced_prefix.shape[2]!=hidden.shape[1] or not 1<=forced_prefix.shape[1]<self.config['codebooks'] or
                forced_prefix.dtype!=torch.long or (forced_prefix<0).any() or
                (forced_prefix>=self.config['entries']).any()):
                raise ValueError('invalid forced codec prefix')
        prefix = torch.zeros_like(hidden)
        codes = []
        for q in range(self.config['codebooks']):
            token = (forced_prefix[:,q] if forced_prefix is not None and q<forced_prefix.shape[1]
                     else self.head(hidden+prefix+self.depth.weight[q]).argmax(-1))
            codes.append(token)
            if q < len(self.code_embeddings):
                prefix = prefix + self.code_embeddings[q](token)
        return torch.stack(codes,1)
