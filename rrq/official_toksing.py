"""Import the pinned official TokSing; adapt only its input contract/output head.

The upstream encoder, decoder, duration/F0 predictors, LF0 enhancement and loss
are executed directly. No copied/reimplemented upstream forward is used here.
"""
from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import torch.nn.functional as F
import yaml

COMMIT = '5f142ef90bf87ee0f4c2d6907f7a9462b3854316'
ROOT = Path(__file__).resolve().parents[1]


def upstream_root():
    path = Path(os.environ.get('ESPNET_ROOT', ROOT/'artifacts/upstream-espnet')).resolve()
    if not (path/'espnet2/svs/discrete/toksing.py').is_file():
        raise RuntimeError('Clone pinned ESPnet and set ESPNET_ROOT; see scripts/setup_official.sh')
    return path


def source_provenance():
    root = upstream_root()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no').splitlines()
    if commit != COMMIT or dirty:
        raise RuntimeError('Official source revision changed or has tracked modifications')
    path = root/'espnet2/svs/discrete/toksing.py'
    return dict(commit=commit, root=str(root), modified_tracked_files=dirty,
                toksing_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def upstream_config(*, phones, singers, languages, codebooks=8, entries=1024):
    source_provenance()
    path = upstream_root()/'egs2/opencpop/svs2/conf/tuning/train_toksing.yaml'
    conf = yaml.safe_load(path.read_text())['svs_conf']
    conf.update(idim=phones, odim=entries, spks=singers, langs=languages,
                discrete_token_layers=codebooks, codec_codebook=0)
    return conf


class PrefixProjection(nn.Module):
    """Keep original projection weights, add preceding-codebook input only.

    Zero initial embeddings preserve the original parallel head. Inference
    predicts each RVQ depth at once across time, then conditions the next depth.
    """
    def __init__(self, projection, codebooks, entries):
        super().__init__()
        if projection.out_features != codebooks*entries:
            raise ValueError('projection size does not match RVQ layout')
        self.projection = projection
        self.codebooks, self.entries = codebooks, entries
        self.prefix_embeddings = nn.ModuleList([
            nn.Embedding(entries, projection.in_features) for _ in range(codebooks-1)])
        for embedding in self.prefix_embeddings:
            nn.init.zeros_(embedding.weight)
        self.context = None
        self.forced_prefix = None

    @contextmanager
    def force_prefix(self, codes):
        if self.context is not None or self.forced_prefix is not None:
            raise RuntimeError('nested prefix context is not supported')
        self.forced_prefix = codes
        try:
            yield
        finally:
            self.forced_prefix = None

    @contextmanager
    def conditioning(self, codes):
        if self.context is not None:
            raise RuntimeError('nested prefix context is not supported')
        self.context = codes
        try:
            yield
        finally:
            self.context = None

    def forward(self, hidden):
        context = self.context
        forced = self.forced_prefix
        if forced is not None:
            if (forced.ndim!=3 or forced.shape[0]!=hidden.shape[0] or forced.shape[2]!=hidden.shape[1]
                    or not 1<=forced.shape[1]<self.codebooks or forced.dtype!=torch.long
                    or (forced<0).any() or (forced>=self.entries).any()):
                raise ValueError('forced prefix must be valid [B,K,T], 1 <= K < Q')
        if context is not None:
            if (context.ndim != 3 or context.shape != (hidden.shape[0], self.codebooks, hidden.shape[1])
                    or context.dtype != torch.long or (context < 0).any() or (context >= self.entries).any()):
                raise ValueError('prefix context must be valid [B,Q,T] tokens')
        prefix = torch.zeros_like(hidden)
        outputs = []
        for q in range(self.codebooks):
            start, end = q*self.entries, (q+1)*self.entries
            value = F.linear(hidden+prefix, self.projection.weight[start:end], self.projection.bias[start:end])
            outputs.append(value)
            if q < self.codebooks-1:
                token = (forced[:,q] if forced is not None and q<forced.shape[1]
                         else context[:, q] if context is not None else value.argmax(-1))
                prefix = prefix + self.prefix_embeddings[q](token)
        return torch.cat(outputs, dim=-1)


class OfficialTokSing(nn.Module):
    def __init__(self, config, prefix_conditioning=False):
        super().__init__()
        self.provenance = source_provenance()
        root = upstream_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = importlib.import_module('espnet2.svs.discrete.toksing')
        if Path(module.__file__).resolve() != root/'espnet2/svs/discrete/toksing.py':
            raise RuntimeError('A different ESPnet installation shadows the pinned original')
        self.config = dict(config)
        if not config.get('predict_pitch') or not config.get('use_discrete_token'):
            raise ValueError('official melody predictor and token pathway must remain enabled')
        self.base = module.TokSing(**config)
        self.codebooks, self.entries = config['discrete_token_layers'], config['odim']
        self.prefix_conditioning = bool(prefix_conditioning)
        if self.prefix_conditioning:
            self.base.linear_projection = PrefixProjection(self.base.linear_projection, self.codebooks, self.entries)

    def _validate_score(self, score):
        phones, ds = score['phones'], score['durations']
        if phones.ndim != 2 or phones.shape[0] != 1:
            raise ValueError('unpadded batch of one phrase per GPU is required')
        if (ds.shape != phones.shape or ds.dtype != torch.long or (ds < 0).any()
                or ds.sum() <= 0 or (ds >= self.config['duration_dim']).any()):
            raise ValueError('invalid duration counts or duration embedding overflow')
        if score['midi'].shape != phones.shape or (phones <= 0).any():
            raise ValueError('invalid phone/MIDI layout; zero phone is reserved for padding')

    def training_arguments(self, score, codes, lf0):
        self._validate_score(score)
        b, q, t = codes.shape
        if q != self.codebooks or t != int(score['durations'].sum()) or codes.dtype != torch.long:
            raise ValueError('target frame count/codebooks differ from score alignment')
        if lf0.shape != (b,t,1) or not torch.isfinite(lf0).all():
            raise ValueError('require finite frame-aligned LF0 supervision')
        device = codes.device
        ilens = torch.tensor([score['phones'].shape[1]], device=device)
        flens = torch.tensor([t], device=device)
        return dict(text=score['phones'], text_lengths=ilens,
                    feats=lf0.new_zeros(b,t,1), feats_lengths=flens,
                    # Official joint_training interface returns logits and takes
                    # flat tensors. It executes the same supervised loss pathway.
                    label=score['phones'], label_lengths=ilens,
                    melody=score['midi'], melody_lengths=ilens,
                    duration=score['durations'], duration_lengths=ilens,
                    pitch=lf0, pitch_lengths=flens,
                    sids=score['singer'], lids=score['language'],
                    discrete_token=codes.permute(0,2,1).reshape(b,t*q),
                    discrete_token_lengths=flens*q, discrete_token_lengths_frame=flens)

    def forward(self, *, score, codes, lf0, context=None):
        from contextlib import nullcontext
        args = self.training_arguments(score, codes, lf0)
        if not self.prefix_conditioning and context is not None:
            raise ValueError('parallel original does not take codec prefix context')
        manager = self.base.linear_projection.conditioning(codes if context is None else context) if self.prefix_conditioning else nullcontext()
        with manager:
            loss, stats, logits = self.base(**args, joint_training=True)
        b, _, t = codes.shape
        return dict(loss=loss, stats=stats, logits=logits.view(b,t,self.codebooks,self.entries).permute(0,2,1,3))

    @torch.no_grad()
    def generate(self, *, phones, midi, durations, singer, language, use_supplied_duration=True,
                 forced_prefix=None):
        from contextlib import nullcontext
        score = dict(phones=phones,midi=midi,durations=durations,singer=singer,language=language)
        self._validate_score(score)
        if forced_prefix is not None and not self.prefix_conditioning:
            raise ValueError('common-prefix diagnostic requires a prefix-conditioned model')
        # Controlled acoustic SVS: supply annotation timing, NEVER target LF0 or tokens.
        # The official duration predictor is retained/trained; unforced generation
        # remains available. The hook changes only LR's duration input for this call.
        hook = None
        if use_supplied_duration:
            hook = self.base.length_regulator.register_forward_pre_hook(lambda _, args: (args[0], durations, *args[2:]))
        try:
            manager=(self.base.linear_projection.force_prefix(forced_prefix)
                     if forced_prefix is not None else nullcontext())
            with manager:
                out = self.base.inference(text=phones, label={'score':phones}, melody={'score':midi},
                                          duration={'score_syb':durations,'lab':durations},
                                          sids=singer,lids=language,use_teacher_forcing=False)
        finally:
            if hook is not None:
                hook.remove()
        codes = out['feat_gen'].reshape(-1,self.codebooks).T.unsqueeze(0).long()
        if forced_prefix is not None:
            # Diagnostic only: common MODEL prediction, never reference codes.
            # This clamp is not used by primary free generation.
            codes[:,:forced_prefix.shape[1]]=forced_prefix
        return dict(codes=codes, lf0=out['pitch'], logits=out['logits'])

    def parameter_report(self):
        values = {name:sum(p.numel() for p in getattr(self.base,name).parameters()) for name in
                  ('encoder','decoder','f0_predictor','proj_pitch','lf0_mapping','duration_predictor','linear_projection')}
        return dict(total=sum(p.numel() for p in self.parameters()), components=values)
