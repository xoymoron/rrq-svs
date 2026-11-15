"""Shared audio/codec objects and cheap, split-independent score alignments."""
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from .dac_codec import FrozenDAC, DACLayer, clean_prefix_audit
from .io import safe_id, sha256
from .toksing_data import frame_condition, load_audio


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def tensor_digest(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(str((str(value.dtype),tuple(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def atomic_save(value,path,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        temporary=Path(f.name)
    try:
        torch.save(value,temporary)
        if exclusive:
            # Atomic publish without replacing an object another process wrote.
            if os.name=='nt':
                os.rename(temporary,path)  # Windows rename fails if destination exists.
            else:
                os.link(temporary,path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encoding_contract():
    # Neither score alignment, split, predictor nor training settings belong here.
    return dict(version=1,sample_rate=44100,hop=512,codebooks=9,entries=1024,
        dtype='float32',tf32=False,deterministic=True,cuda=torch.version.cuda,
        implementation={name:fingerprint(inspect.getsource(fn)) for name,fn in
            [('load_audio',load_audio),('DAC_init',FrozenDAC.__init__),('DAC_encode',FrozenDAC.encode)]},
        packages={name:importlib.metadata.version(name) for name in
            ('torch','torchaudio','numpy','scipy','soundfile','descript-audio-codec')})


def audit_recipe():
    return fingerprint(dict(core=sha256(Path(__file__).with_name('core.py')),
        audit=inspect.getsource(clean_prefix_audit),layers=inspect.getsource(DACLayer)))


class AcousticCache:
    def __init__(self,root,codec_sha256,contract=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.codec_sha256=codec_sha256
        self.contract=encoding_contract() if contract is None else contract
        self.contract_sha256=fingerprint(self.contract)
        self.audit_recipe=audit_recipe()

    def key(self,source_sha256):
        return fingerprint(dict(source=source_sha256,codec=self.codec_sha256,encoding=self.contract_sha256))

    def path(self,source_sha256):
        key=self.key(source_sha256)
        return self.root/key[:2]/(key+'.pt')

    def validate(self,value,source_sha256):
        expected=dict(format='rrq-acoustic-v2',acoustic_key=self.key(source_sha256),
            source_sha256=source_sha256,codec_sha256=self.codec_sha256,
            encoding_contract_sha256=self.contract_sha256,sample_rate=44100,hop=512)
        if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('acoustic cache provenance mismatch')
        codes=value['codes'];samples=value['samples']
        if (not isinstance(samples,int) or samples<=0 or not isinstance(codes,torch.Tensor) or
            codes.dtype!=torch.int16 or tuple(codes.shape)!=(9,1,math.ceil(samples/512)) or
            (codes<0).any() or (codes>=1024).any()):raise ValueError('invalid acoustic token shape/range/clock')
        if tensor_digest(codes)!=value['codes_sha256']:raise ValueError('acoustic token checksum mismatch')
        return value

    def read(self,source_sha256):
        value=torch.load(self.path(source_sha256),map_location='cpu',weights_only=True)
        self.validate(value,source_sha256)
        if value.get('clean_prefix_audit_recipe')!=self.audit_recipe:
            value['clean_prefix_audit']=None
        return value

    def put(self,source_sha256,record):
        if record['codes'].dtype not in (torch.int16,torch.int32,torch.int64):
            raise ValueError('native codec tokens must be integer tensors')
        # Validate before narrowing: int64 overflow must not become a valid token.
        if (record['codes']<0).any() or (record['codes']>=1024).any():raise ValueError('invalid native token range')
        codes=record['codes'].detach().cpu().short().contiguous()
        value=dict(format='rrq-acoustic-v2',acoustic_key=self.key(source_sha256),
            source_sha256=source_sha256,codec_sha256=self.codec_sha256,
            encoding_contract_sha256=self.contract_sha256,codes=codes,codes_sha256=tensor_digest(codes),
            samples=record['samples'],sample_rate=record['sample_rate'],hop=record['hop'],
            clean_prefix_audit=record.get('clean_prefix_audit'),
            clean_prefix_audit_recipe=record.get('clean_prefix_audit_recipe'))
        self.validate(value,source_sha256)
        if not self.path(source_sha256).exists():
            try:
                atomic_save(value,self.path(source_sha256),exclusive=True)
            except FileExistsError:
                pass  # A concurrent writer won; verify that it wrote identical tokens.
        old=self.read(source_sha256)
        if old['codes_sha256']!=value['codes_sha256'] or old['samples']!=value['samples']:
            raise ValueError('refuse to replace immutable acoustic cache with different tokens')
        return old


class AlignmentCache:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.recipe=fingerprint(inspect.getsource(frame_condition))

    def read(self,row,audio):
        frames=audio['codes'].shape[-1]
        key=fingerprint(dict(acoustic_key=audio['acoustic_key'],samples=audio['samples'],
            sample_rate=audio['sample_rate'],hop=audio['hop'],frames=frames,recipe=self.recipe,
            phones=row['phones'],durations=row['phone_durations'],
            midi=row['score'].get('ep_pitches',row['score'].get('notes'))))
        path=self.root/(safe_id(row['id'])+'.pt')
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=True)
            if value.get('alignment_sha256')==key and value.get('id')==row['id']:
                index=value['frame_index']
                if (index.dtype!=torch.int32 or tuple(index.shape)!=(frames,) or
                    (index<0).any() or (index>=len(row['phones'])).any() or
                    tensor_digest(index)!=value['index_sha256']):raise ValueError('alignment cache checksum/shape mismatch')
                return index.long(),False
        condition=frame_condition(row,frames,audio['samples'],audio['sample_rate'],audio['hop'])
        index=torch.from_numpy(condition['phone_index']).int()
        atomic_save(dict(id=row['id'],alignment_sha256=key,acoustic_key=audio['acoustic_key'],
            frame_index=index,index_sha256=tensor_digest(index)),path)
        return index.long(),True
