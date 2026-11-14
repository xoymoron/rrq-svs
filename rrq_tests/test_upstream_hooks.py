"""Execute the modified upstream methods without installing the full Fairseq stack.

Dependency stubs only supply collation, dictionary and frozen text features.
The loader, collator and criterion method bodies are extracted from their real
source files, so this exercises the actual hook wiring, not a copied algorithm.
It is not a full Fairseq trainer integration test.
"""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from einops import rearrange

from rrq.training import CachedTargets
from rrq_tests.test_training import cache

ROOT=Path(__file__).resolve().parents[1]


def method(path,classname,name,namespace):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    func=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=compile(ast.Module(body=[func],type_ignores=[]),str(path),'exec',
                 flags=__future__.annotations.compiler_flag)
    exec(code,namespace)
    return namespace[name]


class Dictionary:
    def __init__(self):
        symbols=['<s>','<pad>','</s>','<unk>']
        symbols += [line.rsplit(' ',1)[0] for line in (ROOT/'infer_tsv/dict_uni.txt').read_text().splitlines()]
        self.indices={name:i for i,name in enumerate(symbols)}
    def index(self,key): return self.indices.get(key,3)
    def pad(self): return 1
    def eos(self): return 2
    def encode_line(self,line,**kwargs): return torch.tensor([self.index(s) for s in line.split()])


def collate(values,pad,*args,**kwargs):
    result=values[0].new_full((len(values),max(len(v) for v in values)),pad)
    for i,value in enumerate(values): result[i,:len(value)]=value
    return result


@pytest.mark.parametrize('arm',['clean','exposure','rrq'])
def test_real_getitem_collater_and_criterion_use_cached_map_and_suffix_loss(tmp_path,arm):
    audio=cache(tmp_path)
    dictionary=Dictionary()
    path='research/PromptSinger/dataset/t2a_sing_t5_config_dataset.py'
    namespace={'np':np,'torch':torch,'rearrange':rearrange,
               'fairseq_data_utils':SimpleNamespace(collate_tokens=collate),
               '_collate_frames':lambda xs:torch.stack(xs)}
    funcs={name:method(path,'T2ASingT5Dataset',name,namespace)
           for name in ('__getitem__','collator_token','collater')}
    sample={'item_name':'x','task':'text_to_acoustic_sing','n_frames':8,'phone':'aa '*8,
            'f0_rescale':'100 '*8,'f0_avg':'100','audio_path':str(audio),'volume':'None',
            'gender':'female','pitch':'low','prompt':'fixed natural language prompt'}
    tags=['text_to_acoustic_sing','prompt_start','continuous_token','prompt_end',
          'f0_rescale_start','f0_rescale_end','text_start','text_end','f0_bias_start',
          'f0_bias_end','acoustic_start','acoustic_end']
    obj=SimpleNamespace(samples=[sample],uni_dict=dictionary,dict=dictionary,
                        rrq_targets=CachedTargets({'cache_dir':str(tmp_path/'cache'),'arm':arm,'rho':1}),
                        text_encoder=SimpleNamespace(encode=lambda text:torch.zeros(1,77,1024)),
                        get_tag_idx=lambda key,d:d.index(key),num_coarse_quantizers=3,
                        batch_max_frames=650,prompt_latent_length=77,
                        spec_tokens={k:torch.tensor([dictionary.index('<'+k+'>')]) for k in tags})
    obj.collator_token=lambda samples:funcs['collator_token'](obj,samples)
    encoded=funcs['__getitem__'](obj,0)
    expected_suffix=2 if arm=='rrq' else 0
    assert encoded['acoustic'][0,1]==dictionary.index(f'aco_{1024+expected_suffix}')
    assert encoded['acoustic'][0,0]==dictionary.index('aco_0' if arm=='clean' else 'aco_1')
    batch=funcs['collater'](obj,[encoded])
    assert batch['ntokens']==8*2
    assert batch['net_input']['target_acoustic_mask'].sum()>batch['target_loss_mask'].sum()
    assert torch.equal(batch['net_input']['src_tokens'],batch['target'])
    criterion=method('research/PromptSinger/criterion/acoustic_language_modeling_cross_entropy.py',
                     'AcousticLanguageModelingCrossEntropyCriterion','get_lprobs_and_targets',{'torch':torch})
    logits=torch.randn(*batch['target'].shape,len(dictionary.indices),requires_grad=True)
    model=SimpleNamespace(get_normalized_probs=lambda output,**kw:output.log_softmax(-1),
                          get_targets=lambda sample,output:sample['target'])
    log_probs,targets=criterion(None,model,logits,batch)
    assert len(targets)==16
    torch.nn.functional.nll_loss(log_probs,targets).backward()
    assert logits.grad[~batch['target_loss_mask']].abs().sum()==0


def test_real_collator_rejects_rrq_random_cropping():
    func=method('research/PromptSinger/dataset/t2a_sing_t5_config_dataset.py',
                'T2ASingT5Dataset','collator_token',{'torch':torch,'np':np,'rearrange':rearrange})
    obj=SimpleNamespace(spec_tokens={'text_to_acoustic_sing':torch.tensor([1])},
                        batch_max_frames=2,rrq_targets=object())
    with pytest.raises(ValueError,match='crop BEFORE'):
        func(obj,[{'task':torch.tensor([1]),'phone':torch.ones(3),
                   'acoustic':torch.ones(3,3),'f0_rescale':torch.ones(3)}])
