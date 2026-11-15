"""Exercise actual optimization/provenance/resume paths without a large codec fixture."""
import json
from pathlib import Path
import shutil

import torch

from rrq.distributed import Distributed
from rrq.experiment import Experiment,model_config
from rrq.io import sha256,write_json
from rrq.toksing_data import annotation_hash
from rrq.toksing_model import TokSingDAC
from rrq.toksing_run import atomic_save
from rrq import matched_run as runner


def make_experiment(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir();codec=tmp_path/'codec.pth';codec.write_bytes(b'test-only-codec')
    rows=[]
    for i,split in enumerate(['train','train','valid','test']):
        rows.append(dict(id=f'item-{i}',corpus='gtsinger',singer='s',language='English',split=split,
                         song_group=f'song-{i}',phones=['a','b'],phone_durations=[.1,.2],
                         score={'ep_pitches':[60,64]},source_sha256='fixture'))
    manifest=prepared/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(prepared/'panels.json',dict(train=['item-0'],valid=['item-2']))
    write_json(prepared/'audit.json',dict(manifest_sha256=sha256(manifest),config={}))
    cfg=json.loads((Path(__file__).parents[1]/'config/gtsinger-rrq-v1.json').read_text())
    cfg.update(world_size=1,branch_epochs=2,gradient_accumulation=2,
               model=dict(dim=8,layers=1,heads=2,dropout=.1))
    config=tmp_path/'config.json';write_json(config,cfg)
    e=Experiment(prepared,tmp_path/'output',codec,config)
    for r in rows:
        atomic_save(dict(id=r['id'],annotation_sha256=annotation_hash(r),source_sha256='fixture',
                         codec_sha256=e.protocol['codec_sha256'],codes=torch.zeros(9,1,4,dtype=torch.int16),
                         frame_index=torch.tensor([0,0,1,1],dtype=torch.int32),samples=2048,sample_rate=44100,hop=512),e.cache/(r['id']+'.pt'))
    write_json(e.root/'cache-complete.json',dict(protocol_sha256=e.hash,n=4))
    (e.root/'common').mkdir();torch.manual_seed(123)
    model=TokSingDAC(**model_config(e.vocab,cfg))
    atomic_save(dict(protocol_sha256=e.hash,model=model.state_dict(),model_config=model.config),e.root/'common/best.pt')
    common_hash=sha256(e.root/'common/best.pt')
    for r in rows[:3]:
        predicted=torch.ones(9,1,4,dtype=torch.int16);corrected=torch.full_like(predicted,2);corrected[0]=1
        atomic_save(dict(id=r['id'],predicted=predicted,corrected=corrected,common_sha256=common_hash,
                         protocol_sha256=e.hash,annotation_sha256=annotation_hash(r),source_sha256='fixture'),e.bank/(r['id']+'.pt'))
    write_json(e.root/'target-bank-complete.json',dict(protocol_sha256=e.hash,common_sha256=common_hash,oracle_room_detected=True))
    return e


def test_three_arms_share_initialization_schedule_and_resume_exactly(tmp_path,monkeypatch):
    torch.set_num_threads(1);e=make_experiment(tmp_path);initial_states=[]
    monkeypatch.setattr(runner,'FrozenDAC',lambda *a:None)
    monkeypatch.setattr(runner,'tracker',lambda *a,**k:None)
    def evaluation(e,d,device,model,codec,ids,common_hash=None,folder=None,ablate=False):
        initial_states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        return dict(language_macro={'generated':{'mrstft_spectral_convergence':.5}},
                    per_language={'English':{'generated':{'mrstft_spectral_convergence':.5}}},
                    diagnostics={'English':{}},samples=[])
    monkeypatch.setattr(runner,'evaluate_panel',evaluation)
    original_save=runner.atomic_save
    def save(value,path):
        original_save(value,path)
        if Path(path).name=='last.pt' and value.get('epoch')==1:
            original_save(value,Path(path).parent/'resume-fixture.pt')
    monkeypatch.setattr(runner,'atomic_save',save)
    records=[]
    for arm in ['baseline','exposure','rrq']:
        first=len(initial_states)
        runner.train_stage(e,Distributed(),torch.device('cpu'),arm)
        initial=initial_states[first]
        if records:
            assert all(torch.equal(initial[k],records[0][0][k]) for k in initial)
        full=e.checkpoint(e.root/arm/'last.pt');assert full['step']==2
        record=json.loads((e.root/arm/'epoch-002.json').read_text())
        records.append((initial,record))
        shutil.copyfile(e.root/arm/'resume-fixture.pt',e.root/arm/'last.pt')
        runner.train_stage(e,Distributed(),torch.device('cpu'),arm)
        resumed=e.checkpoint(e.root/arm/'last.pt')
        assert all(torch.equal(full['model'][k],resumed['model'][k]) for k in full['model'])
    assert len({r['schedule_sha256'] for _,r in records})==1
    assert len({r['shared_predicted_mixture_sha256'] for _,r in records})==1
    assert len({r['parameter_sha256'] for _,r in records})>1
