"""Exercise actual optimization/provenance/resume paths without a large codec fixture."""
import json
from pathlib import Path
import shutil

import torch
import pytest

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
        e.acoustic.put('fixture',dict(codes=torch.zeros(9,1,26,dtype=torch.int16),
                                     samples=13230,sample_rate=44100,hop=512))
    write_json(e.root/'cache-complete.json',dict(protocol_sha256=e.hash,n=4))
    (e.root/'common').mkdir();torch.manual_seed(123)
    model=TokSingDAC(**model_config(e.vocab,cfg))
    atomic_save(dict(protocol_sha256=e.hash,arm='common',model=model.state_dict(),model_config=model.config),e.root/'common/best.pt')
    common_hash=sha256(e.root/'common/best.pt')
    for r in rows[:3]:
        predicted=torch.ones(9,1,26,dtype=torch.int16);corrected=torch.full_like(predicted,2);corrected[0]=1
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
        (e.root/arm/'best.pt').unlink()  # interrupted or missing selection artifact
        runner.train_stage(e,Distributed(),torch.device('cpu'),arm)
        resumed=e.checkpoint(e.root/arm/'last.pt')
        assert all(torch.equal(full['model'][k],resumed['model'][k]) for k in full['model'])
    assert len({r['schedule_sha256'] for _,r in records})==1
    assert len({r['shared_predicted_mixture_sha256'] for _,r in records})==1
    assert len({r['parameter_sha256'] for _,r in records})>1


def test_last_checkpoint_is_written_only_after_best_and_epoch_report(tmp_path,monkeypatch):
    torch.set_num_threads(1);e=make_experiment(tmp_path)
    monkeypatch.setattr(runner,'FrozenDAC',lambda *a:None)
    monkeypatch.setattr(runner,'tracker',lambda *a,**k:None)
    monkeypatch.setattr(runner,'evaluate_panel',lambda *a,**k:dict(
        language_macro={'generated':{'mrstft_spectral_convergence':.5}},per_language={},diagnostics={},samples=[]))
    save=runner.atomic_save
    def ordered_save(value,path):
        path=Path(path)
        if path.name=='last.pt':
            assert (path.parent/'best.pt').exists(), 'last published before best'
            assert (path.parent/f"epoch-{value['epoch']:03d}.json").exists(), 'last published before report'
        save(value,path)
    monkeypatch.setattr(runner,'atomic_save',ordered_save)
    runner.train_stage(e,Distributed(),torch.device('cpu'),'baseline')


def test_changing_imported_selection_code_invalidates_protocol(tmp_path,monkeypatch):
    import rrq.experiment as module
    e=make_experiment(tmp_path);original=module.sha256
    monkeypatch.setattr(module,'sha256',lambda p:'changed' if Path(p).name=='toksing_run.py' else original(p))
    with pytest.raises(ValueError,match='protocol mismatch'):
        Experiment(e.prepared,e.root,e.codec,tmp_path/'config.json')


def test_resumed_cache_still_reports_reconstruction_on_validation_only(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    e=make_experiment(tmp_path)
    for row in e.rows:row['audio']='fixture.wav'
    audio=np.sin(np.arange(13230)*2*np.pi*220/44100).astype(np.float32)
    codec=SimpleNamespace(sample_rate=44100,decode=lambda codes,samples:torch.from_numpy(audio)[None,None])
    monkeypatch.setattr(runner,'FrozenDAC',lambda *a:codec)
    monkeypatch.setattr(runner,'sha256',lambda path:'fixture')
    monkeypatch.setattr(runner,'load_audio',lambda *a:audio.copy())
    runner.cache_stage(e,Distributed(),torch.device('cpu'))
    report=json.loads((e.root/'codec-reconstruction.json').read_text())
    assert [r['id'] for r in report['samples']]==['item-2']
    assert report['language_macro']['codec']['mrstft_spectral_convergence']==0
    assert report['protocol_sha256']==e.hash
    completion=json.loads((e.root/'cache-complete.json').read_text())
    assert completion['acoustic_encoded']==0 and completion['acoustic_reused']==4
    assert report['clean_prefix_audited_samples']==0


def test_revised_split_and_duration_reuse_tokens_but_reject_old_target_bank(tmp_path):
    e=make_experiment(tmp_path);original=e.read_cache(e.by_id['item-0'])
    acoustic_sha=sha256(e.acoustic.path('fixture'))
    revised=tmp_path/'revised';shutil.copytree(e.prepared,revised)
    rows=e.rows
    rows[0]['split']='valid';rows[2]['split']='train';rows[0]['phone_durations']=[.2,.1]
    manifest=revised/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(revised/'panels.json',dict(train=['item-2'],valid=['item-0']))
    write_json(revised/'audit.json',dict(manifest_sha256=sha256(manifest),config={}))
    new=Experiment(revised,tmp_path/'revised-output',e.codec,tmp_path/'config.json')
    changed=new.read_cache(new.by_id['item-0'])
    assert torch.equal(changed['codes'],original['codes'])
    assert not torch.equal(changed['frame_index'],original['frame_index'])
    assert sha256(new.acoustic.path('fixture'))==acoustic_sha
    assert new.hash!=e.hash
    shutil.copyfile(e.bank/'item-0.pt',new.bank/'item-0.pt')
    with pytest.raises(ValueError,match='target bank provenance'):
        new.read_bank(new.by_id['item-0'],sha256(e.root/'common/best.pt'))
