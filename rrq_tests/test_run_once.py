import json
import subprocess

import pytest

from rrq.io import write_json
from rrq.run_once import run_pipeline


def fixture(tmp_path):
    write_json(tmp_path/'protocol.json',{'sha256':'protocol'})
    write_json(tmp_path/'readiness.json',dict(protocol_sha256='protocol',ready_for_common=True,training_started=False))
    return tmp_path


def test_pipeline_runs_once_in_order_then_refuses_second_launch(tmp_path):
    root=fixture(tmp_path);calls=[]
    run_pipeline(root,lambda stage:calls.append(stage))
    assert calls==['common','targets','branches','final-test','publish']
    assert json.loads((root/'experiment-state.json').read_text())['status']=='complete'
    with pytest.raises(ValueError,match='already launched'):run_pipeline(root,lambda s:calls.append(s))
    assert len(calls)==5


def test_failed_gate_never_starts_branches_and_resume_skips_finished_common(tmp_path):
    root=fixture(tmp_path);calls=[]
    def fail(stage):
        calls.append(stage)
        if stage=='targets':raise subprocess.CalledProcessError(1,['targets'])
    with pytest.raises(subprocess.CalledProcessError):run_pipeline(root,fail)
    assert calls==['common','targets']
    state=json.loads((root/'experiment-state.json').read_text())
    assert state['status']=='failed' and state['completed']==['common']
    run_pipeline(root,lambda stage:calls.append(stage),resume=True)
    assert calls==['common','targets','targets','branches','final-test','publish']


def test_resume_rejects_protocol_change_and_untracked_checkpoint(tmp_path):
    root=fixture(tmp_path);(root/'common').mkdir();(root/'common/last.pt').write_bytes(b'checkpoint')
    with pytest.raises(ValueError,match='untracked'):run_pipeline(root,lambda s:None)
    (root/'common/last.pt').unlink()
    run_pipeline(root,lambda s:None)
    write_json(root/'protocol.json',{'sha256':'changed'})
    with pytest.raises(ValueError,match='protocol'):run_pipeline(root,lambda s:None,resume=True)


def test_final_comparison_rejects_mismatched_budget_or_schedule(tmp_path):
    import torch
    from rrq.io import sha256
    from rrq.run_once import verify_matched
    (tmp_path/'common').mkdir();(tmp_path/'common/best.pt').write_bytes(b'common')
    protocol=dict(sha256='p',config={'branch_epochs':2})
    write_json(tmp_path/'protocol.json',protocol)
    for arm in ('baseline','exposure','rrq'):
        folder=tmp_path/arm;folder.mkdir()
        torch.save(dict(protocol_sha256='p',initial_checkpoint_sha256=sha256(tmp_path/'common/best.pt'),
                        arm=arm,epoch=2,step=20),folder/'last.pt')
        for epoch in (1,2):
            write_json(folder/f'epoch-{epoch:03d}.json',dict(protocol_sha256='p',step=epoch*10,
                draws=80,schedule_sha256='schedule',shared_predicted_mixture_sha256='mixture'))
    verify_matched(tmp_path)
    r=json.loads((tmp_path/'rrq/epoch-002.json').read_text());r['schedule_sha256']='different'
    write_json(tmp_path/'rrq/epoch-002.json',r)
    with pytest.raises(ValueError,match='matched'):verify_matched(tmp_path)
