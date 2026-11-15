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
