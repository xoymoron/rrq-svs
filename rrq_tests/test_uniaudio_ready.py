import importlib.util
import pytest


def test_single_gpu_preflight_cannot_claim_two_gpu_readiness():
    assert importlib.util.find_spec('rrq.uniaudio_ready')
    from rrq.uniaudio_ready import validate_preflight
    report=dict(protocol_sha256='p',optimizer_steps=0,status='passed',weights_unchanged=True,
                world_size=1,results=[dict(language='Chinese',arm=arm)
                                     for arm in ('baseline','exposure','rrq')])
    with pytest.raises(ValueError,match='two-GPU'):
        validate_preflight(report,'p',{'Chinese'})
    report.update(world_size=2,results=report['results']*2)
    validate_preflight(report,'p',{'Chinese'})
