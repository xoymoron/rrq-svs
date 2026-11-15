import pytest
import os


def test_global_batches_cover_each_phrase_once_and_pair_ranks():
    from rrq.distributed import global_batches
    rows = list(range(472))
    batches = global_batches(rows, 2)
    assert len(batches) == 236
    assert [row for batch in batches for row in batch] == rows
    assert set(batch[0] for batch in batches).isdisjoint(batch[1] for batch in batches)
    with pytest.raises(ValueError, match='divisible'):
        global_batches(rows[:-1], 2)


def test_validation_gather_requires_exact_coverage_and_language_macro():
    from rrq.distributed import validation_result
    values = [dict(id='a', language='A', ce=1.), dict(id='b', language='A', ce=3.),
              dict(id='c', language='B', ce=8.)]
    result = validation_result(values, ['a', 'b', 'c'])
    assert result['language_macro_ce'] == 5.
    with pytest.raises(ValueError, match='coverage'):
        validation_result(values + values[:1], ['a', 'b', 'c'])
    with pytest.raises(ValueError, match='coverage'):
        validation_result(values[:-1], ['a', 'b', 'c'])


def test_single_process_context_preserves_local_values():
    from rrq.distributed import Distributed
    ctx = Distributed()
    assert ctx.rank == 0 and ctx.world == 1 and ctx.primary
    assert ctx.gather([{'id': 'x'}]) == [{'id': 'x'}]


def test_scheduler_uses_torchrun_for_one_two_gpu_training_job(tmp_path):
    from rrq.experiment_schedule import command_for
    command = command_for(tmp_path, 0, 'rrq.experiment', ['train', '--stage', 'warmup'], 2)
    assert 'torch.distributed.run' in command
    assert '--nproc-per-node=2' in command
    assert command[command.index('--module') + 1] == 'rrq.experiment'
    assert 'torch.distributed.run' not in command_for(tmp_path, 1, 'rrq.experiment', ['render'], 2)


def _gradient_worker(rank, rendezvous):
    import torch
    import torch.distributed as dist
    from rrq.distributed import Distributed
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank))
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    ctx = Distributed()
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.5)
        ddp = ctx.wrap(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # Different valid lengths; phrase mean gives equal weight to each rank.
        x = torch.full((rank * 2 + 1, 1), 1. if rank == 0 else 3.)
        loss = ddp(x).square().mean()
        loss.backward()
        torch.testing.assert_close(model.weight.grad, torch.tensor([[5.]]))
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.zeros(1, 1), atol=1e-7, rtol=0)
        assert len(ctx.assert_synced(model)) == 64
    finally:
        ctx.close()


@pytest.mark.skipif(os.name == 'nt', reason='Linux server validates actual two-process Gloo collectives')
def test_two_process_ddp_matches_unpadded_global_mean_gradient(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_gradient_worker, args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
