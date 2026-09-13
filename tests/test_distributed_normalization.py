"""Exercise repeated distributed merges against the actual pooled samples."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from distributed_normalization import install_finite_action_guard, synchronize_normalizer


def _distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        norm = SimpleNamespace(
            count=torch.tensor(0, dtype=torch.long),
            _mean=torch.zeros(1, 2), _var=torch.ones(1, 2), _std=torch.ones(1, 2),
        )
        history = []
        # Unequal local batch sizes, changing distributions, and a large offset
        # exercise weighted merging and cancellation without a physics runtime.
        for iteration in range(100):
            batches = [torch.arange(2 * n).reshape(n, 2).float() * 0.1
                       + torch.tensor([1000., -1000.]) + iteration * 0.01 + r
                       for r, n in enumerate((3, 5))]
            batch = batches[rank]
            norm.count += len(batch)
            rate = len(batch) / norm.count
            delta = batch.mean(0, keepdim=True) - norm._mean
            norm._mean += rate * delta
            norm._var += rate * (batch.var(0, unbiased=False, keepdim=True) - norm._var
                                + delta * (batch.mean(0, keepdim=True) - norm._mean))
            # Match RSL-RL's rollout-produced buffer before its PPO update.
            with torch.inference_mode():
                norm._std = norm._var.sqrt()
            synchronize_normalizer(norm)
            history.extend(batches)
            pooled = torch.cat(history)
            assert norm.count.item() == 4 * (iteration + 1)
            torch.testing.assert_close(norm._mean, pooled.mean(0, keepdim=True), rtol=0, atol=0.002)
            torch.testing.assert_close(norm._var, pooled.var(0, unbiased=False, keepdim=True), rtol=0, atol=0.002)
            assert torch.isfinite(norm._std).all()
    finally:
        dist.destroy_process_group()


def test_repeated_distributed_merges_preserve_history_without_exponential_counts(tmp_path):
    mp.spawn(_distributed_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_actions_never_reach_physics(bad):
    received = []
    env = SimpleNamespace(step=lambda actions: received.append(actions))
    install_finite_action_guard(env)
    with pytest.raises(FloatingPointError, match="before stepping PhysX"):
        env.step(torch.tensor([[0., bad]]))
    assert not received
    actions = torch.tensor([[0., 1.]])
    env.step(actions)
    assert received == [actions]
