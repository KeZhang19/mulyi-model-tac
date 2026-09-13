"""Check synchronization auditing with real, unequal gradients on two workers."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from distributed_training import install_distributed_training_audit


def _worker(rank, rendezvous, log_dir, mode):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        policy = torch.nn.Linear(2, 1)
        algorithm = SimpleNamespace(
            policy=policy, rnd=None, is_multi_gpu=True, gpu_world_size=2,
            allow_distributed_parameter_sync=mode != "disabled",
        )
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

        def reduce():
            if mode != "noop":
                for parameter in policy.parameters():
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= 2

        def update():
            optimizer.zero_grad()
            policy(torch.full((3, 2), float(rank + 1))).square().mean().backward()
            if mode != "no_reduce":
                algorithm.reduce_parameters()
            optimizer.step()

        algorithm.reduce_parameters = reduce
        algorithm.update = update
        runner = SimpleNamespace(alg=algorithm, device="cpu", is_distributed=True, log_dir=log_dir)
        if mode == "disabled":
            with pytest.raises(RuntimeError, match="not enabled"):
                install_distributed_training_audit(runner, expected_world_size=2)
            return
        install_distributed_training_audit(runner, expected_world_size=2)
        if mode in {"noop", "no_reduce"}:
            message = "do not equal the mean" if mode == "noop" else "without synchronizing"
            with pytest.raises(RuntimeError, match=message):
                algorithm.update()
            return
        for _ in range(4):
            algorithm.update()
        if rank == 0:
            report = json.loads((Path(log_dir) / "distributed_validation.json").read_text())
            assert report["first_gradient_mean_verified"]
            assert len(report["updates"]) == 3
            assert all(item["max_parameter_difference"] == 0 for item in report["updates"])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["correct", "disabled", "noop", "no_reduce"])
def test_distributed_audit_detects_missing_synchronization(tmp_path, mode):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), mode), nprocs=2, join=True)
