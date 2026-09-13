"""Opt-in checks that distributed PPO really shares gradients and parameters."""

import json
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist


def install_distributed_training_audit(runner, *, expected_world_size: int):
    """Fail on disabled synchronization; audit the first three PPO updates."""
    if not dist.is_initialized() or dist.get_world_size() != expected_world_size:
        raise RuntimeError(f"Expected an initialized process group with {expected_world_size} workers")
    algorithm = runner.alg
    rank = dist.get_rank()
    device = torch.device(runner.device)
    status = {
        "rank": rank, "device": str(device), "backend": dist.get_backend(),
        "runner_distributed": bool(runner.is_distributed),
        "algorithm_multi_gpu": bool(algorithm.is_multi_gpu),
        "world_size": algorithm.gpu_world_size,
        "parameter_sync_enabled": bool(getattr(algorithm, "allow_distributed_parameter_sync", True)),
        "per_task_obs_normalizer": bool(getattr(algorithm, "per_task_obs_normalizer", False)),
    }
    workers = [None] * expected_world_size
    dist.all_gather_object(workers, status)
    for worker in workers:
        if (not worker["runner_distributed"] or not worker["algorithm_multi_gpu"]
                or worker["world_size"] != expected_world_size or not worker["parameter_sync_enabled"]
                or worker["per_task_obs_normalizer"]):
            raise RuntimeError(f"Distributed PPO synchronization is not enabled: {workers}")
    # GPU runs must use NCCL with one local device per worker. CPU/Gloo remains
    # supported so the synchronization checks can be exercised without Isaac Sim.
    if device.type == "cuda":
        if dist.get_backend() != "nccl" or len({w["device"] for w in workers}) != expected_world_size:
            raise RuntimeError(f"Expected distinct GPUs communicating through NCCL: {workers}")

    original_reduce = algorithm.reduce_parameters
    original_update = algorithm.update
    report = {"world_size": expected_world_size, "workers": workers, "updates": []}
    reduction_count = 0
    update_count = 0

    def parameters():
        result = list(algorithm.policy.parameters())
        if getattr(algorithm, "rnd", None):
            result.extend(algorithm.rnd.parameters())
        return result

    def reduce_checked(algorithm):
        nonlocal reduction_count
        if (not getattr(algorithm, "allow_distributed_parameter_sync", True)
                or getattr(algorithm, "_distributed_state_compatible", True) is False):
            raise RuntimeError("Distributed gradient synchronization was disabled")
        expected = None
        # Independently calculate the mean once, before the production reducer.
        if reduction_count == 0:
            grads = [p.grad.detach().flatten() for p in parameters() if p.grad is not None]
            if not grads:
                raise RuntimeError("No PPO gradients to synchronize")
            expected = torch.cat(grads).clone()
            dist.all_reduce(expected)
            expected /= expected_world_size
        result = original_reduce()
        if expected is not None:
            actual = torch.cat([p.grad.detach().flatten() for p in parameters() if p.grad is not None])
            mismatch = (~torch.isclose(actual, expected, rtol=0, atol=0)).any().to(torch.int32)
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
            if mismatch.item():
                raise RuntimeError("PPO gradients do not equal the mean across workers")
            report["first_gradient_mean_verified"] = True
        reduction_count += 1
        return result

    def update_checked(algorithm, *args, **kwargs):
        nonlocal update_count
        before = reduction_count
        result = original_update(*args, **kwargs)
        if reduction_count == before:
            raise RuntimeError("PPO update completed without synchronizing gradients")
        update_count += 1
        if update_count <= 3:
            with torch.no_grad():
                actual = torch.cat([p.detach().flatten() for p in parameters()])
                reference = actual.clone()
                dist.broadcast(reference, src=0)
                delta = (actual - reference).abs().max()
                dist.all_reduce(delta, op=dist.ReduceOp.MAX)
                if not torch.isfinite(delta).item() or delta.item() != 0:
                    raise RuntimeError(f"Parameters differ across workers after PPO update: {delta.item()}")
            report["updates"].append({
                "update": update_count, "gradient_reductions": reduction_count - before,
                "max_parameter_difference": delta.item(),
            })
            if rank == 0:
                path = Path(runner.log_dir) / "distributed_validation.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(path)
                print(f"[DISTRIBUTED VERIFIED] update={update_count} world_size={expected_world_size} "
                      f"gradient_reductions={reduction_count - before} max_parameter_difference=0", flush=True)
        return result

    algorithm.reduce_parameters = MethodType(reduce_checked, algorithm)
    algorithm.update = MethodType(update_checked, algorithm)
    if rank == 0:
        print(f"[INFO] Distributed PPO audit enabled: {workers}", flush=True)
