"""Compatibility fixes for RSL-RL forks that synchronize running statistics."""

from types import MethodType

import torch
import torch.distributed as dist


@torch.no_grad()
def synchronize_normalizer(normalizer):
    """Merge moments while keeping the count in equivalent samples per rank.

    Each rank already contains the common history from the previous merge.
    Storing the SUM of their cumulative counts duplicates that history at
    every update and eventually overflows int64. The average count represents
    each rank's share of the merged history on the next local update.
    """
    count = normalizer.count.to(torch.float64)
    total_count = count.clone()
    dist.all_reduce(total_count)
    if total_count.item() <= 0:
        raise ValueError("Distributed observation normalizer has a non-positive sample count")

    mean = normalizer._mean.to(torch.float64)
    merged_mean = mean * count
    dist.all_reduce(merged_mean)
    merged_mean /= total_count

    # Center on the merged mean instead of subtracting two large raw moments.
    merged_var = (normalizer._var.to(torch.float64) + (mean - merged_mean).square()) * count
    dist.all_reduce(merged_var)
    merged_var = (merged_var / total_count).clamp_min_(0)
    normalizer._mean.copy_(merged_mean)
    normalizer._var.copy_(merged_var)
    # RSL-RL replaces _std during rollouts inside inference_mode. Allocate its
    # replacement here; that inference tensor cannot be updated in place later.
    normalizer._std = merged_var.sqrt().to(normalizer._std)
    normalizer.count.copy_((total_count / dist.get_world_size()).round().to(torch.long))


def install_distributed_normalization_fix(runner):
    """Replace only the affected fork's per-update synchronization hook."""
    algorithm = runner.alg
    original = getattr(algorithm, "_sync_empirical_normalization_buffers", None)
    if (original is None or not getattr(algorithm, "is_multi_gpu", False)
            or getattr(algorithm, "per_task_obs_normalizer", False)):
        return False
    normalizer_type = original.__func__.__globals__["EmpiricalNormalization"]

    def synchronize(algorithm):
        modules = list(algorithm.policy.modules())
        if algorithm.rnd:
            modules.extend(algorithm.rnd.modules())
        for module in modules:
            if isinstance(module, normalizer_type):
                synchronize_normalizer(module)

    algorithm._sync_empirical_normalization_buffers = MethodType(synchronize, algorithm)
    print("[INFO] Distributed observation normalization: stable moments and linear sample counts.", flush=True)
    return True


def install_finite_action_guard(env):
    """Reject invalid policy outputs before they enter the GPU physics scene."""
    original_step = env.step

    def step(actions):
        if not torch.isfinite(actions).all().item():
            raise FloatingPointError("Policy produced non-finite actions; stopped before stepping PhysX")
        return original_step(actions)

    env.step = step
