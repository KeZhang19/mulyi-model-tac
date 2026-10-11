# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the module learns input values until the sum of batch sizes
            exceeds it.

        Note: The normalization parameters are computed over the whole batch, not for each environment separately.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values."""

        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        # Keep the registered buffer object instead of replacing it. Rollouts run under
        # torch.inference_mode(), where a newly assigned tensor becomes an inference tensor
        # and cannot be updated later by distributed normalizer synchronization.
        self._std.copy_(torch.sqrt(self._var))

    @torch.jit.unused
    def inverse(self, y):
        """De-normalize values based on empirical values."""

        return y * (self._std + self.eps) + self._mean


@torch.no_grad()
def synchronize_empirical_normalization(norm: EmpiricalNormalization) -> None:
    """Synchronize running moments without multiplying shared history by the world size.

    At the start of a distributed rollout, every rank has the same running statistics. Each rank
    then adds its local rollout. Averaging the cumulative sufficient statistics keeps the shared
    history exactly once while still merging the new samples from every rank. Summing cumulative
    counts would multiply the same history on every update and eventually overflow ``int64``.
    """

    if norm.count.item() < 0:
        raise RuntimeError(
            "EmpiricalNormalization count is negative. The checkpoint is already corrupted by "
            "distributed running-count overflow and must not be resumed."
        )

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return

    # Float64 keeps integer counts exact for all practical training horizons and reduces
    # cancellation when reconstructing variance from the second raw moment.
    count = norm.count.to(dtype=torch.float64)
    mean = norm._mean.to(dtype=torch.float64)
    var = norm._var.to(dtype=torch.float64)

    average_count = count.clone()
    average_sum = mean * count
    average_second_sum = (var + mean.square()) * count

    torch.distributed.all_reduce(average_count, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(average_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(average_second_sum, op=torch.distributed.ReduceOp.SUM)

    inverse_world_size = 1.0 / world_size
    average_count.mul_(inverse_world_size)
    average_sum.mul_(inverse_world_size)
    average_second_sum.mul_(inverse_world_size)

    if not torch.isfinite(average_count) or average_count.item() <= 0.0:
        return

    mean_global = average_sum / average_count
    second_global = average_second_sum / average_count
    var_global = torch.clamp(second_global - mean_global.square(), min=0.0)

    norm._mean.copy_(mean_global.to(dtype=norm._mean.dtype))
    norm._var.copy_(var_global.to(dtype=norm._var.dtype))
    norm._std.copy_(torch.sqrt(var_global).to(dtype=norm._std.dtype))
    norm.count.copy_(average_count.round().to(dtype=torch.long))


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounted rewards
            avg = self.disc_avg.update(rew)
            # update moments from discounted rewards
            self.emp_norm.update(avg)

        # normalize rewards with the empirical std
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


"""
Helper class.
"""


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
