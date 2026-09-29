"""Bounded pre-release costs; pure Torch so recorded trajectories can be audited."""
from dataclasses import dataclass, asdict
import math

import torch


@dataclass(frozen=True)
class ProgressPenaltySettings:
    version: int = 1
    regression_per_turn: float = 2.0
    regression_deadband_turns: float = 0.01
    regression_budget: float = 2.0
    approach_grace_s: float = 3.0
    progress_epsilon_turns: float = 0.02
    stall_grace_s: float = 2.0
    stall_ramp_s: float = 1.0
    stall_rate: float = 0.3
    stall_budget: float = 4.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "version":
                if value != 1:
                    raise ValueError("Unsupported progress penalty version")
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class ProgressPenaltyState:
    """Charge new worst drawdown and time without meaningful new-best progress.

    Recovery never refunds a cost; repeating the same drawdown does not charge
    again. The stall clock restarts only after advancing the episode frontier,
    so reversing/recovering and tiny jitter cannot keep buying grace periods.
    All amounts returned are per-control-step costs, before RewardManager dt.
    """

    def __init__(self, num_envs, device, settings=None):
        self.settings = settings or ProgressPenaltySettings()
        self.best = torch.zeros(num_envs, device=device)
        self.worst_drawdown = torch.zeros_like(self.best)
        self.advance_mark = torch.zeros_like(self.best)
        self.last_advance_s = torch.zeros_like(self.best)
        self.regression_paid = torch.zeros_like(self.best)
        self.stall_paid = torch.zeros_like(self.best)
        self.regression_credit = torch.zeros_like(self.best)
        self.stall_credit = torch.zeros_like(self.best)
        self.last_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("best", "worst_drawdown", "advance_mark", "last_advance_s",
                     "regression_paid", "stall_paid", "regression_credit", "stall_credit"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def update(self, turns, released, failed, elapsed_s, step, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        cfg = self.settings
        valid = torch.isfinite(turns) & torch.isfinite(elapsed_s)
        active = valid & ~released & ~failed
        fresh = self.last_step != step
        update = active & fresh
        turns = torch.nan_to_num(turns, nan=0., posinf=0., neginf=0.).clamp(0, 3)
        elapsed_s = torch.nan_to_num(elapsed_s, nan=0., posinf=0., neginf=0.)
        best = torch.maximum(self.best, turns)
        drawdown = (best - turns - cfg.regression_deadband_turns).clamp_min(0)
        regression = (drawdown - self.worst_drawdown).clamp_min(0) * cfg.regression_per_turn
        regression = torch.minimum(regression, (cfg.regression_budget - self.regression_paid).clamp_min(0))
        advanced = best - self.advance_mark >= cfg.progress_epsilon_turns - 1e-6
        last_advance = torch.where(advanced, elapsed_s, self.last_advance_s)
        wait = elapsed_s - last_advance.clamp_min(cfg.approach_grace_s)
        ramp = ((wait - cfg.stall_grace_s) / cfg.stall_ramp_s).clamp(0, 1)
        stall = torch.minimum(ramp * cfg.stall_rate * dt, (cfg.stall_budget - self.stall_paid).clamp_min(0))
        self.regression_credit.copy_(torch.where(fresh, torch.where(active, regression, 0.), self.regression_credit))
        self.stall_credit.copy_(torch.where(fresh, torch.where(active, stall, 0.), self.stall_credit))
        self.regression_paid += torch.where(update, regression, 0.)
        self.stall_paid += torch.where(update, stall, 0.)
        self.best.copy_(torch.where(update, best, self.best))
        self.worst_drawdown.copy_(torch.where(update, torch.maximum(self.worst_drawdown, drawdown), self.worst_drawdown))
        self.advance_mark.copy_(torch.where(update & advanced, best, self.advance_mark))
        self.last_advance_s.copy_(torch.where(update, last_advance, self.last_advance_s))
        self.last_step[fresh] = step
        return (torch.where(active, self.regression_credit, 0.),
                torch.where(active, self.stall_credit, 0.))
