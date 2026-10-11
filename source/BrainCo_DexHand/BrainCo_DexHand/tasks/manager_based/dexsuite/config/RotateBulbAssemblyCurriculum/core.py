"""Simulator-independent curriculum bookkeeping. No policy observations are added."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


STAGES = ("grasp", "lift", "hover", "insert", "full")
EPISODE_SECONDS = (5.0, 10.0, 8.0, 15.0, 20.0)
SUCCESS_THRESHOLDS = (0.90, 0.80, 0.85, 0.70, 1.0)
# (pregrasp path index, XY half-width in m, yaw half-width in rad).
# The three arm poses are collision-screened in pregrasp.json, not interpolated at reset.
LIFT_LEVELS = ((0, 0., 0.), (1, 0., 0.), (2, 0., 0.), (2, .005, 0.),
               (2, .02, 0.), (2, .05, 0.), (2, .05, math.pi/12),
               (2, .05, math.pi/4), (2, .05, math.pi/2))


@dataclass
class CurriculumSettings:
    stage: int = 0
    fixed_bulb: bool = False
    version: int = 2
    physics_hz: int = 120
    # Explicit lower-fidelity first-stage experiments only. Keep the default
    # and later-stage/insertion physics contract conservative.
    fast_stage0_physics: bool = False
    checkpoint: str = ""
    transfer: bool = False
    evaluation: bool = False
    min_episodes: int = 200
    consecutive_windows: int = 2
    previous_fraction: float = 0.2
    bank_capacity: int = 512
    min_bank_size: int = 16
    validation_steps: int = 3
    grasp_hold_s: float = 1.0
    lift_hold_s: float = 0.3
    hover_hold_s: float = 0.5

    def validate(self):
        if type(self.stage) is not int or not 0 <= self.stage <= 4:
            raise ValueError("curriculum stage must be an integer in [0, 4]")
        if self.version != 2 or self.fixed_bulb:
            raise ValueError("Curriculum v2 starts with a free horizontal bulb; use the frozen v1 workspace for old fixtures")
        if type(self.fast_stage0_physics) is not bool:
            raise ValueError("fast_stage0_physics must be boolean")
        if self.fast_stage0_physics and self.stage != 0:
            raise ValueError("fast_stage0_physics is restricted to Stage0; later stages require separate validation")
        allowed_hz = (90, 120, 240) if self.fast_stage0_physics else (120, 240)
        if type(self.physics_hz) is not int or self.physics_hz not in allowed_hz:
            raise ValueError("physics_hz must be 120 or 240; 90 requires explicit fast_stage0_physics, with 30 Hz control")
        for name in ("min_episodes", "consecutive_windows", "bank_capacity", "min_bank_size", "validation_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_bank_size > self.bank_capacity:
            raise ValueError("min_bank_size exceeds bank capacity")
        if not math.isfinite(self.previous_fraction) or not 0 <= self.previous_fraction <= .5:
            raise ValueError("previous_fraction must be between zero and 0.5")
        for name in ("grasp_hold_s", "lift_hold_s", "hover_hold_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")


class StageScheduler:
    """Training windows nominate promotion; the launcher independently evaluates it.

    Windows count complete episodes from the frontier source and its current
    difficulty only. In-flight episodes retain their original difficulty tag.
    """

    def __init__(self, settings: CurriculumSettings):
        settings.validate()
        self.settings = settings
        self.level = 0
        self.version = 0
        self.count = self.successes = self.passed_windows = 0
        self.ready = False
        self.last_rate = 0.0

    def observe(self, source: int, level: int, success: bool):
        s = self.settings
        if s.evaluation or s.stage == 4 or self.ready or source != s.stage or level != self.level:
            return
        self.count += 1
        self.successes += int(success)
        if self.count < s.min_episodes:
            return
        self.last_rate = self.successes / self.count
        self.passed_windows = self.passed_windows + 1 if self.last_rate >= SUCCESS_THRESHOLDS[s.stage] else 0
        self.count = self.successes = 0
        if self.passed_windows >= s.consecutive_windows:
            self.passed_windows = 0
            if s.stage == 1 and self.level < len(LIFT_LEVELS) - 1:
                self.level += 1
                self.version += 1
            else:
                self.ready = True

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("level", "version", "count", "successes", "passed_windows", "ready", "last_rate")}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or not 0 <= state["level"] < len(LIFT_LEVELS):
            raise ValueError("Invalid curriculum scheduler state")
        for key, value in state.items():
            setattr(self, key, value)


def soft_grasp_quality(forces: torch.Tensor) -> torch.Tensor:
    """Partial single-finger credit; force saturation prevents squeezing for reward."""
    force = torch.nan_to_num(forces, nan=0., posinf=0., neginf=0.).clamp(0., 1.)
    thumb, other = force[:, 0], force[:, 1:].amax(-1)
    return .3 * torch.maximum(thumb, other) + .7 * torch.sqrt(thumb * other)


class HoldCounter:
    def __init__(self, count: int, device):
        self.steps = torch.zeros(count, dtype=torch.long, device=device)

    def update(self, condition):
        self.steps.copy_(torch.where(condition, self.steps + 1, 0))

    def reached(self, seconds: float, dt: float, *, strict: bool = False):
        required = math.floor(seconds / dt + 1e-8) + 1 if strict else math.ceil(seconds / dt - 1e-8)
        return self.steps >= required

    def reset(self, ids):
        self.steps[ids] = 0


class StateBank:
    """Bounded, one sample per successful episode, tensor snapshots on CPU.

    A bank is a reset distribution, never an off-policy PPO transition store.
    Contract equality is checked before loading or sampling any state.
    """

    def __init__(self, capacity: int, contract: dict):
        self.capacity, self.contract = capacity, contract
        self.samples: dict[int, list[dict]] = {2: [], 3: []}
        self.seen = {2: 0, 3: 0}

    def add(self, source: int, tensors: dict, provenance: dict):
        if source not in self.samples:
            raise ValueError("Only successful lift/hover prefixes may populate the bank")
        if provenance.get("source_stage") != source - 1 or not provenance.get("physical_success"):
            raise ValueError("Bank samples require a real preceding-stage success")
        if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Bank state must contain finite tensors")
        item = {"state": {k: v.detach().cpu().clone() for k, v in tensors.items()},
                "provenance": dict(provenance)}
        index = self.seen[source] % self.capacity
        if len(self.samples[source]) < self.capacity:
            self.samples[source].append(item)
        else:
            self.samples[source][index] = item
        self.seen[source] += 1

    def sample(self, source: int, generator: torch.Generator):
        items = self.samples[source]
        if not items:
            raise ValueError(f"Stage{source} has no successful predecessor states")
        index = int(torch.randint(len(items), (), generator=generator))
        return index, items[index]

    def state_dict(self):
        return {"schema": 1, "contract": self.contract, "samples": self.samples, "seen": self.seen}

    def load_state_dict(self, state):
        if state.get("schema") != 1 or state.get("contract") != self.contract:
            raise ValueError("Curriculum bank physics/assets/action/observation contract mismatch")
        for source in (2, 3):
            items = state["samples"][source]
            if len(items) > self.capacity:
                raise ValueError("Restored bank exceeds configured capacity")
            for item in items:
                p = item["provenance"]
                if p.get("source_stage") != source - 1 or not p.get("physical_success"):
                    raise ValueError("Unproven bank prefix")
                if any(not torch.isfinite(t).all() for t in item["state"].values()):
                    raise ValueError("Non-finite bank state")
        self.samples = state["samples"]
        self.seen = state["seen"]


def settings_dict(settings):
    return asdict(settings)
