"""Curriculum-only approach shaping and low-overhead reward reporting."""
import math

import torch
from isaaclab.managers import ManagerTermBase

from ..RotateBulbCustom import assembly
from ..RotateBulbCustom.assembly_env_cfg import assembly_hand_shell_distance
from ..RotateBulbCustom.assembly_reward_adapter import AssemblyRewardAdapter


def grasp_distance(env):
    """Thumb/opposing fingertip and palm distance to the actual shell center."""
    robot, bulb = env.scene["robot"], env.scene["object"]
    if not hasattr(env, "_floor_tip_ids"):
        env._floor_tip_ids = [robot.body_names.index(f"right_{finger}_tip_Link") for finger in
                             ("thumb", "index", "middle", "ring", "little")]
        env._floor_palm_id = robot.body_names.index("palm")
    offset = torch.zeros_like(bulb.data.root_pos_w)
    offset[:, 0] = .04
    shell = bulb.data.root_pos_w + assembly.quat_apply(bulb.data.root_quat_w, offset)
    distance = (robot.data.body_pos_w[:, env._floor_tip_ids] - shell[:, None]).norm(dim=-1)
    pair = .5 * (distance[:, 0] + distance[:, 1:].amin(-1))
    palm = (robot.data.body_pos_w[:, env._floor_palm_id] - shell).norm(dim=-1)
    return .8 * pair + .2 * (palm - .06).clamp_min(0)


class FloorBestReachReward(ManagerTermBase):
    """Dextac historical-minimum distance credit, normalized per control step.

    Adapted from dextac-Depth's object_ee_distance_delta: use the farthest of
    the five fingertips, targeting the bulb shell instead of its screw root.
    The first valid reward sample seeds each episode, so reset geometry earns
    no credit. Retreat/recovery cannot repay an old distance record.

    Return a rate for RewardManager's dt multiplication. With std=.4 and
    weight=40, a new closest distance 1 cm nearer earns exactly one point.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.closest = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.valid[ids] = False

    def __call__(self, env, std: float = .4):
        if not math.isfinite(std) or std <= 0:
            raise ValueError("Historical reach std must be positive and finite")
        distance = assembly_hand_shell_distance(env)
        finite = torch.isfinite(distance)
        baseline = torch.where(self.valid, self.closest, distance)
        progress = torch.where(finite & self.valid, (baseline - distance).clamp_min(0.), 0.)
        self.closest.copy_(torch.where(finite, torch.minimum(baseline, distance), self.closest))
        self.valid.copy_(finite)
        active = ~env._bulb_assembly_state.lifted
        for name in ("non_finite", "object_out_of_bound", "bulb_dropped", "lost_grasp", "curriculum_reset_failed"):
            if name in env.termination_manager.active_terms:
                active &= ~env.termination_manager.get_term(name)
        return progress / (std * env.step_dt) * active


class FloorReachReward(ManagerTermBase):
    """Signed distance progress plus small continuous reach credit.

    Retreat loses the corresponding progress; an out/back cycle cannot earn
    repeated progress. This is task shaping, not a policy-invariance claim.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.valid[ids] = False

    def __call__(self, env):
        distance = torch.nan_to_num(grasp_distance(env), nan=1., posinf=1.).clamp(0., 1.)
        quality = 1 - torch.tanh(distance / .1)
        delta = torch.where(self.valid, self.previous - distance, 0.)
        self.previous.copy_(distance)
        self.valid[:] = True
        state = env._bulb_assembly_state
        # Stop approach shaping only after physical lift. Table contact alone
        # must not remove the incentive to stay at a useful grasp pose.
        active = ~state.lifted
        return (1.0 * quality + 4. * delta / env.step_dt) * active


def arm_path_cost(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_floor_arm_ids"):
        env._floor_arm_ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 8)]
    speed = robot.data.joint_vel[:, env._floor_arm_ids].abs()
    # Integral is actual joint travel; it does not favor a constant action sign.
    return speed[:, 0] + .1 * speed[:, 1:].mean(-1)


def base_excursion_cost(env):
    robot = env.scene["robot"]
    index = robot.joint_names.index("joint1")
    reference = robot.data.default_joint_pos[:, index]
    delta = (robot.data.joint_pos[:, index] - reference).abs()
    # Finite joint coordinates deliberately NOT wrapped modulo 2*pi. A large
    # sweep must retain its cost; this changes no joint limits or collisions.
    return ((delta - math.pi / 3).clamp_min(0.) / (math.pi / 3)).square()


class CurriculumRewardAdapter(AssemblyRewardAdapter):
    def _log_finished(self, ids):
        # Full reward diagnostics are available during frozen evaluation.
        # Training keeps all episode successes/failures in runtime JSONL and
        # tensor aggregate reward/reason metrics without a duplicate JSONL.
        runtime = getattr(self.env, "_bulb_curriculum", None)
        if runtime is not None and runtime.cfg.evaluation:
            return super()._log_finished(ids)
        ledger = self.ledger
        metrics = self.env.extras.setdefault("log", {})
        for name, values in {**ledger.episode_sums, **self.action_sums}.items():
            if name == "reach_progress":
                continue  # Replaced term: actual credit is logged by RewardManager.
            metrics["reward/" + name] = values[ids].mean().clone()
        for stage, reasons in enumerate(ledger.reason_steps):
            for name, values in reasons.items():
                metrics[f"stage{stage}_fail_reason/{name}"] = (
                    values[ids].float() / ledger.elapsed[ids].clamp_min(1)).mean()
