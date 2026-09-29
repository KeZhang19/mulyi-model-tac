"""Task-local randomized reset, goal sampling and observable grasp phases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms
from ..RotateBulbCustom import env_cfg as original
from ..RotateBulbCustom.tactile import initialize_rotate_bulb_tactile_contract

TASK_ID = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0"
ROOT = Path(__file__).resolve().parents[8]
BANK_PATH = ROOT / "assets/rotate_bulb_randomized/workspace_bank_high_z_v2.json"
PHASE_OBSERVATION_DIM = 14
GOAL_Z_RANGE = (.28, .42)
PALM_ABOVE_PARAMS = {
    "xy_tolerance": .02, "xy_scale": .06,
    "height_min": .10, "height_max": .18, "height_scale": .04,
    "hold_rate": .5, "hold_budget": 2., "turn_budget": 2.,
}


def workspace(env):
    """Load the exact validated discrete distribution once, on the env device."""
    if hasattr(env, "_randomized_bulb_workspace"):
        return env._randomized_bulb_workspace
    path = Path(env.cfg.randomized_workspace_path)
    blob = path.read_bytes()
    bank = json.loads(blob)
    if bank["schema"] != "bulb_randomized_workspace_v1" or not bank["validated"]:
        raise ValueError("Randomized bulb requires a validated workspace bank")
    for relative, digest in bank["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Randomized bulb workspace source changed: {relative}")
    expected = bank.get("scene_contract")
    if expected is not None:
        robot, table = env.cfg.scene.robot, env.cfg.scene.table
        comparisons = [(robot.init_state.pos, expected["robot_base"]["position_m"]),
                       (robot.init_state.rot, expected["robot_base"]["quaternion_wxyz"]),
                       (table.init_state.pos, expected["table"]["position_m"]),
                       (table.init_state.rot, expected["table"]["quaternion_wxyz"]),
                       (table.spawn.size, expected["table"]["size_m"]),
                       (env.cfg.events.reset_object.params["support_pose"], expected["support_pose"])]
        if any(tuple(actual) != tuple(wanted) for actual, wanted in comparisons):
            raise ValueError("Randomized workspace scene poses differ from validated geometry")
        if dict(robot.init_state.joint_pos) != expected["joint_positions"]:
            raise ValueError("Randomized workspace requires fixed arm and zero hand defaults")
    rows = torch.tensor(bank["samples"], device=env.device, dtype=torch.float32)
    if rows.ndim != 2 or rows.shape[1] != 5 or rows.shape[0] < 1 or not torch.isfinite(rows).all():
        raise ValueError("Workspace samples must be finite [dx, dy, gx, gy, gz] rows")
    if (rows[:, :2].abs() > .100001).any() or (rows[:, 2:4].abs() > .150001).any():
        raise ValueError("Workspace planar sample exceeds its distribution contract")
    if ((rows[:, 4] < GOAL_Z_RANGE[0] - 1e-6) | (rows[:, 4] > GOAL_Z_RANGE[1] + 1e-6)).any():
        raise ValueError("Workspace goal height exceeds its distribution contract")
    if tuple(bank.get("goal_z_range_m", GOAL_Z_RANGE)) != GOAL_Z_RANGE:
        raise ValueError("Workspace goal height metadata differs from the task contract")
    env._randomized_bulb_workspace = rows
    env._randomized_bulb_workspace_sha256 = hashlib.sha256(blob).hexdigest()
    env._randomized_bulb_sample_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return rows


def reset_randomized_lamp(env, env_ids, x_range, y_range, table_half_height,
                          support_bottom_offset=original._SUPPORT_BOTTOM_OFFSET, support_pose=None):
    """Move the complete fixture; robot joints are reset separately to fixed defaults."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(
        env_ids, device=env.device, dtype=torch.long)
    rows = workspace(env)
    chosen = torch.randint(rows.shape[0], (len(ids),), device=env.device)
    env._randomized_bulb_sample_ids[ids] = chosen
    pose = torch.as_tensor(support_pose, device=env.device, dtype=torch.float32).expand(len(ids), 7).clone()
    pose[:, :2] += rows[chosen, :2]
    original.reset_lamp_on_table(env, ids, x_range=x_range, y_range=y_range,
                                table_half_height=table_half_height,
                                support_bottom_offset=support_bottom_offset, support_pose=pose)
    state(env).reset(ids)


class RandomizedBulbPoseCommand(original.RotateBulbPoseCommand):
    def _resample_command(self, env_ids):
        rows = workspace(self._env)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        offsets = rows[self._env._randomized_bulb_sample_ids[ids], 2:]
        goal_w = self._env.scene["support"].data.root_pos_w[ids] + offsets
        goal_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[ids], self.robot.data.root_quat_w[ids], goal_w)
        self.pose_command_b[ids, :3] = goal_b
        self.pose_command_b[ids, 3:] = 0
        self.pose_command_b[ids, 3] = 1


class RandomizedBulbState:
    """Single per-step state shared by rewards/observations, including partial reset."""
    def __init__(self, env):
        self.env = env
        robot = env.scene["robot"]
        self.arm_ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 8)]
        self.reference = robot.data.joint_pos[:, self.arm_ids].clone()
        self.grasped = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.ever_grasped = self.grasped.clone()
        self.contact_time = torch.zeros(env.num_envs, device=env.device)
        self.lost_time = self.contact_time.clone()
        self.grasp_spent = self.contact_time.clone()
        self.grasp_credit = self.contact_time.clone()
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, ids):
        self.reference[ids] = self.env.scene["robot"].data.default_joint_pos[ids][:, self.arm_ids]
        self.grasped[ids] = False
        self.ever_grasped[ids] = False
        for value in (self.contact_time, self.lost_time, self.grasp_spent, self.grasp_credit):
            value[ids] = 0
        self.last_step[ids] = self.env.common_step_counter

    def update(self):
        env = self.env
        fresh = self.last_step != env.common_step_counter
        forces = original.rotate_bulb_finger_forces(env)
        opposing = (forces[:, 0] > .5) & (forces[:, 1:].amax(-1) > .5)
        released = getattr(env, "_bulb_released", torch.zeros_like(self.grasped))
        contact_time = torch.where(opposing & ~released, self.contact_time + env.step_dt, 0).clamp_max(.12)
        lost_time = torch.where(~opposing, self.lost_time + env.step_dt, 0).clamp_max(.2)
        enter = ~self.grasped & (contact_time >= .12 - 1e-6) & ~released
        grasped = (self.grasped | enter) & (lost_time < .2 - 1e-6) & ~released
        self.reference.copy_(torch.where((fresh & enter)[:, None],
                                        env.scene["robot"].data.joint_pos[:, self.arm_ids], self.reference))
        self.grasped.copy_(torch.where(fresh, grasped, self.grasped))
        self.ever_grasped |= fresh & enter
        self.contact_time.copy_(torch.where(fresh, contact_time, self.contact_time))
        self.lost_time.copy_(torch.where(fresh, lost_time, self.lost_time))
        quality = torch.minimum(forces[:, 0], forces[:, 1:].amax(-1)).clamp(0, 1)
        credit = torch.minimum(quality * .25 * env.step_dt, (1 - self.grasp_spent).clamp_min(0))
        credit *= grasped & ~released
        self.grasp_credit.copy_(torch.where(fresh, credit, self.grasp_credit))
        self.grasp_spent += torch.where(fresh, credit, 0)
        self.last_step[fresh] = env.common_step_counter


def state(env):
    if not hasattr(env, "_randomized_bulb_state"):
        env._randomized_bulb_state = RandomizedBulbState(env)
    return env._randomized_bulb_state


def randomized_phase_observation(env):
    s = state(env)
    s.update()
    released = getattr(env, "_bulb_released", torch.zeros_like(s.grasped))
    phases = torch.stack((~s.grasped & ~released, s.grasped, released), dim=-1).float()
    offset = env.scene["robot"].data.joint_pos[:, s.arm_ids] - s.reference
    offset = torch.where(s.grasped[:, None], offset, 0)
    return torch.cat((phases, offset, (s.contact_time / .12)[:, None], (s.lost_time / .2)[:, None],
                      s.ever_grasped.float()[:, None], s.grasp_spent[:, None]), dim=-1)


def randomized_action_l2(env):
    s = state(env)
    s.update()
    action = env.action_manager.get_term("action")
    values = env.action_manager.action
    arm = values[:, action.arm_action_indices].square().mean(-1)
    hand = values[:, action.hand_action_indices].square().mean(-1)
    multiplier = torch.where(s.grasped, 5., 1.)
    return .5 * (multiplier * arm + hand)


def randomized_arm_motion(env):
    s = state(env)
    s.update()
    data = env.scene["robot"].data
    offset = data.joint_pos[:, s.arm_ids] - s.reference
    pose = ((offset.abs() - .05).clamp_min(0) / .15).clamp_max(1).square().mean(-1)
    speed = ((data.joint_vel[:, s.arm_ids].abs() - .15).clamp_min(0) / .6).clamp_max(1).square().mean(-1)
    return torch.where(s.grasped & ~env.termination_manager.dones, pose + speed, 0)


def randomized_hand_far(env):
    s = state(env)
    s.update()
    # A new location is initially allowed to be far away. After actual grasp,
    # retain the original abandonment guidance, including goal dwell exemption.
    return original.rotate_bulb_hand_far(env) * s.ever_grasped


def randomized_grasp_reward(env):
    s = state(env)
    s.update()
    env.extras.setdefault("log", {}).update({
        "Metrics/randomized/grasped": s.grasped.float().mean(),
        "Metrics/randomized/ever_grasped": s.ever_grasped.float().mean(),
        "Metrics/randomized/grasp_budget": s.grasp_spent.mean(),
    })
    return torch.where(original.rotate_bulb_failed(env), 0, s.grasp_credit) / env.step_dt


class RandomizedPalmAboveSupport(ManagerTermBase):
    """Finite pre-release alignment credit, plus credit for new screw progress.

    The palm body origin is measured against the randomized support in world Z.
    Transport is exempt. Staying still can spend at most hold_budget; rotating
    back and forth cannot collect the turn budget again.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.palm_id = env.scene["robot"].body_names.index("palm")
        self.paid = torch.zeros(env.num_envs, device=env.device)
        self.best_turns = self.paid.clone()
        self.credit = self.paid.clone()
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = 0
        self.best_turns[ids] = 0
        self.credit[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, xy_tolerance=.02, xy_scale=.06, height_min=.10,
                 height_max=.18, height_scale=.04, hold_rate=.5,
                 hold_budget=2., turn_budget=2.):
        relative = (env.scene["robot"].data.body_pos_w[:, self.palm_id]
                    - env.scene["support"].data.root_pos_w)
        finite = torch.isfinite(relative).all(-1)
        relative = torch.nan_to_num(relative, nan=0., posinf=0., neginf=0.)
        radial = relative[:, :2].norm(dim=-1)
        height = relative[:, 2]
        xy_error = (radial - xy_tolerance).clamp_min(0) / xy_scale
        z_error = ((height_min - height).clamp_min(0)
                   + (height - height_max).clamp_min(0)) / height_scale
        alignment = torch.exp(-xy_error.square() - z_error.square()) * finite
        forces = original.rotate_bulb_finger_forces(env)
        contact = (torch.minimum(forces[:, 0], forces[:, 1:].amax(-1)) / .5).clamp(0, 1)
        quality = alignment * (.2 + .8 * contact)
        released = getattr(env, "_bulb_released", torch.zeros_like(finite))
        active = finite & ~released & ~original.rotate_bulb_failed(env)
        turns = torch.nan_to_num(original.rotate_bulb_unscrewed_turns(env),
                                 nan=0., posinf=0., neginf=0.).clamp(0, 3)
        progress = (turns - self.best_turns).clamp_min(0)
        hold = torch.minimum(quality * hold_rate * env.step_dt, (hold_budget - self.paid).clamp_min(0))
        hold = torch.where(active, hold, 0)
        credit = hold + torch.where(active, alignment * progress * (turn_budget / 3), 0)
        fresh = self.last_step != env.common_step_counter
        self.paid += torch.where(fresh, hold, 0)
        self.best_turns.copy_(torch.where(fresh, torch.maximum(self.best_turns, turns), self.best_turns))
        self.credit.copy_(torch.where(fresh, credit, self.credit))
        self.last_step[fresh] = env.common_step_counter
        count = active.sum().clamp_min(1)
        env.extras.setdefault("log", {}).update({
            "Metrics/randomized/palm_xy_error": (radial * active).sum() / count,
            "Metrics/randomized/palm_height_over_support": (height * active).sum() / count,
            "Metrics/randomized/palm_above_score": (alignment * active).sum() / count,
            "Metrics/randomized/palm_above_hold_spent": self.paid.mean(),
            "Metrics/randomized/palm_above_active_fraction": active.float().mean(),
        })
        return self.credit / env.step_dt


def randomized_contract(env):
    rows = workspace(env)
    return {"version": 2, "workspace_sha256": env._randomized_bulb_workspace_sha256,
            "workspace_count": len(rows), "sampling": "uniform_validated_discrete_pairs",
            "fixed_arm_zero_hand": True, "phase_observation_dim": PHASE_OBSERVATION_DIM,
            "arm_phase_multipliers": [1., 5., 1.], "grasp_hold_s": .12, "regrasp_grace_s": .2,
            "contact_threshold_n": .5, "grasp_reward_budget": 1.,
            "goal_z_range_m": list(GOAL_Z_RANGE),
            "palm_above_support": {
                "body": "palm", "frame": "world_z_relative_to_support",
                "pre_release_only": True,
                "weight": env.cfg.rewards.palm_above_support.weight,
                **env.cfg.rewards.palm_above_support.params,
            }}


def initialize_randomized_contract(env, env_ids=None):
    initialize_rotate_bulb_tactile_contract(env, env_ids)
    env.tactile_policy_contract["task"] = TASK_ID
    env.tactile_policy_contract["randomized_task"] = randomized_contract(env)
