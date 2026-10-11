"""Free-body screw progress and stage-three reward adapters.

Adapt the rotation task's force-gated new-best turns / quarter-turn milestones.
Unlike its guided joint, this task must measure both real pose and axial travel.
"""
import math

import torch
from isaaclab.utils.math import quat_conjugate, quat_mul

from .mdp import rewards as teacher_rewards


def socket_twist_phase(obj, socket):
    relative = quat_mul(quat_conjugate(socket.data.root_quat_w), obj.data.root_quat_w)
    return 2 * torch.atan2(relative[:, 1], relative[:, 0])


def screw_contact_quality(env, force_scale):
    # Bulb-filtered forces only; a fingertip can drive rotation while the socket
    # supports the bulb. Large squeezing force does not earn additional credit.
    forces = [env.scene.sensors[f"right_{name}_DIP_Link_object_s"].data.force_matrix_w
              .reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=-1)
              for name in ("thumb", "index", "middle", "ring", "little")]
    return torch.nan_to_num(torch.stack(forces, dim=-1).amax(-1) / force_scale,
                            nan=0., posinf=0., neginf=0.).clamp(0, 1)


class ScrewInsertionState:
    """One update per control step; all cached rewards are amounts before dt."""

    def __init__(self, num_envs, device, scales, cfg):
        self.cfg, self.scales = cfg, scales
        self.names = ("entry_depth", "entry_phase", "previous_phase", "net_turns", "turns",
                      "best", "worst_drawdown", "contact_memory", "force_credit",
                      "progress_credit", "regression_cost", "helix_error")
        for name in self.names:
            setattr(self, name, torch.zeros(num_envs, device=device))
        self.previous_engaged = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.valid = torch.zeros_like(self.previous_engaged)

    def reset(self, ids):
        for name in self.names:
            getattr(self, name)[ids] = 0
        self.previous_engaged[ids] = False
        self.valid[ids] = False

    def begin(self, ids, phase, depth):
        self.reset(ids)
        self.entry_depth[ids] = depth[ids]
        self.entry_phase[ids] = phase[ids]
        self.previous_phase[ids] = phase[ids]
        self.previous_engaged[ids] = True

    def update(self, active, phase, depth, radial, tilt, quality, dt):
        cfg = self.cfg
        pitch = cfg.thread_pitch_m * self.scales
        # Negative twist about outward +X is clockwise as seen from the mouth.
        phase_delta = phase - self.previous_phase
        delta = -torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)) / (2 * math.pi)
        finite = torch.isfinite(phase) & torch.isfinite(depth) & torch.isfinite(radial) & torch.isfinite(tilt)
        engaged = (active & finite & (radial <= cfg.screw_engagement_radial_m)
                   & (tilt <= math.radians(cfg.screw_engagement_tilt_deg))
                   & (depth >= self.entry_depth - cfg.screw_helix_tolerance_m * self.scales)
                   & (depth <= self.entry_depth + cfg.screw_target_turns * pitch + .003 * self.scales))
        sane_step = delta.abs() <= cfg.screw_max_measured_turn_rate * dt
        measurable = engaged & self.previous_engaged & sane_step
        proposed = self.net_turns + torch.where(measurable, delta, 0.)
        error = (depth - self.entry_depth - proposed * pitch).abs()
        # Never bank spinning at the lip/in the air and redeem it with a plunge.
        accept = measurable & ((delta <= 0) | (error <= cfg.screw_helix_tolerance_m * self.scales))
        net = self.net_turns + torch.where(accept, delta, 0.)
        self.net_turns.copy_(torch.where(engaged & sane_step, net, 0.))
        self.helix_error.copy_((depth - self.entry_depth - self.net_turns * pitch).abs())
        self.valid.copy_(engaged & sane_step & (self.helix_error <= cfg.screw_helix_tolerance_m * self.scales))
        axial_turns = ((depth - self.entry_depth) / pitch).clamp_min(0)
        turns = torch.minimum(self.net_turns.clamp_min(0), axial_turns).clamp(max=cfg.screw_target_turns)
        self.turns.copy_(torch.where(self.valid, turns, 0.))
        new_best = torch.maximum(self.best, self.turns)
        delta_best = new_best - self.best
        rate_fraction = (cfg.screw_reward_max_turn_rate * dt / delta_best.clamp_min(1e-8)).clamp_max(1)
        memory = (self.contact_memory - dt / cfg.screw_regrasp_grace_s).clamp_min(0)
        self.contact_memory.copy_(torch.where(self.valid, torch.maximum(quality, memory), 0.))
        self.force_credit.copy_(delta_best * rate_fraction * self.contact_memory)
        segment = cfg.screw_milestone_turns
        crossed = (torch.floor((new_best + 1e-6) / segment) - torch.floor((self.best + 1e-6) / segment)).clamp_min(0)
        self.progress_credit.copy_(crossed * segment * rate_fraction * self.contact_memory)
        drawdown = (new_best - self.turns - cfg.screw_regression_deadband_turns).clamp_min(0)
        self.regression_cost.copy_(torch.where(active, (drawdown - self.worst_drawdown).clamp_min(0), 0.))
        self.worst_drawdown.copy_(torch.maximum(self.worst_drawdown, drawdown))
        # Consume all physical progress, including unheld progress and capped jumps.
        self.best.copy_(new_best)
        self.previous_phase.copy_(torch.nan_to_num(phase))
        self.previous_engaged.copy_(engaged & sane_step)


def screw_in_reward(env, component: str):
    state = env.command_manager.get_term("object_pose").screw
    values = {"force": state.force_credit, "progress": state.progress_credit,
              "regression": state.regression_cost}
    return values[component] / env.step_dt


def final_screw_success_bonus(env):
    """One event per episode; weight=20 pays exactly 20, independent of dt.

    A simultaneous physical failure (not a time limit) suppresses the bonus.
    """
    command = env.command_manager.get_term("object_pose")
    credit = command.final_bonus_credit
    if getattr(env, "termination_manager", None) is not None:
        credit = torch.where(env.termination_manager.terminated, 0., credit)
    return credit / env.step_dt


def carry_velocity_before_screwing(env, object_cfg, palm_cfg, table_asset_cfg,
                                   component: str, contact_threshold: float = 1.,
                                   lift_clearance_m: float = .09,
                                   relative_velocity_clamp_max: float = 1000.):
    funcs = {"linear": teacher_rewards.flexiv_carry_relative_linear_velocity_l2,
             "angular": teacher_rewards.flexiv_carry_relative_angular_velocity_l2}
    value = funcs[component](env, object_cfg, palm_cfg, table_asset_cfg, contact_threshold,
                            lift_clearance_m, relative_velocity_clamp_max)
    return torch.where(env.command_manager.get_term("object_pose").stage_three, 0., value)
