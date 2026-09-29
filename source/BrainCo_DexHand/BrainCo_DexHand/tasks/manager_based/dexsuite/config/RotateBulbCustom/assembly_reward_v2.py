"""Finite assembly reward ledger. Pure tensor code; never writes physical state.

All outputs are *per-step credits*, before the Isaac adapter divides by dt.
Physical success/termination remains owned by assembly.py / curriculum runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


REWARD_VERSION = 2
COMPONENTS = (
    "reach_progress", "contact_stability", "relative_pose_stability", "grasp_stability",
    "contact_persistence", "grasp_hold", "lift_progress", "table_unloading",
    "lift_stability", "lift_hold", "transport_progress", "transport_alignment",
    "transport_stability", "transport_hold", "descent_progress", "entry_alignment",
    "entry_hold", "screw_progress", "installation_stability", "installation_hold",
    "bridge_maintenance", "stage_completion", "final_success",
    "slip_penalty", "velocity_penalty", "time_penalty", "failure_penalty",
)
POTENTIALS = (
    "reach_progress", "contact_stability", "relative_pose_stability", "grasp_stability",
    "lift_progress", "table_unloading", "lift_stability", "transport_progress",
    "transport_alignment", "transport_stability", "descent_progress",
    "entry_alignment", "screw_progress", "installation_stability",
)
HOLD_COMPONENTS = (
    "contact_persistence", "grasp_hold", "lift_hold", "transport_hold",
    "entry_hold", "installation_hold",
)
FAIL_REASONS = (
    ("no_contact", "slip", "pose_error", "velocity", "hold_incomplete", "invalid"),
    ("no_contact", "height", "table_contact", "velocity", "hold_incomplete", "invalid"),
    ("no_contact", "not_lifted", "table_contact", "position", "radial", "angle",
     "velocity", "hold_incomplete", "invalid"),
    ("no_contact", "not_engaged", "depth", "radial", "angle", "helical",
     "velocity", "hold_incomplete", "invalid"),
)


@dataclass
class RewardSample:
    forces: torch.Tensor
    relative_pose: torch.Tensor
    clearance: torch.Tensor
    table_force: torch.Tensor
    linear_speed: torch.Tensor
    angular_speed: torch.Tensor
    pair_distance: torch.Tensor
    depth: torch.Tensor
    radial: torch.Tensor
    angle: torch.Tensor
    helical: torch.Tensor
    relative_angle: torch.Tensor
    advance: torch.Tensor
    lifted: torch.Tensor
    tracking: torch.Tensor
    engaged: torch.Tensor
    installed: torch.Tensor
    socket_contact: torch.Tensor
    final_success: torch.Tensor
    final_hold: torch.Tensor
    axis_in_palm: torch.Tensor
    valid: torch.Tensor
    failed: torch.Tensor
    source: torch.Tensor
    # Read-only authoritative curriculum timers/anchor, when available.
    curriculum_steps: torch.Tensor | None = None
    curriculum_anchor: torch.Tensor | None = None


def quat_angle(a, b):
    return 2 * torch.acos((a * b).sum(-1).abs().clamp(0, 1))


def relative_twist(previous, current, dt):
    """Shortest quaternion increment, expressed in the palm frame."""
    a, b = current[:, 3:], previous[:, 3:].clone()
    b[:, 1:] *= -1
    w = a[:, :1] * b[:, :1] - (a[:, 1:] * b[:, 1:]).sum(-1, keepdim=True)
    xyz = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + torch.cross(a[:, 1:], b[:, 1:], dim=-1)
    xyz = torch.where(w < 0, -xyz, xyz)
    length = xyz.norm(dim=-1, keepdim=True)
    rot = xyz * (2 * torch.atan2(length, w.abs()) / length.clamp_min(1e-8)) / dt
    return (current[:, :3] - previous[:, :3]) / dt, rot


def reward_budgets(stage):
    """Standalone target weights; earlier targets use the same Full prefix."""
    p = dict(zip(POTENTIALS, (.25, .25, .25, .25, 1.5, .5, .5, 1.25, .75, .5, 1., .5, 4., .5)))
    h = [.5, .5, .5, .5, .5, 1.]
    bonus = [2., 3., 4., 2., 3.]
    if stage == 0:
        p.update(dict.fromkeys(POTENTIALS[:4], .5))
        bonus[0] = 8.
    elif stage == 1:
        p.update(lift_progress=3., table_unloading=1., lift_stability=1.)
        h[2], bonus[1] = 1., 24.
    elif stage == 2:
        p.update(transport_progress=2.5, transport_alignment=1.5, transport_stability=1.)
        h[3], bonus[2] = 1., 40.
    if stage < 4:
        keep = (4, 7, 10, 14)[stage]
        for name in POTENTIALS[keep:]:
            p[name] = 0.
        for i in range((2, 3, 4, 6)[stage], 6):
            h[i] = 0.
        for i in range((1, 2, 3, 5)[stage], 5):
            bonus[i] = 0.
    # Stage3 starts from a proven lift; never invent new grasp/lift credit.
    if stage == 3:
        for name in POTENTIALS[:7]:
            p[name] = 0.
        h[:3], bonus[:2] = [0.] * 3, [0.] * 2
    return p, h, bonus


class AssemblyRewardLedger:
    def __init__(self, count, device, dt, stage=4, hold_seconds=(1., .3, .5)):
        if not math.isfinite(dt) or dt <= 0 or stage not in range(5):
            raise ValueError("Invalid assembly reward timing/stage")
        self.count, self.device, self.dt, self.stage = count, device, dt, stage
        self.window = max(1, math.ceil(.1 / dt - 1e-8))
        self.required = [
            math.floor(hold_seconds[0] / dt + 1e-8) + 1,
            math.ceil(hold_seconds[1] / dt - 1e-8),
            math.ceil(hold_seconds[2] / dt - 1e-8),
            math.ceil(.2 / dt - 1e-8),
            math.ceil(1. / dt - 1e-8),
        ]
        if min(self.required) <= self.window:
            raise ValueError("Hold durations must exceed the reward's 0.1s dead zone")
        self.weights, self.hold_weights, self.bonuses = reward_budgets(stage)
        self.best = torch.zeros(count, len(POTENTIALS), device=device)
        self.hold_best = torch.zeros(count, 6, device=device)
        self.steps = torch.zeros(count, 6, dtype=torch.long, device=device)
        self.paid = torch.zeros(count, 6, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)
        self.initial_clearance = torch.zeros(count, device=device)
        self.initial_table_force = torch.full((count,), float("nan"), device=device)
        self.anchor = torch.zeros(count, 7, device=device)
        self.previous = torch.zeros_like(self.anchor)
        self.previous[:, 3] = 1
        self.previous_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.force_history = torch.zeros(self.window, count, device=device)
        self.quality_history = torch.zeros(self.window, count, 6, device=device)
        self.cursor = 0
        self.grasp_grace = torch.zeros(count, dtype=torch.long, device=device)
        self.hover_age = torch.full((count,), -1., device=device)
        self.transport_start = torch.zeros(count, device=device)
        self.keep_paid = torch.zeros(count, device=device)
        self.failure_paid = torch.zeros(count, dtype=torch.bool, device=device)
        self.credits = {name: torch.zeros(count, device=device) for name in COMPONENTS}
        self.episode_sums = {name: torch.zeros(count, device=device) for name in COMPONENTS}
        self.reason_masks = [{r: torch.zeros(count, dtype=torch.bool, device=device) for r in group} for group in FAIL_REASONS]
        self.reason_steps = [{r: torch.zeros(count, dtype=torch.long, device=device) for r in group} for group in FAIL_REASONS]
        self.interruptions = [{r: torch.zeros(count, dtype=torch.long, device=device) for r in group} for group in FAIL_REASONS]
        self.elapsed = torch.zeros(count, dtype=torch.long, device=device)
        self.max_holds = torch.zeros(count, 6, dtype=torch.long, device=device)
        self.stage_success = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids, initial_clearance):
        for name in ("best", "hold_best", "steps", "paid", "initialized", "anchor",
                     "previous_valid", "grasp_grace", "transport_start", "keep_paid",
                     "failure_paid", "elapsed", "max_holds", "stage_success"):
            getattr(self, name)[ids] = 0
        self.initial_clearance[ids] = initial_clearance
        self.initial_table_force[ids] = float("nan")
        self.hover_age[ids] = -1
        self.force_history[:, ids] = 0
        self.quality_history[:, ids] = 0
        for mapping in [self.credits, self.episode_sums, *self.reason_masks, *self.reason_steps, *self.interruptions]:
            for value in mapping.values():
                value[ids] = 0

    def update(self, s: RewardSample):
        dt = self.dt
        for credit in self.credits.values():
            credit.zero_()
        # Infinite helical error before thread tracking is a legitimate sentinel.
        finite = torch.isfinite(torch.stack((s.clearance, s.table_force, s.linear_speed,
            s.angular_speed, s.pair_distance, s.depth, s.radial, s.angle), -1)).all(-1)
        finite &= torch.isfinite(s.relative_pose).all(-1) & torch.isfinite(s.forces).all(-1)
        valid = s.valid & finite & ~s.failed
        first = valid & ~self.initialized
        strict = (s.forces[:, 0] > 1.) & (s.forces[:, 1:].amax(-1) > 1.)
        slow = (s.linear_speed < .05) & (s.angular_speed < .5)
        old_steps = self.steps.clone()
        new_anchor = strict & (self.steps[:, 1] == 0)
        self.anchor[new_anchor] = s.relative_pose[new_anchor]
        if s.curriculum_anchor is not None:
            self.anchor.copy_(s.curriculum_anchor)
        pose_error = (s.relative_pose[:, :3] - self.anchor[:, :3]).norm(dim=-1)
        rot_error = quat_angle(s.relative_pose[:, 3:], self.anchor[:, 3:])
        stable_pose = (pose_error <= .005) & (rot_error <= math.pi / 36)
        vrel, wrel = relative_twist(self.previous, s.relative_pose, dt)
        vrel = torch.where(self.previous_valid[:, None], vrel, 0.)
        wrel = torch.where(self.previous_valid[:, None], wrel, 0.)
        self.previous.copy_(torch.nan_to_num(s.relative_pose))
        self.previous_valid.copy_(finite)
        slip = (vrel.norm(dim=-1) > .01) | (wrel.norm(dim=-1) > .2)
        unloaded = s.table_force < .1
        height_ok = (s.clearance >= .03) & (s.clearance - self.initial_clearance >= .03)
        hover_pos = (s.depth + .1).abs() < .01
        aligned = (s.radial < .006) & (s.angle < .2)
        grasp_ok = strict & slow & stable_pose
        lift_ok = strict & height_ok & unloaded & slow
        hover_ok = strict & unloaded & slow & s.lifted & hover_pos & aligned
        entry_ok = strict & unloaded & slow & (s.depth.abs() < .006) & aligned
        final_ok = s.installed & s.engaged & s.socket_contact & aligned & slow
        final_ok &= ((s.depth - .018).abs() < .004) & (s.helical < .004)
        conditions = torch.stack((strict, grasp_ok, lift_ok, hover_ok, entry_ok, final_ok), -1) & valid[:, None]
        self.steps.copy_(torch.where(conditions, self.steps + 1, 0))
        if s.curriculum_steps is not None:
            self.steps[:, 1:4] = torch.where(valid[:, None], s.curriculum_steps, 0)
        self.max_holds.copy_(torch.maximum(self.max_holds, self.steps))
        completed = torch.stack((
            self.steps[:, 1] >= self.required[0], self.steps[:, 2] >= self.required[1],
            self.steps[:, 3] >= self.required[2], self.steps[:, 4] >= self.required[3],
            s.engaged & s.tracking & s.socket_contact & aligned & (s.advance > .00025)
            & (-s.relative_angle / (2 * math.pi) > .15) & (s.helical < .004),
            s.final_success,
        ), -1) & valid[:, None]
        self.stage_success |= completed[:, (0, 1, 2, 5, 5)[self.stage]]

        # Establish inherited ledgers only after physical bank recovery passes.
        inherited_lift = first & ((s.source == 2) | (s.source == 3))
        inherited_hover = first & (s.source == 3)
        self.paid[inherited_lift, :2] = True
        self.best[inherited_lift, :7] = 1.
        self.hold_best[inherited_lift, :3] = 1.
        self.paid[inherited_hover, 2] = True
        self.best[inherited_hover, 7:10] = 1.
        self.hold_best[inherited_hover, 3] = 1.
        self.hover_age[inherited_hover] = .5
        already = self.paid.clone()

        f = torch.minimum(s.forces[:, 0], s.forces[:, 1:].amax(-1))
        q_raw = torch.nan_to_num((f - .1) / 1.1, nan=0., posinf=0., neginf=0.).clamp(0, 1)
        self.force_history[self.cursor] = torch.where(valid, q_raw, 0.)
        q = self.force_history.amin(0)
        g = lambda x: torch.exp(-x.square())
        q_pose = torch.nan_to_num(g(pose_error / .005) * g(rot_error / (math.pi / 36)))
        q_vel = torch.nan_to_num(g(s.linear_speed / .05) * g(s.angular_speed / .5))
        record_force = valid & torch.isnan(self.initial_table_force) & (s.source != 2) & (s.source != 3) & (s.table_force >= .1)
        self.initial_table_force[record_force] = s.table_force[record_force]
        reference = self.initial_table_force.clamp_min(.2)
        unload = torch.nan_to_num((reference - s.table_force) / (reference - .1), nan=0.).clamp(0, 1)
        height = torch.minimum(((s.clearance - self.initial_clearance) / .03).clamp(0, 1),
                               (s.clearance / .03).clamp(0, 1))
        clearance_gate = (s.clearance / .03).clamp(0, 1)
        a = (s.depth + .1).abs()
        far = (1 - torch.tanh(torch.sqrt(a.square() + s.radial.square()) / .2)) * g(s.angle / .5)
        fine = 1 / (1 + torch.sqrt((a / .01).square() + (s.radial / .006).square() + (s.angle / .2).square()))
        alignment = g(s.radial / .006) * g(s.angle / .2)
        screw = torch.nan_to_num(g(s.helical / .004), nan=0.)
        depth = torch.minimum(s.depth.clamp_min(0), (-s.relative_angle * .006 / (2 * math.pi)).clamp_min(0)).div(.018).clamp(0, 1)
        q_entry = alignment * g(s.depth / .006)
        q_install = (s.installed & s.engaged & s.socket_contact).float() * g((s.depth - .018) / .004) * screw * alignment * q_vel
        quality = torch.stack((strict * q_pose, q * q_pose * q_vel, q * height * unload * q_vel,
                               q * unloaded * fine * q_vel, q * q_entry, q_install), -1)
        self.quality_history[self.cursor] = torch.where(valid[:, None], torch.nan_to_num(quality), 0.)
        smooth = self.quality_history.amin(0)
        self.cursor = (self.cursor + 1) % self.window
        values = torch.stack((1 - torch.tanh(s.pair_distance / .08), q, smooth[:, 0], smooth[:, 1],
            q * height, q * height * unload, smooth[:, 2], q * clearance_gate * far,
            q * clearance_gate * fine, smooth[:, 3], q * alignment * ((s.depth + .1) / .1).clamp(0, 1),
            smooth[:, 4], (s.tracking & s.socket_contact & aligned).float() * depth * screw, smooth[:, 5]), -1)
        values = torch.nan_to_num(values, nan=0., posinf=0., neginf=0.).clamp(0, 1)
        # Seed from instantaneous reset quality, not the zero-filled filter,
        # so warming filters cannot pay for geometry provided by reset.
        baseline = values.clone()
        baseline[:, 1:4] = torch.stack((q_raw, strict * q_pose, q_raw * q_pose * q_vel), -1)
        baseline[:, 4:10] = torch.stack((q_raw * height, q_raw * height * unload, q_raw * height * unload * q_vel,
            q_raw * clearance_gate * far, q_raw * clearance_gate * fine, q_raw * unloaded * fine * q_vel), -1)
        baseline[:, 10:12] = torch.stack((q_raw * alignment * ((s.depth + .1) / .1).clamp(0, 1), q_raw * q_entry), -1)
        baseline[:, 13] = q_install
        self.best[first] = torch.maximum(self.best[first], torch.nan_to_num(baseline[first], nan=0.).clamp(0, 1))
        self.transport_start[first] = self.best[first, 7]

        hover_new = completed[:, 2] & ~already[:, 2]
        self.hover_age[hover_new] = 0.
        handoff = (1 - self.hover_age / .5).clamp(0, 1)
        tail = already[:, 2] | hover_new | s.tracking
        if self.stage < 3:
            tail = torch.zeros_like(tail)
        eligible = valid[:, None].expand(-1, len(POTENTIALS)).clone()
        eligible[:, :4] &= ~already[:, 0, None]
        eligible[:, 2:4] &= (self.steps[:, 0] > self.window)[:, None]
        eligible[:, 4:7] &= ~already[:, 1, None]
        eligible[:, 7:10] &= (handoff > 0)[:, None]
        eligible[:, 10:] &= tail[:, None]
        delta = (values - self.best).clamp_min(0) * eligible
        self.best.copy_(torch.maximum(self.best, torch.where(eligible, values, self.best)))
        for i, name in enumerate(POTENTIALS):
            scale = handoff if 7 <= i < 10 else 1.
            self.credits[name].copy_(self.weights[name] * delta[:, i] * scale)

        denominators = torch.tensor(
            [self.required[i] - self.window for i in (0, 0, 1, 2, 3, 4)], device=self.device)
        holds = ((self.steps - self.window) / denominators).clamp(0, 1)
        # Final timer is owned by assembly_success; never call it from reward.
        holds[:, 5] = ((s.final_hold - self.window * dt) / (1. - self.window * dt)).clamp(0, 1)
        hold_allowed = valid[:, None].expand(-1, 6).clone()
        hold_allowed[:, :2] &= ~already[:, 0, None]
        hold_allowed[:, 2] &= ~already[:, 1]
        hold_allowed[:, 3] &= ~already[:, 2]
        hold_allowed[:, 4] &= tail & ~already[:, 3]
        hold_allowed[:, 5] &= ~already[:, 5]
        hold_delta = (holds - self.hold_best).clamp_min(0) * hold_allowed
        self.hold_best.copy_(torch.maximum(self.hold_best, torch.where(hold_allowed, holds, self.hold_best)))
        for i, name in enumerate(HOLD_COMPONENTS):
            self.credits[name].copy_(self.hold_weights[i] * hold_delta[:, i])
        # Intermediate bonuses remain physical predicates, not best-history tests.
        bonus_events = completed & ~already
        bonus_events[:, 3:5] &= tail[:, None]
        b = torch.tensor(self.bonuses, device=self.device)
        self.credits["stage_completion"].copy_((bonus_events[:, :5] * b).sum(-1))
        if self.stage >= 3:
            self.credits["final_success"].copy_(80. * bonus_events[:, 5])
        self.paid |= completed

        progress = ((self.best[:, 7] - self.transport_start) / (1 - self.transport_start).clamp_min(1e-6)).clamp(0, 1)
        fade = (1 - 3 * progress.square() + 2 * progress.pow(3)) * handoff
        # No saved F0 in existing banks: its unloading auxiliary is disabled,
        # as in the approved plan. Live unloading still gates hover success.
        keep_quality = q * (q_pose + unload) / 2
        keep_active = valid & (s.lifted | already[:, 1]) & ((delta[:, 7] > 0) | (hold_delta[:, 3] > 0))
        keep = torch.minimum((1 - self.keep_paid).clamp_min(0), .25 * dt * fade * keep_quality * keep_active)
        if self.stage < 3:
            keep.zero_()
        self.keep_paid += keep
        self.credits["bridge_maintenance"].copy_(keep)

        self.grasp_grace = torch.where(strict, self.window + 1, (self.grasp_grace - 1).clamp_min(0))
        axis = s.axis_in_palm / s.axis_in_palm.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        projected_v = vrel - (vrel * axis).sum(-1, keepdim=True) * axis
        projected_w = wrel - (wrel * axis).sum(-1, keepdim=True) * axis
        # Permit intentional screw-axis motion only during physical thread tracking.
        v_cost = torch.where(s.tracking[:, None], projected_v, vrel).norm(dim=-1)
        w_cost = torch.where(s.tracking[:, None], projected_w, wrel).norm(dim=-1)
        excess = lambda x: (x - 1).clamp_min(0).square().clamp_max(1)
        slip_cost = .5 * (excess(v_cost / .01) + excess(w_cost / .2))
        near_hover = (a < .03) & (s.radial < .018) & (s.angle < .6)
        near_entry = (s.depth.abs() < .018) & (s.radial < .018) & (s.angle < .6)
        stabilize = torch.where(tail, (near_entry & ~s.tracking) | s.installed,
            torch.where(already[:, 1], near_hover, (height >= .8) | (strict & ~already[:, 0])))
        speed_cost = .5 * (excess(s.linear_speed / .05) + excess(s.angular_speed / .5))
        self.credits["slip_penalty"].copy_(-.3 * dt * torch.nan_to_num(slip_cost) * (self.grasp_grace > 0) * finite)
        self.credits["velocity_penalty"].copy_(-.2 * dt * torch.nan_to_num(speed_cost) * stabilize * finite)
        self.credits["time_penalty"].fill_(-.02 * dt)
        failure = s.failed & ~self.failure_paid
        self.credits["failure_penalty"].copy_(-2. * failure)
        self.failure_paid |= s.failed
        self.initialized |= valid
        self.hover_age = torch.where(self.hover_age >= 0, self.hover_age + dt, self.hover_age)
        self.elapsed += 1
        self._debug(s, valid, strict, slip, stable_pose, slow, height_ok, unloaded,
                    hover_pos, aligned, grasp_ok, lift_ok, hover_ok, final_ok, old_steps)
        for name, credit in self.credits.items():
            self.episode_sums[name] += credit
        return self.credits

    def _debug(self, s, valid, strict, slip, stable_pose, slow, height, unloaded,
               hover_pos, aligned, grasp_ok, lift_ok, hover_ok, final_ok, old_steps):
        masks = (
            (~strict, slip & (self.grasp_grace > 0), ~stable_pose & (self.grasp_grace > 0),
             ~slow & strict, grasp_ok & ~self.paid[:, 0], ~valid),
            (~strict, ~height, ~unloaded, ~slow, lift_ok & ~self.paid[:, 1], ~valid),
            (~strict, ~s.lifted, ~unloaded, ~hover_pos, s.radial >= .006, s.angle >= .2,
             ~slow, hover_ok & ~self.paid[:, 2], ~valid),
            (~s.socket_contact, ~s.engaged, ~s.installed | ((s.depth - .018).abs() >= .004),
             s.radial >= .006, s.angle >= .2, ~(s.helical < .004), ~slow,
             final_ok & ~s.final_success, ~valid),
        )
        for stage, values in enumerate(masks):
            column = (1, 2, 3, 5)[stage]
            interrupted = (old_steps[:, column] > 0) & (self.steps[:, column] == 0)
            for name, value in zip(FAIL_REASONS[stage], values):
                self.reason_masks[stage][name].copy_(value)
                self.reason_steps[stage][name] += value.long()
                self.interruptions[stage][name] += (value & interrupted).long()
