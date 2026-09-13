"""Geometry and bounded rewards for a free rigid D-peg insertion task.

The peg origin is its leading tip; its shaft extends along local +Z.  Socket
coordinates describe a D cavity ``x <= flat, x*x+y*y <= radius*radius``.
No function in this module writes a pose, velocity, force, or joint target.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase


@dataclass(frozen=True)
class DPegGeometryConfig:
    shaft_radius_m: float = 0.012
    shaft_flat_m: float = 0.006
    shaft_length_m: float = 0.035
    tip_chamfer_m: float = 0.001
    hole_radius_m: float = 0.01275
    hole_flat_m: float = 0.00675
    mouth_chamfer_m: float = 0.001
    socket_mouth_height_m: float = 0.05
    socket_floor_height_m: float = 0.02
    insertion_depth_m: float = 0.028
    containment_tolerance_m: float = 1.0e-5
    profile_segments: int = 48


def validate_d_peg_geometry_metadata(path, insertion_depth_m=.028,
                                     socket_mouth_height_m=.05) -> DPegGeometryConfig:
    """Reject assets/configs whose meaning disagrees with the reward geometry.

    The first task version has fixed success tolerances and approved geometry;
    merely hashing a different asset does not make those tolerances correct.
    """
    cfg = DPegGeometryConfig()
    with Path(path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected = dict(shaft_radius=cfg.shaft_radius_m, shaft_flat_x=cfg.shaft_flat_m,
                    shaft_length=cfg.shaft_length_m, tip_chamfer=cfg.tip_chamfer_m,
                    hole_radius=cfg.hole_radius_m, hole_flat_x=cfg.hole_flat_m,
                    hole_depth=cfg.socket_mouth_height_m - cfg.socket_floor_height_m,
                    entry_chamfer=cfg.mouth_chamfer_m)
    dimensions = metadata.get("dimensions_m", {})
    for name, value in expected.items():
        actual = dimensions.get(name)
        if not isinstance(actual, (int, float)) or not math.isclose(actual, value, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"D-peg geometry mismatch for {name}: expected {value}, got {actual}")
    values = dict(insertion_depth_m=(insertion_depth_m, cfg.insertion_depth_m),
                  socket_mouth_height_m=(socket_mouth_height_m, cfg.socket_mouth_height_m),
                  metadata_target_depth=(metadata.get("socket", {}).get("target_depth_m"), cfg.insertion_depth_m))
    for name, (actual, expected_value) in values.items():
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected_value, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"D-peg geometry mismatch for {name}: expected {expected_value}, got {actual}")
    frames = ((metadata.get("peg", {}).get("tip_local"), [0, 0, 0]),
              (metadata.get("socket", {}).get("mouth_local"), [0, 0, cfg.socket_mouth_height_m]),
              (metadata.get("socket", {}).get("hole_bottom_local"), [0, 0, cfg.socket_floor_height_m]),
              (metadata.get("flat_normal_local"), [1, 0, 0]))
    if (metadata.get("schema_version") != 1 or metadata.get("units", {}).get("usd") != "m"
            or metadata.get("up_axis") != "Z" or any(actual != expected for actual, expected in frames)):
        raise ValueError("D-peg metadata must use the approved metre, tip-origin, +Z shaft and +X flat frames")
    return cfg


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply scalar-first quaternions, allowing extra point dimensions."""
    while q.ndim < v.ndim:
        q = q.unsqueeze(-2)
    xyz = q[..., 1:].expand_as(v)
    return v + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1) + q[..., :1] * v, dim=-1)


def _inverse(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _d_profile(radius: float, flat: float, segments: int, device, dtype) -> torch.Tensor:
    """Conservative tangent polygon: it contains the analytic circular D.

    An inscribed sampled circle can incorrectly accept a shaft penetrating a
    wall between samples.  Adjacent circle tangents instead form an outer
    polygon, with the two exact flat/circle intersections as end vertices.
    """
    start = math.acos(flat / radius)
    delta = (2 * math.pi - 2 * start) / segments
    angles = start + (torch.arange(segments, device=device, dtype=dtype) + 0.5) * delta
    arc = radius / math.cos(delta / 2) * torch.stack((angles.cos(), angles.sin()), dim=-1)
    end = math.sqrt(radius * radius - flat * flat)
    return torch.cat((torch.tensor([[flat, end]], device=device, dtype=dtype), arc,
                      torch.tensor([[flat, -end]], device=device, dtype=dtype)))


@lru_cache(maxsize=16)
@torch.inference_mode(False)
def _shaft_mesh(cfg: DPegGeometryConfig, device, dtype):
    """Share immutable local geometry, bounded across configs/devices/dtypes.

    Keep cached tensors usable outside inference mode too.  Callers transform
    these templates out of place; episode poses and reward state are not cached.
    """
    profiles = [
        _d_profile(cfg.shaft_radius_m - cfg.tip_chamfer_m,
                   cfg.shaft_flat_m - cfg.tip_chamfer_m, cfg.profile_segments, device, dtype),
        _d_profile(cfg.shaft_radius_m, cfg.shaft_flat_m, cfg.profile_segments, device, dtype),
    ]
    count = profiles[0].shape[0]
    rings = [torch.cat((profile, torch.full((count, 1), z, device=device, dtype=dtype)), dim=-1)
             for profile, z in zip((profiles[0], profiles[1], profiles[1]),
                                   (0.0, cfg.tip_chamfer_m, cfg.shaft_length_m))]
    index = torch.arange(count, device=device)
    edges = [torch.stack((index + ring * count, index.roll(-1) + ring * count), dim=-1)
             for ring in range(3)]
    for ring in range(2):
        # Include both triangulation diagonals of each side quad as well.
        for shift in (0, 1, -1):
            edges.append(torch.stack((index + ring * count,
                                      index.roll(shift) + (ring + 1) * count), dim=-1))
    return torch.cat(rings), torch.cat(edges)


def d_peg_geometry(peg_pos: torch.Tensor, peg_quat: torch.Tensor,
                   socket_pos: torch.Tensor, socket_quat: torch.Tensor,
                   cfg: DPegGeometryConfig | None = None, *,
                   compute_containment: bool = True) -> dict[str, torch.Tensor]:
    """Evaluate both cavity cross-sections and all shaft/wall overlap vertices.

    Clipping the shaft edges at the cavity mouth and floor handles tilted pegs:
    testing just the original tip ring would miss a shaft striking the rim.
    Observations can request only tip/depth/xy/tilt/yaw/finite by disabling
    containment.  Rewards and terminations always use the complete evaluation.
    """
    cfg = cfg or DPegGeometryConfig()
    device, dtype = peg_pos.device, peg_pos.dtype
    finite = torch.isfinite(torch.cat((peg_pos, peg_quat, socket_pos, socket_quat), -1)).all(-1)
    finite &= (peg_quat.norm(dim=-1) > 0.5) & (socket_quat.norm(dim=-1) > 0.5)
    pq = peg_quat / peg_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sq = socket_quat / socket_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tip = _rotate(_inverse(sq), peg_pos - socket_pos)
    tip = tip - tip.new_tensor([0.0, 0.0, cfg.socket_mouth_height_m])
    axis = _rotate(_inverse(sq), _rotate(pq, torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand_as(peg_pos)))
    flat = _rotate(_inverse(sq), _rotate(pq, torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand_as(peg_pos)))
    tilt = torch.acos(axis[:, 2].clamp(-1.0, 1.0))
    yaw = torch.atan2(flat[:, 1], flat[:, 0]).abs()
    if not compute_containment:
        return dict(tip=tip, depth=-tip[:, 2], xy=tip[:, :2].norm(dim=-1),
                    tilt=tilt, yaw=yaw, finite=finite)
    points, edges = _shaft_mesh(cfg, device, dtype)
    points = _rotate(_inverse(sq), _rotate(pq, points.expand(len(peg_pos), -1, -1))) + tip[:, None, :]
    floor = cfg.socket_floor_height_m - cfg.socket_mouth_height_m
    tolerance = cfg.containment_tolerance_m
    in_slab = (points[..., 2] <= tolerance) & (points[..., 2] >= floor - tolerance)
    a, b = points[:, edges[:, 0]], points[:, edges[:, 1]]
    dz = b[..., 2] - a[..., 2]
    candidates, masks = [points], [in_slab]
    for plane in (0.0, -cfg.mouth_chamfer_m, floor):
        fraction = (plane - a[..., 2]) / torch.where(dz.abs() > 1e-10, dz, torch.ones_like(dz))
        valid = (dz.abs() > 1e-10) & (fraction >= 0.0) & (fraction <= 1.0)
        candidates.append(a + fraction[..., None] * (b - a))
        masks.append(valid)
    clipped, included = torch.cat(candidates, 1), torch.cat(masks, 1)
    expansion = (clipped[..., 2] + cfg.mouth_chamfer_m).clamp(0.0, cfg.mouth_chamfer_m)
    radial_error = clipped[..., :2].norm(dim=-1) - cfg.hole_radius_m - expansion
    flat_error = clipped[..., 0] - cfg.hole_flat_m - expansion
    violation = torch.maximum(radial_error, flat_error)
    max_violation = torch.where(included, violation, -torch.inf).amax(dim=-1)
    overlap = included.any(dim=-1)
    bottom_ok = points[..., 2].amin(dim=-1) >= floor - tolerance
    contained = overlap & (max_violation <= tolerance) & bottom_ok & finite
    depth, xy = -tip[:, 2], tip[:, :2].norm(dim=-1)
    # The old target 20 mm above the mouth saturated both shaping rewards
    # before contact.  Reward the whole approach to the real mouth instead.
    # Once inserted, only physical containment can authorize shaping below it.
    height = tip[:, 2].clamp_min(0.0)
    pre_distance = torch.sqrt(xy.square() + height.square())
    approach = torch.exp(-(pre_distance / .03).square())
    align = torch.exp(-(xy / .01).square() - (tilt / math.radians(10)).square()
                      - (yaw / math.radians(15)).square() - (height / .03).square())
    return dict(tip=tip, depth=depth, xy=xy, tilt=tilt, yaw=yaw, contained=contained,
                overlap=overlap, max_violation=max_violation, bottom_ok=bottom_ok,
                approach=approach, align=align, finite=finite)


def d_peg_fine_alignment(geometry):
    """Resolve the last millimetres/degrees, including after the entry milestone."""
    g = geometry
    return torch.exp(-(g["xy"] / .0015).square()
                     - (g["tilt"] / math.radians(5)).square()
                     - (g["yaw"] / math.radians(5)).square()
                     - (g["tip"][:, 2].clamp_min(0.) / .01).square())


class DPegRewardState:
    """Pure-torch reward history, shared by rewards, metrics and terminations."""

    def __init__(self, num_envs: int, device, cfg: DPegGeometryConfig | None = None):
        self.cfg = cfg or DPegGeometryConfig()
        for name in ("best_approach", "best_align", "best_fine_align", "best_depth", "hold_time", "reward_rate"):
            setattr(self, name, torch.zeros(num_envs, device=device))
        for name in ("entered", "entry_provenance", "entry_paid", "success", "failed", "dropped", "outbound", "nonfinite", "initialized"):
            setattr(self, name, torch.zeros(num_envs, device=device, dtype=torch.bool))
        self.previous_tip = torch.zeros(num_envs, 3, device=device)
        self.last_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.components = {name: torch.zeros(num_envs, device=device)
                           for name in ("approach", "alignment", "fine_alignment", "grasp_loss", "entry", "depth", "stability", "success", "failure", "force", "time")}
        self.episode_sums = {name: torch.zeros(num_envs, device=device) for name in self.components}
        self.episode_diagnostics = {name: torch.zeros(num_envs, device=device) for name in (
            "elapsed_s", "measured_s", "held_s", "thumb_force_ns", "other_force_ns", "socket_force_ns",
            "xy_error_ms", "tilt_error_rad_s", "yaw_error_rad_s", "max_valid_depth_m",
            "max_thumb_force_n", "max_socket_force_n", "first_grasp_loss_s")}
        self.episode_diagnostics["first_grasp_loss_s"].fill_(-1.)
        self.has_held = torch.zeros(num_envs, device=device, dtype=torch.bool)
        metric_names = (
            "duration_s", "held_fraction", "held_time_s", "mean_thumb_force_n", "mean_other_force_n",
            "mean_socket_force_n", "mean_xy_error_m", "mean_tilt_error_rad", "mean_yaw_error_rad",
            "max_valid_depth_m", "first_grasp_loss_s", "experienced_grasp_loss", "success",
            "valid_entry", "held_entry", "dropped", "out_of_bounds", "nonfinite")
        self.diagnostic_totals = self._empty_diagnostic_summary(metric_names, device)
        self.pending_diagnostics = self._empty_diagnostic_summary(metric_names, device)

    @staticmethod
    def _empty_diagnostic_summary(metric_names, device):
        scalar = lambda: torch.zeros((), device=device, dtype=torch.float64)
        return {"count": scalar(), "sums": {name: scalar() for name in metric_names},
                "maxima": {name: scalar() for name in (
                    "max_valid_depth_m", "max_thumb_force_n", "max_socket_force_n")}}

    def _finish_episode_diagnostics(self, ids):
        """Aggregate each nonempty completed episode once, before its reset.

        These additive statistics are independent of Isaac Lab's cached log
        dictionary. A runner can SUM count/sums and MAX maxima across ranks.
        The first-loss time is right-censored at duration for episodes that
        retain the grasp; experienced_grasp_loss reports that distinction.
        """
        d = self.episode_diagnostics
        eligible = self.initialized[ids] & (d["elapsed_s"][ids] > 0)
        duration = d["elapsed_s"][ids].clamp_min(1e-8)
        measured = d["measured_s"][ids].clamp_min(1e-8)
        loss = d["first_grasp_loss_s"][ids]
        metrics = dict(
            duration_s=duration, held_fraction=d["held_s"][ids] / duration,
            held_time_s=d["held_s"][ids], mean_thumb_force_n=d["thumb_force_ns"][ids] / measured,
            mean_other_force_n=d["other_force_ns"][ids] / measured,
            mean_socket_force_n=d["socket_force_ns"][ids] / measured,
            mean_xy_error_m=d["xy_error_ms"][ids] / measured,
            mean_tilt_error_rad=d["tilt_error_rad_s"][ids] / measured,
            mean_yaw_error_rad=d["yaw_error_rad_s"][ids] / measured,
            max_valid_depth_m=d["max_valid_depth_m"][ids],
            first_grasp_loss_s=torch.where(loss >= 0, loss, duration),
            experienced_grasp_loss=(loss >= 0).float(), success=self.success[ids].float(),
            valid_entry=self.entry_paid[ids].float(), held_entry=(self.episode_sums["entry"][ids] > 0).float(),
            dropped=self.dropped[ids].float(), out_of_bounds=self.outbound[ids].float(),
            nonfinite=self.nonfinite[ids].float())
        count = eligible.sum(dtype=torch.float64)
        # Reductions over a zero-length reset slice must still return scalar 0.
        sums = {name: torch.where(eligible, value, 0.).sum(dtype=torch.float64)
                for name, value in metrics.items()}
        maxima = {name: torch.cat((torch.where(eligible, d[name][ids], 0.).reshape(-1),
                                  duration.new_zeros(1))).amax().double()
                  for name in self.diagnostic_totals["maxima"]}
        for summary in (self.diagnostic_totals, self.pending_diagnostics):
            summary["count"] += count
            for name, value in sums.items():
                summary["sums"][name] += value
            for name, value in maxima.items():
                summary["maxima"][name].copy_(torch.maximum(summary["maxima"][name], value))

    def completed_episode_diagnostics(self, *, pop=False):
        """Read lifetime totals, or consume only the completed-episode window."""
        source = self.pending_diagnostics if pop else self.diagnostic_totals
        result = {"count": source["count"].clone(),
                  "sums": {name: value.clone() for name, value in source["sums"].items()},
                  "maxima": {name: value.clone() for name, value in source["maxima"].items()}}
        if pop:
            source["count"].zero_()
            for group in (source["sums"], source["maxima"]):
                for value in group.values():
                    value.zero_()
        return result

    def _update_episode_diagnostics(self, snapshot, observed, entered, valid, dt):
        d, g = self.episode_diagnostics, snapshot["geometry"]
        measured = observed & valid
        held = measured & snapshot["held"]
        next_time = d["elapsed_s"] + observed * dt
        first_loss = measured & self.has_held & ~held & (d["first_grasp_loss_s"] < 0)
        d["first_grasp_loss_s"] = torch.where(first_loss, next_time, d["first_grasp_loss_s"])
        self.has_held |= held
        d["elapsed_s"] = next_time
        d["measured_s"] += measured * dt
        d["held_s"] += held * dt
        forces = snapshot.get("fingertip_forces")
        # Pure geometry callers may omit forces; production snapshots always
        # include all five real contact measurements.
        thumb = forces[:, 0] if forces is not None else torch.zeros_like(next_time)
        other = forces[:, 1:].amax(dim=-1) if forces is not None else torch.zeros_like(next_time)
        measurements = dict(thumb_force_ns=thumb, other_force_ns=other,
                            socket_force_ns=snapshot["socket_force"], xy_error_ms=g["xy"],
                            tilt_error_rad_s=g["tilt"], yaw_error_rad_s=g["yaw"])
        for name, value in measurements.items():
            d[name] += torch.where(measured, value, 0.) * dt
        maximums = dict(max_thumb_force_n=thumb, max_socket_force_n=snapshot["socket_force"],
                        max_valid_depth_m=torch.where(entered, g["depth"].clamp_min(0.), 0.))
        for name, value in maximums.items():
            d[name] = torch.maximum(d[name], torch.where(measured, value, 0.))

    def reset(self, snapshot: dict, env_ids=None, step_id: int = 0):
        ids = slice(None) if env_ids is None else env_ids
        self._finish_episode_diagnostics(ids)
        geometry = snapshot["geometry"]
        self.best_approach[ids] = geometry["approach"][ids]
        self.best_align[ids] = geometry["align"][ids]
        self.best_fine_align[ids] = d_peg_fine_alignment(geometry)[ids]
        self.best_depth[ids] = geometry["depth"][ids].clamp(0.0, self.cfg.insertion_depth_m) / self.cfg.insertion_depth_m
        self.previous_tip[ids] = geometry["tip"][ids]
        for name in ("hold_time", "reward_rate", "entered", "entry_provenance", "entry_paid", "success", "failed", "dropped", "outbound", "nonfinite"):
            getattr(self, name)[ids] = 0
        for value in self.components.values():
            value[ids] = 0
        for value in self.episode_sums.values():
            value[ids] = 0
        for name, value in self.episode_diagnostics.items():
            value[ids] = -1. if name == "first_grasp_loss_s" else 0.
        self.has_held[ids] = snapshot["held"][ids]
        self.initialized[ids] = True
        self.last_step[ids] = step_id

    def update(self, snapshot: dict, dt: float, step_id: int):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("step_dt must be finite and positive")
        if not self.initialized.all():
            self.reset(snapshot, (~self.initialized).nonzero(as_tuple=False).flatten(), step_id - 1)
        selected = self.last_step != step_id
        if not selected.any():
            return self.reward_rate
        observed = selected & ~self.success & ~self.failed
        active = observed.clone()
        g, held = snapshot["geometry"], snapshot["held"]
        valid = g["finite"] & snapshot["finite"]
        drop, outbound = snapshot["dropped"], snapshot["outbound"]
        failure = active & (drop | outbound | ~valid)
        active &= ~failure
        crossing = (self.previous_tip[:, 2] >= 0.0) & (g["tip"][:, 2] < 0.0)
        # Require an actual through-mouth passage, not an initial/side-inserted pose.
        alpha = self.previous_tip[:, 2] / (self.previous_tip[:, 2] - g["tip"][:, 2]).clamp_min(1e-8)
        crossing_point = self.previous_tip + alpha[:, None] * (g["tip"] - self.previous_tip)
        crossed_mouth = crossing_point[:, :2].norm(dim=-1) < self.cfg.hole_radius_m
        entering = crossing & crossed_mouth & g["contained"] & (g["tilt"] < math.radians(10))
        # A transient rim/wall contact invalidates this sample, not the history
        # of a real through-mouth entry. Clear that history on withdrawal,
        # leaving the aperture, floor penetration or episode termination.
        remains_in_aperture = ((g["depth"] >= 0.) & (g["xy"] < self.cfg.hole_radius_m)
                               & g["overlap"] & g["bottom_ok"])
        provenance = ((self.entry_provenance & remains_in_aperture) | entering) & active
        self.entry_provenance = torch.where(selected, provenance, self.entry_provenance)
        entered = provenance & g["contained"]
        self.entered = torch.where(selected, entered, self.entered)
        pre_entry = active & ~self.entry_paid & ((g["tip"][:, 2] >= 0.) | entered)
        approach_delta = (g["approach"] - self.best_approach).clamp_min(0.0) * pre_entry * held
        align_delta = (g["align"] - self.best_align).clamp_min(0.0) * pre_entry * held
        self.best_approach = torch.where(active, torch.maximum(self.best_approach, g["approach"]), self.best_approach)
        self.best_align = torch.where(active, torch.maximum(self.best_align, g["align"]), self.best_align)
        fine_align = d_peg_fine_alignment(g)
        fine_allowed = active & ((g["tip"][:, 2] >= 0.) | entered)
        fine_delta = (fine_align - self.best_fine_align).clamp_min(0.) * fine_allowed * held
        # Preserve the one-time budget and consume unheld progress too. A
        # retreat/regrasp cannot farm the new precision shaping reward.
        self.best_fine_align = torch.where(
            active, torch.maximum(self.best_fine_align, fine_align), self.best_fine_align)
        progress = (g["depth"] / self.cfg.insertion_depth_m).clamp(0., 1.)
        depth_delta = (progress - self.best_depth).clamp_min(0.0) * entered * held
        # Consume unheld progress too; later regrasp cannot reclaim old travel.
        self.best_depth = torch.where(entered, torch.maximum(self.best_depth, progress), self.best_depth)
        # Entry provenance is physical, not a contact flag: dropping the peg
        # through the mouth must consume its travel and entry milestone too.
        entry = entered & (g["depth"] >= .002) & ~self.entry_paid
        self.entry_paid |= entry
        stable = entered & held & (g["depth"] >= .027) & (g["depth"] <= .030)
        stable &= (g["xy"] < .000375) & (g["tilt"] < math.radians(3)) & (g["yaw"] < math.radians(3))
        stable &= (snapshot["linear_speed"] < .02) & (snapshot["angular_speed"] < .2)
        self.hold_time = torch.where(selected, torch.where(stable, self.hold_time + dt, 0.0), self.hold_time)
        success = active & (self.hold_time >= .5 - 1e-6)
        self.success |= success
        self.failed |= failure
        self.dropped |= failure & drop
        self.outbound |= failure & outbound
        self.nonfinite |= failure & ~valid
        self._update_episode_diagnostics(snapshot, observed, entered, valid, dt)
        # Give a small bounded signal for reducing pose and velocity error once
        # the peg is physically inserted. It is gated by real entry and grip,
        # so hovering above the socket or dropping the peg cannot farm reward.
        stability_gate = entered & held & (g["depth"] >= .020) & (g["depth"] <= .030)
        stability_score = torch.exp(
            -(g["xy"] / .00075).square()
            -(g["tilt"] / math.radians(2.5)).square()
            -(g["yaw"] / math.radians(2.5)).square()
            -(snapshot["linear_speed"] / .012).square()
            -(snapshot["angular_speed"] / .12).square())
        stability = 4.0 * stability_score * stability_gate.float() * dt
        values = dict(approach=5 * approach_delta, alignment=8 * align_delta, entry=5 * (entry & held).float(),
                      fine_alignment=8 * fine_delta, grasp_loss=-.2 * dt * (active & ~held).float(),
                      depth=40 * depth_delta, stability=stability, success=80 * success.float(), failure=-20 * failure.float(),
                      force=-2 * ((snapshot["socket_force"] - 30) / 30).clamp(0, 1) * dt * active,
                      time=-.1 * dt * active.float())
        # A NaN pose can yield NaN * False in disabled shaping terms.  Sanitize
        # each component before summing, preserving the finite failure penalty.
        values = {name: torch.nan_to_num(value) for name, value in values.items()}
        for name, value in values.items():
            self.components[name] = torch.where(selected, value, self.components[name])
            self.episode_sums[name] += torch.where(selected, value, 0.0)
        rate = sum(values.values()) / dt
        self.reward_rate = torch.where(selected, torch.nan_to_num(rate), self.reward_rate)
        self.previous_tip = torch.where(selected[:, None], g["tip"], self.previous_tip)
        self.last_step[selected] = step_id
        return self.reward_rate


def _config(env) -> DPegGeometryConfig:
    configured = getattr(env.cfg, "_d_peg_geometry", None)
    if configured is not None:
        result = DPegGeometryConfig(**configured) if isinstance(configured, dict) else configured
        if (env.cfg.insertion_depth_m != result.insertion_depth_m
                or env.cfg.socket_mouth_height_m != result.socket_mouth_height_m):
            raise ValueError("D-peg task parameters changed after geometry validation")
        return result
    return DPegGeometryConfig(insertion_depth_m=getattr(env.cfg, "insertion_depth_m", .028),
                             socket_mouth_height_m=getattr(env.cfg, "socket_mouth_height_m", .05))


def _geometry(env, *, compute_containment=True):
    peg, socket = env.scene["object"].data, env.scene["socket"].data
    return d_peg_geometry(peg.root_pos_w, peg.root_quat_w, socket.root_pos_w, socket.root_quat_w,
                          _config(env), compute_containment=compute_containment)


def _contact_force(env, name: str):
    sensor = env.scene.sensors[name]
    forces = sensor.data.force_matrix_w
    return forces.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=-1)


def _snapshot(env):
    geometry = _geometry(env)
    peg = env.scene["object"].data
    forces = torch.stack([_contact_force(env, f"right_{finger}dip_roll_rubber_link_object_s")
                          for finger in ("thumb", "index", "mid", "ring", "pinky")], dim=-1)
    held = (forces[:, 0] > 1.) & (forces[:, 1:].amax(dim=-1) > 1.)
    socket_force = _contact_force(env, "peg_socket_contact")
    table_force = _contact_force(env, "peg_table_contact")
    local = peg.root_pos_w - env.scene.env_origins
    term = getattr(env.cfg.terminations, "object_out_of_bound", None)
    bounds = getattr(term, "params", {}).get("in_bound_range", {})
    outbound = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for index, axis in enumerate(("x", "y", "z")):
        if axis in bounds:
            low, high = bounds[axis]
            outbound |= (local[:, index] < low) | (local[:, index] > high)
    finite = torch.isfinite(torch.cat((forces, socket_force[:, None], table_force[:, None],
                                      peg.root_lin_vel_w, peg.root_ang_vel_w), dim=-1)).all(-1)
    robot = env.scene["robot"].data
    for value in (robot.joint_pos, robot.joint_vel, robot.root_pos_w, robot.root_quat_w):
        finite &= torch.isfinite(value).flatten(start_dim=1).all(dim=-1)
    return dict(geometry=geometry, held=held, fingertip_forces=forces, socket_force=socket_force,
                dropped=table_force > .5, outbound=outbound, finite=finite,
                linear_speed=peg.root_lin_vel_w.norm(dim=-1), angular_speed=peg.root_ang_vel_w.norm(dim=-1))


def _get_state(env):
    if not hasattr(env, "_d_peg_state"):
        env._d_peg_state = DPegRewardState(env.num_envs, env.device, _config(env))
        env._d_peg_state.reset(_snapshot(env), step_id=int(env.common_step_counter))
    return env._d_peg_state


def _updated_state(env):
    state = _get_state(env)
    if (state.last_step != int(env.common_step_counter)).any():
        state.update(_snapshot(env), env.step_dt, int(env.common_step_counter))
        # Bounded progress/event components are reported as per-step credit;
        # force/time are already integrated over this control step.
        log = env.extras.setdefault("log", {})
        for name, value in state.components.items():
            log[f"DPeg/reward_{name}"] = value.mean()
        log["DPeg/depth_ratio"] = state.best_depth.mean()
        log["DPeg/hold_time"] = state.hold_time.mean()
        log["DPeg/success"] = state.success.float().mean()
    return state


class DPegInsertionState(ManagerTermBase):
    """RewardManager adapter; use weight=1 (action penalties remain in cfg)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state = _get_state(env)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        if self.state.success[ids].numel():
            log["DPeg/EpisodeSuccess"] = self.state.success[ids].float().mean()
            log["DPeg/EpisodeDepth"] = self.state.best_depth[ids].mean()
            for name, value in self.state.episode_sums.items():
                log[f"DPeg/EpisodeReward_{name}"] = value[ids].mean()
        command_manager = getattr(self._env, "command_manager", None)
        if command_manager is not None:
            # CommandManager.reset executes after RewardManager.reset.  Preserve
            # the terminal sample instead of logging the previous step's zero.
            command = command_manager.get_term("object_pose")
            command.metrics["success"][ids] = self.state.success[ids].float()
        self.state.reset(_snapshot(self._env), env_ids, int(self._env.common_step_counter))

    def __call__(self, env):
        return _updated_state(env).reward_rate


def d_peg_success(env):
    return _updated_state(env).success


def d_peg_dropped(env):
    return _updated_state(env).dropped


def d_peg_out_of_bounds(env, in_bound_range=None):
    # The shared evaluator reads this term's configured in_bound_range.
    return _updated_state(env).outbound


def d_peg_nonfinite(env):
    return _updated_state(env).nonfinite


def d_peg_insertion_progress(env):
    """Full-state task observation, deliberately independent of reward creation."""
    g = _geometry(env, compute_containment=False)
    return torch.stack(((g["depth"] / _config(env).insertion_depth_m).clamp(0, 1),
                        (g["xy"] / _config(env).hole_radius_m).clamp(0, 10)), dim=-1)


def d_peg_insertion_phase(env):
    g = _geometry(env, compute_containment=False)
    aligned = (g["xy"] < .002) & (g["tilt"] < math.radians(3)) & (g["yaw"] < math.radians(3))
    time_left = (1 - env.episode_length_buf / env.max_episode_length).clamp(0, 1)
    return torch.stack((aligned.float(), time_left), dim=-1)


def d_peg_state_metrics(env):
    state = _updated_state(env)
    return dict(success=state.success.float(), inserted=state.entered.float(),
                depth=state.best_depth, hold_time=state.hold_time,
                **{f"reward_{name}": value for name, value in state.components.items()})


def d_peg_diagnostic_totals(env):
    """Read completed-episode count, sums and maxima without advancing state.

    All ranks expose the same scalar keys, even when no episode has completed.
    This function performs no distributed operation and returns copies.
    """
    return _get_state(env).completed_episode_diagnostics()


def d_peg_pop_episode_diagnostics(env):
    """Consume completed-episode statistics since the previous runner update.

    This never clears in-progress episode history or the lifetime totals.
    Only the training runner may call distributed collectives on this result.
    """
    return _get_state(env).completed_episode_diagnostics(pop=True)
