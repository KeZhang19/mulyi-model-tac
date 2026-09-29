"""Finite per-episode lift bonuses for tabletop assembly; no physics writes."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from .assembly_guidance import assembly_soft_grasp_gate


@configclass
class AssemblyLiftCfg:
    """Distances are measured from the reset collision surface, not root Z."""

    target_height_m: float = 0.03
    progress_budget: float = 8.0
    stable_bonus: float = 3.0
    stable_duration_s: float = 0.3
    contact_force_scale_n: float = 1.0
    max_table_force_n: float = 0.1


def bulb_collision_vertices(stage, root_path: str, device) -> torch.Tensor:
    """Cache support vertices of the enabled collision meshes in body frame.

    Taking a linear minimum over convex-hull vertices is identical to taking
    it over every original vertex, including for this concave threaded mesh.
    This hull is only a reward measurement; it never replaces the collider.
    USD is read once; live body poses subsequently come from PhysX tensors.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Missing bulb rigid body: {root_path}")
    cache = UsdGeom.XformCache()
    clouds = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Lift measurement requires mesh colliders: {prim.GetPath()}")
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid bulb collision vertices: {prim.GetPath()}")
        transform, _ = cache.ComputeRelativeTransform(prim, root)
        matrix = np.asarray(transform)
        clouds.append(points @ matrix[:3, :3] + matrix[3, :3])
    if not clouds:
        raise ValueError(f"No enabled bulb collision meshes below {root_path}")
    points = np.concatenate(clouds)
    points = points[ConvexHull(points).vertices]
    return torch.tensor(points, dtype=torch.float32, device=device)


def bulb_table_clearance(env, vertices: torch.Tensor, env_ids=None) -> torch.Tensor:
    """Lowest collision vertex above the horizontal table, using live poses."""
    ids = slice(None) if env_ids is None else env_ids
    bulb = env.scene["object"].data
    quat = bulb.root_quat_w[ids]
    norm = quat.norm(dim=-1, keepdim=True)
    w, x, y, z = (quat / norm.clamp_min(1.0e-8)).unbind(dim=-1)
    # World-up expressed in the bulb frame = row 3 of the rotation matrix.
    up_local = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                            1 - 2 * (x.square() + y.square())), dim=-1)
    bottom_z = bulb.root_pos_w[ids, 2] + (up_local @ vertices.T).amin(dim=-1)
    table_z = env.scene["table"].data.root_pos_w[ids, 2] + env.cfg.scene.table.spawn.size[2] / 2
    clearance = bottom_z - table_z
    valid = torch.isfinite(quat).all(dim=-1) & (norm.squeeze(-1) > 1.0e-8)
    valid &= torch.isfinite(bulb.root_pos_w[ids]).all(dim=-1)
    return torch.where(valid, clearance, torch.full_like(clearance, float("nan")))


class AssemblyLiftReward(ManagerTermBase):
    """Best-so-far grasped lift, or a once-only sustained-lift bonus.

    Each component owns its history; evaluation order cannot affect payout.
    Outputs divide by step_dt because RewardManager multiplies by step_dt.
    With weight=1, total episode payouts are capped at 8 and 3 respectively.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params.get("component") not in ("progress", "stable"):
            raise ValueError("component must be 'progress' or 'stable'")
        self.lift = env.cfg.assembly_lift
        for name in ("target_height_m", "progress_budget", "stable_bonus", "stable_duration_s",
                     "contact_force_scale_n", "max_table_force_n"):
            value = getattr(self.lift, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"assembly_lift.{name} must be positive and finite")
        if not math.isfinite(env.step_dt) or env.step_dt <= 0:
            raise ValueError("step_dt must be positive and finite")
        # Both components share immutable geometry, but never episode history.
        if not hasattr(env, "_assembly_lift_vertices"):
            env._assembly_lift_vertices = bulb_collision_vertices(
                env.scene.stage, env.scene["object"].root_physx_view.prim_paths[0], env.device,
            )
        self.vertices = env._assembly_lift_vertices
        self.initial_clearance = torch.zeros(env.num_envs, device=env.device)
        self.best_quality = torch.zeros_like(self.initial_clearance)
        self.stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # RewardManager resets AFTER reset events write the new root poses.
        # Capture each episode's actual pose, including random XY and yaw.
        self.initial_clearance[ids] = bulb_table_clearance(self._env, self.vertices, env_ids)
        self.best_quality[ids] = 0
        self.stable_steps[ids] = 0
        self.paid[ids] = False

    def __call__(self, env, component: str):
        cfg = self.lift
        height = bulb_table_clearance(env, self.vertices) - self.initial_clearance
        grasp = assembly_soft_grasp_gate(env, cfg.contact_force_scale_n)
        valid = torch.isfinite(height) & torch.isfinite(grasp)
        quality = torch.where(valid, grasp * (height / cfg.target_height_m).clamp(0, 1), 0.0)
        if component == "progress":
            progress = (quality - self.best_quality).clamp_min(0)
            self.best_quality = torch.maximum(self.best_quality, quality)
            return cfg.progress_budget * progress / env.step_dt
        if component != "stable":
            raise ValueError("component must be 'progress' or 'stable'")

        forces = env.scene.sensors["bulb_table_contact"].data.force_matrix_w
        if forces is None:
            unsupported = torch.zeros_like(self.paid)
        else:
            # Sum magnitudes rather than resultant: opposing contacts must not
            # cancel. Missing/non-finite filtered data cannot certify a lift.
            table_force = forces.norm(dim=-1).flatten(start_dim=1).sum(dim=-1)
            unsupported = torch.isfinite(table_force) & (table_force < cfg.max_table_force_n)
        held = valid & (height >= cfg.target_height_m) & (grasp >= 1) & unsupported
        self.stable_steps = torch.where(held, self.stable_steps + 1, 0)
        required_steps = math.ceil(cfg.stable_duration_s / env.step_dt - 1.0e-9)
        credit = (self.stable_steps >= required_steps) & ~self.paid
        self.paid |= credit
        return cfg.stable_bonus * credit.float() / env.step_dt
