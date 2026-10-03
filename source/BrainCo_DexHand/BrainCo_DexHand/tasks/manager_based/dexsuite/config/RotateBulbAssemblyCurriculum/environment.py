"""Curriculum-only environment lifecycle; never advances physics inside reset."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils

from ..RotateBulbCustom.env_cfg import configure_bulb_connection_mode, spawn_tabletop_threaded_lamp
from ..RotateBulbCustom.environment import collision_stack_bytes
from ..RotateBulbCustom.self_collision import spawn_bulb_robot_with_self_collision
from .env_cfg import configure_curriculum
from .runtime import CurriculumRuntime


@sim_utils.clone
def spawn_fixed_curriculum_lamp(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A world-fixed Stage0 bulb, authored once before PhysX views exist."""
    from pxr import UsdPhysics

    prim = spawn_tabletop_threaded_lamp(prim_path, cfg, translation, orientation, **kwargs)
    bulb = prim.GetStage().GetPrimAtPath(str(prim.GetPath()) + "/Bulb")
    UsdPhysics.RigidBodyAPI(bulb).CreateKinematicEnabledAttr(True)
    return prim


class AssemblyCurriculumEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self._bulb_curriculum = None
        super().__init__(cfg, **kwargs)
        self._bulb_curriculum = CurriculumRuntime(self)

    def _reset_idx(self, env_ids):
        runtime = self._bulb_curriculum
        records = runtime.finish(env_ids) if runtime is not None else []
        super()._reset_idx(env_ids)
        if runtime is not None:
            runtime.after_reset(env_ids)
        if records:
            for source in sorted({r["source_stage"] for r in records}):
                selected = [r for r in records if r["source_stage"] == source]
                prefix = f"Curriculum/source{source}"
                self.extras["log"][prefix + "/success"] = torch.tensor(
                    sum(r["success"] for r in selected) / len(selected), device=self.device)
                self.extras["log"][prefix + "/episodes"] = torch.tensor(len(selected), device=self.device)
            self.extras["log"]["Curriculum/reset_failure"] = torch.tensor(
                sum(r["reset_failure"] for r in records) / len(records), device=self.device)


def make_curriculum_env(cfg, **kwargs):
    configure_curriculum(cfg)
    if cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is not True:
        raise ValueError("Curriculum preserves robot self collisions")
    configure_bulb_connection_mode(cfg)
    cfg.scene.robot.spawn = cfg.scene.robot.spawn.replace(func=spawn_bulb_robot_with_self_collision)
    if cfg.bulb_curriculum.fixed_bulb:
        cfg.scene.lamp.spawn.func = spawn_fixed_curriculum_lamp
    cfg.sim.physx.gpu_collision_stack_size = collision_stack_bytes(
        cfg.scene.num_envs, max(cfg.sim.physx.gpu_collision_stack_size,
                               min(cfg.scene.num_envs * 2**20, 15 * 2**28)))
    # Horizontal hand/table contacts exceeded 590 MiB at 1024 environments.
    # Keep 1 MiB/environment (1 GiB at 1024, capped at 3.75 GiB), preserving larger
    # explicit reservations. This is PhysX's pinned host collision stack,
    # distinct from device VRAM. Native logs must still be checked for overflow.
    # Pair buffers are device allocations; scale with this worker's scene,
    # never with the combined environment count of all distributed workers.
    for name in ("gpu_found_lost_pairs_capacity", "gpu_found_lost_aggregate_pairs_capacity",
                 "gpu_total_aggregate_pairs_capacity"):
        setattr(cfg.sim.physx, name, max(getattr(cfg.sim.physx, name), cfg.scene.num_envs * 4096))
    if cfg.scene.num_envs >= 4096:
        cfg.sim.physx.gpu_max_rigid_contact_count = max(cfg.sim.physx.gpu_max_rigid_contact_count, 2**24)
        cfg.sim.physx.gpu_max_rigid_patch_count = max(cfg.sim.physx.gpu_max_rigid_patch_count, 2**20)
    return AssemblyCurriculumEnv(cfg=cfg, **kwargs)
