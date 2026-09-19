"""Size Custom bulb collision buffers before Isaac Lab creates the simulation."""


def collision_stack_bytes(num_envs: int, configured_bytes: int) -> int:
    """Keep large SDF-contact scenes above their observed stack requirement.

    The 4096-environment run exceeded 1.16 GiB after policy contacts developed.
    Budget 512 KiB per environment (2 GiB at 4096), with a 256 MiB floor for
    small scenes. This is per worker, not multiplied by distributed world size.
    Preserve larger explicit reservations. PhysX exposes this size as PxU32.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("Rotate-Bulb-Custom requires a positive integer scene.num_envs")
    if not isinstance(configured_bytes, int) or configured_bytes < 0:
        raise ValueError("gpu_collision_stack_size must be a non-negative integer")
    # PhysX stores this reservation in a uint32.  The 512 KiB/environment
    # heuristic reaches 2**32 at the required 8192-environment scale and
    # would reject the task before simulation construction.  Keep the
    # empirically validated 2 GiB process reservation as the automatic cap;
    # callers may still request a larger representable value explicitly.
    capacity = max(2**28, min(num_envs * 2**19, 2**31), configured_bytes)
    if capacity > 2**32 - 1:
        raise ValueError(
            "Rotate-Bulb-Custom collision stack exceeds the PhysX uint32 limit; "
            "reduce environments per worker or an oversized explicit stack reservation"
        )
    return capacity


def make_rotate_bulb_env(cfg, **kwargs):
    """Resolve physics mode and buffer capacity after all CLI/Hydra overrides."""
    from isaaclab.envs import ManagerBasedRLEnv

    if hasattr(cfg, "bulb_pure_collision"):
        from .env_cfg import configure_bulb_connection_mode
        from .self_collision import spawn_bulb_robot_with_self_collision

        if cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is not True:
            raise ValueError("Revised Rotate-Bulb-Custom requires robot self collisions to remain enabled")
        configure_bulb_connection_mode(cfg)
        cfg.scene.robot.spawn = cfg.scene.robot.spawn.replace(func=spawn_bulb_robot_with_self_collision)
        print(f"[INFO][RotateBulbCustom] bulb_pure_collision={cfg.bulb_pure_collision}", flush=True)
    previous = cfg.sim.physx.gpu_collision_stack_size
    capacity = collision_stack_bytes(cfg.scene.num_envs, previous)
    cfg.sim.physx.gpu_collision_stack_size = capacity
    print(
        f"[INFO][RotateBulbCustom] {cfg.scene.num_envs} environments per worker: "
        f"collision stack {previous} -> {capacity} bytes",
        flush=True,
    )
    return ManagerBasedRLEnv(cfg=cfg, **kwargs)
