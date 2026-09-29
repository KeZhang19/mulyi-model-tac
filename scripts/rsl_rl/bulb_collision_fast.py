"""Collision-only opt-in configuration, applied before native scene creation."""
from pathlib import Path


def configure_collision_fast(cfg, profile):
    """Change only the lamp collision asset and optional robot self collisions.

    Return an undo callback for the process-local spawn hook. Source USDs,
    shared task defaults, joints, materials, control and task terms are intact.
    """
    from isaaclab.sim import clone
    from pxr import Usd, UsdPhysics, PhysxSchema
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom import self_collision

    if set(profile) - {"lamp_asset", "robot_self_collision", "robot_asset"}:
        raise ValueError("collision_fast accepts collision fields only")
    path = Path(profile["lamp_asset"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg.scene.lamp.spawn.usd_path = str(path)
    if profile.get("robot_asset"):
        robot_path = Path(profile["robot_asset"]).resolve()
        if not robot_path.is_file():
            raise FileNotFoundError(robot_path)
        cfg.scene.robot.spawn.usd_path = str(robot_path)
    original = self_collision.spawn_bulb_robot_with_self_collision
    if not profile.get("robot_self_collision", True):
        @clone
        def spawn_without_self_collision(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
            prim = self_collision._spawn_bulb_robot_source(prim_path, spawn_cfg, translation, orientation, **kwargs)
            roots = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if not roots:
                raise RuntimeError("Robot articulation root not found")
            for root in roots:
                PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
            return prim
        self_collision.spawn_bulb_robot_with_self_collision = spawn_without_self_collision

    def restore():
        self_collision.spawn_bulb_robot_with_self_collision = original
    return restore


def collision_runtime_manifest(raw):
    """Read effective native settings, shapes, geometry, and pair exclusions."""
    from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom, PhysicsSchemaTools, UsdUtils
    from omni.physx.bindings._physx import PhysicsSceneStats
    import omni.physx
    stage = raw.sim.stage
    stats = PhysicsSceneStats()
    ok = omni.physx.get_physx_statistics_interface().get_physx_scene_statistics(
        UsdUtils.StageCache.Get().Insert(stage).ToLongInt(),
        PhysicsSchemaTools.sdfPathToInt(raw.cfg.sim.physics_prim_path), stats)
    rows, filters, roots = [], [], []
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            mesh = UsdGeom.Mesh(p) if p.IsA(UsdGeom.Mesh) else None
            rows.append({"path": str(p.GetPath()), "type": p.GetTypeName(),
                "enabled": UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get(),
                "approximation": p.GetAttribute('physics:approximation').Get(),
                "sdf_resolution": p.GetAttribute('physxSDFMeshCollision:sdfResolution').Get(),
                "faces": len(mesh.GetFaceVertexCountsAttr().Get()) if mesh else None})
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append({"path": str(p.GetPath()), "self_collision": PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get()})
        rel = p.GetRelationship('physics:filteredPairs')
        if rel and rel.GetTargets():
            filters.append({"path": str(p.GetPath()), "targets": [str(x) for x in rel.GetTargets()]})
    native = {k: getattr(stats, k) for k in dir(stats) if not k.startswith('_') and isinstance(getattr(stats, k), (int, float, bool))} if ok else None
    return {"colliders": rows, "articulations": roots, "filters": filters, "native_scene_stats": native}
