"""Check collision exclusions on the composed donor geometry without Isaac Sim."""

import hashlib
import importlib.util
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CUSTOM = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom"
ASSET = ROOT / "assets/rotate_bulb_custom/robot/usd/rizon4_training.usda"
SPEC = importlib.util.spec_from_file_location("bulb_self_collision", CUSTOM / "self_collision.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def scene():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    root = "/World/envs/env_0/Robot"
    prim = stage.DefinePrim(root, "Xform")
    prim.GetReferences().AddReference(str(ASSET))
    return stage, root


def filtered(stage, root):
    from pxr import Usd

    return {str(p.GetPath()): tuple(map(str, p.GetRelationship("physics:filteredPairs").GetTargets()))
            for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
            if p.GetRelationship("physics:filteredPairs")}


def test_restored_pairs_match_nonadjacent_arm_and_proximal_hand_contract(scene):
    from pxr import UsdPhysics

    stage, root = scene
    source_digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    before = filtered(stage, root)
    report = MODULE.restore_bulb_arm_self_collisions(stage, root)
    assert report["removed_count"] == report["restored_pair_count"] == 192
    after = filtered(stage, root)
    for name in ("base_link", "link1", "link2", "link3", "link4", "link5"):
        assert after[root + "/" + name] == ()
    assert after[root + "/link6"] == tuple(root + "/" + name for name in
                                           ("flange", "palm_link", "right_base_link", "palm"))
    changed = {root + "/" + name for name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")}
    assert {k: v for k, v in after.items() if k not in changed} == {
        k: v for k, v in before.items() if k not in changed}
    for index in range(1, 8):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(root + f"/joints/joint{index}"))
        assert joint.GetCollisionEnabledAttr().Get() is False
    assert MODULE.restore_bulb_arm_self_collisions(stage, root)["removed_count"] == 0
    assert filtered(stage, root) == after
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == source_digest


def test_reverse_pair_is_removed_but_external_and_hand_pairs_are_retained(scene):
    from pxr import Sdf, UsdPhysics

    stage, root = scene
    link5 = stage.GetPrimAtPath(root + "/link5")
    relation = UsdPhysics.FilteredPairsAPI.Apply(link5).GetFilteredPairsRel()
    relation.AddTarget(Sdf.Path(root + "/link3"))
    relation.AddTarget(Sdf.Path("/World/ExternalObstacle"))
    report = MODULE.restore_bulb_arm_self_collisions(stage, root)
    assert report["removed_count"] == 193
    assert list(map(str, relation.GetTargets())) == ["/World/ExternalObstacle"]
    hand_relation = stage.GetPrimAtPath(root + "/right_thumb_CMP_Link").GetRelationship("physics:filteredPairs")
    assert root + "/right_thumb_DIP_Link" in list(map(str, hand_relation.GetTargets()))


def test_incomplete_robot_fails_before_authoring_collision_overrides(scene):
    stage, root = scene
    stage.GetPrimAtPath(root + "/link4").SetActive(False)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="missing body prims"):
        MODULE.restore_bulb_arm_self_collisions(stage, root)
    assert stage.GetRootLayer().ExportToString() == before


def test_repaired_source_is_copied_to_two_independent_environment_roots(scene):
    from pxr import Sdf, Usd

    stage, root = scene
    MODULE.restore_bulb_arm_self_collisions(stage, root)
    # Flatten composition as copy_from_source does: the second environment must
    # carry the source's stronger filter opinions even without live inheritance.
    flattened = stage.Flatten()
    second = "/World/envs/env_1/Robot"
    stage.DefinePrim("/World/envs/env_1", "Xform")
    assert Sdf.CopySpec(flattened, Sdf.Path(root), stage.GetRootLayer(), Sdf.Path(second))
    for instance in (root, second):
        assert MODULE.restore_bulb_arm_self_collisions(stage, instance)["removed_count"] == 0
        assert not stage.GetPrimAtPath(instance + "/link3").GetRelationship("physics:filteredPairs").GetTargets()
        retained = stage.GetPrimAtPath(instance + "/link6").GetRelationship("physics:filteredPairs").GetTargets()
        assert all(str(p).startswith(instance + "/") for p in retained)
    # Opening the unchanged asset again still sees the donor's exclusions.
    donor = Usd.Stage.Open(str(ASSET))
    assert len(donor.GetPrimAtPath("/flexiv_rizon4/base_link").GetRelationship("physics:filteredPairs").GetTargets()) == 31


def test_saved_grasp_has_clearance_for_every_restored_collision_pair(scene):
    from pxr import Usd, UsdGeom, UsdPhysics

    # Independent joint-frame FK is already checked against 37 editor frames.
    fk_checks = runpy.run_path(str(ROOT / "tests/test_rotate_bulb_grasp_fk.py"))
    reference = json.loads((ROOT / "assets/rotate_bulb_custom/grasp_reference/simulation_state.json").read_text())["robot"]
    angles = dict(zip(reference["joint_names"], reference["joint_positions_rad"], strict=True))
    stage = Usd.Stage.Open(str(ASSET))
    base = reference["base"]
    poses = fk_checks["_usd_forward_kinematics"](
        stage, angles, fk_checks["_pose_matrix"](base["position_m"], base["quaternion_wxyz"]))
    cache = UsdGeom.XformCache()
    points = {}
    for collider in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = collider.GetPath().pathString.split("/")[2]
        body_prim = stage.GetPrimAtPath("/flexiv_rizon4/" + body)
        for mesh_prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            vertices = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=float)
            local = np.asarray(cache.GetLocalToWorldTransform(mesh_prim)
                               * cache.GetLocalToWorldTransform(body_prim).GetInverse()).T
            world = poses[body] @ local
            points.setdefault(body, []).append(vertices @ world[:3, :3].T + world[:3, 3])
    bounds = {body: (np.vstack(values).min(0), np.vstack(values).max(0)) for body, values in points.items()}
    clearances = []
    for first, second in MODULE.RESTORED_PAIRS:
        if first not in bounds or second not in bounds:
            continue  # Flange and palm frames have no collider of their own.
        a, b = bounds[first], bounds[second]
        separating_gap = np.maximum(np.maximum(a[0], b[0]) - np.minimum(a[1], b[1]), 0.0)
        lower_bound = float(np.linalg.norm(separating_gap))
        assert lower_bound >= 0.001, (first, second, lower_bound)
        clearances.append(lower_bound)
    assert len(clearances) == 180
    assert min(clearances) > 0.011
