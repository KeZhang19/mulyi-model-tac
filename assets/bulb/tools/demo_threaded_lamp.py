#!/usr/bin/env python3
"""Drive only bulb rotation; native PhysX coupling produces screw advance.

Examples (from repository root):
  python assets/bulb/tools/demo_threaded_lamp.py --headless --validate
  python assets/bulb/tools/demo_threaded_lamp.py
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser(__doc__)
PARSER.add_argument("--validate", action="store_true")
PARSER.add_argument("--report", type=Path)
PARSER.add_argument("--cycles", type=int, default=1)
PARSER.add_argument("--collision-probe", action="store_true",
                    help="Test-only: misphase the male thread and isolate thread contacts; do not save changes to USD.")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
if ARGS.cycles < 1:
    PARSER.error("--cycles must be positive")
if not ARGS.device.startswith("cuda"):
    PARSER.error("SDF thread contacts require GPU dynamics (--device cuda:0)")
LAUNCHER = AppLauncher(ARGS)
APP = LAUNCHER.app


def run():
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils, PhysxSchema

    from build_threaded_lamp import SOURCE, OUTPUT, ROOT, PITCH, TRAVEL, INITIAL_EXTENSION, source_meshes

    asset_stage = Usd.Stage.Open(str(OUTPUT))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(OUTPUT))
    assert len(layers) == 1 and not assets and not unresolved
    expected = source_meshes(Usd.Stage.Open(str(SOURCE)))
    for name, arrays in expected.items():
        mesh = UsdGeom.Mesh(asset_stage.GetPrimAtPath(ROOT + "/" + name))
        np.testing.assert_array_equal(np.array(mesh.GetPointsAttr().Get()), arrays[0])
        np.testing.assert_array_equal(np.array(mesh.GetFaceVertexCountsAttr().Get()), arrays[1])
        np.testing.assert_array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()), arrays[2])
        if "Thread" not in name:
            approximation = UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get()
            assert approximation == "sdf"
    for name, approximation in (("Support/InternalThreadCollider", "none"),
                                 ("Bulb/ExternalThreadCollider", "sdf")):
        prim = asset_stage.GetPrimAtPath(ROOT + "/" + name)
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == approximation
        assert PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get() == 0
    female = UsdGeom.Mesh(asset_stage.GetPrimAtPath(ROOT + "/Support/InternalThreadCollider"))
    source_female = expected["Support/InternalThread"]
    np.testing.assert_array_equal(np.array(female.GetFaceVertexIndicesAttr().Get()), source_female[2])
    untouched = source_female[0][:, 0] <= source_female[0][:, 0].max() - .001
    np.testing.assert_array_equal(np.array(female.GetPointsAttr().Get())[untouched], source_female[0][untouched])

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1 / 240, device=ARGS.device, use_fabric=False,
        physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=32,
                                min_velocity_iteration_count=8,
                                gpu_max_rigid_contact_count=2**18,
                                gpu_max_rigid_patch_count=2**16),
    ))
    sim.set_camera_view(eye=(.20, -.23, .19), target=(.035, 0, 0))
    lamp = Articulation(ArticulationCfg(
        prim_path="/World/Lamp",
        spawn=sim_utils.UsdFileCfg(usd_path=str(OUTPUT)),
        actuators={"turn": ImplicitActuatorCfg(
            joint_names_expr=["screw_turn"], stiffness=.05, damping=.005, effort_limit_sim=.25,
        )},
    ))
    if ARGS.collision_probe:
        # Deliberately put crests against crests. With only the male/female
        # thread colliders enabled, measured forces cannot come from shells.
        stage = sim_utils.get_current_stage()
        stage.GetPrimAtPath("/World/Lamp/Bulb").GetAttribute("xformOp:orient").Set(Gf.Quatf(1))
        stage.GetPrimAtPath("/World/Lamp/Joints/screw_turn").GetAttribute("physics:localRot1").Set(Gf.Quatf(1))
        for name in ("ShellA", "ShellB"):
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/World/Lamp/Bulb/" + name)).GetCollisionEnabledAttr().Set(False)
    sensor = ContactSensor(ContactSensorCfg(prim_path="/World/Lamp/Bulb", update_period=0.0,
                                            filter_prim_paths_expr=["/World/Lamp/Support/InternalThreadCollider"]))
    light = sim_utils.DomeLightCfg(intensity=1500)
    light.func("/World/Light", light)
    contacts = {"max_net_force_n": 0.0, "contact_steps": 0}
    print("THREAD resetting GPU physics", flush=True)
    sim.reset()
    lamp.reset()
    sensor.reset()
    print("THREAD joint_names", lamp.joint_names, "body_names", lamp.body_names, flush=True)
    print("THREAD fixed_base", lamp.is_fixed_base, flush=True)
    assert lamp.is_fixed_base
    turn_id = lamp.joint_names.index("screw_turn")
    slide_id = lamp.joint_names.index("screw_slide")
    assert lamp.num_joints == 2
    records = []
    max_target = -2 * math.pi * TRAVEL / PITCH
    ramp_steps = 600 if ARGS.collision_probe else 2400
    if ARGS.collision_probe:
        max_target = -2 * math.pi

    for cycle in range(ARGS.cycles):
        phases = (("insert", 0.0, max_target),) if ARGS.collision_probe else (("insert", 0.0, max_target), ("remove", max_target, 0.0))
        for phase, start, end in phases:
            for step in range(ramp_steps + 240):
                alpha = min((step + 1) / ramp_steps, 1.0)
                target = start + (end - start) * alpha
                lamp.set_joint_position_target(torch.tensor([[target]], device=sim.device), joint_ids=[turn_id])
                lamp.write_data_to_sim()
                sim.step(render=not ARGS.headless)
                lamp.update(sim.get_physics_dt())
                sensor.update(sim.get_physics_dt())
                force = float(torch.linalg.vector_norm(sensor.data.net_forces_w).item())
                contacts["max_net_force_n"] = max(contacts.get("max_net_force_n", 0.0), force)
                contacts["contact_steps"] += int(force > .01)
                q = lamp.data.joint_pos[0].detach().cpu().numpy()
                assert np.isfinite(q).all(), q
                if step % 120 == 0 or step == ramp_steps + 239:
                    turn, slide = float(q[turn_id]), float(q[slide_id])
                    record = {"cycle": cycle, "phase": phase, "step": step,
                              "target_rad": target, "turn_rad": turn,
                              "slide_m": slide, "extension_m": INITIAL_EXTENSION + slide,
                              "lead_error_m": slide - turn * PITCH / (2 * math.pi)}
                    records.append(record)
                    record["net_force_n"] = force
                    print("THREAD sample", json.dumps(record), flush=True)
            final = records[-1]
            if ARGS.validate and not ARGS.collision_probe:
                assert abs(final["turn_rad"] - end) < .15, final
                assert abs(final["slide_m"] - end * PITCH / (2 * math.pi)) < .0002, final

    max_lead_error = max(abs(r["lead_error_m"]) for r in records)
    report = {"usd": OUTPUT.name, "usd_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              "device": sim.device, "pitch_m": PITCH, "fixed_base": lamp.is_fixed_base,
              "travel_m": TRAVEL, "max_lead_error_m": max_lead_error,
              "source_geometry_preserved": True, "contacts": contacts, "samples": records,
              "test_mode": "misphased_thread_contact" if ARGS.collision_probe else "insert_and_remove"}
    if ARGS.collision_probe:
        assert contacts["max_net_force_n"] > .1, report
        assert abs(records[-1]["turn_rad"] - max_target) > .5, records[-1]
    elif ARGS.validate:
        assert max_lead_error < .0002, report
    if ARGS.report:
        ARGS.report.parent.mkdir(parents=True, exist_ok=True)
        ARGS.report.write_text(json.dumps(report, indent=2) + "\n")
    print("THREAD PASS", json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
    # Let SimulationApp close directly; stopping this GPU articulation with an
    # active contact tensor view can stall the Kit stop callback on Isaac 5.1.


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import os
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    APP.close(skip_cleanup=True)
