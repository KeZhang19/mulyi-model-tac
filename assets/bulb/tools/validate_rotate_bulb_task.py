#!/usr/bin/env python3
"""Check screw attachment, physical release, nearby goals, gravity and partial resets.

Run from the repository root with the brainco environment:
  python assets/bulb/tools/validate_rotate_bulb_task.py --headless --skip-taxim-rgb

The RGB opt-out is for installations without the optional torch_scatter package;
it only affects this validation process, not the task configuration.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


def validate(args):
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude
    from pxr import UsdPhysics, UsdShade

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/BrainCo_DexHand"))
    import BrainCo_DexHand.tasks  # noqa: F401
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_rotate_bulb import (
        DexsuiteRevo3RotateBulbEnvCfg, DexsuiteRevo3RotateBulbEnvCfg_PLAY,
        _SCREW_EXIT_EXTENSION, _SCREW_PITCH, release_unscrewed_bulb,
        rotate_bulb_dropped, rotate_bulb_screw_progress,
    )

    cfg = (DexsuiteRevo3RotateBulbEnvCfg_PLAY if args.play else DexsuiteRevo3RotateBulbEnvCfg)()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # Geometry/physics checks need no goal-frame visual. Avoid downloading its
    # remote USD during headless resets; task defaults remain unchanged.
    cfg.commands.object_pose.debug_vis = False
    if args.turns:
        # Deliberately override AFTER construction, as training's Hydra CLI does.
        cfg.initial_screw_turns_range = tuple(args.turns)
        cfg.episode_length_s = 30.0
    if args.skip_taxim_rgb:
        cfg.observations.proprio.rl_ours_taxim_rgb = None
    env = gym.make("BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0", cfg=cfg).unwrapped
    lamp, bulb, support, table = (env.scene[name] for name in ("lamp", "object", "support", "table"))
    turn, slide = (lamp.joint_names.index(name) for name in ("screw_turn", "screw_slide"))
    ids = torch.arange(env.num_envs, device=env.device)
    records = []
    assert lamp.is_fixed_base and "Bulb" not in lamp.body_names
    assert "ScrewFollower" in lamp.body_names
    assert cfg.curriculum is None and cfg.events.variable_gravity is None
    assert cfg.sim.gravity == (0.0, 0.0, -9.81)
    assert cfg.commands.object_pose.resampling_time_range[0] > cfg.episode_length_s
    assert cfg.rewards.orientation_tracking is None
    assert cfg.observations.policy.bulb_phase is not None
    for i in ids.tolist():
        path = f"/World/envs/env_{i}/Lamp"
        attachment = UsdPhysics.FixedJoint.Get(env.sim.stage, path + "/Joints/bulb_attachment")
        assert attachment.GetExcludeFromArticulationAttr().Get()
        assert not env.sim.stage.GetPrimAtPath(path + "/Bulb").GetAttribute("physxRigidBody:disableGravity").Get()
        for name in ("ShellA", "ShellB", "ExternalThreadCollider"):
            collider = env.sim.stage.GetPrimAtPath(path + "/Bulb/" + name)
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            expected = "ThreadContact" if name == "ExternalThreadCollider" else "GripContact"
            assert material.GetPrim().GetName() == expected
    for name in ("support", "table"):
        prim = env.sim.stage.GetPrimAtPath(env.scene[name].cfg.prim_path.replace(".*", "0"))
        assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()

    def advance(steps=1):
        for _ in range(steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

    def check_reset(selected):
        assert not env._bulb_released[selected].any()
        assert not env._bulb_success[selected].any()
        assert all(env._bulb_attachment_attrs[i].Get() for i in selected.tolist())
        turns = -lamp.data.joint_pos[selected, turn] / (2 * math.pi)
        lo, hi = cfg.initial_screw_turns_range
        assert ((turns >= lo - 1e-5) & (turns <= hi + 1e-5)).all()
        torch.testing.assert_close(rotate_bulb_screw_progress(env)[selected], torch.zeros(len(selected), 2, device=env.device))
        torch.testing.assert_close(lamp.data.joint_pos[selected, slide], -turns * _SCREW_PITCH, atol=1e-6, rtol=0)
        torch.testing.assert_close(bulb.data.root_pos_w[selected, 2] - support.data.root_pos_w[selected, 2],
                                   _SCREW_EXIT_EXTENSION - turns * _SCREW_PITCH, atol=1e-5, rtol=0)
        torch.testing.assert_close(support.data.root_pos_w[selected, 2] - table.data.root_pos_w[selected, 2],
                                   torch.full_like(turns, cfg.scene.table.spawn.size[2] / 2 + .044732484966516495),
                                   atol=1e-5, rtol=0)
        assert lamp.data.joint_effort_target[selected].abs().max() == 0
        assert lamp.data.joint_vel_target[selected].abs().max() == 0

    # Startup reset, then repeated ordinary resets; these must use identical rules.
    check_reset(ids)
    from tabletop_validation import TabletopGeometry, export_live_scene

    geometry = TabletopGeometry(env)
    assert env.scene["robot"].is_fixed_base
    geometry.check(ids)
    initial_locations = support.data.root_pos_w.clone()
    samples = []
    for _ in range(20):
        env.reset()
        check_reset(ids)
        geometry.check(ids)
        samples.append((support.data.root_pos_w - env.scene.env_origins).clone())
    assert (support.data.root_pos_w - initial_locations).norm(dim=-1).min() > 1e-5
    sampled = torch.stack(samples)
    assert sampled[..., :2].std(dim=0).min() > .001
    # Robot joint defaults are applied by env.reset(), whereas the socket also
    # has a startup event. Compare/export the actual task reset posture.
    records.append(dict(phase="preserved_hand_pose", **geometry.compare_original_hand_pose()))
    if args.scene_export:
        records.append(dict(phase="scene_geometry", **export_live_scene(env, args.scene_export)))
    # Exercise the real reset event at all four region corners, including its
    # table-local to environment-local translation. Restore normal sampling.
    params = env.event_manager.get_term_cfg("reset_object").params
    x_range, y_range = params["x_range"], params["y_range"]
    try:
        for x in x_range:
            for y in y_range:
                params["x_range"], params["y_range"] = (x, x), (y, y)
                env.reset()
                check_reset(ids)
                geometry.check(ids)
                expected = torch.tensor([x, y], device=env.device)
                torch.testing.assert_close(support.data.root_pos_w[:, :2] - table.data.root_pos_w[:, :2],
                                           expected.expand(env.num_envs, -1), atol=1e-5, rtol=0)
    finally:
        params["x_range"], params["y_range"] = x_range, y_range
    env.reset()
    records.append(dict(phase="tabletop_resets", full_resets=20, corners=4,
                        max_bottom_error_m=geometry.max_bottom_error, min_edge_margin_m=geometry.min_edge_margin,
                        sampled_min=sampled.amin(dim=(0, 1)).cpu().tolist(),
                        sampled_max=sampled.amax(dim=(0, 1)).cpu().tolist()))
    command = env.command_manager.get_term("object_pose")
    bounds = torch.tensor([cfg.terminations.object_out_of_bound.params["in_bound_range"][axis]
                           for axis in ("x", "y", "z")], device=env.device)
    max_goal_distance = 0.0
    for _ in range(50):
        command._resample_command(ids)
        goal, _ = combine_frame_transforms(env.scene["robot"].data.root_pos_w,
                                           env.scene["robot"].data.root_quat_w, command.command[:, :3])
        offset = goal - support.data.root_pos_w
        assert (offset[:, :2].abs() <= .04001).all()
        assert ((offset[:, 2] >= .11999) & (offset[:, 2] <= .16001)).all()
        local = goal - env.scene.env_origins
        assert ((local > bounds[:, 0] + .05) & (local < bounds[:, 1] - .05)).all()
        max_goal_distance = max(max_goal_distance, float((goal - bulb.data.root_pos_w).norm(dim=-1).max()))
    assert max_goal_distance < .18
    records.append(dict(phase="goals", samples=50 * env.num_envs, max_transport_m=max_goal_distance))

    # Isolate the fixture mechanics from accidental contact with an uncommanded
    # hand. Ordinary task resets below restore the actual robot start pose.
    robot = env.scene["robot"]
    parked = robot.data.root_pose_w.clone()
    parked[:, 1] += 1.0
    robot.write_root_pose_to_sim(parked)
    # Shallow initial states must not screw themselves fully in under gravity
    # while the policy is still approaching the object.
    initial_turns = lamp.data.joint_pos[:, turn].clone() / (2 * math.pi)
    advance(1200)
    idle_drift = float((lamp.data.joint_pos[:, turn] / (2 * math.pi) - initial_turns).abs().max())
    assert idle_drift < .02, idle_drift
    records.append(dict(phase="passive_hold", seconds=5, max_drift_turns=idle_drift))

    # A lateral impulse before release must NOT bypass the screw constraint.
    initial_bulb = bulb.data.root_pose_w.clone()
    velocity = torch.zeros(env.num_envs, 6, device=env.device)
    velocity[:, 0] = .3
    bulb.write_root_velocity_to_sim(velocity)
    advance(24)
    assert not env._bulb_released.any()
    lateral_drift = float((bulb.data.root_pos_w[:, :2] - initial_bulb[:, :2]).norm(dim=-1).max())
    assert lateral_drift < .001, lateral_drift
    records.append(dict(phase="attached", max_lateral_drift_m=lateral_drift))

    # Controlled torque tests mechanical capability, not learned hand behavior.
    initial_support = support.data.root_pose_w.clone()
    max_lead_error = 0.0
    support_force = torch.zeros(env.num_envs, 1, 3, device=env.device)
    lamp.set_joint_effort_target(torch.full((env.num_envs, 1), .02, device=env.device), joint_ids=[turn])
    for _ in range(600):
        advance(cfg.decimation)
        attached = ~env._bulb_released
        error = lamp.data.joint_pos[:, slide] - lamp.data.joint_pos[:, turn] * _SCREW_PITCH / (2 * math.pi)
        max_lead_error = max(max_lead_error, float(error.abs().max()))
        # At the exit the lowest external-thread collider point clears the mouth.
        ready = (rotate_bulb_screw_progress(env) >= .999).all(dim=-1) & attached
        for i in ready.nonzero(as_tuple=False).flatten().tolist():
            mesh = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Lamp/Bulb/ExternalThreadCollider")
            points = torch.tensor(mesh.GetAttribute("points").Get(), device=env.device)
            points_w = quat_apply(bulb.data.root_quat_w[i].expand(len(points), -1), points) + bulb.data.root_pos_w[i]
            assert points_w[:, 2].min() >= support.data.root_pos_w[i, 2] - .0001
        release_unscrewed_bulb(env)
        # Support the weight of released bulbs while waiting for the others.
        # This test-only wrench is removed before checking gravity/free motion.
        released_ids = ready.nonzero(as_tuple=False).flatten()
        if len(released_ids):
            release_velocity = torch.zeros(len(released_ids), 6, device=env.device)
            bulb.write_root_velocity_to_sim(release_velocity, env_ids=released_ids)
            support_force[:, 0, 2] = bulb.data.default_mass[:, 0].to(env.device) * 9.81 * env._bulb_released
            bulb.set_external_force_and_torque(support_force, torch.zeros_like(support_force), is_global=True)
        if env._bulb_released.all():
            break
    assert env._bulb_released.all(), lamp.data.joint_pos
    assert all(not attr.Get() for attr in env._bulb_attachment_attrs)
    assert max_lead_error < .0002
    assert (support.data.root_pos_w - initial_support[:, :3]).norm(dim=-1).max() < 1e-5
    assert quat_error_magnitude(support.data.root_quat_w, initial_support[:, 3:]).max() < 1e-5
    lamp.set_joint_effort_target(torch.zeros_like(lamp.data.joint_pos))
    bulb.set_external_force_and_torque(torch.zeros_like(support_force), torch.zeros_like(support_force), is_global=True)
    records.append(dict(phase="released", max_lead_error_m=max_lead_error))

    # Released bulbs retain independent velocity, including normal gravity.
    initial_bulb = bulb.data.root_pos_w.clone()
    velocity.zero_(); velocity[:, 0] = .3; velocity[:, 2] = 1.0
    bulb.write_root_velocity_to_sim(velocity)
    advance(20)
    displacement = bulb.data.root_pos_w - initial_bulb
    assert (displacement[:, 0] > .02).all(), displacement
    assert (displacement[:, 2] > .02).all(), displacement
    torch.testing.assert_close(bulb.data.root_lin_vel_w[:, 2],
                               torch.full((env.num_envs,), 1.0 - 9.81 * 20 * env.physics_dt, device=env.device),
                               atol=.04, rtol=0)
    records.append(dict(phase="free_motion", min_lateral_m=float(displacement[:, 0].min()),
                        min_rise_m=float(displacement[:, 2].min())))

    # Partial reset reconnects one subset, without changing the other's state.
    reset_ids, untouched = ids[::2], ids[1::2]
    before = bulb.data.root_pose_w[untouched].clone()
    goal_before = command.command[untouched].clone()
    support_before = support.data.root_pose_w[untouched].clone()
    robot_before = robot.data.root_pose_w[untouched].clone()
    table_before = table.data.root_pose_w[untouched].clone()
    lamp.set_joint_effort_target(torch.ones_like(lamp.data.joint_pos[reset_ids]), env_ids=reset_ids)
    env._reset_idx(reset_ids)
    check_reset(reset_ids)
    geometry.check(reset_ids)
    assert env._bulb_released[untouched].all()
    torch.testing.assert_close(bulb.data.root_pose_w[untouched], before)
    torch.testing.assert_close(command.command[untouched], goal_before)
    torch.testing.assert_close(support.data.root_pose_w[untouched], support_before)
    torch.testing.assert_close(robot.data.root_pose_w[untouched], robot_before)
    torch.testing.assert_close(table.data.root_pose_w[untouched], table_before)
    before = bulb.data.root_pos_w[reset_ids].clone()
    bulb.write_root_velocity_to_sim(velocity[reset_ids], env_ids=reset_ids)
    advance(20)
    assert (bulb.data.root_pos_w[reset_ids, :2] - before[:, :2]).norm(dim=-1).max() < .001
    records.append(dict(phase="partial_reset", restored=int(len(reset_ids)), untouched=int(len(untouched))))

    # Let the released subset fall away from the socket onto the table.
    velocity.zero_(); velocity[:, 0] = -.4; velocity[:, 2] = -.2
    bulb.write_root_velocity_to_sim(velocity[untouched], env_ids=untouched)
    saw_drop = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for _ in range(120):
        advance()
        saw_drop |= rotate_bulb_dropped(env)
    assert saw_drop[untouched].all(), saw_drop
    assert not saw_drop[reset_ids].any()
    records.append(dict(phase="drop_detection", dropped=int(saw_drop.sum())))

    env.reset()
    check_reset(ids)
    for _ in range(5):
        obs, reward, *_ = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert not command.metrics["success"].any()
    report = dict(passed=True, num_envs=env.num_envs, play=args.play,
                  initial_turns=list(cfg.initial_screw_turns_range), episode_s=cfg.episode_length_s,
                  taxim_rgb_skipped=args.skip_taxim_rgb, phases=records)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("BULB SCENE PASS", json.dumps(report), flush=True)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--turns", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--skip-taxim-rgb", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scene-export", type=Path, help="Export initial live scene geometry as GLB for visual checks.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or not args.device.startswith("cuda"):
        parser.error("Use at least two environments and a CUDA device for SDF contacts.")
    app = AppLauncher(args).app
    try:
        validate(args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
