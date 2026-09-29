"""Opt-in, versioned fast physics for the randomized bulb task.

Run from a frozen workspace with BULB_FAST_CONFIG pointing to a case JSON.
The task's actions, observations, rewards, resets and success terms are retained.
Checkpoint transfer permits only a declared change of physics decimation while
holding the control period fixed; all other interface fields must match exactly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path.cwd()
CASE = json.loads(Path(os.environ["BULB_FAST_CONFIG"]).read_text())
OUT = Path(CASE["output"])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
import gymnasium as gym

ORIGINAL_MAKE = gym.make


def configure_fast_train(task, **kwargs):
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
    import duration_logging
    from bulb_training import prepare_bulb_policy_run
    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom import env_cfg as bulb

    assert task == "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0"
    cfg = kwargs["cfg"]
    assert not cfg.bulb_pure_collision and not cfg.tactile_policy_enabled
    assert tuple(cfg.initial_screw_turns_range) == (3., 3.)
    physics = CASE["physics"]
    hz = int(physics["hz"])
    assert hz in (60, 90, 120, 240)
    cfg.sim.dt, cfg.decimation = 1 / hz, hz // 30
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.min_position_iteration_count = cfg.sim.physx.max_position_iteration_count = int(physics["position"])
    cfg.sim.physx.min_velocity_iteration_count = cfg.sim.physx.max_velocity_iteration_count = int(physics["velocity"])
    cfg.sim.physx.gpu_max_num_partitions = int(physics.get("partitions", 8))
    changed = []
    original_reset = SimulationContext.reset
    applied = False

    def reset_with_geometry(sim, *args, **kw):
        nonlocal applied
        if not applied:
            applied = True
            for i in range(cfg.scene.num_envs):
                base = f"/World/envs/env_{i}"
                if physics.get("palm_box", False):
                    bounds = physics["palm_box_bounds_m"]
                    owner = sim.stage.GetPrimAtPath(base + "/Robot/right_base_link")
                    disabled = 0
                    for prim in Usd.PrimRange(owner):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                            disabled += 1
                    assert disabled == 28, "The audited palm collision topology changed"
                    cube = UsdGeom.Cube.Define(sim.stage, str(owner.GetPath()) + "/FastPalmCollider")
                    cube.CreateSizeAttr(1.)
                    cube.AddTranslateOp().Set(Gf.Vec3d(*[(a+b)/2 for a, b in zip(*bounds)]))
                    cube.AddScaleOp().Set(Gf.Vec3d(*[b-a for a, b in zip(*bounds)]))
                    cube.CreateVisibilityAttr("invisible")
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                    collision.CreateContactOffsetAttr(.0002)
                    collision.CreateRestOffsetAttr(-.0001)
                    changed.append(str(cube.GetPath()))
                if not physics.get("self_collision", True):
                    for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(base + "/Robot")):
                        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
                            changed.append(str(prim.GetPath()))
                meshes = (["ShellA", "ShellB"] if physics.get("shell_convex", False) else [])
                if physics.get("thread_convex", False):
                    meshes.append("ExternalThreadCollider")
                for name in meshes:
                    prim = sim.stage.GetPrimAtPath(base + "/Lamp/Bulb/" + name)
                    assert prim and prim.HasAPI(UsdPhysics.CollisionAPI)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                    prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
                    changed.append(str(prim.GetPath()))
        return original_reset(sim, *args, **kw)

    SimulationContext.reset = reset_with_geometry
    try:
        env = ORIGINAL_MAKE(task, **kwargs)
    finally:
        SimulationContext.reset = original_reset
    raw = env.unwrapped
    assert abs(raw.step_dt - 1 / 30) < 1e-12 and abs(raw.max_episode_length_s - 20) < 1e-9
    assert sum(raw.observation_manager.group_obs_dim[g][0] for g in ("policy", "proprio", "perception")) == 687
    scene = PhysxSchema.PhysxSceneAPI(raw.sim.stage.GetPrimAtPath(cfg.sim.physics_prim_path))
    limits = [scene.GetMinPositionIterationCountAttr().Get(), scene.GetMaxPositionIterationCountAttr().Get(),
              scene.GetMinVelocityIterationCountAttr().Get(), scene.GetMaxVelocityIterationCountAttr().Get()]
    assert limits == [physics["position"]] * 2 + [physics["velocity"]] * 2
    for name, sensor in raw.scene.sensors.items():
        if "warpsdf" in name:
            assert abs(sensor.cfg.update_period - 1 / 30) < 1e-12
    success_params = raw.reward_manager.get_term_cfg("success").params
    assert success_params["pos_std"] == .03 and success_params["hold_duration"] == 1.
    manifest = {"schema": "bulb_fast_train_physics", "version": 1, "physics": physics,
        "task": task, "num_envs": raw.num_envs, "physics_dt": raw.physics_dt, "decimation": cfg.decimation,
        "control_dt": raw.step_dt, "episode_s": raw.max_episode_length_s, "solver_limits": limits,
        "gpu_max_num_partitions": scene.GetGpuMaxNumPartitionsAttr().Get(), "changed_prim_count": len(changed),
        "robot_native_shapes_per_env": raw.scene["robot"].root_physx_view.max_shapes,
        "bulb_native_shapes_per_env": raw.scene["object"].root_physx_view.max_shapes,
        "changed_prim_examples": changed[:5], "observation_dim": 687, "action_dim": 28,
        "success": {"required_turns": 3, "released": True, "goal_distance_m": .03, "hold_s": 1.},
        "reward_weights": {n: raw.reward_manager.get_term_cfg(n).weight for n in raw.reward_manager.active_terms},
        "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_checkpoint": CASE["checkpoint"], "physics_approximation_explicit": True}
    if physics.get("palm_box", False):
        assert manifest["robot_native_shapes_per_env"] == 31, "Expected 58 - 28 + 1 robot shapes"
    (OUT / "physics.json").write_text(json.dumps(manifest, indent=2))
    (Path(cfg.log_dir) / "fast_train_physics.json").write_text(json.dumps(manifest, indent=2))

    robot, lamp, obj = (raw.scene[n] for n in ("robot", "lamp", "object"))
    turn, slide = (lamp.joint_names.index(n) for n in ("screw_turn", "screw_slide"))
    follower = lamp.body_names.index("ScrewFollower")
    peak = torch.zeros(raw.num_envs, device=raw.device)
    released = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    goal_hold = torch.zeros_like(peak)
    active = torch.ones_like(released)
    counts = torch.zeros(16, device=raw.device, dtype=torch.float64)
    bulb_turns = torch.zeros_like(peak)
    previous_angle = torch.zeros_like(peak)
    initial_axial = torch.zeros_like(peak)
    failure_counts = {n: torch.zeros((), device=raw.device) for n in raw.termination_manager.active_terms}
    # Per-episode measurements run before automatic resets. No writes to physical state.
    original_termination = raw.termination_manager.compute

    def measure_episodes(*args, **kw):
        done = original_termination(*args, **kw)
        eligible = active if CASE.get("mode") == "eval" else torch.ones_like(active)
        physical_turns = bulb.rotate_bulb_unscrewed_turns(raw)
        peak.copy_(torch.maximum(peak, torch.where(~released, physical_turns, peak)))
        first_release = raw._bulb_released & ~released
        local_pos, local_quat = subtract_frame_transforms(
            lamp.data.root_pos_w, lamp.data.root_quat_w, obj.data.root_pos_w, obj.data.root_quat_w)
        angle = 2 * torch.atan2(local_quat[:, 1], local_quat[:, 0])
        first_step = raw.episode_length_buf <= 1
        previous_angle.copy_(torch.where(first_step, angle, previous_angle))
        initial_axial.copy_(torch.where(first_step, local_pos[:, 0], initial_axial))
        delta = torch.remainder(angle - previous_angle + math.pi, 2 * math.pi) - math.pi
        bulb_turns.add_(torch.where(~released, delta / (2 * math.pi), 0.))
        previous_angle.copy_(angle)
        released_now = first_release & eligible
        counts[12].add_(torch.where(released_now, bulb_turns, 0.).sum())
        counts[13].add_(released_now.sum())
        counts[14].add_(torch.where(released_now, local_pos[:, 0] - initial_axial, 0.).sum())
        counts[15].add_(torch.where(released_now,
            (obj.data.root_pos_w - lamp.data.body_pos_w[:, follower]).norm(dim=-1), 0.).sum())
        counts[8].add_((first_release & ~bulb.rotate_bulb_screw_complete(raw) & eligible).sum())
        released.logical_or_(raw._bulb_released)
        command = raw.command_manager.get_command("object_pose")
        goal, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        distance = (obj.data.root_pos_w - goal).norm(dim=-1)
        valid_goal = (distance < .03) & raw._bulb_released & ~bulb.rotate_bulb_failed(raw)
        goal_hold.copy_(torch.where(valid_goal, goal_hold + raw.step_dt, 0.))
        success = raw.termination_manager.get_term("success")
        counts[9].add_((success & (goal_hold + 1e-6 < 1.) & eligible).sum())
        finite = torch.isfinite(robot.data.joint_pos).all(-1) & torch.isfinite(robot.data.joint_vel).all(-1)
        finite &= torch.isfinite(obj.data.root_state_w).all(-1) & torch.isfinite(lamp.data.joint_pos).all(-1)
        counts[7].add_((~finite & eligible).sum())
        counts[10].add_(((obj.data.root_pos_w - raw.scene.env_origins).abs().amax(-1) > 10).logical_and(eligible).sum())
        forces = bulb.rotate_bulb_finger_forces(raw)
        counts[5].add_(((forces.amax(-1) > .1) & eligible).sum())
        counts[6].add_(eligible.sum())
        counts[11].add_((valid_goal & eligible).sum())
        ended = done & eligible
        counts[0].add_(ended.sum())
        counts[1].add_((success & ended).sum())
        counts[2].add_((released & ended).sum())
        counts[3].add_(torch.where(ended, peak, 0.).sum())
        counts[4].add_((ended & (peak >= 2.)).sum())
        for name, value in failure_counts.items():
            value.add_((raw.termination_manager.get_term(name) & ended).sum())
        active.logical_and_(~done)
        peak.masked_fill_(done, 0.)
        released.masked_fill_(done, False)
        goal_hold.masked_fill_(done, 0.)
        bulb_turns.masked_fill_(done, 0.)
        return done

    raw.termination_manager.compute = measure_episodes

    def statistics():
        vals = counts.tolist()
        return dict(zip(("episodes", "successes", "released_episodes", "sum_max_turns", "reached_two_turns",
            "contact_env_steps", "env_steps", "nonfinite_env_steps", "premature_releases", "invalid_successes",
            "outlier_env_steps", "goal_env_steps", "sum_bulb_turns_at_release", "release_events",
            "sum_bulb_axial_travel_at_release_m", "sum_attachment_error_at_release_m"), vals),
            terminations={n: float(v) for n, v in failure_counts.items()},
            bulb_speed_max_m_s=float(obj.data.root_lin_vel_w.norm(dim=-1).max()),
            attachment_position_max_m=float((obj.data.root_pos_w - lamp.data.body_pos_w[:, follower]).norm(dim=-1).max()))

    cls = duration_logging.DurationLoggingOnPolicyRunner
    original_init, original_log, original_learn = cls.__init__, cls.log, cls.learn
    records = []
    began = None

    class BudgetReached(Exception):
        pass

    def initialize(runner, *args, **kw):
        original_init(runner, *args, **kw)
        checkpoint = Path(CASE["checkpoint"])
        prepare_policy_run(raw, cfg.log_dir, resume_path=str(checkpoint))
        expected = prepare_bulb_policy_run(raw, cfg.log_dir)
        saved = json.loads((checkpoint.parent / "bulb_action_contract.json").read_text())
        comparable = copy.deepcopy(saved)
        old_decimation = comparable["action"]["decimation"]
        comparable["action"]["decimation"] = expected["action"]["decimation"]
        assert comparable == expected, "Only declared physics decimation may differ during transfer"
        assert abs(saved["action"]["control_dt"] - raw.step_dt) < 1e-12
        runner.load(str(checkpoint))
        rates = [g["lr"] for g in runner.alg.optimizer.param_groups]
        assert len(set(rates)) == 1
        runner.alg.learning_rate = rates[0]
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = runner.alg.policy.state_dict()
        assert all(torch.equal(v.cpu(), loaded["model_state_dict"][n]) for n, v in model.items())
        pairs = [(loaded["optimizer_state_dict"], runner.alg.optimizer.state_dict())]
        while pairs:
            saved_value, actual = pairs.pop()
            if isinstance(saved_value, torch.Tensor):
                assert torch.equal(saved_value, actual.detach().cpu()) and bool(torch.isfinite(actual).all())
            elif isinstance(saved_value, dict):
                assert saved_value.keys() == actual.keys()
                pairs.extend((value, actual[name]) for name, value in saved_value.items())
            elif isinstance(saved_value, (list, tuple)):
                assert len(saved_value) == len(actual)
                pairs.extend(zip(saved_value, actual))
            else:
                assert saved_value == actual
        transfer = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "loaded_iteration": runner.current_learning_iteration, "model_normalizer_exact": True,
            "optimizer_exact": True, "adaptive_lr": rates[0], "decimation_from": old_decimation,
            "decimation_to": cfg.decimation, "allowed_contract_difference": "action.decimation only",
            "parent_physics": json.loads((checkpoint.parent / "fast_train_physics.json").read_text())
                if (checkpoint.parent / "fast_train_physics.json").exists() else "original 240Hz TGS24/4"}
        (OUT / "transfer.json").write_text(json.dumps(transfer, indent=2))
        print("[FAST_TRANSFER]", json.dumps(transfer), flush=True)

    def log(runner, locs, *args, **kw):
        st = statistics()
        if st["nonfinite_env_steps"] or st["invalid_successes"] or st["premature_releases"]:
            raise RuntimeError("Invalid fast simulation state or success semantics: " + json.dumps(st))
        row = {k: float(locs[k]) for k in ("it", "collection_time", "learn_time")}
        row.update(relative_iteration=len(records), training_wall_s=time.monotonic() - began,
                   timestamp=time.time(), statistics=st)
        records.append(row)
        with (OUT / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        answer = original_log(runner, locs, *args, **kw)
        if CASE.get("train_seconds") and row["training_wall_s"] >= CASE["train_seconds"]:
            runner.current_learning_iteration = int(locs["it"])
            runner.save(str(Path(cfg.log_dir) / f"model_{int(locs['it'])}.pt"))
            raise BudgetReached()
        return answer

    def learn(runner, *args, **kw):
        nonlocal began
        if CASE.get("mode") == "eval":
            raw.reset(seed=raw.cfg.seed)
            counts.zero_(); peak.zero_(); released.zero_(); goal_hold.zero_(); active.fill_(True)
            bulb_turns.zero_(); previous_angle.zero_(); initial_axial.zero_()
            for value in failure_counts.values():
                value.zero_()
            policy = runner.get_inference_policy(device=raw.device)
            obs = runner.env.get_observations()
            before = {n: v.clone() for n, v in runner.alg.policy.state_dict().items()}
            with torch.inference_mode():
                for _ in range(raw.max_episode_length + 2):
                    if not bool(active.any()):
                        break
                    obs, *_ = runner.env.step(policy(obs))
            assert all(torch.equal(v, before[n]) for n, v in runner.alg.policy.state_dict().items())
            result = {"completed": not bool(active.any()), "mode": "eval", "statistics": statistics(),
                      "policy_unchanged": True, "num_envs": raw.num_envs}
        else:
            began = time.monotonic()
            try:
                original_learn(runner, *args, **kw)
            except BudgetReached:
                pass
            result = {"completed": True, "mode": "train", "iterations": len(records),
                      "training_wall_s": time.monotonic() - began, "statistics": statistics()}
        (OUT / "result.json").write_text(json.dumps(result, indent=2))
        if runner.writer is not None:
            runner.writer.flush()
        return None

    cls.__init__, cls.log, cls.learn = initialize, log, learn
    return env


gym.make = configure_fast_train
sys.argv = [str(ROOT / "scripts/rsl_rl/train.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
