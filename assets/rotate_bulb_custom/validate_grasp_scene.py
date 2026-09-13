#!/usr/bin/env python3
"""Compare the live custom task's startup/reset states with its grasp export.

Run from the repository root in the Isaac Lab environment::

    python assets/rotate_bulb_custom/validate_grasp_scene.py --headless --num_envs 2

The default geometry-only mode retains the configured scene, sensors, reset
events and native screw dynamics, but keeps only joint position/velocity
observations. Use --full-observations to also initialize the tactile encoder.
No physics rollout or grasp-stability claim is made by this state check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
DEFAULT_TASK = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0"


def validate_grasp_scene(args, report):
    """Create the registered task, compare all environments, then partial reset."""
    import gymnasium as gym
    import torch
    from isaaclab.managers import ObservationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
    import BrainCo_DexHand.tasks  # noqa: F401

    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    if reference.get("schema") != "revo3_simulation_state":
        raise ValueError("Expected a revo3_simulation_state reference export")
    report.update(
        reference=str(args.reference.resolve()),
        reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        source_validation=reference["validation"],
        geometry_only=not args.full_observations,
        physics_rollout_validated=False,
        removed_observations=[],
        checks=[],
    )
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # The command's remote frame USD is only a debug visual; no task geometry
    # or reset parameter is changed by disabling its visualization here.
    cfg.commands.object_pose.debug_vis = False
    if not args.full_observations:
        for group_name, group in list(vars(cfg.observations).items()):
            if group is None:
                continue
            if group_name != "proprio":
                setattr(cfg.observations, group_name, None)
                report["removed_observations"].append(group_name + ".*")
                continue
            for term_name, term in list(vars(group).items()):
                if isinstance(term, ObservationTermCfg) and term_name not in {"joint_pos", "joint_vel"}:
                    setattr(group, term_name, None)
                    report["removed_observations"].append(group_name + "." + term_name)
        cfg.events.zz_initialize_tactile_contract = None

    def tensor(value):
        return torch.as_tensor(value, device=args.device, dtype=torch.float64)

    def compare(name, actual, expected, tolerance):
        actual = torch.as_tensor(actual, dtype=torch.float64, device=args.device)
        expected = tensor(expected)
        error = float((actual - expected).abs().max())
        report["checks"].append(dict(name=name, max_absolute_error=error, tolerance=tolerance,
                                     passed=bool(torch.isfinite(actual).all()) and error <= tolerance))

    def compare_pose(name, actual, expected, origins):
        actual = actual.to(torch.float64)
        target_pos = tensor(expected["position_m"])
        target_quat = tensor(expected["quaternion_wxyz"])
        compare(name + ".position_m", actual[:, :3] - origins, target_pos, args.position_tolerance)
        actual_quat = actual[:, 3:7] / actual[:, 3:7].norm(dim=-1, keepdim=True)
        target_quat = target_quat / target_quat.norm()
        dot = (actual_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
        angle = float((2.0 * torch.acos(dot)).max())
        report["checks"].append(dict(name=name + ".rotation_rad", max_absolute_error=angle,
                                     tolerance=args.rotation_tolerance,
                                     passed=bool(torch.isfinite(actual_quat).all()) and angle <= args.rotation_tolerance))

    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        robot = env.scene["robot"]
        ids = torch.arange(env.num_envs, device=env.device)
        expected_joints = dict(zip(reference["robot"]["joint_names"], reference["robot"]["joint_positions_rad"], strict=True))
        if set(robot.joint_names) != set(expected_joints):
            raise AssertionError(f"Joint name sets differ: actual={robot.joint_names}, reference={list(expected_joints)}")
        joint_values = tensor([expected_joints[name] for name in robot.joint_names])
        fixture = next(item for item in reference["fixtures"] if item["name"] == "lamp_support")
        root_poses = {"robot": reference["robot"]["base"], "table": reference["table"],
                      "support": fixture, "lamp": fixture, "object": reference["object"]}
        report["joint_names_in_simulator_order"] = list(robot.joint_names)
        report["expected_joint_positions_rad_in_simulator_order"] = joint_values.cpu().tolist()
        report["sensor_names"] = list(env.scene.sensors)
        report["environment_origins_m"] = env.scene.env_origins.cpu().tolist()
        report["tolerances"] = dict(position_m=args.position_tolerance,
                                    rotation_rad=args.rotation_tolerance, joint_rad=args.joint_tolerance)

        compare("configuration.table.size_m", cfg.scene.table.spawn.size, reference["table"]["size_m"], 1e-10)
        for name, expected in root_poses.items():
            if name not in {"support", "object"}:
                asset_cfg = getattr(cfg.scene, name)
                compare_pose("configuration." + name, tensor([(*asset_cfg.init_state.pos, *asset_cfg.init_state.rot)]),
                             expected, tensor([[0.0, 0.0, 0.0]]))
        compare("configuration.robot.default_joint_positions_rad", robot.data.default_joint_pos,
                joint_values, args.joint_tolerance)

        def refresh_kinematics():
            env.scene.write_data_to_sim()
            # Update link poses without advancing dynamics/contact solving.
            env.sim.physics_sim_view.update_articulations_kinematic()
            env.sim.forward()

        def check_state(phase, selected):
            refresh_kinematics()
            origins = env.scene.env_origins[selected].to(torch.float64)
            for name, expected in root_poses.items():
                compare_pose(phase + "." + name, env.scene[name].data.root_link_pose_w[selected], expected, origins)
            compare(phase + ".robot.joint_positions_rad", robot.data.joint_pos[selected], joint_values, args.joint_tolerance)
            compare(phase + ".robot.joint_velocities_rad_s", robot.data.joint_vel[selected], 0.0, args.joint_tolerance)
            compare(phase + ".robot.joint_position_targets_rad", robot.data.joint_pos_target[selected], joint_values,
                    args.joint_tolerance)
            source_screw = reference["assembly"]["source_configuration"]
            for joint_name, expected in (("screw_turn", source_screw["screw_turn_rad"]),
                                         ("screw_slide", source_screw["screw_slide_m"])):
                joint_id = env.scene["lamp"].joint_names.index(joint_name)
                compare(phase + ".lamp." + joint_name, env.scene["lamp"].data.joint_pos[selected, joint_id],
                        expected, args.joint_tolerance)
            links = robot.data.body_link_pose_w
            checked_links, unavailable_links = [], []
            for link in reference["robot"]["links"]:
                name = link["body_name"].removeprefix("hand-")
                if name not in robot.body_names:
                    unavailable_links.append(name)
                    continue
                index = robot.body_names.index(name)
                compare_pose(phase + ".link." + name, links[selected, index], link, origins)
                checked_links.append(name)
            report["checked_robot_links"] = checked_links
            report["unavailable_reference_links"] = unavailable_links
            if not any("DIP" in name for name in checked_links):
                raise AssertionError("No fingertip DIP body frames were checked")
            report.setdefault("phases", []).append(dict(name=phase, env_ids=selected.cpu().tolist()))

        check_state("startup", ids)
        for reset_index in range(3):
            env.reset()
            check_state(f"full_reset_{reset_index + 1}", ids)

        # Perturb both groups without stepping physics. A partial reset must
        # recover the reference in one group while preserving the other group.
        selected, untouched = ids[::2], ids[1::2]
        shift = (ids.to(torch.float32) + 1.0) * 0.003
        for name in root_poses:
            asset = env.scene[name]
            pose = asset.data.root_link_pose_w.clone()
            pose[:, 0] += shift
            asset.write_root_pose_to_sim(pose)
        perturbed_joints = robot.data.joint_pos.clone()
        perturbed_joints[:, 0] += 0.01
        robot.write_joint_state_to_sim(perturbed_joints, torch.zeros_like(perturbed_joints))
        robot.set_joint_position_target(perturbed_joints)
        refresh_kinematics()
        before = {name: env.scene[name].data.root_link_pose_w[untouched].clone() for name in root_poses}
        before_joints = robot.data.joint_pos[untouched].clone()
        before_targets = robot.data.joint_pos_target[untouched].clone()
        env._reset_idx(selected)
        check_state("partial_reset", selected)
        for name, expected in before.items():
            compare("partial_reset.untouched." + name, env.scene[name].data.root_link_pose_w[untouched], expected, 1e-7)
        compare("partial_reset.untouched.robot.joints", robot.data.joint_pos[untouched], before_joints, 1e-7)
        compare("partial_reset.untouched.robot.targets", robot.data.joint_pos_target[untouched], before_targets, 1e-7)
        failures = [entry for entry in report["checks"] if not entry["passed"]]
        report["failed_checks"] = failures
        if failures:
            raise AssertionError(f"{len(failures)} alignment checks failed; inspect the JSON report")
        report["passed"] = True
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=2)
    parser.add_argument("--reference", type=Path, default=ASSETS / "grasp_reference/simulation_state.json")
    parser.add_argument("--report", type=Path, default=ASSETS / "grasp_reference/runtime_validation.json")
    parser.add_argument("--full-observations", action="store_true")
    parser.add_argument("--position-tolerance", type=float, default=2e-4)
    parser.add_argument("--rotation-tolerance", type=float, default=2e-3)
    parser.add_argument("--joint-tolerance", type=float, default=2e-5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("Use at least two environments to verify partial resets")
    if not args.device.startswith("cuda"):
        parser.error("Use a CUDA device for the task's native SDF contacts")
    if min(args.position_tolerance, args.rotation_tolerance, args.joint_tolerance) <= 0.0:
        parser.error("Tolerances must be positive")
    report = dict(task=args.task, num_envs=args.num_envs,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), passed=False)
    app = AppLauncher(args).app
    try:
        validate_grasp_scene(args, report)
    except Exception as exc:
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"GRASP SCENE {'PASS' if report['passed'] else 'FAIL'}: {args.report}", flush=True)
        # Kit shutdown otherwise replaces a failed validation's exit code by 0.
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(0 if report["passed"] else 1)
        app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
