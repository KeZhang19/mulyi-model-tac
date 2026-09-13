#!/usr/bin/env python3
"""Calibrate and measure an unconstrained, gravity-loaded D-peg grasp in PhysX.

This offline tool may step the simulator. The task reset never calls this tool.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/d_peg_insertion"
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))


def main(args):
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.optimize import least_squares
    from isaaclab.utils.math import quat_error_magnitude
    from pxr import UsdPhysics
    import BrainCo_DexHand.tasks  # noqa
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_insert_d_peg import DexsuiteRevo3InsertDPegEnvCfg
    from solve_pregrasp import Kinematics

    cfg = DexsuiteRevo3InsertDPegEnvCfg(pregrasp_path=str(args.seed))
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.events.zz_initialize_tactile_contract = None
    cfg.observations.proprio = None
    cfg.observations.perception = None
    # Keep ordinary contact sensing and precisely the production robot material.
    for name, value in list(vars(cfg.scene).items()):
        if name.endswith(("_warpsdf_s", "_tacmap_surface_s", "_tacmap_object_s", "_hydroshear_marker_surface_s", "_hydroshear_marker_object_s")) or name == "object_point_cloud":
            setattr(cfg.scene, name, None)
    env = gym.make("BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0", cfg=cfg).unwrapped
    robot, peg = env.scene["robot"], env.scene["object"]
    env.reset()
    seed = json.loads(args.seed.read_text())
    kin = Kinematics()
    names = robot.joint_names
    qseed = np.array([seed["robot_joint_positions"][n] for n in kin.names])
    if args.thumb_only:
        qseed = np.array([seed["robot_joint_targets"][n] for n in kin.names])
    center = np.array(seed["seed_hand_center"])
    axis = np.array(seed.get("seed_hand_axis",[0.,1.,0.]))
    if args.thumb_only:
        from scipy.spatial.transform import Rotation
        root = np.eye(4)
        root[:3,:3] = Rotation.from_euler("z",-np.pi/2).as_matrix()
        root[:3,3] = [1,0,.766]
        hand_pose = kin.fk(seed["robot_joint_positions"],root)["base_link"]
        peg_rotation = Rotation.from_quat(np.array(seed["peg_pose"]["quaternion"])[[1,2,3,0]]).as_matrix()
        grip = np.array(seed["peg_pose"]["position"]) + peg_rotation @ [0,0,.06]
        center = hand_pose[:3,:3].T @ (grip-hand_pose[:3,3])
        axis = hand_pose[:3,:3].T @ peg_rotation[:,2]
    base = robot.data.joint_pos.clone()
    targets = base.clone()
    if args.verify_restored or args.thumb_only:
        targets[:] = torch.tensor([seed['robot_joint_targets'][n] for n in names],device=env.device)
    # Convert a prescribed surface indentation into a bounded joint preload.
    for i, preload in enumerate(np.linspace(args.preload_min, args.preload_max, args.num_envs)):
        if args.verify_restored:
            break
        for finger in (("thumb",) if args.thumb_only else ("thumb", "index", "mid")):
            ids = [j for j,n in enumerate(kin.names) if n.startswith("right_"+finger)]
            surface = np.array({"thumb": [0.,.033,-.0002], "index": [.009,.001,.021], "mid": [.010,.001,.050]}[finger])
            def point(q):
                p = kin.fk(q, hand_only=True)[f"right_{finger}dip_roll_rubber_link"]
                return p[:3,:3] @ surface + p[:3,3]
            start = point(qseed)
            jac = np.stack([(point(qseed + np.eye(len(qseed))[j]*1e-5)-start)/1e-5 for j in ids], axis=1)
            inward = center-start
            inward -= axis*np.dot(inward,axis)
            inward /= np.linalg.norm(inward)
            depth = preload * (args.thumb_factor if finger == "thumb" else 1.)
            goal = start + inward*depth
            def residual(qpart):
                q = qseed.copy(); q[ids] = qpart
                return np.r_[(point(q)-goal)*100, (qpart-qseed[ids])*.03]
            solution = least_squares(residual,np.clip(qseed[ids],kin.bounds[ids,0]+1e-6,kin.bounds[ids,1]-1e-6),bounds=kin.bounds[ids].T)
            for j,value in zip(ids,solution.x):
                targets[i,names.index(kin.names[j])] = float(value)
    robot.set_joint_position_target(targets)
    initial = peg.data.root_pose_w.clone()
    for prim in env.sim.stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            for rel in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel()):
                assert not any("/Object" in str(path) for path in rel.GetTargets()), prim.GetPath()

    history = []
    def sample(t):
        contacts = torch.stack([env.scene[f"right_{f}dip_roll_rubber_link_object_s"].data.force_matrix_w.reshape(env.num_envs,3).norm(dim=-1) for f in ("thumb","index","mid","ring","pinky")],dim=-1)
        history.append(dict(time=t,pose=peg.data.root_pose_w.cpu().tolist(),forces=contacts.cpu().tolist()))
        print(json.dumps(dict(time=t,position=peg.data.root_pos_w.cpu().tolist(),forces=contacts.cpu().tolist())),flush=True)
        return contacts

    # Settling is only part of offline calibration, never part of environment reset.
    for step in range(round(args.settle_s/env.physics_dt)):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        if step % 120 == 119:
            sample((step+1)*env.physics_dt)
    settled = peg.data.root_pose_w.clone()
    qsettled = robot.data.joint_pos.clone()
    max_drift = torch.zeros(env.num_envs,device=env.device)
    max_angle = torch.zeros_like(max_drift)
    min_contacts = torch.full((env.num_envs,5),float("inf"),device=env.device)
    # Contact reporters are cleared on reset. Allow one ordinary control frame
    # for them to repopulate, while measuring drift from the exact reset state.
    contact_warmup = env.cfg.decimation if args.settle_s == 0 else 0
    for step in range(round(2/env.physics_dt)+contact_warmup):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        max_drift = torch.maximum(max_drift,(peg.data.root_pos_w-settled[:,:3]).norm(dim=-1))
        max_angle = torch.maximum(max_angle,quat_error_magnitude(peg.data.root_quat_w,settled[:,3:]))
        contacts = torch.stack([env.scene[f"right_{f}dip_roll_rubber_link_object_s"].data.force_matrix_w.reshape(env.num_envs,3).norm(dim=-1) for f in ("thumb","index","mid","ring","pinky")],dim=-1)
        if step >= contact_warmup:
            min_contacts = torch.minimum(min_contacts,contacts)
        if step % 120 == 119:
            sample(args.settle_s+(step+1)*env.physics_dt)
    passed = (max_drift<.003)&(max_angle<np.deg2rad(5))&(min_contacts[:,0]>1)&(min_contacts[:,1:].amax(dim=-1)>1)
    report = dict(passed=passed.cpu().tolist(),max_drift_m=max_drift.cpu().tolist(),max_angle_deg=torch.rad2deg(max_angle).cpu().tolist(),minimum_finger_force_N=min_contacts.cpu().tolist(),history=history,
                  contact_report_warmup_s=contact_warmup*env.physics_dt,drift_measured_from_reset=args.settle_s==0,
                  root_body_poses={n:robot.data.body_pose_w[0,i].cpu().tolist() for i,n in enumerate(robot.body_names)},
                  joint_positions=robot.data.joint_pos.cpu().tolist(),joint_names=names)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print("PREGRASP_REPORT",json.dumps({k:v for k,v in report.items() if k not in ("history","root_body_poses","joint_positions","joint_names")}),flush=True)
    if args.write and passed.any():
        chosen = int(torch.where(passed)[0][0])
        calibrated = dict(seed)
        calibrated["robot_joint_positions"] = dict(zip(names,qsettled[chosen].cpu().tolist()))
        calibrated["robot_joint_targets"] = dict(zip(names,targets[chosen].cpu().tolist()))
        pose = settled[chosen].clone()
        pose[:3] -= env.scene.env_origins[chosen]
        calibrated["peg_pose"] = dict(position=pose[:3].cpu().tolist(),quaternion=pose[3:].cpu().tolist())
        calibrated["validated"] = True
        calibrated["validation"] = {"gravity_m_s2":9.81,"hold_seconds":2.,"report":str(args.report.resolve().relative_to(ROOT)),"env_index":chosen}
        (ASSETS/"pregrasp.json").write_text(json.dumps(calibrated,indent=2)+"\n")
    env.close()
    if args.verify_restored and not bool(passed.all()):
        raise RuntimeError("Restored pregrasp failed the gravity/contact acceptance checks")
    if args.write and not bool(passed.any()):
        raise RuntimeError("No candidate passed; no calibrated pregrasp was written")


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=Path,default=ASSETS/"pregrasp_seed.json")
    parser.add_argument("--report",type=Path,default=ASSETS/"validation/pregrasp.json")
    parser.add_argument("--num-envs",type=int,default=4)
    parser.add_argument("--preload-min",type=float,default=.002)
    parser.add_argument("--preload-max",type=float,default=.008)
    parser.add_argument("--thumb-factor",type=float,default=1.)
    parser.add_argument("--settle-s",type=float,default=1.)
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--verify-restored",action="store_true")
    parser.add_argument("--thumb-only",action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        main(args)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        # Kit may swallow a pending exception or hang during partial-startup
        # shutdown; preserve a failing process status for this offline tool.
        import os
        os._exit(1)
    finally:
        app.close()
