"""Geometry checks and mesh export for the rotate-bulb tabletop installation.

Uses authored mesh-to-body transforms and live PhysX body poses, so Fabric's
stale USD world transforms cannot make a floating base appear to pass.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    return transform


def body_meshes_local(stage, body_path):
    """Return visible USD triangle meshes expressed in their rigid body frame."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(body_path)
    assert root.IsValid(), body_path
    cache = UsdGeom.XformCache()
    inverse_body = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)).T)
    result = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or any(x in path.lower() for x in ("collision", "collider")):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy"):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        faces, offset = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset:offset + count]
            faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
            offset += count
        if not faces:
            continue
        local = inverse_body @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points = points @ local[:3, :3].T + local[:3, 3]
        result.append((path, points, np.asarray(faces)))
    return result


class TabletopGeometry:
    def __init__(self, env):
        from pxr import Usd

        self.env = env
        self.robot, self.table, self.support = (env.scene[name] for name in ("robot", "table", "support"))
        source = Usd.Stage.Open(self.robot.cfg.spawn.usd_path)
        base_meshes = body_meshes_local(source, str(source.GetDefaultPrim().GetPath()) + "/Base_R")
        support_meshes = body_meshes_local(env.sim.stage, "/World/envs/env_0/Lamp/Support")
        self.points = {
            name: torch.tensor(np.concatenate([mesh[1] for mesh in meshes]), dtype=torch.float32, device=env.device)
            for name, meshes in (("robot", base_meshes), ("support", support_meshes))
        }
        self.max_bottom_error = 0.0
        self.min_edge_margin = float("inf")

    def check(self, selected):
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        env, table = self.env, self.table
        half = torch.tensor(table.cfg.spawn.size, device=env.device) / 2
        for name, asset in (("robot", self.robot), ("support", self.support)):
            pos, quat = subtract_frame_transforms(
                table.data.root_pos_w[selected], table.data.root_quat_w[selected],
                asset.data.root_pos_w[selected], asset.data.root_quat_w[selected],
            )
            points = self.points[name].expand(len(selected), -1, -1)
            local = quat_apply(quat[:, None, :].expand(-1, points.shape[1], -1), points) + pos[:, None, :]
            low, high = local.amin(dim=1), local.amax(dim=1)
            bottom_error = (low[:, 2] - half[2]).abs().max().item()
            margin = torch.minimum(low[:, :2] + half[:2], half[:2] - high[:, :2]).min().item()
            assert bottom_error < .001, (name, bottom_error)
            assert margin > .05, (name, margin)
            self.max_bottom_error = max(self.max_bottom_error, bottom_error)
            self.min_edge_margin = min(self.min_edge_margin, margin)
        support_local = self.support.data.root_pos_w[selected] - env.scene.env_origins[selected]
        assert ((support_local[:, 0] >= .43 - 1e-5) & (support_local[:, 0] <= .47 + 1e-5)).all()
        assert ((support_local[:, 1] >= .08 - 1e-5) & (support_local[:, 1] <= .12 + 1e-5)).all()
        root_local = self.robot.data.root_pos_w[selected] - env.scene.env_origins[selected]
        torch.testing.assert_close(root_local, torch.tensor([1., 0., .766], device=env.device).expand_as(root_local),
                                   atol=1e-5, rtol=0)

    def compare_original_hand_pose(self):
        """Compare both configurations using PhysX forward kinematics, without stepping contacts."""
        from isaaclab.utils.math import quat_error_magnitude
        from BrainCo_DexHand.assets.tianji_revo3_right import TIANJI_REVO3_RIGHT_CFG

        env, robot = self.env, self.robot
        root = robot.data.root_pose_w.clone()
        joint_pos, joint_vel = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
        limits = robot.data.soft_joint_pos_limits
        assert ((joint_pos >= limits[..., 0] - 1e-5) & (joint_pos <= limits[..., 1] + 1e-5)).all()
        body_ids, names = robot.find_bodies(["Link7_R", "right_palm"], preserve_order=True)
        assert len(body_ids) == 2
        env.sim.forward()
        current = robot.data.body_pose_w[:, body_ids].clone()
        old_root = root.clone()
        old_root[:, 2] -= .186
        old_joints = joint_pos.clone()
        for name, angle in TIANJI_REVO3_RIGHT_CFG.init_state.joint_pos.items():
            old_joints[:, robot.joint_names.index(name)] = angle
        try:
            robot.write_root_pose_to_sim(old_root)
            robot.write_joint_state_to_sim(old_joints, torch.zeros_like(old_joints))
            env.sim.forward()
            previous = robot.data.body_pose_w[:, body_ids].clone()
            errors = (current[..., :3] - previous[..., :3]).norm(dim=-1)
            angles = quat_error_magnitude(current[..., 3:], previous[..., 3:])
            assert errors.max() < .001, errors
            assert angles.max() < np.deg2rad(1), angles
            return dict(bodies=names, max_position_error_m=float(errors.max()),
                        max_orientation_error_deg=float(angles.max()) * 180 / np.pi,
                        old_pose=previous[0].cpu().tolist(), new_pose=current[0].cpu().tolist())
        finally:
            robot.write_root_pose_to_sim(root)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            env.sim.forward()


def export_live_scene(env, path):
    """Export env_0's actual USD geometry at live body poses to a standalone GLB."""
    import trimesh

    scene = trimesh.Scene()
    env.sim.forward()
    robot = env.scene["robot"]
    table = env.scene["table"]
    top = float(table.data.root_pos_w[0, 2]) + table.cfg.spawn.size[2] / 2
    min_robot_clearance = float("inf")
    for asset_name, asset in (("robot", robot), ("support", env.scene["support"]), ("object", env.scene["object"])):
        if asset_name == "robot":
            bodies = [(name, f"/World/envs/env_0/Robot/{name}", asset.data.body_pose_w[0, i])
                      for i, name in enumerate(asset.body_names)]
        else:
            name = "Support" if asset_name == "support" else "Bulb"
            bodies = [(name, f"/World/envs/env_0/Lamp/{name}", asset.data.root_pose_w[0])]
        for name, body_path, pose in bodies:
            transform = pose_matrix(pose.cpu().numpy())
            pieces = []
            for _, points, faces in body_meshes_local(env.sim.stage, body_path):
                mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
                mesh.apply_transform(transform)
                pieces.append(mesh)
            if not pieces:
                continue
            mesh = trimesh.util.concatenate(pieces)
            mesh.visual.vertex_colors = {"robot": [220, 225, 233, 255], "support": [68, 83, 100, 255],
                                         "object": [247, 198, 64, 255]}[asset_name]
            if asset_name == "robot":
                clearance = float(mesh.bounds[0, 2]) - top
                assert clearance > -.001, (name, clearance)
                if name != "Base_R":
                    min_robot_clearance = min(min_robot_clearance, clearance)
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    mesh = trimesh.creation.box(extents=table.cfg.spawn.size, transform=pose_matrix(table.data.root_pose_w[0].cpu().numpy()))
    mesh.visual.vertex_colors = [117, 139, 156, 255]
    scene.add_geometry(mesh, node_name="Table", geom_name="Table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))
    return dict(meshes=len(scene.geometry), min_nonbase_table_clearance_m=min_robot_clearance, file=str(path))
