from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import omni.usd as omni_usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from revo2_tactile_env_cfg import Revo2TactilePressEnvCfg
from tacmap_sensor.sharpa_tacmap_link_surface import SharpaTacmapLinkSurface, SharpaTacmapLinkSurfaceCfg
from tacmap_sensor.sharpa_tacmap_vbts import SharpaTacmap
from pxr import Gf, Usd, UsdGeom
from torch_jit_utils import deform_quantize


class Revo2TactilePressEnv(DirectRLEnv):
    cfg: Revo2TactilePressEnvCfg

    def __init__(self, cfg: Revo2TactilePressEnvCfg, render_mode: str | None = None, **kwargs):
        self._touch_center_l_np, self._touch_normal_l_np = self._load_touch_center_and_normal(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.touch_link not in self.hand.body_names:
            raise RuntimeError(
                f"Touch link {self.cfg.touch_link!r} not found in Revo2 body names: {self.hand.body_names}"
            )
        self.touch_body_idx = self.hand.body_names.index(self.cfg.touch_link)
        self.touch_center_l = torch.tensor(
            self._touch_center_l_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.touch_normal_l = torch.tensor(
            self._touch_normal_l_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.press_local_offset_l = torch.tensor(
            getattr(self.cfg, "press_local_offset", (0.0, 0.0, 0.0)), dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.object_rot_l = torch.tensor(
            self.cfg.object_rot_in_touch_frame, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.object_flip_l = torch.tensor(
            self.cfg.object_flip_quat_in_touch_frame, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        slide_axis = torch.tensor(self.cfg.press_slide_axis_l, dtype=torch.float32, device=self.device)
        slide_axis_norm = torch.linalg.norm(slide_axis).clamp_min(1.0e-8)
        self.press_slide_axis_l = (slide_axis / slide_axis_norm).unsqueeze(0).repeat(self.num_envs, 1)

        self.press_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        height, width = self._tacmap_image_shape()
        sensor_count = len(self._vbts_sensor)
        self.vbts_deform = torch.zeros((self.num_envs, sensor_count, height, width), dtype=torch.uint8, device=self.device)
        self.tacmap_raw = torch.zeros((self.num_envs, sensor_count, height, width), dtype=torch.float32, device=self.device)
        self.tacmap_surface_raw = torch.zeros_like(self.tacmap_raw)
        self.tacmap_object_raw = torch.zeros_like(self.tacmap_raw)
        self.contact_forces = torch.zeros((self.num_envs, len(self._contact_sensor), 3), device=self.device)
        self.contact_pos = torch.zeros((self.num_envs, len(self._contact_sensor), 3), device=self.device)

    @staticmethod
    def _load_touch_center_and_normal(cfg: Revo2TactilePressEnvCfg) -> tuple[np.ndarray, np.ndarray]:
        points = np.load(cfg.points_npy).astype(np.float64) * float(cfg.vbts_sensor[0].correction_scale)
        normals = np.load(cfg.normals_npy).astype(np.float64)
        valid = (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(normals).all(axis=-1)
            & (np.linalg.norm(points, axis=-1) > 1e-12)
            & (np.linalg.norm(normals, axis=-1) > 0.5)
        )
        if not np.any(valid):
            raise RuntimeError(f"No valid tactile samples found in {cfg.points_npy}")

        pts = points[valid]
        nrms = normals[valid]
        nrms = nrms / (np.linalg.norm(nrms, axis=-1, keepdims=True) + 1e-12)
        center = np.mean(pts, axis=0)
        normal = np.mean(nrms, axis=0)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        print(f"[INFO] Revo2 tactile center(local,m): {center}, normal(local): {normal}")
        return center.astype(np.float32), normal.astype(np.float32)

    def _setup_scene(self):
        self.object = RigidObject(self.cfg.object_cfg)
        self.hand = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        if bool(getattr(self.cfg, "enable_touch_materials", True)):
            self._apply_touch_contact_materials()
        if bool(getattr(self.cfg, "tacmap_use_xform_anchor", False)):
            self._create_tacmap_anchor_prims()

        self._contact_sensor = []
        if bool(getattr(self.cfg, "enable_contact_sensor", True)):
            for sensor_id, sensor_cfg in enumerate(self.cfg.contact_sensor):
                sensor = ContactSensor(sensor_cfg)
                self._contact_sensor.append(sensor)
                self.scene.sensors[f"contact_sensor_{sensor_id}"] = sensor

        self._vbts_sensor = []
        self._tacmap_surface_sensor = []
        for sensor_id, sensor_cfg in enumerate(self.cfg.vbts_sensor):
            if str(self.cfg.tacmap_ray_mode) == "link_surface":
                common_kwargs = self._link_surface_sensor_kwargs(sensor_cfg)
                surface_prim_path = str(getattr(self.cfg, "tacmap_surface_prim_path", "") or sensor_cfg.prim_path)
                surface_cfg = SharpaTacmapLinkSurfaceCfg(
                    **common_kwargs,
                    ray_hit_index=2,
                    mesh_prim_paths=[
                        SharpaTacmapLinkSurfaceCfg.RaycastTargetCfg(
                            prim_expr=surface_prim_path,
                            track_mesh_transforms=bool(getattr(self.cfg, "tacmap_surface_track_mesh_transforms", True)),
                        )
                    ],
                )
                object_targets = [
                    SharpaTacmapLinkSurfaceCfg.RaycastTargetCfg(
                        prim_expr=str(target.prim_expr),
                        track_mesh_transforms=bool(getattr(target, "track_mesh_transforms", True)),
                    )
                    for target in sensor_cfg.mesh_prim_paths
                ]
                object_cfg = SharpaTacmapLinkSurfaceCfg(
                    **common_kwargs,
                    ray_hit_index=1,
                    mesh_prim_paths=object_targets,
                )
                surface_sensor = SharpaTacmapLinkSurface(surface_cfg)
                object_sensor = SharpaTacmapLinkSurface(object_cfg)
                self._tacmap_surface_sensor.append(surface_sensor)
                self._vbts_sensor.append(object_sensor)
                self.scene.sensors[f"tacmap_surface_sensor_{sensor_id}"] = surface_sensor
                self.scene.sensors[f"tacmap_object_sensor_{sensor_id}"] = object_sensor
            else:
                if isinstance(sensor_cfg, SharpaTacmapLinkSurfaceCfg):
                    sensor = SharpaTacmapLinkSurface(sensor_cfg)
                else:
                    sensor = SharpaTacmap(sensor_cfg)
                self._vbts_sensor.append(sensor)
                self.scene.sensors[f"vbts_sensor_{sensor_id}"] = sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _tacmap_image_shape(self) -> tuple[int, int]:
        if str(self.cfg.tacmap_ray_mode) == "link_surface":
            return int(self.cfg.tacmap_link_surface_height), int(self.cfg.tacmap_link_surface_width)
        size = 240 // int(self.cfg.resolution_step)
        return size, size

    def _create_tacmap_anchor_prims(self) -> None:
        stage = omni_usd.get_context().get_stage()
        anchor_expr = str(getattr(self.cfg, "tacmap_anchor_prim_path", "/World/envs/env_.*/TacmapAnchor"))
        anchor_leaf = anchor_expr.rsplit("/", 1)[-1]
        for env_id in range(int(self.cfg.scene.num_envs)):
            env_path = f"/World/envs/env_{env_id}"
            touch_path = f"{env_path}/Robot/{self.cfg.touch_link}"
            anchor_path = f"{env_path}/{anchor_leaf}"
            env_prim = stage.GetPrimAtPath(env_path)
            touch_prim = stage.GetPrimAtPath(touch_path)
            if not env_prim.IsValid() or not touch_prim.IsValid():
                continue
            env_world = UsdGeom.Xformable(env_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            touch_world = UsdGeom.Xformable(touch_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local = touch_world * env_world.GetInverse()
            translation = local.ExtractTranslation()
            rotation_quat = local.ExtractRotation().GetQuat()
            rotation_imag = rotation_quat.GetImaginary()
            anchor = UsdGeom.Xform.Define(stage, anchor_path)
            xformable = UsdGeom.Xformable(anchor.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(translation)
            xformable.AddOrientOp().Set(
                Gf.Quatf(
                    float(rotation_quat.GetReal()),
                    Gf.Vec3f(float(rotation_imag[0]), float(rotation_imag[1]), float(rotation_imag[2])),
                )
            )
            xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    def _link_surface_sensor_kwargs(self, sensor_cfg) -> dict:
        return {
            "prim_path": str(sensor_cfg.prim_path),
            "update_period": float(sensor_cfg.update_period),
            "pattern_cfg": sensor_cfg.pattern_cfg,
            "offset": SharpaTacmapLinkSurfaceCfg.OffsetCfg(
                pos=tuple(sensor_cfg.offset.pos),
                rot=tuple(sensor_cfg.offset.rot),
                convention=str(sensor_cfg.offset.convention),
            ),
            "data_types": ["distance_along_normal", "distance_along_normal_raw"],
            "points_npy": str(sensor_cfg.points_npy),
            "normals_npy": str(sensor_cfg.normals_npy),
            "resolution_step": 1,
            "max_distance": float(sensor_cfg.max_distance),
            "cpd_max_dist": float(sensor_cfg.cpd_max_dist),
            "correction_scale": float(sensor_cfg.correction_scale),
            "pts_offsets": float(sensor_cfg.pts_offsets),
            "image_width": int(self.cfg.tacmap_link_surface_width),
            "image_height": int(self.cfg.tacmap_link_surface_height),
            "ray_axis": str(self.cfg.tacmap_link_surface_ray_axis),
            "ray_direction": self.cfg.tacmap_link_surface_ray_direction,
            "grid_u_axis": str(self.cfg.tacmap_link_surface_grid_u_axis),
            "grid_v_axis": str(self.cfg.tacmap_link_surface_grid_v_axis),
            "grid_u_size": float(self.cfg.tacmap_link_surface_grid_u_size),
            "grid_v_size": float(self.cfg.tacmap_link_surface_grid_v_size),
            "grid_center": tuple(float(v) for v in self.cfg.tacmap_link_surface_grid_center),
        }

    def _apply_touch_contact_materials(self):
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            compliant_contact_stiffness=float(self.cfg.compliant_contact_stiffness),
            compliant_contact_damping=float(self.cfg.compliant_contact_damping),
        )
        collision_cfg = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(self.cfg.touch_contact_offset),
            rest_offset=float(self.cfg.touch_rest_offset),
        )

        for env_id in range(int(self.cfg.scene.num_envs)):
            robot_path = f"/World/envs/env_{env_id}/Robot"
            try:
                sim_utils.make_uninstanceable(robot_path)
            except Exception as exc:
                print(f"[WARN] Could not make {robot_path} uninstanceable before material binding: {exc}")

            for rel_path in self.cfg.touch_collision_paths:
                collision_path = f"{robot_path}/{rel_path}"
                material_path = f"{collision_path}/compliant_material"
                try:
                    material_cfg.func(material_path, material_cfg)
                    sim_utils.modify_collision_properties(collision_path, collision_cfg)
                    sim_utils.bind_physics_material(collision_path, material_path)
                    if env_id == 0:
                        print(
                            f"[INFO] touch compliant material applied: {collision_path}, "
                            f"stiffness={self.cfg.compliant_contact_stiffness}, damping={self.cfg.compliant_contact_damping}"
                        )
                except Exception as exc:
                    print(f"[WARN] Could not bind touch compliant material on {collision_path}: {exc}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.press_counter += 1
        self._write_presser_pose()

    def _apply_action(self) -> None:
        joint_pos = self.hand.data.default_joint_pos
        self.hand.set_joint_position_target(joint_pos)

    def _write_presser_pose(self):
        touch_state = self.hand.data.body_link_state_w[:, self.touch_body_idx, :7]
        touch_pos_w = touch_state[:, :3]
        touch_quat_w = touch_state[:, 3:7]

        denom = max(1, int(self.cfg.press_steps) - 1)
        alpha = torch.clamp(self.press_counter.float() / float(denom), 0.0, 1.0)
        offset = self.cfg.press_start_offset + alpha * (self.cfg.press_end_offset - self.cfg.press_start_offset)
        object_pos_l = self.touch_center_l + self.press_local_offset_l + self.touch_normal_l * offset.unsqueeze(-1)
        slide_steps = max(0, int(self.cfg.press_slide_steps))
        slide_distance = float(self.cfg.press_slide_distance)
        if slide_steps > 0 and abs(slide_distance) > 0.0:
            slide_counter = torch.clamp(self.press_counter - (max(1, int(self.cfg.press_steps)) - 1), min=0)
            slide_alpha = torch.clamp(slide_counter.float() / float(max(1, slide_steps)), 0.0, 1.0)
            object_pos_l = object_pos_l + self.press_slide_axis_l * (slide_distance * slide_alpha).unsqueeze(-1)

        object_pos_w = quat_apply(touch_quat_w, object_pos_l) + touch_pos_w
        object_quat_l = quat_mul(self.object_rot_l, self.object_flip_l)
        object_quat_w = quat_mul(touch_quat_w, object_quat_l)

        object_state = self.object.data.default_root_state.clone()
        object_state[:, :3] = object_pos_w
        object_state[:, 3:7] = object_quat_w
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7])
        self.object.write_root_velocity_to_sim(object_state[:, 7:])

    def _get_observations(self) -> dict:
        self._refresh_tactile()
        obs = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        return {
            "policy": obs,
            "vbts_deform": self.vbts_deform,
            "tacmap": self.vbts_deform,
            "tacmap_raw": self.tacmap_raw,
            "tacmap_surface_raw": self.tacmap_surface_raw,
            "tacmap_object_raw": self.tacmap_object_raw,
            "tactile_forces": self.contact_forces,
            "tactile_points": self.contact_pos,
        }

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        reset = self.press_counter >= self._press_total_steps()
        timeout = torch.zeros_like(reset)
        return reset, timeout

    def _press_total_steps(self) -> int:
        return max(1, int(self.cfg.press_steps) + max(0, int(self.cfg.press_slide_steps)))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        dof_pos = self.hand.data.default_joint_pos[env_ids]
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.press_counter[env_ids] = 0
        self._write_presser_pose()
        self._refresh_tactile()

    def _refresh_tactile(self):
        height, width = self._tacmap_image_shape()
        if str(self.cfg.tacmap_ray_mode) == "link_surface":
            surface_raw = torch.cat(
                [
                    sensor.data.output["distance_along_normal_raw"].reshape(self.num_envs, height, width).unsqueeze(1)
                    for sensor in self._tacmap_surface_sensor
                ],
                dim=1,
            ).float()
            object_raw = torch.cat(
                [
                    sensor.data.output["distance_along_normal_raw"].reshape(self.num_envs, height, width).unsqueeze(1)
                    for sensor in self._vbts_sensor
                ],
                dim=1,
            ).float()
            object_raw_for_projection = torch.where(object_raw > 0.0, object_raw, surface_raw)
            penetration = torch.clamp(surface_raw - object_raw_for_projection, min=0.0)
            penetration = torch.where(surface_raw > 0.0, penetration, torch.zeros_like(penetration))
            self.tacmap_surface_raw = torch.nan_to_num(surface_raw, nan=0.0, posinf=0.0, neginf=0.0)
            self.tacmap_object_raw = torch.nan_to_num(object_raw_for_projection, nan=0.0, posinf=0.0, neginf=0.0)
            self.tacmap_raw = torch.nan_to_num(penetration, nan=0.0, posinf=0.0, neginf=0.0)
            quantized = deform_quantize(self.tacmap_raw.clone().reshape(-1, height * width, 1))
            self.vbts_deform = quantized.reshape(self.num_envs, len(self._vbts_sensor), height, width).to(torch.uint8)
        else:
            self.vbts_deform = torch.cat(
                [
                    sensor.data.output["distance_along_normal"].reshape(self.num_envs, height, width).unsqueeze(1)
                    for sensor in self._vbts_sensor
                ],
                dim=1,
            ).to(torch.uint8)
            raw_list = []
            for sensor in self._vbts_sensor:
                raw = sensor.data.output.get("distance_along_normal_raw")
                if raw is None:
                    raw = sensor.data.output["distance_along_normal"].float()
                raw_list.append(raw.reshape(self.num_envs, height, width).unsqueeze(1))
            self.tacmap_raw = torch.nan_to_num(torch.cat(raw_list, dim=1).float(), nan=0.0, posinf=0.0, neginf=0.0)
            self.tacmap_object_raw = self.tacmap_raw.clone()
            self.tacmap_surface_raw = torch.zeros_like(self.tacmap_raw)

        if self._contact_sensor:
            self.contact_forces = torch.cat(
                [sensor.data.net_forces_w_history[:, 0, 0, :].unsqueeze(1) for sensor in self._contact_sensor],
                dim=1,
            )

            contact_pos_w = torch.cat(
                [sensor.data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for sensor in self._contact_sensor],
                dim=1,
            )
            contact_pos_w = torch.nan_to_num(contact_pos_w, nan=0.0)

            touch_state = self.hand.data.body_link_state_w[:, self.touch_body_idx, :7]
            touch_pos_w = touch_state[:, :3].unsqueeze(1)
            touch_quat_w = touch_state[:, 3:7].unsqueeze(1).repeat(1, len(self._contact_sensor), 1)
            local_delta = contact_pos_w - touch_pos_w
            local_contact = quat_apply(quat_inv(touch_quat_w.reshape(-1, 4)), local_delta.reshape(-1, 3))
            self.contact_pos = local_contact.reshape(self.num_envs, len(self._contact_sensor), 3)
        else:
            self.contact_forces = torch.zeros((self.num_envs, 0, 3), dtype=torch.float32, device=self.device)
            self.contact_pos = torch.zeros((self.num_envs, 0, 3), dtype=torch.float32, device=self.device)
