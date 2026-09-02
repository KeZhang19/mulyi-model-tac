from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.utils.warp import raycast_dynamic_meshes
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster

from torch_jit_utils import deform_quantize
import torchvision.transforms as transforms

if TYPE_CHECKING:
    from .sharpa_multi_mesh_vbts_cfg import SharpaMultiMeshVBTSCfg

import omni.usd as omni_usd
from pxr import Gf, UsdGeom, Sdf, Vt


class SharpaTacmap(MultiMeshRayCaster):
    """Sharpa Vision-Based Tactile Sensor (VBTS) based on ray casting from surface points and normals."""

    cfg: SharpaMultiMeshVBTSCfg
    UNSUPPORTED_TYPES: ClassVar[set[str]] = set()

    def __init__(self, cfg: SharpaMultiMeshVBTSCfg):
        # check data_types, no need to write additional functions as ray_caster_camera.py
        for name in cfg.data_types:
            if name not in ["distance_along_normal", "distance_along_normal_raw"]:
                raise ValueError(f"Unsupported data type: {name}")
        # initialize
        super().__init__(cfg)
        MultiMeshRayCaster.__init__(self, cfg)
        # create empty variables for storing output data
        self._data = MultiMeshRayCasterData()

    # view.count = num_envs * sensors_per_env
    def __str__(self) -> str:
        return (
            f"Sharpa VBTS @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )
    """
    Properties
    """

    @property
    def data(self) -> MultiMeshRayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data
    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # Reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    # num_envs: number of environments created
    # N: number of points/norms

    # _w: world reference frame
    # _tar: target reference frame
    # _att: attached body reference frame
    # _b: baked reference frame

    def _initialize_rays_impl(self):
        # Create all indices buffer and Create frame count buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        # read the points and normals on the surface points array (load on CPU then copy to device)
        resolution_step = self.cfg.resolution_step
        pts_np = np.load(self.cfg.points_npy)[::resolution_step, ::resolution_step, :]
        nrm_np = np.load(self.cfg.normals_npy)[::resolution_step, ::resolution_step, :]
        # (x,y,3) reshape to (N, 3), N=x*y
        pts_np = np.asarray(pts_np).reshape(-1, 3)
        nrm_np = np.asarray(nrm_np).reshape(-1, 3)

        assert pts_np.shape[-1] == 3 and nrm_np.shape[-1] == 3
        assert pts_np.shape[0] == nrm_np.shape[0]
        # normalize normals just in case, may be uunnecessary
        nrm_norm = np.linalg.norm(nrm_np, axis=-1, keepdims=True) + 1e-12
        nrm_np = nrm_np / nrm_norm
        # to torch tensor and copy to device
        pts = torch.tensor(pts_np, dtype=torch.float32, device=self._device)
        nrms = torch.tensor(nrm_np, dtype=torch.float32, device=self._device)

        blur_kernel = max(1, 9 // int(self.cfg.resolution_step))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        self.blur = transforms.GaussianBlur(kernel_size=(blur_kernel, blur_kernel), sigma=(1.5, 1.5))

        self.pts = pts.clone()

        # correct scale
        pts = pts * self.cfg.correction_scale  # mm to m, or other dimension in the sim world
        nrms = -nrms  # flip normals as the normal directions are point outside
        pts_offset = self.cfg.pts_offsets  # meters
        pts = pts + pts_offset * nrms

        # copy to every envs（num_envs, N, 3）
        self.num_rays = pts.shape[0]
        # create buffers after self.num_rays is properly set
        self._create_buffers()

        # ray_starts and ray_directions size become:
        # (num_envs, N, 3)
        # include the value of pts and nrms, in AttachedBody reference frame
        self.ray_starts_att = pts.unsqueeze(0).repeat(self._view.count, 1, 1)
        self.ray_directions_att = nrms.unsqueeze(0).repeat(self._view.count, 1, 1)
        # create buffer for storing ray hits (num_envs, N, 3)
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

        # offset pose (same as camera)
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        
        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._surface_normals_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

        # visualization of tactile sensing points (in debug_viz), downsampled for USD display
        max_viz_points = max(1, int(getattr(self.cfg, "debug_viz_max_points", 5000)))
        self._viz_stride = max(1, (self.num_rays + max_viz_points - 1) // max_viz_points)

        # how many samples per env when downsample with [::self._viz_stride]
        self._viz_count = (self.num_rays + self._viz_stride - 1) // self._viz_stride
        self._viz_indices = torch.arange(
            0, self.num_rays, self._viz_stride, device=self._device, dtype=torch.long
        )[: self._viz_count]

        # Create a per-env buffer for visualization hits in WORLD coordinates
        # Shape: (num_envs, viz_count, 3)
        if not hasattr(self._data, "ray_hits_w"):
            self._data.ray_hits_w = torch.zeros(
                (self._view.count, self._viz_count, 3),
                device=self._device,
                dtype=torch.float32,
            )

        if getattr(self.cfg, "debug_viz", False):
            self._setup_surface_points_viz_usd(
                env_id=int(getattr(self.cfg, "debug_viz_env_id", 0))
            )
            if getattr(self.cfg, "debug_viz_normals", True):
                self._setup_surface_normals_viz_usd(
                    env_id=int(getattr(self.cfg, "debug_viz_env_id", 0))
                )

    def _create_buffers(self):
        """
        Create buffers for storing data. used in initialization (_initialize_rays_impl).
        Note: different from ray_caster_camera, here num_rays is determined after loading points.npy
        """
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w = torch.zeros((self._view.count, 4), device=self._device)
        self._data.output = {}

        # data structure: [Num_envs, Num_rays, Mesh_id]
        if "distance_along_normal" in self.cfg.data_types:
            self._data.output["distance_along_normal"] = torch.zeros(
                (self._view.count, self.num_rays, 1), device=self._device, dtype=torch.uint8
            )
        if "distance_along_normal_raw" in self.cfg.data_types:
            self._data.output["distance_along_normal_raw"] = torch.zeros(
                (self._view.count, self.num_rays, 1), device=self._device, dtype=torch.float32
            )
            
        # may switch to [Num_envs, H, W, Mesh_id] in future
        self._data.image_mesh_ids = torch.zeros(self._num_envs, self.num_rays, 1, device=self.device, dtype=torch.int16)

        # required by raycaster
        self.drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.bias  = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_bias  = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
    
    def _update_ray_infos(self, env_ids: Sequence[int]):
        """Updates the ray information buffers."""

        # compute poses from current view
        # pos_w sensor parent position in world frame
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        # update the data
        self._data.pos_w[env_ids] = pos_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts_att[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions_att[env_ids])
        surface_normals_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), -self.ray_directions_att[env_ids])

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w
        self._surface_normals_w[env_ids] = surface_normals_w

        if getattr(self.cfg, "debug_viz", False):
            self._update_surface_points_viz_usd(env_ids)
            if getattr(self.cfg, "debug_viz_normals", True):
                self._update_surface_normals_viz_usd(env_ids)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data. main sensing logic implemented here."""
        self._update_ray_infos(env_ids)
        # increment frame count
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        self._frame[env_ids] += 1
        debug_hit_stats = bool(getattr(self.cfg, "debug_hit_stats", False))
        need_hit_normals = debug_hit_stats

        # follow the multi_mesh_ray_caster
        # Update the mesh positions and rotations
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # update position of the target meshes
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:  # Mesh is not global, i.e. we have different meshes for each env
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        # ray cast and store the hits
        self.ray_hits_w[env_ids], ray_depth, ray_normal, _, ray_mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=True,
            return_normal=need_hit_normals,
            return_mesh_id=need_hit_normals or bool(getattr(self.cfg, "update_mesh_ids", False)),
        )

        cpd_hits = self.ray_hits_w[env_ids]            # (N,3)
        cpd_distance = ray_depth            # (N,)
        cpd_has_hit_1st = torch.isfinite(cpd_distance)

        # CPD: advance-start second raycast just beyond the first hit
        cpd_eps = -1e-4          # tiny step beyond the surface

        # new start points = start + dir * (dist + eps)
        cpd_advance = (cpd_distance.clamp_min(0.0) + cpd_eps).unsqueeze(-1)            # (N,1)
            # use the same starts as first pass
        cpd_ray_starts = self._ray_starts_w[env_ids] + self._ray_directions_w[env_ids] * cpd_advance              # (N,3)

        # second pass from advanced starts
        _, cpd_ray_depth, cpd_ray_normal, _, _ = raycast_dynamic_meshes(
            cpd_ray_starts,
            -self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.cpd_max_dist, # cpd_max_dist is defined in cfg, which should provide a relatively large value
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=True,
            return_normal=need_hit_normals,
            return_mesh_id=False,
        )

        # Invalidate when contact is detected on CPD pass
        cpd_keep = torch.isfinite(cpd_distance) & torch.isfinite(cpd_ray_depth)

        hit_axis_dot = None
        hit_ray_abs_dot = None
        cpd_axis_dot = None
        if need_hit_normals and ray_normal is not None:
            normal_finite = torch.isfinite(ray_normal).all(dim=-1)
            normal_norm = torch.linalg.norm(ray_normal, dim=-1, keepdim=True).clamp_min(1.0e-12)
            hit_normals_w = ray_normal / normal_norm

            ray_dir_norm = torch.linalg.norm(self._ray_directions_w[env_ids], dim=-1, keepdim=True).clamp_min(1.0e-12)
            ray_dirs_w = self._ray_directions_w[env_ids] / ray_dir_norm
            hit_ray_abs_dot = torch.abs(torch.sum(hit_normals_w * ray_dirs_w, dim=-1))

            # Diagnostic-only bucket axis for the cylinder presser. This does
            # not affect cpd_keep or the exported TacMap.
            axis_l = torch.tensor((0.0, 0.0, 1.0), device=self._device, dtype=torch.float32)
            axis_l = axis_l / torch.linalg.norm(axis_l).clamp_min(1.0e-12)
            axes_l = axis_l.view(1, 1, 3).repeat(len(env_ids), self._mesh_orientations_w.shape[1], 1)
            axes_w = math_utils.quat_apply(
                self._mesh_orientations_w[env_ids].reshape(-1, 4),
                axes_l.reshape(-1, 3),
            ).reshape(len(env_ids), self._mesh_orientations_w.shape[1], 3)

            if ray_mesh_ids is None:
                axis_per_ray_w = axes_w[:, :1, :].expand(-1, self.num_rays, -1)
            else:
                safe_mesh_ids = ray_mesh_ids.to(torch.long).clamp(0, axes_w.shape[1] - 1)
                axis_per_ray_w = torch.gather(axes_w, 1, safe_mesh_ids.unsqueeze(-1).expand(-1, -1, 3))
            hit_axis_dot = torch.abs(torch.sum(hit_normals_w * axis_per_ray_w, dim=-1))

            if cpd_ray_normal is not None:
                cpd_normal_norm = torch.linalg.norm(cpd_ray_normal, dim=-1, keepdim=True).clamp_min(1.0e-12)
                cpd_normals_w = cpd_ray_normal / cpd_normal_norm
                cpd_axis_dot = torch.abs(torch.sum(cpd_normals_w * axes_w[:, :1, :], dim=-1))

            if debug_hit_stats:
                self._print_hit_debug_stats(
                    env_ids=env_ids,
                    first_hit_mask=cpd_has_hit_1st,
                    cpd_keep=cpd_keep,
                    cpd_distance=cpd_distance,
                    hit_axis_dot=hit_axis_dot,
                    cpd_axis_dot=cpd_axis_dot,
                    hit_ray_abs_dot=hit_ray_abs_dot,
                )

        # filter out the "odd situation" that the points is outside the object mesh
        ray_depth_afterCPD = torch.where(cpd_keep, cpd_distance, torch.zeros_like(cpd_distance)) 

        # distances clipping policy
        export_ray_depth = ray_depth_afterCPD.clone()
        export_ray_depth = torch.where(torch.isfinite(export_ray_depth), export_ray_depth, torch.zeros_like(export_ray_depth))

        if "distance_along_normal_raw" in self._data.output:
            self._data.output["distance_along_normal_raw"][env_ids] = export_ray_depth.view(-1, self.num_rays, 1)

        if "distance_along_normal" in self._data.output:
            export_ray_depth_quantize = deform_quantize(export_ray_depth.view(-1, self.num_rays, 1))
            export_ray_depth_quantize = export_ray_depth_quantize.reshape(len(env_ids), 240//self.cfg.resolution_step, 240//self.cfg.resolution_step)
            export_ray_depth_quantize = self.blur(export_ray_depth_quantize)

            self._data.output["distance_along_normal"][env_ids] = export_ray_depth_quantize.view(-1, self.num_rays, 1)

        if getattr(self.cfg, "debug_viz", False) and getattr(self.cfg, "debug_viz_active_points", False):
            self._update_surface_points_active_viz_usd(env_ids, export_ray_depth)

        if self.cfg.update_mesh_ids:
            self._data.image_mesh_ids[env_ids] = ray_mesh_ids.view(-1, self.num_rays, 1)
        

        # visualize hit points in USD
        # if getattr(self.cfg, "debug_viz", False):
        if False:
            E = 0  # visualize env_0
            stride = self._viz_stride
            hits_w = self.ray_hits_w[E, ::stride, :]   # [K,3]
            depth  = ray_depth[E, ::stride]            # [K]
            hit_mask = torch.isfinite(depth) & (depth > 0)

            hit_pts = self._ray_starts_w[E, ::stride, :]

            self._update_hit_viz_usd(hit_pts)

    def _print_hit_debug_stats(
        self,
        *,
        env_ids: torch.Tensor,
        first_hit_mask: torch.Tensor,
        cpd_keep: torch.Tensor,
        cpd_distance: torch.Tensor,
        hit_axis_dot: torch.Tensor,
        cpd_axis_dot: torch.Tensor | None,
        hit_ray_abs_dot: torch.Tensor,
    ) -> None:
        every = max(1, int(getattr(self.cfg, "debug_hit_stats_every", 50)))
        frame = int(self._frame[env_ids][0].detach().cpu().item())
        if frame % every != 0:
            return

        for local_i, env_id in enumerate(env_ids.detach().cpu().tolist()):
            keep = cpd_keep[local_i]
            first = first_hit_mask[local_i]
            total_count = int(first.numel())
            first_count = int(torch.count_nonzero(first).detach().cpu().item())
            keep_count = int(torch.count_nonzero(keep).detach().cpu().item())
            no_first_count = total_count - first_count
            first_no_second_mask = first & ~keep
            first_no_second_count = int(torch.count_nonzero(first_no_second_mask).detach().cpu().item())
            reject_bucket_text = ""
            if first_no_second_count > 0:
                reject_axis_dot = hit_axis_dot[local_i][first_no_second_mask]
                reject_side = int(torch.count_nonzero(reject_axis_dot < 0.35).detach().cpu().item())
                reject_bevel = int(
                    torch.count_nonzero((reject_axis_dot >= 0.35) & (reject_axis_dot < 0.85)).detach().cpu().item()
                )
                reject_cap = int(torch.count_nonzero(reject_axis_dot >= 0.85).detach().cpu().item())
                reject_bucket_text = f" reject(side/bevel/cap)={reject_side}/{reject_bevel}/{reject_cap}"
            if keep_count == 0:
                print(
                    f"[TacMapHitStats frame={frame:06d} env={int(env_id)}] "
                    f"total={total_count} first={first_count} keep=0 "
                    f"invalid(no_first/first_no_second)={no_first_count}/{first_no_second_count}"
                    f"{reject_bucket_text}",
                    flush=True,
                )
                continue

            axis_dot = hit_axis_dot[local_i][keep]
            ray_dot = hit_ray_abs_dot[local_i][keep]
            depth_mm = cpd_distance[local_i][keep] * 1000.0
            side = int(torch.count_nonzero(axis_dot < 0.35).detach().cpu().item())
            bevel = int(torch.count_nonzero((axis_dot >= 0.35) & (axis_dot < 0.85)).detach().cpu().item())
            cap = int(torch.count_nonzero(axis_dot >= 0.85).detach().cpu().item())
            pair_text = ""
            if cpd_axis_dot is not None:
                second_axis_dot = cpd_axis_dot[local_i][keep]
                first_side = axis_dot < 0.35
                first_cap = axis_dot >= 0.85
                second_side = second_axis_dot < 0.35
                second_cap = second_axis_dot >= 0.85
                side_side = int(torch.count_nonzero(first_side & second_side).detach().cpu().item())
                side_cap = int(torch.count_nonzero(first_side & second_cap).detach().cpu().item())
                cap_side = int(torch.count_nonzero(first_cap & second_side).detach().cpu().item())
                cap_cap = int(torch.count_nonzero(first_cap & second_cap).detach().cpu().item())
                pair_text = (
                    f"pairs(ss/sc/cs/cc)={side_side}/{side_cap}/{cap_side}/{cap_cap} "
                )
            axis_q = torch.quantile(axis_dot, torch.tensor([0.05, 0.50, 0.95], device=axis_dot.device))
            ray_q = torch.quantile(ray_dot, torch.tensor([0.05, 0.50, 0.95], device=ray_dot.device))
            depth_q = torch.quantile(depth_mm, torch.tensor([0.05, 0.50, 0.95], device=depth_mm.device))

            axis_q = axis_q.detach().cpu().numpy()
            ray_q = ray_q.detach().cpu().numpy()
            depth_q = depth_q.detach().cpu().numpy()
            depth_max = float(depth_mm.max().detach().cpu().item())
            print(
                f"[TacMapHitStats frame={frame:06d} env={int(env_id)}] "
                f"total={total_count} first={first_count} keep={keep_count} "
                f"invalid(no_first/first_no_second)={no_first_count}/{first_no_second_count}"
                f"{reject_bucket_text} "
                f"side={side} bevel={bevel} cap={cap} "
                f"{pair_text}"
                f"axis_dot_q05/50/95={axis_q[0]:.3f}/{axis_q[1]:.3f}/{axis_q[2]:.3f} "
                f"ray_absdot_q05/50/95={ray_q[0]:.3f}/{ray_q[1]:.3f}/{ray_q[2]:.3f} "
                f"depth_mm_q05/50/95/max={depth_q[0]:.3f}/{depth_q[1]:.3f}/{depth_q[2]:.3f}/{depth_max:.3f}",
                flush=True,
            )

    def _compute_view_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self._view, XFormPrim):
            if isinstance(env_ids, slice):
                env_ids = self._ALL_INDICES
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        return pos_w.clone(), quat_w.clone()
    
    """
    Additional Functions
    """

    def _setup_hit_viz_usd(self, env_id: int = 0):
        """Create a USD PointInstancer under /World/envs/env_{env_id}/... for visualizing points."""
        stage = omni_usd.get_context().get_stage()

        self._viz_env_id = int(env_id)
        self._viz_instancer_path = Sdf.Path(f"/World/envs/env_{self._viz_env_id}/VBTS_hit_points")

        inst = UsdGeom.PointInstancer.Define(stage, self._viz_instancer_path)
        inst.CreatePositionsAttr()
        inst.CreateProtoIndicesAttr()
        inst.CreateScalesAttr()

        # prototype sphere
        proto_root = self._viz_instancer_path.AppendPath("Prototypes")
        UsdGeom.Xform.Define(stage, proto_root)

        sphere_path = proto_root.AppendPath("Sphere")
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(0.002)  # 2mm radius (tune)

        inst.CreatePrototypesRel().SetTargets([sphere_path])

        K = int(self._viz_count)
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0] * K))
        inst.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * K))
        inst.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * K))

        self._viz_instancer = inst

    def _update_hit_viz_usd(self, pts_w: torch.Tensor):
        """Update the instancer positions. pts_w should be [K,3] in world coordinates."""
        if not hasattr(self, "_viz_instancer"):
            return

        K = int(self._viz_count)
        if pts_w.ndim != 2 or pts_w.shape[-1] != 3:
            return

        # ensure exactly K points
        if pts_w.shape[0] < K:
            pad = torch.zeros((K - pts_w.shape[0], 3), device=pts_w.device, dtype=pts_w.dtype)
            pts_w = torch.cat([pts_w, pad], dim=0)
        else:
            pts_w = pts_w[:K]

        pts = pts_w.detach().cpu().numpy()
        self._viz_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        )

    def _setup_surface_points_viz_usd(self, env_id: int = 0):
        """Create a USD PointInstancer for downsampled tactile sensing points."""
        stage = omni_usd.get_context().get_stage()

        self._surface_viz_env_id = max(0, min(int(env_id), int(self._view.count) - 1))
        sensor_name = self.cfg.prim_path.rstrip("/").split("/")[-1]
        sensor_name = (
            sensor_name.replace("*", "all")
            .replace(".", "_")
            .replace("[", "_")
            .replace("]", "_")
        )
        self._surface_viz_instancer_path = Sdf.Path(
            f"/Visuals/TacmapSurfacePoints/env_{self._surface_viz_env_id}/{sensor_name}"
        )

        UsdGeom.Xform.Define(stage, self._surface_viz_instancer_path.GetParentPath())
        inst = UsdGeom.PointInstancer.Define(stage, self._surface_viz_instancer_path)
        inst.CreatePositionsAttr()
        inst.CreateProtoIndicesAttr()
        inst.CreateScalesAttr()

        proto_root = self._surface_viz_instancer_path.AppendPath("Prototypes")
        UsdGeom.Xform.Define(stage, proto_root)

        sphere_path = proto_root.AppendPath("SurfacePoint")
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(float(getattr(self.cfg, "debug_viz_point_radius", 0.0008)))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(1.0, 0.0, 0.0)])
        )

        inst.CreatePrototypesRel().SetTargets([sphere_path])

        k = int(self._viz_count)
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0] * k))
        inst.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * k))
        inst.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * k))

        self._surface_viz_instancer = inst
        self._set_surface_points_viz_colors()

    def _set_surface_points_viz_colors(self):
        """Color points by flattened TacMap order, or red when disabled."""
        if not hasattr(self, "_surface_viz_instancer"):
            return

        if not getattr(self.cfg, "debug_viz_order_colors", True):
            colors = [Gf.Vec3f(1.0, 0.0, 0.0)] * int(self._viz_count)
        else:
            width = max(1, int(round(240 / max(1, int(self.cfg.resolution_step)))))
            height = max(1, int(round(240 / max(1, int(self.cfg.resolution_step)))))
            idx = self._viz_indices.detach().cpu().numpy()
            rows = idx // width
            cols = idx % width
            color_mode = getattr(self.cfg, "debug_viz_color_mode", "gradient")
            if color_mode == "stripes":
                interval = max(1, int(getattr(self.cfg, "debug_viz_stripe_interval", 10)))
                colors = []
                for row, col in zip(rows, cols):
                    is_row_stripe = row % interval == 0
                    is_col_stripe = col % interval == 0
                    if is_row_stripe and is_col_stripe:
                        colors.append(Gf.Vec3f(1.0, 1.0, 0.0))
                    elif is_row_stripe:
                        colors.append(Gf.Vec3f(1.0, 0.0, 0.0))
                    elif is_col_stripe:
                        colors.append(Gf.Vec3f(0.0, 0.2, 1.0))
                    else:
                        colors.append(Gf.Vec3f(0.8, 0.8, 0.8))
            else:
                denom_row = max(1, height - 1)
                denom_col = max(1, width - 1)
                colors = [
                    Gf.Vec3f(float(col / denom_col), float(row / denom_row), 1.0 - float(row / denom_row))
                    for row, col in zip(rows, cols)
                ]

        self._surface_viz_base_colors = colors
        self._apply_surface_points_viz_colors(colors)

    def _apply_surface_points_viz_colors(self, colors: list[Gf.Vec3f]):
        if not hasattr(self, "_surface_viz_instancer"):
            return

        k = int(self._viz_count)
        if len(colors) < k:
            colors = colors + [Gf.Vec3f(0.25, 0.25, 0.25)] * (k - len(colors))
        elif len(colors) > k:
            colors = colors[:k]

        primvar = UsdGeom.PrimvarsAPI(self._surface_viz_instancer.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.Vec3fArray(colors))

    def _update_surface_points_active_viz_usd(self, env_ids: torch.Tensor, ray_depth_after_cpd: torch.Tensor):
        """Color downsampled tactile surface points by the current valid-depth mask."""
        if not hasattr(self, "_surface_viz_instancer"):
            return

        env_id = int(self._surface_viz_env_id)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        matches = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
        if matches.numel() == 0:
            return

        local_i = int(matches[0].detach().cpu().item())
        if ray_depth_after_cpd.ndim != 2 or local_i >= ray_depth_after_cpd.shape[0]:
            return

        threshold = float(getattr(self.cfg, "debug_viz_active_threshold", 0.0))
        depths = ray_depth_after_cpd[local_i, self._viz_indices]
        active = torch.isfinite(depths) & (depths > threshold)
        active_np = active.detach().cpu().numpy().astype(bool)

        active_color = Gf.Vec3f(1.0, 0.0, 0.0)
        inactive_color = Gf.Vec3f(0.25, 0.25, 0.25)
        colors = [active_color if is_active else inactive_color for is_active in active_np]
        self._apply_surface_points_viz_colors(colors)

    def _update_surface_points_viz_usd(self, env_ids: Sequence[int]):
        """Update visualized tactile sensing points for the configured env."""
        if not hasattr(self, "_surface_viz_instancer"):
            return

        env_id = int(self._surface_viz_env_id)
        if isinstance(env_ids, torch.Tensor):
            if not torch.any(env_ids == env_id).item():
                return
        elif isinstance(env_ids, slice):
            pass
        elif env_id not in env_ids:
            return

        pts_w = self._ray_starts_w[env_id, :: self._viz_stride, :]
        k = int(self._viz_count)
        if pts_w.shape[0] < k:
            pad = torch.zeros((k - pts_w.shape[0], 3), device=pts_w.device, dtype=pts_w.dtype)
            pts_w = torch.cat([pts_w, pad], dim=0)
        else:
            pts_w = pts_w[:k]

        pts = pts_w.detach().cpu().numpy()
        self._surface_viz_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        )

    def _setup_surface_normals_viz_usd(self, env_id: int = 0):
        """Create USD basis curves for downsampled outward surface normals."""
        stage = omni_usd.get_context().get_stage()

        self._normal_viz_env_id = max(0, min(int(env_id), int(self._view.count) - 1))
        self._normal_viz_indices = torch.arange(
            0,
            self.num_rays,
            max(1, int(getattr(self.cfg, "debug_viz_normal_stride", 40))),
            device=self._device,
            dtype=torch.long,
        )
        sensor_name = self.cfg.prim_path.rstrip("/").split("/")[-1]
        sensor_name = (
            sensor_name.replace("*", "all")
            .replace(".", "_")
            .replace("[", "_")
            .replace("]", "_")
        )
        self._normal_viz_path = Sdf.Path(
            f"/Visuals/TacmapSurfaceNormals/env_{self._normal_viz_env_id}/{sensor_name}"
        )

        UsdGeom.Xform.Define(stage, self._normal_viz_path.GetParentPath())
        curves = UsdGeom.BasisCurves.Define(stage, self._normal_viz_path)
        curves.CreateTypeAttr("linear")
        curves.CreateBasisAttr("bezier")
        curves.CreateCurveVertexCountsAttr(Vt.IntArray([2] * int(len(self._normal_viz_indices))))
        curves.CreateWidthsAttr(
            Vt.FloatArray([float(getattr(self.cfg, "debug_viz_point_radius", 0.0008)) * 0.35])
        )
        UsdGeom.Gprim(curves.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(1.0, 0.55, 0.0)])
        )
        zero_points = [Gf.Vec3f(0.0, 0.0, 0.0)] * int(len(self._normal_viz_indices) * 2)
        curves.CreatePointsAttr(Vt.Vec3fArray(zero_points))
        self._normal_viz_curves = curves

    def _update_surface_normals_viz_usd(self, env_ids: Sequence[int]):
        """Update visualized outward normal line segments."""
        if not hasattr(self, "_normal_viz_curves"):
            return

        env_id = int(self._normal_viz_env_id)
        if isinstance(env_ids, torch.Tensor):
            if not torch.any(env_ids == env_id).item():
                return
        elif isinstance(env_ids, slice):
            pass
        elif env_id not in env_ids:
            return

        starts = self._ray_starts_w[env_id, self._normal_viz_indices, :]
        normals = self._surface_normals_w[env_id, self._normal_viz_indices, :]
        length = float(getattr(self.cfg, "debug_viz_normal_length", 0.004))
        ends = starts + normals * length
        segments = torch.stack([starts, ends], dim=1).reshape(-1, 3)
        pts = segments.detach().cpu().numpy()
        self._normal_viz_curves.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        )
