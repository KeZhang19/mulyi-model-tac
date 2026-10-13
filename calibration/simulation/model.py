"""Geometry reference, provisional pressure/optics, and the existing HydroShear adapter.

This is a prescribed-motion elastomer model, not a PhysX/FEM force oracle.
"""
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from geometry import ROOT, HEADS, first_hit, load_assembly, sensor_grid, hand_asset, sensor_link

sys.path.insert(0, str(ROOT / 'integrate'))
from curved_hydroshear_adapter import RevoCurvedHydroShearAdapter, RevoCurvedHydroShearCfg


class MechanicalCalibrationBench:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hand_urdf = hand_asset(cfg)
        self.frames, self.hand, self.paths = load_assembly(
            self.hand_urdf, cfg['joint_positions_rad'],
            mesh_only=cfg.get('mesh_visuals_only', False),
            excluded_links=cfg.get('excluded_hand_links', ()))
        self.link = sensor_link(cfg)
        self.surface, self.normals, self.valid, self.area = sensor_grid(self.hand[self.link], *cfg['grid_hw'])
        self.tool_frames, self.tool, paths = load_assembly(HEADS[cfg['head']])
        self.paths |= paths
        self.tip = self.tool_frames['press_head_tip'][:3, 3]
        self.head = self.tool['press_head']
        # Persistent material samples are required for shear and rotations.
        samples, _ = __import__('trimesh').sample.sample_surface(self.head, 1024, seed=cfg['seed'])
        h, w = cfg['grid_hw']
        rows, cols = cfg['marker_rows_cols']
        u, v = np.meshgrid(np.linspace(3, w - 4, cols), np.linspace(3, h - 4, rows))
        self.marker_uv = np.stack((u.ravel(), v.ravel()), -1).astype(np.float32)
        self.marker_ij = np.rint(self.marker_uv[:, ::-1]).astype(int)
        self.marker_valid = self.valid[self.marker_ij[:, 0], self.marker_ij[:, 1]]
        self.hydro = RevoCurvedHydroShearAdapter(RevoCurvedHydroShearCfg(
            width=w, height=h, marker_rows=rows, marker_cols=cols, marker_uv=self.marker_uv,
            device='cpu', object_sample_points_l=samples, object_sample_roi_count=128,
            **cfg['hydroshear']))
        self.pixel_pitch = np.array([self.surface[0, 1, 0] - self.surface[0, 0, 0],
                                     self.surface[1, 0, 1] - self.surface[0, 0, 1]])
        self.reset()

    def reset(self):
        self.hydro.reset()
        self.previous_depth = np.zeros(self.cfg['grid_hw'])

    def contact_zero(self, offset):
        """Find first sampled contact over the actual mesh footprint, world Z."""
        center = (self.hand[self.link].bounds[0] + self.hand[self.link].bounds[1]) / 2
        xy = center[:2] + np.asarray(offset)
        origins = self.surface.reshape(-1, 3).copy()
        origins[:, :2] -= xy
        origins[:, 2] = -0.08
        rays = np.tile([0., 0., 1.], (len(origins), 1))
        d, _ = first_hit(self.head, origins, rays)
        hits = np.isfinite(d) & self.valid.ravel()
        if not hits.any():
            raise ValueError('Tool misses selected sensor')
        z = np.max(self.surface.reshape(-1, 3)[hits, 2] - (-.08 + d[hits]))
        return np.r_[xy, z]

    def pose(self, zero, displacement):
        rotation = Rotation.from_euler('xyz', displacement[3:]).as_matrix()
        zero = np.asarray(zero).copy()
        if self.cfg.get('follow_surface_for_slide', False) and np.any(np.asarray(displacement[:2]) != 0):
            center = self.hand[self.link].bounds.mean(axis=0)
            offset = zero[:2] - center[:2] + np.asarray(displacement[:2])
            zero[2] = self.contact_zero(offset)[2]
        # Virtual rotations pivot about the working tip, not the flange.
        pivot = zero + self.tip + np.array([displacement[0], displacement[1], -displacement[2]])
        pos = pivot - rotation @ self.tip
        bounds = np.asarray(self.cfg['axis_bounds_m'])
        if np.any(pos < bounds[:, 0]) or np.any(pos > bounds[:, 1]):
            raise ValueError('Virtual XYZ travel limit')
        matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = pos
        return matrix

    def collision_guard(self, matrix):
        """Sampled non-target containment, not continuous collision certification."""
        problems = []
        for tool_name, mesh in self.tool.items():
            moved = mesh.copy(); moved.apply_transform(matrix)
            for name, obstacle in self.hand.items():
                if name == self.link or np.any(moved.bounds[1] < obstacle.bounds[0]) or np.any(obstacle.bounds[1] < moved.bounds[0]):
                    continue
                if not obstacle.is_watertight:
                    problems.append(f'uncertified_open_mesh:{name}')
                    continue
                vertices = moved.vertices[::max(1, len(moved.vertices) // 256)]
                if obstacle.contains(vertices).any():
                    problems.append(f'non_target_collision:{tool_name}:{name}')
        return sorted(set(problems))

    def sample(self, matrix):
        c = self.cfg
        p, n = self.surface.reshape(-1, 3), self.normals.reshape(-1, 3)
        # Start below the deepest allowed indentation and cast outwards. The
        # first underside hit, NOT the top of the shaft, determines intrusion.
        ray_length = 0.004
        origins = p - ray_length * n
        local_origins = (origins - matrix[:3, 3]) @ matrix[:3, :3]
        local_dirs = n @ matrix[:3, :3]
        distance, _ = first_hit(self.head, local_origins, local_dirs)
        hit = np.isfinite(distance) & self.valid.ravel()
        signed_gap = np.where(hit, distance - ray_length, 0).reshape(c['grid_hw'])
        depth = np.where(hit.reshape(c['grid_hw']), np.maximum(-signed_gap, 0), 0)
        velocity = (depth - self.previous_depth) * c['sample_hz']
        self.previous_depth = depth.copy()
        pressure_unsat = c['young_modulus_pa'] * (depth / c['shell_thickness_m']) ** c['pressure_gamma']
        pressure = np.minimum(pressure_unsat, c['pressure_saturation_pa'])
        taxel_force = pressure * self.area
        force = -(taxel_force[..., None] * self.normals).sum((0, 1)) # force on sensor
        contact = depth > 1e-8
        uv = np.indices(depth.shape)[::-1].transpose(1, 2, 0)
        total = taxel_force.sum()
        cop = (uv * taxel_force[..., None]).sum((0, 1)) / max(total, 1e-20)
        object_points = self.surface + signed_gap[..., None] * self.normals
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        with torch.no_grad():
            displacement = self.hydro.step_displacement_only(
                depth.astype(np.float32), self.surface.astype(np.float32), self.valid,
                object_points.astype(np.float32), hit.reshape(depth.shape),
                object_pose_wxyz=np.r_[matrix[:3, 3], quat][None].astype(np.float32),
                surface_raw_m=(self.surface[..., 2].max() + .01 - self.surface[..., 2]).astype(np.float32),
                surface_normals_w=self.normals.astype(np.float32), ray_directions_w=np.array([[0, 0, -1]], dtype=np.float32),
            )[0].cpu().numpy()
        flow = displacement[:, :2] / self.pixel_pitch
        flow[~self.marker_valid] = 0
        marker = np.c_[self.marker_uv, flow, self.marker_valid].astype(np.float32)
        # Explicit uncalibrated three-light Lambertian diagnostic renderer.
        gy, gx = np.gradient(depth, self.pixel_pitch[1], self.pixel_pitch[0])
        optical_n = np.stack((-gx, -gy, np.ones_like(depth)), -1)
        optical_n /= np.linalg.norm(optical_n, axis=-1, keepdims=True)
        lights = np.array([[.6, 0, .8], [-.3, .52, .8], [-.3, -.52, .8]])
        rgb = np.clip(30 + 180 * optical_n @ lights.T, 0, 255).astype(np.uint8)
        for row in marker:
            if not row[4]:
                continue
            x, y = np.rint(row[:2] + row[2:4]).astype(int)
            if 0 <= y < len(rgb) and 0 <= x < rgb.shape[1]:
                rgb[y, x] = 15
        rgb[~self.valid] = 0
        return dict(depth_m=depth.astype(np.float32), depth_velocity_mps=velocity.astype(np.float32),
                    signed_gap_m=signed_gap.astype(np.float32), ray_hit_valid=hit.reshape(depth.shape),
                    contact_mask=contact, pressure_pa=pressure.astype(np.float32),
                    taxel_force_n=taxel_force.astype(np.float32), total_normal_load_n=np.float32(total),
                    force_on_sensor_w_n=force.astype(np.float32), center_of_pressure_px=cop.astype(np.float32),
                    cop_valid=np.bool_(total > 0), pressure_saturated=pressure_unsat >= c['pressure_saturation_pa'],
                    marker=marker, marker_displacement_w_m=displacement, rgb=rgb,
                    T_world_flange=matrix, T_world_tip=matrix @ self.tool_frames['press_head_tip'])
