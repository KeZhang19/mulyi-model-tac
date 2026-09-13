"""Direct-task binding of the calibrated collection observation pipeline."""

import tempfile

import numpy as np
import torch

from isaaclab.utils.math import quat_apply_inverse
from BrainCo_DexHand.tactile_representation.direct_calibration import (
    CALIBRATION_FINGERS, DIRECT_LINKS, RUBBER_STEMS, prepare_direct_calibration,
)
from BrainCo_DexHand.tactile_representation.policy import marker_flow_to_features_tensor
from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp import observations as tactile_obs
from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_grasp import (
    _make_tacmap_link_surface_cfg,
)
from tacmap_sensor.sharpa_tacmap_link_surface import SharpaTacmapLinkSurface


def project_hydroshear_flow(adapter, displacement, depth, points, valid, marker_points, marker_normals, marker_valid):
    """Same projection as render_displacement_output, without CPU debug geometry."""
    from BrainCo_DexHand.tactile_representation.simulation_utils import project_marker_flow

    return project_marker_flow(adapter, displacement, depth, points, valid, marker_points, marker_normals, marker_valid)


class DirectVisuotactileRuntime:
    """Reuse calibrated TacMap, local refinement, HydroShear and Taxim on Direct assets.

    Internal order matches the collector. Output order matches the Repose policy.
    Only scene bindings differ; no ManagerBased environment is instantiated.
    """

    def __init__(self, env):
        self.env = env
        self._calibration_dir = tempfile.TemporaryDirectory(prefix="brainco-direct-calibration-")
        self.layout_path = prepare_direct_calibration(
            env.cfg.tactile_marker_layout_path, env.cfg.tactile_calibration_urdf, self._calibration_dir.name,
        )
        self.sensors = []
        sensor_names = {key: [] for key in ("surface", "object", "marker_surface", "marker_object")}
        for finger, stem, link in zip(CALIBRATION_FINGERS, RUBBER_STEMS, DIRECT_LINKS, strict=True):
            for key in sensor_names:
                marker = key.startswith("marker_")
                kind = key.removeprefix("marker_")
                cfg = _make_tacmap_link_surface_cfg(
                    f"right_{stem}dip_roll_rubber_link", sensor_kind=kind,
                    image_rows=10 if marker else 32, image_cols=10 if marker else 24,
                    ray_layout_prefix=f"{finger}_marker" if marker else finger,
                )
                cfg.prim_path = f"/World/envs/env_.*/Robot/{link}"
                cfg.mesh_prim_paths[0].prim_expr = (
                    "/World/envs/env_.*/object" if kind == "object" else cfg.prim_path
                )
                cfg.ray_layout_npz = str(self.layout_path)
                name = f"visuotactile_{finger}_{key}"
                sensor = SharpaTacmapLinkSurface(cfg)
                env.scene.sensors[name] = sensor
                self.sensors.append(sensor)
                sensor_names[key].append(name)
        self.component_cfg = {
            "tacmap_sensor_names": sensor_names["object"],
            "tacmap_surface_sensor_names": sensor_names["surface"],
            "hydroshear_marker_sensor_names": sensor_names["marker_object"],
            "hydroshear_marker_surface_sensor_names": sensor_names["marker_surface"],
            "hydroshear_marker_finger_link_names": list(DIRECT_LINKS),
            "hydroshear_marker_layout_path": str(self.layout_path),
            "tacmap_rows": 32, "tacmap_cols": 24,
            "local_tacmap_enabled": True,
            "local_tacmap_reference_rows": 240, "local_tacmap_reference_cols": 320,
            "local_tacmap_rows": 25, "local_tacmap_cols": 40,
            "local_tacmap_contact_threshold_m": 0.02e-3,
            "local_tacmap_roi_margin_m": 1.0e-3,
            "local_tacmap_penetration_deadband_m": 1.0e-6,
            "local_tacmap_max_distance_m": 0.05,
            "taxim_rgb_render_rows": 240, "taxim_rgb_render_cols": 320,
            "taxim_rgb_render_chunk_size": env.cfg.tactile_taxim_chunk_size,
            "taxim_rgb_background_path": env.cfg.tactile_rgb_background_path,
            "taxim_rgb_with_shadow": env.cfg.tactile_taxim_with_shadow,
            "hydroshear_render_rows": 240, "hydroshear_render_cols": 320,
            "hydroshear_marker_rows": 10, "hydroshear_marker_cols": 10,
            "hydroshear_use_object_surface_samples": True,
            "hydroshear_cache_debug_output": False,
        }
        self._markers_bound = False
        self._output_order = (3, 2, 0, 1, 4)

    def _bind_markers_to_direct_surface(self):
        """Reproject marker rays onto the actual target hand, not the reference mesh."""
        env = self.env
        points, normals, valid = [], [], []
        for name, link in zip(self.component_cfg["hydroshear_marker_surface_sensor_names"], DIRECT_LINKS, strict=True):
            sensor = env.scene.sensors[name]
            sensor.data  # update the rays before reading their geometry
            pose = env.hand.data.body_link_state_w[0, env.hand.body_names.index(link), :7]
            hit = sensor.ray_hits_w[0]
            normal = sensor.ray_normals_w[0]
            mask = sensor.ray_hit_valid[0] & torch.isfinite(hit).all(-1)
            if not mask.any():
                raise RuntimeError(f"Calibrated marker rays miss Direct fingertip {link}; check sensor mounting")
            quat = pose[3:7].expand(len(hit), -1)
            points.append(quat_apply_inverse(quat, torch.nan_to_num(hit) - pose[:3]).cpu().numpy())
            normals.append(quat_apply_inverse(quat, torch.nan_to_num(normal)).cpu().numpy())
            valid.append(mask.cpu().numpy())
        with np.load(self.layout_path, allow_pickle=False) as layout:
            uv = layout["pixels_distorted"].copy()
            valid = np.stack(valid) & layout["distortion_valid"][None, :]
        env._brainco_rl_hydroshear_marker_layout = (uv, np.stack(points), np.stack(normals), valid)
        env._brainco_rl_hydroshear_marker_layout_key = (str(self.layout_path.resolve()), tuple(DIRECT_LINKS))
        self._markers_bound = True

    @torch.no_grad()
    def observe(self, normalization):
        env, cfg = self.env, self.component_cfg
        if not self._markers_bound:
            self._bind_markers_to_direct_surface()
        displacement = tactile_obs.ours_rl_hydroshear_obs(env, cfg).reshape(env.num_envs * 5, 100, 3)
        adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
        if adapter is None:
            raise RuntimeError("Calibrated HydroShear failed to initialize")
        # Materializes exactly the dense metric field/background used by collection.
        rgb = tactile_obs.ours_rl_taxim_rgb_obs(env, cfg).reshape(env.num_envs, 5, 3, 240, 320)
        depth = env._rl_ours_dense_tacmap_depth_m.unsqueeze(2)
        calibrated = tactile_obs._hydroshear_calibrated_marker_world(
            env, marker_layout_path=self.layout_path, finger_link_names=DIRECT_LINKS,
            sensor_count=5, robot_cfg=tactile_obs.SceneEntityCfg("robot"),
        )
        flow = project_hydroshear_flow(
            adapter, displacement, env._rl_tacmap_penetration_m.reshape(-1, 32, 24),
            env._rl_tacmap_surface_points_w.reshape(-1, 32, 24, 3),
            env._rl_tacmap_surface_valid.reshape(-1, 32, 24),
            calibrated[1], calibrated[2], calibrated[3],
        )
        marker, valid = marker_flow_to_features_tensor(flow, calibrated[3])
        marker, valid = marker.reshape(env.num_envs, 5, 100, 5), valid.reshape(env.num_envs, 5, 100)
        rgb = (rgb * 255.0).round().clamp(0, 255).to(torch.uint8)
        reference = env._rl_ours_taxim_rgb_background_cache["real"].permute(2, 0, 1)
        reference = reference[None, None].expand_as(rgb)
        if normalization.marker_count != 100:
            raise ValueError("Direct calibrated sensor has 100 markers; encoder checkpoint is incompatible")
        if (normalization.image_height, normalization.image_width) != (240, 320):
            raise ValueError("Direct calibration requires a 240x320 encoder checkpoint")
        order = self._output_order
        return {
            "rgb": rgb[:, order], "depth_m": depth[:, order],
            "marker": marker[:, order], "marker_valid": valid[:, order],
            "rgb_reference": reference[:, order],
        }

    def reset(self, env_ids):
        for sensor in self.sensors:
            sensor.reset(env_ids)
        tactile_obs.invalidate_ours_tactile_cache_on_reset(self.env, env_ids)
        adapter = getattr(self.env, "_brainco_rl_hydroshear_adapter", None)
        if adapter is not None and adapter._batch_state_valid is not None:
            for env_id in torch.as_tensor(env_ids).cpu().tolist():
                for finger in range(5):
                    adapter._reset_hydrosoft_state(int(env_id) * 5 + finger)

    def close(self):
        self._calibration_dir.cleanup()
