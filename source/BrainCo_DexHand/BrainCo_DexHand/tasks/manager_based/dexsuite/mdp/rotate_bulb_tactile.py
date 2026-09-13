"""Rotate-Bulb observations from the pretrained multimodal restoration encoder."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from BrainCo_DexHand.tactile_representation.policy import (
    file_sha256, load_sim_policy_encoder, marker_flow_to_features_tensor,
)
from BrainCo_DexHand.tactile_representation.simulation_utils import ensure_taxim_scatter, project_marker_flow
from . import observations as tactile_obs


# Collector/sensor order: middle, index, ring, pinky, thumb.
# Portable encoder order: little, ring, middle, index, thumb.
_POLICY_FINGER_INDICES = (3, 2, 0, 1, 4)


@torch.no_grad()
def observe_rotate_bulb_tactile(env, component_cfg):
    """Use the collection calibration and pixel convention, without rendering debug images."""
    cfg = component_cfg
    displacement = tactile_obs.ours_rl_hydroshear_obs(env, cfg).reshape(env.num_envs * 5, 100, 3)
    adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
    if adapter is None:
        raise RuntimeError("Rotate-Bulb calibrated HydroShear did not initialize")
    rgb = tactile_obs.ours_rl_taxim_rgb_obs(env, cfg).reshape(env.num_envs, 5, 3, 240, 320)
    depth = env._rl_ours_dense_tacmap_depth_m.unsqueeze(2)
    calibrated = tactile_obs._hydroshear_calibrated_marker_world(
        env, marker_layout_path=cfg["hydroshear_marker_layout_path"],
        finger_link_names=cfg["hydroshear_marker_finger_link_names"],
        sensor_count=5, robot_cfg=SceneEntityCfg("robot"),
    )
    flow = project_marker_flow(
        adapter, displacement, env._rl_tacmap_penetration_m.reshape(-1, 32, 24),
        env._rl_tacmap_surface_points_w.reshape(-1, 32, 24, 3),
        env._rl_tacmap_surface_valid.reshape(-1, 32, 24),
        calibrated[1], calibrated[2], calibrated[3],
    )
    marker, valid = marker_flow_to_features_tensor(flow, calibrated[3])
    marker, valid = marker.reshape(env.num_envs, 5, 100, 5), valid.reshape(env.num_envs, 5, 100)
    order = _POLICY_FINGER_INDICES
    return {
        "rgb": (rgb[:, order] * 255.0).round().clamp(0, 255).to(torch.uint8),
        "depth_m": depth[:, order], "marker": marker[:, order], "marker_valid": valid[:, order],
    }


class RotateBulbPretrainedTactile(ManagerTermBase):
    """Frozen five-finger encoder with one sensor update per control step/reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.encoder = load_sim_policy_encoder(
            env.cfg.tactile_policy_checkpoint, chunk_size=env.cfg.tactile_encoder_chunk_size,
        ).to(env.device)
        norm = self.encoder.normalization
        if (norm.image_height, norm.image_width, norm.marker_count) != (240, 320, 100):
            raise ValueError("Rotate-Bulb calibration requires a 240x320 encoder with 100 markers")
        self.component_cfg = copy.deepcopy(cfg.params["component_cfg"])
        self.component_cfg.update(
            taxim_rgb_render_chunk_size=env.cfg.tactile_taxim_chunk_size,
            hydroshear_cache_debug_output=False, hydroshear_render_debug_marker_images=False,
        )
        ensure_taxim_scatter()
        self._cache_key = None
        self._cached = None
        env._rotate_bulb_tactile_term = self
        env.latest_tactile_reconstruction = None
        print(
            f"[INFO] Rotate-Bulb pretrained tactile encoder: {env.cfg.tactile_policy_checkpoint}; "
            f"feature={self.encoder.feature_mode}, per_finger={self.encoder.projection_dim}, "
            f"frozen=True, sha256={self.encoder.bundle_sha256}", flush=True,
        )

    @torch.no_grad()
    def __call__(self, env, component_cfg):
        key = (env.common_step_counter, getattr(env, "_rl_ours_tactile_reset_epoch", 0))
        if self._cache_key == key:
            return self._cached
        sample = observe_rotate_bulb_tactile(env, self.component_cfg)
        latent = self.encoder(**sample)
        if not torch.isfinite(latent).all():
            raise RuntimeError("Rotate-Bulb pretrained tactile encoder produced non-finite features")
        env.latest_tactile_inputs = sample
        env.latest_tactile_latent = latent
        if env.cfg.tactile_reconstruction_diagnostics:
            reference = env._rl_ours_taxim_rgb_background_cache["real"].permute(2, 0, 1)
            env.latest_tactile_reconstruction = self.encoder.reconstruct(
                **sample, rgb_reference=reference[None, None].expand_as(sample["rgb"]),
            )
        self._cached = latent.flatten(1)
        self._cache_key = key
        return self._cached

    def reset(self, env_ids=None):
        self._cache_key = None
        self._cached = None
        env = self._env
        tactile_obs.invalidate_ours_tactile_cache_on_reset(env, env_ids)
        adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
        if adapter is not None and adapter._batch_state_valid is not None:
            ids = torch.arange(env.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu()
            for index in ids.tolist():
                for finger in range(5):
                    adapter._reset_hydrosoft_state(int(index) * 5 + finger)


def initialize_rotate_bulb_tactile_contract(env, env_ids=None):
    """Capture the actual manager layout after its observation dimensions are resolved."""
    term = env._rotate_bulb_tactile_term
    manager = env.observation_manager
    groups = ("policy", "proprio", "perception")
    dimension = sum(int(manager.group_obs_dim[name][0]) for name in groups)
    contract = term.encoder.observation_contract(
        state_dim=dimension - 5 * term.encoder.projection_dim, history_length=1,
    )
    contract.update(
        task="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0",
        frame_order=list(groups), observation_dim=dimension,
        simulation_encoder_sha256=term.encoder.bundle_sha256,
        observation_groups={
            group: [dict(name=name, shape=[int(size) for size in shape]) for name, shape in zip(
                manager.active_terms[group], manager.group_obs_term_dim[group], strict=True,
            )] for group in groups
        },
    )
    cfg = term.component_cfg
    marker_path = Path(cfg["hydroshear_marker_layout_path"])
    contract["simulation_sensor_assets"] = {
        "marker_layout": file_sha256(marker_path),
        "camera_rectangles": file_sha256(marker_path.with_name("camera_ray_rectangles_320x240.json")),
        "rgb_background": file_sha256(cfg["taxim_rgb_background_path"]),
    }
    # Inference chunk sizes may vary with hardware without changing observations.
    contract["sensor_parameters"] = {
        key: value for key, value in cfg.items()
        if "chunk_size" not in key and not key.endswith("_path") and not key.startswith("tactile_resnet_")
    }
    env.tactile_policy_contract = contract
