# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct Revo3 cube task with five-finger Depth, Marker Motion and Taxim RGB."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import torch

from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import InHandManipulationEnv
from BrainCo_DexHand.tactile_representation.policy import (
    FrozenTactilePolicyEncoder, TactileObservationHistory, file_sha256,
)
from .brainco.visuotactile_runtime import DirectVisuotactileRuntime

from .brainco.brainco_hand_visuotactile_env_cfg import TACMAP_ROOT

if str(TACMAP_ROOT) not in sys.path:
    sys.path.append(str(TACMAP_ROOT))

if TYPE_CHECKING:
    from .brainco.brainco_hand_visuotactile_env_cfg import BrainCoVisuotactileHandEnvCfg


def _ensure_torch_scatter_compatibility() -> None:
    """Provide the Taxim scatter-min operation when torch-scatter is absent."""
    from BrainCo_DexHand.tactile_representation.simulation_utils import ensure_taxim_scatter

    ensure_taxim_scatter()


class VisuotactileInHandManipulationEnv(InHandManipulationEnv):
    """Original cube dynamics with a frozen aligned five-finger tactile tower."""

    cfg: BrainCoVisuotactileHandEnvCfg

    def __init__(self, cfg: BrainCoVisuotactileHandEnvCfg, render_mode: str | None = None, **kwargs):
        self._repose_training = None
        # Resolve dimensions before DirectRLEnv creates Gym spaces and buffers.
        encoder = FrozenTactilePolicyEncoder(
            cfg.tactile_policy_checkpoint, domain="sim", chunk_size=cfg.tactile_encoder_chunk_size,
        )
        norm = encoder.normalization
        if (norm.image_height, norm.image_width, norm.marker_count) != (240, 320, 100):
            raise ValueError("Direct calibrated tactile observations require H=240, W=320, K=100")
        if cfg.tactile_encoder_id and cfg.tactile_encoder_id != encoder.bundle_sha256:
            raise ValueError("Configured tactile encoder ID differs from the loaded bundle")
        cfg.tactile_encoder_id = encoder.bundle_sha256
        cfg.tactile_projection_dim = encoder.projection_dim
        cfg.visuotactile_observation_terms = (
            "state", "aligned_tactile_z" if encoder.feature_mode == "aligned" else "pretrained_tactile_h",
        )
        cfg.single_frame_observation_dim = cfg.state_observation_dim + cfg.tactile_finger_count * encoder.projection_dim
        cfg.observation_space = cfg.single_frame_observation_dim * cfg.observation_history_length
        self.tactile_policy_contract = encoder.observation_contract(
            state_dim=cfg.state_observation_dim, history_length=cfg.observation_history_length,
        )
        self.tactile_policy_contract["simulation_encoder_sha256"] = encoder.bundle_sha256
        self.tactile_policy_contract["simulation_sensor_assets"] = {
            "marker_layout": file_sha256(cfg.tactile_marker_layout_path),
            "mounting_urdf": file_sha256(cfg.tactile_calibration_urdf),
            "camera_rectangles": file_sha256(
                Path(cfg.tactile_marker_layout_path).with_name("camera_ray_rectangles_320x240.json")
            ),
            "rgb_background": file_sha256(cfg.tactile_rgb_background_path),
        }
        _ensure_torch_scatter_compatibility()
        super().__init__(cfg, render_mode, **kwargs)
        self._policy_encoder = encoder.to(self.device)
        self._history = TactileObservationHistory(
            self.num_envs, cfg.state_observation_dim, encoder.projection_dim,
            cfg.observation_history_length, self.device,
        )
        self._observation_cache_key = None
        self.latest_tactile_reconstruction = None
        if cfg.repose_training_revision == 2:
            from .repose_training_runtime import ReposeTrainingRuntime

            self._repose_training = ReposeTrainingRuntime(self)
            self.tactile_policy_contract["repose_task"] = {
                "revision": 2,
                "action": "bounded_joint_velocity_increment",
                "action_speed_rad_s": cfg.repose_action_speed,
                "action_filter": cfg.repose_action_filter,
                "step_dt": self.step_dt,
            }
        elif cfg.repose_training_revision != 1:
            raise ValueError("Unsupported Repose training revision")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._repose_training is None:
            return super()._pre_physics_step(actions)
        self._repose_training.prepare_action(actions)

    def _apply_action(self) -> None:
        if self._repose_training is None:
            return super()._apply_action()
        self._repose_training.apply_action()

    def _get_rewards(self) -> torch.Tensor:
        if self._repose_training is None:
            return super()._get_rewards()
        return self._repose_training.rewards()

    def _reset_target_pose(self, env_ids):
        if self._repose_training is None:
            return super()._reset_target_pose(env_ids)
        self._repose_training.reset_target(env_ids)

    def _setup_scene(self):
        super()._setup_scene()
        self._tactile_runtime = DirectVisuotactileRuntime(self)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        key = (self.common_step_counter, getattr(self, "_rl_ours_tactile_reset_epoch", 0))
        if key == self._observation_cache_key:
            return {"policy": self._history.frames.flatten(1)}
        sample = self._compute_visuotactile_observations()
        inputs = {name: sample[name] for name in ("rgb", "depth_m", "marker", "marker_valid")}
        z = self._policy_encoder(**inputs)
        policy = self._history.append(self.compute_full_observations(), z)
        self.latest_tactile_depth = sample["depth_m"]
        self.latest_marker_motion = sample["marker"]
        self.latest_taxim_rgb = sample["rgb"]
        self.latest_tactile_latent = z
        if self.cfg.tactile_reconstruction_diagnostics:
            self.latest_tactile_reconstruction = self._policy_encoder.reconstruct(
                **inputs, rgb_reference=sample["rgb_reference"],
            )
        self._observation_cache_key = key
        return {"policy": policy}

    def _compute_visuotactile_observations(self) -> dict[str, torch.Tensor]:
        return self._tactile_runtime.observe(self._policy_encoder.normalization)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        if self._repose_training is None:
            super()._reset_idx(env_ids)
        else:
            self._repose_training.reset(env_ids)
        if hasattr(self, "_tactile_runtime"):
            self._tactile_runtime.reset(env_ids)
        if hasattr(self, "_history"):
            self._history.reset(env_ids)
        self._observation_cache_key = None
        self.latest_tactile_reconstruction = None

    def close(self):
        try:
            super().close()
        finally:
            if hasattr(self, "_tactile_runtime"):
                self._tactile_runtime.close()
