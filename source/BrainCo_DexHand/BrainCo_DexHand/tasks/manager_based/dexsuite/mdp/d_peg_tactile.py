"""D-peg task contract around the existing calibrated, frozen tactile runtime."""

from dataclasses import asdict
import copy
import math
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase

from BrainCo_DexHand.tactile_representation.policy import file_sha256
from .d_peg_insertion import validate_d_peg_geometry_metadata
from .d_peg_tactile_runtime import enable_d_peg_fast_taxim
from .rotate_bulb_tactile import (
    RotateBulbPretrainedTactile,
    initialize_rotate_bulb_tactile_contract,
    tactile_obs,
)


class DPegPretrainedTactile(RotateBulbPretrainedTactile):
    """Share sensor calibration, frozen inference, cache and partial-reset behavior."""

    _HISTORY_BUFFERS = ("_batch_hydrosoft_forces", "_batch_prev_sdf", "_batch_prev_indenter_points",
                        "_batch_prev_normals", "_batch_state_valid", "_batch_prev_sample_ids")
    _HISTORY_LISTS = ("_hydrosoft_forces", "_prev_sdf", "_prev_indenter_points", "_prev_normals")

    def __init__(self, cfg, env):
        # Read the final configuration at manager construction, after Hydra
        # overrides. Disabled observations must not even load encoder assets.
        self.enabled = bool(env.cfg.tactile_policy_enabled)
        if self.enabled:
            super().__init__(cfg, env)
            return
        ManagerTermBase.__init__(self, cfg, env)
        self.encoder = None
        self.component_cfg = copy.deepcopy(cfg.params["component_cfg"])
        self._cache_key = self._cached = None
        self._disabled_obs = torch.empty((env.num_envs, 0), device=env.device)
        env._rotate_bulb_tactile_term = self
        env.latest_tactile_inputs = None
        env.latest_tactile_latent = None
        env.latest_tactile_reconstruction = None
        print("[INFO] D-peg restoration observations disabled: skipping RGB/Depth/marker "
              "generation and encoder inference.", flush=True)

    def reset(self, env_ids=None):
        env = self._env
        if not getattr(self, "enabled", True):
            tactile_obs.invalidate_ours_tactile_cache_on_reset(env, env_ids)
            return
        all_ids = torch.arange(env.num_envs, device=env.device)
        ids = all_ids if env_ids is None else (all_ids[env_ids] if isinstance(env_ids, slice)
                                              else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
        pending = getattr(self, "_d_peg_pending_reset", None)
        if pending is not None and pending["step"] != env.common_step_counter:
            pending = None
        cached_now = self._cache_key is not None and self._cache_key[0] == env.common_step_counter
        # Normal training resets before this step's observations, so this
        # snapshot path is only needed for a same-step re-read (e.g. recording).
        if cached_now and self._cached is not None and len(ids) < env.num_envs:
            adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
            pending = dict(step=env.common_step_counter, keep=torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
                           adapter=adapter, latent=self._cached,
                           inputs=getattr(env, "latest_tactile_inputs", None), buffers={}, lists={})
            if adapter is not None:
                pending["buffers"] = {name: value.clone() for name in self._HISTORY_BUFFERS
                                      if isinstance(value := getattr(adapter, name, None), torch.Tensor)}
                pending["lists"] = {name: [value.clone() if isinstance(value, torch.Tensor) else value for value in values]
                                    for name in self._HISTORY_LISTS
                                    if isinstance(values := getattr(adapter, name, None), list)}
        if pending is not None:
            pending["keep"][ids] = False
            if not pending["keep"].any():
                pending = None
        self._d_peg_pending_reset = pending
        super().reset(ids)

    @torch.no_grad()
    def __call__(self, env, component_cfg):
        if not getattr(self, "enabled", True):
            return self._disabled_obs
        if getattr(getattr(env, "cfg", None), "d_peg_fast_taxim", False):
            adapter = tactile_obs._ours_taxim_rgb_adapter(env, self.component_cfg)
            env._rl_ours_taxim_rgb_adapter = enable_d_peg_fast_taxim(adapter)
        pending = getattr(self, "_d_peg_pending_reset", None)
        result = super().__call__(env, component_cfg)
        self._d_peg_pending_reset = None
        if pending is None or pending["step"] != env.common_step_counter:
            return result
        keep = pending["keep"]
        slots = keep.repeat_interleave(5)
        adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
        if pending["adapter"] is not None:
            if adapter is not pending["adapter"]:
                raise RuntimeError("HydroShear adapter changed during a same-step partial reset")
            for name, previous in pending["buffers"].items():
                current = getattr(adapter, name)
                if current.shape != previous.shape:
                    raise RuntimeError(f"HydroShear {name} shape changed during partial reset")
                current[slots] = previous[slots]
            slot_ids = slots.nonzero(as_tuple=False).flatten().tolist()
            for name, previous in pending["lists"].items():
                current = getattr(adapter, name)
                for index in slot_ids:
                    if index < len(previous):
                        current[index] = previous[index]
        result[keep] = pending["latent"][keep]
        latest = getattr(env, "latest_tactile_inputs", None)
        if pending["inputs"] is not None and latest is not None:
            for name, previous in pending["inputs"].items():
                latest[name][keep] = previous[keep]
        return result


def initialize_d_peg_tactile_contract(env, env_ids=None):
    """Record the insertion semantics and assets even when dimensions match bulb PPO."""
    # Hydra applies overrides after cfg.__post_init__: validate the final values.
    env.cfg._d_peg_geometry = asdict(validate_d_peg_geometry_metadata(
        env.cfg.geometry_metadata_path, env.cfg.insertion_depth_m, env.cfg.socket_mouth_height_m,
    ))
    enabled = bool(getattr(env.cfg, "tactile_policy_enabled", True))
    if enabled:
        initialize_rotate_bulb_tactile_contract(env, env_ids)
    else:
        manager = env.observation_manager
        groups = ("policy", "proprio", "perception")
        dimension = sum(int(manager.group_obs_dim[group][0]) for group in groups)
        env.tactile_policy_contract = dict(
            schema_version=1, feature="disabled", tactile_policy_enabled=False,
            projection_dim=0, state_dim=dimension, observation_dim=dimension,
            history_length=1, frame_order=list(groups),
            observation_groups={group: [dict(name=name, shape=[int(size) for size in shape])
                for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group], strict=True)]
                for group in groups},
        )
    contract = env.tactile_policy_contract
    expected = {"policy": 43, "proprio": 1714 if enabled else 434, "perception": 192}
    actual = {group: sum(math.prod(term["shape"]) for term in contract["observation_groups"][group])
              for group in expected}
    dimension = 1949 if enabled else 669
    projection = 256 if enabled else 0
    if (actual != expected or contract["observation_dim"] != dimension
            or contract["projection_dim"] != projection or contract["history_length"] != 1):
        raise ValueError(f"D-peg requires {dimension} current-frame observations and {projection} features per finger: {actual}")
    # Custom scene variants own their checkpoint contract while the original
    # task keeps its historical identifier for backwards compatibility.
    contract["task"] = getattr(
        env.cfg, "task_id", "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0"
    )
    # This describes the implemented reward/state-machine semantics, so it
    # cannot be overridden to claim compatibility with a v2 checkpoint.
    contract["task_schema_version"] = 3
    contract["action_schema"] = env.action_manager.get_term("action").action_contract()
    contract["reward_geometry"] = dict(env.cfg._d_peg_geometry)
    contract["task_assets"] = {
        name: file_sha256(Path(path)) for name, path in {
            "peg": env.cfg.peg_usd_path,
            "socket": env.cfg.socket_usd_path,
            "geometry": env.cfg.geometry_metadata_path,
            "pregrasp": env.cfg.pregrasp_path,
        }.items()
    }
    # Custom scene variants must pin the robot USD as well as the peg/socket;
    # this prevents loading a checkpoint with the legacy Flexiv asset whose
    # joint and tactile-link inventory is different.
    if getattr(env.cfg, "task_id", "").endswith("-Custom-v0"):
        robot_spawn = getattr(getattr(env.scene["robot"], "cfg", None), "spawn", None)
        robot_path = getattr(robot_spawn, "usd_path", None)
        if robot_path:
            contract["task_assets"]["robot"] = file_sha256(Path(str(robot_path)))
    contract["task_parameters"] = {
        "socket_mouth_height_m": env.cfg.socket_mouth_height_m,
        "insertion_depth_m": env.cfg.insertion_depth_m,
        "socket_xy_randomization_m": env.cfg.socket_xy_randomization_m,
        "socket_yaw_randomization_deg": env.cfg.socket_yaw_randomization_deg,
        "episode_length_s": env.cfg.episode_length_s,
        "action_scale": env.cfg.actions.action.scale,
    }
