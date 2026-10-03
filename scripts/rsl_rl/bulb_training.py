"""Custom bulb bounded policy, explicit legacy diagnostics and export contract."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from repose_training import ReposeActorCritic, _BoundLatentMean


BULB_POLICY_VERSION = 3
BULB_TASK = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0"
BULB_ASSEMBLY_TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"
BULB_CURRICULUM_TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0"
BULB_RANDOMIZED_TASK = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0"
BULB_TASKS = frozenset((BULB_TASK, BULB_ASSEMBLY_TASK, BULB_RANDOMIZED_TASK, BULB_CURRICULUM_TASK))
CONTRACT_FILENAME = "bulb_action_contract.json"


def is_bulb_task(task_name: str) -> bool:
    """Return whether a task uses the bounded Custom bulb policy contract."""
    return task_name.split(":")[-1] in BULB_TASKS


def _task_id(env) -> str:
    """Resolve the registered task id for the contract written beside a run."""
    raw = env.unwrapped
    for owner in (env, raw):
        spec = getattr(owner, "spec", None)
        task = getattr(spec, "id", None)
        if task in BULB_TASKS:
            return task
    if getattr(getattr(raw, "cfg", None), "bulb_curriculum", None) is not None:
        return BULB_CURRICULUM_TASK
    if getattr(getattr(raw, "cfg", None), "bulb_randomized_task", False):
        return BULB_RANDOMIZED_TASK
    if getattr(getattr(raw, "cfg", None), "bulb_assembly_task", False):
        return BULB_ASSEMBLY_TASK
    return BULB_TASK


class BulbActorCritic(ReposeActorCritic):
    """Tanh Gaussian with the existing noise floor and no noise ceiling.

    Reuse Repose's mean transform, tanh density and entropy, but initialize the
    Gaussian directly so the saved Repose v2 policy retains its capped noise.
    """

    maximum_std = None

    def __init__(self, *args, init_noise_std: float = 0.35,
                 noise_std_type: str = "log", state_dependent_std: bool = False, **kwargs):
        if state_dependent_std:
            raise ValueError("BulbActorCritic requires a state-independent standard deviation")
        if not math.isfinite(init_noise_std) or init_noise_std <= self.minimum_std:
            raise ValueError("Initial Bulb noise must be finite and greater than 0.05")
        ActorCritic.__init__(self, *args, init_noise_std=init_noise_std, noise_std_type="log",
                             state_dependent_std=False, **kwargs)
        # Inverse softplus: std = 0.05 + softplus(parameter), with no upper bound.
        delta = init_noise_std - self.minimum_std
        initial_parameter = delta + math.log(-math.expm1(-delta))
        self.noise_unconstrained = nn.Parameter(torch.full_like(self.log_std, initial_parameter))
        del self.log_std
        self.latent_mean_bound = _BoundLatentMean()
        self.register_buffer("_bulb_policy_version", torch.tensor(BULB_POLICY_VERSION, dtype=torch.int64))
        output_layer = [layer for layer in self.actor.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.orthogonal_(output_layer.weight, gain=0.01)
        nn.init.zeros_(output_layer.bias)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.latent_mean_bound(self.actor(obs))
        std = self.minimum_std + nn.functional.softplus(self.noise_unconstrained)
        self.distribution = Normal(mean, std.expand_as(mean))


def register_bulb_policy() -> None:
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.BulbActorCritic = BulbActorCritic


def _load_contract(checkpoint_path: str | Path) -> dict:
    path = Path(checkpoint_path).parent / CONTRACT_FILENAME
    if not path.is_file():
        raise ValueError(f"Bounded Bulb checkpoint is missing its action contract: {path}")
    contract = json.loads(path.read_text())
    if (contract.get("schema") != "rotate_bulb_custom_action_contract"
            or contract.get("version") != BULB_POLICY_VERSION):
        raise ValueError("Unsupported Bulb action contract schema or version")
    if contract.get("legacy_diagnostic"):
        raise ValueError("Legacy diagnostic artifacts cannot be used as a bounded Bulb checkpoint")
    return contract


def configure_bulb_checkpoint(agent_cfg, checkpoint_path: str | None = None, *,
                              legacy_diagnostic: bool = False, training: bool = False) -> bool:
    """Select the policy before runner construction; never migrate old PPO state.

    Returns True only for explicitly requested legacy diagnostic play. That mode
    loads the old ActorCritic unchanged and clips its inference output separately.
    """
    if (getattr(agent_cfg, "class_name", None) != "OnPolicyRunner"
            or getattr(getattr(agent_cfg, "algorithm", None), "class_name", None) != "PPO"):
        raise ValueError("Custom bulb bounded training/play supports only OnPolicyRunner with PPO")
    register_bulb_policy()
    if legacy_diagnostic and (training or checkpoint_path is None):
        raise ValueError("--bulb-legacy-diagnostic is only supported when playing an old checkpoint")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        if "_repose_policy_version" in state:
            raise ValueError("A Repose checkpoint cannot be loaded as a Custom bulb policy")
        marker = state.get("_bulb_policy_version")
        if marker is None:
            if training:
                raise ValueError("Legacy Bulb checkpoints cannot resume bounded PPO; start a new training run")
            if not legacy_diagnostic:
                raise ValueError("Legacy Bulb checkpoint requires explicit --bulb-legacy-diagnostic")
            if "noise_logit" in state or not any(key in state for key in ("std", "log_std")):
                raise ValueError("Checkpoint has no recognized legacy Gaussian policy state")
            config_path = Path(checkpoint_path).parent / "params" / "agent.yaml"
            if not config_path.is_file():
                raise ValueError(f"Legacy diagnostic requires the saved agent configuration: {config_path}")
            saved = yaml.safe_load(config_path.read_text())
            if saved.get("class_name") != "OnPolicyRunner" or saved.get("algorithm", {}).get("class_name") != "PPO":
                raise ValueError("Legacy diagnostic requires a saved OnPolicyRunner/PPO configuration")
            saved_policy = saved.get("policy", {})
            if saved_policy.get("class_name") != "ActorCritic" or saved_policy.get("state_dependent_std", False):
                raise ValueError("Legacy diagnostic only supports the saved state-independent ActorCritic")
            for name in ("actor_hidden_dims", "critic_hidden_dims", "activation", "actor_obs_normalization",
                         "critic_obs_normalization", "init_noise_std"):
                if name not in saved_policy:
                    raise ValueError(f"Legacy agent configuration is missing policy.{name}")
                setattr(agent_cfg.policy, name, copy.deepcopy(saved_policy[name]))
            if "obs_groups" not in saved:
                raise ValueError("Legacy agent configuration is missing obs_groups")
            agent_cfg.obs_groups = copy.deepcopy(saved["obs_groups"])
            agent_cfg.empirical_normalization = saved.get("empirical_normalization")
            agent_cfg.policy.class_name = "ActorCritic"
            agent_cfg.policy.noise_std_type = "log" if "log_std" in state else "scalar"
            agent_cfg.policy.state_dependent_std = False
            agent_cfg.clip_actions = None
            print("[Bulb] LEGACY DIAGNOSTIC: original weights/normalizer, clipped actions, corrected task; "
                  "this is not an evaluation under the original training dynamics.", flush=True)
            return True
        if marker.numel() == 1 and int(marker.item()) == 2:
            raise ValueError("Bulb v2 checkpoint uses small action scales and capped noise; "
                             "the v3 action/noise contract requires a new training run")
        if marker.numel() != 1 or int(marker.item()) != BULB_POLICY_VERSION:
            raise ValueError(f"Unsupported Bulb policy version: {marker}")
        if legacy_diagnostic:
            raise ValueError("--bulb-legacy-diagnostic cannot be applied to a bounded Bulb checkpoint")
        _load_contract(checkpoint_path)
    agent_cfg.policy.class_name = "BulbActorCritic"
    agent_cfg.policy.noise_std_type = "log"
    agent_cfg.policy.state_dependent_std = False
    agent_cfg.policy.init_noise_std = 0.35
    agent_cfg.clip_actions = None
    return False


def prepare_bulb_policy_run(env, log_dir: str | Path, *, resume_path: str | None = None,
                          legacy_diagnostic: bool = False) -> dict:
    """Validate the runtime action contract and persist it beside run artifacts."""
    raw = env.unwrapped
    action = raw.action_manager.get_term("action")
    if not callable(getattr(action, "action_contract", None)):
        raise ValueError("Custom bulb bounded policies require BulbJointPositionAction")
    dimensions = raw.observation_manager.group_obs_dim
    observation_dim = sum(int(dimensions[name][0]) for name in ("policy", "proprio", "perception"))
    guidance = "rotation_subgoal" in getattr(raw.observation_manager, "active_terms", {}).get("policy", [])
    expected_dim = 673 if guidance else 669
    curriculum = getattr(getattr(raw, "cfg", None), "bulb_curriculum", None)
    if curriculum is not None and getattr(curriculum, "version", 1) == 2:
        # v2 state-only removes 285 taxels and 15 contact-force columns.
        # An enabled five-finger encoder adds 5 * 128 latent columns.
        expected_dim = 1309 if raw.cfg.tactile_policy_enabled else 369
    randomized = getattr(getattr(raw, "cfg", None), "bulb_randomized_task", False)
    if randomized:
        if not guidance or legacy_diagnostic:
            raise ValueError("Randomized bulb requires guided rotation and its own policy contract")
        expected_dim += 14
    if observation_dim != expected_dim:
        raise ValueError(f"Custom bulb requires a {expected_dim}-dimensional observation, got {observation_dim}")
    contract = {
        "schema": "rotate_bulb_custom_action_contract", "version": BULB_POLICY_VERSION,
        "task": _task_id(env), "observation_dim": observation_dim,
        "legacy_diagnostic": bool(legacy_diagnostic),
        "policy_transform": "clip_normalized_action" if legacy_diagnostic else "bounded_latent_mean_then_tanh",
        "latent_std_bounds": None if legacy_diagnostic else [BulbActorCritic.minimum_std, None],
        "action": action.action_contract(),
    }
    if curriculum is not None and getattr(curriculum, "version", 1) == 2:
        contract["assembly_curriculum"] = {
            "version": 2, "reset": "free_horizontal_tabletop",
            "sensor_mode": "tactile" if raw.cfg.tactile_policy_enabled else "state_only",
        }
    if guidance:
        if legacy_diagnostic or "Bulb-Assembly" in contract["task"]:
            raise ValueError("Rotation subgoals require a new guided-rotation policy, not an assembly or legacy policy")
        contract["rotation_guidance"] = {
            "version": 1,
            "observation": ["frontier_fraction", "next_target_fraction", "segment_distance_div_12", "contact_memory"],
            "turn_reward_params": dict(raw.reward_manager.get_term_cfg("unscrew_turns").params),
        }
    directory = Path(log_dir)
    if randomized:
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbRandomized.mdp import randomized_contract
        contract["randomized_task"] = randomized_contract(raw)
    if legacy_diagnostic and resume_path is not None:
        source = Path(resume_path).resolve().parent
        if directory.resolve() == source or source in directory.resolve().parents:
            raise ValueError("Legacy diagnostic output must be outside the checkpoint directory")
        contract["source_checkpoint"] = str(Path(resume_path).resolve())
    if resume_path is not None and not legacy_diagnostic:
        saved = _load_contract(resume_path)
        if saved != contract:
            raise ValueError("Bulb checkpoint action/observation contract does not match this environment")
    path = directory / CONTRACT_FILENAME
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f"Refusing to overwrite a different Bulb action contract: {path}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def bulb_inference_policy(policy, *, legacy_diagnostic: bool = False):
    """Adapt only old diagnostic outputs; new policy already applies its tanh."""
    if not legacy_diagnostic:
        return policy

    def clipped(obs):
        return policy(obs).clamp(-1.0, 1.0)

    return clipped


def bulb_policy_for_export(policy, *, legacy_diagnostic: bool = False):
    """Preserve inference transforms in native JIT/ONNX actor-only exports."""
    if isinstance(policy, BulbActorCritic):
        if legacy_diagnostic:
            raise ValueError("A bounded Bulb policy cannot use legacy diagnostic export")
        transforms = [copy.deepcopy(policy.latent_mean_bound), nn.Tanh()]
    elif legacy_diagnostic and not isinstance(policy, ReposeActorCritic):
        transforms = [nn.Hardtanh(-1.0, 1.0)]
    else:
        raise ValueError("Unrecognized Bulb policy; legacy export requires explicit diagnostic mode")
    # list(actor) retains repeated activation modules; children() deduplicates them.
    actor = nn.Sequential(*copy.deepcopy(list(policy.actor)), *transforms)
    return SimpleNamespace(actor=actor, is_recurrent=False)
