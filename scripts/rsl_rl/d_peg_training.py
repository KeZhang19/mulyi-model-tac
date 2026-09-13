"""D-peg-only PPO initialization and observation-only distributed diagnostics.

The hooks observe the original rollout, minibatch iterator and log-probability
evaluation. They do not resample actions, change likelihoods, normalize inputs,
adjust learning rates, or replace the PPO loss/optimizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist


def _require_d_peg_runner(runner):
    env = runner.env.unwrapped
    contract = getattr(env, "tactile_policy_contract", {})
    if contract.get("task") not in {
        "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0",
        "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-Custom-v0",
        "BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0",
    }:
        raise ValueError("D-peg PPO hooks require the D-peg tactile policy contract")


def initialize_d_peg_policy(runner):
    """Set the fresh actor's deterministic residual to zero; leave critic/noise intact.

    Call only for a new training run, before learning. Resume and play must load
    their saved actor without applying this initialization.
    """
    _require_d_peg_runner(runner)
    policy = runner.alg.policy
    if getattr(policy, "state_dependent_std", False):
        raise ValueError("D-peg initialization requires a separate action-noise parameter")
    layers = [module for module in policy.actor.modules() if isinstance(module, torch.nn.Linear)]
    if not layers or layers[-1].out_features != 28:
        raise ValueError("D-peg actor must end with a 28-output Linear layer")
    with torch.no_grad():
        layers[-1].weight.zero_()
        if layers[-1].bias is not None:
            layers[-1].bias.zero_()
    return {"actor_output_initialized_to_zero": True, "action_dim": 28}


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _normalizer_snapshot(policy):
    result = {}
    for name in ("actor_obs_normalizer", "critic_obs_normalizer"):
        normalizer = getattr(policy, name, None)
        if normalizer is not None and all(hasattr(normalizer, key) for key in ("_mean", "_std", "count")):
            result[name] = {key: getattr(normalizer, key).detach().clone()
                            for key in ("_mean", "_std", "count")}
    return result


def _normalizer_deltas(before, after, prefix):
    result = {}
    for name in before.keys() & after.keys():
        short = name.removesuffix("_obs_normalizer")
        for field, label in (("_mean", "mean"), ("_std", "std")):
            result[f"{prefix}_{short}_{label}_max"] = (after[name][field] - before[name][field]).abs().max()
        result[f"{prefix}_{short}_count_delta"] = (after[name]["count"] - before[name]["count"]).double()
    return result


@torch.no_grad()
def _normalizer_rank_differences(snapshot):
    """Observe rank-local running-moment drift without synchronizing buffers."""
    values = [(name, field, snapshot[name][field]) for name in sorted(snapshot)
              for field in ("_mean", "_std")]
    if not values:
        return {}
    packed = torch.cat([value.reshape(-1) for _, _, value in values])
    reference = packed.clone()
    if _distributed():
        dist.broadcast(reference, src=0)
    result, offset = {}, 0
    for name, field, value in values:
        length = value.numel()
        result[f"normalizer_{name.removesuffix('_obs_normalizer')}_rank0{field}_difference_max"] = (
            packed[offset:offset + length] - reference[offset:offset + length]).abs().max()
        offset += length
    return result


@torch.no_grad()
def _reduce_scalars(values, operation):
    """Reduce a fixed, sorted scalar schema with one collective."""
    if not values:
        return {}
    keys = sorted(values)
    packed = torch.stack([values[key].detach().reshape(()).double() for key in keys])
    if _distributed():
        dist.all_reduce(packed, op=operation)
    return dict(zip(keys, packed.cpu().tolist(), strict=True))


@torch.no_grad()
def reduce_episode_diagnostics(payload, device):
    """Aggregate completed-episode sums with their true cross-rank denominator.

    The provider drains a rollout's completed episodes and returns count, sums
    and maxima. Every rank must return the same metric keys, including when its
    count is zero. Empty averages/maxima are represented by None in JSON.
    """
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    sums = {name: tensor(value) for name, value in payload.get("sums", {}).items()}
    sums["__episode_count"] = tensor(payload["count"])
    summed = _reduce_scalars(sums, dist.ReduceOp.SUM)
    count = summed.pop("__episode_count")
    maxima = _reduce_scalars({name: tensor(value) for name, value in payload.get("maxima", {}).items()},
                            dist.ReduceOp.MAX)
    return {"completed_episodes": int(count),
            "means": {name: value / count if count else None for name, value in summed.items()},
            "maxima": {name: value if count else None for name, value in maxima.items()}}


def install_d_peg_training_diagnostics(runner, *, episode_metrics_provider=None):
    """Write one globally reduced JSONL/TensorBoard record per PPO update.

    ``episode_metrics_provider`` is an optional zero-argument callback returning
    the schema documented by ``reduce_episode_diagnostics``. Install this after
    checkpoint load and existing distributed audit/normalizer hooks, on ALL ranks.
    """
    _require_d_peg_runner(runner)
    if getattr(runner, "_d_peg_training_diagnostics_installed", False):
        raise RuntimeError("D-peg diagnostics were already installed")
    algorithm, policy = runner.alg, runner.alg.policy
    if any(getattr(policy, name, False) for name in ("is_recurrent", "is_sequence_model", "is_transformerxl")):
        raise ValueError("D-peg diagnostics support the approved feedforward MLP only")
    if getattr(algorithm, "symmetry", None) or getattr(algorithm, "rnd", None):
        raise ValueError("D-peg diagnostics expect PPO without symmetry augmentation or RND")
    storage = algorithm.storage
    if storage.actions.shape[-1] != 28:
        raise ValueError("D-peg diagnostics require 28 actions")
    original_act, original_update = algorithm.act, algorithm.update
    original_generator = storage.mini_batch_generator
    original_log_prob = policy.get_actions_log_prob
    rank = dist.get_rank() if _distributed() else 0
    start_iteration = int(runner.current_learning_iteration)
    state = {"active": False, "batch": None, "rollout_start": None, "updates": 0}

    def act(algorithm, *args, **kwargs):
        if state["rollout_start"] is None:
            state["rollout_start"] = _normalizer_snapshot(policy)
        return original_act(*args, **kwargs)

    def generator(storage, *args, **kwargs):
        for index, batch in enumerate(original_generator(*args, **kwargs)):
            state["batch"] = batch
            state["batch_index"] = index
            state["batch_observed"] = False
            if index == 0:
                state["maxima"].update(_normalizer_deltas(
                    state["update_start"], _normalizer_snapshot(policy), "normalizer_update_entry"))
            yield batch
            lr = float(algorithm.optimizer.param_groups[0]["lr"])
            state["lr_min"] = min(state["lr_min"], lr)
            state["lr_max"] = max(state["lr_max"], lr)
        state["batch"] = None

    def log_prob(policy, actions):
        result = original_log_prob(actions)
        if not state["active"] or state["batch"] is None or state["batch_observed"]:
            return result
        state["batch_observed"] = True
        batch = state["batch"]
        with torch.no_grad():
            old_log_prob, old_mean, old_std = batch[5], batch[6], batch[7]
            mean, std = policy.action_mean, policy.action_std
            # Match the scheduler's exact epsilon convention, including its tiny
            # positive self-KL; report this so it can be compared with desired_kl.
            kl = (torch.log(std / old_std + 1.0e-5)
                  + (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - .5).sum(-1)
            ratio = (result.detach() - old_log_prob.squeeze(-1)).exp()
            clipped = ((ratio - 1).abs() > algorithm.clip_param).double()
            sums, maxima = state["sums"], state["maxima"]
            for key, value in {"kl_sum": kl.sum(), "ratio_clip_count": clipped.sum(),
                               "sample_count": kl.new_tensor(kl.numel()), "minibatches": kl.new_tensor(1)}.items():
                sums[key] = sums.get(key, 0) + value.double()
            maxima["kl_max"] = torch.maximum(maxima.get("kl_max", kl.max()), kl.max())
            if state["batch_index"] == 0:
                sums.update(first_minibatch_kl_sum=kl.sum().double(),
                            first_minibatch_sample_count=kl.new_tensor(kl.numel()).double(),
                            first_minibatch_ratio_clip_count=clipped.sum())
                maxima["first_minibatch_kl_max"] = kl.max()
        return result

    def update(algorithm, *args, **kwargs):
        state.update(active=True, batch=None, sums={}, maxima={},
                     lr_min=float("inf"), lr_max=float("-inf"),
                     update_start=_normalizer_snapshot(policy))
        state["maxima"].update(_normalizer_deltas(
            state["rollout_start"] or state["update_start"], state["update_start"], "normalizer_rollout"))
        state["maxima"].update(_normalizer_rank_differences(state["update_start"]))
        actions = storage.actions.detach()
        with torch.no_grad():
            state["sums"].update(action_sum=actions.double().sum(), action_square_sum=actions.double().square().sum(),
                                 action_count=actions.new_tensor(actions.numel()).double(),
                                 action_over_one=(actions.abs() > 1).double().sum(),
                                 policy_std_sum=storage.sigma.double().sum())
            state["maxima"]["action_abs_max"] = actions.abs().max()
        lr_before = float(algorithm.learning_rate)
        try:
            result = original_update(*args, **kwargs)
        finally:
            state["active"] = False
            state["batch"] = None
            state["rollout_start"] = None
        if "first_minibatch_kl_sum" not in state["sums"]:
            raise RuntimeError("PPO completed without the expected minibatch log-probability evaluation")
        sums = _reduce_scalars(state["sums"], dist.ReduceOp.SUM)
        maxima = _reduce_scalars(state["maxima"], dist.ReduceOp.MAX)
        count, first_count, action_count = sums["sample_count"], sums["first_minibatch_sample_count"], sums["action_count"]
        action_mean = sums["action_sum"] / action_count
        metrics = {
            "first_minibatch_pre_update_kl_mean": sums["first_minibatch_kl_sum"] / first_count,
            "first_minibatch_ratio_clip_frac": sums["first_minibatch_ratio_clip_count"] / first_count,
            "kl_mean": sums["kl_sum"] / count,
            "ratio_clip_frac": sums["ratio_clip_count"] / count,
            "learning_rate_before": lr_before, "learning_rate_after": float(algorithm.learning_rate),
            "learning_rate_used_min": state["lr_min"], "learning_rate_used_max": state["lr_max"],
            "action_mean": action_mean,
            "action_std": max(0., sums["action_square_sum"] / action_count - action_mean ** 2) ** .5,
            "action_over_one_frac": sums["action_over_one"] / action_count,
            "policy_noise_std_mean": sums["policy_std_sum"] / action_count,
            **maxima,
        }
        episode = reduce_episode_diagnostics(episode_metrics_provider(), runner.device) if episode_metrics_provider else None
        iteration = start_iteration + state["updates"]
        record = {"schema_version": 1, "iteration": iteration, "world_size": dist.get_world_size() if _distributed() else 1,
                  "minibatches_per_rank": int(sums["minibatches"] / (dist.get_world_size() if _distributed() else 1)),
                  "ppo": metrics, "episodes": episode}
        # Enforce finite JSON before any record is published. A failure here is a
        # numerical failure in the original PPO computation, not silently hidden.
        serialized = json.dumps(record, allow_nan=False)
        runner.d_peg_training_diagnostics = record
        if rank == 0 and runner.log_dir:
            path = Path(runner.log_dir) / "d_peg_training.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
            writer = getattr(runner, "writer", None)
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(f"DPegPPO/{name}", value, iteration)
                if episode:
                    writer.add_scalar("DPegGlobal/completed_episodes", episode["completed_episodes"], iteration)
                    for kind in ("means", "maxima"):
                        for name, value in episode[kind].items():
                            if value is not None:
                                writer.add_scalar(f"DPegGlobal/{kind}/{name}", value, iteration)
        state["updates"] += 1
        return result

    algorithm.act = MethodType(act, algorithm)
    algorithm.update = MethodType(update, algorithm)
    storage.mini_batch_generator = MethodType(generator, storage)
    policy.get_actions_log_prob = MethodType(log_prob, policy)
    runner._d_peg_training_diagnostics_installed = True
    return True
