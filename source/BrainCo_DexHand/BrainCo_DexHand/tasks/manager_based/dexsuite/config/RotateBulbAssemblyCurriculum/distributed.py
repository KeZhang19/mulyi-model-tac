"""Synchronize curriculum only at PPO boundaries, never during scene resets."""
from copy import deepcopy

import torch.distributed as dist


def restore_worker_state(saved, *, seed, world_size, rank, evaluation=False):
    """Select the originating worker's counters/RNG without relaxing physics checks."""
    metadata = saved.get("distributed")
    if evaluation:
        return saved
    if metadata is None:
        if world_size != 1:
            raise ValueError("A single-rank curriculum checkpoint cannot resume distributed training")
        return saved
    if metadata.get("schema") != 1 or metadata["world_size"] != world_size:
        raise ValueError("Curriculum resume requires the saved distributed world size")
    worker = metadata["ranks"][rank]
    if worker["rank"] != rank or worker["seed"] != seed:
        raise ValueError("Curriculum resume rank/seed mismatch")
    bank = saved["bank"]
    # All workers share the same physically proven bank. Its contract seed is
    # the sampler seed; physical source/asset/action fields stay unchanged.
    bank = {**bank, "contract": {**bank["contract"], "seed": seed}}
    return {**saved, "bank": bank, **{key: worker[key] for key in ("rng", "episodes", "transitions")}}


class DistributedCurriculum:
    """Merge real episode results and successful prefixes in deterministic rank order.

    The simulator remains rank-local. No collective is allowed inside reset:
    different workers finish different numbers of episodes. Checkpoint data is
    cached at a boundary so rank-zero-only RSL save calls cannot deadlock.
    """

    def __init__(self, runtime):
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise ValueError("Distributed curriculum requires an initialized process group")
        self.runtime = runtime
        self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
        self.observations, self.prefixes = [], []
        self.observe = runtime.scheduler.observe
        self.add = runtime.bank.add
        self.staging = type(runtime.bank)(1, runtime.bank.contract)
        signature = {**runtime.contract, "seed": 0}
        signatures = [None] * self.world_size
        dist.all_gather_object(signatures, (signature, runtime.state_dict()["settings"],
                                            runtime.scheduler.state_dict(), runtime.bank.seen))
        if any(item != signatures[0] for item in signatures):
            raise ValueError("Distributed workers differ in physics, settings, scheduler or bank history")
        runtime.scheduler.observe = self.defer_observation
        runtime.bank.add = self.defer_prefix
        self.cached_state = None
        self.synchronize()

    def defer_observation(self, source, level, success):
        self.observations.append((source, level, bool(success)))

    def defer_prefix(self, source, tensors, provenance):
        # Reuse the authoritative finite/provenance checks and CPU snapshot.
        self.staging.add(source, tensors, {**provenance, "rank": self.rank})
        self.prefixes.append((source, self.staging.samples[source][0]))

    def synchronize(self):
        runtime = self.runtime
        worker = {"rank": self.rank, "seed": int(runtime.env.cfg.seed), "rng": runtime.rng.get_state(),
                  "episodes": runtime.episodes, "transitions": runtime.transitions}
        payloads = [None] * self.world_size
        dist.all_gather_object(payloads, (self.observations, self.prefixes, worker))
        for observations, prefixes, _ in payloads:
            for source, level, success in observations:
                self.observe(source, level, success)
            for source, item in prefixes:
                self.add(source, item["state"], item["provenance"])
        self.observations.clear()
        self.prefixes.clear()
        states = [None] * self.world_size
        state = (runtime.scheduler.state_dict(), runtime.bank.seen,
                 {source: len(items) for source, items in runtime.bank.samples.items()})
        dist.all_gather_object(states, state)
        if any(item != states[0] for item in states):
            raise RuntimeError("Curriculum state diverged across workers after PPO update")
        self.cached_state = deepcopy(runtime.state_dict())
        self.cached_state["distributed"] = {"schema": 1, "world_size": self.world_size,
                                             "ranks": [p[2] for p in payloads]}
        self.cached_state["episodes"] = sum(p[2]["episodes"] for p in payloads)
        self.cached_state["transitions"] = sum(p[2]["transitions"] for p in payloads)
        return self.cached_state
