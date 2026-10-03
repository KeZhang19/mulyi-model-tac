"""Checkpoint and stage-boundary integration for the existing RSL-RL entry point."""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType


class CurriculumStageComplete(Exception):
    """Normal early completion, raised only after a finished PPO update."""


def install_curriculum_runner(runner, env):
    runtime = env._bulb_curriculum
    distributed = bool(getattr(runner, "is_distributed", False))
    coordinator = None
    if distributed:
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbAssemblyCurriculum.distributed import DistributedCurriculum
        coordinator = DistributedCurriculum(runtime)
        original_update, original_iteration = runner.alg.update, runner.env.set_iteration

        def update(self, *args, **kwargs):
            result = original_update(*args, **kwargs)
            coordinator.synchronize()
            return result

        def set_iteration(iteration):
            if ready():
                raise CurriculumStageComplete()
            return original_iteration(iteration)

        runner.alg.update = MethodType(update, runner.alg)
        runner.env.set_iteration = set_iteration
    original_save, original_log, original_learn = runner.save, runner.log, runner.learn

    def ready():
        enough_bank = (runtime.cfg.stage not in (1, 2)
                       or len(runtime.bank.samples[runtime.cfg.stage + 1]) >= runtime.cfg.min_bank_size)
        return runtime.scheduler.ready and enough_bank

    def save(self, path, infos=None):
        if distributed and self.gpu_global_rank != 0:
            return
        data = dict(infos or {})
        data["bulb_curriculum"] = coordinator.cached_state if coordinator else runtime.state_dict()
        return original_save(path, infos=data)

    def log(self, locs, *args, **kwargs):
        result = original_log(locs, *args, **kwargs)
        if self.writer is not None:
            metrics = {"stage": runtime.cfg.stage, "level": runtime.scheduler.level,
                       "nomination_rate": runtime.scheduler.last_rate,
                       "physics_transitions": runtime.transitions,
                       "lift_bank_size": len(runtime.bank.samples[2]),
                       "hover_bank_size": len(runtime.bank.samples[3])}
            for key, value in metrics.items():
                self.writer.add_scalar("Curriculum/" + key, value, locs["it"])
        if ready() and not distributed:
            raise CurriculumStageComplete()
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("Curriculum episodes must start at a real reset, not a random episode clock")
        nominated = False
        try:
            result = original_learn(num_learning_iterations, init_at_random_ep_len=False)
        except CurriculumStageComplete:
            nominated, result = True, None
        nominated = nominated or ready()
        if distributed and self.gpu_global_rank != 0:
            return result
        output = Path(self.log_dir)
        checkpoint = output / f"model_{self.current_learning_iteration}.pt"
        self.save(str(checkpoint))
        status = {"schema": 1, "stage": runtime.cfg.stage, "fixed_bulb": runtime.cfg.fixed_bulb,
                  "promotion_nominated": nominated, "full_budget_completed": runtime.cfg.stage == 4,
                  "checkpoint": str(checkpoint.resolve()), "scheduler": runtime.scheduler.state_dict(),
                  "bank_sizes": {str(k): len(v) for k, v in runtime.bank.samples.items()},
                  "episodes": coordinator.cached_state["episodes"] if coordinator else runtime.episodes,
                  "physics_transitions": coordinator.cached_state["transitions"] if coordinator else runtime.transitions,
                  "world_size": coordinator.world_size if coordinator else 1}
        temp = output / "curriculum_status.json.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(output / "curriculum_status.json")
        if self.writer is not None:
            self.writer.flush()
        return result

    runner.save = MethodType(save, runner)
    runner.log = MethodType(log, runner)
    runner.learn = MethodType(learn, runner)
