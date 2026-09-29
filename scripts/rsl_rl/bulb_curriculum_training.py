"""Checkpoint and stage-boundary integration for the existing RSL-RL entry point."""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType


class CurriculumStageComplete(Exception):
    """Normal early completion, raised only after a finished PPO update."""


def install_curriculum_runner(runner, env):
    if getattr(runner, "is_distributed", False):
        raise ValueError("Curriculum banks and promotion currently support one rank only")
    runtime = env._bulb_curriculum
    original_save, original_log, original_learn = runner.save, runner.log, runner.learn

    def save(self, path, infos=None):
        data = dict(infos or {})
        data["bulb_curriculum"] = runtime.state_dict()
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
        enough_bank = (runtime.cfg.stage not in (1, 2)
                       or len(runtime.bank.samples[runtime.cfg.stage + 1]) >= runtime.cfg.min_bank_size)
        if runtime.scheduler.ready and enough_bank:
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
        output = Path(self.log_dir)
        checkpoint = output / f"model_{self.current_learning_iteration}.pt"
        self.save(str(checkpoint))
        status = {"schema": 1, "stage": runtime.cfg.stage, "fixed_bulb": runtime.cfg.fixed_bulb,
                  "promotion_nominated": nominated, "full_budget_completed": runtime.cfg.stage == 4,
                  "checkpoint": str(checkpoint.resolve()), "scheduler": runtime.scheduler.state_dict(),
                  "bank_sizes": {str(k): len(v) for k, v in runtime.bank.samples.items()},
                  "episodes": runtime.episodes, "physics_transitions": runtime.transitions}
        temp = output / "curriculum_status.json.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(output / "curriculum_status.json")
        if self.writer is not None:
            self.writer.flush()
        return result

    runner.save = MethodType(save, runner)
    runner.log = MethodType(log, runner)
    runner.learn = MethodType(learn, runner)
