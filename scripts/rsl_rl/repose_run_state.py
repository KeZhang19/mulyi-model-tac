"""Preserve Repose curriculum progress in ordinary RSL-RL checkpoints."""
from types import MethodType


def install_repose_run_state(runner, env):
    runtime = env._repose_training
    original_save, original_load = runner.save, runner.load

    def save(self, path, infos=None):
        payload = dict(infos or {})
        payload["repose_training_state"] = runtime.state_dict()
        return original_save(path, infos=payload)

    def load(self, path, *args, **kwargs):
        infos = original_load(path, *args, **kwargs)
        if not isinstance(infos, dict) or "repose_training_state" not in infos:
            raise ValueError("Repose v2 checkpoint is missing its curriculum state")
        runtime.load_state_dict(infos["repose_training_state"])
        return infos

    runner.save = MethodType(save, runner)
    runner.load = MethodType(load, runner)
