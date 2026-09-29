"""RewardManager adapter for the randomized guided-screw task only."""
from isaaclab.managers import ManagerTermBase

from ..RotateBulbCustom import env_cfg as original
from .progress_penalty import ProgressPenaltySettings, ProgressPenaltyState


class RandomizedProgressPenalty(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        settings = ProgressPenaltySettings(**env.cfg.unscrew_progress_penalty)
        if not env.cfg.bulb_randomized_task or env.cfg.bulb_pure_collision:
            raise ValueError("Progress penalties require the randomized guided task")
        if not hasattr(env, "_randomized_progress_penalty"):
            env._randomized_progress_penalty = ProgressPenaltyState(env.num_envs, env.device, settings)
        self.state = env._randomized_progress_penalty
        if self.state.settings != settings:
            raise ValueError("Progress penalty terms must share the same settings")

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env, component: str):
        if component not in ("regression", "stall"):
            raise ValueError("component must be regression or stall")
        regression, stall = self.state.update(
            original.rotate_bulb_unscrewed_turns(env), env._bulb_released,
            original.rotate_bulb_failed(env), env.episode_length_buf * env.step_dt,
            env.common_step_counter, env.step_dt,
        )
        # Each cost is already integrated in seconds/turns. RewardManager will
        # multiply by dt and the configured -1 weight exactly once.
        return (regression if component == "regression" else stall) / env.step_dt
