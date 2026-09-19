"""Small explicit diagnostic logs for the Custom bulb compatibility replay."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class BulbDiagnosticRecorder:
    """Write one JSON record per control step, without changing the environment."""

    def __init__(self, env, directory):
        self.env = env.unwrapped
        self.path = Path(directory) / "rollout.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w")
        self.step = 0

    def record(self, actions, rewards, dones):
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom import env_cfg as bulb

        env = self.env
        loss = env.termination_manager.get_term_cfg("lost_grasp").func
        row = {
            "step": self.step,
            "seconds": (self.step + 1) * env.step_dt,
            "actions": actions.detach().cpu().tolist(),
            "reward": rewards.detach().cpu().tolist(),
            "done": dones.detach().cpu().tolist(),
            "termination": {name: env.termination_manager.get_term(name).cpu().tolist()
                            for name in env.termination_manager.active_terms},
            # Isaac Lab resets finished environments inside step(); distances
            # and contact below describe the returned (possibly reset) state.
            "returned_state_contact_scores": bulb.rotate_bulb_contact_scores(env).cpu().tolist(),
            "returned_state_released": env._bulb_released.cpu().tolist(),
            "returned_state_opposing_distance_m": bulb.rotate_bulb_hand_shell_distance(env).cpu().tolist(),
            "returned_state_no_contact_time_s": loss.no_contact_time.cpu().tolist(),
            "returned_state_far_time_s": loss.far_time.cpu().tolist(),
            "step_reward": {
                name: (env.reward_manager._step_reward[:, i] * env.step_dt).cpu().tolist()
                for i, name in enumerate(env.reward_manager.active_terms)
            },
            "finite": bool(torch.isfinite(actions).all() & torch.isfinite(rewards).all()),
        }
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        self.step += 1

    def close(self):
        self.stream.close()
