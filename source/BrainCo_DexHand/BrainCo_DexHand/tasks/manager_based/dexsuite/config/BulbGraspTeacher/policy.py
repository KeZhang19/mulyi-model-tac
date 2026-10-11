"""Optional single-frame LSTM teacher; leaves the frozen MLP defaults intact."""
import torch

from rsl_rl.modules import ActorCriticRecurrent


def configure_bulb_teacher_lstm(env_cfg, agent_cfg):
    """Apply the explicit --lstm mode before constructing the environment/runner."""
    for group in vars(env_cfg.observations).values():
        if hasattr(group, "history_length"):
            group.history_length = 1
    policy = agent_cfg.policy
    policy.class_name = "BulbGraspTeacherLSTM"
    policy.rnn_type = "lstm"
    policy.rnn_hidden_dim = 256
    policy.rnn_num_layers = 1


class BulbGraspTeacherLSTM(ActorCriticRecurrent):
    """RSL recurrent PPO with the teacher's noise bounds and bootstrap semantics."""

    def __init__(self, *args, min_noise_std=0.05, max_noise_std=0.5, **kwargs):
        # This task uses a learned, observation-independent std per action.
        if kwargs.get("state_dependent_std", False) or kwargs.get("noise_std_type", "scalar") != "scalar":
            raise ValueError("BulbGraspTeacherLSTM requires state-independent scalar noise.")
        if not 0 < min_noise_std <= max_noise_std:
            raise ValueError("Expected 0 < min_noise_std <= max_noise_std.")
        super().__init__(*args, **kwargs)
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

    def update_distribution(self, obs):
        # The upstream recurrent class does not implement the teacher's bounds.
        with torch.no_grad():
            self.std.clamp_(self.min_noise_std, self.max_noise_std)
        super().update_distribution(obs)

    def evaluate(self, obs, masks=None, hidden_states=None):
        if isinstance(masks, str) and masks == "compute_returns":
            # The task's PPO fork uses this sentinel for the rollout bootstrap.
            # Peek at V(s_T) without consuming s_T twice in the persistent memory.
            obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
            output, _ = self.memory_c.rnn(obs.unsqueeze(0), self.memory_c.hidden_states)
            return self.critic(output.squeeze(0))
        return super().evaluate(obs, masks=masks, hidden_states=hidden_states)

    def load_state_dict(self, state_dict, strict=True):
        if "memory_a.rnn.weight_ih_l0" not in state_dict:
            raise ValueError("--lstm requires an LSTM checkpoint; start a fresh run instead of loading a 5-frame MLP.")
        return super().load_state_dict(state_dict, strict=strict)
