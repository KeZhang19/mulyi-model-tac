"""Teacher policies with fixed-scale task state and optional single-frame LSTM."""
import torch

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.networks import EmpiricalNormalization


def stack_bulb_observation_frames(obs, groups):
    """Join same-time channels, then flatten frames oldest to newest.

    Isaac's manager flattens each term's history before joining terms/groups.
    Undo that packing explicitly: each network frame is policy(39), proprio(149),
    perception(192), task_state(15), rather than adjacent histories of each term.
    """
    widths = {"policy": (4, 7, 28), "proprio": (15, 28, 28, 78),
              "perception": (192,), "task_state": (15,)}
    if list(groups) != list(widths):
        raise ValueError("Teacher v12 observation groups/order do not match the frame layout.")
    history, remainder = divmod(obs["task_state"].shape[-1], 15)
    if remainder or history not in (1, 5):
        raise ValueError("Teacher v12 requires matching 1- or 5-frame histories.")
    terms = []
    for group, term_widths in widths.items():
        value = obs[group]
        if value.shape[-1] != sum(term_widths) * history:
            raise ValueError(f"Teacher v12 has inconsistent history/width for {group}: {value.shape}.")
        for term, width in zip(value.split([w * history for w in term_widths], dim=-1), term_widths):
            terms.append(term.reshape(*term.shape[:-1], history, width))
    return torch.cat(terms, dim=-1).flatten(start_dim=-2)


class BulbObservationFrames:
    """Use the same time-major layout for actor, critic, normalization and PPO."""

    def get_actor_obs(self, obs):
        if "task_state" not in self.obs_groups["policy"]:
            return super().get_actor_obs(obs)
        return stack_bulb_observation_frames(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs):
        if "task_state" not in self.obs_groups["critic"]:
            return super().get_critic_obs(obs)
        return stack_bulb_observation_frames(obs, self.obs_groups["critic"])


class TaskStateNormalization(EmpiricalNormalization):
    """Normalize the 380 original channels in each frame, preserving its 15 new ones.

    Being an EmpiricalNormalization also preserves native DDP moment syncing.
    Moment buffers follow the new time-major layout; old checkpoints require migration.
    """

    def forward(self, x):
        frames = x.reshape(*x.shape[:-1], self._mean.shape[-1] // 380, 395)
        original = super().forward(frames[..., :380].flatten(start_dim=-2))
        return torch.cat((original.reshape(*frames.shape[:-1], 380), frames[..., 380:]), dim=-1).flatten(start_dim=-2)

    def update(self, x):
        frames = x.reshape(*x.shape[:-1], self._mean.shape[-1] // 380, 395)
        super().update(frames[..., :380].flatten(start_dim=-2))

    def inverse(self, y):
        frames = y.reshape(*y.shape[:-1], self._mean.shape[-1] // 380, 395)
        original = super().inverse(frames[..., :380].flatten(start_dim=-2))
        return torch.cat((original.reshape(*frames.shape[:-1], 380), frames[..., 380:]), dim=-1).flatten(start_dim=-2)


def _configure_task_normalization(policy, obs, obs_groups):
    """Keep the added task-state channels out of running moment estimates."""
    for role, prefix in (("policy", "actor"), ("critic", "critic")):
        groups = obs_groups[role]
        if "task_state" not in groups:
            continue  # Preserve the optional legacy LSTM diagnostic interface.
        stack_bulb_observation_frames(obs, groups)  # Fail early on mismatched histories/order.
        if getattr(policy, prefix + "_obs_normalization"):
            width = sum(obs[key].shape[-1] for key in groups[:-1])
            setattr(policy, prefix + "_obs_normalizer", TaskStateNormalization(width))


class BulbGraspTeacherActorCritic(BulbObservationFrames, ActorCritic):
    """Complete 395-channel frames feed the original actor/critic hidden sizes."""

    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        _configure_task_normalization(self, obs, obs_groups)


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


class BulbGraspTeacherLSTM(BulbObservationFrames, ActorCriticRecurrent):
    """RSL recurrent PPO with the teacher's noise bounds and bootstrap semantics."""

    def __init__(self, obs, obs_groups, num_actions, min_noise_std=0.05, max_noise_std=0.5, **kwargs):
        # This task uses a learned, observation-independent std per action.
        if kwargs.get("state_dependent_std", False) or kwargs.get("noise_std_type", "scalar") != "scalar":
            raise ValueError("BulbGraspTeacherLSTM requires state-independent scalar noise.")
        if not 0 < min_noise_std <= max_noise_std:
            raise ValueError("Expected 0 < min_noise_std <= max_noise_std.")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        _configure_task_normalization(self, obs, obs_groups)
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
