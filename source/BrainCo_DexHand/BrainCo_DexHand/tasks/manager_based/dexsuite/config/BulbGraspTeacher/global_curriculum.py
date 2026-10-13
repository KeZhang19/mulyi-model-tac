"""Whole-population reward curriculum; no simulator imports or action control."""
from collections import deque
import math


class GlobalStageCurriculum:
    """Advance once at a PPO boundary using exact, globally reduced episode counts.

    A window contains the most recent PPO updates (including empty updates).
    Promotion clears it, so evidence from one objective cannot promote the next.
    """

    def __init__(self, threshold=0.8, window_updates=100, min_episodes=8192):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if int(window_updates) != window_updates or window_updates < 1:
            raise ValueError("window_updates must be a positive integer")
        if int(min_episodes) != min_episodes or min_episodes < 1:
            raise ValueError("min_episodes must be a positive integer")
        self.settings = dict(threshold=float(threshold), window_updates=int(window_updates),
                             min_episodes=int(min_episodes))
        self.stage = 1
        self.window = deque(maxlen=int(window_updates))
        self.last_report = None

    def observe(self, successes, episodes):
        if int(successes) != successes or int(episodes) != episodes or not 0 <= successes <= episodes:
            raise ValueError("Expected integer counts: 0 <= successes <= episodes")
        self.window.append((int(successes), int(episodes)))
        success_sum = sum(row[0] for row in self.window)
        episode_sum = sum(row[1] for row in self.window)
        promoted = (self.stage < 3 and episode_sum >= self.settings["min_episodes"]
                    and success_sum >= self.settings["threshold"] * episode_sum)
        report = dict(evaluated_stage=self.stage, successes=success_sum, episodes=episode_sum,
                      success_rate=success_sum / episode_sum if episode_sum else None,
                      promoted=promoted)
        if promoted:
            self.stage += 1
            self.window.clear()
        self.last_report = {**report, "stage": self.stage}
        return self.last_report

    def state_dict(self):
        return dict(version=1, settings=dict(self.settings), stage=self.stage,
                    window=list(self.window))

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("settings") != self.settings:
            raise ValueError("Global bulb curriculum checkpoint settings/version do not match")
        stage, window = state.get("stage"), state.get("window")
        if stage not in (1, 2, 3) or not isinstance(window, (list, tuple)):
            raise ValueError("Invalid global bulb curriculum state")
        if len(window) > self.window.maxlen:
            raise ValueError("Global bulb curriculum window is too long")
        rows = []
        for success, count in window:
            if int(success) != success or int(count) != count or not 0 <= success <= count:
                raise ValueError("Invalid global bulb curriculum episode counts")
            rows.append((int(success), int(count)))
        self.stage = int(stage)
        self.window.clear()
        self.window.extend(rows)
        self.last_report = None


def synchronize_episode_counts(local_counts):
    """All ranks call once per update, even when a rank completed zero episodes."""
    import torch.distributed as dist

    counts = local_counts.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts.tolist()
