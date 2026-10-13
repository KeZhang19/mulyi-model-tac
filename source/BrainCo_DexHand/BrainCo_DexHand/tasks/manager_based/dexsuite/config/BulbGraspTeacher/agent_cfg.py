"""Frozen teacher PPO options, independent of newer Isaac Lab default fields."""
import json
from pathlib import Path
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from .env_cfg import ConfigNode


@configclass
class BulbGraspTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    def __post_init__(self):
        values = json.loads(Path(__file__).with_name("agent_snapshot.json").read_text())
        for key, value in values.items():
            if key in ("policy", "algorithm"):
                group = ConfigNode()
                for name, setting in value.items():
                    setattr(group, name, setting)
                value = group
            setattr(self, key, value)
        # Discount at 30 Hz; retain more credit across grasp, insert and screw.
        self.algorithm.gamma = 0.995
        for name in ("policy", "critic"):
            self.obs_groups[name].append("task_state")
        self.policy.class_name = "BulbGraspTeacherActorCritic"
        self.class_name = "BulbGraspTeacherRunner"
        self.experiment_name = "bulb_grasp_teacher_v2test"
        self.run_name = ""
        self.resume = False
        self.load_checkpoint = "model_.*.pt"
