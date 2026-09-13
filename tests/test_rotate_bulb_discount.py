"""Check the configured discount in physical task time without loading simulation."""

import ast
from pathlib import Path

import pytest


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3/agents"
    / "rsl_rl_ppo_cfg_rotate_bulb.py"
)


@pytest.mark.parametrize("delay_seconds, minimum_weight", [(10, .5), (30, .1)])
def test_later_task_stages_retain_useful_discount_weight(delay_seconds, minimum_weight):
    tree = ast.parse(CONFIG.read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assignments = {
        node.targets[0].id: node.value
        for node in runner.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    algorithm = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in assignments["algorithm"].keywords
    }
    # The task keeps its 60 Hz control rate; only the physical discount horizon
    # changes. A short rollout still bootstraps from the critic at its boundary.
    assert 0.0 < algorithm["gamma"] < 1.0
    assert algorithm["gamma"] ** (60 * delay_seconds) >= minimum_weight
    assert ast.literal_eval(assignments["num_steps_per_env"]) == 32
    assert algorithm["lam"] == .95
