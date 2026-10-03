#!/usr/bin/env python3
"""CPU behavioral checks for floor curriculum scheduling and reward direction."""
import ast
import json
import math
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum"
core = runpy.run_path(str(CONFIG / "core.py"))


class ManagerStub:
    def __init__(self, cfg, env):
        self._env = env


# Execute the production reward definitions with minimal simulator bindings;
# native tests independently validate manager ordering and physical resets.
tree = ast.parse((CONFIG / "rewards.py").read_text())
selected = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                            and n.name in ("FloorReachReward", "arm_path_cost", "base_excursion_cost")], type_ignores=[])
namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerStub,
             "grasp_distance": lambda env: env.distance}
exec(compile(selected, str(CONFIG / "rewards.py"), "exec"), namespace)


class FloorContractChecks(unittest.TestCase):
    def test_no_fixture_or_unvalidated_physics(self):
        Settings = core["CurriculumSettings"]
        Settings().validate()
        for kwargs in ({"fixed_bulb": True}, {"version": 1}, {"physics_hz": 60}):
            with self.assertRaises(ValueError):
                Settings(**kwargs).validate()

    def test_promotion_requires_frontier_and_consecutive_windows(self):
        settings = core["CurriculumSettings"](stage=1, min_episodes=10)
        scheduler = core["StageScheduler"](settings)
        for _ in range(25):
            scheduler.observe(0, 0, True)
        self.assertEqual(scheduler.count, 0)
        for _ in range(20):
            scheduler.observe(1, 0, True)
        self.assertEqual(scheduler.level, 1)
        self.assertFalse(scheduler.ready)
        for _ in range(25):
            scheduler.observe(1, 0, True)
        self.assertEqual(scheduler.count, 0)

    def test_reach_retreat_and_no_cycle_progress(self):
        env = NS(num_envs=1, device="cpu", step_dt=1/30,
                 distance=torch.tensor([.2]), _bulb_assembly_state=NS(lifted=torch.tensor([False])))
        term = namespace["FloorReachReward"](None, env)
        term(env)
        credits = []
        for distance in (.1, .2):
            env.distance.fill_(distance)
            proximity = 1 - torch.tanh(env.distance / .1)
            credits.append(float((term(env) - proximity) * env.step_dt))
        self.assertGreater(credits[0], 0)
        self.assertLess(credits[1], 0)
        self.assertAlmostEqual(sum(credits), 0., places=6)
        term.reset(torch.tensor([0]))
        env.distance.fill_(.8)
        self.assertAlmostEqual(float(term(env)), float(1 - torch.tanh(env.distance/.1)), places=6)
        env._bulb_assembly_state.lifted[:] = True
        self.assertEqual(float(term(env)), 0.)

    def test_joint_costs_bidirectional_and_no_angle_wrapping(self):
        data = NS(joint_pos=torch.zeros(4, 7), default_joint_pos=torch.zeros(4, 7),
                  joint_vel=torch.zeros(4, 7))
        robot = NS(data=data, joint_names=[f"joint{i}" for i in range(1, 8)])
        env = NS(scene={"robot": robot})
        data.joint_pos[:, 0] = torch.tensor([0., 2., -2., 2*math.pi])
        costs = namespace["base_excursion_cost"](env)
        self.assertEqual(float(costs[0]), 0.)
        self.assertEqual(float(costs[1]), float(costs[2]))
        self.assertGreater(float(costs[3]), float(costs[1]))
        data.joint_vel[:, 0] = torch.tensor([0., 1., -1., 2.])
        self.assertTrue(torch.equal(namespace["arm_path_cost"](env), torch.tensor([0., 1., 1., 2.])))

    def test_reset_geometry_and_small_base_path(self):
        data = json.loads((ROOT / "assets/rotate_bulb_assembly_curriculum/floor_pregrasp_v2.json").read_text())
        bottom = min(v[2] for v in data["bulb_collision_vertices"])
        self.assertAlmostEqual(data["bulb_position_m"][2] + bottom - data["table_top_m"], .002, places=7)
        w, x, y, z = data["bulb_quaternion_wxyz"]
        self.assertAlmostEqual(2*(x*z-w*y), 0.)
        self.assertLess(max(abs(v) for v in data["joint1_path_range_deg"]), 30.)

    def test_policy_contract_keeps_legacy_dimensions_strict(self):
        path = ROOT / "scripts/rsl_rl/bulb_training.py"
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == "prepare_bulb_policy_run")
        ns = {"Path": Path, "json": json, "BULB_POLICY_VERSION": 3, "CONTRACT_FILENAME": "contract.json",
              "BulbActorCritic": NS(minimum_std=.05), "_task_id": lambda env: "test-task"}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        action = NS(action_contract=lambda: {"version": 3})
        for version, dim, tactile in ((None, 669, False), (2, 369, False), (2, 1309, True)):
            cfg = NS(tactile_policy_enabled=tactile)
            if version:
                cfg.bulb_curriculum = NS(version=version)
            raw = NS(cfg=cfg, action_manager=NS(get_term=lambda name: action),
                     observation_manager=NS(group_obs_dim={"policy": (dim,), "proprio": (0,), "perception": (0,)},
                                            active_terms={"policy": []}))
            env = NS(unwrapped=raw)
            with tempfile.TemporaryDirectory() as directory:
                result = ns["prepare_bulb_policy_run"](env, directory)
                self.assertEqual(result["observation_dim"], dim)
                self.assertEqual("assembly_curriculum" in result, version == 2)
                raw.observation_manager.group_obs_dim["policy"] = (dim + 1,)
                with self.assertRaises(ValueError):
                    ns["prepare_bulb_policy_run"](env, directory)


if __name__ == "__main__":
    unittest.main()
