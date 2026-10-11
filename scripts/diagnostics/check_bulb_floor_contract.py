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
                            and n.name in ("FloorBestReachReward", "FloorReachReward", "arm_path_cost", "base_excursion_cost")], type_ignores=[])
namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerStub,
             "grasp_distance": lambda env: env.distance,
             "assembly_hand_shell_distance": lambda env: env.distance}
exec(compile(selected, str(CONFIG / "rewards.py"), "exec"), namespace)

# Run the real filtered-force reader and contact predicates with tensor-backed
# sensor data; unrelated total contact forces must not earn bulb contact credit.
contact_namespace = {"torch": torch, "math": math, "ManagerBasedRLEnv": NS}
for filename, names in (("env_cfg.py", ("rotate_bulb_finger_forces",)),
                        ("assembly_env_cfg.py", ("assembly_any_finger_contact", "assembly_good_finger_contact"))):
    path = CONFIG.parent / "RotateBulbCustom" / filename
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), contact_namespace)


class FloorContractChecks(unittest.TestCase):
    def test_any_contact_filtered_single_finger_and_rate(self):
        forces = torch.zeros(9, 5)
        forces[1, 0] = 1.0  # Strict > threshold, not >=.
        for i in range(5):
            forces[i + 2, i] = 1.1
        forces[7, :2] = 2.0
        forces[8, :2] = torch.tensor([float("nan"), float("inf")])
        sensors = {}
        for i, finger in enumerate(("thumb", "index", "mid", "ring", "pinky")):
            matrix = torch.zeros(9, 1, 1, 3)
            matrix[:, 0, 0, 0] = forces[:, i]
            sensors[f"right_{finger}dip_roll_rubber_link_object_s"] = NS(
                data=NS(force_matrix_w=matrix, net_forces_w=torch.full((9, 1, 3), 1000.)))
        env = NS(num_envs=9, device="cpu", scene=NS(sensors=sensors))
        any_contact = contact_namespace["assembly_any_finger_contact"]
        expected = torch.tensor([0., 0., 1., 1., 1., 1., 1., 1., 0.])
        torch.testing.assert_close(any_contact(env), expected)
        torch.testing.assert_close(contact_namespace["assembly_good_finger_contact"](env),
                                   torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0.]))
        for hz in (30, 60, 120):
            torch.testing.assert_close(any_contact(env) * .7 / hz * (5 * hz), expected * 3.5)
        for sensor in sensors.values():
            sensor.data.force_matrix_w = None
        self.assertTrue(torch.equal(any_contact(env), torch.zeros(9)))

    def test_best_reach_only_new_records_and_frequency_independent(self):
        for hz in (30, 60, 120):
            env = NS(num_envs=1, device="cpu", step_dt=1/hz, distance=torch.tensor([.08]),
                     _bulb_assembly_state=NS(lifted=torch.tensor([False])),
                     termination_manager=NS(active_terms=[]))
            term = namespace["FloorBestReachReward"](None, env)
            self.assertEqual(float(term(env)), 0.)
            credits = []
            for distance in (.07, .075, .07, .06, .06):
                env.distance.fill_(distance)
                credits.append(float(term(env) * 40 * env.step_dt))
            torch.testing.assert_close(torch.tensor(credits), torch.tensor([1., 0., 0., 1., 0.]))

    def test_best_reach_partial_reset_and_invalid_sample(self):
        env = NS(num_envs=2, device="cpu", step_dt=1/30, distance=torch.tensor([.08, .08]),
                 _bulb_assembly_state=NS(lifted=torch.tensor([False, False])),
                 termination_manager=NS(active_terms=[]))
        term = namespace["FloorBestReachReward"](None, env)
        term(env)
        env.distance[:] = .07
        term(env)
        term.reset(torch.tensor([0]))
        env.distance[:] = torch.tensor([.02, .06])
        torch.testing.assert_close(term(env) * 40 * env.step_dt, torch.tensor([0., 1.]))
        env.distance[:] = torch.tensor([float("nan"), float("inf")])
        self.assertTrue(torch.equal(term(env), torch.zeros(2)))
        env.distance[:] = .01
        self.assertTrue(torch.equal(term(env), torch.zeros(2)))
        term.reset()
        env.distance[:] = .005
        self.assertTrue(torch.equal(term(env), torch.zeros(2)))
        for std in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                term(env, std=std)

    def test_best_reach_suppressed_on_lift_or_failure(self):
        failed = torch.tensor([False, False, True])
        env = NS(num_envs=3, device="cpu", step_dt=1/30, distance=torch.full((3,), .08),
                 _bulb_assembly_state=NS(lifted=torch.tensor([False, True, False])),
                 termination_manager=NS(active_terms=["bulb_dropped"], get_term=lambda name: failed))
        term = namespace["FloorBestReachReward"](None, env)
        term(env)
        env.distance[:] = .07
        torch.testing.assert_close(term(env) * 40 * env.step_dt, torch.tensor([1., 0., 0.]))
        self.assertTrue(torch.equal(env._bulb_assembly_state.lifted, torch.tensor([False, True, False])))

    def test_best_reach_stage0_wiring_keeps_costs(self):
        # Exercise both production configuration functions: the ledger adapter
        # clears earlier rewards, so the new term must be installed afterwards.
        custom = CONFIG.parent / "RotateBulbCustom"
        ledger = runpy.run_path(str(custom / "assembly_reward_v2.py"))
        def term(**kwargs):
            return NS(params={}, **kwargs) if "params" not in kwargs else NS(**kwargs)
        def rewards():
            # Read the actual inherited action cost definitions, not copies of
            # their weights embedded in this test's fixture.
            node = next(n for n in ast.parse((custom / "assembly_env_cfg.py").read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == "DexsuiteRevo3BulbAssemblyRewardCfg")
            assignments = [n for n in node.body if isinstance(n, ast.Assign)
                           and n.targets[0].id in ("action_l2", "action_rate_l2")]
            scope = {"RewTerm": term, "rotate_bulb_action_l2_balanced": object(),
                     "rotate_bulb_action_rate_l2_balanced": object()}
            exec(compile(ast.Module(body=assignments, type_ignores=[]), "action_costs", "exec"), scope)
            return NS(**{name: scope[name] for name in ("action_l2", "action_rate_l2")},
                      insertion_tracking=term(func=object(), weight=5.))
        scope = {**namespace, **contact_namespace, "RewardTermCfg": term, "TerminationTermCfg": term,
                 "COMPONENTS": ledger["COMPONENTS"], "reward_component": object(),
                 "AssemblyRewardAdapter": object(), "CurriculumRewardAdapter": object(),
                 "DexsuiteRevo3BulbAssemblyRewardCfg": rewards, "CurriculumInsertionReward": object(),
                 "EPISODE_SECONDS": core["EPISODE_SECONDS"],
                 "runtime": NS(curriculum_reward=object(), curriculum_stage_complete=object(),
                               curriculum_reset_failed=object(), reset_curriculum_scene=object())}
        for path, name in ((custom / "assembly_reward_adapter.py", "configure_assembly_rewards"),
                           (CONFIG / "env_cfg.py", "configure_curriculum")):
            node = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        for stage in range(5):
            cfg = NS(bulb_curriculum=core["CurriculumSettings"](stage=stage), sim=NS(), scene=NS(),
                     observations=NS(perception=NS(object_point_cloud=NS(params={})), proprio=NS()),
                     tactile_policy_enabled=False, bulb_pure_collision=True, bulb_assembly_task=True,
                     OLD_REWARD=False, terminations=NS(),
                     events=NS(reset_object=NS(replace=lambda **kwargs: NS(**kwargs))))
            scope["configure_curriculum"](cfg)
            self.assertEqual(hasattr(cfg.rewards, "reach_best_progress"), stage == 0)
            self.assertEqual(getattr(cfg.rewards, "any_finger_contact", None) is not None, stage == 0)
            if stage == 0:
                reach = cfg.rewards.reach_best_progress
                self.assertIs(reach.func, namespace["FloorBestReachReward"])
                self.assertEqual(reach.weight / reach.params["std"], 100.)
                contact = cfg.rewards.any_finger_contact
                self.assertIs(contact.func, contact_namespace["assembly_any_finger_contact"])
                self.assertEqual(contact.weight, .7)
                self.assertEqual(contact.params["threshold"], 1.)
            self.assertEqual(cfg.rewards.action_l2.weight, -.02)
            self.assertEqual(cfg.rewards.action_rate_l2.weight, -.01)
            self.assertEqual(cfg.rewards.arm_path.weight, -.08)
            self.assertEqual(cfg.rewards.base_excursion.weight, -.5)
            self.assertIs(cfg.rewards.reach_progress.func, namespace["FloorReachReward"])

    def test_no_fixture_or_unvalidated_physics(self):
        Settings = core["CurriculumSettings"]
        Settings().validate()
        for kwargs in ({"fixed_bulb": True}, {"version": 1}, {"physics_hz": 60}):
            with self.assertRaises(ValueError):
                Settings(**kwargs).validate()

    def test_fast_clock_requires_stage0_opt_in(self):
        Settings = core["CurriculumSettings"]
        Settings(stage=0, physics_hz=90, fast_stage0_physics=True).validate()
        for kwargs in ({"physics_hz": 90},
                       {"physics_hz": 60, "fast_stage0_physics": True},
                       {"physics_hz": 90, "fast_stage0_physics": "true"},
                       *({"stage": stage, "physics_hz": 90, "fast_stage0_physics": True}
                         for stage in range(1, 5))):
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
