"""Custom turn budgets and episode-weighted metrics, including terminal resets."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rotate_bulb_rewards import CUSTOM_CONFIG, reward_code
from test_rotate_bulb_reward_balance import _task_env, _term
from test_rotate_bulb_grasp_reset import reset_code, _grasp_reset_env, _saved_reset_kwargs


@pytest.fixture
def code(monkeypatch):
    return reward_code.__wrapped__(monkeypatch, SimpleNamespace(param=CUSTOM_CONFIG))


def _set_turns(env, turns, ids=None):
    ids = slice(None) if ids is None else ids
    turns = torch.as_tensor(turns)
    names = env.scene["lamp"].joint_names
    travel = torch.zeros_like(env._bulb_start_joints[ids])
    travel[:, names.index("screw_turn")] = turns * (2 * torch.pi)
    travel[:, names.index("screw_slide")] = turns * .006
    env.scene["lamp"].data.joint_pos[ids] = env._bulb_start_joints[ids] + travel


@pytest.fixture
def configured(code):
    tree = ast.parse(CUSTOM_CONFIG.read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "DexsuiteRevo3RotateBulbRewardCfg")
    cfg.decorator_list = []
    namespace = dict(vars(code), SceneEntityCfg=lambda name: SimpleNamespace(name=name),
                     mdp=SimpleNamespace(action_l2_clamped=None, action_rate_l2_clamped=None))
    namespace["RewTerm"] = lambda func, weight, params=None: SimpleNamespace(
        func=func, weight=weight, params=params or {})
    exec(compile(ast.Module(body=[cfg], type_ignores=[]), str(CUSTOM_CONFIG), "exec"), namespace)
    return namespace[cfg.name]()


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_configured_turn_split_has_fixed_budget_with_and_without_grasp(code, configured, hz):
    env = _task_env(count=3, hz=hz)
    thumb = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w
    thumb[:, 0, 0, 2] = torch.tensor([0., .25, 1.])
    terms = [configured.unscrew_turns, configured.unscrew_progress]
    funcs = [term.func(term, env) for term in terms]
    sums = torch.zeros(2, 3)
    for step in range(1, hz + 1):
        _set_turns(env, [3 * step / hz] * 3)
        for i, (term, func) in enumerate(zip(terms, funcs)):
            sums[i] += func(env, **term.params) * term.weight * env.step_dt
    torch.testing.assert_close(sums[0], torch.full((3,), 1.5))
    torch.testing.assert_close(sums[1], torch.tensor([0., 1.125, 4.5]))
    torch.testing.assert_close(sums.sum(0), torch.tensor([1.5, 2.625, 6.]))
    for term, func in zip(terms, funcs):
        assert not func(env, **term.params).any()  # Holding at three turns pays nothing.


def test_physical_turns_reject_backtracking_axial_mismatch_and_free_follower(code):
    env = _task_env(count=2)
    term = _term(code, "RotateBulbPhysicalTurnProgress", env)
    _set_turns(env, [1., 1.])
    env.scene["lamp"].data.joint_pos[1, 1] = env._bulb_start_joints[1, 1]
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([1., 0.]))
    for turns in (.5, 1., .75, 1.):
        _set_turns(env, [turns], [0])
        assert term(env)[0] == 0
    _set_turns(env, [1.5], [0])
    assert term(env)[0] * env.step_dt == pytest.approx(.5)
    env._bulb_released[:] = True
    _set_turns(env, [3., 3.])
    assert not term(env).any()
    torch.testing.assert_close(term.best, torch.tensor([1.5, 0.]))
    # A partial reset re-arms only that environment, using its new seated pose.
    env._bulb_released[0] = False
    env._bulb_start_joints[0] = torch.tensor([-4 * torch.pi, -.012])
    _set_turns(env, [0.], [0])
    term.reset(torch.tensor([0]))
    _set_turns(env, [4.], [0])  # Clamp to this episode's two-turn engagement.
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([2., 0.]))


@pytest.fixture
def command_type(code):
    class Parent:
        def __init__(self, cfg, env):
            self._env, self.cfg = env, cfg
            self.num_envs, self.device = env.num_envs, env.device
            self.metrics = {}
            self.success_visualizer = SimpleNamespace(visualize=lambda *a, **kw: None)
            self.success_vis_asset = env.scene["object"]

        def reset(self, env_ids=None):
            return {"success": 0., "position_error": 0.}

        def _update_metrics(self):
            pass

    tree = ast.parse(CUSTOM_CONFIG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RotateBulbPoseCommand")
    ns = dict(vars(code), ObjectUniformPoseCommand=Parent)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(CUSTOM_CONFIG), "exec"), ns)
    return ns[cls.name]


def _command_env(count=3):
    env = _task_env(count=count)
    env.failure_terms["success"] = torch.zeros(count, dtype=torch.bool)
    env.episode_length_buf = torch.zeros(count, dtype=torch.long)
    env.common_step_counter = 0
    env.extras = {}
    return env


def _log_reset(command, ids=None):
    metrics = command.reset(ids)
    command._env.extras["log"] = {f"Metrics/object_pose/{k}": v for k, v in metrics.items()}
    return command._env.extras["log"]


def test_metrics_capture_terminal_step_backtracking_and_reset_independently(command_type):
    env = _command_env()
    command = command_type(None, env)
    initial = _log_reset(command)
    assert initial["Metrics/object_pose/unscrew_turns_max"].numel() == 0
    env.common_step_counter = 1
    env.episode_length_buf[:] = 1
    _set_turns(env, [2., 1., .5])
    command._update_metrics()
    # The terminal step happens before CommandManager.compute. Record at reset.
    _set_turns(env, [1.5, 3., .5])
    command.record_turn_metrics(torch.tensor([0, 1]))
    _set_turns(env, [0., 0.], [0, 1])  # Scene and reward reset precede command reset.
    log = _log_reset(command, torch.tensor([0, 1]))
    torch.testing.assert_close(log["Metrics/object_pose/unscrew_turns_max"], torch.tensor([2., 3.]))
    torch.testing.assert_close(log["Metrics/object_pose/unscrew_turns_final"], torch.tensor([1.5, 3.]))
    torch.testing.assert_close(log["Metrics/object_pose/unscrew_reached_3_turns"], torch.tensor([0., 1.]))
    torch.testing.assert_close(command.max_turns, torch.tensor([0., 0., .5]))
    command._update_metrics()
    torch.testing.assert_close(log["Metrics/object_pose/unscrew_turns_final"], torch.tensor([1.5, 3.]))


def test_release_captures_and_freezes_actual_turns_without_reward_terms(code, command_type):
    env = _command_env(count=1)
    command = command_type(None, env)
    env._bulb_attachment_attrs = [SimpleNamespace(Set=lambda value: None)]
    env.episode_length_buf[:] = 4
    _set_turns(env, [2.])
    command._update_metrics()
    _set_turns(env, [2.998])  # Physical exit permits the existing 0.1% tolerance.
    code.release_unscrewed_bulb(env)
    assert env._bulb_released.all()
    for turns in (0., 3., 1.):
        _set_turns(env, [turns])
        command._update_metrics()
        torch.testing.assert_close(command.turns, torch.tensor([2.998]))
    metrics = command.reset()
    torch.testing.assert_close(metrics["unscrew_turns_final"], torch.tensor([2.998]))


def test_reset_event_captures_terminal_joints_before_overwriting_them(reset_code):
    env = _grasp_reset_env()
    selected = torch.tensor([0, 2])
    env._bulb_start_joints = env.scene["lamp"].data.default_joint_pos.clone()
    before = env.scene["lamp"].data.joint_pos.clone()
    captured = []
    env._bulb_turn_metrics_command = SimpleNamespace(record_turn_metrics=lambda ids: captured.append(
        (ids.clone(), env.scene["lamp"].data.joint_pos.clone())))
    reset_code.reset_lamp_on_table(env, selected, **_saved_reset_kwargs())
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], selected)
    torch.testing.assert_close(captured[0][1], before)
    assert not torch.equal(env.scene["lamp"].data.joint_pos[selected], before[selected])


def test_logs_count_each_episode_once_and_weight_unequal_reset_batches(command_type):
    env = _command_env()
    command = command_type(None, env)
    logs = [_log_reset(command)]
    env.episode_length_buf[:] = 1
    env.common_step_counter = 1
    _set_turns(env, [3., 0., 0.])
    command.record_turn_metrics()
    _set_turns(env, [0.], [0])
    first = _log_reset(command, torch.tensor([0]))
    command._update_metrics()
    logs.append(env.extras["log"])
    for step in (2, 3, 4):
        env.common_step_counter = step
        command._update_metrics()
        logs.append(env.extras["log"])
    # A later batch contains two zero-progress failed episodes.
    logs.append(_log_reset(command, torch.tensor([1, 2])))
    key = "Metrics/object_pose/unscrew_turns_max"
    torch.testing.assert_close(first[key], torch.tensor([3.]))  # Old logs retain their samples.
    samples = torch.cat([log[key] for log in logs])
    torch.testing.assert_close(samples, torch.tensor([3., 0., 0.]))
    assert samples.mean() == pytest.approx(1.)  # Mean of batch means would incorrectly report 1.5.
    reached = torch.cat([log["Metrics/object_pose/unscrew_reached_3_turns"] for log in logs])
    assert reached.mean() == pytest.approx(1 / 3)

    # Execute the installed RSL-RL logger's episode-info block, including its
    # handling of empty vectors. Verify the scalar that TensorBoard receives.
    import importlib.util
    spec = importlib.util.find_spec("rsl_rl")
    assert spec is not None
    path = Path(next(iter(spec.submodule_search_locations))) / "runners/on_policy_runner.py"
    tree = ast.parse(path.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnPolicyRunner")
    log_method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "log")
    block = next(n for n in log_method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "locs['ep_infos']")
    written = {}
    writer = SimpleNamespace(add_scalar=lambda key, value, step: written.update({key: float(value)}))
    ns = dict(torch=torch, self=SimpleNamespace(device="cpu", writer=writer),
              locs={"ep_infos": logs, "it": 1}, ep_info_means={}, ep_string="", pad=35)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), ns)
    assert written[key] == pytest.approx(1.)
    assert written["Metrics/object_pose/unscrew_reached_3_turns"] == pytest.approx(1 / 3)
