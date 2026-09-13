"""Same-step reset must not advance untouched fingers' physical memory twice."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = (Path(__file__).resolve().parents[1] / "source/BrainCo_DexHand/BrainCo_DexHand/tasks"
          / "manager_based/dexsuite/mdp/d_peg_tactile.py")


def runtime(n=3):
    class CalibratedRuntime:
        def reset(self, env_ids=None):
            self._cache_key, self._cached = None, None
            self._env.epoch += 1
            adapter = self._env._brainco_rl_hydroshear_adapter
            for index in env_ids.tolist():
                slots = slice(index * 5, (index + 1) * 5)
                adapter._batch_state_valid[slots] = False
                for name in self._HISTORY_LISTS:
                    for slot in range(index * 5, (index + 1) * 5):
                        getattr(adapter, name)[slot] = None

        def __call__(self, env, component_cfg):
            key = (env.common_step_counter, env.epoch)
            if self._cache_key == key:
                return self._cached
            adapter = env._brainco_rl_hydroshear_adapter
            for name in self._HISTORY_BUFFERS:
                value = getattr(adapter, name)
                if value.dtype != torch.bool:
                    value += 1
            adapter._batch_state_valid[:] = True
            for name in self._HISTORY_LISTS:
                values = getattr(adapter, name)
                for index, value in enumerate(values):
                    values[index] = torch.ones(1) if value is None else value + 1
            feature = adapter._batch_hydrosoft_forces.reshape(env.num_envs, 5, -1).sum((1, 2))
            self._cached = torch.stack((feature, feature * 2), dim=-1)
            self._cache_key = key
            env.latest_tactile_inputs = {"marker": self._cached[:, None, :].clone()}
            env.latest_tactile_latent = self._cached[:, None, :]
            return self._cached

    tree = ast.parse(SOURCE.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DPegPretrainedTactile")
    ns = dict(torch=torch, RotateBulbPretrainedTactile=CalibratedRuntime)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), ns)
    cls = ns["DPegPretrainedTactile"]
    term = object.__new__(cls)
    adapter = SimpleNamespace()
    for name in cls._HISTORY_BUFFERS:
        dtype = torch.bool if name == "_batch_state_valid" else torch.float32
        setattr(adapter, name, torch.zeros(n * 5, 1, dtype=dtype))
    for name in cls._HISTORY_LISTS:
        setattr(adapter, name, [torch.zeros(1) for _ in range(n * 5)])
    env = SimpleNamespace(num_envs=n, device="cpu", common_step_counter=7, epoch=0,
                          _brainco_rl_hydroshear_adapter=adapter)
    term._env, term._cache_key, term._cached = env, None, None
    return term, env, adapter


def test_same_step_partial_reset_preserves_untouched_memory_latent_and_inputs():
    term, env, adapter = runtime()
    first = term(env, {}).clone()
    old_inputs = env.latest_tactile_inputs["marker"].clone()
    buffers = {name: getattr(adapter, name).clone() for name in term._HISTORY_BUFFERS}
    term.reset(torch.tensor([1]))
    result = term(env, {})
    keep_slots = torch.tensor([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    for name, before in buffers.items():
        torch.testing.assert_close(getattr(adapter, name)[keep_slots], before[keep_slots])
    for name in term._HISTORY_LISTS:
        assert all(getattr(adapter, name)[index].item() == 1 for index in keep_slots.tolist())
    torch.testing.assert_close(result[[0, 2]], first[[0, 2]])
    torch.testing.assert_close(env.latest_tactile_inputs["marker"][[0, 2]], old_inputs[[0, 2]])
    torch.testing.assert_close(env.latest_tactile_latent[[0, 2], 0], first[[0, 2]])
    # Selected fingers are recomputed; an ordinary cached reread changes none.
    assert (adapter._batch_hydrosoft_forces[5:10] == 2).all()
    torch.testing.assert_close(term(env, {}), result)


def test_consecutive_partial_resets_merge_the_preserved_environment_mask():
    term, env, adapter = runtime()
    first = term(env, {}).clone()
    term.reset(torch.tensor([0]))
    term.reset(torch.tensor([1]))
    result = term(env, {})
    torch.testing.assert_close(result[2], first[2])
    assert (adapter._batch_hydrosoft_forces[10:] == 1).all()
    assert (adapter._batch_hydrosoft_forces[:10] == 2).all()


def test_next_physics_step_advances_all_environments_normally():
    term, env, adapter = runtime()
    term(env, {})
    term.reset(torch.tensor([1]))
    env.common_step_counter += 1
    term(env, {})
    assert (adapter._batch_hydrosoft_forces == 2).all()


def test_full_reset_does_not_retain_any_previous_episode_latent():
    term, env, adapter = runtime()
    first = term(env, {}).clone()
    term.reset(slice(None))
    result = term(env, {})
    assert (adapter._batch_hydrosoft_forces == 2).all()
    torch.testing.assert_close(result, first * 2)
