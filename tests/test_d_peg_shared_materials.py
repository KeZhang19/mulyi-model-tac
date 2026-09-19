"""Large Flexiv insertion scenes share identical contact materials."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config"


@pytest.fixture
def material_runtime():
    created, bindings, collisions, expanded = {}, [], [], []
    stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: path in created))

    def create(path, cfg):
        assert path not in created
        created[path] = (cfg.compliant_contact_stiffness, cfg.compliant_contact_damping)

    sim = SimpleNamespace(
        get_current_stage=lambda: stage,
        RigidBodyMaterialCfg=lambda **kwargs: SimpleNamespace(func=create, **kwargs),
        CollisionPropertiesCfg=lambda **kwargs: SimpleNamespace(**kwargs),
        make_uninstanceable=lambda path, **kwargs: expanded.append(path),
        modify_collision_properties=lambda path, cfg, **kwargs: collisions.append((path, cfg)),
        bind_physics_material=lambda path, material, **kwargs: bindings.append((path, material)),
    )
    path = CONFIG / "RotateBulbCustom/robot_setup.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "apply_revo3_touch_compliant_materials")
    ns = dict(sim_utils=sim, ManagerBasedRLEnv=object,
              _collision_prim_paths_under=lambda stage, path: [path + "/collision"])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
    return SimpleNamespace(apply=ns[function.name], created=created, bindings=bindings,
                           collisions=collisions, expanded=expanded)


PARAMS = dict(link_names=("thumb", "index", "middle", "ring", "little"),
              stiffness=280., damping=20., contact_offset=.0002, rest_offset=-.0001)
SHARED = "/World/Physics_Materials/DPegTouchCompliant"


def test_8192_environments_bind_one_material_without_changing_contact_parameters(material_runtime):
    runtime = material_runtime
    env = SimpleNamespace(num_envs=8192)
    runtime.apply(env, None, shared_material_path=SHARED, **PARAMS)
    assert runtime.created == {SHARED: (280., 20.)}
    assert len(runtime.bindings) == 8192 * 5
    assert {material for _, material in runtime.bindings} == {SHARED}
    assert runtime.bindings[0][0] == "/World/envs/env_0/Robot/thumb/collision"
    assert runtime.bindings[-1][0] == "/World/envs/env_8191/Robot/little/collision"
    assert len(runtime.collisions) == len(runtime.bindings)
    assert all(cfg.collision_enabled and cfg.contact_offset == .0002 and cfg.rest_offset == -.0001
               for _, cfg in runtime.collisions)
    runtime.apply(env, [8191], shared_material_path=SHARED, **PARAMS)
    assert len(runtime.created) == 1
    assert len(runtime.bindings) == 8192 * 5 + 5


def test_default_material_ownership_and_partial_environment_selection_are_preserved(material_runtime):
    runtime = material_runtime
    runtime.apply(SimpleNamespace(num_envs=3), [1], **PARAMS)
    assert len(runtime.created) == 5
    assert runtime.expanded == ["/World/envs/env_1/Robot"]
    assert all(material == path + "/compliant_material" for path, material in runtime.bindings)
    assert set(runtime.created.values()) == {(280., 20.)}


def test_empty_selection_does_not_create_materials(material_runtime):
    material_runtime.apply(SimpleNamespace(num_envs=8192), [], shared_material_path=SHARED, **PARAMS)
    assert not material_runtime.created
    assert not material_runtime.bindings


def test_only_flexiv_insertion_opts_into_shared_compliance():
    tree = ast.parse((CONFIG / "InsertDPegCustom/env_cfg.py").read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Subscript)
                           and ast.unparse(target) == "cfg.events.zz_revo3_touch_compliant_materials.params['shared_material_path']"
                           for target in node.targets)]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == SHARED
    # The mixin itself does not enable sharing for the existing bulb task.
    setup = ast.parse((CONFIG / "RotateBulbCustom/robot_setup.py").read_text())
    mixin = next(node for node in setup.body if isinstance(node, ast.ClassDef) and node.name == "Revo3MixinCfg")
    assert "shared_material_path" not in ast.unparse(mixin)
