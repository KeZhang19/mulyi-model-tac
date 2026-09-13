"""Explicit D-peg migrations preserve optimizer/normalizers and reject scope drift."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch


TOOL = Path(__file__).resolve().parents[1] / "assets/d_peg_insertion/tools/prepare_optimized_resume.py"
spec = importlib.util.spec_from_file_location("_d_peg_resume", TOOL)
code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code)


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    socket = tmp_path / "socket.usd"
    socket.write_bytes(b"original socket")
    peg = tmp_path / "peg.usd"
    peg.write_bytes(b"unchanged peg")
    contract = {
        "task": code.TASK, "task_schema_version": 2, "observation_dim": 1949,
        "action_schema": {"type": "pregrasp_target_residual_position", "scale": .1},
        "reward_geometry": {"insertion_depth_m": .028}, "task_parameters": {"episode_length_s": 30.},
        "task_assets": {"socket": code.file_sha256(socket), "peg": code.file_sha256(peg),
                        "geometry": "geometry hash", "pregrasp": "pregrasp hash"},
    }
    contract_path = run / "tactile_policy_contract.json"
    contract_path.write_text(json.dumps(contract))
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    state = model.state_dict()
    state.update({"actor_obs_normalizer._mean": torch.tensor([.1, .2, .3]),
                  "actor_obs_normalizer._var": torch.tensor([2., 3., 4.]),
                  "actor_obs_normalizer.count": torch.tensor(1234)})
    checkpoint = run / "model_10.pt"
    torch.save({"model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(),
                "iter": 10, "infos": {"nested": ("preserve", [1, 2])}}, checkpoint)
    return dict(checkpoint=checkpoint, socket=socket, peg=peg, contract=contract,
                contract_path=contract_path, destination=tmp_path / "seed")


def validated_socket(source):
    socket = source["socket"].with_name("optimized.usd")
    socket.write_bytes(b"validated simplified socket")
    report = socket.with_suffix(".json")
    value = dict(passed=True, checks={key: True for key in code.PHYSICS_CHECKS},
                 socket_sha256=code.file_sha256(socket), peg_sha256=code.file_sha256(source["peg"]))
    report.write_text(json.dumps(value))
    return socket, report, value


def migrate(source, **kwargs):
    return code.prepare_resume(source["checkpoint"], source["destination"], source["socket"], **kwargs)


def test_geometry_only_preserves_all_weights_optimizer_normalizers_and_source_bytes(source):
    original_files = {path: path.read_bytes() for path in (source["checkpoint"], source["contract_path"], source["socket"])}
    result = migrate(source, total_iterations=15000)
    before = torch.load(source["checkpoint"], weights_only=True)
    after = torch.load(source["destination"] / "model_11.pt", weights_only=True)
    assert after["iter"] == 11
    code._assert_same_tree(before, dict(after, iter=10))
    assert result["recommended_training"]["additional_iterations"] == 14989
    assert result["recommended_training"]["required_hydra_override"] == "agent.algorithm.learning_rate=1.5e-05"
    assert result["contract_changed_fields"] == []
    assert result["mode"] == "geometry_only"
    assert json.loads((source["destination"] / "tactile_policy_contract.json").read_text()) == source["contract"]
    for path, data in original_files.items():
        assert path.read_bytes() == data
    assert (source["destination"] / "source_tactile_policy_contract.json").read_bytes() == original_files[source["contract_path"]]
    assert json.loads((source["destination"] / "resume_manifest.json").read_text()) == result


def test_socket_migration_changes_only_one_contract_leaf_and_audits_physics(source):
    socket, report, _ = validated_socket(source)
    result = migrate(source, socket_usd=socket, validation_report=report)
    target = json.loads((source["destination"] / "tactile_policy_contract.json").read_text())
    assert target["task_assets"]["socket"] == code.file_sha256(socket)
    target["task_assets"]["socket"] = source["contract"]["task_assets"]["socket"]
    code.validate_policy_contract(source["contract"], target)
    assert result["contract_changed_fields"] == ["task_assets.socket"]
    assert result["physics_validation"]["sha256"] == code.file_sha256(report)
    assert result["source"]["checkpoint_sha256"] == code.file_sha256(source["checkpoint"])
    assert result["target"]["checkpoint_sha256"] == code.file_sha256(source["destination"] / "model_11.pt")


def test_report_can_bind_exact_current_asset_paths(source):
    socket, report, value = validated_socket(source)
    value.pop("socket_sha256")
    value.pop("peg_sha256")
    value.update(socket_usd=str(socket), peg_usd=str(source["peg"]))
    report.write_text(json.dumps(value))
    result = migrate(source, socket_usd=socket, validation_report=report)
    assert result["physics_validation"]["socket_identity"] == "current_report_path_sha256"


def test_contact_probe_without_physics_changes_is_accepted(source):
    socket, report, value = validated_socket(source)
    value["contact_probe"] = dict(disable_peg_sleep=False,
        additional_down_force_n_last_second_blocked_cases=0.0, sample_period_s=1 / 240)
    report.write_text(json.dumps(value))
    assert migrate(source, socket_usd=socket, validation_report=report)["physics_validation"]["passed"]


@pytest.mark.parametrize("probe", [
    {"disable_peg_sleep": True},
    {"additional_down_force_n_last_second_blocked_cases": .1},
    {"additional_down_force_n_last_second_blocked_cases": -.1},
    {"additional_down_force_n_last_second_blocked_cases": "0"},
])
def test_modified_sleep_or_stress_forces_cannot_authorize_canonical_migration(source, probe):
    socket, report, value = validated_socket(source)
    value["contact_probe"] = probe
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="sleep and applied forces unchanged"):
        migrate(source, socket_usd=socket, validation_report=report)
    assert not source["destination"].exists()


@pytest.mark.parametrize("field", ["action_schema", "reward_geometry", "task_parameters", "observation_dim", "task_assets", "extra"])
def test_target_contract_cannot_change_anything_except_socket(source, field):
    socket, report, _ = validated_socket(source)
    target = deepcopy(source["contract"])
    target["task_assets"]["socket"] = code.file_sha256(socket)
    if field == "task_assets":
        target[field]["peg"] = "different peg"
    else:
        target[field] = "not the source contract"
    path = source["destination"].with_suffix(".json")
    path.write_text(json.dumps(target))
    with pytest.raises(ValueError, match="contract mismatch"):
        migrate(source, socket_usd=socket, validation_report=report, target_contract_path=path)
    assert not source["destination"].exists()


@pytest.mark.parametrize("change", ["failed", "failed_check", "missing_check", "wrong_socket", "wrong_peg", "missing_identity", "stale_path"])
def test_unvalidated_or_mismatched_collision_assets_are_rejected(source, change):
    socket, report, value = validated_socket(source)
    if change == "failed":
        value["passed"] = False
    elif change == "failed_check":
        value["checks"]["wrong_yaw_is_blocked"] = False
    elif change == "missing_check":
        value["checks"].pop("socket_stays_fixed")
    elif change == "wrong_socket":
        value["socket_sha256"] = "different socket"
    elif change == "wrong_peg":
        value["peg_sha256"] = "different peg"
    elif change == "missing_identity":
        value.pop("socket_sha256")
    elif change == "stale_path":
        value["socket_usd"] = str(source["socket"])
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        migrate(source, socket_usd=socket, validation_report=report)
    assert not source["destination"].exists()


def test_collision_requires_explicit_report_and_original_socket_identity(source):
    socket, report, _ = validated_socket(source)
    with pytest.raises(ValueError, match="requires --validation-report"):
        migrate(source, socket_usd=socket)
    source["socket"].write_bytes(b"silently replaced original")
    with pytest.raises(ValueError, match="Source socket bytes"):
        migrate(source, socket_usd=socket, validation_report=report)


def test_existing_output_is_never_overwritten(source):
    source["destination"].mkdir()
    valuable = source["destination"] / "model.pt"
    valuable.write_bytes(b"existing run")
    with pytest.raises(FileExistsError):
        migrate(source)
    assert valuable.read_bytes() == b"existing run"


def test_dangling_output_symlink_does_not_redirect_migration(source):
    target = source["destination"].with_name("unexpected_destination")
    source["destination"].symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        migrate(source)
    assert source["destination"].is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("change", ["missing_optimizer", "empty_optimizer", "nan_model", "nan_optimizer", "unequal_lr", "no_remaining_updates"])
def test_incomplete_or_nonfinite_training_state_is_rejected(source, change):
    payload = torch.load(source["checkpoint"], weights_only=True)
    if change == "missing_optimizer":
        payload.pop("optimizer_state_dict")
    elif change == "empty_optimizer":
        payload["optimizer_state_dict"]["state"] = {}
    elif change == "nan_model":
        payload["model_state_dict"]["weight"][0, 0] = torch.nan
    elif change == "nan_optimizer":
        next(iter(payload["optimizer_state_dict"]["state"].values()))["exp_avg"].fill_(torch.nan)
    elif change == "unequal_lr":
        group = deepcopy(payload["optimizer_state_dict"]["param_groups"][0])
        group["lr"] = 1e-4
        payload["optimizer_state_dict"]["param_groups"].append(group)
    else:
        payload["iter"] = 14999
    torch.save(payload, source["checkpoint"])
    with pytest.raises(ValueError):
        migrate(source)
    assert not source["destination"].exists()


@pytest.mark.parametrize("change", [{"task": "rotate bulb"}, {"task_schema_version": 1}, {"observation_dim": 1432}])
def test_other_task_and_legacy_sources_are_rejected(source, change):
    source["contract_path"].write_text(json.dumps({**source["contract"], **change}))
    with pytest.raises(ValueError, match="approved"):
        migrate(source)
