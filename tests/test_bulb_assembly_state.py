"""CPU-level invariants for the independent tabletop assembly task."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/assembly.py"
CFG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/assembly_env_cfg.py"
INIT = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/__init__.py"


def test_assembly_is_independent_and_starts_on_desktop():
    source = ASSEMBLY.read_text()
    assert "reset_bulb_assembly_on_table" in source
    assert "DESKTOP_OFFSET_Y_M = -0.18" in source
    assert "THREAD_MIN_X_M = -0.03002103604376316" in source
    assert "env._bulb_assembly_state" in source
    assert "INSTALLATION_DEPTH_M = 0.018" in source


def test_engagement_requires_contact_alignment_rotation_and_advance():
    source = ASSEMBLY.read_text()
    for required in ("self.tracking_thread & aligned_now & contact & valid_progress", "ENGAGEMENT_TURN_MIN", "HELICAL_TOLERANCE_M", "self.installed"):
        assert required in source
    assert "ENTRY_ROOT_X_M = SOCKET_MOUTH_X_M - THREAD_MIN_X_M" in source
    assert "current_advance = ENTRY_ROOT_X_M - rel_pos[:, 0]" in source
    assert "write_root_pose_to_sim" in source
    assert "set_joint" not in source


def test_registration_uses_separate_task_id_and_config():
    source = INIT.read_text() + CFG.read_text()
    assert "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0" in source
    assert "DexsuiteRevo3BulbAssemblyEnvCfg" in source
    assert "self.events.release_bulb = None" in source
