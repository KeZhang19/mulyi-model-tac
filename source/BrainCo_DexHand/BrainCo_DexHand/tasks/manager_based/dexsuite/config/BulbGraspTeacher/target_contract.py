"""Validate the position-command meaning before loading teacher checkpoints."""
from pathlib import Path


def validate_target_contract(checkpoint_path, curriculum_version, allow_v8_migration=False,
                             allow_v9_migration=False):
    """A same-width policy is insufficient when targets or stage gates change."""
    import yaml

    metadata = Path(checkpoint_path).resolve().parent / "params/env.yaml"
    saved_version = None
    if metadata.is_file():
        # Isaac dumps !!python/tuple nodes. Inspect scalar metadata without
        # constructing Python objects or executing any YAML constructors.
        saved = yaml.compose(metadata.read_text(), Loader=yaml.SafeLoader)
        if isinstance(saved, yaml.MappingNode):
            for key, value in saved.value:
                if key.value == "bulb_grasp_curriculum_version" and isinstance(value, yaml.ScalarNode):
                    saved_version = int(value.value)
    v8_migration = saved_version == 8 and curriculum_version == 9 and allow_v8_migration
    v9_migration = saved_version == 9 and curriculum_version == 10 and allow_v9_migration
    if saved_version != curriculum_version and not (v8_migration or v9_migration):
        raise ValueError(
            f"Checkpoint task contract is v{saved_version or 'unknown'}, "
            f"but this environment is v{curriculum_version} (tip targets and stage gates). "
            f"Use a matching checkpoint with {metadata}, or replay the old checkpoint "
            "in its original frozen checkout. Start a fresh run for the new target contract."
        )
    return {"source_version": saved_version, "target_version": curriculum_version,
            "v8_to_v9_migration": bool(v8_migration), "v9_to_v10_migration": bool(v9_migration)}
