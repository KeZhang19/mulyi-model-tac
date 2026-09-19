"""Restore arm collision pairs only on newly spawned Custom bulb robots.

The donor asset and its shared spawn callback are also used by D-Peg.  This
module edits the new scene instance before cloning, never the asset files.
Isaac Lab imports remain lazy so the USD contract can be checked on the CPU.
"""

from __future__ import annotations

from functools import lru_cache


ARM_LINKS = ("base_link", *(f"link{i}" for i in range(1, 8)))
WRIST_LINKS = ("flange", "palm_link", "right_base_link", "palm")
FINGER_LINKS = tuple(
    f"right_{finger}_{segment}_Link"
    for finger, segments in (
        ("thumb", ("CMP", "CMR", "MCP", "PIP", "DIP")),
        ("index", ("MPR", "MCP", "PIP", "DIP")),
        ("middle", ("MPR", "MCP", "PIP", "DIP")),
        ("ring", ("MPR", "MCP", "PIP", "DIP")),
        ("little", ("MPR", "MCP", "PIP", "DIP")),
    )
    for segment in segments
)


def _pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


# Restore every nonadjacent arm pair and proximal-arm/hand pair.  Link6's
# rigidly mounted wrist assembly remains excluded, as does the donor's
# internal hand filtering.  The seven adjacent arm joints remain unchanged.
RESTORED_PAIRS = frozenset(
    [_pair(owner, target)
     for index, owner in enumerate(ARM_LINKS[:6])
     for target in (*ARM_LINKS[index + 2:], *WRIST_LINKS, *FINGER_LINKS)]
    + [_pair("link6", target) for target in FINGER_LINKS]
)


def _body_name(path, root_path) -> str | None:
    if not path.HasPrefix(root_path) or path == root_path:
        return None
    return path.MakeRelativePath(root_path).pathString.split("/", 1)[0]


def restore_bulb_arm_self_collisions(stage, robot_root_path: str) -> dict[str, int]:
    """Remove the 192 specified pair exclusions from one scene instance.

    Both relationship directions and collider-level targets are recognized.
    All other targets, adjacent-joint settings and hand filters are retained.
    Repeated calls are harmless. Missing expected body prims fail before edits.
    """
    from pxr import Sdf, Usd

    root_path = Sdf.Path(robot_root_path)
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"Custom bulb robot root does not exist: {root_path}")
    missing = [name for name in (*ARM_LINKS, *WRIST_LINKS, *FINGER_LINKS)
               if not stage.GetPrimAtPath(root_path.AppendChild(name))
               or not stage.GetPrimAtPath(root_path.AppendChild(name)).IsActive()]
    if missing:
        raise ValueError(f"Custom bulb collision contract is missing body prims: {missing}")

    updates = []
    removed_count = 0
    remaining_count = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        relationship = prim.GetRelationship("physics:filteredPairs")
        if not relationship:
            continue
        owner = _body_name(prim.GetPath(), root_path)
        targets = relationship.GetTargets()
        retained = []
        for target in targets:
            other = _body_name(target, root_path)
            if owner is not None and other is not None and _pair(owner, other) in RESTORED_PAIRS:
                removed_count += 1
            else:
                retained.append(target)
        remaining_count += len(retained)
        if retained != targets:
            if prim.IsInstanceProxy():
                raise ValueError(f"Expand robot collider instances before restoring collisions: {prim.GetPath()}")
            updates.append((relationship, retained))

    # Author stronger opinions in the scene layer, not in the referenced USD.
    with Usd.EditContext(stage, stage.GetRootLayer()):
        for relationship, retained in updates:
            if not relationship.SetTargets(retained):
                raise RuntimeError(f"Could not restore bulb arm collisions: {relationship.GetPath()}")

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        owner = _body_name(prim.GetPath(), root_path)
        for target in prim.GetRelationship("physics:filteredPairs").GetTargets():
            other = _body_name(target, root_path)
            if owner is not None and other is not None and _pair(owner, other) in RESTORED_PAIRS:
                raise RuntimeError(f"A restored collision pair remains filtered: {prim.GetPath()} -> {target}")
    return {
        "removed_count": removed_count,
        "restored_pair_count": len(RESTORED_PAIRS),
        "remaining_filter_count": remaining_count,
    }


def _spawn_bulb_robot_source(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from .robot_setup import spawn_revo3_robot_for_touch

    # The outer clone wrapper gives this callback a concrete source path, so
    # the original decorated spawn creates only that source at this point.
    prim = spawn_revo3_robot_for_touch(prim_path, cfg, translation, orientation, **kwargs)
    report = restore_bulb_arm_self_collisions(prim.GetStage(), prim.GetPath().pathString)
    print(f"[INFO][RotateBulbCustom] restored arm collision pairs: {report}", flush=True)
    return prim


@lru_cache(maxsize=1)
def _cloned_bulb_spawn():
    from isaaclab.sim import clone

    return clone(_spawn_bulb_robot_source)


def spawn_bulb_robot_with_self_collision(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Repair the source before Isaac Lab clones it into every environment."""
    return _cloned_bulb_spawn()(prim_path, cfg, translation, orientation, **kwargs)
