# Bulb assets

This directory contains two locked bulb assets and a screw-in HiveBoard variant:

| Asset | Recommended entry point | Structure |
| --- | --- | --- |
| Lightbulb with metal socket | `lightbulb_with_socket/lightbulb_with_socket.usd` | One rigid body, zero joints |
| HiveBoard lamp with large base | `hiveboard_lamp/hiveboard_lamp_locked.usd` | One rigid body, zero joints |
| HiveBoard screw-in/out lamp | `hiveboard_lamp/hiveboard_lamp_threaded.usd` | Fixed base, coupled rotation/translation, thread collisions |

For the new **可旋入 / 可旋出版本**, see [使用说明](hiveboard_lamp/THREADED.md).
It preserves both visible thread meshes, uses native PhysX screw coupling,
and includes an isolated thread-contact test. Load it as an **Articulation**,
not a `RigidObject`. The locked files remain unchanged.

The two locked USD files are self-contained, use metres and Z-up, and place the
`PhysicsRigidBodyAPI` on the default prim. Internal geometry is therefore
locked in simulation while the complete object can still be posed or moved as
one rigid body. The accompanying URDF files provide the same joint-free
contract for non-USD tools.

## Isaac Lab

Use the checked-in USD directly with `UsdFileCfg`:

```python
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

bulb_usd = Path("assets/bulb/lightbulb_with_socket/lightbulb_with_socket.usd").resolve()

bulb_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Lightbulb",
    spawn=sim_utils.UsdFileCfg(usd_path=str(bulb_usd)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
)
```

Replace `bulb_usd` with
`assets/bulb/hiveboard_lamp/hiveboard_lamp_locked.usd` to load the large-base
lamp. To keep the whole lamp fixed to the scene, override its root rigid-body
properties with `kinematic_enabled=True`; this is separate from the already
locked internal geometry.

## Locked geometry contract

- The lightbulb glass uses scale `(1.01, 1, 1)`. The socket uses scale
  `(1.05, 1, 1)` and translation `(0.002, 0, 0)` metres. These are the exact
  transforms from the Dexonomy MJCF adapter and are baked into the USD points.
- The HiveBoard asset is frozen at `RevoluteJoint=0 rad` and
  `PrismaticJoint=0 m`. Its source body transforms are preserved, the joint
  prims are removed, and all colliders belong to one compound rigid body.
- `manifest.json` records the default prim, units, mass, joint count, and
  locked transforms so consumers do not have to infer them from a loader.

The original HiveBoard articulation, meshes, printable STL files, and editable
STEP files remain under `hiveboard_lamp/source/`. The merged object_sim STL and
its split visual/collision meshes remain under `lightbulb_with_socket/meshes/`.

## Rebuild and validate the locked variants

The outputs were generated with Isaac Sim 5.1 from the packaged source files:

```bash
conda run -n brainco python assets/bulb/tools/build_locked_usd.py
conda run -n brainco python assets/bulb/tools/validate_assets.py
conda run -n brainco python assets/bulb/tools/smoke_spawn.py
```

The validator checks that each recommended USD opens, has exactly one rigid
body, has zero joints, contains collision geometry, uses metres/Z-up, and has
no external asset dependencies. It also checks that each URDF is a single-link,
joint-free model whose referenced meshes are present.
`smoke_spawn.py` goes one step further and instantiates both recommended USDs
through Isaac Lab's `RigidObject` API, resets them, and advances physics.

## Provenance and licenses

- `lightbulb_with_socket`: `vikashplus/object_sim` at commit
  `5257bbdbe7e3af32aae5ad67ffa1d4402ac2e871`, Apache-2.0.
- `hiveboard_lamp`: `EESC-LabRoM/HiveBoard` at commit
  `cb6526a619157ac81329a72566267a5d6285685d`, MIT.

See each asset's `UPSTREAM.md` and the files under `licenses/`.
