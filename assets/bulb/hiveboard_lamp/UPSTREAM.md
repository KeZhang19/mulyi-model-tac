# HiveBoard lamp asset provenance

The files under `CAD/Lamp`, `STL/Lamp`, and `Simulation/Lamp` were copied
without modification from the HiveBoard repository:

- Repository: https://github.com/EESC-LabRoM/HiveBoard
- Commit: `cb6526a619157ac81329a72566267a5d6285685d`
- Commit date: 2026-09-07
- License: MIT (see `LICENSE` in this directory)

## Asset roles

- `CAD/Lamp`: editable STEP sources for the lamp shell, screw, base, and assembly.
- `STL/Lamp`: printable lamp half-shell, screw, and base meshes. STL dimensions
  are expressed in millimetres.
- `Simulation/Lamp`: URDF/USD assembly and OBJ meshes in metres. The URDF
  contains a fixed base (`World`), a continuously rotating screw axis, and a
  prismatic insertion axis with a 0.024 m range.

The upstream directory layout is preserved. These source assets are not yet
wired into the Dexonomy smoke scenario; that scenario continues to use
`assets/object/object_sim/lightbulb` until a MuJoCo adapter and grasp-only mesh
are prepared and validated for the HiveBoard geometry.
