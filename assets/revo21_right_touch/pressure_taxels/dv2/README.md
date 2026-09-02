# DV2 pressure-taxel layouts

These point and normal maps come from `DV2pointcollection.dxf`; fingertip
electrodes are intentionally excluded.

`dv2_pressure_taxel_layout.urdf` is the canonical metadata entry point used by
the Revo3 RL environment. It resolves all NPY files relative to this directory
and yields 11 pressure regions / 285 observation values in runtime link order.

- `thumbmcp`: right-thumb large finger-pad pattern, 43 taxels.
- `thumbpip`: right-thumb small finger-pad pattern, 10 taxels.
- Other `mcp` links: shared four-finger large pattern, 39 taxels each.
- Other `pip` links: shared four-finger small pattern, 10 taxels each.
- `right_hand_rubber_link`: right-palm pattern, 36 taxels: 31 large electrodes
  (IDs 01-18 and 24-36) and five small electrodes (IDs 19-23).

Each NPY file has shape `(1, taxel_count, 3)`. Point values are millimetre
offsets from the corresponding URDF `<pressure_pad><origin>`; use
`correction_scale="0.001"` when loading them from a pressure-pad declaration.
Normals are unit vectors in the touch-link frame.

The palm also has `right_hand_rubber_link_radii_m.npy`, shape `(1, 36)`, in SI
metres. It records the nominal DXF outer footprint radius: `3.5 mm` for the 31
large electrodes and `2.5 mm` for IDs 19-23. This radius is visualization and
footprint metadata; the current WarpSDF backend still queries penetration at
the stored center point rather than integrating a finite circular support.

The pruned maps preserve the surviving original DXF order. The four-finger
MCP pattern is recentered on each URDF pressure-pad origin in the tangent
plane, then reprojected onto that link's outer STL surface along local `-X`.
The thumb MCP layout is reprojected with its transverse center aligned and its
distal center at `-5.9422 mm`, leaving a `2.5 mm` center-to-edge margin on the
closest longitudinal side to match the DXF while all 43 points remain on the
unscaled STL. The PIP layouts are cropped without recentering:

- `TMCP`: removed original IDs 44-60.
- `TPIP`: removed original IDs 11-13.
- `FMCP`: removed original IDs 40-48.
- `FPIP`: removed original IDs 11-17.

The finger patterns are centred on their active-taxel bounds without scaling.
DXF `+y` points distally, while DXF `+x` follows positive surface arc length
across the touch STL. Points are wrapped onto each link's actual outer STL
surface, so every finger has its own map even when it uses the shared four-
finger pattern. The palm uses DXF `x -> link +Y`, DXF `y -> link +Z`, keeps the
horizontal scale at `1.0`, and uses vertical scale `0.968` plus the accepted
translation so the top, bottom, and right center-to-edge padding each include
the requested extra `2 mm`.

Taxel order follows the DXF entity order except palm ID 36, whose exploded
`4 ARC + 12 LINE` geometry is appended to preserve the already reviewed IDs
01-35. The drawing contains no channel or pin identifiers, so this order must
not be treated as a hardware channel map.
