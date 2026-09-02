# SDF-like Pressure Tactile Pipeline

This note records the reproducible pressure trace and L1 diagnostic/reference workflow for the `ziyiwang54/pressure-sdf-tactile` branch. References in this file are model or benchmark references unless explicitly marked as real measured pressure ground truth.

Command examples use portable variables rather than user-specific absolute paths:

```bash
export REPO_ROOT=${REPO_ROOT:-$PWD}
export ISAACLAB_SH=${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}
export ISAACLAB_CONDA_PREFIX=${ISAACLAB_CONDA_PREFIX:-$CONDA_PREFIX}
```

## Core Trace Contract

Saved pressure traces use SI units and the main arrays follow `(T, sensor, H, W)`:

- `penetration_m`: SDF/taxel penetration depth.
- `signed_distance_m`: signed distance before clamping.
- `penetration_velocity_mps`: finite-difference penetration velocity.
- `pressure_raw_n`: calibrated normal force in Newtons before normalization. The name is retained for compatibility; semantically this is force, not pressure in Pascals.
- `pressure_norm`: normalized display/learning map.
- `total_force_n`: summed raw force per sensor.
- `center_of_pressure_px`: force-weighted center of pressure.
- `tacmap_raw_m`: optional TacMap dense reference depth.
- `physx_contact_count`: optional sparse PhysX contact sanity channel.
- `geometry_normal_ray_sample_support_fraction`: optional finite-area geometry normal-ray support ratio per taxel.
- `geometry_normal_ray_sample_active_count`: optional number of finite-area samples inside the closed geometry per taxel.
- `geometry_normal_ray_sample_mean_penetration_m`: optional mean finite-area sampled penetration, including zero samples.
- `geometry_normal_ray_sample_positive_mean_penetration_m`: optional mean penetration over active finite-area samples only.
- `geometry_normal_ray_sample_max_penetration_m`: optional maximum finite-area sampled penetration.
- `geometry_normal_ray_sample_penetrations_m`: optional raw finite-area sample penetrations with shape `(T, sensor, H, W, sample)`.
- `geometry_normal_ray_sample_offsets_l`: optional finite-area sample offsets in the touch-link frame with shape `(T, sensor, H, W, sample, 3)`.
- `geometry_normal_ray_sample_points_l_m`: optional finite-area sample local positions with shape `(T, sensor, H, W, sample, 3)`.
- `reference_layer`: optional verifier/report metadata for dense comparisons, inferred from `reference_key` when omitted. Current values are `L0_analytic`, `L1_model_reference`, `L1_sparse_sanity`, `L2_offline_oracle`, `L3_real_calibration`, or `benchmark_reference`.
- `pressure_taxel_points_l_m` / `pressure_taxel_normals_l`: static pressure taxel layout in the touch-link frame.
- `tacmap_grid_points_l_m` / `tacmap_ray_directions_l` / `tacmap_grid_axes_l`: static TacMap link-surface ray grid in the same local frame, read from the live TacMap sensor when available.
- `normal_ray_alignment_source_index` / `normal_ray_alignment_valid_mask` / `normal_ray_alignment_nn_distance_m`: optional alignment from pressure taxels to TacMap link-surface ray origins.
- `geometry_normal_ray_origin_points_l_m` / `geometry_normal_ray_origin_valid_mask`: optional diagnostic ray origins aligned from the TacMap link-surface grid onto the pressure taxel grid.

The trace contract is now executable via `validate_pressure_trace_v1()`. A valid v1 trace must include core `(T, S, H, W)` arrays, `total_force_n`, `center_of_pressure_px`, pressure taxel layout arrays, and metadata keys `pressure_trace_schema_version`, `pressure_backend_id`, `pressure_layout_id`, and `pressure_calibration_id`. Older traces may report metadata warnings, but malformed core arrays are errors. Both `scripts/force_map/verify_pressure_trace.py` and the integrated runner's `--verify-pressure-trace` output include a top-level `contract` block.

## Revised Validation Plan

The pressure work should optimize for Isaac/geometry-reproducible dense pressure maps first. Use `GT` only for strict analytic checks, explicitly declared offline physical oracles, or real pressure-pad measurements with matching layout/calibration. L1 Isaac/TacMap/normal-ray outputs should be called dense model references, not physical dense pressure-map GT. Vision-based tactile sensing datasets are useful for later real-domain calibration, but they should not be used as the primary dense pressure-map GT unless the same hand layout, contact geometry, material model, and label generation process are reproduced.

Execution order:

- Keep the Warp-based pressure sensor as the online implementation, but use the normal-ray `surface_gap` contact model by default: pressure depth is `max(shell_thickness - normal_gap_to_object_surface, 0)`, not `clamp(-signed_sdf, 0)`. The default response maps this shell-closure fraction directly to pressure, so it does not require a Kelvin-Voigt spring/damper to create pressure. Use signed penetration only for historical comparisons.
- Freeze `pressure_trace_v1` as the handoff format between backends, verifiers, and future URDF pressure pads. URDF tags discover/bind pressure pads; calibration and force/depth semantics live in sidecar metadata and trace fields.
- Use `L0 analytic` traces for strict CI-style regression on simple pressers.
- Use `L1 model reference` traces from TacMap/normal-ray alignment to diagnose dense footprint, centroid, onset, and depth behavior inside Isaac-compatible runs.
- Treat `GeometryNormalRayPenetrationSource` as a candidate diagnostic backend until its ray-origin semantics and closed-mesh assumptions match the chosen reference. Do not promote it to physical dense GT just because it is generated from object geometry.
- Use `geometry_normal_ray_sample_*` trace arrays to explain finite-area taxel boundary cases. Aggregate fields show whether a taxel failed because its center missed contact, only a small fraction of its area was supported, or the aggregated depth model was too conservative. Raw per-sample fields preserve the local sample positions and depths needed for richer area-fraction or boundary-shape analysis.
- Keep PhysX contact as sparse sanity only: onset/total force/contact count, not dense taxel GT.
- Bring Hydroelastic/FEM/UIPC-style traces in as small offline `L2 oracle` cases only after L0/L1 is stable.
- Use GelSight/DIGIT/Sparsh/FeelAnyForce/real pad data as `L3 real calibration` for gain, gamma, saturation, hysteresis, crosstalk, and sim-to-real transfer. Stiffness/damping are now only needed when explicitly running the historical `penetration_kv` response.

## Reference And Ground Truth Policy

Do not treat vision-based tactile force-estimation datasets as the default dense pressure-map ground truth for this SDF-like pipeline. GelSight/DIGIT/Sparsh/FeelAnyForce-style work is valuable, but its force labels are usually external F/T readings, learned force regressions, or FEA-derived pseudo-labels from tactile images. Those labels are useful for real-domain calibration and sanity validation, not as a direct IsaacLab-reproducible `(sensor, H, W)` pressure oracle.

Use this hierarchy instead:

- `L0 analytic`: closed-form or scripted indenter checks for footprint, centroid, onset, depth, and monotonic force. This is the strict regression oracle for simple shapes.
- `L1 simulation reference`: SDF/normal-ray/geometric penetration maps generated from the same taxel layout, object geometry, and poses as the simulation. This is the main dense model reference for the current project, not physical GT.
- `L1 sanity`: PhysX/Isaac contact data. Use it for contact onset, total force trend, and sparse contact sanity only; do not promote it to dense pressure-map ground truth.
- `L2 offline oracle`: hydroelastic or FEM/UIPC traces for a small set of representative poses. These can provide higher-fidelity pressure or traction fields, but they are offline validation targets rather than the online IsaacLab training path.
- `L3 real calibration`: new hand pressure pads, pressure films, Tekscan/PPS-style arrays, or load-cell experiments. Use these to calibrate `surface_gap -> pressure` parameters such as shell thickness, deadband, gain, gamma, saturation, hysteresis, and crosstalk.

TacSL/Isaac Lab visuo-tactile can output force fields, but it is still a sensor model based on SDF/penetration and penalty-style contact. Treat it as a comparable model-generated tactile reference, not as independent physical ground truth. TacMap/normal-ray traces in this document are likewise diagnostic/model references unless they are explicitly compared against analytic, hydroelastic/FEM, or real pad data. A geometry-derived reference is only acceptable when the mesh topology, taxel ray origins, normals, and contact/deformation semantics are explicitly matched to the pressure layout.

The verifier records this distinction in every dense comparison report via `reference_layer`. For example, `analytic_depth_m` is `L0_analytic`, while `tacmap_raw_m`, `normal_ray_penetration_m`, and `geometry_normal_ray_penetration_m` are `L1_model_reference`. PhysX contact channels remain `L1_sparse_sanity` and are not accepted as dense pressure-map GT.

The default dense acceptance gate is now executable, not just prose:

- `evaluate_pressure_trace_report()` only allows `L0_analytic`, `L1_model_reference`, and `L2_offline_oracle` to drive dense spatial pass/fail by default.
- `L1_sparse_sanity`, `L3_real_calibration`, and `benchmark_reference` may still be compared and reported, but they fail the `reference.layer_allows_dense_acceptance` check unless explicitly allowed.
- Dense scalar metrics compare `penetration_key` against `reference_key`, so those arrays must use the same scalar meaning and unit. For example, compare `penetration_m` against `uipc_penetration_m`, or compare force/pressure maps only after choosing a matching prediction key and thresholds.
- Automatic L0 analytic inference is limited to positive depth/penetration-like arrays such as `analytic_depth_m`; masks and signed-distance arrays are not auto-accepted as L0 dense references.
- Automatic L1 model-reference inference is limited to depth-like keys such as `normal_ray_penetration_m`, `geometry_normal_ray_depth_m`, or known TacMap depth keys such as `tacmap_raw_m`; pressure maps such as `normal_ray_pressure_norm` remain `benchmark_reference` unless explicitly labeled.
- Automatic L2 inference is intentionally limited to depth-like keys such as `uipc_penetration_m` or `hydroelastic_depth_m`; pressure/traction fields such as `uipc_pressure_mpa` and raw distance fields such as `uipc_distance_m` require an explicit reference layer and matching comparison keys.
- Velocity arrays and raw per-sample penetration tensors such as `normal_ray_penetration_velocity_mps` or `geometry_normal_ray_sample_penetrations_m` are not auto-accepted as dense references. If a raw per-sample tensor is passed as `reference_key`, the verifier reports `reference.available=false` instead of treating it as a dense `(T, sensor, H, W)` map.
- Aggregated sample penetration maps such as `geometry_normal_ray_sample_mean_penetration_m` remain eligible because they are `(T, sensor, H, W)` positive depth maps; the raw per-sample tensor stays diagnostic-only.
- Signed-distance arrays such as `normal_ray_signed_distance_m` are not auto-accepted as dense references because the default verifier treats references as positive penetration/depth maps.
- L2 depth-like keys and L3 real-calibration keys take precedence over generic `contact` naming, so `uipc_contact_penetration_m` and `real_pad_contact_pressure_n` are not misclassified as PhysX-style sparse sanity.
- Real-domain tactile calibration keys such as `real_pad_*`, `tekscan_*`, `pps_*`, `pressure_film_*`, `load_cell_*`, `gelsight_*`, `digit_*`, `sparsh_*`, `tacbench_*`, `feats_*`, `feelanyforce_*`, and `vision_tactile_*` are inferred as `L3_real_calibration`, which keeps them usable for calibration/validation while excluding them from default dense Isaac/SDF acceptance.
- Isaac/TacSL-style model sensor outputs such as `tacsl_*`, `visuo_tactile_*`, `visuotactile_*`, and `vbts_*` are inferred as `benchmark_reference` unless explicitly relabeled. They can be compared for diagnostics, but they are not real calibration evidence and do not drive default dense acceptance.
- L1 model-reference keys also take precedence over generic `contact` naming, so `tacmap_contact_depth_m` and `normal_ray_contact_penetration_m` remain model references.
- Sparse-sanity inference is limited to PhysX/contact-force/count style keys; generic contact masks are not auto-promoted to dense references.
- The current test suite has an L2 acceptance fixture for `uipc_penetration_m`; it verifies the gate, not a real FEM/UIPC oracle integration.
- If a dense reference has an alignment valid mask, the verifier reports `alignment_valid_taxel_fraction_*`, `alignment_contact_region_valid_fraction_*`, `alignment_invalid_contact_fraction_max`, and `alignment_invalid_zero_fill_fraction_*`.
- If the trace also has pressure taxel points, TacMap grid points, and `normal_ray_alignment_*`, the verifier reports `reference.origin_alignment`: `delta_direction = tacmap_grid_points_l_m - pressure_taxel_points_l_m`, plus `delta_norm_m_*`, signed `normal_delta_m_*`, absolute normal delta, tangent delta, and nearest-neighbor distance stats. This is diagnostic evidence for ray-origin semantics; it is not an acceptance metric by itself.
- Contact-region alignment is gated by default: `alignment_contact_region_valid_fraction_min` must be `>= 1.0`, and `alignment_invalid_contact_fraction_max` must be `<= 0.0`. Overall valid taxel fraction is reported, and can be gated explicitly with `--reference-alignment-valid-fraction-threshold`.
- Use `scripts/force_map/verify_pressure_trace.py --dense-reference-layer <layer>` or `integrate/run_integrated_tactile.py --pressure-verify-dense-reference-layer <layer>` only for deliberate diagnostics or a future real dense pressure-pad calibration gate.
- Use `--reference-valid-mask-key <key>` or `--pressure-verify-reference-valid-mask-key <key>` to bind a specific alignment mask. If omitted, the verifier uses `<reference-key>_valid_mask` when present, and also recognizes the internal `normal_ray_alignment_valid_mask` fallback for `normal_ray_*` references.
- Manually relabeling an inferred reference layer fails the `reference.layer_override_is_safe` check by default, even when both layers are dense-capable. Use `--allow-reference-layer-override` or `--pressure-verify-allow-reference-layer-override` only when the override is intentional and documented.
- CLI smoke confirms this gate: `analytic_depth_m --reference-layer L1_model_reference --fail-on-threshold` exits `2`, and the same command with `--allow-reference-layer-override` exits `0`.

Multi-agent and source review notes:

- Isaac Lab `ContactSensor` is body/filter scoped and is documented as returning net contact force on rigid bodies, so it is appropriate for sparse sanity rather than dense taxel GT: [Isaac Lab Contact Sensor](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html).
- Isaac Lab/Isaac Sim contact buffers can lose accuracy in contact-rich cases unless buffer sizes are raised; this supports keeping PhysX contact data out of dense acceptance: [Isaac Lab ContactSensor API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html).
- Isaac Lab `VisuoTactileSensor`/TacSL can provide tactile RGB, penetration, and force-field style outputs, but the documented pipeline is still sensor-model based rather than independent physical GT: [Isaac Lab Visuo-Tactile Sensor](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html), [Isaac Lab contrib sensor API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_contrib/isaaclab_contrib.sensors.html), [TacSL](https://arxiv.org/html/2408.06506v1).
- TacMap is useful here because it makes penetration/deform maps the shared representation, not because vision-based tactile images are dense pressure GT by themselves: [TacMap](https://arxiv.org/abs/2602.21625).
- Hydroelastic/FEM-style work is closer to a dense pressure-field oracle, but it should enter as small offline L2 cases unless the same geometry, material, and boundary conditions are reproduced in this project: [HydroelasticTouch](https://arxiv.org/html/2501.08077v1).
- Sparsh/GelSight/DIGIT/FeelAnyForce style datasets are valuable real-domain evidence, but they mainly provide tactile images plus external force labels or learned force labels, not an IsaacLab-reproducible pressure map for this hand layout: [Sparsh](https://github.com/facebookresearch/sparsh), [FeelAnyForce](https://prg.cs.umd.edu/FeelAnyForce).
- FEATS is closer to dense force distribution, but its labels come from FEA inferred from GelSight Mini data; use it as an L2/L3 methodology reference unless the same geometry/material/boundary conditions are reproduced: [FEATS](https://feats-ai.github.io/).

## Revo21 Pressure-Origin Convention

Current simulation convention for `assets/revo21_right_touch`:

- URDF `*_touch_link` / toucher is the physical pressure-sensing pad link. In this document and code, a `pressure_pad` is not a second surface, extra link, or replacement mesh; it is the taxel layout and calibration metadata attached to that same toucher link.
- Therefore `pressure_layout_link`, `press_touch_link`, and the URDF link carrying `<pressure_pad>` should normally name the same link, for example `right_midpip_roll_touch_link`. The runner uses that link both as the rigid body whose pose defines the pressure frame and as the link whose collision/visual surface represents the pad.
- Pressure tactile origin means the pad surface taxel point. It is the SDF/ray origin used for contact onset and pressure-map footprint.
- Do not use the URDF fixed-joint origin as the pressure origin. In `assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf`, every `*_touch_joint` has `xyz="0 0 0"` and `rpy="0 0 0"`, while the touch STL meshes are offset from that frame. The joint origin is a CAD/link-frame anchor, not the pad contact surface.
- Fingertips are reserved for visuo-tactile sensing. Pressure tactile covers the remaining declared touch pads, currently the `mcp` and `pip` touch links such as `right_indexmcp_roll_touch_link`, `right_indexpip_roll_touch_link`, `right_midmcp_roll_touch_link`, `right_midpip_roll_touch_link`, `right_ringmcp_roll_touch_link`, `right_ringpip_roll_touch_link`, `right_pinkymcp_roll_touch_link`, `right_pinkypip_roll_touch_link`, `right_thumbmcp_roll_touch_link`, and `right_thumbpip_roll_touch_link`.
- Internal pad/taxel points may still be useful for real pressure-pad calibration, but they are not the online SDF contact origin. If the real sensor later reports an internal taxel layer, store the offset/depth as calibration metadata and project to the surface for SDF onset.
- The loader accepts `origin_semantics`, `origin_type`, or `taxel_origin` on pressure-pad tags. The default is `pad_surface`; accepted normalized values are `pad_surface`, `pad_internal`, and `link_surface`. Do not use the URDF `<origin>` transform for this semantic label.
- A future URDF may declare grid resolution as `rows/cols`, `num_rows/num_cols`, `h/w`, or `resolution="HxW"` / `taxel_resolution="HxW"`. It may declare square spacing as `point_distance`, `taxel_pitch`, or `pitch`, or rectangular spacing as `row_pitch` / `col_pitch` and their distance aliases. It may also declare `pad_size="<row_extent> <col_extent>"`; by default this means the pressure-sensitive surface footprint is divided by `rows/cols` (`pad_size_semantics="cell_extent"`), while `pad_size_semantics="center_span"` treats the size as the distance between the outermost taxel centers. A pressure-pad child `<origin xyz="..." rpy="...">` is supported as the pad-local frame pose inside the link frame; grid points and normals are transformed through this pose before they reach WarpSDF. It may also declare `taxel_count`; the loader treats count as a consistency check against `rows * cols` or the loaded point map size. It does not infer a layout from count alone. Supported declaration forms include pressure-pad tags under a link, `<sensor type="pressure" link="...">`, and child `<origin>`, `<grid>`, `<taxels>`, `<map>`, or `<calibration>` tags.
- The loader can also list current tactile candidates from URDF link names: all `touch_link` names are discoverable, and pressure-touch candidates are the toucher links intended to carry pressure-pad metadata. They currently exclude only explicit `tip` / `fingertip` links. This keeps `pip` and `mcp` touch links available for pressure tactile while leaving fingertip links for visuo-tactile sensing. Explicit pressure-pad declarations are still parsed even when the link name does not match `touch_link`; `inspect_pressure_urdf.py` reports these under `pressure_pad_links_not_in_pressure_touch_candidates` for audit because such a declaration would no longer be attached to an obvious toucher link.
- Inspect a new URDF before running simulation with `python scripts/force_map/inspect_pressure_urdf.py <hand.urdf>`. The output lists all `touch_links`, pressure-touch candidates, declared pad `rows/cols/taxel_count/origin_semantics`, spacing (`point_distance` or `row_distance/col_distance`), optional `pad_size`, pad frame `origin_xyz/origin_rpy`, `normal_axis/normal_sign`, and parsed calibration fields. Use `--require-pressure-layouts` once the new URDF is expected to be complete; it exits `2` if a pressure candidate lacks a pad declaration, a pad only declares count without an actual grid/point-map layout, `taxel_count` disagrees with `rows * cols`, grid layout fields are invalid (`rows/cols <= 0`, pitch/derived pitch <= 0, or `normal_axis` outside `0..2`), a pressure pad is duplicated for one link, a pressure pad is declared on an explicit `tip` / `fingertip` link, or a pressure pad origin is not `pad_surface`. Add `--check-layout-files` when a URDF declares `points_npy/normals_npy`; it loads those files and checks their shape/count through the same taxel-map loader used by the simulation.
- Render the declared taxel point lattice with `python scripts/force_map/visualize_pressure_pad_layout.py <hand.urdf> --out-dir outputs/pressure_pad_layout`. This uses the same `UrdfPressurePadSpec.to_taxel_map()` path as simulation, writes `pressure_pad_layout.svg` for human inspection, and writes `pressure_pad_layout.json` with each pad's `rows/cols/taxel_count`, pitch, optional `pad_size`, pad frame pose, local bounds, projection axes, and projected taxel coordinates. Use `--link <pressure_link>` to inspect one pad, and `--include-points` when the full local `points_l_m/normals_l` arrays are needed for debugging.
- For the current integrated WarpSDF runner, a grid pressure pad can be applied with `--robot-urdf <hand.urdf> --pressure-layout-urdf <hand.urdf> --pressure-layout-link <pressure_link>`. This updates the selected pressure link, touch collision path, grid resolution, square or rectangular spacing, pad frame pose, normal axis/offset/sign, scalar calibration fields, and press center/normal before explicit CLI calibration overrides are applied. `--normal-sign` is resolved while loading the URDF layout so the old press-motion module and the WarpSDF taxel normals agree on which side of the pad is outside. `--point-distance` remains a square-pitch override; `--row-distance` and `--col-distance` can override rectangular pitch explicitly. The selected pad must declare `origin_semantics="pad_surface"` or use the default `pad_surface`. Arbitrary `points_npy/normals_npy` pressure maps are intentionally not coerced into the current uniform-grid runner, and old TacMap point files/fallbacks are not reused for a URDF grid pad.
- Saved pressure traces record `pressure_layout_source`, `pressure_layout_urdf`, `pressure_layout_link`, `pressure_layout_name`, `pressure_origin_semantics`, declared/actual taxel count, actual `press_touch_link`, `press_center_l/press_normal_l`, `press_patch_pos_l/press_patch_quat_l`, `row_distance_m/col_distance_m`, pad frame pose, and optional pad-size metadata so runs can be audited without relying on the `--finger` label.
- `link-surface` points remain a model-reference frame for TacMap/normal-ray comparison. They are only equivalent to pressure origins after explicit alignment/projection. The current L1 blocker exists because pressure-taxel origins and TacMap link-surface origins differ by about `10.066 mm` along the tactile normal.

## Right Midpip Pressure-Pad Smoke

The current generated pressure-pad URDF has an 8x12, `1.5 mm` pitch, `pad_surface` grid on `right_midpip_roll_touch_link`. For this provisional pad, the declared frame is on the local `-Z` outside surface of the STL bounds at `origin xyz="0.00194958 0.00010799 0.00468356"` with `normal_sign="-1"`, so the old press-motion module places the cylinder on the visible pad side instead of the back side. This is a pressure pad on the middle finger PIP touch link, not a fingertip visuo-tactile sensor. All generated pressure-pad touch links and their matching outer `rubber`/`tubber` skin links use visual material `pressure_pad_cyan` with `rgba="0 0.85 1 1"` so the marked pad locations are visible on the outer hand surface after URDF import; this visual color does not change collision geometry or pressure calibration.

Isaac's URDF-to-USD converter does not reliably preserve the URDF `<material><color>` on these generated touch links. After conversion, bind the material directly on the converted USD:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
$ISAACLAB_SH -p scripts/force_map/color_pressure_pad_usd.py \
  scripts/force_map/official_replay/output/revo2_urdf/revo21_dv2_urdf_right-touch.SLDASM.pressurepad.usd \
  --urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --out-json outputs/pressurepad_usd_color_bind.json
```

The expected result is `passed=true`, `pressure_pad_count=10`, `colored_visual_link_count=20`, and all pressure `*/touch_link/visuals` plus matching outer `*/rubber_link/visuals` or `*/tubber_link/visuals` prims bound to `/Looks/pressure_pad_cyan`. Open `scripts/force_map/official_replay/output/revo2_urdf/revo21_dv2_urdf_right-touch.SLDASM.pressurepad.usd` in Isaac Sim to see the cyan pads on the visible surface.

The integrated runner also binds the same material to the live Isaac stage after spawning the robot. A smoke run should print `pressure pad visual material applied: 20/20 links`; this confirms the GUI scene is colored even if the URDF importer ignores URDF material colors.

To view the Revo21 middle-finger PIP pressure pad in the same simple press scenario as main, drive the presser from the URDF pressure layout and hold the alternate-URDF robot joints at zero pose:

```bash
DISPLAY=:1 CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --mode press --finger middle \
  --robot-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-link right_midpip_roll_touch_link \
  --press-hold-joint-pose zero \
  --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 700 \
  --show-sample-points --show-sample-axes \
  --live_web --live_port 8090 \
  --disable-tacmap
```

For GUI placement work, use `--press-setup-before-motion`. The runner opens an editable setup phase first; select `/World/Plug` in the Isaac Sim stage, drag/rotate it to the desired initial world pose, then press Enter in the terminal to start the scripted press trajectory from that current pose:

```bash
DISPLAY=:1 CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --mode press --finger middle \
  --robot-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-link right_midpip_roll_touch_link \
  --press-hold-joint-pose zero \
  --press-object-control scripted \
  --press-setup-before-motion \
  --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 700 \
  --presser-pose-save-file outputs/pressure_gui_demo/right_midpip_presser_pose.json \
  --press-debug-log outputs/pressure_gui_demo/right_midpip_press_debug.jsonl \
  --press-debug-log-every 5 \
  --live_web --live_port 8090 \
  --disable-tacmap
```

This setup flow does not need JSON to drive the pose. `--presser-pose-save-file` is only an optional record of the final `/World/Plug` root pose, useful for notes or later exact replay. If you need to trigger the start without terminal focus, add `--press-setup-start-file outputs/pressure_gui_demo/start_press.flag` and create that file when the world is ready.

For collaborative GUI debugging, keep `--press-debug-log` enabled. The JSONL log records setup frames, the Enter transition, run frames, `/World/Plug` pose, selected toucher pose, middle/index joint deltas, middle/index body deltas, press counters, and pressure-map activity. This makes it clear whether a run is moving the Plug, the middle finger, the index finger, or nothing.

For manual action tuning, run the GUI without `--no-local-ui` and add `--live-control-file`. The runner writes or reloads a JSON file while Isaac is running; editing it changes the presser target on the next frame while the tactile force-map panel and optional web view update from the same simulation:

```bash
DISPLAY=:1 CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --mode press --finger middle \
  --robot-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-urdf 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf' \
  --pressure-layout-link right_midpip_roll_touch_link \
  --press-hold-joint-pose zero \
  --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset -0.006 --press-steps 600 --max_steps 1200 \
  --show-sample-points --show-sample-axes \
  --live-control-file outputs/pressure_gui_demo/right_midpip_live_control.json \
  --write-live-control-template \
  --live_web --live_port 8090 \
  --disable-tacmap
```

Edit `outputs/pressure_gui_demo/right_midpip_live_control.json` while the command is running. `center_offset_l`, `center_l`, `normal_l`, and `slide_offset_l` are in `right_midpip_roll_touch_link` local coordinates and meters; `offset_m` overrides the scripted press depth when it is not `null`. Negative `offset_m` values press into the pad. Use the Isaac force-map panel or `http://localhost:8090` to watch the pressure map while tuning.

The verified command shape is:

```bash
PRESSURE_URDF="$REPO_ROOT/assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf"

CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle \
  --robot-urdf "$PRESSURE_URDF" \
  --pressure-layout-urdf "$PRESSURE_URDF" \
  --pressure-layout-link right_midpip_roll_touch_link \
  --press-hold-joint-pose zero \
  --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset -0.006 --press-steps 60 --max_steps 20 \
  --save_maps --save-pressure-trace --verify-pressure-trace \
  --pressure-trace-dir $REPO_ROOT/outputs/right_midpip_pressurepad/cylinder_D4_final \
  --pressure-verify-out $REPO_ROOT/outputs/right_midpip_pressurepad/cylinder_D4_final
```

For `square_4`, use the same command with `--presser square_4 --presser-extra-rot-axis y --presser-extra-rot-deg -90 --press-end-offset -0.02 --max_steps 14` and write to `outputs/right_midpip_pressurepad/square_4_deep_final`.

Current evidence:

- Cylinder trace: `outputs/right_midpip_pressurepad_smoke_surface_gate/cylinder_D4/20260624_113123.npz`, schema valid, no pre-contact leakage, `normal_l=[0, 0, -1]`, peak active taxels `24/96`, max total force `90.0258 N`, onset step `7`.
- Square trace: `outputs/right_midpip_pressurepad_smoke_surface_gate/square_4/20260624_113129.npz`, schema valid, no pre-contact leakage, peak active taxels `24/96`, max total force `90.0208 N`, onset step `5`.
- Peak-frame comparison image: `outputs/right_midpip_pressurepad/comparison/right_midpip_peak_contact_sheet_labeled.png`. Left is `cylinder_D4`, right is `square_4`.

## L0 Analytic Check

Generate an analytic trace:

```bash
python3 scripts/force_map/generate_pressure_l0_trace.py   --out-dir outputs/pressure_l0_smoke   --run-id square_l0   --presser square   --steps 8   --indent-start 0   --indent-end 0.002
```

Verify it strictly:

```bash
python3 scripts/force_map/verify_pressure_trace.py outputs/pressure_l0_smoke/square_l0.npz   --reference-key analytic_depth_m   --reference-iou-threshold 1.0   --centroid-error-threshold-px 0.0   --bbox-error-threshold-px 0.0   --depth-rmse-threshold-m 0.0   --onset-error-threshold-frames 0   --offset-error-threshold-frames 0   --fail-on-threshold
```

Current evidence: strict L0 passes all checks.

Geometry normal-ray sanity against L0:

- `outputs/pressure_geometry_l1_sanity/square_box_geometry.npz` uses the existing geometry normal-ray path on a one-frame analytic square press represented as an equivalent box.
- Verification against `analytic_depth_m` passes with `active_mask_iou_min = 1.0`, zero centroid/bbox/onset/offset error, and `depth_rmse_m_max = 2.33e-10 m` under a `1e-9 m` numerical tolerance.
- The unit test `test_write_geometry_normal_ray_trace_watertight_mesh_matches_l0_analytic` covers the same contract through the explicit mesh path: a watertight box mesh is reported as watertight and passes dense comparison against `analytic_depth_m`.
- `scripts/force_map/generate_pressure_watertight_mesh.py` can generate `.vertices.npy/.triangles.npy` for simple closed `box` or `cylinder` pressers. The cylinder smoke in `outputs/pressure_watertight_mesh_smoke/cylinder_geometry_mesh.verify.json` passes strict mask/centroid/bbox/onset checks against L0 analytic cylinder, with `depth_rmse_m_max = 1.80e-9 m` under a `1e-8 m` numerical tolerance.
- `integrate/run_integrated_tactile.py` accepts `--geometry-normal-ray-vertices-npy` and `--geometry-normal-ray-triangles-npy`, so an Isaac smoke can compare explicit watertight meshes against the USD-loaded presser geometry.
- This proves the explicit primitive/mesh ray-distance semantics match flat analytic L0 traces. It does not prove the current USD presser mesh or a naive watertight replacement is a valid online L1 model reference; the L1 reverify section below remains authoritative for that.

## L1 TacMap Diagnostic Smoke

Run one short integrated WarpSDF + TacMap trace. This checks agreement against a model-generated deformation/depth reference, not real pressure ground truth:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm timeout 240s $ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py   --headless --mode press --finger middle --presser cylinder_D4   --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 20 --max_steps 19   --tacmap-ray-mode link_surface --no-local-ui --print-every 5   --save-pressure-trace --verify-pressure-trace --pressure-verify-reference-key tacmap_raw_m   --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_tacmap_smoke   --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_tacmap_smoke
```

Observed on `outputs/pressure_l1_tacmap_smoke/20260623_125227.npz`:

- `precontact_leakage_fraction = 0.0` passes.
- Force-depth Spearman passes.
- SDF pressure onset is step 6; TacMap diagnostic reference onset is step 12.
- Active mask IoU, centroid, bbox, and depth RMSE fail strict L1 thresholds.

This means the failure is not raw data leakage; it is a geometry/model-reference alignment issue.

## L1 Geometry Alignment

The raw TacMap image is 240x240 while the pressure map is 30x30, and the TacMap link-surface grid stores ray starts rather than the physical taxel surface. Align the model reference onto the pressure taxel grid before using spatial metrics:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand python3 scripts/force_map/align_pressure_reference_geometry.py \
  outputs/pressure_l1_true_grid_smoke/20260623_132929.npz \
  --out-trace outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.npz \
  --out-json outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.json \
  --distance-mode plane \
  --max-distance-m 0.00075
```

Then verify with the aligned reference:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand python3 scripts/force_map/verify_pressure_trace.py \
  outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.npz \
  --reference-key tacmap_raw_aligned_m \
  --out-json outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.verify.json
```

Current evidence from `outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.verify.json`:

- `precontact_leakage_fraction = 0.0` passes.
- Force-depth Spearman passes.
- `centroid_error_px_max = 0.5` passes.
- `bbox_error_px_max = 2.0` passes.
- `active_mask_iou_min = 0.0` still fails because SDF onset remains earlier than TacMap.
- `depth_rmse_m_max = 0.0011303853` still fails, so the SDF depth scale is not yet aligned to the TacMap diagnostic depth target.
- SDF pressure onset is step 6; aligned TacMap onset is step 12.

Important implementation detail: the trace writer now reads `ray_starts_att` / `ray_directions_att` from the live TacMap link-surface sensor when available. This avoids incorrectly reconstructing the grid from cfg when TacMap uses the mean tactile normal as ray direction.

## L1 Deadband Fit

Fit a penetration deadband against the TacMap diagnostic reference:

```bash
python3 scripts/force_map/fit_pressure_l1_deadband.py   outputs/pressure_l1_tacmap_smoke/20260623_125227.npz   --reference-key tacmap_raw_m   --candidate-count 80   --out-json outputs/pressure_l1_tacmap_smoke/20260623_125227.deadband_fit.json   --fit-trace-out outputs/pressure_l1_tacmap_smoke/20260623_125227.deadband_fit.npz
```

Current fitted result on the older raw TacMap diagnostic reference:

- Suggested runner arg: `--penetration-deadband 0.0010873604`.
- Onset error improves from 6 frames to 2 frames.
- Spatial IoU and shape metrics still fail.

Current fitted result on the geometry-aligned true ray-grid trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand python3 scripts/force_map/fit_pressure_l1_deadband.py \
  outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.npz \
  --reference-key tacmap_raw_aligned_m \
  --candidate-count 80 \
  --out-json outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.deadband_fit.json \
  --fit-trace-out outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.deadband_fit.npz
```

- Suggested runner arg: `--penetration-deadband 0.000315903832`.
- Geometry remains acceptable: `centroid_error_px_max = 1.0`, `bbox_error_px_max = 2.0`.
- Onset error is still 5 frames, so deadband alone does not solve the SDF-vs-TacMap timing difference.
- Depth RMSE still fails (`0.0013560118 m`), so a depth/force calibration curve or a geometry-consistent normal-ray backend is still required.

Conclusion: geometry alignment fixes most spatial-frame mismatch, and deadband is useful but insufficient. The next required implementation step is a geometry-consistent NormalRay/penetration source or a contact/onset gate that makes SDF onset/depth comparable to TacMap before enforcing strict L1 thresholds.

## L1 Normal-Ray Diagnostic Reference

Generate a pressure trace directly from the geometry-aligned TacMap deformation map. This is an internal diagnostic model reference for the trace contract and verifier, not an independent real pressure GT:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand python3 scripts/force_map/apply_pressure_normal_ray_reference.py \
  outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.npz \
  --deformation-key tacmap_raw_aligned_m \
  --out-trace outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.normal_ray.npz \
  --out-json outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.normal_ray.json \
  --stiffness 1 \
  --max-force 1
```

Verify it strictly:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand python3 scripts/force_map/verify_pressure_trace.py \
  outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.normal_ray.npz \
  --pressure-key normal_ray_pressure_norm \
  --raw-pressure-key normal_ray_pressure_raw_n \
  --penetration-key normal_ray_penetration_m \
  --reference-key tacmap_raw_aligned_m \
  --reference-iou-threshold 1.0 \
  --centroid-error-threshold-px 0.0 \
  --bbox-error-threshold-px 0.0 \
  --depth-rmse-threshold-m 0.0 \
  --onset-error-threshold-frames 0 \
  --offset-error-threshold-frames 0 \
  --out-json outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.normal_ray.verify.json
```

Current evidence from `outputs/pressure_l1_true_grid_smoke/20260623_132929.geom_aligned.normal_ray.verify.json`:

- `precontact_leakage_fraction = 0.0` passes.
- Force-depth Spearman is `1.0`.
- Active mask IoU is `1.0`.
- Centroid error, bbox error, depth RMSE, onset error, and offset error are all `0`.

This proves the trace contract, geometry alignment, pressure calibration path, and verifier can pass strict L1 when the penetration source is geometry-consistent. The remaining failure is therefore specific to the current WarpSDF penetration source relative to this model reference: it detects contact around step 6 while the aligned TacMap normal-ray/deformation reference starts at step 12.

## L1 Online Normal-Ray Backend

The integrated runner can now build the TacMap-grid-to-pressure-taxel alignment once at startup, convert the live TacMap deformation into the pressure taxel grid inside the simulation loop, and optionally use that force map for the displayed pressure strip:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm timeout 240s \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 20 --max_steps 19 \
  --tacmap-ray-mode link_surface --no-local-ui --print-every 5 \
  --enable-normal-ray-pressure --pressure-view-source normal_ray \
  --normal-ray-pressure-max-distance 0.00075 \
  --save-pressure-trace --verify-pressure-trace \
  --pressure-verify-pressure-key normal_ray_pressure_norm \
  --pressure-verify-raw-key normal_ray_pressure_raw_n \
  --pressure-verify-penetration-key normal_ray_penetration_m \
  --pressure-verify-reference-key normal_ray_penetration_m \
  --pressure-verify-reference-iou-threshold 1.0 \
  --pressure-verify-centroid-threshold-px 0.0 \
  --pressure-verify-bbox-threshold-px 0.0 \
  --pressure-verify-depth-rmse-threshold-m 0.0 \
  --pressure-verify-onset-threshold-frames 0 \
  --pressure-verify-offset-threshold-frames 0 \
  --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_online_normal_ray_smoke \
  --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_online_normal_ray_smoke
```

Current evidence from `outputs/pressure_l1_online_normal_ray_smoke/20260623_134127.verify.json`:

- `precontact_leakage_fraction = 0.0` passes.
- Force-depth Spearman is `1.0`.
- Active mask IoU is `1.0`.
- Centroid error, bbox error, depth RMSE, onset error, and offset error are all `0`.
- The pressure trace contains `normal_ray_penetration_m`, `normal_ray_signed_distance_m`, `normal_ray_penetration_velocity_mps`, `normal_ray_pressure_raw_n`, `normal_ray_pressure_norm`, `normal_ray_total_force_n`, and `normal_ray_center_of_pressure_px`.
- The trace metadata records the online alignment arrays: `normal_ray_alignment_source_index`, `normal_ray_alignment_valid_mask`, and `normal_ray_alignment_nn_distance_m`.

This confirms the online pressure path can produce a strict L1-pass force map with the same `(T, sensor, H, W)` trace contract when compared against its own aligned model reference. WarpSDF remains the default backend because it is independent of TacMap at runtime, but `--pressure-view-source normal_ray` is now available as a geometry-consistent reference/debug view.

## Independent Geometry Normal-Ray Source

The pressure source layer now also has `GeometryNormalRayPenetrationSource`. It is independent from TacMap and Isaac runtime sensors: callers provide taxel ray origins/directions plus object geometry in the same local frame, and it ray-casts sphere, axis-aligned box, or triangle mesh geometry to produce the same canonical `PenetrationFrame`.

Covered behavior in `tests/test_pressure_taxel_map.py`:

- Miss rays and `inf` hit distances do not create pre-contact pressure.
- Sphere ray casts create a footprint whose center taxel has the maximum penetration.
- Axis-aligned box and equivalent triangle mesh produce the same penetration.
- `max_distance_m` clips far hits into misses.

This is the pressure-side API used by the online geometry normal-ray backend below. It traces object geometry directly along pressure taxel normals without using TacMap deformation as test-time ground truth.

There is also a CLI wrapper for offline/benchmark traces with explicit geometry:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

out = Path("outputs/pressure_geometry_normal_ray_smoke/geometry_center_traj.npy")
out.parent.mkdir(parents=True, exist_ok=True)
center = np.zeros((19, 3), dtype=np.float32)
center[:, 0] = np.linspace(0.018, 0.007, 19, dtype=np.float32)
center[:, 1] = 0.0
center[:, 2] = 0.0012
np.save(out, center)
print(out)
PY
```

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/apply_pressure_geometry_normal_ray.py \
  outputs/pressure_l1_online_normal_ray_smoke/20260623_134127.npz \
  --geometry sphere \
  --center-npy outputs/pressure_geometry_normal_ray_smoke/geometry_center_traj.npy \
  --radius 0.002 \
  --rest-distance-m 0.01 \
  --max-distance-m 0.02 \
  --stiffness 1 \
  --max-force 1 \
  --out-trace outputs/pressure_geometry_normal_ray_smoke/geometry_sphere_traj.npz \
  --out-json outputs/pressure_geometry_normal_ray_smoke/geometry_sphere_traj.json
```

Verify the generated geometry-normal-ray trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/verify_pressure_trace.py \
  outputs/pressure_geometry_normal_ray_smoke/geometry_sphere_traj.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --out-json outputs/pressure_geometry_normal_ray_smoke/geometry_sphere_traj.verify.json
```

Current evidence from `outputs/pressure_geometry_normal_ray_smoke/geometry_sphere_traj.verify.json`:

- `precontact_leakage_fraction = 0.0` passes.
- Force-depth Spearman is `0.9999999999999999`.
- Onset is step 6 for this explicit sphere trajectory.
- This is not TacMap GT; it is an explicit geometry trace generated from taxel layout, ray directions, and user-supplied geometry parameters.

## L1 Online Geometry Normal-Ray Backend

The integrated runner can now load the presser USD mesh, read the live object pose and live touch-link pose, transform the mesh into the touch-link local frame, and produce `geometry_normal_ray_*` pressure arrays in the same `(T, sensor, H, W)` trace contract. The environment observation includes `press_touch_pose` so this transform does not depend on TacMap deformation.

The default online geometry mode is `inside_exit`: a taxel only becomes active if its ray origin is inside the closed object mesh in both `+normal` and `-normal` ray directions, and penetration is measured to the nearest mesh exit along either direction. The bidirectional inside check avoids accepting one-sided/open-surface ray hits as true closed-volume contact. Using the nearest normal-axis boundary avoids treating the full thickness of a large presser as local indentation depth. It also avoids the failure mode of the simpler baseline ray-distance mode, where nearby but non-intersecting geometry can light up the whole taxel grid before contact. Baseline mode remains available for debugging.

The runner exposes `--geometry-normal-ray-origin-source pressure_taxel|tacmap_aligned` for A/B diagnostics. The default `pressure_taxel` preserves the original pressure layout. `tacmap_aligned` reuses the TacMap link-surface alignment and writes `geometry_normal_ray_origin_points_l_m` plus `geometry_normal_ray_origin_valid_mask` into the trace, but it should not be treated as a fix unless the chosen reference semantics are declared.

Run a headless online mesh trace:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm timeout 240s \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 20 --max_steps 19 \
  --tacmap-ray-mode link_surface --no-local-ui --print-every 5 \
  --enable-geometry-normal-ray-pressure --pressure-view-source geometry_normal_ray \
  --geometry-normal-ray-mode inside_exit --geometry-normal-ray-max-distance 0.03 \
  --pressure-stiffness 1000 --pressure-max-force 10 \
  --save-pressure-trace --verify-pressure-trace \
  --pressure-verify-pressure-key geometry_normal_ray_pressure_norm \
  --pressure-verify-raw-key geometry_normal_ray_pressure_raw_n \
  --pressure-verify-penetration-key geometry_normal_ray_penetration_m \
  --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_geometry_normal_ray_inside_smoke \
  --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_geometry_normal_ray_inside_smoke
```

Historical evidence from `outputs/pressure_l1_geometry_normal_ray_inside_smoke/20260623_140438.verify.json` before the nearest-boundary fix:

- `precontact_leakage_fraction = 0.0` passes.
- Force-depth Spearman is `0.9674208144796379`, above the `0.95` threshold.
- Onset is step 6, with active taxels growing from 8 to 14 and no active taxels before onset.
- The trace metadata records `geometry_normal_ray_mode = inside_exit`, the loaded presser USD path, mesh bounds, `1878` vertices, and `626` triangles.
- The run used `--pressure-stiffness 1000 --pressure-max-force 10` to avoid early saturation while checking monotonicity.

After the nearest-boundary fix, run geometry and TacMap-derived normal-ray together to measure spatial agreement instead of only no-reference leakage/monotonicity:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm timeout 240s \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 20 --max_steps 19 \
  --tacmap-ray-mode link_surface --no-local-ui --print-every 5 \
  --enable-normal-ray-pressure --normal-ray-pressure-max-distance 0.00075 \
  --enable-geometry-normal-ray-pressure --pressure-view-source geometry_normal_ray \
  --geometry-normal-ray-mode inside_exit --geometry-normal-ray-max-distance 0.03 \
  --pressure-stiffness 1000 --pressure-max-force 10 \
  --save-pressure-trace --verify-pressure-trace \
  --pressure-verify-pressure-key geometry_normal_ray_pressure_norm \
  --pressure-verify-raw-key geometry_normal_ray_pressure_raw_n \
  --pressure-verify-penetration-key geometry_normal_ray_penetration_m \
  --pressure-verify-reference-key normal_ray_penetration_m \
  --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_geometry_vs_normal_ray_smoke \
  --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_geometry_vs_normal_ray_smoke
```

Current evidence from `outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.verify.json`:

- Local geometry penetration max is now `0.0044197207 m`, not the earlier near-object-thickness value.
- `precontact_leakage_fraction = 0.0` and force-depth Spearman is `1.0`.
- Geometry onset is step 6 while TacMap-derived normal-ray reference onset is step 12, so strict L1 still fails.
- Centroid and bbox pass (`0.5 px`, `2.0 px`), but active mask IoU and depth RMSE fail (`IoU min = 0.0`, `depth_rmse_m_max = 0.0022731508 m`).

Fit a geometry deadband against the TacMap-derived normal-ray reference:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_l1_deadband.py \
  outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --candidate-count 120 \
  --suggested-arg-name=--geometry-normal-ray-deadband \
  --out-json outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_deadband_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_deadband_fit.npz
```

Current best fit:

- Suggested runner arg: `--geometry-normal-ray-deadband 0.00233985215`.
- Onset aligns to step 12 and offset remains aligned.
- Centroid and bbox stay within threshold (`0.5143 px`, `1.0 px`).
- Strict L1 still fails on IoU and depth scale (`IoU min = 0.6667`, `depth_rmse_m_max = 0.0007534928 m`).

The same deadband was also verified in the live runner at `outputs/pressure_l1_geometry_deadband_live_smoke/20260623_141202.verify.json` with the same conclusion: onset, centroid, bbox, leakage, and monotonicity pass, while IoU and depth RMSE still fail.

The fitter also supports an optional affine depth search:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_l1_deadband.py \
  outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --candidate-count 120 \
  --scale-min 0.4 --scale-max 1.4 --scale-count 31 \
  --suggested-arg-name=--geometry-normal-ray-deadband \
  --out-json outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_affine_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_affine_fit.npz
```

Current affine-fit evidence from `outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_affine_fit.verify.json`:

- Best scale is `0.966666667` and best deadband is `0.0022415542 m`.
- Onset/offset, centroid, bbox, leakage, and force-depth monotonicity pass.
- Strict L1 still fails on active mask IoU and depth RMSE (`IoU min = 0.6667`, `depth_rmse_m_max = 0.0007531821 m`).
- This means a global scalar depth calibration is not enough; the remaining mismatch is spatial/shape-related or reference-frame/model-related.

## L1 Finite-Area Taxel Diagnostic

The online geometry normal-ray backend can optionally sample a small finite support around each taxel center before aggregating penetration. This is meant to model the fact that a real pressure taxel has area, while the original diagnostic uses a single point sample. The default remains `center`, so old runs are reproducible unless the sampling arguments are explicitly passed.

Example live run:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
timeout 240s $ISAACLAB_SH -p \
  $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 \
  --press-steps 20 --max_steps 19 \
  --tacmap-ray-mode link_surface --no-local-ui --print-every 100 \
  --enable-normal-ray-pressure \
  --normal-ray-pressure-max-distance 0.00075 \
  --enable-geometry-normal-ray-pressure \
  --pressure-view-source geometry_normal_ray \
  --geometry-normal-ray-mode inside_exit \
  --geometry-normal-ray-max-distance 0.03 \
  --geometry-normal-ray-taxel-samples cross_5 \
  --geometry-normal-ray-sample-spacing 0.00025 \
  --geometry-normal-ray-sample-aggregation max \
  --pressure-stiffness 1000 --pressure-max-force 10 \
  --save-pressure-trace --verify-pressure-trace \
  --pressure-verify-pressure-key geometry_normal_ray_pressure_norm \
  --pressure-verify-raw-key geometry_normal_ray_pressure_raw_n \
  --pressure-verify-penetration-key geometry_normal_ray_penetration_m \
  --pressure-verify-reference-key normal_ray_penetration_m \
  --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke \
  --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke
```

Affine fit for that trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_l1_deadband.py \
  outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --candidate-count 120 \
  --scale-min 0.2 --scale-max 1.2 --scale-count 31 \
  --suggested-arg-name=--geometry-normal-ray-deadband \
  --out-json outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.geometry_sampled_affine_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.geometry_sampled_affine_fit.npz
```

Current finite-area evidence from `outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.geometry_sampled_affine_fit.verify.json`:

- Best scale is `0.333333333` and best deadband is `0.00078852524 m`.
- Onset/offset, centroid, bbox, leakage, and force-depth monotonicity pass.
- The boundary footprint improves versus the point-sampled affine fit (`IoU min = 0.7143` instead of `0.6667`).
- Strict L1 still fails on active mask IoU and depth RMSE (`depth_rmse_m_max = 0.0011302545 m`), so finite taxel support is not an acceptance fix.
- A larger `cross_5` spacing of `0.0005 m` is too aggressive for this case: it triggers contact around step 6 and drops the fitted IoU minimum to `0.5`.

Conclusion: finite-area support is a useful diagnostic and future pressure-pad modeling option, but it should not be promoted to accepted reference alignment by itself. The next correction needs a better geometry depth law, a physically justified onset/contact gate, or an L2 oracle.

The geometry normal-ray backend also supports an explicit support gate:

- `--geometry-normal-ray-sample-min-support-fraction <0..1>` leaves default behavior unchanged at `0`.
- It applies after `max`, `mean`, or `positive_mean` aggregation and suppresses a taxel unless enough finite-area samples report contact.
- This is meant to test contact-support hypotheses without changing the default pressure source or older trace reproducibility.

Example support-gated run:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm \
timeout 240s $ISAACLAB_SH -p \
  $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 \
  --press-steps 20 --max_steps 19 \
  --tacmap-ray-mode link_surface --no-local-ui --print-every 100 \
  --enable-normal-ray-pressure \
  --normal-ray-pressure-max-distance 0.00075 \
  --enable-geometry-normal-ray-pressure \
  --pressure-view-source geometry_normal_ray \
  --geometry-normal-ray-mode inside_exit \
  --geometry-normal-ray-max-distance 0.03 \
  --geometry-normal-ray-taxel-samples cross_5 \
  --geometry-normal-ray-sample-spacing 0.00025 \
  --geometry-normal-ray-sample-aggregation mean \
  --geometry-normal-ray-sample-min-support-fraction 0.4 \
  --pressure-stiffness 1000 --pressure-max-force 10 \
  --save-pressure-trace --verify-pressure-trace \
  --pressure-verify-pressure-key geometry_normal_ray_pressure_norm \
  --pressure-verify-raw-key geometry_normal_ray_pressure_raw_n \
  --pressure-verify-penetration-key geometry_normal_ray_penetration_m \
  --pressure-verify-reference-key normal_ray_penetration_m \
  --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_geometry_mean_support04_vs_normal_ray_smoke \
  --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_geometry_mean_support04_vs_normal_ray_smoke
```

Current support-gated evidence from `outputs/pressure_l1_geometry_mean_support04_vs_normal_ray_smoke/20260623_150835.geometry_mean_support04_affine_fit.json` and `.diagnostics.json`:

- Raw trace still starts early at step `6`; affine fit chooses scale `0.8` and deadband `0.0017456038 m`, aligning onset to step `12`.
- Strict L1 still fails on IoU and depth (`IoU min = 0.6667`, `depth_rmse_m_max = 0.0008690176 m`).
- Per-frame diagnostics after fit show pure `under_prediction`: `false_negative_taxels_total = 16`, `false_positive_taxels_total = 0`.
- This is safer than naive finite-area `max` in the sense that it avoids post-fit false positives, but it does not improve the boundary footprint over point sampling and has worse depth RMSE than the point-sampled affine fit.

Current online trace-contract evidence from `outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.npz`:

- The live integrated runner writes `geometry_normal_ray_sample_support_fraction`, `geometry_normal_ray_sample_active_count`, `geometry_normal_ray_sample_mean_penetration_m`, `geometry_normal_ray_sample_positive_mean_penetration_m`, and `geometry_normal_ray_sample_max_penetration_m`.
- All five arrays have shape `(19, 1, 30, 30)`, matching the geometry normal-ray pressure map.
- `geometry_normal_ray_sample_support_fraction` ranges from `0.0` to `1.0`, and `geometry_normal_ray_sample_active_count` ranges from `0.0` to `5.0` for the `cross_5` sampler.
- Strict L1 against `normal_ray_penetration_m` still fails before affine fitting: onset is step `6` versus reference step `12`, `false_positive_taxels_total = 106`, `false_negative_taxels_total = 6`, and `depth_rmse_m_max = 0.0020190689 m`.
- This confirms that finite-area support diagnostics are now observable in raw traces, but the live model still needs an onset/contact gate and area-weighted depth law before it can be used as an accepted dense reference.

Current raw per-sample evidence from `outputs/pressure_l1_geometry_sample_raw_smoke/20260623_163232.npz`:

- The live integrated runner also writes `geometry_normal_ray_sample_penetrations_m`, `geometry_normal_ray_sample_offsets_l`, and `geometry_normal_ray_sample_points_l_m`.
- With `cross_5`, those arrays have shape `(19, 1, 30, 30, 5)`, `(19, 1, 30, 30, 5, 3)`, and `(19, 1, 30, 30, 5, 3)`.
- Use the raw-sample boundary diagnostic when strict L1 fails:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/diagnose_pressure_sample_boundary.py \
  outputs/pressure_l1_geometry_sample_raw_smoke/20260623_163232.npz \
  --out-json outputs/pressure_l1_geometry_sample_raw_smoke/20260623_163232.sample_boundary_diagnostics.json \
  --out-csv outputs/pressure_l1_geometry_sample_raw_smoke/20260623_163232.sample_boundary_diagnostics.csv
```

Observed diagnostic summary:

- Geometry-normal-ray pressure onset is step `6`; the normal-ray model reference onset is step `12`.
- False positives dominate: `106` FP taxels versus `6` FN taxels.
- FP taxels have mean raw sample support about `0.83`; true positives have mean support about `0.98`.
- FN taxels have zero geometry sample support but nonzero reference depth, with mean reference depth about `0.000974 m`.

This means the current dense L1 failure is not just a low-support boundary artifact. The mismatch points to onset/reference-geometry semantics or object/taxel geometry alignment, so further tuning should not be accepted until the reference choice and geometry source agree on first contact.

## L1 Origin-Source Diagnostic

The latest A/B check shows that ray-origin semantics are a first-order issue, not a cosmetic frame detail. This is now reported automatically by `pressure_trace_report()` under `reference.origin_alignment` whenever the trace contains pressure taxel points, TacMap grid points, and `normal_ray_alignment_*` arrays.

Use the existing reference diagnostic script for a short report:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/diagnose_pressure_reference.py \
  outputs/pressure_l1_geometry_welded_usd_smoke/20260623_193846.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --top-k 2
```

On the raw-sample trace `outputs/pressure_l1_geometry_sample_raw_smoke/20260623_163232.npz`, pressure taxel centers and the aligned TacMap link-surface origins differ mostly along the tactile normal:

- Normal offset magnitude is about `0.01006637 m` for valid taxels. In the executable report the signed direction is `tacmap_grid_points_l_m - pressure_taxel_points_l_m`, so the mean `normal_delta_m` is about `-0.01006637 m`.
- Tangential mismatch is small by comparison, around `0.0000958 m` mean.
- This explains why a pressure-taxel-origin geometry trace can begin at step `6` while the TacMap/normal-ray model reference begins at step `12`.

Two surface-origin smoke runs then exposed the tradeoff:

- With `--geometry-normal-ray-origin-source tacmap_aligned --geometry-normal-ray-mode inside_exit`, the bidirectional closed-volume test produced zero active geometry taxels on the current `cylinder_D4` mesh while the TacMap/normal-ray reference became active at step `12`. This suggests the TacMap-aligned origin semantics are not yet a reliable closed-mesh inside reference for this presser.
- With `--geometry-normal-ray-origin-source tacmap_aligned --geometry-normal-ray-mode baseline`, the trace became a proximity field: `336` taxels active from step `1`, versus only `14` final active taxels in the normal-ray reference. This is not an accepted contact-footprint reference.
- With the default pressure-taxel origin and `inside_exit`, pre-contact leakage remains `0.0`, but geometry onset is still step `6` while the normal-ray reference onset is step `12`; strict L1 fails on IoU and depth RMSE.

Conclusion: `tacmap_aligned` origin support is useful for diagnosing frame/origin mismatch, and the bidirectional inside check prevents false closed-volume hits. It does not yet make online geometry normal-ray an accepted dense L1 model reference for the current presser mesh. The accepted dense L1 path remains TacMap/normal-ray model reference plus explicit verifier thresholds until mesh topology and ray-origin semantics are solved.

## Mesh Topology Diagnostic

`GeometryNormalRayPenetrationSource` in `inside_exit` mode assumes the presser mesh behaves like a closed volume. That assumption is now auditable through `triangle_mesh_topology_diagnostics()` and through the integrated runner metadata. USD-loaded geometry summaries include:

- `topology.is_edge_watertight`
- `topology.is_orientable_watertight`
- `topology.boundary_edge_count`
- `topology.nonmanifold_edge_count`
- `topology.degenerate_triangle_count`
- `topology.connected_component_count`
- `topology.euler_characteristic`

The integrated runner also prints the key topology fields when `--enable-geometry-normal-ray-pressure` is used. Offline mesh traces created by `scripts/force_map/apply_pressure_geometry_normal_ray.py --geometry mesh` include the same topology report in their JSON summary.

Use this before promoting any geometry-normal-ray run to accepted L1 reference. A mesh with boundary edges, nonmanifold edges, or degenerate triangles is not a reliable closed-volume reference for bidirectional `inside_exit`, even if ray hits can still be computed.

Current integrated smoke evidence:

```bash
CONDA_PREFIX=$ISAACLAB_CONDA_PREFIX TERM=xterm timeout 300s \
$ISAACLAB_SH -p $REPO_ROOT/integrate/run_integrated_tactile.py \
  --headless --mode press --finger middle --presser cylinder_D4 \
  --press-start-offset 0.025 --press-end-offset 0.018 --press-steps 20 --max_steps 19 \
  --tacmap-ray-mode link_surface --no-local-ui --print-every 5 \
  --enable-normal-ray-pressure --normal-ray-pressure-max-distance 0.00075 \
  --enable-geometry-normal-ray-pressure --pressure-view-source geometry_normal_ray \
  --geometry-normal-ray-mode inside_exit --geometry-normal-ray-max-distance 0.03 \
  --pressure-stiffness 1000 --pressure-max-force 10 \
  --save-pressure-trace --verify-pressure-trace \
  --pressure-verify-pressure-key geometry_normal_ray_pressure_norm \
  --pressure-verify-raw-key geometry_normal_ray_pressure_raw_n \
  --pressure-verify-penetration-key geometry_normal_ray_penetration_m \
  --pressure-verify-reference-key normal_ray_penetration_m \
  --pressure-trace-dir $REPO_ROOT/outputs/pressure_l1_geometry_welded_usd_smoke \
  --pressure-verify-out $REPO_ROOT/outputs/pressure_l1_geometry_welded_usd_smoke
```

Observed on `outputs/pressure_l1_geometry_welded_usd_smoke/20260623_193846.npz`:

- `pressure_trace_v1` contract passes with no errors or warnings.
- `geometry_normal_ray_pressure_norm` and `geometry_normal_ray_penetration_m` both have shape `(19, 1, 30, 30)`.
- The USD-loaded `cylinder_D4` stores split triangle vertices (`1878` input vertices for `626` triangles). The geometry loader now welds duplicate positions to `307` vertices before topology diagnostics and ray checks.
- After welding, the USD mesh reports `is_edge_watertight = true`, `is_orientable_watertight = true`, `boundary_edge_count = 0`, `connected_component_count = 1`, `triangle_count = 626`, and `vertex_count = 307`.
- Strict L1 comparison against `normal_ray_penetration_m` still fails: geometry onset is step `6`, the normal-ray model reference onset is step `12`, `active_mask_iou_min = 0.0`, and `depth_rmse_m_max = 0.0022731509 m`.

Conclusion: the old non-watertight diagnosis was a loader artifact from unwelded USD vertices. The remaining blocker is not topology; it is that the closed-volume geometry-normal-ray onset/depth semantics still do not match the TacMap-derived normal-ray model reference closely enough to accept this backend as L1.

Additional watertight NPY smoke evidence:

- `outputs/pressure_l1_geometry_watertight_npy_smoke/20260623_193015.npz` uses a closed cylinder generated from the current USD bounds. Topology is watertight, but the trace fails with `geometry_normal_ray_blocker = watertight_mesh_precontact_inside_volume`: geometry onset is step `0`, normal-ray model-reference onset is step `12`, and many active geometry taxels lie outside the aligned contact region.
- `outputs/pressure_l1_geometry_d4_watertight_npy_smoke/20260623_193104.npz` uses a true D4-scale closed cylinder. Topology is watertight, but the trace fails with `geometry_normal_ray_blocker = watertight_mesh_no_overlap_with_model_reference`: geometry never becomes active while the normal-ray model reference onsets at step `12`.

Conclusion update: watertight topology is only a prerequisite. The online geometry-normal-ray backend also needs a presser mesh whose object-local geometry, scale, and pose semantics match the actual Isaac contact object before it can be promoted from diagnostic reference to accepted L1 model reference.


Fit a finite-area support/depth law against the same trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_sample_support_law.py \
  outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --support-key geometry_normal_ray_sample_support_fraction \
  --depth-source-key geometry_normal_ray_penetration_m \
  --depth-source-key geometry_normal_ray_sample_mean_penetration_m \
  --depth-source-key geometry_normal_ray_sample_positive_mean_penetration_m \
  --depth-source-key geometry_normal_ray_sample_max_penetration_m \
  --support-thresholds '0,0.2,0.4,0.6,0.8,1.0' \
  --mask-deadbands-m '0,0.0005,0.001,0.0015,0.002,0.0025' \
  --depth-deadbands-m '0,0.0005,0.001,0.0015,0.002,0.0025' \
  --scale-candidates '0.2,0.4,0.6,0.8,1.0,1.2' \
  --support-powers '0,0.5,1.0' \
  --active-floors-m '0,1e-6' \
  --out-json outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.sample_support_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.sample_support_fit.npz
```

Current sample-support fit evidence:

- Best candidate uses `depth_source_key = geometry_normal_ray_penetration_m`, `mask_deadband_m = 0.002`, `depth_deadband_m = 0.0025`, `scale = 1.0`, `support_power = 0.0`, and `support_threshold = 0.0`.
- It fixes onset and offset against the L1 model reference: both pressure and reference onset are step `12`, and offset is step `18`.
- It passes leakage, force-depth monotonicity, centroid, bbox, onset, and offset checks.
- It still fails strict dense acceptance: `active_mask_iou_min = 0.3333` and `depth_rmse_m_max = 0.0008800259 m`.
- Diagnostics on `20260623_152828.sample_support_fit.diagnostics.json` show the failure has changed from early false positives to residual boundary/depth mismatch: `false_positive_taxels_total = 8`, `false_negative_taxels_total = 14`, with worst late-frame failures dominated by under-prediction.

Conclusion: stored support statistics are enough to make onset/contact gating reproducible, but this simple scalar support/depth law still does not solve dense boundary shape or depth. The next candidate should be spatial rather than scalar: local footprint dilation/erosion in taxel coordinates, per-edge area fraction modeling, or a higher-fidelity L2 oracle to define the target for partially covered taxels.

Fit a spatial footprint law against the same trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_spatial_footprint_law.py \
  outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --support-key geometry_normal_ray_sample_support_fraction \
  --depth-source-key geometry_normal_ray_penetration_m \
  --depth-source-key geometry_normal_ray_sample_mean_penetration_m \
  --depth-source-key geometry_normal_ray_sample_max_penetration_m \
  --support-thresholds '0,0.2,0.4' \
  --mask-deadbands-m '0.0015,0.002,0.0025' \
  --depth-deadbands-m '0,0.0005,0.001,0.0015,0.002' \
  --scale-candidates '0.4,0.6,0.8,1.0' \
  --support-powers '0' \
  --active-floors-m '0' \
  --transition-depths-m '0,0.001,0.002,0.003' \
  --early-ops 'none,erode' \
  --late-ops 'none,dilate' \
  --early-iterations '0,1' \
  --late-iterations '0,1' \
  --out-json outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.spatial_footprint_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.spatial_footprint_fit.npz
```

Current spatial-footprint fit evidence:

- The fitter supports taxel-grid erosion/dilation split by a peak-depth transition, and writes `pressure_norm_spatial_footprint_fit`, `pressure_raw_spatial_footprint_fit_n`, and `penetration_spatial_footprint_fit_m`.
- The best real-trace candidate still does not use meaningful morphology (`transition_depth_m = 0`, effective iterations `0`), which is itself evidence that simple global erosion/dilation is not a good fit for this mismatch.
- It still passes leakage, force-depth monotonicity, centroid, bbox, onset, and offset.
- It still fails strict dense acceptance: `active_mask_iou_min = 0.3333` and `depth_rmse_m_max = 0.0009866225 m`.
- Diagnostics on `20260623_152828.spatial_footprint_fit.diagnostics.json` show the same structural pattern as scalar sample-support fitting: step `12` is over-predicted (`false_positive_taxels = 4`), while late steps `16..18` are under-predicted (`false_negative_taxels = 4` per frame).

Conclusion: local binary morphology is a useful diagnostic control, but not the missing physical model.

Fit sampled area-fraction/depth integration against the same trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_area_fraction_law.py \
  outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --area-modes 'mean,support_positive,support_max,support_center,positive,max,blend_mean_support_max,blend_mean_support_center' \
  --blend-weights '0,0.25,0.5,0.75,1.0' \
  --support-thresholds '0,0.2,0.4,0.6,0.8,1.0' \
  --mask-deadbands-m '0,0.0005,0.001,0.0015,0.002,0.0025' \
  --depth-deadbands-m '0,0.0005,0.001,0.0015,0.002,0.0025' \
  --scale-candidates '0.2,0.4,0.6,0.8,1.0,1.2' \
  --active-floors-m '0' \
  --out-json outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.area_fraction_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_sample_stats_smoke/20260623_152828.area_fraction_fit.npz
```

Current area-fraction fit evidence:

- The best candidate uses `area_mode = blend_mean_support_max`, `blend_weight = 0.25`, `scale = 1.2`, `mask_deadband_m = 0`, and `depth_deadband_m = 0.0025`.
- It improves mean footprint metrics compared with scalar sample support: `active_mask_iou_mean = 0.7537` and `bbox_error_px_mean = 0.5714`.
- It still passes leakage, force-depth monotonicity, centroid, bbox, onset, and offset checks.
- It still fails strict dense acceptance: `active_mask_iou_min = 0.3333` and `depth_rmse_m_max = 0.0009193945 m`.
- Diagnostics on `20260623_152828.area_fraction_fit.diagnostics.json` show the same hard mismatch remains: step `12` has four high-confidence false-positive taxels, while late step `18` still misses four reference taxels.

Conclusion: the sampled area-fraction fields are useful and improve average alignment, but the old aggregate-only `cross_5` trace does not contain enough information to perfectly separate early false positives from late missing edge taxels. The trace contract now includes raw per-sample positions and penetrations so the next credible step is to use those richer fields for boundary-shape analysis, or to compare against a small L2 hydroelastic/FEM/UIPC oracle that defines what pressure/depth should be assigned to partially covered taxels.

## L1 Per-Frame Failure Diagnostics

Use the per-frame diagnostic tool to separate onset, footprint, and depth errors instead of reading only aggregate verifier metrics:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/diagnose_pressure_reference.py \
  outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_affine_fit.npz \
  --pressure-key pressure_norm_fit \
  --penetration-key penetration_fit_m \
  --reference-key normal_ray_penetration_m \
  --top-k 4 \
  --out-json outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_affine_fit.diagnostics.json \
  --out-csv outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_affine_fit.diagnostics.csv
```

Current point-sampled affine diagnostic:

- Onset and offset are aligned at steps `12` and `18`.
- `false_positive_taxels_total = 0`, `false_negative_taxels_total = 16`.
- The remaining footprint failures are pure `under_prediction` on steps `15..18`; worst IoU is `0.6667`.
- Depth bias is negative (`depth_bias_m_mean = -0.0001398917`), and worst depth RMSE is step `18` (`0.0007531821 m`).

Run the same diagnostic on the finite-area `cross_5 0.00025 m` affine trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/diagnose_pressure_reference.py \
  outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.geometry_sampled_affine_fit.npz \
  --pressure-key pressure_norm_fit \
  --penetration-key penetration_fit_m \
  --reference-key normal_ray_penetration_m \
  --top-k 4 \
  --out-json outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.geometry_sampled_affine_fit.diagnostics.json \
  --out-csv outputs/pressure_l1_geometry_sampled025_vs_normal_ray_smoke/20260623_145106.geometry_sampled_affine_fit.diagnostics.csv
```

Current finite-area diagnostic:

- Onset and offset are still aligned.
- False negatives drop from `16` to `8`, but false positives rise from `0` to `6`.
- Worst IoU improves from `0.6667` to `0.7143`, but depth gets worse (`depth_rmse_m_max = 0.0011302545 m`, `depth_bias_m_mean = -0.0004397658`).

Conclusion: the next L1 implementation should not be a single global deadband or scale. The evidence points to an edge-support/depth-law problem: center sampling is too conservative at the contact boundary, while naive finite-area max support changes the fitted depth scale and creates early over-prediction. A better candidate is to separate contact support estimation from force/depth magnitude, or to introduce area-weighted taxel support with an independently calibrated depth law.

## L1 Mask/Depth Split Diagnostic

The offline split fitter tests whether active support and reported depth can be calibrated separately on an existing trace:

```bash
PYTHONPATH=$REPO_ROOT/source/BrainCo_DexHand \
python3 scripts/force_map/fit_pressure_mask_depth_split.py \
  outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.npz \
  --pressure-key geometry_normal_ray_pressure_norm \
  --raw-pressure-key geometry_normal_ray_pressure_raw_n \
  --penetration-key geometry_normal_ray_penetration_m \
  --reference-key normal_ray_penetration_m \
  --mask-deadbands-m '0.000315,0.0008,0.0012,0.0016,0.0020,0.00224,0.00234' \
  --depth-deadbands-m '0.0020,0.0022,0.00224,0.00234,0.0025' \
  --scale-candidates '0.8,0.966666666667,1.1' \
  --active-floors-m '0,1e-9,1e-5,5e-5,1e-4' \
  --out-json outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_mask_depth_split_fit.json \
  --fit-trace-out outputs/pressure_l1_geometry_vs_normal_ray_smoke/20260623_141027.geometry_mask_depth_split_fit.npz
```

Current split-fit evidence:

- Best fit chooses `mask_deadband_m = 0.00234`, `depth_deadband_m = 0.00224`, `scale = 0.966666666667`, `active_floor_m = 0.0001`.
- It still fails strict L1 (`IoU min = 0.6667`, `depth_rmse_m_max = 0.0007531793 m`).
- Per-frame diagnostics remain pure `under_prediction`: `false_negative_taxels_total = 16`, `false_positive_taxels_total = 0`.
- The fitted mask threshold stays high instead of preserving weaker edge taxels, so splitting scalar mask/depth thresholds alone does not recover the missing boundary support.

Conclusion: the remaining miss is unlikely to be fixed by scalar thresholds alone on the point-sampled geometry trace. The raw trace now stores per-taxel finite-area support fractions and sampled depth summaries, so the next meaningful candidate should use those fields for an onset/contact gate, local footprint/area model, or area-weighted depth law. An L2 oracle is still needed if we want to justify how much physical pressure to assign to partially covered taxels beyond model-reference matching.

## Next Implementation Target

Do not mark L1 complete until one of these is true and verified:

- WarpSDF taxels are projected into the same link-surface frame as the model reference before comparison.
- Or an independently justified contact/onset gate makes the WarpSDF pressure onset and depth scale pass the same strict L1 checks without using TacMap as test-time ground truth.
- Or the online `GeometryNormalRayPenetrationSource` is validated as an accepted L1 geometry model reference: watertight mesh behavior is checked, ray origins are declared and aligned, bidirectional inside tests do not erase valid contact, and per-taxel area/depth integration passes the same spatial/depth checks against analytic and model-reference traces.
- Or a small L2 offline oracle is added from hydroelastic/FEM and the SDF-like backend is shown to match it within declared tolerances on representative poses.
- Real vision-based tactile datasets are not a substitute for these gates; they should enter later as calibration/validation evidence for total force, center of pressure, contact footprint, and sim-to-real robustness.

Acceptance should then use the existing verifier thresholds: IoU, centroid, bbox, depth RMSE, onset/offset, pre-contact leakage, force-depth monotonicity, and the alignment-health checks above.
