# Flexiv D-Peg Custom

`BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0` copies the D-shaped peg/socket
interaction while using the table, plane, lighting and sensor conventions from
`RotateBulbCustom`. The peg and socket assets remain task-owned under
`assets/d_peg_insertion`; only the robot is replaced by the local Flexiv Rizon4 +
Revo3 right-hand asset `assets/rotate_bulb_custom/robot/usd/rizon4_training.usda`.

The configuration uses a Flexiv-specific 28-joint pregrasp seed, maps the legacy
calibration names only for backward-compatible input files, and uses a task-owned
residual action contract and tactile observation pipeline. Its PPO model and
hyperparameters are independent from both the original D-peg and Rotate-Bulb runs.

`assets/d_peg_insertion/pregrasp_flexiv.json` contains the editor pose from
`20260913_142053_9886dc87`: all 28 measured joint positions and the peg/socket
poses remain unchanged. Its drive targets now carry a bounded, finger-only
preload (largest on the index finger, which had the largest contact gap) so the
position controller continues closing after contact settles. The unmodified
editor candidate is archived under
`assets/d_peg_insertion/flexiv_grasp_reference/20260913_142053_9886dc87/`.

This remains a `preload_candidate`, not a physically calibrated grasp. The
editor optimizer converged but its pose acceptance failed (18.43 mm maximum
contact-anchor error, 3.80 mm maximum penetration). The preload is a small,
reproducible starting point for Isaac gravity-hold tuning; run the Flexiv
zero-action check and adjust targets from measured fingertip forces before a
long run. No training was launched.

The two-environment Isaac check passed all 33 startup/reset comparisons with the
original export, including the full 669-dimensional policy input. Its two-second
zero-action grasp check failed: maximum peg drift was 6.59/11.56 mm and rotation
14.23/7.26 degrees, with insufficient contact on some fingers. See the archived
`full_runtime.json` and `README_CN.md`; the pose remains explicitly unvalidated.

To import a subsequent export with geometry, joint, frame and provenance checks:

```bash
python assets/d_peg_insertion/tools/import_flexiv_pregrasp.py --export /path/to/simulation_state.json
```
