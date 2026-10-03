# Randomized bulb unscrewing

Task: `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0`.

The arm base and seven arm joints retain the current Dexonomy arm pose. All 21
hand joint positions and drive targets start at zero. The policy must approach,
grasp, unscrew three turns, and transport the released bulb to its goal.

The fixture and seated bulb move together within world X/Y ±0.10 m. Goals are
relative to that fixture within X/Y ±0.15 m and Z +0.28–0.42 m. Reset samples
uniformly from `assets/rotate_bulb_randomized/workspace_bank_high_z_v2.json`; this is a
finite, screened distribution, not continuous uniform sampling of the whole
box. Its source hashes and scene poses are checked before use. Rejected
candidate counts, joint solutions, and the exact validation scope are in the
bank. No IK solution drives or resets the robot during training.
The raised bank contains 3456 pairs at the same 432 support positions as the
original bank, which remains available for replay of old runs. Height is
resampled with each pair; this is not independent continuous XYZ sampling.

`mdp.py` owns reset, goal sampling and grasp state. Opposing contact above 0.5 N
for 0.12 s latches the grasp pose; loss for 0.2 s allows approach again. The arm
action-cost multipliers are 1/5/1 for approach/engaged/transport. The engaged arm
drift cost uses the latched grasp pose. The original strong distance cost starts
only after a real grasp. The one-point grasp-budget state remains in the
observation contract; its separate pre-release payout is disabled in reward
profile 3, described below.
Existing three-turn release, 18 mm screw travel, 3 cm/1 s success and failure
criteria are preserved. Observation width is 687, including 14 explicit phase,
reference-offset, timer and budget values. Checkpoints carry a separate task ID
and workspace hash; original Custom or Assembly checkpoints are not resumes.

Before release, `RandomizedPalmAboveSupport` rewards the actual `palm` body
origin being over the fixture in world XY and 0.10–0.18 m above its root in
world Z. Alignment is flat within 2 cm horizontally, then decays with a 6 cm
scale; height outside the band decays with a 4 cm scale. Approach receives
20% of the holding rate, rising with opposing finger contact to 0.5 points/s.
Holding can earn at most 2 points per episode. New unscrewing progress can earn
another 2 across three turns; reversing cannot earn that progress again.
Failed or released states earn no new credit, so transport is exempt. This is
soft reward guidance, not a changed success condition. Task contract version 2
records these parameters and the new bank; old version 1 is not a direct resume.

The 2026-09-28 local timing profile uses 60 Hz physics, two physics steps per
action, 30 Hz control and pressure, and 12 s episodes (360 control steps).
It retains the fresh pose-only pressure provider and TGS 24/4. The user chose
fresh training on GPU4 because the action contract's decimation changes;
GPU6's existing frozen run retains 240 Hz physics, decimation 8 and 20 s episodes.
Train and PLAY use the same scene and sampling distribution. Assets, sensors
and guided screw implementation are shared without changing existing tasks.

Reward profile 3 (`contact_transport_reward_version=3`, 2026-09-28) separates
the auxiliary contact reward by physical release state. `phase_rewards.py`
supplies `ReleasedContactGuidance`: zero before detachment, then up to 1 point/s
and 4 points per episode. Bulb-filtered thumb and strongest other fingertip
forces saturate at 0.5 N. Single-sided contact pays at most 0.25/s; opposing
contact pays the full rate. It does not require clearance or a hold timer, so
early post-release contact can receive guidance. Dropped/out-of-bounds/lost-grasp
failures earn zero. Partial resets clear only the corresponding budget; repeated
reads within one control step cannot spend twice. The TensorBoard reward name
remains `any_finger_contact`, with new `Metrics/phase_contact/*` diagnostics.

`prerelease_arm_motion` penalizes arm velocity even before an opposing grasp
is latched, retaining the 0.15 rad/s deadband and 0.6 rad/s saturation scale.
Position drift is charged only relative to a real latched grasp, leaving the
initial approach free to change pose. Weight is -0.3 (previously -0.1), and this
term becomes zero after release so transport can move the arm. Action scales,
the existing action costs and all phase observations are unchanged.

The separate `randomized_grasp` reward weight is now zero. Physical turn
progress still requires genuine fingertip contact; palm guidance is retained.
Existing strict release-grasp/stability bonuses, transport weight 40, failure
penalty -6 and final task criteria are unchanged. Reward parameters are recorded
in env.yaml/version metadata separately from the unchanged action/observation
contract, allowing the authorized profile-2 checkpoint continuation. Replay old
experiments from their own frozen workspace. CPU invariants additionally live
in `tests/test_rotate_bulb_phase_rewards.py`.

CPU tests: `tests/test_rotate_bulb_randomized.py`. Native validation:
`scripts/diagnostics/validate_bulb_palm_high_z.py --out <new-output-directory> --headless --num_envs 8`.
The native probe checks 16 resets, reward gating, contracts and 40 physics steps;
it is not an independent task-success evaluation.
