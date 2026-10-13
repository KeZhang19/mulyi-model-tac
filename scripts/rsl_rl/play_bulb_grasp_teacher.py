#!/usr/bin/env python3
"""Replay the bulb teacher; optionally record bounded headless video with metrics."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/BrainCo_DexHand'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=None)
parser.add_argument('--lstm', action='store_true', help='Replay a single-frame LSTM teacher; requires an explicit LSTM --checkpoint.')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=0, help='0: run until the viewer is closed.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--report', type=Path, default=None)
parser.add_argument('--real-time', action='store_true')
parser.add_argument('--video', type=Path, default=None, help='New MP4 path; requires one environment and positive --steps.')
parser.add_argument('--video-resolution', type=int, nargs=2, default=(1280, 960), metavar=('WIDTH', 'HEIGHT'))
parser.add_argument('--camera-eye', type=float, nargs=3, default=(-1.15, -1.20, 1.50))
parser.add_argument('--camera-lookat', type=float, nargs=3, default=(-0.22, 0.08, 1.00))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.lstm and args.checkpoint is None:
    parser.error('--lstm requires --checkpoint pointing to a model trained with --lstm')
if args.checkpoint is None:
    args.checkpoint = ROOT / 'checkpoints/bulb_grasp_teacher_20261004/model_22200.pt'
if args.lstm:
    sys.path.insert(0, str(ROOT / 'runtime/bulb_grasp_teacher'))
if args.video is not None:
    if args.num_envs != 1 or args.steps <= 0:
        parser.error('--video requires --num_envs 1 and positive --steps')
    if args.video.exists():
        parser.error(f'Refusing to overwrite video: {args.video}')
    if min(args.video_resolution) < 64 or any(v % 2 for v in args.video_resolution):
        parser.error('--video-resolution dimensions must be even and at least 64')
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
import BrainCo_DexHand  # registers this checkout's tasks
from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.policy import BulbGraspTeacherActorCritic

TASK = 'BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-Play-v0'
report = {'task': TASK, 'checkpoint': str(args.checkpoint.resolve()), 'passed': False}
env = None
video_writer = None
video_trace = []
try:
    cfg = load_cfg_from_registry(TASK, 'env_cfg_entry_point')
    agent_cfg = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
    if args.lstm:
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.policy import (
            BulbGraspTeacherLSTM, configure_bulb_teacher_lstm,
        )
        configure_bulb_teacher_lstm(cfg, agent_cfg)
    cfg.scene.num_envs = args.num_envs
    report['curriculum_version'] = getattr(cfg, 'bulb_grasp_curriculum_version', 0)
    if report['curriculum_version'] >= 9:
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.target_contract import validate_target_contract
        validate_target_contract(args.checkpoint, report['curriculum_version'])
    if report['curriculum_version'] >= 5:
        report['stage_time_limits_s'] = [cfg.commands.object_pose.stage12_duration_s,
                                        cfg.commands.object_pose.stage3_duration_s]
        report['screw_target_turns'] = cfg.commands.object_pose.screw_target_turns
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.video is not None:
        cfg.viewer.eye = tuple(args.camera_eye)
        cfg.viewer.lookat = tuple(args.camera_lookat)
        cfg.viewer.origin_type = 'env'
        cfg.viewer.env_index = 0
        cfg.viewer.resolution = tuple(args.video_resolution)
        cfg.commands.object_pose.debug_vis = True
    # Restore the teacher's training-time gravity before the first scene reset.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    iteration = int(checkpoint['iter'])
    gravity_state = checkpoint['v2_gravity_curriculum_state']
    trigger = gravity_state['trigger_iteration']
    gravity_cfg = cfg.v2_gravity_curriculum
    progress = 0.0 if trigger is None else min(max((iteration - trigger) / gravity_cfg.ramp_iterations, 0.0), 1.0)
    gravity = gravity_cfg.initial_gravity_z + progress * (gravity_cfg.target_gravity_z - gravity_cfg.initial_gravity_z)
    cfg.sim.gravity = (0.0, 0.0, gravity)
    cfg.events.variable_gravity.params['initial_gravity_z'] = gravity
    # The source run's ADR state is not in the checkpoint. Keep its saved initial
    # difficulty (0.7317); this is a compatibility smoke, not a matched evaluation.
    env = gym.make(TASK, cfg=cfg, render_mode='rgb_array' if args.video else None)
    base = env.unwrapped
    base._rsl_rl_training_iteration = iteration
    base._v2_gravity_curriculum_trigger_iteration = trigger
    base._v2_gravity_curriculum_gravity_z = gravity
    command = base.command_manager.get_term('object_pose')
    if getattr(command, 'global_curriculum', None) is not None:
        state = checkpoint.get('bulb_global_stage_curriculum')
        if state is None:
            raise ValueError('Checkpoint has no global reward-stage state; replay its original task version.')
        command.global_curriculum.load_state_dict(state)
        report['global_reward_stage'] = command.global_curriculum.stage
    if 'training_iteration' in base.event_manager.available_modes:
        base.event_manager.apply(mode='training_iteration')
    obs, _ = env.reset()
    history = cfg.observations.policy.history_length
    assert obs['task_state'].shape == (args.num_envs, 15 * history)
    expected = {name: width * history for name, width in {'policy': 39, 'proprio': 149, 'perception': 192}.items()}
    for group, width in expected.items():
        assert obs[group].shape == (args.num_envs, width), (group, obs[group].shape)
    assert base.action_manager.total_action_dim == 28
    policy_kwargs = agent_cfg.policy.to_dict()
    policy_kwargs.pop('class_name', None)
    policy_class = BulbGraspTeacherLSTM if args.lstm else BulbGraspTeacherActorCritic
    policy = policy_class(obs, agent_cfg.obs_groups, 28, **policy_kwargs).to(base.device)
    policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    policy.eval()
    if args.video is not None:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageFont
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.progress import (
            draw_stage_progress, stage_progress_snapshot,
        )
        args.video.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(args.video), fps=round(1 / base.step_dt),
                                          codec='libx264', quality=8, macro_block_size=1,
                                          output_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        hud_font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
        base.command_manager.get_term('object_pose')._update_metrics()
        # Warm up rendering without advancing physics or consuming policy steps.
        env.render()
        for _ in range(8):
            base.sim.render()
        report['video'] = {'path': str(args.video.resolve()), 'fps': round(1 / base.step_dt),
                           'resolution': args.video_resolution, 'camera_eye': args.camera_eye,
                           'camera_lookat': args.camera_lookat, 'continuous_unfiltered': True}
    report.update({'iteration': iteration, 'num_envs': args.num_envs,
                   'observation_shapes': {k: list(obs[k].shape) for k in expected},
                   'policy_class': agent_cfg.policy.class_name,
                   'observation_history_length': history,
                   'joint_names': list(base.scene['robot'].joint_names),
                   'action_dim': 28, 'control_dt': base.step_dt, 'gravity_z': gravity,
                   'robot_usd': cfg.scene.robot.spawn.usd_path,
                   'bulb_usd': cfg.scene.object.spawn.assets_cfg[0].usd_path,
                   'strict_checkpoint_load': True})
    completed = 0
    steps = 0
    start = time.monotonic()
    terminal_episodes = []
    saw_stage_two = False
    with torch.inference_mode():
        while app.is_running() and (args.steps == 0 or steps < args.steps):
            tick = time.monotonic()
            if video_writer is not None:
                command = base.command_manager.get_term('object_pose')
                stage = int(command.metrics['stage'][0].item())
                position_cm = float(command.metrics['position_error'][0].item()) * 100
                tilt_deg = float(command.metrics['thread_down_error_deg'][0].item())
                success = bool(command.metrics['success'][0].item())
                insertion_metrics = {name: float(command.metrics[name][0].item()) for name in (
                    'insertion_depth_m', 'insertion_radial_error_m', 'insertion_tilt_deg',
                    'insertion_speed_m_s', 'insertion_angular_speed_rad_s', 'insertion_hold_s', 'insertion_tip_radial_error_m'
                    , 'screw_turns', 'screw_angular_turns', 'screw_best_turns', 'screw_depth_target_m',
                    'screw_helix_error_m', 'screw_hold_s', 'stage3_elapsed_s'
                ) if name in command.metrics}
                progress = stage_progress_snapshot(command)
                video_trace.append({'frame': steps, 'time_s': steps * base.step_dt,
                                    'episode': completed + 1, 'stage': stage,
                                    'position_error_cm': position_cm, 'thread_tilt_deg': tilt_deg,
                                    'success': success, **insertion_metrics, 'progress': progress})
                frame = draw_stage_progress(Image.fromarray(env.render()), progress,
                    f'Bulb v{report["curriculum_version"]} | model {iteration} | Episode {completed + 1} | {steps * base.step_dt:.1f}s',
                    hud_font)
                video_writer.append_data(np.asarray(frame))
                if steps == 0 or (stage == 2 and not saw_stage_two):
                    suffix = '.preview.png' if steps == 0 else '.stage2.png'
                    frame.save(args.video.with_suffix(suffix))
                saw_stage_two |= stage == 2
            assert all(torch.isfinite(obs[k]).all() for k in expected)
            actions = policy.act_inference(obs)
            assert actions.shape == (args.num_envs, 28) and torch.isfinite(actions).all()
            obs, reward, terminated, truncated, _ = env.step(actions)
            policy.reset(terminated | truncated)
            assert torch.isfinite(reward).all()
            if args.video is not None and bool((terminated | truncated).any()):
                terminal_episodes.append({'episode': completed + 1, 'step': steps + 1,
                                          'success': bool(base.command_manager.get_term('object_pose').terminal_success[0].item()),
                                          'time_out': bool(truncated[0].item()),
                                          'terminated': bool(terminated[0].item())})
            completed += int((terminated | truncated).sum().item())
            steps += 1
            if args.real_time:
                time.sleep(max(0.0, base.step_dt - (time.monotonic() - tick)))
    assert all(torch.isfinite(obs[k]).all() for k in expected)
    report.update({'passed': True, 'steps': steps, 'completed_episodes': completed,
                   'elapsed_seconds': time.monotonic() - start,
                   'note': 'Finite-state/checkpoint compatibility check only; not a success-rate evaluation.'})
    if args.video is not None:
        report.update({'video_frames': len(video_trace), 'saw_stage_two': saw_stage_two,
                       'terminal_episodes': terminal_episodes})
    print('BULB_GRASP_TEACHER_REPORT ' + json.dumps(report), flush=True)
except Exception as exc:
    report['error'] = repr(exc)
    raise
finally:
    if video_writer is not None:
        video_writer.close()
        args.video.with_suffix('.trace.json').write_text(json.dumps(video_trace, indent=2) + '\n')
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    if env is not None:
        env.close()
    app.close()
