#!/usr/bin/env python3
"""Train/resume the transplanted teacher with its original gravity-aware runner."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import os
import json
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/BrainCo_DexHand'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=None, help='Omit for a fresh run; supply for full-state resume.')
migrations = parser.add_mutually_exclusive_group()
migrations.add_argument('--migrate-v8-tip-command', action='store_true',
                    help='Explicitly resume a v8 checkpoint into v9 direct tip commands; records the MDP change.')
migrations.add_argument('--migrate-v9-entry-window', action='store_true',
                    help='Explicitly resume v9 into v10 with the 12 mm radial / 0..10 mm depth entry window.')
parser.add_argument('--lstm', action='store_true', help='Use 1 observation frame and a single 256-unit LSTM before each actor/critic MLP.')
parser.add_argument('--num_envs', type=int, default=8192)
parser.add_argument('--max_iterations', type=int, required=True, help='Additional PPO iterations.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_dir', type=Path, default=None)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--expected_world_size', type=int, default=None)
parser.add_argument('--initial_adr_difficulty', type=float, default=None,
                    help='Explicit initial ADR difficulty; checkpoints do not save per-environment ADR state.')
parser.add_argument('--physx_buffer_scale', type=int, default=1, choices=(1, 2, 4),
                    help='Per-rank multiplier for rigid contact/pair, heap and temporary buffer capacities.')
parser.add_argument('--physx_collision_stack_size', type=int, default=None,
                    help='Per-rank collision stack bytes; must be positive and below 2**32.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.migrate_v8_tip_command and args.checkpoint is None:
    parser.error('--migrate-v8-tip-command requires an explicit --checkpoint')
if args.migrate_v9_entry_window and args.checkpoint is None:
    parser.error('--migrate-v9-entry-window requires an explicit --checkpoint')
if args.num_envs < 1 or args.max_iterations < 1:
    parser.error('num_envs and max_iterations must be positive')
if args.physx_collision_stack_size is not None and not 0 < args.physx_collision_stack_size < 2**32:
    parser.error('physx_collision_stack_size must be positive and below 2**32')
rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
if args.expected_world_size is not None and world_size != args.expected_world_size:
    parser.error(f'Expected {args.expected_world_size} ranks, got {world_size}')
if world_size > 1 and not args.distributed:
    parser.error('torchrun launches require --distributed')
if args.lstm:
    # Pin the tested recurrent PPO implementation for both single- and multi-GPU entry points.
    sys.path.insert(0, str(ROOT / 'runtime/bulb_grasp_teacher'))
app = AppLauncher(args).app
import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import BrainCo_DexHand
from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.teacher_runner import OnPolicyRunner

TASK = 'BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0'
env = None
try:
    cfg = load_cfg_from_registry(TASK, 'env_cfg_entry_point')
    target_migration = None
    if args.checkpoint and cfg.bulb_grasp_curriculum_version >= 9:
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.target_contract import validate_target_contract
        target_migration = validate_target_contract(args.checkpoint, cfg.bulb_grasp_curriculum_version,
                                                   allow_v8_migration=args.migrate_v8_tip_command,
                                                   allow_v9_migration=args.migrate_v9_entry_window)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed + rank
    cfg.sim.device = args.device
    capacity_fields = (
        'gpu_max_rigid_contact_count', 'gpu_max_rigid_patch_count',
        'gpu_found_lost_pairs_capacity', 'gpu_found_lost_aggregate_pairs_capacity',
        'gpu_total_aggregate_pairs_capacity', 'gpu_heap_capacity', 'gpu_temp_buffer_capacity',
    )
    for field in capacity_fields:
        setattr(cfg.sim.physx, field, getattr(cfg.sim.physx, field) * args.physx_buffer_scale)
    if args.physx_collision_stack_size is not None:
        cfg.sim.physx.gpu_collision_stack_size = args.physx_collision_stack_size
    if args.initial_adr_difficulty is not None:
        minimum = cfg.curriculum.adr.params.get('min_difficulty', 0)
        maximum = cfg.curriculum.adr.params.get('max_difficulty', 10)
        if not math.isfinite(args.initial_adr_difficulty) or not minimum <= args.initial_adr_difficulty <= maximum:
            raise ValueError(f'initial_adr_difficulty must be finite and in [{minimum}, {maximum}]')
        cfg.curriculum.adr.params['init_difficulty'] = args.initial_adr_difficulty
    elif args.checkpoint is None:
        cfg.curriculum.adr.params['init_difficulty'] = 0.0
    agent = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
    if args.lstm:
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.BulbGraspTeacher.policy import configure_bulb_teacher_lstm
        configure_bulb_teacher_lstm(cfg, agent)
    agent.seed = cfg.seed
    agent.device = args.device
    agent.max_iterations = args.max_iterations
    out = args.log_dir or ROOT / 'logs/rsl_rl/bulb_grasp_teacher_v2test' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if world_size > 1:
        out = out / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    if target_migration is not None:
        (out / 'target_contract_migration.json').write_text(json.dumps({
            **target_migration, 'checkpoint': str(args.checkpoint.resolve()),
            'restore_mode': 'full_state_resume',
            'note': 'Intentional task migration; not a same-MDP continuation.',
        }, indent=2) + '\n')
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=None)
    runner_cfg = agent.to_dict()
    # Older installed RSL-RL has no CNN-sharing option. False is the exact
    # equivalent for this state-only MLP; never silently discard an enabled option.
    import inspect
    from rsl_rl.algorithms import PPO
    if "share_cnn_encoders" not in inspect.signature(PPO).parameters:
        assert runner_cfg["algorithm"].pop("share_cnn_encoders", False) is False
    runner = OnPolicyRunner(env, runner_cfg, log_dir=str(out), device=args.device)
    distributed_contract = {
        'task': TASK, 'rank': rank, 'world_size': world_size, 'seed': cfg.seed,
        'num_envs_per_rank': cfg.scene.num_envs,
        'total_envs': cfg.scene.num_envs * world_size,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'from_scratch': args.checkpoint is None,
        'policy_class': agent.policy.class_name,
        'observation_history_length': cfg.observations.policy.history_length,
        'initial_adr_difficulty': cfg.curriculum.adr.params['init_difficulty'],
        'initial_gravity_z': cfg.v2_gravity_curriculum.initial_gravity_z,
        'physx_capacities_per_rank': {
            field: getattr(cfg.sim.physx, field)
            for field in (*capacity_fields, 'gpu_collision_stack_size')
        },
    }
    (out / 'launch_contract.json').write_text(json.dumps(distributed_contract, indent=2) + '\n')
    print('TEACHER_DISTRIBUTED_CONTRACT ' + json.dumps(distributed_contract), flush=True)
    if world_size > 1:
        # Verify actual gradient reduction and equal parameters after the first
        # optimizer update; a second PID alone is not evidence of joint training.
        import hashlib
        import torch
        sync_probe = {'gradient_reductions': 0, 'verified': False}
        original_reduce = runner.alg.reduce_parameters
        def checked_reduce():
            result = original_reduce()
            sync_probe['gradient_reductions'] += 1
            return result
        runner.alg.reduce_parameters = checked_reduce
        def verify_first_update(optimizer, hook_args, hook_kwargs):
            if sync_probe['verified']:
                return
            digest = hashlib.sha256()
            for parameter in runner.alg.policy.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            hashes = [None] * world_size
            torch.distributed.all_gather_object(hashes, digest.hexdigest())
            assert sync_probe['gradient_reductions'] > 0, 'No PPO gradient all-reduce occurred'
            assert len(set(hashes)) == 1, f'Parameters diverged across ranks: {hashes}'
            sync_probe['verified'] = True
            evidence = {**distributed_contract, **sync_probe, 'parameter_sha256_by_rank': hashes}
            (out / 'distributed_verified.json').write_text(json.dumps(evidence, indent=2) + '\n')
            print('TEACHER_DDP_VERIFIED ' + json.dumps(evidence), flush=True)
        runner.alg.optimizer.register_step_post_hook(verify_first_update)
    if args.checkpoint:
        runner.load(str(args.checkpoint), map_location=args.device)
    # Publish restored gravity before resetting initial episodes.
    runner._apply_env_training_iteration(runner.current_learning_iteration)
    env.reset()
    dump_yaml(str(out / 'params/env.yaml'), cfg)
    dump_yaml(str(out / 'params/agent.yaml'), agent)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    (out / 'completed.txt').write_text(f'Completed {args.max_iterations} additional PPO iterations.\n')
except Exception as exc:
    import traceback
    message = traceback.format_exc()
    print(message, flush=True)
    if "out" in globals() and out.is_dir():
        (out / 'error.txt').write_text(message)
    raise
finally:
    if env is not None:
        env.close()
    import torch
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    app.close()
