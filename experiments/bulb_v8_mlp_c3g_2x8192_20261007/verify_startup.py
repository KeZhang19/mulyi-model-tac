"""Read-only audit of the v8 two-GPU five-frame MLP startup and saved checkpoint."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import subprocess

root = Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out = root / 'experiments/bulb_v8_mlp_c3g_2x8192_20261007'
run = root / 'logs/rsl_rl/bulb_grasp_teacher_v8_mlp/2026-10-07_scratch_2x8192'
text = (out / 'train.log').read_text(errors='replace')
iterations = sorted(set(map(int, re.findall(r'Learning iteration (\d+)/', text))))
native_paths = re.findall(r'^\[Info\] \[carb\] Logging to file: (.+)$', text, re.M)
required = [run / f'rank{rank}' / name for rank in (0, 1)
            for name in ('launch_contract.json', 'distributed_verified.json', 'params/env.yaml', 'params/agent.yaml')]
required.append(run / 'rank0/model_0.pt')
missing = [str(p) for p in required if not p.is_file()]
if missing or len(iterations) < 10:
    print(json.dumps({'ready': False, 'iterations': iterations[-12:], 'missing': missing,
                      'tail': text.splitlines()[-8:]}))
    raise SystemExit(0)

import torch
import yaml

checks = {}
contracts = []
distributed = []
expected_capacities = dict(gpu_collision_stack_size=4026531840,
    gpu_max_rigid_contact_count=16777216, gpu_max_rigid_patch_count=3000000,
    gpu_found_lost_pairs_capacity=4194304, gpu_found_lost_aggregate_pairs_capacity=67108864,
    gpu_total_aggregate_pairs_capacity=4194304, gpu_heap_capacity=134217728,
    gpu_temp_buffer_capacity=33554432)
for rank in (0, 1):
    r = run / f'rank{rank}'
    cfg = yaml.load((r / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load((r / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    contract = json.loads((r / 'launch_contract.json').read_text())
    ddp = json.loads((r / 'distributed_verified.json').read_text())
    contracts.append(contract)
    distributed.append(ddp)
    checks[f'rank{rank}_task'] = (
        int(cfg['bulb_grasp_curriculum_version']) == 8
        and float(cfg['commands']['object_pose']['shallow_insertion_tilt_tolerance_deg']) == 30
        and float(cfg['commands']['object_pose']['insertion_depth_m']) == .0055
        and float(cfg['commands']['object_pose']['insertion_tilt_tolerance_deg']) == 5
        and float(cfg['commands']['object_pose']['insertion_hold_s']) == .2
        and int(cfg['scene']['num_envs']) == 8192
        and float(cfg['sim']['dt']) == 1 / 120
        and int(cfg['decimation']) == 4
        and float(cfg['episode_length_s']) == 14
        and float(cfg['curriculum']['adr']['params']['init_difficulty']) == 0
        and float(cfg['rewards']['success']['params']['orientation_floor']) == .2
        and float(cfg['rewards']['stage2_success_bonus']['weight']) == 10
        and float(cfg['rewards']['final_screw_success']['weight']) == 20
    )
    checks[f'rank{rank}_mlp'] = (
        agent['policy']['class_name'] == 'ActorCritic'
        and list(map(int,agent['policy']['actor_hidden_dims'])) == [512,256,128]
        and list(map(int,agent['policy']['critic_hidden_dims'])) == [512,256,128]
        and int(agent['num_steps_per_env']) == 32
        and int(agent['max_iterations']) == 30000
        and all(int(cfg['observations'][group]['history_length']) == 5
                for group in ('policy', 'proprio', 'perception'))
    )
    checks[f'rank{rank}_ddp'] = (
        ddp['verified'] and ddp['gradient_reductions'] > 0
        and len(set(ddp['parameter_sha256_by_rank'])) == 1
        and len(ddp['parameter_sha256_by_rank']) == 2
        and contract['world_size'] == 2 and contract['total_envs'] == 16384
        and contract['seed'] == 42 + rank and contract['from_scratch']
        and contract['checkpoint'] is None and contract['num_envs_per_rank'] == 8192
        and contract['cuda_visible_devices'] == str(rank)
    )
    checks[f'rank{rank}_capacities'] = all(int(cfg['sim']['physx'][k]) == v
                                         for k, v in expected_capacities.items())

checkpoint = run / 'rank0/model_0.pt'
saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
state = saved['model_state_dict']
optimizer = saved['optimizer_state_dict']['state']
steps = sorted(set(int(v['step']) for v in optimizer.values()))
checks['optimizer_updated'] = steps == [20]
checks['optimizer_finite'] = all(torch.isfinite(v).all().item()
                               for s in optimizer.values() for v in s.values() if isinstance(v, torch.Tensor))
checks['model_finite'] = all(torch.isfinite(v).all().item() for v in state.values())
checks['actor_critic_mlp_checkpoint'] = (all(tuple(state[f'{side}.0.weight'].shape) == (512,1900) for side in ('actor','critic')) and not any(k.startswith('memory_') for k in state))
checks['normalizers_observed_samples'] = all(state[f'{side}_obs_normalizer.count'].item() > 0
                                             for side in ('actor', 'critic'))
old_stop = json.loads((out / 'old_stopped.json').read_text())
checks['old_checkpoint_preserved'] = hashlib.sha256(Path(old_stop['latest_checkpoint']).read_bytes()).hexdigest() == old_stop['sha256_before']
log_scan = {}
fatal = re.compile(r'Traceback \(most recent call last\)|CUDA.*out of memory|CUDA error|PhysX error|'
                   r'\[Error\].*(?:physx|omni\.physx)|(?:contact|patch|pair|collision|foundLost).*'
                   r'(?:overflow|capacity.*exceed|buffer.*too small|lost contacts|dropped contacts)', re.I)
for path in [out / 'train.log', *map(Path, native_paths)]:
    lines = path.read_text(errors='replace').splitlines()
    log_scan[str(path)] = {'bytes': path.stat().st_size,
                          'fatal_or_capacity_matches': [s for s in lines if fatal.search(s)][:30],
                          'error_lines': [s for s in lines if '[Error]' in s][:30]}
checks['no_fatal_or_capacity_errors'] = not any(x['fatal_or_capacity_matches'] for x in log_scan.values())
checks['no_error_lines'] = not any(x['error_lines'] for x in log_scan.values())
checks['two_native_logs_scanned'] = len(native_paths) == 2
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                               '--format=csv,noheader'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                     '--format=csv,noheader'], text=True)
report = {'verified_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'passed': all(checks.values()), 'checks': checks, 'iterations_observed': iterations,
          'optimizer_steps': steps, 'checkpoint': str(checkpoint),
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'contracts': contracts, 'distributed': distributed, 'gpu': gpu, 'gpu_processes': processes,
          'log_scan': log_scan, 'recent_stage_metrics': [s.strip() for s in text.splitlines()[-200:]
              if any(k in s for k in ('stage1_success:', 'stage2_success:', 'stage3_success:',
                                      'Curriculum/adr:', 'GravityCurriculum/gravity_z:'))],
          'limitations': 'Startup and initial capacity only; no independent task-success evaluation.'}
(out / 'startup_verified.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks, 'iterations': iterations[-12:],
                  'optimizer_steps': steps, 'gpu': gpu, 'error_lines':
                  {k: v['error_lines'] for k, v in log_scan.items()}}, ensure_ascii=False))
if not report['passed']:
    raise SystemExit(1)
