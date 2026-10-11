"""Read-only startup audit; does not launch, interrupt, or change training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import subprocess

root = Path('/home/admin/workspace/xinyu/mulyi-model-tac-v6-20261007')
out = root / 'experiments/bulb_v6_dsw_resume_20261007'
run = root / 'logs/rsl_rl/bulb_grasp_teacher_v6/2026-10-07_resume9500_2x8192'
text = (out / 'train.log').read_text(errors='replace')
iterations = sorted(set(map(int, re.findall(r'Learning iteration (\d+)/', text))))
native_paths = re.findall(r'^\[Info\] \[carb\] Logging to file: (.+)$', text, re.M)
required = [run / f'rank{rank}' / name for rank in (0, 1)
            for name in ('launch_contract.json', 'distributed_verified.json', 'params/env.yaml')]
required.append(run / 'rank0/model_9500.pt')
missing = [str(p) for p in required if not p.is_file()]
if missing or len(iterations) < 10:
    print(json.dumps({'ready': False, 'iterations': iterations[-12:], 'missing': missing,
                      'tail': text.splitlines()[-8:]}, ensure_ascii=False))
    raise SystemExit(0)

import torch
import yaml

checks = {}
parent_meta = json.loads((out / 'resume_source.json').read_text())
parent_path = Path(parent_meta['checkpoint'])
parent_cfg = yaml.load((parent_path.parent / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
parent_root = '/home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006'
expected_capacities = dict(gpu_collision_stack_size=4026531840,
    gpu_max_rigid_contact_count=16777216, gpu_max_rigid_patch_count=3000000,
    gpu_found_lost_pairs_capacity=4194304, gpu_found_lost_aggregate_pairs_capacity=67108864,
    gpu_total_aggregate_pairs_capacity=4194304, gpu_heap_capacity=134217728,
    gpu_temp_buffer_capacity=33554432)
configs = []
contracts = []
distributed = []
for rank in (0, 1):
    r = run / f'rank{rank}'
    cfg = yaml.load((r / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    contract = json.loads((r / 'launch_contract.json').read_text())
    ddp = json.loads((r / 'distributed_verified.json').read_text())
    configs.append(cfg)
    contracts.append(contract)
    distributed.append(ddp)
    checks[f'rank{rank}_config'] = (
        int(cfg['bulb_grasp_curriculum_version']) == 6
        and int(cfg['scene']['num_envs']) == 8192
        and float(cfg['sim']['dt']) == 1 / 120
        and int(cfg['decimation']) == 4
        and float(cfg['episode_length_s']) == 14
        and float(cfg['curriculum']['adr']['params']['init_difficulty']) == 0
        and cfg['rewards']['success']['func'].endswith(':stage_two_position_orientation_reward')
        and float(cfg['rewards']['success']['params']['orientation_floor']) == .2
        and float(cfg['rewards']['success']['weight']) == 10
        and float(cfg['rewards']['thread_down']['weight']) == 5
        and float(cfg['rewards']['final_screw_success']['weight']) == 20
    )
    checks[f'rank{rank}_ddp'] = (
        ddp['verified'] and ddp['gradient_reductions'] > 0
        and len(set(ddp['parameter_sha256_by_rank'])) == 1
        and len(ddp['parameter_sha256_by_rank']) == 2
        and contract['world_size'] == 2 and contract['total_envs'] == 16384
        and contract['seed'] == 42 + rank and not contract['from_scratch']
    )
    checks[f'rank{rank}_capacities'] = all(int(cfg['sim']['physx'][k]) == v
                                         for k, v in expected_capacities.items())
    checks[f'rank{rank}_parent_action_observation_command_contract'] = (
        cfg['actions'] == parent_cfg['actions']
        and json.dumps(cfg['observations'], sort_keys=True)
            == json.dumps(parent_cfg['observations'], sort_keys=True).replace(parent_root, str(root))
        and cfg['commands'] == parent_cfg['commands']
        and cfg['sim']['physx'] == parent_cfg['sim']['physx']
    )
    checks[f'rank{rank}_other_rewards_unchanged'] = all(
        value == parent_cfg['rewards'][name] for name, value in cfg['rewards'].items() if name != 'success')

parent = torch.load(parent_path, weights_only=False, map_location='cpu')
new_path = run / 'rank0/model_9500.pt'
new = torch.load(new_path, weights_only=False, map_location='cpu')
old_steps = sorted(set(int(v['step']) for v in parent['optimizer_state_dict']['state'].values()))
new_steps = sorted(set(int(v['step']) for v in new['optimizer_state_dict']['state'].values()))
checks['parent_unchanged'] = hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_meta['sha256']
checks['optimizer_resumed_and_advanced'] = min(new_steps) > max(old_steps) == 190020
checks['model_finite'] = all(torch.isfinite(v).all().item() for v in new['model_state_dict'].values())
checks['model_updated'] = any(not torch.equal(parent['model_state_dict'][k], v)
                              for k, v in new['model_state_dict'].items() if k.startswith('actor.'))
checks['normalizers_retained'] = all(
    new['model_state_dict'][f'{side}_obs_normalizer.count'].item()
    >= parent['model_state_dict'][f'{side}_obs_normalizer.count'].item() > 0
    for side in ('actor', 'critic'))
checks['gravity_state_restored'] = new['v2_gravity_curriculum_state']['trigger_iteration'] == parent['v2_gravity_curriculum_state']['trigger_iteration']

log_scan = {}
fatal = re.compile(r'Traceback \(most recent call last\)|CUDA.*out of memory|CUDA error|PhysX error|'
                   r'\[Error\].*(?:physx|omni\.physx)|(?:contact|patch|pair|collision|foundLost).*'
                   r'(?:overflow|capacity.*exceed|buffer.*too small|lost contacts|dropped contacts)', re.I)
for path in [out / 'train.log', *map(Path, native_paths)]:
    content = path.read_text(errors='replace')
    lines = content.splitlines()
    log_scan[str(path)] = {'bytes': path.stat().st_size,
                          'fatal_or_capacity_matches': [s for s in lines if fatal.search(s)][:30],
                          'error_lines': [s for s in lines if '[Error]' in s][:30]}
checks['no_fatal_or_capacity_log_errors'] = not any(x['fatal_or_capacity_matches'] for x in log_scan.values())
checks['two_native_logs_scanned'] = len(native_paths) == 2
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                               '--format=csv,noheader'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                     '--format=csv,noheader'], text=True)
report = {'verified_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'passed': all(checks.values()), 'checks': checks,
          'iterations_observed': iterations, 'parent_optimizer_steps': old_steps,
          'new_optimizer_steps': new_steps, 'new_checkpoint': str(new_path),
          'new_checkpoint_sha256': hashlib.sha256(new_path.read_bytes()).hexdigest(),
          'contracts': contracts, 'distributed': distributed,
          'gpu': gpu, 'gpu_processes': processes, 'log_scan': log_scan,
          'recent_stage_metrics': [s.strip() for s in text.splitlines()[-170:]
              if any(k in s for k in ('stage1_success:', 'stage2_success:', 'stage3_success:',
                                      'Curriculum/adr:', 'GravityCurriculum/gravity_z:'))],
          'limitations': 'Startup/continuation and initial capacity checks only; no independent task success evaluation.'}
(out / 'startup_verified.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks, 'iterations': iterations[-12:],
                  'optimizer_steps': new_steps, 'gpu': gpu, 'error_lines':
                  {k: v['error_lines'] for k, v in log_scan.items()}}, ensure_ascii=False))
if not report['passed']:
    raise SystemExit(1)
