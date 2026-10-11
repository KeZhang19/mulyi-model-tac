"""Bounded network-only GPU benchmark. No Isaac, optimizer update or training edits."""
from pathlib import Path
import argparse
import contextlib
import datetime
import gc
import hashlib
import importlib.util
import io
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime/bulb_grasp_teacher'))
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mlp-checkpoint', type=Path, required=True)
parser.add_argument('--lstm-checkpoint', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=8)
parser.add_argument('--batch-sizes', type=int, nargs='+', default=[64, 256, 1024])
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()
assert not args.output.exists()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
torch.set_num_threads(8)
torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(0.04, 0)
task = ROOT / 'source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher'
spec = importlib.util.spec_from_file_location('benchmark_teacher_policy', task / 'policy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cfg = json.loads((task / 'agent_snapshot.json').read_text())
states = {}
provenance = {}
for name, path in [('mlp', args.mlp_checkpoint), ('lstm', args.lstm_checkpoint)]:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    states[name] = saved['model_state_dict']
    provenance[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    del saved

def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader'], text=True)

result = {'started_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'pid': os.getpid(), 'gpu_before': gpu_snapshot(), 'checkpoint_sources': provenance,
          'torch_version': torch.__version__, 'cudnn_version': torch.backends.cudnn.version(),
          'gpu': torch.cuda.get_device_name(), 'steps_per_sample': 32, 'repeats': args.repeats,
          'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
          'conditions': 'Synthetic observations drawn using each checkpoint normalizer mean/std. Real policy and PPO rollout code. No physics, reward computation, DDP or optimizer updates. Shares GPU0 with production; CUDA event intervals include scheduling interference. Small batches cannot be linearly extrapolated to8192.',
          'cases': []}
started = time.monotonic()
with torch.inference_mode():
    for n in args.batch_sizes:
        assert 1 <= n <= 1024
        policies, obs, algs = {}, {}, {}
        for name, history in [('mlp', 5), ('lstm', 1)]:
            example = TensorDict({g: torch.zeros(n, width * history, device='cuda')
                                  for g, width in [('policy', 39), ('proprio', 149), ('perception', 192)]}, [n])
            kw = cfg['policy'].copy()
            kw.pop('class_name')
            if name == 'lstm':
                kw.update(rnn_type='lstm', rnn_hidden_dim=256, rnn_num_layers=1)
            cls = ActorCritic if name == 'mlp' else module.BulbGraspTeacherLSTM
            with contextlib.redirect_stdout(io.StringIO()):
                net = cls(example, cfg['obs_groups'], 28, **kw).to('cuda')
            net.load_state_dict(states[name], strict=True)
            net.train()
            flat = net.actor_obs_normalizer._mean + net.actor_obs_normalizer._std * torch.randn(n, 380 * history, device='cuda')
            offset = 0
            for group, width in [('policy', 39), ('proprio', 149), ('perception', 192)]:
                example[group] = flat[:, offset:offset + width * history].contiguous()
                offset += width * history
            policies[name], obs[name] = net, example
            algorithm = cfg['algorithm'].copy()
            algorithm.pop('class_name')
            assert algorithm.pop('share_cnn_encoders') is False
            alg = PPO(net, device='cuda', **algorithm)
            alg.init_storage('rl', n, 32, example, [28])
            algs[name] = alg
        rewards = torch.zeros(n, device='cuda')
        # Deterministic asynchronous episode ends, roughly1/200 per step.
        dones = [((torch.arange(n, device='cuda') + j) % 200 == 0).long() for j in range(32)]

        def run_block(name, mode):
            net, x, alg = policies[name], obs[name], algs[name]
            alg.storage.clear()
            for step in range(32):
                if mode == 'policy_forward':
                    actions = net.act(x)
                    values = net.evaluate(x)
                    log_prob = net.get_actions_log_prob(actions)
                else:
                    actions = alg.act(x)
                    alg.process_env_step(x, rewards, dones[step], {})
            return actions

        for mode in ['policy_forward', 'neural_rollout']:
            for name in ['mlp', 'lstm']:
                policies[name].reset()
                for _ in range(2):
                    actions = run_block(name, mode)
                assert torch.isfinite(actions).all().item()
            samples = {'mlp': [], 'lstm': []}
            for repeat in range(args.repeats):
                for name in (['mlp', 'lstm'] if repeat % 2 == 0 else ['lstm', 'mlp']):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    wall_start = time.perf_counter()
                    begin.record()
                    actions = run_block(name, mode)
                    end.record()
                    end.synchronize()
                    samples[name].append({'cuda_ms': begin.elapsed_time(end), 'wall_ms': (time.perf_counter() - wall_start) * 1000})
            summary = {name: {key: statistics.median(v[key] for v in values) for key in ['cuda_ms', 'wall_ms']}
                       for name, values in samples.items()}
            summary['paired_cuda_delta_median_ms'] = statistics.median(
                samples['lstm'][i]['cuda_ms'] - samples['mlp'][i]['cuda_ms'] for i in range(args.repeats))
            row = {'num_envs': n, 'mode': mode, 'samples': samples, 'summary_per32steps': summary,
                   'torch_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                   'torch_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
            result['cases'].append(row)
            if args.profile:
                row['profiles'] = {}
                for name in ['mlp', 'lstm']:
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        actions = run_block(name, mode)
                        torch.cuda.synchronize()
                    gpu_events = [event for event in prof.events()
                                  if event.device_type == torch.autograd.DeviceType.CUDA]
                    totals = {}
                    for event in gpu_events:
                        totals[event.name] = totals.get(event.name, 0.0) + event.device_time_total
                    row['profiles'][name] = {
                        'cuda_kernel_count': len(gpu_events),
                        'sum_cuda_event_ms': sum(event.device_time_total for event in gpu_events) / 1000,
                        'top_cuda_events_ms': [(key, value / 1000) for key, value
                                              in sorted(totals.items(), key=lambda pair: -pair[1])[:10]],
                    }
            print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            if time.monotonic() - started > 180:
                raise RuntimeError('Diagnostic exceeded180s runtime budget')
        del policies, obs, algs, alg, net, example, flat, actions, dones, rewards
        gc.collect()
        torch.cuda.empty_cache()

result['elapsed_seconds'] = time.monotonic() - started
result['gpu_after'] = gpu_snapshot()
result['checkpoint_unchanged'] = {name: hashlib.sha256(Path(p['path']).read_bytes()).hexdigest() == p['sha256']
                                  for name, p in provenance.items()}
result['completed'] = True
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'completed': True, 'elapsed_seconds': result['elapsed_seconds'],
                  'checkpoint_unchanged': result['checkpoint_unchanged']}), flush=True)
