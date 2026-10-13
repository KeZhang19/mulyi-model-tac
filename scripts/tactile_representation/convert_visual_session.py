#!/usr/bin/env python3
"""Convert every saved visual-calibration frame into raw-unit CLIP sim shards."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/BrainCo_DexHand'))
from BrainCo_DexHand.tactile_representation.training.data import TactileNormalization
from BrainCo_DexHand.tactile_representation.training.alignment_domain import AlignmentDomain, FIELDS


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def convert_visual_session(source, output, normalization, *, shard_size=128):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a new output directory separate from the source session')
    if shard_size < 1:
        raise ValueError('shard-size must be positive')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('schema') != 'visual_calibration_tag_v1' or manifest.get('status') not in ('closed', 'error'):
        raise ValueError('Expected a closed visual_calibration_tag_v1 source session')
    reports = manifest.get('episodes', [])
    ids = [r['episode_id'] for r in reports]
    if not reports or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Source episode IDs are missing, invalid or duplicated')
    files = [source / n for n in ('manifest.json', 'config.json', 'coordinate_contract.json', 'layout.npz')]
    for i in ids:
        files.extend(source / f'episode_{i:05d}{suffix}' for suffix in ('.npz', '.json', '_plan.json'))
    before = {str(p.relative_to(source)): digest(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    domain = output / 'sim'
    (domain / 'shards').mkdir(parents=True)
    (output / 'provenance').mkdir()
    for p in files:
        if p.suffix != '.npz' or p.name == 'layout.npz':
            shutil.copy2(p, output / 'provenance' / p.name)
    state = {'schema': 'visual_to_clip_sim_v1', 'status': 'writing',
             'source_session': str(source), 'source_sha256': before,
             'source_schema': manifest['schema'], 'source_complete': manifest['complete'],
             'normalization_expected': asdict(normalization),
             'pair_valid': False, 'training_ready': False,
             'pair_id_semantics': 'source-local episode_id/frame_id; cross-domain correspondence unresolved',
             'selection': 'every saved frame, including positioning, baseline, loading, unloading and failed episodes',
             'resize': 'bilinear align_corners=True; RGB rounded uint8; depth stays metres; Marker pixel axes scaled',
             'unavailable': ['canonical real pairing', 'measured real pose/force', 'validated optics/material/TCP',
                             'confirmed real unloaded reference', 'independent cross-domain validation']}
    write_json(output / 'conversion.json', state)
    shapes = {'rgb': ('uint8', [3, normalization.image_height, normalization.image_width]),
              'depth_m': ('float32', [1, normalization.image_height, normalization.image_width]),
              'marker_2d': ('float32', [normalization.marker_count, 5]),
              'marker_valid': ('bool', [normalization.marker_count]), 'pair_id': ('int64', [2]),
              'pair_valid': ('bool', []), 'training_ready': ('bool', [])}
    fields = {k: {'dtype': dtype, 'sample_shape': shape} for k, (dtype, shape) in shapes.items()}
    shards, episodes, phase_counts = [], [], Counter()
    try:
        for report in reports:
            episode = report['episode_id']
            path = source / f'episode_{episode:05d}.npz'
            with np.load(path, allow_pickle=False) as a:
                n = len(a['frame_id'])
                if n != report['frames']:
                    raise ValueError('Source frame/report count mismatch')
                if n == 0:
                    episodes.append({'episode_id': episode, 'sample_count': 0, 'source_report': report})
                    continue
                rgb, depth, marker = (a[k] for k in ('rgb', 'depth_m', 'marker'))
                h, w = depth.shape[1:]
                if (rgb.dtype != np.uint8 or rgb.shape != (n, h, w, 3) or depth.dtype != np.float32
                        or marker.dtype != np.float32 or marker.shape != (n, normalization.marker_count, 5)):
                    raise ValueError('Source RGB/depth/Marker shape or dtype is incompatible')
                if (min(h, w) < 2 or not np.isfinite(depth).all() or not np.isfinite(marker).all()
                        or (depth < 0).any() or not np.isin(marker[..., 4], [0, 1]).all()):
                    raise ValueError('Source has invalid depth/Marker values')
                if (not np.array_equal(a['frame_id'], np.arange(n)) or not np.all(a['episode_id'] == episode)
                        or len(a['phase']) != n):
                    raise ValueError('Source frame keys or phases are inconsistent')
                phase_counts.update(a['phase'].tolist())
                # Keep the original phase strings, coordinate/force channels and failure reports.
                trace_keys = ('frame_id', 'time_s', 'phase', 'episode_id', 'machine_position_mm',
                              'command_machine_position_mm', 'T_tag_tip', 'T_tag_flange',
                              'total_normal_load_n', 'force_on_sensor_w_n', 'contact_mask')
                with (output / 'provenance' / f'episode_{episode:05d}_trace.npz').open('xb') as stream:
                    np.savez_compressed(stream, **{k: a[k] for k in trace_keys})
                start_shard = len(shards)
                for start in range(0, n, shard_size):
                    stop = min(n, start + shard_size)
                    size = stop - start
                    target = (normalization.image_height, normalization.image_width)
                    image = torch.from_numpy(rgb[start:stop].copy()).permute(0, 3, 1, 2).float()
                    rgb_out = F.interpolate(image, target, mode='bilinear', align_corners=True).round().clamp(0, 255).byte().numpy()
                    depth_out = F.interpolate(torch.from_numpy(depth[start:stop].copy())[:, None], target,
                                              mode='bilinear', align_corners=True).numpy()
                    marker_out = marker[start:stop].copy()
                    marker_out[..., [0, 2]] *= (target[1] - 1) / (w - 1)
                    marker_out[..., [1, 3]] *= (target[0] - 1) / (h - 1)
                    valid = marker_out[..., 4].astype(bool)
                    values = dict(rgb=rgb_out, depth_m=depth_out, marker_2d=marker_out,
                                  marker_valid=valid, pair_id=np.c_[np.full(size, episode, dtype=np.int64),
                                                                 a['frame_id'][start:stop].astype(np.int64)],
                                  pair_valid=np.zeros(size, dtype=bool), training_ready=np.zeros(size, dtype=bool))
                    assert set(values) == set(FIELDS)
                    name = f'shards/shard_{len(shards):06d}.npz'
                    with (domain / name).open('xb') as stream:
                        np.savez_compressed(stream, **values)
                    shards.append({'file': name, 'sha256': digest(domain / name), 'sample_count': size})
                episodes.append({'episode_id': episode, 'sample_count': n, 'first_shard': start_shard,
                                 'shard_count': len(shards) - start_shard, 'source_report': report})
                print(f'converted episode={episode} frames={n} shards={len(shards)}', flush=True)
        count = sum(s['sample_count'] for s in shards)
        if not count:
            raise ValueError('Source session contains no saved frames')
        if before != {str(p.relative_to(source)): digest(p) for p in files}:
            raise ValueError('Source session changed during conversion')
        info = {'schema': 'tactile_alignment_domain_npz_v1', 'status': 'complete', 'domain': 'sim',
                'sample_count': count, 'fields': fields, 'shards': shards,
                'units': {'rgb': 'uint8 0-255 CHW', 'depth_m': 'm',
                          'marker_2d': 'target image pixels: u0,v0,du,dv,valid'},
                'normalization_expected': asdict(normalization), 'pair_valid': False, 'training_ready': False,
                'source_conversion': '../conversion.json'}
        write_json(domain / 'manifest.json', info)
        checked = AlignmentDomain(output, domain / 'manifest.json', normalization, domain='sim')
        expected = np.concatenate([np.c_[np.full(r['frames'], r['episode_id'], dtype=np.int64),
                                             np.arange(r['frames'], dtype=np.int64)] for r in reports])
        if not np.array_equal(checked.arrays['pair_id'], expected):
            raise ValueError('Conversion changed or dropped source frame keys')
        state.update(status='complete', sample_count=count, shard_count=len(shards), episodes=episodes,
                     phase_counts=dict(phase_counts), input_files_unchanged=len(before),
                     physical_calibration_validated=False, geometry_validated=False,
                     reference_unloaded_confirmed=False, independent_validation=False)
        write_json(output / 'conversion.json', state)
        write_json(output / 'verification.json', {'passed': True, 'samples': count, 'episodes': len(reports),
                   'all_source_frames_preserved': True, 'all_source_phases_preserved': True,
                   'all_shards_validated': len(shards), 'input_files_unchanged': len(before),
                   'pair_valid': False, 'training_ready': False})
        return state
    except BaseException as exc:
        state.update(status='error', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'conversion.json', state)
        # A failed conversion never exposes a completed domain manifest.
        (domain / 'manifest.json').unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Original visual session with episode NPZs')
    p.add_argument('--output', type=Path, required=True, help='New dataset directory; existing paths refused')
    p.add_argument('--checkpoint', type=Path, help='Read the original encoder normalization and input geometry')
    p.add_argument('--shard-size', type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(1)
    norm = TactileNormalization(240, 320, 100)
    checkpoint_hash = None
    if args.checkpoint:
        cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        norm = TactileNormalization(**cp['normalization'])
        checkpoint_hash = digest(args.checkpoint)
    result = convert_visual_session(args.source, args.output, norm, shard_size=args.shard_size)
    if args.checkpoint:
        if digest(args.checkpoint) != checkpoint_hash:
            raise ValueError('Normalization checkpoint changed during conversion')
        result['normalization_checkpoint_sha256'] = checkpoint_hash
        write_json(args.output / 'conversion.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'sample_count', 'shard_count', 'training_ready')}, indent=2))


if __name__ == '__main__':
    main()
