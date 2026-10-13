#!/usr/bin/env python3
"""Verify the saved Tag-frame contract and data consistency, not physical accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    cfg = json.loads((folder / 'config.json').read_text())
    contract = json.loads((folder / 'coordinate_contract.json').read_text())
    with np.load(folder / 'layout.npz', allow_pickle=False) as layout:
        area = layout['taxel_area_m2']
    checks = []
    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
    transform = np.asarray(contract['T_tag_machine_m'])
    rotation = transform[:3, :3]
    origin = np.asarray(contract['tag_origin_machine_mm'])
    bounds = np.asarray(contract['axis_bounds_mm'])
    check('Tag root and current M6x16 asset', manifest['frame'].startswith('world=tag_center')
          and cfg['hand_asset'] == 'revo21_touch_tag_m6x16')
    check('proper orthonormal frame', np.allclose(rotation.T @ rotation, np.eye(3))
          and np.isclose(np.linalg.det(rotation), 1))
    check('no physical/training claim', not manifest['pressure_calibrated']
          and not manifest['hardware_connected'] and not manifest['training_ready'])
    check('closed session', manifest['status'] in ('closed', 'error'))
    complete = 0
    total_frames = 0
    for report in manifest['episodes']:
        prefix = f'episode_{report["episode_id"]:05d}'
        with np.load(folder / (prefix + '.npz'), allow_pickle=False) as a:
            n = len(a['frame_id'])
            total_frames += n
            check(prefix + ': frame/report count', n == report['frames'])
            check(prefix + ': completion/fault contract', report['completed'] ==
                  (n == report['planned_frames'] and not report['faults']))
            check(prefix + ': no collision certification claim', not report['geometry_clearance_certified'])
            if n == 0:
                check(prefix + ': zero-frame interruptions labelled', bool(report['faults']))
                continue
            for key in ('time_s', 'frame_id', 'phase', 'rgb', 'depth_m', 'pressure_pa', 'marker',
                        'machine_position_mm', 'command_machine_position_mm', 'T_tag_tip', 'T_tag_flange',
                        'total_normal_load_n', 'force_on_sensor_w_n'):
                check(prefix + ': ' + key + ' synchronized', len(a[key]) == n)
                if a[key].dtype.kind in 'fiu':
                    check(prefix + ': ' + key + ' finite', np.isfinite(a[key]).all())
            check(prefix + ': simulation clock', np.allclose(a['time_s'], np.arange(n) / cfg['sample_hz']))
            check(prefix + ': configured travel', np.all(a['machine_position_mm'] >= bounds[:, 0] - 1e-7)
                  and np.all(a['machine_position_mm'] <= bounds[:, 1] + 1e-7))
            expected = (a['machine_position_mm'] - origin) @ rotation.T / 1000
            check(prefix + ': mm -> Tag tip', np.allclose(a['T_tag_tip'][:, :3, 3], expected, atol=1e-9))
            check(prefix + ': Tag tip -> mm', np.allclose(a['T_tag_tip'][:, :3, 3] @ rotation * 1000 + origin,
                                                        a['machine_position_mm'], atol=1e-6))
            check(prefix + ': command matches prescribed position',
                  np.array_equal(a['machine_position_mm'], a['command_machine_position_mm']))
            delta = np.diff(a['machine_position_mm'], axis=0)
            check(prefix + ': serialized axes', np.all((np.abs(delta) > 1e-7).sum(1) <= 1))
            check(prefix + ': model load = pressure x area',
                  np.allclose((a['pressure_pa'] * area).sum((1, 2)), a['total_normal_load_n'], atol=1e-6))
            if report['completed']:
                complete += 1
                check(prefix + ': all load/unload stages retained',
                      set(('baseline', 'approach', 'load', 'hold', 'unload', 'retract', 'release')).issubset(set(a['phase'])))
                check(prefix + ': unloaded endpoints', not a['contact_mask'][0].any() and not a['contact_mask'][-1].any())
                check(prefix + ': model depth limit', a['depth_m'].max() <= cfg['max_depth_m'] + 1e-8)
                check(prefix + ': model force limit', a['total_normal_load_n'].max() <= cfg['max_model_force_n'])
            else:
                check(prefix + ': failure reason retained', bool(report['faults']))
    check('session completion agrees with episodes', manifest['complete'] ==
          (bool(manifest['episodes']) and complete == len(manifest['episodes']) and manifest['status'] != 'error'))
    result = dict(passed=all(x['passed'] for x in checks), checks=checks,
                  check_count=len(checks), completed_episodes=complete, total_episodes=len(manifest['episodes']),
                  total_frames=total_frames, physical_accuracy_validated=False)
    destination = folder / 'visual_verification.json'
    with destination.open('x') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    result = verify(parser.parse_args().folder)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
