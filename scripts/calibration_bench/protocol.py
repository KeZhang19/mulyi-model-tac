"""Finite, deterministic trajectories and episode-group dataset partitions."""
import itertools
import numpy as np

PROTOCOLS = ('normal', 'hold', 'shear_x', 'shear_y', 'slip_x', 'slip_y', 'twist', 'roll')


def validate_config(c):
    if c['schema_version'] != 1 or c['head'] not in ('spherical', 'triangle', 'conical'):
        raise ValueError('Unsupported config schema/head')
    if c['finger'] not in ('index', 'middle', 'ring', 'little', 'thumb') or c['segment'] not in ('DIP', 'MCP', 'PIP'):
        raise ValueError('Unknown finger/segment')
    for key in ('sample_hz', 'normal_speed_mps', 'slide_speed_mps', 'shell_thickness_m',
                'young_modulus_pa', 'max_depth_m', 'max_model_force_n', 'pressure_gamma',
                'pressure_saturation_pa', 'clearance_m', 'hold_s', 'baseline_s', 'release_s',
                'slide_m', 'slip_scan_m', 'twist_rad', 'roll_rad'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not c['depths_m'] or not all(0 < d <= c['max_depth_m'] < c['shell_thickness_m'] for d in c['depths_m']):
        raise ValueError('Depth must be positive and bounded below shell thickness')
    if not c['protocols'] or set(c['protocols']) - set(PROTOCOLS):
        raise ValueError('Unknown or empty protocols')
    for key in ('grid_hw', 'marker_rows_cols'):
        if len(c[key]) != 2 or any(int(x) != x or x < 4 for x in c[key]):
            raise ValueError(f'Invalid {key}')
    if c['repeats'] < 1 or int(c['repeats']) != c['repeats']:
        raise ValueError('repeats must be a positive integer')
    bounds = np.asarray(c['axis_bounds_m'])
    offsets = np.asarray(c['offsets_xy_m'])
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
        raise ValueError('Invalid axis bounds')
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError('Invalid offsets')
    if not all(np.isfinite(v) and v >= 0 for v in c['hydroshear'].values()):
        raise ValueError('Invalid HydroShear parameters')


def episode_plans(c):
    plans = []
    # Repetitions of the same contact/trajectory always stay in one partition.
    groups = list(itertools.product(c['protocols'], c['depths_m'], c['offsets_xy_m']))
    order = np.random.default_rng(c['seed']).permutation(len(groups))
    partitions = {}
    for rank, group_index in enumerate(order):
        partitions[group_index] = 'train' if rank < int(.7 * len(groups)) else 'val' if rank < int(.85 * len(groups)) else 'test'
    for group, (kind, depth, offset) in enumerate(groups):
        for repeat in range(c['repeats']):
            plans.append(dict(episode_id=len(plans), group_id=group, split=partitions[group],
                              protocol=kind, depth_m=depth, offset_xy_m=offset, repeat=repeat,
                              capability='virtual_rotation' if kind in ('twist', 'roll') else 'xyz'))
    return plans


def trajectory(plan, c):
    """Return phase and [dx,dy,positive inward depth,roll,pitch,yaw]."""
    hz, clear, depth = c['sample_hz'], c['clearance_m'], plan['depth_m']
    out, now = [], np.array([0., 0., -clear, 0., 0., 0.])
    def go(phase, target, seconds):
        nonlocal now
        target = np.array(target, dtype=float)
        steps = max(1, int(np.ceil(seconds * hz)))
        for a in np.linspace(0, 1, steps + 1)[1:]:
            out.append((phase, now + a * (target - now)))
        now = target
    go('baseline', now, c['baseline_s'])
    go('approach', [0, 0, 0, 0, 0, 0], clear / c['normal_speed_mps'])
    go('load', [0, 0, depth, 0, 0, 0], depth / c['normal_speed_mps'])
    go('hold', now, c['hold_s'] * (5 if plan['protocol'] == 'hold' else 1))
    kind = plan['protocol']
    if kind.startswith(('shear', 'slip')):
        distance = c['slip_scan_m'] if kind.startswith('slip') else c['slide_m']
        axis = 0 if kind.endswith('x') else 1
        for phase, pos in [('slide_forward', distance), ('slide_reverse', -distance), ('slide_return', 0)]:
            target = now.copy(); target[axis] = pos
            go(phase, target, abs(target[axis] - now[axis]) / c['slide_speed_mps'])
    if kind in ('twist', 'roll'):
        axis, angle = (5, c['twist_rad']) if kind == 'twist' else (3, c['roll_rad'])
        for pos in (angle, -angle, 0):
            target = now.copy(); target[axis] = pos
            go(kind, target, .5)
    go('unload', [0, 0, 0, 0, 0, 0], depth / c['normal_speed_mps'])
    go('retract', [0, 0, -clear, 0, 0, 0], clear / c['normal_speed_mps'])
    go('release', now, c['release_s'])
    return out
