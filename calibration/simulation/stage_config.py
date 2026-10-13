"""Offline coordinate/limit contract. Never import a ROS node or hardware SDK."""
import itertools
import os
from pathlib import Path

import numpy as np
import yaml

from geometry import ROOT, fingerprint

COORDINATES = ROOT / 'calibration/config/坐标转换.yaml'
# Explicit CLI paths still take precedence; otherwise follow the active checkout.
ROS_WORKSPACE = Path(os.environ.get('ROS_WS_ROOT') or
                     os.environ.get('ROS_TACTILE_WORKSPACE') or
                     ROOT.parent / 'ros_ws').expanduser().resolve()
PARAMS_RELATIVE = 'src/py_force_ws/py_force_control/config/params.yaml'


class StageCoordinates:
    """Configured machine mm <-> nominal Tag metres; display point = nominal tip.

    This assumption is saved in every session. It is not a calibrated TCP claim.
    """
    def __init__(self, coordinates=COORDINATES, ros_workspace=ROS_WORKSPACE):
        self.coordinate_path = Path(coordinates).resolve()
        self.params_path = (Path(ros_workspace) / PARAMS_RELATIVE).resolve()
        coordinate = yaml.safe_load(self.coordinate_path.read_text())
        axes = coordinate['轴向关系']
        expected = {'标定台X正向': 'TagY正向', '标定台Y正向': 'TagX负向',
                    '标定台Z正向': 'TagZ正向'}
        if axes != expected:
            raise ValueError('Unsupported axis convention; update the transform explicitly')
        self.rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.tag_origin_machine_mm = np.array([
            coordinate['Tag原点_标定台系'][axis] for axis in ('x', 'y', 'z')], dtype=float)
        if not np.isfinite(self.tag_origin_machine_mm).all():
            raise ValueError('Non-finite Tag origin')
        self.T_tag_machine_m = np.eye(4)
        self.T_tag_machine_m[:3, :3] = self.rotation
        self.T_tag_machine_m[:3, 3] = -self.rotation @ self.tag_origin_machine_mm / 1000
        params = yaml.safe_load(self.params_path.read_text())['/**']['ros__parameters']
        # Whitelist only motion/limit values; never copy serial/network/SDK settings.
        keys = [f'{axis}_limit_{side}_mm' for axis in 'xyz' for side in ('lower', 'upper')]
        keys += ['pulses_per_mm', 'axes_share_rs485_bus', 'allow_parallel_xy_motion',
                 'watchdog_threshold_n', 'camera_area_x_min_mm', 'camera_area_x_max_mm',
                 'camera_area_y_min_mm', 'camera_area_y_max_mm',
                 'turntable_1_limit_enabled', 'turntable_1_limit_lower_deg',
                 'turntable_1_limit_upper_deg', 'turntable_2_limit_enabled',
                 'turntable_2_limit_lower_deg', 'turntable_2_limit_upper_deg']
        self.parameters = {key: params[key] for key in keys if key in params}
        self.bounds_mm = np.array([[params[f'{axis}_limit_lower_mm'],
                                    params[f'{axis}_limit_upper_mm']] for axis in 'xyz'], dtype=float)
        if not np.isfinite(self.bounds_mm).all() or np.any(self.bounds_mm[:, 0] >= self.bounds_mm[:, 1]):
            raise ValueError('Invalid configured machine limits')
        self.workspace_corners_tag_m = self.machine_to_tag(
            np.array(list(itertools.product(*self.bounds_mm))))
        self.tag_bounds_m = np.stack((self.workspace_corners_tag_m.min(0),
                                      self.workspace_corners_tag_m.max(0)), axis=1)
        anchor = coordinate['安装孔坐标']
        machine = np.array([anchor['标定台系'][a] for a in 'xyz'])
        tag = np.array([anchor['Tag系'][a] for a in 'xyz']) / 1000
        if not np.allclose(self.machine_to_tag(machine), tag, atol=1e-9):
            raise ValueError('Mount-hole anchor disagrees with Tag transform')

    def machine_to_tag(self, machine_mm):
        value = np.asarray(machine_mm, dtype=float)
        return (value - self.tag_origin_machine_mm) @ self.rotation.T / 1000

    def tag_to_machine(self, tag_m):
        return np.asarray(tag_m, dtype=float) @ self.rotation * 1000 + self.tag_origin_machine_mm

    def faults(self, machine_mm):
        value = np.asarray(machine_mm, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            return ['non_finite_or_invalid_machine_position']
        return [f'machine_{a}_limit:{v:.6f}mm' for a, v, b in zip('xyz', value, self.bounds_mm)
                if not b[0] - 1e-9 <= v <= b[1] + 1e-9]

    def require(self, machine_mm):
        faults = self.faults(machine_mm)
        if faults:
            raise ValueError('; '.join(faults))

    def snapshot(self):
        return dict(frame='tag_center', machine_units='mm', simulation_units='m',
                    display_point='nominal_working_tip; actual TCP/extrinsics unverified',
                    source='ROS params.yaml configured limits; live node parameters not queried',
                    T_tag_machine_m=self.T_tag_machine_m.tolist(),
                    tag_origin_machine_mm=self.tag_origin_machine_mm.tolist(),
                    axis_bounds_mm=self.bounds_mm.tolist(), tag_bounds_m=self.tag_bounds_m.tolist(),
                    parameters=self.parameters,
                    sources=fingerprint({self.coordinate_path, self.params_path}),
                    historical_platform_z_used=False, extra_180deg_rotation_applied=False,
                    turntables_simulated=False)


def configure_tag_bench(config, coordinates):
    """Opt in to the current asset; keep legacy collector configs unchanged."""
    config = dict(config)
    config['hand_asset'] = 'revo21_touch_tag_m6x16'
    config['mesh_visuals_only'] = True
    config['excluded_hand_links'] = ['connection_flange', 'press_head', 'press_head_tip']
    # The legacy model bounds check addresses the flange, while ROS addresses the
    # nominal tip. The runtime checks tip limits independently on every frame.
    from geometry import HEADS, load_assembly
    frames, _, _ = load_assembly(HEADS[config['head']])
    tip = frames['press_head_tip'][:3, 3]
    config['axis_bounds_m'] = (coordinates.tag_bounds_m - tip[:, None]).tolist()
    if set(config['protocols']) & {'twist', 'roll'}:
        raise ValueError('Rotational protocols need calibrated turntable axes; use XYZ protocols')
    return config
