"""Shared CPU state machine for the interactive Isaac viewer and offline capture."""
from collections import deque
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from geometry import ROOT, fingerprint
from model import MechanicalCalibrationBench
from protocol import episode_plans, trajectory, validate_config
from stage_config import configure_tag_bench


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class VisualSession:
    def __init__(self, config, coordinates, output):
        self.coordinates = coordinates
        self.config = configure_tag_bench(config, coordinates)
        validate_config(self.config)
        self.bench = MechanicalCalibrationBench(self.config)
        self.zero = self.bench.contact_zero([0, 0])
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.queue = deque()
        self.recording = False
        self.paused = False
        self.rows, self.phases, self.commands = [], [], []
        self.reports = []
        self.notes = set()
        self.message = 'Ready. Configured limits; nominal tip coordinates.'
        self.last_arrays = None
        self.replay_index = None
        self.step_count = 0
        self.closed = False
        self.started = time.monotonic()
        self.machine_mm = self.coordinates.tag_to_machine(self.zero + self.bench.tip +
                                  [0, 0, self.config['clearance_m']])
        self.coordinates.require(self.machine_mm)
        self.data = self.bench.sample(self.matrix(self.machine_mm))
        self.force_history = []
        self.manifest = dict(schema='visual_calibration_tag_v1', status='open', complete=False,
                             started_utc=datetime.now(timezone.utc).isoformat(),
                             frame='world=tag_center; T maps local to Tag; lengths m, machine mm',
                             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                               cwd=ROOT, text=True).strip(),
                             coordinate_contract=coordinates.snapshot(),
                             dynamics='prescribed motion; no PhysX/FEM contact solver',
                             channels='synthetic RGB, normal-ray depth, HydroShear marker, model pressure/force',
                             pressure_calibrated=False, rgb_calibrated=False,
                             hardware_connected=False, training_ready=False,
                             collision_guard='sampled containment every 15 frames; no continuous guarantee',
                             limitations=['actual TCP and extrinsics unverified',
                                          'no real mechanical gantry geometry',
                                          'no calibrated turntable axes',
                                          'open meshes are uncertified',
                                          'slip protocol is not a measured slip label'],
                             source_hashes=fingerprint(set(Path(__file__).parent.glob('*.py')) |
                                 self.bench.paths | {ROOT / 'integrate/curved_hydroshear_adapter.py'}),
                             episodes=[])
        write_json(self.output / 'config.json', self.config)
        write_json(self.output / 'coordinate_contract.json', coordinates.snapshot())
        np.savez_compressed(self.output / 'layout.npz', surface_points_tag_m=self.bench.surface,
                            surface_normals_tag=self.bench.normals, surface_valid=self.bench.valid,
                            taxel_area_m2=self.bench.area, marker_reference_uv_px=self.bench.marker_uv,
                            marker_valid=self.bench.marker_valid)
        self.persist()

    def persist(self):
        self.manifest['episodes'] = self.reports
        write_json(self.output / 'manifest.json', self.manifest)

    def matrix(self, machine_mm):
        matrix = np.eye(4)
        matrix[:3, 3] = self.coordinates.machine_to_tag(machine_mm) - self.bench.tip
        return matrix

    def interpolate(self, start, target, phase, speed_mps):
        """Serialize X/Y/Z. No simultaneous shared-bus XY motion is implied."""
        now = np.array(start, dtype=float)
        out = []
        for axis in range(3):
            distance = target[axis] - now[axis]
            if abs(distance) < 1e-9:
                continue
            count = max(1, int(np.ceil(abs(distance) / (speed_mps * 1000) * self.config['sample_hz'])))
            origin = now.copy()
            for a in np.linspace(0, 1, count + 1)[1:]:
                value = origin.copy()
                value[axis] += a * distance
                out.append((phase, value))
            now[axis] = target[axis]
        return out

    def positioning(self, target):
        if np.allclose(self.machine_mm[:2], target[:2], atol=1e-9, rtol=0):
            return self.interpolate(self.machine_mm, target, 'position_z', self.config['normal_speed_mps'])
        # Traverse above all hand/fixture visual geometry before changing XY.
        safe_tag_z = max(mesh.bounds[1, 2] for mesh in self.bench.hand.values()) + .008
        safe_z = self.coordinates.tag_to_machine([0, 0, safe_tag_z])[2]
        safe_z = max(safe_z, self.machine_mm[2], target[2])
        self.coordinates.require([target[0], target[1], safe_z])
        now = self.machine_mm.copy()
        out = []
        for phase, goal in [('position_retract', [now[0], now[1], safe_z]),
                            ('position_xy', [target[0], target[1], safe_z]),
                            ('position_descend', target)]:
            goal = np.asarray(goal)
            out.extend(self.interpolate(now, goal, phase, self.config['normal_speed_mps']))
            now = goal
        return out

    def move(self, target):
        if self.recording or self.queue:
            raise ValueError('Finish or stop the current motion before positioning')
        self.coordinates.require(target)
        self.queue = deque(self.positioning(np.asarray(target, dtype=float)))
        self.replay_index = None
        self.paused = False
        self.message = 'Positioning: retract, serialized XY, descend.'

    def start(self, plan):
        if self.recording or self.queue:
            raise ValueError('Session is busy')
        if plan['protocol'] not in self.config['protocols']:
            raise ValueError('Protocol is not in this session config')
        if not np.isfinite(plan['depth_m']) or not 0 < plan['depth_m'] <= self.config['max_depth_m']:
            raise ValueError('Requested depth must be positive and within the model depth limit')
        offset = np.asarray(plan['offset_xy_m'])
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError('Offset must be two finite values')
        self.bench.reset()
        self.zero = self.bench.contact_zero(plan['offset_xy_m'])
        waypoints = []
        now = None
        for phase, command in trajectory(plan, self.config):
            pose = self.bench.pose(self.zero, command)
            tip_tag = (pose @ self.bench.tool_frames['press_head_tip'])[:3, 3]
            target = self.coordinates.tag_to_machine(tip_tag)
            self.coordinates.require(target)
            if now is None:
                waypoints += self.positioning(target)
            changed = 0 if now is None else np.count_nonzero(np.abs(target - now) > 1e-9)
            # Following a curved surface couples Z with the scan axis. Serialize
            # these increments, retain every intermediate observation.
            if changed > 1:
                waypoints += self.interpolate(now, target, phase, self.config['slide_speed_mps'])
            else:
                waypoints.append((phase, target.copy()))
            now = target
        self.plan = dict(plan, episode_id=len(self.reports))
        self.planned_frames = len(waypoints)
        self.queue = deque(waypoints)
        self.recording, self.paused = True, False
        self.rows, self.phases, self.commands, self.force_history = [], [], [], []
        self.notes = set()
        self.replay_index = None
        self.message = f'Collecting {plan["protocol"]}; complete load/unload cycle.'
        write_json(self.output / f'episode_{self.plan["episode_id"]:05d}_plan.json',
                   dict(plan=self.plan, planned_frames=self.planned_frames))
        self.manifest['status'] = 'collecting'
        self.persist()

    def step(self):
        if self.paused or not self.queue:
            return False
        phase, target = self.queue.popleft()
        pose = self.matrix(target)
        data = self.bench.sample(pose)
        self.machine_mm, self.data = target.copy(), data
        faults = self.coordinates.faults(target)
        if not all(np.isfinite(value).all() for value in data.values()):
            faults.append('non_finite')
        if data['depth_m'].max() > self.config['max_depth_m'] + 1e-8:
            faults.append('depth_limit')
        if data['total_normal_load_n'] > self.config['max_model_force_n']:
            faults.append('model_force_limit')
        if self.step_count % 15 == 0 or not self.queue:
            guard = self.bench.collision_guard(pose)
            faults += [x for x in guard if x.startswith('non_target_collision:')]
            self.notes.update(x for x in guard if not x.startswith('non_target_collision:'))
        self.step_count += 1
        self.force_history.append(float(data['total_normal_load_n']))
        if self.recording:
            self.rows.append(dict(data, machine_position_mm=target.copy(),
                                  T_tag_tip=data['T_world_tip'].copy(),
                                  T_tag_flange=pose.copy()))
            self.phases.append(phase)
            self.commands.append(target.copy())
        self.message = f'{phase} | {data["total_normal_load_n"]:.4f} N (model)'
        if faults:
            self.queue.clear()
            if self.recording:
                self.finish(faults)
            else:
                self.message = 'Motion fault: ' + '; '.join(faults)
                self.manifest.setdefault('manual_motion_faults', []).append(
                    dict(faults=faults, machine_position_mm=target.tolist()))
                self.persist()
        elif not self.queue and self.recording:
            self.finish([])
        return True

    def finish(self, faults):
        self.recording = False
        count = len(self.rows)
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]} if count else {}
        arrays.update(time_s=np.arange(count) / self.config['sample_hz'], frame_id=np.arange(count),
                      phase=np.asarray(self.phases), episode_id=np.full(count, self.plan['episode_id']),
                      command_machine_position_mm=np.asarray(self.commands).reshape(-1, 3),
                      contact_zero_flange_tag_m=self.zero,
                      flange_velocity_tag_mps=np.vstack((np.zeros((1, 3)),
                        np.diff(arrays['T_tag_flange'][:, :3, 3], axis=0) * self.config['sample_hz'])) if count else np.empty((0, 3)))
        name = f'episode_{self.plan["episode_id"]:05d}'
        with (self.output / (name + '.npz')).open('xb') as file:
            np.savez_compressed(file, **arrays)
        report = dict(**self.plan, frames=count, planned_frames=self.planned_frames,
                      completed=count == self.planned_frames and not faults, faults=sorted(set(faults)),
                      geometry_notes=sorted(self.notes), geometry_clearance_certified=False,
                      peak_model_load_n=max(self.force_history, default=0),
                      max_depth_m=float(arrays['depth_m'].max()) if count else 0,
                      contact_frames=int(np.any(arrays['contact_mask'], axis=(1, 2)).sum()) if count else 0)
        write_json(self.output / (name + '.json'), report)
        self.reports.append(report)
        self.last_arrays = arrays
        self.manifest['status'] = 'ready'
        self.persist()
        self.message = ('Saved complete episode: ' if report['completed'] else 'Saved failed/interrupted episode: ') + name

    def stop(self, reason='user_stopped'):
        self.queue.clear()
        self.paused = False
        if self.recording:
            self.finish([reason])
        else:
            self.message = 'Stopped. Use positioning to retract.'

    def replay(self, index):
        if self.recording or self.queue:
            raise ValueError('Stop motion before replay')
        if not self.last_arrays or not len(self.last_arrays['frame_id']):
            raise ValueError('No saved frames in this session')
        index = int(np.clip(index, 0, len(self.last_arrays['frame_id']) - 1))
        self.replay_index = index
        self.data = {key: self.last_arrays[key][index] for key in self.rows[0]}
        self.machine_mm = self.last_arrays['machine_position_mm'][index].copy()
        self.message = f'Replay {index}/{len(self.last_arrays["frame_id"]) - 1}: {self.last_arrays["phase"][index]}'

    def close(self, error=None):
        if self.closed:
            return
        if self.recording:
            self.stop('application_error' if error else 'application_closed')
        self.manifest['status'] = 'error' if error else 'closed'
        self.manifest['wall_time_s'] = time.monotonic() - self.started
        self.manifest['complete'] = bool(self.reports) and all(r['completed'] for r in self.reports) and not error
        if error:
            self.manifest['error'] = str(error)
        self.persist()
        self.closed = True


def finite_capture(session, episodes):
    for plan in episode_plans(session.config)[:episodes]:
        session.start(plan)
        while session.queue:
            session.step()
        if not session.reports[-1]['completed']:
            break
    return all(r['completed'] for r in session.reports)
