"""Read-only URDF geometry and a reproducible virtual, top-down sensor grid."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / 'calibration/assets/revo3_breadboard_mount/generated_calibration_v4/urdf/revo3_right_calibration_lower_mount_v4.urdf'
TOUCH_HAND = ROOT / 'calibration/assets/revo3_breadboard_mount/generated_calibration_v4_touch/urdf/revo21_touch_calibration_lower_mount_v4.urdf'
HEADS = {name: ROOT / f'calibration/assets/head/{name}_press_head_with_flange.urdf'
         for name in ('spherical', 'triangle', 'conical')}


def hand_asset(cfg):
    """Missing version means a historical V4 recording, never silently retarget it."""
    version = cfg.get('hand_asset', 'revo3_v4')
    return {'revo3_v4': HAND, 'revo21_touch_v4': TOUCH_HAND}[version]


def sensor_link(cfg):
    if cfg.get('hand_asset', 'revo3_v4') == 'revo21_touch_v4':
        stem = {'middle': 'mid', 'little': 'pinky'}.get(cfg['finger'], cfg['finger'])
        suffix = 'rubber_link' if cfg['segment'] == 'DIP' else 'touch_link'
        return f'right_{stem}{cfg["segment"].lower()}_roll_{suffix}'
    return f'right_{cfg["finger"]}_{cfg["segment"]}_' + ('rubber_link' if cfg['segment'] == 'DIP' else 'touch_link')


def transform(origin):
    out = np.eye(4)
    if origin is not None:
        out[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        out[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return out


def load_assembly(path, joint_positions=None):
    """Preserve source frames/scales; hold hand at named URDF joint positions."""
    joint_positions = joint_positions or {}
    tree = ET.parse(path).getroot()
    joints = list(tree.findall('joint'))
    children = {j.find('child').get('link') for j in joints}
    roots = {l.get('name') for l in tree.findall('link')} - children
    if len(roots) != 1:
        raise ValueError(f'Expected one URDF root: {roots}')
    movable = {j.get('name') for j in joints if j.get('type') != 'fixed'}
    if set(joint_positions) - movable:
        raise ValueError('Unknown or fixed joint in joint_positions_rad')
    frames = {roots.pop(): np.eye(4)}
    while joints:
        progress = False
        for j in joints[:]:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent not in frames:
                continue
            local = transform(j.find('origin'))
            if j.get('type') != 'fixed':
                q = float(joint_positions.get(j.get('name'), 0.0))
                limit = j.find('limit')
                if limit is not None and not float(limit.get('lower', '-inf')) <= q <= float(limit.get('upper', 'inf')):
                    raise ValueError(f'Joint outside URDF limits: {j.get("name")}={q}')
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                motion = np.eye(4)
                if j.get('type') == 'prismatic':
                    motion[:3, 3] = q * axis
                else:
                    motion[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
                local = local @ motion
            frames[child] = frames[parent] @ local
            joints.remove(j)
            progress = True
        if not progress:
            raise ValueError('Disconnected or cyclic URDF')
    meshes, paths = {}, {Path(path).resolve()}
    for link in tree.findall('link'):
        name = link.get('name')
        parts = []
        for visual in link.findall('visual'):
            spec = visual.find('geometry/mesh')
            if spec is None:
                raise ValueError('This asset loader expects mesh visuals')
            mesh_path = (Path(path).parent / spec.get('filename')).resolve()
            paths.add(mesh_path)
            mesh = trimesh.load(mesh_path, force='mesh', process=True)
            mesh.apply_scale(np.fromstring(spec.get('scale', '1 1 1'), sep=' '))
            mesh.apply_transform(frames[name] @ transform(visual.find('origin')))
            parts.append(mesh)
        if parts:
            meshes[name] = trimesh.util.concatenate(parts)
    return frames, meshes, paths


def first_hit(mesh, origins, directions):
    """First forward triangle intersection; misses have infinite distance."""
    xyz, rays, triangles = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    distance = np.full(len(origins), np.inf)
    faces = np.full(len(origins), -1, dtype=np.int64)
    if len(rays):
        d = np.einsum('ij,ij->i', xyz - origins[rays], directions[rays])
        for i in np.argsort(d)[::-1]:
            if d[i] >= -1e-10:
                distance[rays[i]], faces[rays[i]] = max(0, d[i]), triangles[i]
    return distance, faces


def sensor_grid(mesh, height, width, margin=0.001):
    """Virtual orthographic UV, NOT the existing real camera calibration."""
    lo, hi = mesh.bounds
    xs = np.linspace(lo[0] + margin, hi[0] - margin, width)
    ys = np.linspace(lo[1] + margin, hi[1] - margin, height)
    x, y = np.meshgrid(xs, ys)
    origins = np.stack((x, y, np.full_like(x, hi[2] + 0.01)), -1)
    directions = np.tile([0., 0., -1.], (height * width, 1))
    d, f = first_hit(mesh, origins.reshape(-1, 3), directions)
    valid = np.isfinite(d)
    points = origins.reshape(-1, 3).copy()
    points[valid] += directions[valid] * d[valid, None]
    normals = np.tile([0., 0., 1.], (len(d), 1))
    normals[valid] = mesh.face_normals[f[valid]]
    valid &= normals[:, 2] > 0.35
    # Surface area of a height-field cell, not projected XY area.
    area = (xs[1] - xs[0]) * (ys[1] - ys[0]) / np.maximum(normals[:, 2], 0.35)
    area[~valid] = 0
    if valid.sum() < height * width * 0.3:
        raise ValueError('Too little upward-visible tactile surface; change the fixture pose')
    return points.reshape(height, width, 3), normals.reshape(height, width, 3), valid.reshape(height, width), area.reshape(height, width)


def fingerprint(paths):
    return {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
