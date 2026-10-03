#!/usr/bin/env python3
"""Export the minimal scene as a portable URDF with an invisible XYZ tool chain."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from geometry import hand_asset, load_assembly, transform

REPO = Path(__file__).resolve().parents[2]
JOINT_NAMES = ('press_x_joint', 'press_y_joint', 'press_z_joint')


def reframe_tool_element(element, source_to_shared):
    """Express every tool link in the same tip frame, preserving physical geometry.

    source_to_shared maps each new shared frame into its original link frame.
    Geometry uses B_link^-1 T; joints use B_parent^-1 T B_child.
    Source assets are not modified; this operates on the export's deep copy.
    """
    if element.tag == 'link':
        basis = source_to_shared[element.get('name')]
        owners = [node for node in element if node.tag in ('visual', 'collision', 'inertial')]
    elif element.tag == 'joint':
        if element.get('type') != 'fixed':
            raise ValueError('Tool frame rebasing requires a fixed tool assembly')
        basis = source_to_shared[element.find('parent').get('link')]
        owners = [element]
    else:
        return
    for owner in owners:
        origin = owner.find('origin')
        matrix = np.linalg.inv(basis) @ transform(origin)
        if element.tag == 'joint':
            matrix = matrix @ source_to_shared[element.find('child').get('link')]
            if not np.allclose(matrix, np.eye(4), atol=1e-12):
                raise ValueError('Tool fixed joint did not collapse to the common tip frame')
            matrix = np.eye(4)
        if origin is None:
            origin = ET.SubElement(owner, 'origin')
        origin.set('xyz', ' '.join(f'{v:.17g}' for v in matrix[:3, 3]))
        origin.set('rpy', ' '.join(f'{v:.17g}' for v in Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz')))


def export_calibration_urdf(scene_dir, output):
    config = json.loads((scene_dir / 'config.json').read_text())
    hand_urdf = hand_asset(config)
    head_urdf = REPO / f'assets/head/{config["head"]}_press_head_with_flange.urdf'
    with np.load(scene_dir / 'episode_00000.npz', allow_pickle=False) as recording:
        matrices = recording['T_world_flange'].copy()
        times = recording['time_s'].copy()
        phases = recording['phase'].copy()
    if not np.isfinite(matrices).all() or not np.allclose(matrices[:, :3, :3], np.eye(3), atol=1e-10):
        raise ValueError('This XYZ URDF export requires fixed, world-aligned tool orientation')
    if any(float(q) != 0 for q in config.get('joint_positions_rad', {}).values()):
        raise ValueError('This current-scene export expects the source hand at URDF zero positions')
    flange_origin = matrices[0, :3, 3]
    # Columns are the new stage axes in the original/world frame: -Y, +X, +Z.
    stage_rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    tool_frames, _, _ = load_assembly(head_urdf)
    shared_frame = np.eye(4)
    shared_frame[:3, :3] = stage_rotation
    shared_frame[:3, 3] = tool_frames['press_head_tip'][:3, 3]
    source_to_shared = {name: np.linalg.inv(frame) @ shared_frame for name, frame in tool_frames.items()}
    origin = (matrices[0] @ shared_frame)[:3, 3]
    # Bounds are specified at the original flange; moving the coordinate origin
    # must not alter the allowed physical motion or the displacement commands.
    world_bounds = np.asarray(config['axis_bounds_m']) - flange_origin[:, None]
    travel_bounds = np.array([[-world_bounds[1, 1], -world_bounds[1, 0]],
                              world_bounds[0], world_bounds[2]])
    positions = (matrices[:, :3, 3] - flange_origin) @ stage_rotation
    if np.any(positions < travel_bounds[:, 0]) or np.any(positions > travel_bounds[:, 1]):
        raise ValueError('Recorded trajectory outside configured simulation bounds')
    output.mkdir(parents=True, exist_ok=False)
    urdf_dir = output / 'urdf'
    urdf_dir.mkdir()
    root = ET.Element('robot', name='revo3_calibration_scene_xyz')
    root.append(ET.Comment('SI units. Kinematic XYZ chain; no physical gantry geometry or calibrated moving-body inertias.'))
    ET.SubElement(root, 'link', name='calibration_world')
    fixed = ET.SubElement(root, 'joint', name='world_to_fixture', type='fixed')
    ET.SubElement(fixed, 'parent', link='calibration_world')
    ET.SubElement(fixed, 'child', link='lower_mount')
    ET.SubElement(fixed, 'origin', xyz='0 0 0', rpy='0 0 0')
    root.append(ET.Comment('EDIT TOOL COORDINATE FRAME HERE: world_to_press_stage origin xyz / rpy.'))
    ET.SubElement(root, 'link', name='press_stage_base')
    mount = ET.SubElement(root, 'joint', name='world_to_press_stage', type='fixed')
    ET.SubElement(mount, 'parent', link='calibration_world')
    ET.SubElement(mount, 'child', link='press_stage_base')
    ET.SubElement(mount, 'origin', xyz=' '.join(f'{x:.15g}' for x in origin), rpy=f'0 0 {-np.pi / 2:.17g}')
    ET.SubElement(root, 'link', name='press_x_carriage')
    ET.SubElement(root, 'link', name='press_y_carriage')
    chain = (('press_stage_base', 'press_x_carriage'), ('press_x_carriage', 'press_y_carriage'),
             ('press_y_carriage', 'connection_flange'))
    root.append(ET.Comment('XYZ effort=20 N and velocity=0.01 m/s are simulation placeholders, not measured machine limits.'))
    for i, (name, (parent, child)) in enumerate(zip(JOINT_NAMES, chain)):
        joint = ET.SubElement(root, 'joint', name=name, type='prismatic')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        # Every carriage and tool link uses the same stage basis. Geometry is
        # compensated below, never by rotating a joint out of this basis.
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(joint, 'axis', xyz=' '.join(str(int(x)) for x in np.eye(3)[i]))
        ET.SubElement(joint, 'limit', lower=f'{travel_bounds[i, 0]:.15g}',
                      upper=f'{travel_bounds[i, 1]:.15g}', effort='20', velocity='0.01')
    assets = {}
    for source_urdf, category, mesh_root in ((hand_urdf, 'hand', hand_urdf.parent.parent / 'meshes'),
                                              (head_urdf, 'head', head_urdf.parent)):
        source = ET.parse(source_urdf).getroot()
        for element in source:
            copied = copy.deepcopy(element)
            if category == 'head':
                reframe_tool_element(copied, source_to_shared)
            for mesh in copied.iter('mesh'):
                original = (source_urdf.parent / mesh.get('filename')).resolve()
                relative = original.relative_to(mesh_root.resolve())
                destination = output / 'meshes' / category / relative
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, destination)
                mesh.set('filename', Path(os.path.relpath(destination, urdf_dir)).as_posix())
                assets[str(destination.relative_to(output))] = dict(
                    source=str(original.relative_to(REPO)), sha256=hashlib.sha256(original.read_bytes()).hexdigest())
            root.append(copied)
    ET.indent(root, space='  ')
    target = urdf_dir / 'revo3_calibration_scene_xyz.urdf'
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    with (output / 'press_trajectory.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('time_s', *JOINT_NAMES, 'phase'))
        for time, q, phase in zip(times, positions, phases):
            writer.writerow((f'{time:.12g}', *(f'{v:.15g}' for v in q), str(phase)))
    initial = {j.get('name'): 0.0 for j in root.findall('joint') if j.get('type') != 'fixed'}
    (output / 'initial_joint_positions.json').write_text(json.dumps(initial, indent=2) + '\n')
    manifest = dict(schema='calibration_scene_urdf_v3', source_scene=str(scene_dir.relative_to(REPO)),
        source_recording_sha256=hashlib.sha256((scene_dir / 'episode_00000.npz').read_bytes()).hexdigest(),
        source_urdf_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (hand_urdf, head_urdf)},
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        urdf=str(target.relative_to(output)), root_link='calibration_world',
        editable_frame_joint='world_to_press_stage', editable_frame_xyz_m=origin.tolist(),
        editable_frame_rpy_rad=[0., 0., -float(np.pi / 2)],
        stage_axes_in_original_frame=stage_rotation.tolist(),
        stage_axis_convention='X_new=-Y_old; Y_new=+X_old; Z_new=+Z_old',
        tool_link_basis_in_source_frame=stage_rotation.tolist(),
        shared_tool_frame='press_head_tip; all fixed tool links have coincident origins and axes',
        T_source_flange_shared_tip=shared_frame.tolist(),
        T_source_link_shared_tip={name: value.tolist() for name, value in source_to_shared.items()},
        tool_frames='All stage/carriage frames coincide at tip at q=0; flange/head/tip remain coincident at any q',
        tool_geometry_compensation='Rebased visual/collision/inertial origins; no compensating motion-joint rotation',
        motion_joint_order=list(JOINT_NAMES), motion_units='m', negative_z_is_press=True,
        bounds_source='Original flange world bounds transformed into stage displacements; physical travel unchanged; NOT mechanical travel limits',
        moving_inertias='unknown; omitted, no physical dynamics claim', assets=assets,
        trajectory_frames=len(times), trajectory_duration_s=float(times[-1]),
        new_geometry_added=False, physical_dynamics_validated=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=REPO / 'outputs/calibration_bench/visuotactile_v1')
    parser.add_argument('--output', type=Path, required=True, help='New self-contained asset directory')
    args = parser.parse_args()
    print(export_calibration_urdf(args.scene_dir.resolve(), args.output.resolve()))
