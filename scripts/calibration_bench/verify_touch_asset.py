#!/usr/bin/env python3
"""Check source preservation, task frames, exported XYZ and sampled fixture fit."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from geometry import ROOT, HAND, TOUCH_HAND, load_assembly, transform

SOURCE = ROOT / 'assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf'
FIXTURE = HAND.with_name('lower_calibration_v4.urdf')
PACKAGE = ROOT / 'assets/calibration_bench_visuotactile_urdf'
SCENE = ROOT / 'outputs/calibration_bench/visuotactile_v1'
EXPORT = PACKAGE / 'urdf/revo3_calibration_scene_xyz.urdf'


def normalized(element, source):
    element = copy.deepcopy(element)
    for node in element.iter():
        node.text = None
        node.tail = None
        if node.tag == 'mesh':
            node.set('filename', hashlib.sha256((source.parent / node.get('filename')).resolve().read_bytes()).hexdigest())
    return ET.tostring(element)


def verify_touch_asset():
    assembly, exported = ET.parse(TOUCH_HAND).getroot(), ET.parse(EXPORT).getroot()
    preserved = 0
    for path, destination in ((SOURCE, assembly), (FIXTURE, assembly), (TOUCH_HAND, exported)):
        for element in ET.parse(path).getroot():
            selector = element.tag if element.get('name') is None else f"{element.tag}[@name='{element.get('name')}']"
            target = destination.find(selector)
            assert target is not None, element.attrib
            assert normalized(element, path) == normalized(target, TOUCH_HAND if destination is assembly else EXPORT), element.attrib
            preserved += 1
    paths = set()
    for mesh in exported.iter('mesh'):
        path = (EXPORT.parent / mesh.get('filename')).resolve()
        assert path.is_file() and path.is_relative_to(PACKAGE)
        paths.add(path)
    # The old and DV2 wrist origins share all four finger-base datum positions.
    old_xml, source_xml = ET.parse(HAND), ET.parse(SOURCE)
    for stem, finger in (('mid', 'middle'), ('index', 'index'), ('ring', 'ring'), ('pinky', 'little')):
        old = old_xml.find(f"joint[@name='right_{finger}_MPR_joint']/origin")
        new = source_xml.find(f"joint[@name='right_{stem}mcp_yaw_joint']/origin")
        np.testing.assert_allclose(transform(old), transform(new), atol=1e-14)
    mount = transform(assembly.find("joint[@name='lower_mount_to_right_hand']/origin"))
    frames, meshes, _ = load_assembly(TOUCH_HAND)
    source_frames, source_meshes, _ = load_assembly(SOURCE)
    error = max(float(np.abs(frames[n] - mount @ t).max()) for n, t in source_frames.items())
    assert error < 1e-12
    # Validate nonzero joint poses too: a zero-only match cannot establish axes.
    q = {j.get('name'): 0.5 * (float(j.find('limit').get('lower')) + float(j.find('limit').get('upper')))
         for j in source_xml.findall('joint') if j.get('type') == 'revolute'}
    posed, _, _ = load_assembly(TOUCH_HAND, q)
    source_posed, _, _ = load_assembly(SOURCE, q)
    pose_error = max(float(np.abs(posed[n] - mount @ t).max()) for n, t in source_posed.items())
    assert pose_error < 1e-12
    module_path = ROOT / 'source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/direct_calibration.py'
    spec = importlib.util.spec_from_file_location('task_direct_calibration', module_path)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    task_frames = {}
    for finger, stem, alias in zip(task.CALIBRATION_FINGERS, task.RUBBER_STEMS, task.DIRECT_LINKS, strict=True):
        p, r = task.rubber_to_dip_transform(SOURCE, stem)
        t = np.eye(4); t[:3, :3] = r; t[:3, 3] = p
        rubber = f'right_{stem}dip_roll_rubber_link'
        dip = f'right_{stem}dip_roll_link'
        np.testing.assert_allclose(frames[dip] @ t, frames[rubber], atol=1e-12)
        task_frames[finger] = dict(source_surface=rubber, task_body=alias, T_dip_rubber=t.tolist())
    reference = PACKAGE / 'reference'
    reference.mkdir(exist_ok=True)
    layout = SOURCE.parent.parent / 'marker_positions/vitai_4fingers/marker_positions.npz'
    mapped = task.prepare_direct_calibration(layout, SOURCE, reference / 'direct_task')
    with np.load(layout) as a, np.load(mapped) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if not key.endswith(('_points_link_m', '_starts_link_m', '_origin_link_m', '_normals_link', '_directions_link', '_camera_rotation_link')):
                np.testing.assert_array_equal(a[key], b[key])
    (reference / 'task_frame_alignment.json').write_text(json.dumps(dict(
        source=str(module_path.relative_to(ROOT)), source_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        convention='T_dip_rubber maps rubber-frame points into task DIP frame; world mount applies separately',
        fingers=task_frames, active_collection_layout='virtual orthographic; reference layout not active',
        task_usd_mesh_identity_validated=False), indent=2) + '\n')
    # Independently reconstruct the exported XYZ chain and original tool TCP.
    tstage = transform(exported.find("joint[@name='world_to_press_stage']/origin"))
    expected_axes = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(tstage[:3, :3], expected_axes, atol=1e-12)
    np.testing.assert_allclose(np.cross(expected_axes[:, 0], expected_axes[:, 1]), expected_axes[:, 2])
    ttool = transform(exported.find("joint[@name='flange_to_press_head']/origin"))
    ttip = ttool @ transform(exported.find("joint[@name='press_head_to_tip']/origin"))
    basis = np.eye(4)
    basis[:3, :3] = expected_axes
    config = json.loads((SCENE / 'config.json').read_text())
    head_path = ROOT / f'assets/head/{config["head"]}_press_head_with_flange.urdf'
    source_tool_frames, source_tool_meshes, _ = load_assembly(head_path)
    basis[:3, 3] = source_tool_frames['press_head_tip'][:3, 3]
    scene_frames, scene_meshes, _ = load_assembly(EXPORT)
    aligned_links = ('press_stage_base', 'press_x_carriage', 'press_y_carriage',
                     'connection_flange', 'press_head', 'press_head_tip')
    for name in aligned_links:
        np.testing.assert_allclose(scene_frames[name][:3, :3], expected_axes, atol=1e-12)
        np.testing.assert_allclose(scene_frames[name], scene_frames['press_head_tip'], atol=1e-12)
    np.testing.assert_allclose(ttool, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(ttip, np.eye(4), atol=1e-12)
    for axis in 'xyz':
        joint = exported.find(f"joint[@name='press_{axis}_joint']")
        np.testing.assert_allclose(transform(joint.find('origin')), np.eye(4), atol=1e-12)
    geometry_error = 0.
    # Mesh objects, including each collision shape, must retain their world pose.
    source_xml_tool = ET.parse(head_path).getroot()
    with np.load(SCENE / 'episode_00000.npz') as recording:
        original_zero = recording['T_world_flange'][0].copy()
    for link in source_xml_tool.findall('link'):
        name = link.get('name')
        for kind in ('visual', 'collision'):
            for source_node, output_node in zip(link.findall(kind), exported.find(f"link[@name='{name}']").findall(kind), strict=True):
                before = original_zero @ source_tool_frames[name] @ transform(source_node.find('origin'))
                after = scene_frames[name] @ transform(output_node.find('origin'))
                geometry_error = max(geometry_error, float(np.abs(before - after).max()))
                assert source_node.find('geometry/mesh').get('scale') == output_node.find('geometry/mesh').get('scale')
    assert geometry_error < 1e-12
    rows = list(csv.DictReader((PACKAGE / 'press_trajectory.csv').open()))
    with np.load(SCENE / 'episode_00000.npz') as data:
        assert len(rows) == len(data['time_s'])
        trajectory_error = 0.
        for i, row in enumerate(rows):
            t = tstage.copy()
            for axis in 'xyz':
                joint = exported.find(f"joint[@name='press_{axis}_joint']")
                t = t @ transform(joint.find('origin'))
                motion = np.eye(4)
                motion[:3, 3] = float(row[f'press_{axis}_joint']) * np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                t = t @ motion
            expected_flange = data['T_world_flange'][i] @ basis
            expected_tip = expected_flange
            trajectory_error = max(trajectory_error, float(np.abs(t-expected_flange).max()),
                                   float(np.abs(t @ ttip-expected_tip).max()))
        assert trajectory_error < 1e-12
    interference = {}
    for version, geometry in (('touch', meshes), ('legacy', load_assembly(HAND)[1])):
        support = geometry['lower_mount']; findings = []
        for name, mesh in geometry.items():
            if name == 'lower_mount' or np.any(mesh.bounds[1] < support.bounds[0]) or np.any(support.bounds[1] < mesh.bounds[0]):
                continue
            points, _ = trimesh.sample.sample_surface(mesh, 500, seed=42)
            depth = trimesh.proximity.signed_distance(support, points)
            findings.append(dict(link=name, samples_over_0_1mm=int((depth > 0.0001).sum()), max_penetration_m=max(0., float(depth.max()))))
        interference[version] = findings
    initial = json.loads((PACKAGE / 'initial_joint_positions.json').read_text())
    assert len(initial) == 24 and all(v == 0 for v in initial.values())
    report = dict(passed=True, preserved_elements=preserved, links=len(exported.findall('link')),
        joints=len(exported.findall('joint')), mesh_references=len(list(exported.iter('mesh'))), unique_meshes=len(paths),
        movable_joints=len(initial), camera_frames=[n for n in frames if 'camera' in n],
        zero_pose_max_error=error, nonzero_pose_max_error=pose_error, trajectory_frames=len(rows),
        trajectory_max_error=trajectory_error, task_rubber_to_dip_transforms_checked=5,
        aligned_tool_links=list(aligned_links), tool_geometry_pose_max_error=geometry_error,
        shared_origin='press_head_tip', tool_fixed_joint_transforms_identity=True,
        fixture_surface_sample_count_per_link=500, fixture_sample_seed=42, fixture_interference=interference,
        fit_status='REVIEW_REQUIRED', physical_fit_validated=False, dynamics_validated=False)
    (PACKAGE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'fixture_interference'}, indent=2))


if __name__ == '__main__':
    verify_touch_asset()
