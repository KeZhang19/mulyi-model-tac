#!/usr/bin/env python3
"""Assemble the task's DV2 visuotactile hand on the unchanged V4 fixture."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'assets/revo21_right_touch'
V4 = ROOT / 'calibration/assets/revo3_breadboard_mount/generated_calibration_v4'
OUTPUT = ROOT / 'calibration/assets/revo3_breadboard_mount/generated_calibration_v4_touch'


def build_touch_fixture(output):
    hand_path = SOURCE / 'urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf'
    fixture_path = V4 / 'urdf/lower_calibration_v4.urdf'
    old_path = V4 / 'urdf/revo3_right_calibration_lower_mount_v4.urdf'
    hand = ET.parse(hand_path).getroot()
    fixture = ET.parse(fixture_path).getroot()
    mount = copy.deepcopy(ET.parse(old_path).find("joint[@name='lower_mount_to_right_hand']"))
    mount.find('child').set('link', 'base_link')
    # Both CAD exports share the wrist origin and the four MCP-yaw joint datums.
    # Preserve DV2 axes/limits/inertias, rather than translating old joint values.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'urdf').mkdir()
    assembled = ET.Element('robot', name='revo21_touch_calibration_lower_mount_v4')
    hashes = {}
    for path, source, category, mesh_root in (
        (fixture_path, fixture, 'mount', V4 / 'meshes'),
        (hand_path, hand, 'hand', SOURCE),
    ):
        for element in source:
            copied = copy.deepcopy(element)
            for mesh in copied.iter('mesh'):
                original = (path.parent / mesh.get('filename')).resolve()
                destination = output / 'meshes' / category / original.relative_to(mesh_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(original, destination)
                mesh.set('filename', Path(os.path.relpath(destination, output / 'urdf')).as_posix())
                hashes[str(destination.relative_to(output))] = hashlib.sha256(original.read_bytes()).hexdigest()
            assembled.append(copied)
    assembled.append(mount)
    ET.indent(assembled, space='  ')
    target = output / 'urdf/revo21_touch_calibration_lower_mount_v4.urdf'
    ET.ElementTree(assembled).write(target, encoding='utf-8', xml_declaration=True)
    # Carry the original camera/marker references alongside the geometric asset.
    shutil.copytree(SOURCE / 'marker_positions', output / 'reference/marker_positions')
    shutil.copytree(SOURCE / 'pressure_taxels', output / 'reference/pressure_taxels')
    stems = {'middle': 'mid', 'index': 'index', 'ring': 'ring', 'little': 'pinky', 'thumb': 'thumb'}
    cameras = dict(middle='right_middip_camera_link', index='right_index_camera',
                   ring='right_ring_camera_link', little='right_pinky_camera_link', thumb='right_thumbdip_camera_link')
    alignment = dict(
        schema='calibration_dv2_asset_alignment_v1',
        source_hand=str(hand_path.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (hand_path, fixture_path, old_path)},
        mesh_sha256=hashes,
        wrist_mount=mount.find('origin').attrib,
        source_elements_preserved_except_mesh_paths=True,
        task_references=[
            'tacmap/revo21_dv2_tactile_env_cfg.py',
            'source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/brainco_hand_visuotactile_env_cfg.py',
            'source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/robot_contract.py'],
        fingers={f: dict(surface_link=f'right_{s}dip_roll_rubber_link', camera_link=cameras[f],
                        task_body_alias=f'right_{f}_DIP_Link') for f, s in stems.items()},
        task_joint_values_interchangeable=False,
        task_note='DV2 source geometry/camera frames retained. Task USD names are aliases, not proof of identical axes or joint limits.',
        marker_note='Original reference files retained with original metadata; not activated as the virtual orthographic collection layout.',
        physical_fit_validated=False,
    )
    (output / 'alignment.json').write_text(json.dumps(alignment, indent=2, ensure_ascii=False) + '\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    print(build_touch_fixture(parser.parse_args().output.resolve()))
