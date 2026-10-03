#!/usr/bin/env python3
"""Standalone mechanical calibration; --plan-only/--help never start Isaac."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from protocol import episode_plans, trajectory, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--episodes', type=int, help='Bounded prefix, NOT a full campaign')
    parser.add_argument('--head', choices=('spherical', 'triangle', 'conical'))
    parser.add_argument('--finger', choices=('index', 'middle', 'ring', 'little', 'thumb'))
    parser.add_argument('--segment', choices=('DIP', 'MCP', 'PIP'))
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--no-scene', action='store_true')
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    for key in ('head','finger','segment','protocols'):
        if getattr(args,key) is not None:
            c[key]=getattr(args,key)
    validate_config(c)
    plans = episode_plans(c)
    if args.episodes is not None:
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        plans=plans[:args.episodes]
    summary=dict(episodes=len(plans), frames=sum(len(trajectory(p,c)) for p in plans),
                 finger=c['finger'], segment=c['segment'], head=c['head'], protocols=c['protocols'],
                 grid_hw=c['grid_hw'], limited_prefix=args.episodes is not None,
                 simulation='prescribed_motion_elastomer_model', physical_calibration=False)
    if args.plan_only:
        print(json.dumps(summary,indent=2)); return 0
    if args.output is None:
        parser.error('--output is required; existing directories are refused')
    args.output.mkdir(parents=True,exist_ok=False)
    def write(name,obj):
        (args.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write('config.json',c);write('plan.json',dict(summary=summary,episodes=plans))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from geometry import ROOT, fingerprint
    from model import MechanicalCalibrationBench
    from scene import export_scene, plot_episode
    started=time.time(); reports=[]; bench=None
    manifest=dict(schema='mechanical_calibration_v1', status='running', complete=False,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=sys.argv, frames='world=lower_mount; SI; T maps local to world; quaternion=wxyz',
        depth_backend='actual_mesh_normal_ray_L1_model_reference',
        pressure_backend='provisional_shell_fraction_power_law; taxel_force=Pa*surface_area_m2',
        marker_backend='integrate/curved_hydroshear_adapter.py; persistent tool surface samples',
        rgb_backend='uncalibrated_three_light_Lambertian; synthetic_only',
        sensor_layout='virtual_top_down_orthographic; NOT existing real-camera calibration',
        dynamics='prescribed_motion; no PhysX force or FEM solver',
        limitations=['No physical ground truth','No hardware/SDK connection','Not encoder checkpoint compatible',
                     'No viscous relaxation model','Slip protocol name is not a measured slip label',
                     'Sampled geometry guard is not continuous collision certification'],
        unavailable_channels=['real_rgb','measured_wrench','physx_contact_force','independent_dense_pressure_gt'],
        source_hashes=fingerprint(set(Path(__file__).parent.glob('*.py')) | {args.config.resolve(), ROOT/'integrate/curved_hydroshear_adapter.py'}))
    write('manifest.json',manifest)
    try:
        bench=MechanicalCalibrationBench(c)
        manifest['asset_hashes']=fingerprint(bench.paths)
        import xml.etree.ElementTree as ET
        manifest['hand_asset']=str(bench.hand_urdf.relative_to(ROOT))
        manifest['sensor_link']=bench.link
        joint_names=[j.get('name') for j in ET.parse(bench.hand_urdf).findall('joint') if j.get('type')!='fixed']
        joint_pos=np.asarray([c['joint_positions_rad'].get(j,0.0) for j in joint_names])
        manifest['hand_joint_order']=joint_names
        write('frames.json',{k:v.tolist() for k,v in bench.frames.items()})
        np.savez_compressed(args.output/'layout.npz', surface_points_w_m=bench.surface,
            surface_normals_w=bench.normals, surface_valid=bench.valid, taxel_area_m2=bench.area,
            marker_reference_uv_px=bench.marker_uv, marker_valid=bench.marker_valid,
            T_world_sensor_link=bench.frames[bench.link], T_flange_tip=bench.tool_frames['press_head_tip'])
        for p in plans:
            bench.reset(); rows=[]; phases=[]; faults=[]; notes=set(); zero=bench.contact_zero(p['offset_xy_m'])
            steps=trajectory(p,c)
            for i,(phase,command) in enumerate(steps):
                matrix=bench.pose(zero,command)
                data=bench.sample(matrix)
                # Keep offending observations before aborting; never trim hard samples.
                rows.append(data);phases.append(phase)
                if not all(np.isfinite(v).all() for v in data.values()):
                    faults.append('non_finite')
                if float(data['depth_m'].max()) > c['max_depth_m'] + 1e-8:
                    faults.append('depth_limit')
                if float(data['total_normal_load_n']) > c['max_model_force_n']:
                    faults.append('model_force_limit')
                if i % 15 == 0 or i == len(steps)-1:
                    collisions=bench.collision_guard(matrix)
                    faults.extend(x for x in collisions if x.startswith('non_target_collision:'))
                    notes.update(x for x in collisions if not x.startswith('non_target_collision:'))
                if faults:
                    break
            arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
            arrays.update(time_s=np.arange(len(rows))/c['sample_hz'], frame_id=np.arange(len(rows)),
                phase=np.asarray(phases), episode_id=np.full(len(rows),p['episode_id']),
                command_xyzrpy=np.asarray([x[1] for x in steps[:len(rows)]]),
                contact_zero_flange_w_m=zero, surface_valid=bench.valid,
                joint_pos_rad=np.tile(joint_pos,(len(rows),1)),joint_vel_rad_s=np.zeros((len(rows),len(joint_names))),
                flange_velocity_w_mps=np.vstack((np.zeros((1,3)),np.diff(np.stack([r['T_world_flange'][:3,3] for r in rows]),axis=0)*c['sample_hz'])))
            name=f'episode_{p["episode_id"]:05d}'
            np.savez_compressed(args.output/(name+'.npz'),**arrays)
            report=dict(**p, completed=len(rows)==len(steps) and not faults, faults=sorted(set(faults)),
                geometry_notes=sorted(notes), frames=len(rows), planned_frames=len(steps),
                contact_frames=int(np.any(arrays['contact_mask'],axis=(1,2)).sum()),
                max_depth_m=float(arrays['depth_m'].max()), peak_model_load_n=float(arrays['total_normal_load_n'].max()),
                collision_guard='sampled_every_15_frames; no continuous guarantee',
                geometry_clearance_certified=False)
            reports.append(report);write(name+'.json',report)
            if len(reports)==1:
                plot_episode(args.output/'overview.png',bench,arrays)
                if not args.no_scene:
                    export_scene(args.output/'scene.usda',bench,arrays['T_world_flange'])
            print(json.dumps(report,ensure_ascii=False),flush=True)
        manifest.update(status='complete' if all(r['completed'] for r in reports) else 'completed_with_failures',
                        complete=all(r['completed'] for r in reports), episodes=reports)
    except Exception as e:
        manifest.update(status='error',error=f'{type(e).__name__}: {e}',episodes=reports)
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest['wall_time_s']=time.time()-started
        write('manifest.json',manifest)
    return 0 if manifest['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
