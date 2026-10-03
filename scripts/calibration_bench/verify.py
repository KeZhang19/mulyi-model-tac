#!/usr/bin/env python3
"""Verify collection completeness and units; never certify physical accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def verify_run(folder):
    c=json.loads((folder/'config.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    plan=json.loads((folder/'plan.json').read_text())
    checks=[]
    def check(name, condition):
        checks.append(dict(name=name,passed=bool(condition)))
    check('manifest_complete', manifest.get('complete'))
    groups={}
    for p in plan['episodes']:
        groups.setdefault(p['group_id'],set()).add(p['split'])
    check('no_group_split_leakage',all(len(v)==1 for v in groups.values()))
    with np.load(folder/'layout.npz',allow_pickle=False) as layout:
        area=layout['taxel_area_m2'];valid=layout['surface_valid'];normals=layout['surface_normals_w']
    for p in plan['episodes']:
        name=f'episode_{p["episode_id"]:05d}'
        if not (folder/(name+'.npz')).is_file():
            check(name+'.exists',False);continue
        meta=json.loads((folder/(name+'.json')).read_text())
        with np.load(folder/(name+'.npz'),allow_pickle=False) as a:
            count=len(a['time_s']);depth=a['depth_m'];pressure=a['pressure_pa'];force=a['taxel_force_n']
            check(name+'.finite',all(np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind in 'fiu'))
            check(name+'.complete',meta['completed'] and count==meta['planned_frames'])
            check(name+'.synchronized_shapes',depth.shape==(count,*c['grid_hw']) and a['rgb'].shape==(*depth.shape,3)
                  and a['marker'].shape==(count,np.prod(c['marker_rows_cols']),5))
            check(name+'.clock',np.allclose(np.diff(a['time_s']),1/c['sample_hz'],atol=1e-12)
                  and np.array_equal(a['frame_id'],np.arange(count)))
            check(name+'.bounds',depth.min()>=0 and depth.max()<=c['max_depth_m']+1e-8 and pressure.min()>=0)
            check(name+'.pressure_area_units',np.allclose(force,pressure*area,atol=1e-8))
            check(name+'.normal_load_sum',np.allclose(force.sum((1,2)),a['total_normal_load_n'],atol=1e-6))
            expected=-(force[...,None]*normals).sum((1,2))
            check(name+'.force_frame_sign',np.allclose(expected,a['force_on_sensor_w_n'],atol=1e-6))
            check(name+'.invalid_surface_zero',not np.any(depth[:,~valid]) and not np.any(force[:,~valid]))
            idle=np.isin(a['phase'],['baseline','release'])
            check(name+'.zero_before_after',idle.any() and np.max(depth[idle])<1e-8
                  and np.max(np.abs(a['marker'][idle,:,2:4]))<1e-6)
            check(name+'.contact_observed',np.any(a['contact_mask']))
            check(name+'.pose_rotation',np.allclose(np.linalg.det(a['T_world_flange'][:,:3,:3]),1,atol=1e-7))
            if p['protocol']=='normal':
                loading=a['phase']=='load'
                check(name+'.monotonic_normal_load',np.all(np.diff(a['total_normal_load_n'][loading])>=-1e-6))
    return dict(schema='mechanical_calibration_verification_v1',passed=all(c['passed'] for c in checks),
                checks=checks,physical_accuracy_validated=False,collision_clearance_certified=False,
                note='Model consistency and complete collection only; not independent calibration evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args()
    report=verify_run(args.folder)
    (args.folder/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
