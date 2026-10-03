#!/usr/bin/env python3
"""Small analytic/regression checks independent of generated experiment output."""
import json
from pathlib import Path
import unittest
import numpy as np
import trimesh
from geometry import first_hit, HEADS, load_assembly
from protocol import episode_plans, trajectory, validate_config


class CalibrationChecks(unittest.TestCase):
    def setUp(self):
        self.c=json.loads(Path(__file__).with_name('config.json').read_text())

    def test_sphere_ray_analytic(self):
        radius=.0035
        mesh=trimesh.creation.icosphere(subdivisions=4,radius=radius)
        origins=np.array([[0,0,-.02],[.001,0,-.02],[.01,0,-.02]])
        d,_=first_hit(mesh,origins,np.tile([0,0,1.],(3,1)))
        expected=.02-np.sqrt(radius**2-origins[:2,0]**2)
        np.testing.assert_allclose(d[:2],expected,atol=8e-6)
        self.assertTrue(np.isinf(d[2]))

    def test_repetitions_do_not_leak(self):
        plans=episode_plans(self.c);groups={}
        for p in plans:groups.setdefault(p['group_id'],set()).add(p['split'])
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        self.assertEqual({p['split'] for p in plans},{'train','val','test'})
        self.assertEqual(plans,episode_plans(self.c))

    def test_complete_protocols_and_speed(self):
        for kind in ('normal','hold','shear_x','shear_y','slip_x','slip_y','twist','roll'):
            plan={'protocol':kind,'depth_m':.0005}
            rows=trajectory(plan,self.c);phases=[p for p,_ in rows];a=np.array([v for _,v in rows])
            for phase in ('baseline','approach','load','hold','unload','retract','release'):self.assertIn(phase,phases)
            np.testing.assert_allclose(a[0],a[-1])
            speed=abs(np.diff(a[:,:3],axis=0))*self.c['sample_hz']
            self.assertLessEqual(speed.max(),max(self.c['normal_speed_mps'],self.c['slide_speed_mps'])+1e-10)

    def test_reject_invalid_depth(self):
        self.c['depths_m']=[self.c['max_depth_m']*2]
        with self.assertRaises(ValueError):validate_config(self.c)

    def test_tip_matches_source_geometry(self):
        for path in HEADS.values():
            frames,meshes,_=load_assembly(path)
            self.assertAlmostEqual(frames['press_head_tip'][2,3],meshes['press_head'].bounds[0,2],places=7)
            self.assertTrue(meshes['press_head'].is_watertight)


if __name__=='__main__':unittest.main()
