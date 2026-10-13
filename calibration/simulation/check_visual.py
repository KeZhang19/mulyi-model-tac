#!/usr/bin/env python3
"""Bounded CPU regression for coordinates, failures, pause and replay isolation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stage_config import StageCoordinates
from visual_session import VisualSession
from protocol import episode_plans

CONFIG = Path(__file__).resolve().parents[1] / 'config/simulation/visual_config.json'


class VisualChecks(unittest.TestCase):
    def test_coordinate_anchors_and_limits(self):
        c = StageCoordinates()
        np.testing.assert_allclose(c.machine_to_tag([-267.5, -14.5, -308.2]), 0, atol=1e-12)
        np.testing.assert_allclose(c.machine_to_tag([-232.5, 137.5, -343.7]), [-.152, .035, -.0355])
        np.testing.assert_allclose(c.tag_to_machine([0, 0, .0005]), [-267.5, -14.5, -307.7])
        np.testing.assert_allclose(c.bounds_mm, [[-340, 0], [0, 170], [-310, 0]])
        random = np.random.default_rng(14).uniform(c.bounds_mm[:, 0], c.bounds_mm[:, 1], (100, 3))
        np.testing.assert_allclose(c.tag_to_machine(c.machine_to_tag(random)), random, atol=1e-10)
        for axis in range(3):
            value = c.bounds_mm.mean(1)
            for sign in (0, 1):
                value[axis] = c.bounds_mm[axis, sign]
                c.require(value)
                value[axis] += -.001 if sign == 0 else .001
                self.assertTrue(c.faults(value))
        self.assertTrue(c.faults([np.nan, 1, 2]))

    def test_pause_replay_stop_and_force_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads(CONFIG.read_text())
            cfg['max_model_force_n'] = 1e-6
            s = VisualSession(cfg, StageCoordinates(), Path(directory) / 'session')
            before = s.machine_mm.copy()
            with self.assertRaises(ValueError):
                s.move([100, 0, 0])
            np.testing.assert_array_equal(s.machine_mm, before)
            invalid = dict(episode_plans(s.config)[0], depth_m=-.001)
            with self.assertRaises(ValueError):
                s.start(invalid)
            self.assertFalse(s.recording)
            s.start(episode_plans(s.config)[0])
            s.paused = True
            count = len(s.queue)
            s.step()
            self.assertEqual(len(s.queue), count)
            self.assertEqual(len(s.rows), 0)
            s.paused = False
            while s.queue:
                s.step()
            self.assertIn('model_force_limit', s.reports[0]['faults'])
            self.assertFalse(s.reports[0]['completed'])
            self.assertGreater(s.last_arrays['total_normal_load_n'][-1], cfg['max_model_force_n'])
            samples = len(s.rows)
            s.replay(0)
            self.assertEqual(len(s.rows), samples)
            s.start(episode_plans(s.config)[0])
            s.step()
            s.stop('test_interrupted')
            self.assertIn('test_interrupted', s.reports[1]['faults'])
            self.assertEqual(s.reports[1]['frames'], 1)
            s.close()
            manifest = json.loads((s.output / 'manifest.json').read_text())
            self.assertFalse(manifest['complete'])
            with self.assertRaises(FileExistsError):
                VisualSession(cfg, StageCoordinates(), s.output)


if __name__ == '__main__':
    unittest.main()
