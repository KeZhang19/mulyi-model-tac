"""CPU-only regression checks; no camera, SDK import, or robot access."""
import copy
from pathlib import Path
import tempfile
import threading
import unittest
from collections import deque

import cv2
import numpy as np

from marker_template import FixedMarkerMatcher, create_marker_template, save_marker_batch
from marker_roi_live import MarkerROISession


class MarkerTemplateTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema": "revo3-marker-roi-v1", "camera_serial": "test-camera",
            "coordinate_space": "raw_rgb_pixels", "image_hw": [480, 640],
            "include_polygons": [[[0, 0], [639, 0], [639, 479], [0, 479]]],
            "exclude_polygons": [],
        }
        self.points = np.array([[20., 20.], [60., 20.], [100., 20.]])
        self.template = create_marker_template(self.points, self.config)
        self.matcher = FixedMarkerMatcher(self.template, self.config)

    def match(self, points):
        return self.matcher.match(points, "test-camera", [480, 640])

    def test_reorder_extra_missing_recovery(self):
        moved = self.points + [2, 1]
        first = self.match(moved[[2, 0, 1]])
        np.testing.assert_allclose(first["marker"][:3, :4], np.c_[self.points, np.tile([2, 1], (3, 1))])
        second = self.match(np.vstack([moved[[2, 0]], [500, 400]]))
        np.testing.assert_array_equal(second["marker"][:, 4], [1, 0, 1] + [0] * 97)
        self.assertEqual(second["extra"], 1)
        self.assertTrue((second["marker"][1] == 0).all())
        np.testing.assert_array_equal(self.match(moved)["marker"], first["marker"])

    def test_ambiguity_is_invalid_not_arbitrary_choice(self):
        result = self.match(np.vstack([self.points, self.points[0] + [1, 0]]))
        self.assertEqual(result["ambiguous"], 1)
        self.assertEqual(result["matched_candidate_indices"][0], -1)
        self.assertEqual(result["model_valid"], 2)

    def test_large_motion_and_empty_frame(self):
        self.assertEqual(self.match(self.points + [13, 0])["matched"], 0)
        self.assertTrue((self.match([])["marker"] == 0).all())
        self.assertEqual(self.match([[np.nan, 20]])["matched"], 0)

    def test_more_than_100_selection_is_fixed_and_all_references_retained(self):
        points = np.array([[20 + x * 30, 20 + y * 30] for y in range(11) for x in range(11)])
        template = create_marker_template(points, self.config)
        matcher = FixedMarkerMatcher(template, self.config)
        self.assertEqual(len(template["reference_xy"]), 121)
        self.assertEqual(len(set(template["output_ids"])), 100)
        a = matcher.match(points, "test-camera", [480, 640])
        b = matcher.match(points[::-1], "test-camera", [480, 640])
        np.testing.assert_array_equal(a["marker"], b["marker"])
        self.assertEqual(a["matched"], 121)
        self.assertEqual(a["model_valid"], 100)

    def test_configuration_and_template_mismatch(self):
        for serial, shape in [("other-camera", [480, 640]), ("test-camera", [240, 320])]:
            with self.assertRaises(ValueError):
                self.matcher.match(self.points, serial, shape)
        for key, value in [("output_ids", [0] * 100), ("gate_radius_px", [50] * 3)]:
            changed = copy.deepcopy(self.template)
            changed[key] = value
            with self.assertRaises(ValueError):
                FixedMarkerMatcher(changed, self.config)
        with self.assertRaises(ValueError):
            create_marker_template([[20, 20], [20, 20]], self.config)
        other_roi = copy.deepcopy(self.config)
        other_roi["exclude_polygons"] = [[[300, 300], [320, 300], [320, 320]]]
        with self.assertRaises(ValueError):
            FixedMarkerMatcher(self.template, other_roi)

    def test_gates_cannot_assign_one_candidate_to_two_references(self):
        template = create_marker_template([[20, 20], [30, 20]], self.config)
        matcher = FixedMarkerMatcher(template, self.config)
        result = matcher.match([[25, 20]], "test-camera", [480, 640])
        self.assertEqual(result["matched"], 0)

    def test_export_and_session_reference_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Construct only session state, bypassing the camera-opening constructor.
            session = MarkerROISession.__new__(MarkerROISession)
            session.serial, session.config, session.output = "test-camera", self.config, Path(tmp)
            session.lock, session.save_lock = threading.Lock(), threading.Lock()
            session.matcher, session.template_count, session.template_folder = None, 0, None
            session.saved = []
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            mask = np.zeros((480, 640), dtype=np.uint8)
            for point in self.points.astype(int):
                cv2.circle(mask, tuple(point), 2, 255, -1)
            sample = {"sequence": 1, "host_read_end_unix_ns": 123456789,
                      "centers": self.points.tolist(), "bgr": image, "mask": mask}
            session.samples, session.frozen = deque([sample]), {"frozen": sample}
            body = {"config": self.config, "snapshot_id": "frozen", "points": self.points.tolist()}
            with self.assertRaises(ValueError):
                session.set_template(dict(body, snapshot_id="expired"))
            result = session.set_template(body)
            template_id = result["template"]["template_id"]
            with self.assertRaises(ValueError):
                session.save_selection(body)  # stale page without active template ID
            report = session.save_selection(dict(body, template_id=template_id))
            folder = Path(report["output"])
            data = np.load(folder / "markers_100.npz", allow_pickle=False)
            self.assertEqual(data["marker"].shape, (1, 100, 5))
            np.testing.assert_array_equal(data["candidate_xy"], self.points)
            np.testing.assert_array_equal(data["candidate_offsets"], [0, 3])
            np.testing.assert_array_equal(data["output_ids"], [0, 1, 2] + [-1] * 97)
            self.assertTrue((data["marker"][0, 3:] == 0).all())
            np.testing.assert_array_equal(cv2.imread(str(folder / "reference.png")), image)
            np.testing.assert_array_equal(np.load(folder / "recent_frames.npz")["rgb"][0], image[..., ::-1])
            changed_roi = copy.deepcopy(self.config)
            changed_roi["exclude_polygons"] = [[[300, 300], [320, 300], [320, 320]]]
            with self.assertRaises(ValueError):
                session.save_selection(dict(body, config=changed_roi, template_id=template_id))
            session.set_template({"clear": True})
            self.assertIsNone(session.matcher)
            self.assertTrue((Path(result["output"]) / "template.json").exists())


if __name__ == "__main__":
    unittest.main()
