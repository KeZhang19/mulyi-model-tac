#!/usr/bin/env python3
"""Local, bounded UVC camera preview and polygon ROI calibration.

Only the selected camera is opened. No robot/ROS/serial API is imported.
Start with the existing Revo3 SDK Python environment; --help opens no device.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from marker_roi import filter_marker_components, load_marker_roi, marker_centers_in_roi, validate_marker_roi
from marker_template import FixedMarkerMatcher, create_marker_template, roi_identity, save_marker_batch


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class MarkerROISession:
    def __init__(self, serial, output, duration, roi=None):
        if roi is not None:
            validate_marker_roi(roi)
            if roi["camera_serial"] != serial or roi["image_hw"] != [480, 640]:
                raise ValueError("Initial ROI belongs to another camera or image size")
        matches = list(Path("/dev/v4l/by-id").glob(f"*ViTai-GFBC_{serial}-video-index0"))
        if len(matches) != 1:
            raise ValueError(f"Expected one camera for {serial}, found {len(matches)}")
        self.serial, self.camera = serial, matches[0]
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.samples = deque(maxlen=60)
        self.frozen = {}
        self.error = None
        self.started = time.monotonic()
        self.duration = duration
        self.saved = []
        self.matcher = None
        self.template_count = 0
        self.template_folder = None
        self.config = roi or {
            "schema": "revo3-marker-roi-v1", "camera_serial": serial,
            "coordinate_space": "raw_rgb_pixels", "image_hw": [480, 640],
            "include_polygons": [[[0, 0], [639, 0], [639, 479], [0, 479]]],
            "exclude_polygons": [], "calibration_status": "unselected",
        }
        write_json(self.output / "session.json", {
            "camera_serial": serial, "camera_by_id": str(self.camera),
            "device": str(self.camera.resolve()),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "duration_limit_seconds": duration, "robot_motion_commands": 0,
            "serial_or_ros_opened": False, "output": str(self.output),
        })
        self.thread = threading.Thread(target=self.capture, daemon=True)

    def capture(self):
        import vitai_marker.dfs_tracking as dfs
        cap = cv2.VideoCapture(str(self.camera), cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open the selected UVC camera")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            next_preview, sequence = 0.0, 0
            for _ in range(5):
                cap.read()
            while not self.stop.is_set():
                ok, frame = cap.read()
                read_ns = time.time_ns()
                if not ok or frame is None:
                    raise RuntimeError("Camera returned an empty frame")
                if frame.shape != (480, 640, 3):
                    raise RuntimeError(f"Unexpected image shape {frame.shape}")
                now = time.monotonic()
                if now < next_preview:
                    continue
                next_preview = now + 0.2
                mask = dfs.find_marker(frame)
                _, _, _, centers = cv2.connectedComponentsWithStats(
                    (mask > 0).astype(np.uint8), connectivity=8
                )
                sdk_centers = np.asarray(dfs.marker_center(mask), dtype=float).reshape(-1, 2)
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not encoded:
                    raise RuntimeError("Preview JPEG encoding failed")
                sample = {
                    "sequence": sequence, "host_read_end_unix_ns": read_ns,
                    "centers": centers[1:].tolist(), "sdk_centers": sdk_centers.tolist(),
                    "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                    "bgr": frame.copy(), "mask": mask.copy(),
                }
                with self.lock:
                    self.samples.append(sample)
                sequence += 1
                self.ready.set()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            cap.release()

    def payload(self, sample):
        result = {key: value for key, value in sample.items() if key not in ("bgr", "mask")}
        if self.matcher is not None:
            matched = self.matcher.match(sample["centers"], self.serial, sample["bgr"].shape[:2])
            result["tracking"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in matched.items() if key != "marker"}
            result["tracking"]["template_id"] = self.matcher.template["template_id"]
        return result

    def set_template(self, body):
        with self.save_lock:
            if body.get("clear"):
                with self.lock:
                    self.matcher = None
                    self.template_folder = None
                return {"cleared": True}
            config = body["config"]
            validate_marker_roi(config)
            if config["camera_serial"] != self.serial or config["image_hw"] != [480, 640]:
                raise ValueError("Reference belongs to another camera or image size")
            with self.lock:
                sample = self.frozen.get(body.get("snapshot_id"))
                if sample is None:
                    raise ValueError("Freeze the reference frame before fixing IDs")
            template = create_marker_template(body["points"], config)
            template.update(template_id=uuid.uuid4().hex, reference_sequence=sample["sequence"],
                            reference_host_read_end_unix_ns=sample["host_read_end_unix_ns"],
                            reference_selection="user_reviewed_points",
                            contact_state="user_confirmed_unloaded" if body.get("unloaded") is True else "unverified")
            detected = np.asarray(sample["centers"], dtype=float).reshape(-1, 2)
            template["reference_source"] = [
                "sdk_component" if len(detected) and np.linalg.norm(detected - point, axis=1).min() < .01
                else "manual" for point in np.asarray(template["reference_xy"])]
            matcher = FixedMarkerMatcher(template, config)
            folder = self.output / f"template_{self.template_count + 1:03d}"
            folder.mkdir(exist_ok=False)
            write_json(folder / "template.json", template)
            write_json(folder / "roi.json", config)
            cv2.imwrite(str(folder / "reference.png"), sample["bgr"])
            with self.lock:
                self.config = json.loads(json.dumps(config))
                self.matcher = matcher
                self.template_folder = folder
                self.template_count += 1
            return {"template": template, "output": str(folder)}

    def current(self, freeze=False):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if not self.samples:
                raise RuntimeError("Waiting for the first camera frame")
            sample = self.samples[-1]
            result = self.payload(sample)
            if freeze:
                snapshot_id = uuid.uuid4().hex
                if len(self.frozen) >= 5:
                    self.frozen.pop(next(iter(self.frozen)))
                self.frozen[snapshot_id] = sample
                result["snapshot_id"] = snapshot_id
        result.update({"camera_serial": self.serial, "image_hw": [480, 640],
                       "remaining_seconds": max(0, int(self.duration - (time.monotonic() - self.started)))})
        return result

    def save_selection(self, body):
        config = body["config"]
        validate_marker_roi(config)
        if config["camera_serial"] != self.serial or config["image_hw"] != [480, 640]:
            raise ValueError("Selected ROI belongs to another camera or image size")
        with self.save_lock:
            with self.lock:
                frozen = self.frozen.get(body.get("snapshot_id"))
                if frozen is None:
                    raise ValueError("Freeze a current frame before saving its selection")
                samples = list(self.samples)
                matcher, template_folder = self.matcher, self.template_folder
            if body.get("template_id") != (matcher.template["template_id"] if matcher else None):
                raise ValueError("Active template changed; reload the page before saving")
            if matcher is not None and roi_identity(config) != matcher.template["roi_sha256"]:
                raise ValueError("ROI changed: create a new reference template before exporting markers")
            folder = self.output / f"selection_{len(self.saved) + 1:03d}"
            folder.mkdir(exist_ok=False)
            config = dict(config, calibration_status="user_selected_detection_roi",
                          selection_sequence=frozen["sequence"],
                          selection_host_read_end_unix_ns=frozen["host_read_end_unix_ns"])
            write_json(folder / "roi.json", config)
            cv2.imwrite(str(folder / "selection_raw.png"), frozen["bgr"])
            filtered = filter_marker_components(frozen["mask"], config)
            cv2.imwrite(str(folder / "selected_components.png"), filtered)
            overlay = frozen["bgr"].copy()
            for key, color in (("include_polygons", (40, 210, 40)),
                               ("exclude_polygons", (50, 50, 235))):
                for poly in config.get(key, []):
                    cv2.polylines(overlay, [np.rint(poly).astype(np.int32)], True, color, 2)
            accepted = marker_centers_in_roi(frozen["centers"], config)
            for center, keep in zip(frozen["centers"], accepted):
                point = tuple(np.rint(center).astype(int))
                cv2.drawMarker(overlay, point, (40, 210, 40) if keep else (50, 50, 235),
                               cv2.MARKER_CROSS if not keep else cv2.MARKER_TILTED_CROSS, 10, 1)
            cv2.imwrite(str(folder / "selection_overlay.png"), overlay)
            np.savez_compressed(folder / "recent_frames.npz",
                                rgb=np.stack([s["bgr"][..., ::-1] for s in samples]),
                                host_read_end_unix_ns=np.array([s["host_read_end_unix_ns"] for s in samples], np.int64))
            records = []
            for sample in samples:
                kept = int(marker_centers_in_roi(sample["centers"], config).sum())
                records.append({"sequence": sample["sequence"],
                                "host_read_end_unix_ns": sample["host_read_end_unix_ns"],
                                "candidates": len(sample["centers"]), "kept": kept})
            report = {
                "camera_serial": self.serial, "selection_sequence": frozen["sequence"],
                "selection_candidates": len(frozen["centers"]),
                "selection_kept": int(accepted.sum()), "recent_frames": len(samples),
                "recent_kept_range": [min(r["kept"] for r in records), max(r["kept"] for r in records)],
                "tracking_validated": False, "physical_marker_identity_validated": False,
                "contact_state": "unverified", "timestamp_source": "host UVC read completion",
                "scope": "User-selected detection region; counts are not tracking or grid-order verification",
                "output": str(folder), "records": records,
            }
            if matcher is not None:
                write_json(folder / "template.json", matcher.template)
                (folder / "reference.png").write_bytes((template_folder / "reference.png").read_bytes())
                report["marker_export"] = save_marker_batch(folder / "markers_100.npz", matcher, samples)
                report["reference_contact_state"] = matcher.template["contact_state"]
                report["scope"] = "Fixed reference gate association in raw pixels; physical identity and large deformation unverified"
            report["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in folder.iterdir() if p.is_file()}
            write_json(folder / "verification.json", report)
            self.saved.append(report)
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="WTUVL3197X260010B")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", type=Path, help="Reuse an existing camera-specific ROI; reference IDs are calibrated afresh")
    args = parser.parse_args()
    if not args.serial.isalnum() or not 1 <= args.duration <= 7200:
        parser.error("serial must be alphanumeric; duration must be 1..7200 seconds")
    session = MarkerROISession(args.serial, args.output, args.duration,
                               load_marker_roi(args.roi) if args.roi else None)
    html_path = Path(__file__).with_suffix(".html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *arguments):
            pass

        def reply(self, value, status=200, html=False):
            payload = value.encode() if html else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            try:
                if self.path == "/":
                    with session.lock:
                        initial = {"config": session.config,
                                   "template": session.matcher.template if session.matcher else None}
                    self.reply(html_path.read_text().replace("__INITIAL_CONFIG__", json.dumps(initial)), html=True)
                elif self.path == "/api/frame":
                    self.reply(session.current())
                elif self.path == "/api/status":
                    self.reply({"error": session.error, "saved": session.saved,
                                "preview_samples": len(session.samples)})
                else:
                    self.reply({"error": "Not found"}, 404)
            except Exception as exc:
                self.reply({"error": str(exc)}, 503)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://127.0.0.1:{server.server_port}":
                    self.reply({"error": "Origin mismatch"}, 403)
                    return
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= 65536:
                    raise ValueError("Request body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/freeze":
                    self.reply(session.current(freeze=True))
                elif self.path == "/api/template":
                    self.reply(session.set_template(body))
                elif self.path == "/api/save":
                    self.reply(session.save_selection(body))
                elif self.path == "/api/stop":
                    self.reply({"stopped": True})
                    session.stop.set()
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    self.reply({"error": "Not found"}, 404)
            except Exception as exc:
                self.reply({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    session.thread.start()
    if not session.ready.wait(10) or session.error:
        session.stop.set()
        server.server_close()
        raise RuntimeError(session.error or "Camera startup timed out")
    timer = threading.Timer(args.duration, server.shutdown)
    timer.daemon = True
    timer.start()
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}",
                      "camera": str(session.camera), "serial": session.serial,
                      "duration_seconds": args.duration, "output": str(session.output)}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        session.stop.set()
        timer.cancel()
        server.server_close()
        session.thread.join(timeout=2)
        write_json(session.output / "closed.json", {
            "closed_utc": datetime.now(timezone.utc).isoformat(),
            "capture_thread_stopped": not session.thread.is_alive(),
            "saved_selections": len(session.saved), "error": session.error,
        })


if __name__ == "__main__":
    main()
