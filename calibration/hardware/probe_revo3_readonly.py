#!/usr/bin/env python3
"""Bounded Revo3 state/touch and ViTai UVC reads; no motion/configuration writes.

Use the existing SDK Python environment for dependencies. The copied native SDK
packages take precedence over its packages. Each hardware channel runs in a
separate process so even a blocked native call has a finite deadline.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "vendor/brainco-revo3-sdk/runtime/python"
sys.path.insert(0, str(RUNTIME))


def revo3_json_value(value, depth=0):
    """Serialize SDK snapshot data properties, never invoke SDK methods."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if depth > 8:
        raise ValueError("SDK snapshot nesting exceeded")
    if isinstance(value, (list, tuple)):
        return [revo3_json_value(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): revo3_json_value(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "int_value"):
        return {"name": str(value), "value": value.int_value}
    if hasattr(value, "name"):
        return str(value.name)
    fields = {}
    for name in dir(value):
        if name.startswith("_") or not inspect.isdatadescriptor(getattr(type(value), name, None)):
            continue
        field = getattr(value, name)
        if not callable(field):
            fields[name] = revo3_json_value(field, depth + 1)
    return fields or str(value)


def revo3_write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


async def revo3_hand_read_worker(args):
    from bc_revo3_sdk import main_mod as sdk

    result = {"sdk_module": sdk.__file__, "motion_commands": 0, "errors": []}
    manager, hand = sdk.Manager(), None
    try:
        devices = await manager.discover(
            port=args.port, protocol=sdk.ProtocolType.Modbus,
            modbus_baudrate=sdk.Rs485Baudrate.Baud5Mbps,
            slave_id=args.slave_id, scan_all=True,
        )
        result["discovered"] = revo3_json_value(devices)
        if len(devices) != 1:
            raise RuntimeError(f"Expected one endpoint, discovered {len(devices)}")
        hand = await manager.connect(devices[0])
        for name in ("device_info", "firmware_info", "joint_layout"):
            result[name] = revo3_json_value(getattr(hand, name))
        result["touch_layout"] = revo3_json_value(hand.touch.layout)
        for name, operation in (("health", hand.health.snapshot),
                                ("enabled_mask", hand.touch.enabled_mask),
                                ("read_mode", hand.touch.read_mode)):
            try:
                result[name] = revo3_json_value(await operation())
            except Exception as exc:
                result["errors"].append({"operation": name, "error": str(exc)})
        result["value_modes"] = {}
        if hand.touch.layout is not None:
            for module in hand.touch.layout.modules:
                if module.layout_id.startswith(("pressure_array_", "high_density_matrix_")):
                    try:
                        mode = await hand.touch.value_mode(module.module_id)
                        result["value_modes"][str(module.module_id)] = revo3_json_value(mode)
                    except Exception as exc:
                        result["errors"].append({"operation": f"value_mode:{module.module_id}", "error": str(exc)})
        result["samples"] = []
        for index in range(3):
            sample = {"index": index}
            for name, operation in (("state", hand.state.snapshot), ("touch", hand.touch.snapshot)):
                start = time.monotonic_ns()
                try:
                    sample[name] = {"host_start_monotonic_ns": start,
                                    "data": revo3_json_value(await operation()),
                                    "host_end_monotonic_ns": time.monotonic_ns(),
                                    "host_end_unix_ns": time.time_ns()}
                except Exception as exc:
                    sample[name] = {"error": str(exc)}
                    result["errors"].append({"operation": f"{name}:{index}", "error": str(exc)})
            result["samples"].append(sample)
    except Exception as exc:
        result["errors"].append({"operation": "connect_or_metadata", "error": str(exc)})
    finally:
        try:
            if hand is not None:
                await hand.close()
        finally:
            await manager.close()
    result["touch_module_availability"] = []
    for sample in result.get("samples", []):
        modules = sample.get("touch", {}).get("data", {}).get("modules", [])
        result["touch_module_availability"].append([
            {"module_id": m["module_id"], "sample_state": m["sample_state"],
             "point_count_received": len(m["points"]) if m["points"] is not None else 0}
            for m in modules])
    result["passed"] = not result["errors"] and len(result.get("samples", [])) == 3
    result["passed_scope"] = "Read API completion only; inspect sample_state for each tactile module"
    revo3_write_json(args.output / "hand.json", result)


def revo3_camera_read_worker(args):
    import cv2
    import numpy as np

    path = Path(args.camera)
    serial = path.name.split("ViTai-GFBC_", 1)[1].rsplit("-video-index", 1)[0]
    folder = args.output / serial
    folder.mkdir()
    result = {"serial": serial, "by_id": str(path), "device": str(path.resolve()),
              "finger": None, "requested": {"width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
              "color_order_npz": "RGB", "timestamp_source": "host read completion; not hardware exposure", "errors": []}
    cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError("V4L2 open failed")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        result["negotiated"] = {"width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                "reported_fps": cap.get(cv2.CAP_PROP_FPS)}
        images, monotonic, unix, starts = [], [], [], []
        for index in range(args.frames + 5):
            start = time.monotonic_ns()
            ok, bgr = cap.read()
            end, wall = time.monotonic_ns(), time.time_ns()
            if not ok or bgr is None or bgr.size == 0:
                raise RuntimeError(f"Empty frame at read {index}")
            if index < 5:
                continue
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            starts.append(start)
            monotonic.append(end)
            unix.append(wall)
        rgb = np.stack(images)
        np.savez_compressed(folder / "frames.npz", rgb=rgb,
                            host_read_start_monotonic_ns=np.array(starts, dtype=np.int64),
                            host_read_end_monotonic_ns=np.array(monotonic, dtype=np.int64),
                            host_read_end_unix_ns=np.array(unix, dtype=np.int64))
        if not cv2.imwrite(str(folder / "last_frame.png"), cv2.cvtColor(rgb[-1], cv2.COLOR_RGB2BGR)):
            raise RuntimeError("PNG save failed")
        result.update({"frames": len(images), "shape": list(rgb.shape), "dtype": str(rgb.dtype),
                       "pixel_min": int(rgb.min()), "pixel_max": int(rgb.max()),
                       "pixel_std": float(rgb.std()),
                       "distinct_frame_hashes": len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in rgb}),
                       "host_completion_fps_short_sample": (len(images)-1)*1e9/(monotonic[-1]-monotonic[0])})
    except Exception as exc:
        result["errors"].append(str(exc))
    finally:
        cap.release()
    result["passed"] = not result["errors"] and result.get("frames") == args.frames
    revo3_write_json(folder / "camera.json", result)


def probe_revo3_readonly():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; refuses overwrite")
    parser.add_argument("--port", default="/dev/serial/by-id/usb-BrainCo_USB__-__RS485_Converter-if00-port0")
    parser.add_argument("--slave-id", type=lambda s: int(s, 0), default=127)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--expected-cameras", type=int, default=5)
    parser.add_argument("--worker", choices=("hand", "camera", "finder"), help=argparse.SUPPRESS)
    parser.add_argument("--camera", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 2 <= args.frames <= 60 or not 1 <= args.slave_id <= 247 or args.expected_cameras < 1:
        parser.error("frames must be 2..60, slave-id 1..247, expected-cameras >=1")
    args.output = args.output.resolve()
    if args.worker == "hand":
        asyncio.run(revo3_hand_read_worker(args))
        return 0
    if args.worker == "camera":
        revo3_camera_read_worker(args)
        return 0
    if args.worker == "finder":
        from pyvitaisdk import VTSDeviceFinder
        devices = VTSDeviceFinder().get_devices()
        revo3_write_json(args.output / "vitai_finder.json", [vars(d) for d in devices])
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    from importlib.metadata import version
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {n: version(n) for n in ("bc-revo3-sdk", "pyvitaisdk", "numpy", "opencv-python")},
              "motion_commands": 0, "tare_or_device_config_writes": 0,
              "sampling": "Sequential channels; no cross-channel hardware synchronization", "workers": []}
    cameras = sorted(Path("/dev/v4l/by-id").glob("*ViTai-GFBC*-video-index0"))
    jobs = [("hand", [], 40), ("finder", [], 25)]
    jobs.extend(("camera", ["--camera", str(camera)], 20) for camera in cameras)
    for worker, extra, timeout in jobs:
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                   "--port", args.port, "--slave-id", str(args.slave_id), "--frames", str(args.frames),
                   "--worker", worker, *extra]
        log_name = f"worker_{len(report['workers']):02d}_{worker}.log"
        with (args.output / log_name).open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                status = {"returncode": process.returncode}
            except subprocess.TimeoutExpired:
                status = {"timeout_seconds": timeout}
        report["workers"].append({"worker": worker, "arguments": extra, "log": log_name, **status})
        print(worker, extra, status, flush=True)
        revo3_write_json(args.output / "report.json", report)
    hand_path = args.output / "hand.json"
    report["hand"] = json.loads(hand_path.read_text()) if hand_path.exists() else {"passed": False}
    report["cameras"] = [json.loads(p.read_text()) for p in sorted(args.output.glob("*/camera.json"))]
    report["expected_cameras"] = args.expected_cameras
    report["passed_scope"] = "Read API completion and requested UVC frame count; not all-module health, calibration, or synchronization"
    report["passed"] = (report["hand"]["passed"] and len(cameras) == args.expected_cameras
                        and len(report["cameras"]) == len(cameras)
                        and all(c["passed"] for c in report["cameras"])
                        and all(w.get("returncode") == 0 for w in report["workers"]))
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    revo3_write_json(args.output / "report.json", report)
    print(f"Evidence: {args.output}; passed={report['passed']}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(probe_revo3_readonly())
