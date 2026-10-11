#!/usr/bin/env python3
"""Infer RGB/depth/marker from a readonly capture, without opening hardware.

By default frame 0 is the reference and frames 1..N are processed.
With --reference-capture, use that capture's frame 0 and process all current frames.
The reference condition must be stated explicitly. This is not physical calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brainco-revo3-sdk/runtime/python"))


def trimodal_write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def trimodal_infer_worker(args):
    import cv2
    import numpy as np
    from pyvitaisdk import VTSensor, VTSensorType, VTSDataType as D
    from pyvitaisdk.vts.core import GFBCI, GFBCT
    from pyvitaisdk.vts.core.base.profile import VideoStreamFmt

    source = args.capture / args.serial
    folder = args.output / args.serial
    folder.mkdir()
    result = {"serial": args.serial, "physical_finger": None, "errors": [],
              "reference_state": args.reference_state, "reference_frame_index": 0,
              "physical_calibration_validated": False, "hardware_opened": False}
    sensor = prototype = None
    try:
        camera = json.loads((source / "camera.json").read_text())
        if not camera["passed"] or camera["color_order_npz"] != "RGB":
            raise ValueError("Source must be a successful RGB readonly capture")
        raw_path = source / "frames.npz"
        result["source_npz"] = str(raw_path)
        result["source_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        first = 0 if args.reference_capture else 1
        with np.load(raw_path, allow_pickle=False) as raw:
            rgb = raw["rgb"].copy()
            times = {k: raw[k][first:].copy() for k in raw.files if k.startswith("host_read_")}
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or len(rgb) < 2:
            raise ValueError("Invalid source RGB shape/dtype/count")
        reference_source = (args.reference_capture or args.capture) / args.serial
        reference_camera = json.loads((reference_source / "camera.json").read_text())
        if (not reference_camera["passed"] or reference_camera["serial"] != args.serial
                or reference_camera["color_order_npz"] != "RGB"):
            raise ValueError("Reference camera identity/format mismatch")
        reference_path = reference_source / "frames.npz"
        with np.load(reference_path, allow_pickle=False) as raw:
            reference_rgb = raw["rgb"][0].copy()
        if reference_rgb.shape != rgb[0].shape or reference_rgb.dtype != np.uint8:
            raise ValueError("Reference image shape/dtype mismatch")
        result["reference_source_npz"] = str(reference_path)
        result["reference_source_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        # Use the vendor's product detection, not an inferred physical finger map.
        product = VTSensor._detect_product(args.serial)
        if product not in {"GFBCI", "GFBCT"}:
            raise ValueError(f"Unsupported SDK product {product}")
        prototype = VTSensor(sensor_type=VTSensorType[product])
        cfg = prototype._config
        result["sdk_factory_profile"] = asdict(cfg)
        sensor_cls = {"GFBCI": GFBCI, "GFBCT": GFBCT}[product]
        # Preserve factory marker/output/model settings, but match actual input size.
        sensor = sensor_cls(
            model_path=cfg.model_path, marker_size=list(cfg.marker_size),
            marker_offsets=list(cfg.marker_offsets), output_format=cfg.output_format,
            stream_format=VideoStreamFmt(rgb.shape[2], rgb.shape[1],
                                         int(camera["requested"]["fps"]), "MJPG"),
        )
        result.update(product=product, profile=asdict(sensor._config),
                      marker_tracker=type(sensor._marker_tracker).__name__,
                      model_sha256=hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest())
        baseline = sensor.calibrate(cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR))
        if baseline is None:
            raise ValueError("SDK returned no reference image")
        sensor.enable_lowcost_mode(False)
        np.savez_compressed(folder / "reference.npz", rgb=reference_rgb, warped_bgr=baseline)
        ip = sensor._image_processor
        result["warp"] = {k: (getattr(ip, k).tolist() if isinstance(getattr(ip, k), np.ndarray)
                              else getattr(ip, k)) for k in
                          ("_warp_src", "_warp_dsize", "_warp_scale", "_warp_paddings",
                           "_is_auto_warp_mode", "_need_warpPerspective")}
        types = {"raw_bgr": D.RAW_IMG, "warped_bgr": D.WARPED_IMG, "depth": D.DEPTH_MAP,
                 "marker_origin": D.MARKER_ORIGIN_VECTOR, "marker_current": D.MARKER_CURRENT_VECTOR,
                 "marker_offset": D.MARKER_OFFSET_VECTOR}
        arrays = {k: [] for k in types}
        inference_times = []
        for index in range(first, len(rgb)):
            bgr = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR)
            start = time.monotonic_ns()
            data = sensor.collect_sensor_data(*types.values(), frame=bgr)
            inference_times.append((start, time.monotonic_ns()))
            for name, key in types.items():
                value = np.asarray(data[key])
                if value.size == 0 or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {name} at source frame {index}")
                arrays[name].append(value.copy())
            if not np.array_equal(arrays["raw_bgr"][-1], bgr):
                raise ValueError("SDK raw frame does not match supplied source BGR")
            if arrays["depth"][-1].ndim != 2 or arrays["marker_offset"][-1].shape[-1] != 2:
                raise ValueError("Unexpected depth or marker shape")
        packed = {k: np.stack(v) for k, v in arrays.items()}
        packed.update(rgb=rgb[first:], source_frame_index=np.arange(first, len(rgb)),
                      inference_start_end_monotonic_ns=np.asarray(inference_times, dtype=np.int64), **times)
        np.savez_compressed(folder / "trimodal.npz", **packed)
        result["frames"] = len(rgb) - first
        result["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                "min": float(v.min()), "max": float(v.max()),
                                "nonzero": int(np.count_nonzero(v))}
                            for k, v in packed.items() if k in types or k == "rgb"}
        result["marker_offset_equals_current_minus_origin"] = bool(np.array_equal(
            packed["marker_offset"], packed["marker_current"] - packed["marker_origin"]))
        result["color_order_verified"] = "SDK RAW_IMG equals supplied BGR; saved rgb is source RGB"
        result["units"] = {"depth": "SDK native; physical scale unverified",
                           "marker": "SDK image coordinates; physical scale unverified"}
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        for obj in (sensor, prototype):
            if obj is not None:
                try:
                    obj.release()
                except Exception as exc:
                    result["errors"].append(f"release: {exc}")
        result["passed"] = not result["errors"] and result.get("frames", 0) > 0
        trimodal_write_json(folder / "inference.json", result)
    return 0 if result["passed"] else 2


def probe_revo3_trimodal():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--reference-capture", type=Path,
                        help="Reuse frame 0 per matching serial from a previous raw capture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-state", required=True,
                        choices=("unverified-current", "unloaded-confirmed"))
    parser.add_argument("--expected-cameras", type=int, default=5)
    parser.add_argument("--serial", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.capture, args.output = args.capture.resolve(), args.output.resolve()
    if args.reference_capture:
        args.reference_capture = args.reference_capture.resolve()
    if args.expected_cameras < 1:
        parser.error("expected-cameras must be positive")
    if args.serial:
        return trimodal_infer_worker(args)
    source_report = json.loads((args.capture / "report.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": [],
              "source_capture": str(args.capture), "reference_state": args.reference_state,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "passed_scope": "SDK finite outputs from captured frames; not unloaded/physical calibration or live synchronization",
              "physical_calibration_validated": False, "hardware_opened": False}
    for camera in source_report["cameras"]:
        serial = camera["serial"]
        # The serial is also a directory component and must not escape the output.
        if not serial.isalnum():
            raise ValueError("Invalid serial in capture report")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--capture", str(args.capture),
               "--output", str(args.output), "--reference-state", args.reference_state, "--serial", serial]
        if args.reference_capture:
            cmd.extend(["--reference-capture", str(args.reference_capture)])
        with (args.output / f"{serial}.log").open("w") as log:
            try:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=45)
                status = {"serial": serial, "returncode": completed.returncode}
            except subprocess.TimeoutExpired:
                status = {"serial": serial, "timeout_seconds": 45}
        result["workers"].append(status)
        print(status, flush=True)
        trimodal_write_json(args.output / "report.json", result)
    result["cameras"] = [json.loads(p.read_text()) for p in sorted(args.output.glob("*/inference.json"))]
    result["passed"] = (len(result["workers"]) == args.expected_cameras
                        and len(result["cameras"]) == args.expected_cameras
                        and all(w.get("returncode") == 0 for w in result["workers"])
                        and all(c["passed"] for c in result["cameras"]))
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    trimodal_write_json(args.output / "report.json", result)
    print(f"Evidence: {args.output}; passed={result['passed']}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(probe_revo3_trimodal())
