from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pressure_sources import NormalRayPenetrationSource
from .pressure_taxel_map import PressureCalibration, calibrate_penetration, pressure_map_stats
from .pressure_trace_metrics import load_pressure_trace_npz


def apply_normal_ray_reference_to_trace(
    trace: Mapping[str, Any] | str | Path,
    *,
    deformation_key: str = "tacmap_raw_aligned_m",
    output_prefix: str = "normal_ray",
    calibration: PressureCalibration | None = None,
    contact_deadband_m: float = 0.0,
) -> dict[str, np.ndarray]:
    """Add normal-ray pressure arrays derived from a deformation map.

    This is an L1 model-reference helper path: a geometry-consistent deformation
    map such as aligned TacMap depth is converted through the same pressure
    calibration contract as other backends. It lets the verifier distinguish
    reference/source mismatch from bugs in trace IO, shape handling, or metrics.
    """

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if deformation_key not in data:
        raise KeyError(f"deformation_key {deformation_key!r} not found in trace")

    deformation = _as_tshw(data[deformation_key], name=deformation_key)
    source = NormalRayPenetrationSource(contact_deadband_m=max(0.0, float(contact_deadband_m)))
    frame = source.frame_from_deformation(deformation)
    penetration = _as_tshw(frame.penetration_m, name="normal_ray_penetration_m")
    signed_distance = _as_tshw(frame.signed_distance_m, name="normal_ray_signed_distance_m")

    raw, pressure = calibrate_penetration(
        penetration,
        calibration or PressureCalibration(stiffness=1.0, max_force=1.0),
        normalize=True,
    )
    raw = np.asarray(raw, dtype=np.float32)
    pressure = np.asarray(pressure, dtype=np.float32)
    total_force, center = pressure_map_stats(raw)

    out = {str(key): value for key, value in data.items()}
    prefix = str(output_prefix).strip() or "normal_ray"
    out[f"{prefix}_source_key"] = np.asarray(str(deformation_key))
    out[f"{prefix}_penetration_m"] = penetration.astype(np.float32)
    out[f"{prefix}_signed_distance_m"] = signed_distance.astype(np.float32)
    out[f"{prefix}_contact_mask"] = (penetration > 0.0).astype(np.uint8)
    out[f"{prefix}_pressure_raw_n"] = raw
    out[f"{prefix}_pressure_norm"] = pressure
    out[f"{prefix}_total_force_n"] = np.asarray(total_force, dtype=np.float32)
    out[f"{prefix}_center_of_pressure_px"] = np.asarray(center, dtype=np.float32)
    return out


def write_normal_ray_reference_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write a trace with normal-ray pressure arrays and return a summary."""

    out = apply_normal_ray_reference_to_trace(trace, **kwargs)
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)

    prefix = str(kwargs.get("output_prefix", "normal_ray")).strip() or "normal_ray"
    penetration = _as_tshw(out[f"{prefix}_penetration_m"], name=f"{prefix}_penetration_m")
    pressure = _as_tshw(out[f"{prefix}_pressure_norm"], name=f"{prefix}_pressure_norm")
    summary = {
        "out_trace": str(path),
        "output_prefix": prefix,
        "deformation_key": str(kwargs.get("deformation_key", "tacmap_raw_aligned_m")),
        "penetration_shape": [int(v) for v in penetration.shape],
        "pressure_shape": [int(v) for v in pressure.shape],
        "max_penetration_m": float(np.max(penetration)) if penetration.size else 0.0,
        "max_pressure_norm": float(np.max(pressure)) if pressure.size else 0.0,
        "active_taxel_count": int(np.count_nonzero(pressure > 0.0)),
    }
    return path, summary


def _as_tshw(value: Any, *, name: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 2:
        return arr[None, None, :, :]
    raise ValueError(f"{name} must have shape (T,S,H,W), (T,H,W), or (H,W); got {arr.shape}")
