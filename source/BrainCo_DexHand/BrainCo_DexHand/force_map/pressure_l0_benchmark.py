from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .pressure_taxel_map import PressureFrame, PressureTaxelMap

PresserKind = Literal["square", "cylinder", "sphere", "rectangle"]


@dataclass(frozen=True)
class AnalyticPresserSpec:
    """Analytic L0 presser definition in SI units."""

    kind: PresserKind = "square"
    size_m: float = 0.004
    radius_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None

    def footprint_label(self) -> str:
        if self.kind == "sphere":
            return f"sphere_R{float(self.effective_radius_m()):g}m"
        if self.kind == "cylinder":
            return f"cylinder_D{float(self.effective_diameter_m()):g}m"
        if self.kind == "rectangle":
            return f"rectangle_{float(self.effective_width_m()):g}x{float(self.effective_height_m()):g}m"
        return f"square_{float(self.effective_width_m()):g}m"

    def effective_radius_m(self) -> float:
        if self.radius_m is not None:
            return float(self.radius_m)
        return 0.5 * float(self.size_m)

    def effective_diameter_m(self) -> float:
        if self.radius_m is not None:
            return 2.0 * float(self.radius_m)
        return float(self.size_m)

    def effective_width_m(self) -> float:
        return float(self.width_m if self.width_m is not None else self.size_m)

    def effective_height_m(self) -> float:
        return float(self.height_m if self.height_m is not None else self.size_m)


@dataclass(frozen=True)
class AnalyticPressTrajectory:
    """Commanded L0 indentation and optional slide trajectory."""

    steps: int = 32
    indentation_start_m: float = 0.0
    indentation_end_m: float = 0.002
    center_start_uv_m: tuple[float, float] = (0.0, 0.0)
    center_end_uv_m: tuple[float, float] | None = None
    dt_s: float = 1.0 / 120.0

    def centers_uv_m(self) -> np.ndarray:
        start = np.asarray(self.center_start_uv_m, dtype=np.float32)
        end = np.asarray(self.center_end_uv_m if self.center_end_uv_m is not None else self.center_start_uv_m, dtype=np.float32)
        if int(self.steps) <= 1:
            return start.reshape(1, 2)
        alpha = np.linspace(0.0, 1.0, int(self.steps), dtype=np.float32).reshape(-1, 1)
        return start.reshape(1, 2) + alpha * (end.reshape(1, 2) - start.reshape(1, 2))

    def indentations_m(self) -> np.ndarray:
        if int(self.steps) <= 1:
            return np.asarray([float(self.indentation_end_m)], dtype=np.float32)
        return np.linspace(float(self.indentation_start_m), float(self.indentation_end_m), int(self.steps), dtype=np.float32)


@dataclass(frozen=True)
class L0PressureTrace:
    """In-memory pressure trace compatible with saved WarpSDF pressure traces."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, out_dir: str | Path, *, run_id: str = "l0_pressure") -> tuple[Path, Path]:
        out = Path(out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        npz_path = out / f"{run_id}.npz"
        metadata_path = out / f"{run_id}.metadata.json"
        metadata_json = json.dumps(self.metadata, sort_keys=True)
        payload = dict(self.arrays)
        payload["metadata_json"] = np.asarray(metadata_json)
        np.savez_compressed(npz_path, **payload)
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return npz_path, metadata_path


def generate_l0_pressure_trace(
    taxel_map: PressureTaxelMap,
    presser: AnalyticPresserSpec,
    trajectory: AnalyticPressTrajectory,
    *,
    sensor_id: int = 0,
    tangent_axes: tuple[int, int] | None = None,
    normalize: bool = True,
    metadata: dict[str, Any] | None = None,
) -> L0PressureTrace:
    """Generate an analytic flat-pad pressure trace.

    The returned arrays follow the same ``(T, S, H, W)`` convention used by
    ``--save-pressure-trace``. This is intended as L0 ground truth for unit and
    CI checks, not as a high-fidelity elastomer model.
    """

    if int(trajectory.steps) <= 0:
        raise ValueError("trajectory.steps must be positive")

    points = np.asarray(taxel_map.points_l, dtype=np.float32)
    uv = _taxel_uv(points, np.asarray(taxel_map.normals_l, dtype=np.float32), tangent_axes=tangent_axes)
    h, w = taxel_map.image_shape
    indentations = trajectory.indentations_m()
    centers = trajectory.centers_uv_m()
    penetration_flat = []
    for depth_m, center_uv in zip(indentations, centers, strict=True):
        penetration_flat.append(analytic_presser_penetration(uv, presser, indentation_m=float(depth_m), center_uv_m=center_uv))
    penetration = np.stack(penetration_flat, axis=0).reshape(int(trajectory.steps), h, w).astype(np.float32)
    velocity = _penetration_velocity(penetration, dt_s=float(trajectory.dt_s))

    pressure_out = taxel_map.force_from_penetration(
        penetration,
        penetration_velocity=velocity,
        normalize=normalize,
    )
    pressure_frame = PressureFrame.from_maps(
        pressure_out.raw_force_map[:, None, :, :],
        pressure_out.force_map[:, None, :, :],
        calibration_id="l0_analytic",
    )

    arrays = {
        "step": np.arange(int(trajectory.steps), dtype=np.int64),
        "penetration_m": penetration[:, None, :, :].astype(np.float32),
        "signed_distance_m": (-penetration)[:, None, :, :].astype(np.float32),
        "penetration_velocity_mps": velocity[:, None, :, :].astype(np.float32),
        "pressure_raw_n": pressure_frame.pressure_raw_n.astype(np.float32),
        "pressure_norm": pressure_frame.pressure_norm.astype(np.float32),
        "total_force_n": pressure_frame.total_force_n.astype(np.float32),
        "center_of_pressure_px": pressure_frame.center_of_pressure_px.astype(np.float32),
        "analytic_depth_m": penetration[:, None, :, :].astype(np.float32),
        "analytic_contact_mask": (penetration[:, None, :, :] > 0.0),
        "commanded_indentation_m": indentations.astype(np.float32),
        "commanded_center_uv_m": centers.astype(np.float32),
        "pressure_taxel_points_l_m": points.reshape(1, h, w, 3).astype(np.float32),
        "pressure_taxel_normals_l": np.asarray(taxel_map.normals_l, dtype=np.float32).reshape(1, h, w, 3),
        "pressure_taxel_layout_valid": np.ones((1, h, w), dtype=np.uint8),
    }
    trace_metadata = {
        "pressure_trace_schema_version": "pressure_trace_v1",
        "pressure_backend_id": "l0_analytic",
        "pressure_layout_id": f"{taxel_map.link_name}:{h}x{w}",
        "pressure_calibration_id": "l0_analytic",
        "source": "l0_analytic_pressure",
        "sensor_id": int(sensor_id),
        "link_name": str(taxel_map.link_name),
        "image_shape": [int(h), int(w)],
        "presser": presser.footprint_label(),
        "presser_kind": presser.kind,
        "trajectory_steps": int(trajectory.steps),
        "dt_s": float(trajectory.dt_s),
        "indentation_start_m": float(trajectory.indentation_start_m),
        "indentation_end_m": float(trajectory.indentation_end_m),
    }
    if metadata:
        trace_metadata.update(metadata)
    return L0PressureTrace(arrays=arrays, metadata=trace_metadata)


def analytic_presser_penetration(
    taxel_uv_m: np.ndarray,
    presser: AnalyticPresserSpec,
    *,
    indentation_m: float,
    center_uv_m: np.ndarray | tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Analytic normal indentation at taxel coordinates for simple pressers."""

    uv = np.asarray(taxel_uv_m, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[-1] != 2:
        raise ValueError(f"taxel_uv_m must have shape (P, 2), got {uv.shape}")
    depth = float(indentation_m)
    if depth <= 0.0:
        return np.zeros((uv.shape[0],), dtype=np.float32)

    center = np.asarray(center_uv_m, dtype=np.float32).reshape(1, 2)
    rel = uv - center
    if presser.kind == "sphere":
        radius = presser.effective_radius_m()
        if radius <= 0.0:
            raise ValueError("sphere radius must be positive")
        r2 = np.sum(rel * rel, axis=-1)
        inside_sphere = r2 <= radius * radius
        sagitta = np.full((uv.shape[0],), np.inf, dtype=np.float32)
        sagitta[inside_sphere] = radius - np.sqrt(np.maximum(radius * radius - r2[inside_sphere], 0.0))
        return np.maximum(depth - sagitta, 0.0).astype(np.float32)

    if presser.kind == "cylinder":
        radius = 0.5 * presser.effective_diameter_m()
        if radius <= 0.0:
            raise ValueError("cylinder diameter must be positive")
        active = np.sum(rel * rel, axis=-1) <= radius * radius
    elif presser.kind in {"square", "rectangle"}:
        half_w = 0.5 * presser.effective_width_m()
        half_h = 0.5 * presser.effective_height_m()
        if half_w <= 0.0 or half_h <= 0.0:
            raise ValueError("flat presser width/height must be positive")
        active = (np.abs(rel[:, 0]) <= half_w) & (np.abs(rel[:, 1]) <= half_h)
    else:
        raise ValueError(f"unsupported analytic presser kind {presser.kind!r}")
    return np.where(active, depth, 0.0).astype(np.float32)


def _taxel_uv(points_l: np.ndarray, normals_l: np.ndarray, *, tangent_axes: tuple[int, int] | None) -> np.ndarray:
    points = np.asarray(points_l, dtype=np.float32)
    if tangent_axes is not None:
        return points[:, [int(tangent_axes[0]), int(tangent_axes[1])]].astype(np.float32)
    normal = np.mean(np.asarray(normals_l, dtype=np.float32), axis=0)
    normal_axis = int(np.argmax(np.abs(normal)))
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    return points[:, axes].astype(np.float32)


def _penetration_velocity(penetration: np.ndarray, *, dt_s: float) -> np.ndarray:
    dt = max(float(dt_s), 1.0e-12)
    velocity = np.zeros_like(penetration, dtype=np.float32)
    if penetration.shape[0] > 1:
        velocity[1:] = (penetration[1:] - penetration[:-1]) / dt
    return velocity
