from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pressure_taxel_map import pressure_map_stats


MAX_PRESS_INDENT_DEPTH_M = 0.0005


def validate_press_indent_depth(indent_depth_m: float, *, max_depth_m: float = MAX_PRESS_INDENT_DEPTH_M) -> float:
    depth = float(indent_depth_m)
    limit = float(max_depth_m)
    if not np.isfinite(depth) or depth < 0.0:
        raise ValueError("--press-indent-depth must be a finite non-negative distance in meters.")
    if depth > limit + 1.0e-12:
        raise ValueError("--press-indent-depth must be <= 0.5mm for RL ball-probe pressure calibration.")
    return depth


@dataclass
class BallProbePressureCalibState:
    search_distance_m: float
    indent_depth_m: float
    indent_steps: int
    settle_steps: int
    contact_penetration_threshold_m: float
    contact_force_threshold_n: float
    contact_found: bool = False
    contact_step: int = -1
    contact_axis_disp_m: float = 0.0
    indent_index: int = 0
    settle_count: int = 0
    done: bool = False
    failed: bool = False
    failure_reason: str = ""
    _indent_levels_m: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.search_distance_m = _finite_non_negative(self.search_distance_m, "search_distance_m")
        self.indent_depth_m = validate_press_indent_depth(self.indent_depth_m)
        self.indent_steps = max(1, int(self.indent_steps))
        self.settle_steps = max(1, int(self.settle_steps))
        self.contact_penetration_threshold_m = max(0.0, float(self.contact_penetration_threshold_m))
        self.contact_force_threshold_n = max(0.0, float(self.contact_force_threshold_n))
        self._indent_levels_m = np.linspace(0.0, self.indent_depth_m, self.indent_steps, dtype=np.float32)

    @property
    def phase(self) -> str:
        if self.done:
            return "done"
        if self.failed:
            return "failed"
        return "indent" if self.contact_found else "search"

    @property
    def commanded_indent_m(self) -> float:
        if not self.contact_found:
            return 0.0
        idx = min(max(0, int(self.indent_index)), len(self._indent_levels_m) - 1)
        return float(self._indent_levels_m[idx])

    @property
    def desired_travel_m(self) -> float:
        if not self.contact_found:
            return float(self.search_distance_m)
        return float(self.contact_axis_disp_m + self.commanded_indent_m)

    def actual_post_contact_indent_m(self, actual_axis_disp_m: float) -> float:
        if not self.contact_found:
            return 0.0
        return max(0.0, float(actual_axis_disp_m) - float(self.contact_axis_disp_m))

    def observe(
        self,
        *,
        step: int,
        penetration_max_m: float,
        normal_force_n: float,
        actual_axis_disp_m: float,
    ) -> bool:
        if self.done or self.failed:
            return False

        actual_disp = max(0.0, float(actual_axis_disp_m))
        has_contact = (
            float(penetration_max_m) > self.contact_penetration_threshold_m
            or float(normal_force_n) > self.contact_force_threshold_n
        )
        if not self.contact_found:
            if has_contact:
                self.contact_found = True
                self.contact_step = int(step)
                self.contact_axis_disp_m = actual_disp
                self.settle_count = 0
                return False
            if actual_disp >= self.search_distance_m - 1.0e-8:
                self.failed = True
                self.done = True
                self.failure_reason = "contact was not found within press_contact_search_distance"
            return False

        self.settle_count += 1
        return self.settle_count >= self.settle_steps

    def mark_sample_recorded(self) -> None:
        if self.done or self.failed:
            return
        if self.indent_index >= len(self._indent_levels_m) - 1:
            self.done = True
            return
        self.indent_index += 1
        self.settle_count = 0


class RlPressureTraceRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        metadata: dict[str, object],
        *,
        layout_arrays: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir).expanduser()
        self.metadata = dict(metadata)
        self.run_id = str(self.metadata.get("run_id") or "rl_ball_probe_pressure_calib")
        self.metadata["run_id"] = self.run_id
        self.metadata.setdefault("pressure_trace_schema_version", "pressure_trace_v1")
        self.layout_arrays = {str(key): np.asarray(value) for key, value in (layout_arrays or {}).items()}
        self._rows: list[dict[str, object]] = []

    def record(
        self,
        *,
        step: int,
        phase: str,
        pressure_norm: np.ndarray,
        pressure_raw_n: np.ndarray,
        penetration_m: np.ndarray,
        signed_distance_m: np.ndarray,
        commanded_indent_m: float,
        actual_axis_disp_m: float,
        actual_post_contact_indent_m: float,
        contact_force_n: float,
        contact_normal_force_n: float,
        contact_found: bool,
    ) -> None:
        self._rows.append(
            {
                "step": int(step),
                "phase": str(phase),
                "pressure_norm": _float_array(pressure_norm),
                "pressure_raw_n": _float_array(pressure_raw_n),
                "penetration_m": _float_array(penetration_m),
                "signed_distance_m": _float_array(signed_distance_m),
                "commanded_indent_m": float(commanded_indent_m),
                "actual_axis_disp_m": float(actual_axis_disp_m),
                "actual_post_contact_indent_m": float(actual_post_contact_indent_m),
                "contact_force_n": float(contact_force_n),
                "contact_normal_force_n": float(contact_normal_force_n),
                "contact_found": bool(contact_found),
            }
        )

    def close(self) -> tuple[Path, Path]:
        if not self._rows:
            raise RuntimeError("no pressure trace samples recorded")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.out_dir / f"{self.run_id}.npz"
        metadata_path = self.out_dir / f"{self.run_id}.metadata.json"

        pressure_raw = np.stack([row["pressure_raw_n"] for row in self._rows], axis=0).astype(np.float32)
        penetration = np.stack([row["penetration_m"] for row in self._rows], axis=0).astype(np.float32)
        total_force, center_of_pressure = pressure_map_stats(pressure_raw)
        payload = {
            "step": np.asarray([row["step"] for row in self._rows], dtype=np.int64),
            "phase": np.asarray([row["phase"] for row in self._rows], dtype="<U16"),
            "penetration_m": penetration,
            "signed_distance_m": np.stack([row["signed_distance_m"] for row in self._rows], axis=0).astype(np.float32),
            "penetration_velocity_mps": np.zeros_like(penetration, dtype=np.float32),
            "pressure_raw_n": pressure_raw,
            "pressure_norm": np.stack([row["pressure_norm"] for row in self._rows], axis=0).astype(np.float32),
            "total_force_n": total_force.astype(np.float32),
            "center_of_pressure_px": center_of_pressure.astype(np.float32),
            "commanded_indent_m": np.asarray([row["commanded_indent_m"] for row in self._rows], dtype=np.float32),
            "actual_axis_disp_m": np.asarray([row["actual_axis_disp_m"] for row in self._rows], dtype=np.float32),
            "actual_post_contact_indent_m": np.asarray(
                [row["actual_post_contact_indent_m"] for row in self._rows], dtype=np.float32
            ),
            "contact_force_n": np.asarray([row["contact_force_n"] for row in self._rows], dtype=np.float32),
            "contact_normal_force_n": np.asarray(
                [row["contact_normal_force_n"] for row in self._rows], dtype=np.float32
            ),
            "contact_found": np.asarray([row["contact_found"] for row in self._rows], dtype=np.bool_),
        }
        payload.update(self.layout_arrays)
        metadata_json = json.dumps(self.metadata, sort_keys=True)
        payload["metadata_json"] = np.asarray(metadata_json)
        np.savez_compressed(npz_path, **payload)
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return npz_path, metadata_path


def _finite_non_negative(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite non-negative distance in meters.")
    return out


def _float_array(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
