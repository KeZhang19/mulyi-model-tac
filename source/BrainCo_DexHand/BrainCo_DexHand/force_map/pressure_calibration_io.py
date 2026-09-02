from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pressure_taxel_map import coerce_taxel_calibration_value


PRESSURE_CALIBRATION_ALIASES: dict[str, tuple[str, ...]] = {
    "stiffness": ("stiffness", "k", "pressure_stiffness"),
    "damping": ("damping", "c", "pressure_damping"),
    "max_force": ("max_force", "force_max", "pressure_max_force"),
    "gain": ("gain", "pressure_gain"),
    "bias": ("bias", "pressure_bias"),
    "gamma": ("gamma", "pressure_gamma"),
    "threshold": ("threshold", "pressure_threshold"),
    "taxel_area": ("taxel_area", "area"),
}

_NESTED_KEYS = ("pressure", "calibration", "default")
_FILE_SUFFIX_KEYS = ("_npy", "_file", "_path", "_map")


def load_pressure_calibration_overrides(
    path_arg: str | Path | None,
    *,
    image_shape: tuple[int, int] | None = None,
    base_dir: str | Path | None = None,
    backend_like: Any | None = None,
) -> dict[str, Any]:
    """Load scalar or per-taxel pressure calibration overrides.

    The accepted schema is deliberately small and explicit. Fields can be stored
    at the top level or under ``pressure``, ``calibration``, or ``default``.
    Every calibration field accepts scalar values, nested lists, or a path to a
    ``.npy``/``.npz`` array. If ``image_shape`` is provided, non-scalar arrays
    are validated and flattened to ``(H * W,)`` for the WarpSDF backend.
    """

    if not path_arg:
        return {}

    path = Path(path_arg).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pressure calibration file not found: {path}")
    path = path.resolve()
    resolved_base = Path(base_dir).expanduser().resolve() if base_dir is not None else path.parent

    payload = _merged_calibration_mapping(_load_mapping(path))
    overrides: dict[str, Any] = {}
    for canonical, aliases in PRESSURE_CALIBRATION_ALIASES.items():
        found, raw_value = _lookup_calibration_value(payload, aliases)
        if not found:
            continue
        value = _load_calibration_value(raw_value, base_dir=resolved_base)
        if image_shape is not None:
            value = coerce_taxel_calibration_value(
                value,
                image_shape,
                backend_like=backend_like,
                field_name=canonical,
            )
        else:
            value = _scalar_or_array(value)
        overrides[canonical] = value
    return overrides


def pressure_calibration_value_summary(value: Any) -> float | dict[str, float | list[int]]:
    """Return a JSON-friendly scalar or compact array summary."""

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "shape": list(arr.shape),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
        except ModuleNotFoundError:
            payload = _parse_simple_yaml_scalars(text)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Pressure calibration must be a mapping, got {type(payload).__name__}")
    return dict(payload)


def _parse_simple_yaml_scalars(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value:
            payload[key] = value
    return payload


def _merged_calibration_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    for key in _NESTED_KEYS:
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            merged.update(dict(nested))
    return merged


def _lookup_calibration_value(payload: Mapping[str, Any], aliases: tuple[str, ...]) -> tuple[bool, Any]:
    for key in _expanded_aliases(aliases):
        if key in payload and payload[key] is not None:
            return True, payload[key]
    return False, None


def _expanded_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    for alias in aliases:
        keys.append(alias)
        keys.extend(f"{alias}{suffix}" for suffix in _FILE_SUFFIX_KEYS)
    return tuple(keys)


def _load_calibration_value(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, Mapping):
        for key in ("value", "values"):
            if key in value:
                return _load_calibration_value(value[key], base_dir=base_dir)
        for key in ("npy", "npz", "file", "path", "map"):
            if key in value:
                return _load_array_path(value[key], base_dir=base_dir, npz_key=value.get("key"))
        return dict(value)

    if isinstance(value, str):
        stripped = value.strip()
        path = _resolve_maybe_path(stripped, base_dir)
        if path is not None:
            return _load_array_path(path, base_dir=base_dir)
        return float(stripped)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return value


def _resolve_maybe_path(raw: str | Path, base_dir: Path) -> Path | None:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"} or path.exists():
        return path.resolve()
    return None


def _load_array_path(raw_path: str | Path, *, base_dir: Path, npz_key: Any | None = None) -> np.ndarray:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        key = str(npz_key) if npz_key is not None else data.files[0]
        return np.asarray(data[key], dtype=np.float32)
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def _scalar_or_array(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return np.asarray(arr, dtype=np.float32)
