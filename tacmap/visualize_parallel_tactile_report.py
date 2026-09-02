from __future__ import annotations

"""Generate a visual report for parallel TacMap validation traces."""

import argparse
import csv
import html
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "outputs" / "tactile_parallel_benchmark"
DEFAULT_PATTERNS = [
    "*projection_fixed_ref_envs_1_record100*.npz",
    "*projection_fixed_envs_4_record100*.npz",
    "*projection_fixed_envs_16_record100_sparse*.npz",
    "*projection_fixed_envs_64_record100_sparse*.npz",
    "*projection_fixed_envs_256_record100_sparse*.npz",
]
DEFAULT_LOG_PATTERNS = ["1024_retry_240x240.log", "1024_retry_240x240_replicate.log"]


def _path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _latest(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_trace(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        arrays = {name: data[name].copy() for name in data.files if name != "metadata"}
    return metadata, arrays


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_json_path(trace_path: Path) -> Path:
    return trace_path.with_name(trace_path.stem + "__metrics.json")


def _summary_json_path(trace_path: Path) -> Path:
    return trace_path.with_name(trace_path.stem + "__summary.json")


def _font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _heatmap(raw: np.ndarray, vmax: float, size: int = 240) -> Image.Image:
    if vmax <= 0:
        vmax = 1.0
    x = np.clip(raw.astype(np.float32) / float(vmax), 0.0, 1.0)
    r = np.interp(x, [0.0, 0.20, 0.55, 1.0], [5, 20, 245, 255])
    g = np.interp(x, [0.0, 0.20, 0.55, 1.0], [8, 110, 220, 40])
    b = np.interp(x, [0.0, 0.20, 0.55, 1.0], [14, 235, 70, 20])
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[x <= 0.0] = np.array([5, 8, 14], dtype=np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if size != raw.shape[-1]:
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def _centroid(raw: np.ndarray, threshold: float = 1.0e-9) -> tuple[float, float] | None:
    mask = raw > threshold
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    lines = text.split("\n")
    widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    w = max(widths) if widths else 0
    h = sum(heights) + max(0, len(lines) - 1) * 4
    x, y = xy
    draw.rectangle((x - 6, y - 4, x + w + 6, y + h + 8), fill=(0, 0, 0, 170))
    yy = y
    for line, line_h in zip(lines, heights):
        draw.text((x, yy), line, font=font, fill=(245, 248, 255))
        yy += line_h + 4


def _line_safe(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], scale: float, color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    scaled = [(int(round(x * scale)), int(round(y * scale))) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(scaled) >= 2:
        draw.line(scaled, fill=color, width=3)
        x, y = scaled[-1]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def _make_sequence_gif(trace_path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    raw = arrays["tacmap_raw"][:, 0, 0]
    steps = arrays["recorded_step_indices"]
    vmax = max(float(np.max(raw)), 1.0e-6)
    mean_step = float(np.mean(arrays.get("step_times_s", np.array([], dtype=float))))
    sim_hz = 1.0 / mean_step if mean_step > 0 else 0.0
    font = _font(15)
    small = _font(12)
    scale = 2.0
    frames: list[Image.Image] = []
    path_points: list[tuple[float, float]] = []
    pixel_area = float(metadata.get("pixel_area_m2", 0.0))
    for frame_i, step in enumerate(steps):
        img = _heatmap(raw[frame_i], vmax, size=int(raw.shape[-1] * scale)).convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")
        c = _centroid(raw[frame_i])
        if c is not None:
            path_points.append(c)
        _line_safe(draw, path_points, scale, (255, 255, 255))
        active = int(np.count_nonzero(raw[frame_i] > 1.0e-9))
        max_depth_mm = float(np.max(raw[frame_i]) * 1000.0)
        volume = float(np.sum(raw[frame_i]) * pixel_area)
        label = (
            f"1-env reference | step {int(step)}\n"
            f"max depth {max_depth_mm:.3f} mm | active {active}\n"
            f"volume {volume:.3e} m3 | sim {sim_hz:.1f} Hz"
        )
        _draw_text_box(draw, (12, 12), label, font)
        draw.text((12, img.height - 22), trace_path.name[:80], font=small, fill=(210, 220, 235, 230))
        frames.append(img.convert("RGB"))
    out_path = out_dir / "tactile_sequence_env1.gif"
    imageio.mimsave(out_path, [np.asarray(f) for f in frames], duration=0.45, loop=0)
    return out_path


def _make_montage_gif(trace_path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    raw = arrays["tacmap_raw"][:, :, 0]
    steps = arrays["recorded_step_indices"]
    saved_indices = list(metadata.get("saved_env_indices", range(raw.shape[1])))
    vmax = max(float(np.max(raw)), 1.0e-6)
    mean_step = float(np.mean(arrays.get("step_times_s", np.array([], dtype=float))))
    sim_hz = 1.0 / mean_step if mean_step > 0 else 0.0
    env_steps_s = float(metadata.get("num_envs", 0)) * sim_hz
    font = _font(14)
    small = _font(11)
    tile = 170
    cols = min(4, raw.shape[1])
    rows = int(math.ceil(raw.shape[1] / cols))
    header_h = 78
    frames: list[Image.Image] = []
    for frame_i, step in enumerate(steps):
        canvas = Image.new("RGB", (cols * tile, header_h + rows * tile), (14, 16, 20))
        draw = ImageDraw.Draw(canvas, "RGBA")
        title = (
            f"{metadata.get('num_envs')} envs | saved maps {raw.shape[1]} | step {int(step)} | "
            f"mean sim {sim_hz:.2f} Hz | throughput {env_steps_s:.1f} env-steps/s"
        )
        draw.text((14, 12), title, font=font, fill=(245, 248, 255))
        draw.text((14, 40), "RGB heatmap is TacMap raw depth, normalized to this run's max depth", font=small, fill=(196, 206, 220))
        for env_slot in range(raw.shape[1]):
            row = env_slot // cols
            col = env_slot % cols
            x0 = col * tile
            y0 = header_h + row * tile
            tile_img = _heatmap(raw[frame_i, env_slot], vmax, size=tile - 10)
            canvas.paste(tile_img, (x0 + 5, y0 + 5))
            d = ImageDraw.Draw(canvas, "RGBA")
            c = _centroid(raw[frame_i, env_slot])
            if c is not None:
                sx = (tile - 10) / raw.shape[-1]
                sy = (tile - 10) / raw.shape[-2]
                cx = x0 + 5 + int(round(c[0] * sx))
                cy = y0 + 5 + int(round(c[1] * sy))
                d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=(255, 255, 255), width=2)
            env_label = saved_indices[env_slot] if env_slot < len(saved_indices) else env_slot
            max_mm = float(np.max(raw[frame_i, env_slot]) * 1000.0)
            label = f"env {env_label} | {max_mm:.3f} mm"
            d.rectangle((x0 + 5, y0 + tile - 24, x0 + tile - 5, y0 + tile - 5), fill=(0, 0, 0, 160))
            d.text((x0 + 10, y0 + tile - 22), label, font=small, fill=(245, 248, 255))
        frames.append(canvas)
    out_path = out_dir / f"tactile_montage_{metadata.get('num_envs')}_envs.gif"
    imageio.mimsave(out_path, [np.asarray(f) for f in frames], duration=0.45, loop=0)
    return out_path


def _make_projection_panel(metadata: dict[str, Any], arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    raw = arrays["tacmap_raw"][:, 0, 0]
    surface = arrays["tacmap_surface_raw"][:, 0, 0]
    obj = arrays["tacmap_object_raw"][:, 0, 0]
    formula = np.maximum(surface - obj, 0.0)
    err = np.abs(raw - formula)
    steps = arrays["recorded_step_indices"]
    contact_i = int(np.argmax(raw.reshape(raw.shape[0], -1).max(axis=1)))
    vmax = max(float(raw.max()), float(formula.max()), 1.0e-6)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    panels = [
        (0, raw[0], "Reset raw"),
        (0, formula[0], "Reset max(surface-object,0)"),
        (0, err[0] * 1000.0, "Reset abs error (mm)"),
        (contact_i, raw[contact_i], f"Contact raw step {int(steps[contact_i])}"),
        (contact_i, formula[contact_i], "Contact formula"),
        (contact_i, err[contact_i] * 1000.0, "Contact abs error (mm)"),
    ]
    for ax, (_, data, title) in zip(axes.ravel(), panels):
        if "error" in title.lower():
            im = ax.imshow(data, cmap="magma", vmin=0.0, vmax=max(float(err.max() * 1000.0), 1.0e-9))
        else:
            im = ax.imshow(data * 1000.0, cmap="inferno", vmin=0.0, vmax=vmax * 1000.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"Projection correctness | direct max error {float(err.max()):.3e} m | precontact raw max {float(raw[0].max()):.3e} m",
        fontsize=13,
    )
    out_path = out_dir / "projection_formula_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _extract_metric(trace_path: Path) -> dict[str, Any] | None:
    return _load_json(_metric_json_path(trace_path))


def _trace_row(trace_path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    metric = _extract_metric(trace_path)
    raw_metric = metric.get("tacmap_raw", {}) if metric else {}
    failures = "; ".join(metric.get("failures", [])) if metric else ""
    step_times = arrays.get("step_times_s", np.array([], dtype=float))
    mean_step = float(np.mean(step_times)) if step_times.size else 0.0
    sim_hz = 1.0 / mean_step if mean_step > 0 else 0.0
    env_steps_s = float(metadata.get("num_envs", 0)) * sim_hz
    raw = arrays["tacmap_raw"]
    direct = np.maximum(arrays["tacmap_surface_raw"] - arrays["tacmap_object_raw"], 0.0)
    direct_err = float(np.max(np.abs(raw - direct)))
    steps = arrays["recorded_step_indices"]
    press_steps = int(metadata.get("press_steps", 0))
    press_mask = steps <= press_steps
    max_depth = arrays.get("all_env_max_depth_m", raw.max(axis=(-2, -1)))[:, 0, 0]
    active = arrays.get("all_env_active_pixels", (raw > 1e-9).sum(axis=(-2, -1)))[:, 0, 0]
    volume = arrays.get("all_env_volume_m3", raw.sum(axis=(-2, -1)) * float(metadata.get("pixel_area_m2", 0.0)))[:, 0, 0]
    def min_delta(values: np.ndarray) -> float:
        vals = values[press_mask]
        if vals.size < 2:
            return 0.0
        return float(np.nanmin(np.diff(vals)))
    cx = arrays.get("all_env_centroid_x_px")[:, 0, 0]
    cy = arrays.get("all_env_centroid_y_px")[:, 0, 0]
    slide_mask = steps >= press_steps
    slide_dx = slide_dy = float("nan")
    if np.count_nonzero(slide_mask) >= 2:
        slide_cx = cx[slide_mask]
        slide_cy = cy[slide_mask]
        finite = np.isfinite(slide_cx) & np.isfinite(slide_cy)
        if np.count_nonzero(finite) >= 2:
            slide_dx = float(slide_cx[finite][-1] - slide_cx[finite][0])
            slide_dy = float(slide_cy[finite][-1] - slide_cy[finite][0])
    return {
        "trace": trace_path.name,
        "num_envs": metadata.get("num_envs"),
        "saved_env_count": metadata.get("saved_env_count"),
        "map_shape": "x".join(str(x) for x in metadata.get("map_shape", [])),
        "recorded_frames": int(len(steps)),
        "mean_step_ms": mean_step * 1000.0,
        "sim_hz": sim_hz,
        "env_steps_s": env_steps_s,
        "cuda_reserved_gb": (metadata.get("cuda_memory_reserved") or 0) / 1.0e9,
        "projection_direct_error_m": direct_err,
        "precontact_raw_max_m": float(raw[0].max()),
        "press_max_depth_min_delta_mm": min_delta(max_depth) * 1000.0,
        "press_active_min_delta_px": min_delta(active),
        "press_volume_min_delta_m3": min_delta(volume),
        "slide_centroid_dx_px": slide_dx,
        "slide_centroid_dy_px": slide_dy,
        "pass": metric.get("pass") if metric else "baseline",
        "active_iou_global": raw_metric.get("active_iou_global", ""),
        "depth_rmse_mm": raw_metric.get("depth_rmse_m", 0.0) * 1000.0 if raw_metric else "",
        "max_depth_error_mm": raw_metric.get("max_depth_error_m", 0.0) * 1000.0 if raw_metric else "",
        "centroid_error_px_max": raw_metric.get("centroid_error_px_max", ""),
        "slide_centroid_rmse_px": raw_metric.get("slide_centroid_rmse_px", ""),
        "env_index_rmse_slope": raw_metric.get("env_index_rmse_slope", ""),
        "env_origin_rmse_slope_m_per_m": raw_metric.get("env_origin_rmse_slope_m_per_m", ""),
        "failures": failures,
    }


def _make_performance_chart(rows: list[dict[str, Any]], log_paths: list[Path], out_dir: Path) -> Path:
    rows = sorted(rows, key=lambda r: int(r["num_envs"]))
    envs = np.array([int(r["num_envs"]) for r in rows], dtype=float)
    step_ms = np.array([float(r["mean_step_ms"]) for r in rows], dtype=float)
    env_steps = np.array([float(r["env_steps_s"]) for r in rows], dtype=float)
    sim_hz = np.array([float(r["sim_hz"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(envs, step_ms, marker="o", linewidth=2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Step Time")
    axes[0].set_xlabel("num_envs")
    axes[0].set_ylabel("mean step time (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(envs, sim_hz, marker="o", color="#27806f", linewidth=2)
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Simulation Frequency")
    axes[1].set_xlabel("num_envs")
    axes[1].set_ylabel("steps/s")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(envs, env_steps, marker="o", color="#b15c14", linewidth=2)
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("Parallel Throughput")
    axes[2].set_xlabel("num_envs")
    axes[2].set_ylabel("env-steps/s")
    axes[2].grid(True, alpha=0.3)
    if log_paths:
        for ax in axes:
            ax.axvline(1024, color="#a11", linestyle="--", alpha=0.45)
        axes[2].text(1024, max(env_steps) * 0.7, "1024 launch failed before trace", rotation=90, va="center", color="#a11")
    out_path = out_dir / "performance_scaling.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _make_drift_chart(ref_arrays: dict[str, np.ndarray], cur_meta: dict[str, Any], cur_arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    steps = cur_arrays["recorded_step_indices"]
    ref_depth = ref_arrays["all_env_max_depth_m"][:, 0, 0]
    ref_cx = ref_arrays["all_env_centroid_x_px"][:, 0, 0]
    ref_cy = ref_arrays["all_env_centroid_y_px"][:, 0, 0]
    cur_depth = cur_arrays["all_env_max_depth_m"][:, :, 0]
    cur_cx = cur_arrays["all_env_centroid_x_px"][:, :, 0]
    cur_cy = cur_arrays["all_env_centroid_y_px"][:, :, 0]
    depth_rmse = np.sqrt(np.nanmean((cur_depth - ref_depth[:, None]) ** 2, axis=0)) * 1000.0
    centroid_err = np.sqrt((cur_cx - ref_cx[:, None]) ** 2 + (cur_cy - ref_cy[:, None]) ** 2)
    centroid_rmse = np.sqrt(np.nanmean(centroid_err ** 2, axis=0))
    envs = np.arange(cur_depth.shape[1])
    origins = np.asarray(cur_meta.get("env_origins") or np.zeros((len(envs), 3)), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(envs, depth_rmse, linewidth=1.4)
    axes[0, 0].set_title("Depth Drift vs Env Index")
    axes[0, 0].set_xlabel("env index")
    axes[0, 0].set_ylabel("max-depth RMSE (mm)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(envs, centroid_rmse, color="#7851a9", linewidth=1.4)
    axes[0, 1].set_title("Centroid Drift vs Env Index")
    axes[0, 1].set_xlabel("env index")
    axes[0, 1].set_ylabel("centroid RMSE (px)")
    axes[0, 1].grid(True, alpha=0.3)
    sc0 = axes[1, 0].scatter(origins[:, 0], origins[:, 1], c=depth_rmse, s=35, cmap="viridis")
    axes[1, 0].set_title("Depth Drift vs Env Origin")
    axes[1, 0].set_xlabel("origin x (m)")
    axes[1, 0].set_ylabel("origin y (m)")
    fig.colorbar(sc0, ax=axes[1, 0], label="depth RMSE (mm)")
    sc1 = axes[1, 1].scatter(origins[:, 0], origins[:, 1], c=centroid_rmse, s=35, cmap="magma")
    axes[1, 1].set_title("Centroid Drift vs Env Origin")
    axes[1, 1].set_xlabel("origin x (m)")
    axes[1, 1].set_ylabel("origin y (m)")
    fig.colorbar(sc1, ax=axes[1, 1], label="centroid RMSE (px)")
    fig.suptitle(f"All-env scalar drift for {cur_meta.get('num_envs')} envs, frames {int(steps[0])}-{int(steps[-1])}", fontsize=13)
    out_path = out_dir / f"drift_env_index_origin_{cur_meta.get('num_envs')}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _write_summary_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    out_path = out_dir / "summary_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.001):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _html_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    parts = ["<table><thead><tr>"]
    for field in fields:
        parts.append(f"<th>{html.escape(field)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for field in fields:
            parts.append(f"<td>{html.escape(_format_cell(row.get(field, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _copy_logs(log_paths: list[Path], out_dir: Path) -> list[tuple[str, str]]:
    copied: list[tuple[str, str]] = []
    for log_path in log_paths:
        if not log_path.exists():
            continue
        dst = out_dir / log_path.name
        shutil.copy2(log_path, dst)
        lines = dst.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-80:])
        copied.append((dst.name, tail))
    return copied


def _write_html(
    out_dir: Path,
    rows: list[dict[str, Any]],
    media: dict[str, Path],
    logs: list[tuple[str, str]],
    trace_paths: list[Path],
) -> Path:
    fields = [
        "num_envs",
        "saved_env_count",
        "map_shape",
        "mean_step_ms",
        "sim_hz",
        "env_steps_s",
        "projection_direct_error_m",
        "precontact_raw_max_m",
        "press_max_depth_min_delta_mm",
        "press_active_min_delta_px",
        "slide_centroid_dx_px",
        "slide_centroid_dy_px",
        "pass",
        "active_iou_global",
        "depth_rmse_mm",
        "max_depth_error_mm",
        "centroid_error_px_max",
        "slide_centroid_rmse_px",
    ]
    fail_fields = ["num_envs", "pass", "failures"]
    css = """
body { margin: 0; font-family: Inter, Arial, sans-serif; background: #f7f8fa; color: #20242a; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 6px; font-size: 28px; }
h2 { margin-top: 34px; font-size: 20px; }
p { line-height: 1.45; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; align-items: start; }
.figure { background: #fff; border: 1px solid #d9dee8; border-radius: 6px; padding: 14px; }
.figure img { width: 100%; height: auto; display: block; }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: 12px; }
th, td { border: 1px solid #d9dee8; padding: 7px 8px; text-align: right; vertical-align: top; }
th { background: #eef2f7; font-weight: 650; }
td:first-child, th:first-child { text-align: left; }
pre { background: #111820; color: #dce7f7; padding: 14px; overflow: auto; border-radius: 6px; max-height: 420px; }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.note { background: #fff8e5; border: 1px solid #e0c878; padding: 12px 14px; border-radius: 6px; }
"""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Parallel tactile verification report</title>",
        f"<style>{css}</style></head><body><main>",
        "<h1>Parallel Tactile Verification Report</h1>",
        f"<p>Generated {html.escape(datetime.now().isoformat(timespec='seconds'))}. Core checks: projection formula, precontact zero, monotonic press metrics, slide centroid direction, multi-env drift, env-index/origin effects, and simulation frequency scaling.</p>",
        "<div class='note'><strong>1024-env status:</strong> the 240x240 run did not reach trace capture in this runtime. The non-replicated launch stopped during USD composition; the replicate-physics attempt reported missing rigid bodies for high env indices. Logs are copied below, so this is recorded as a launch/runtime limit rather than a tactile-accuracy datapoint.</div>",
        "<h2>Metric Summary</h2>",
        _html_table(rows, fields),
        "<h2>Failures</h2>",
        _html_table(rows, fail_fields),
        "<h2>Visual Evidence</h2><div class='grid'>",
    ]
    for title, key in [
        ("1-env tactile RGB sequence", "sequence"),
        ("256-env sampled tactile RGB montage", "montage"),
        ("Projection formula check", "projection"),
        ("Performance scaling", "performance"),
        ("Env-index and origin drift", "drift"),
    ]:
        path = media.get(key)
        if path is None:
            continue
        rel = path.name
        parts.append(f"<section class='figure'><h3>{html.escape(title)}</h3><img src='{html.escape(rel)}' alt='{html.escape(title)}'></section>")
    parts.append("</div>")
    parts.append("<h2>Trace Inputs</h2><ul>")
    for trace_path in trace_paths:
        parts.append(f"<li class='code'>{html.escape(str(trace_path))}</li>")
    parts.append("</ul>")
    parts.append("<h2>1024 Launch Logs</h2>")
    for name, tail in logs:
        parts.append(f"<h3 class='code'>{html.escape(name)}</h3><pre>{html.escape(tail)}</pre>")
    parts.append("</main></body></html>")
    out_path = out_dir / "index.html"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visual report for parallel tactile benchmark traces.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--trace", action="append", type=Path, default=[])
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    benchmark_dir = _path(args.benchmark_dir)
    trace_paths = [_path(p) for p in args.trace]
    if not trace_paths:
        for pattern in DEFAULT_PATTERNS:
            match = _latest(pattern, benchmark_dir)
            if match is not None:
                trace_paths.append(match.resolve())
    trace_paths = sorted(dict.fromkeys(trace_paths), key=lambda p: json.loads(str(np.load(p, allow_pickle=False)["metadata"].item())).get("num_envs", 0))
    if not trace_paths:
        raise SystemExit(f"No traces found in {benchmark_dir}")

    log_paths = [_path(p) for p in args.log]
    if not log_paths:
        for pattern in DEFAULT_LOG_PATTERNS:
            match = _latest(pattern, benchmark_dir)
            if match is not None:
                log_paths.append(match.resolve())

    out_dir = _path(args.output_dir) if args.output_dir else benchmark_dir / ("visual_report_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]] = []
    rows: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        metadata, arrays = _load_trace(trace_path)
        loaded.append((trace_path, metadata, arrays))
        rows.append(_trace_row(trace_path, metadata, arrays))
    _write_summary_csv(rows, out_dir)

    media: dict[str, Path] = {}
    ref = loaded[0]
    media["sequence"] = _make_sequence_gif(ref[0], ref[1], ref[2], out_dir)
    highest = max(loaded, key=lambda item: int(item[1].get("num_envs", 0)))
    if int(highest[1].get("num_envs", 0)) > 1:
        media["montage"] = _make_montage_gif(highest[0], highest[1], highest[2], out_dir)
        media["drift"] = _make_drift_chart(ref[2], highest[1], highest[2], out_dir)
    media["projection"] = _make_projection_panel(highest[1], highest[2], out_dir)
    media["performance"] = _make_performance_chart(rows, log_paths, out_dir)
    copied_logs = _copy_logs(log_paths, out_dir)
    html_path = _write_html(out_dir, rows, media, copied_logs, trace_paths)
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
