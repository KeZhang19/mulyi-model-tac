#!/usr/bin/env python3
"""Measure VRAM scaling as OURS cumulatively enables depth, RGB, and marker."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import benchmark_marker_vram_comparison_sweep as shared


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_revo3_tactile_standalone.py"
DEFAULT_ENV_COUNTS = (16, 32, 64, 128, 256, 512, 1024, 2048)
VARIANTS = ("depth", "depth_pressure", "depth_pressure_marker", "depth_pressure_marker_rgb")
VARIANT_MODALITIES = {
    "depth": ("depth",),
    "depth_pressure": ("depth", "pressure"),
    "depth_pressure_marker": ("depth", "pressure", "marker"),
    "depth_pressure_marker_rgb": ("depth", "pressure", "marker", "rgb"),
}
VARIANT_LABELS = {
    "depth": "Depth",
    "depth_pressure": "Depth + Pressure",
    "depth_pressure_marker": "Depth + Pressure + Marker",
    "depth_pressure_marker_rgb": "Depth + Pressure + Marker + RGB",
}
VARIANT_COLORS = {
    "depth": "#4C78A8",
    "depth_pressure": "#F58518",
    "depth_pressure_marker": "#54A24B",
    "depth_pressure_marker_rgb": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep cumulative OURS tactile modalities from 16 to 2048 environments."
    )
    parser.add_argument("--env-counts", nargs="+", type=int, default=DEFAULT_ENV_COUNTS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-finger", default="index")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Rerun timeout points when resuming while preserving confirmed OOM points.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    env_counts = [int(value) for value in args.env_counts]
    if not env_counts or any(value <= 0 for value in env_counts):
        raise ValueError("--env-counts must contain positive integers")
    if len(env_counts) != len(set(env_counts)):
        raise ValueError("--env-counts must not contain duplicates")
    if not args.variants or len(args.variants) != len(set(args.variants)):
        raise ValueError("--variants must contain unique variant names")
    if int(args.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not math.isfinite(float(args.measure_seconds)) or float(args.measure_seconds) <= 0.0:
        raise ValueError("--measure-seconds must be finite and positive")
    if not math.isfinite(float(args.child_timeout_seconds)) or float(args.child_timeout_seconds) <= 0.0:
        raise ValueError("--child-timeout-seconds must be finite and positive")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python}")
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"Standalone benchmark is missing: {BENCHMARK}")


def output_directory(args: argparse.Namespace) -> Path:
    requested = Path(args.output).expanduser()
    path = requested if requested.is_absolute() else REPO_ROOT / requested
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"Output directory is not empty: {path}. Use --resume or a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_paths(output: Path, *, variant: str, num_envs: int) -> tuple[Path, Path]:
    point_dir = output / variant / f"envs-{num_envs:04d}"
    point_dir.mkdir(parents=True, exist_ok=True)
    return point_dir / "result.json", point_dir / "run.log"


def child_command(
    args: argparse.Namespace,
    *,
    variant: str,
    num_envs: int,
    result_path: Path,
) -> list[str]:
    return [
        str(Path(args.python).resolve()),
        "-u",
        str(BENCHMARK),
        "--implementation",
        "ours",
        "--modalities",
        *VARIANT_MODALITIES[variant],
        "--num-envs",
        str(int(num_envs)),
        "--focus-finger",
        str(args.focus_finger),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--measure-seconds",
        f"{float(args.measure_seconds):.9g}",
        "--device",
        str(args.device),
        "--output",
        str(result_path),
    ]


def success_row(
    variant: str,
    result: dict[str, Any],
    result_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    cfg = result["configuration"]
    metrics = result["metrics"]
    gpu = result["gpu"]
    nvml = gpu.get("nvml") or {}
    observations = result.get("observations") or {}
    shapes = {
        name: value.get("shape")
        for name, value in observations.items()
        if isinstance(value, dict) and value.get("shape") is not None
    }
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "modalities": "+".join(VARIANT_MODALITIES[variant]),
        "num_envs": int(cfg["num_envs"]),
        "status": "success",
        "process_peak_vram_gib": shared.gib(nvml.get("process_current_peak_bytes")),
        "torch_peak_allocated_gib": shared.gib(gpu.get("peak_allocated_bytes")),
        "torch_peak_reserved_gib": shared.gib(gpu.get("peak_reserved_bytes")),
        "device_total_vram_gib": shared.gib(gpu.get("total_memory_bytes")),
        "batch_hz": float(metrics["sim_hz"]),
        "env_steps_per_s": float(metrics["env_steps_per_s"]),
        "mean_step_ms": float(metrics["mean_step_s"]) * 1000.0,
        "measured_steps": int(cfg["measured_steps"]),
        "measured_time_s": float(metrics["total_time_s"]),
        "observation_shapes": json.dumps(shapes, separators=(",", ":")),
        "result": str(result_path),
        "log": str(log_path),
        "return_code": 0,
    }


def failure_row(
    variant: str,
    num_envs: int,
    *,
    status: str,
    result_path: Path,
    log_path: Path,
    return_code: int | None,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "modalities": "+".join(VARIANT_MODALITIES[variant]),
        "num_envs": int(num_envs),
        "status": status,
        "process_peak_vram_gib": None,
        "torch_peak_allocated_gib": None,
        "torch_peak_reserved_gib": None,
        "device_total_vram_gib": None,
        "batch_hz": None,
        "env_steps_per_s": None,
        "mean_step_ms": None,
        "measured_steps": None,
        "measured_time_s": None,
        "observation_shapes": None,
        "result": str(result_path),
        "log": str(log_path),
        "return_code": return_code,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gpu_capacity_gib(rows: list[dict[str, Any]]) -> float:
    values = [
        float(row["device_total_vram_gib"])
        for row in rows
        if row.get("device_total_vram_gib") is not None
    ]
    return max(values, default=31.36)


def plot_vram(path: Path, rows: list[dict[str, Any]], env_counts: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    capacity = gpu_capacity_gib(rows)
    figure, axis = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    for variant in VARIANTS:
        selected = {int(row["num_envs"]): row for row in rows if row["variant"] == variant}
        xs: list[int] = []
        ys: list[float] = []
        oom_x: list[int] = []
        for num_envs in env_counts:
            row = selected.get(num_envs)
            if row is None:
                continue
            status = str(row["status"])
            value = row.get("process_peak_vram_gib")
            if status == "success" and value is not None:
                xs.append(num_envs)
                ys.append(float(value))
                continue
            if status in ("oom", "inferred_oom"):
                xs.append(num_envs)
                ys.append(capacity)
                oom_x.append(num_envs)
                break
        if not xs:
            continue
        axis.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.1,
            markersize=5.5,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
        if oom_x:
            axis.scatter(
                oom_x,
                [capacity] * len(oom_x),
                marker="X",
                s=70,
                color=VARIANT_COLORS[variant],
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )

    axis.axhline(capacity, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8)
    axis.text(
        env_counts[0],
        capacity * 1.006,
        f"GPU capacity / OOM ({capacity:.2f} GiB)",
        fontsize=9,
        color="#444444",
        va="bottom",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(env_counts)
    axis.set_xticklabels([str(value) for value in env_counts], rotation=30)
    axis.set_ylim(0.0, capacity * 1.075)
    axis.set_xlabel("Parallel environments")
    axis.set_ylabel("Peak process VRAM (GiB)")
    axis.set_title("Cumulative Tactile Modalities VRAM Scaling (3 s)")
    axis.grid(True, which="major", alpha=0.27)
    axis.legend(ncol=2)
    figure.savefig(path, dpi=210)
    plt.close(figure)


def plot_throughput(path: Path, rows: list[dict[str, Any]], env_counts: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    for variant in VARIANTS:
        selected = sorted(
            (
                row
                for row in rows
                if row["variant"] == variant
                and row["status"] == "success"
                and row.get("env_steps_per_s") is not None
            ),
            key=lambda row: int(row["num_envs"]),
        )
        if selected:
            axis.plot(
                [int(row["num_envs"]) for row in selected],
                [float(row["env_steps_per_s"]) for row in selected],
                marker="o",
                linewidth=2.1,
                markersize=5.5,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(env_counts)
    axis.set_xticklabels([str(value) for value in env_counts], rotation=30)
    axis.set_xlabel("Parallel environments")
    axis.set_ylabel("Throughput (env-steps/s)")
    axis.set_title("Cumulative Tactile Modalities Throughput Scaling (3 s)")
    axis.grid(True, which="major", alpha=0.27)
    axis.legend(ncol=2)
    figure.savefig(path, dpi=210)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output = output_directory(args)
    env_counts = sorted(int(value) for value in args.env_counts)
    variants = [str(value) for value in args.variants]
    rows: list[dict[str, Any]] = []
    prior_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if bool(args.resume):
        prior_summary = shared.load_json(output / "summary.json")
        if prior_summary is not None:
            prior_rows = {
                (str(row["variant"]), int(row["num_envs"])): row
                for row in prior_summary.get("results", [])
                if isinstance(row, dict)
                and row.get("variant") is not None
                and row.get("num_envs") is not None
            }

    for variant in variants:
        oom_seen = False
        for num_envs in env_counts:
            result_path, log_path = result_paths(output, variant=variant, num_envs=num_envs)
            existing_result = shared.load_json(result_path) if bool(args.resume) else None
            if existing_result is not None:
                row = success_row(variant, existing_result, result_path, log_path)
                rows.append(row)
                print(
                    f"[resume] variant={variant} envs={num_envs} "
                    f"vram={row['process_peak_vram_gib']:.3f} GiB",
                    flush=True,
                )
                continue

            prior_row = prior_rows.get((variant, num_envs))
            prior_status = None if prior_row is None else str(prior_row.get("status"))
            retry_prior = prior_status == "timeout" and bool(args.retry_timeouts)
            if prior_row is not None and prior_status != "success" and not retry_prior:
                rows.append(prior_row)
                oom_seen = oom_seen or prior_status in ("oom", "inferred_oom")
                print(f"[preserve] variant={variant} envs={num_envs} status={prior_status}", flush=True)
                continue

            if oom_seen:
                rows.append(
                    failure_row(
                        variant,
                        num_envs,
                        status="inferred_oom",
                        result_path=result_path,
                        log_path=log_path,
                        return_code=None,
                    )
                )
                print(f"[skip-oom] variant={variant} envs={num_envs}", flush=True)
                continue

            command = child_command(
                args,
                variant=variant,
                num_envs=num_envs,
                result_path=result_path,
            )
            return_code, timed_out = shared.run_child(
                command,
                log_path=log_path,
                timeout_s=float(args.child_timeout_seconds),
            )
            result = shared.load_json(result_path)
            if return_code == 0 and result is not None:
                row = success_row(variant, result, result_path, log_path)
                rows.append(row)
                print(
                    f"[success] variant={variant} envs={num_envs} "
                    f"vram={row['process_peak_vram_gib']:.3f} GiB "
                    f"throughput={row['env_steps_per_s']:.1f}",
                    flush=True,
                )
                continue

            oom = shared.is_cuda_oom(log_path)
            status = "oom" if oom else "timeout" if timed_out else "failed"
            rows.append(
                failure_row(
                    variant,
                    num_envs,
                    status=status,
                    result_path=result_path,
                    log_path=log_path,
                    return_code=return_code,
                )
            )
            print(
                f"[{status}] variant={variant} envs={num_envs} "
                f"return_code={return_code} log={log_path}",
                flush=True,
            )
            oom_seen = oom

    rows.sort(key=lambda row: (VARIANTS.index(str(row["variant"])), int(row["num_envs"])))
    write_csv(output / "aggregate.csv", rows)
    plot_vram(output / "cumulative_modalities_vram_scaling.png", rows, env_counts)
    plot_vram(output / "cumulative_modalities_vram_scaling.pdf", rows, env_counts)
    plot_throughput(output / "cumulative_modalities_throughput_scaling.png", rows, env_counts)
    plot_throughput(output / "cumulative_modalities_throughput_scaling.pdf", rows, env_counts)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "implementation": "ours",
            "variants": variants,
            "variant_modalities": {name: list(VARIANT_MODALITIES[name]) for name in variants},
            "env_counts": env_counts,
            "warmup_steps": int(args.warmup_steps),
            "measure_seconds": float(args.measure_seconds),
            "device": str(args.device),
            "sequential_isolated_processes": True,
            "rl_environment": False,
            "stop_variant_after_first_cuda_oom": True,
            "oom_plot_value": "device_total_vram_gib",
        },
        "gpu_capacity_gib": gpu_capacity_gib(rows),
        "results": rows,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    print(
        f"[done] statuses={status_counts} "
        f"plot={output / 'cumulative_modalities_vram_scaling.png'} summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
