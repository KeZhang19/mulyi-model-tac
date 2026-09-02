#!/usr/bin/env python3
"""Compare marker-method VRAM scaling in isolated standalone processes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_revo3_tactile_standalone.py"
DEFAULT_ENV_COUNTS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
METHODS = ("ours", "tacsl", "fots", "hydroshear")
METHOD_LABELS = {
    "ours": "Ours marker",
    "tacsl": "TacSL",
    "fots": "FOTS",
    "hydroshear": "HydroShear",
}
METHOD_COLORS = {
    "ours": "#4C78A8",
    "tacsl": "#E45756",
    "fots": "#54A24B",
    "hydroshear": "#F58518",
}
OOM_PATTERNS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "outofmemoryerror",
    "out of gpu memory",
    "cumemalloc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep OURS marker, TacSL, FOTS, and HydroShear VRAM from 16 to 4096 environments."
    )
    parser.add_argument("--env-counts", nargs="+", type=int, default=DEFAULT_ENV_COUNTS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-finger", default="index")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="When resuming, rerun prior timeout points while preserving confirmed OOM points.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    env_counts = [int(value) for value in args.env_counts]
    if not env_counts or any(value <= 0 for value in env_counts):
        raise ValueError("--env-counts must contain positive integers")
    if len(env_counts) != len(set(env_counts)):
        raise ValueError("--env-counts must not contain duplicates")
    if not args.methods or len(args.methods) != len(set(args.methods)):
        raise ValueError("--methods must contain unique method names")
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


def result_paths(output: Path, *, method: str, num_envs: int) -> tuple[Path, Path]:
    point_dir = output / method / f"envs-{num_envs:04d}"
    point_dir.mkdir(parents=True, exist_ok=True)
    return point_dir / "result.json", point_dir / "run.log"


def child_command(
    args: argparse.Namespace,
    *,
    method: str,
    num_envs: int,
    result_path: Path,
) -> list[str]:
    command = [
        str(Path(args.python).resolve()),
        "-u",
        str(BENCHMARK),
        "--implementation",
        str(method),
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
    if method == "ours":
        command.extend(("--mode", "marker"))
    return command


def run_child(command: list[str], *, log_path: Path, timeout_s: float) -> tuple[int, bool]:
    print(f"[run] {shlex.join(command)}", flush=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = int(process.wait(timeout=float(timeout_s)))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = int(process.wait(timeout=10.0))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = int(process.wait())
    return return_code, timed_out


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_cuda_oom(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return any(pattern in text for pattern in OOM_PATTERNS)


def gib(value: Any) -> float | None:
    return None if value is None else float(value) / float(1024**3)


def success_row(method: str, result: dict[str, Any], result_path: Path, log_path: Path) -> dict[str, Any]:
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
        "method": method,
        "method_label": METHOD_LABELS[method],
        "num_envs": int(cfg["num_envs"]),
        "status": "success",
        "process_peak_vram_gib": gib(nvml.get("process_current_peak_bytes")),
        "torch_peak_allocated_gib": gib(gpu.get("peak_allocated_bytes")),
        "torch_peak_reserved_gib": gib(gpu.get("peak_reserved_bytes")),
        "device_total_vram_gib": gib(gpu.get("total_memory_bytes")),
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
    method: str,
    num_envs: int,
    *,
    status: str,
    result_path: Path,
    log_path: Path,
    return_code: int | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
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
    for method in METHODS:
        selected = {int(row["num_envs"]): row for row in rows if row["method"] == method}
        xs: list[int] = []
        ys: list[float] = []
        statuses: list[str] = []
        for num_envs in env_counts:
            row = selected.get(num_envs)
            if row is None:
                continue
            status = str(row["status"])
            value = row.get("process_peak_vram_gib")
            if status == "success" and value is not None:
                y = float(value)
            elif status in ("oom", "inferred_oom"):
                y = capacity
            else:
                continue
            xs.append(num_envs)
            ys.append(y)
            statuses.append(status)
            # Draw only the first OOM point.  It connects the last successful
            # measurement to GPU capacity without extending horizontally
            # across larger, unmeasured environment counts.
            if status in ("oom", "inferred_oom"):
                break
        if not xs:
            continue
        axis.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.1,
            markersize=5.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        oom_x = [x for x, status in zip(xs, statuses, strict=True) if status in ("oom", "inferred_oom")]
        if oom_x:
            axis.scatter(
                oom_x,
                [capacity] * len(oom_x),
                marker="X",
                s=70,
                color=METHOD_COLORS[method],
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
    axis.set_title("Tactile Field Simulation VRAM Scaling (3 s)")
    axis.grid(True, which="major", alpha=0.27)
    axis.legend(ncol=2)
    figure.savefig(path, dpi=210)
    plt.close(figure)


def plot_throughput(path: Path, rows: list[dict[str, Any]], env_counts: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    for method in METHODS:
        selected = sorted(
            (
                row
                for row in rows
                if row["method"] == method
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
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(env_counts)
    axis.set_xticklabels([str(value) for value in env_counts], rotation=30)
    axis.set_xlabel("Parallel environments")
    axis.set_ylabel("Throughput (env-steps/s)")
    axis.set_title("Marker simulation throughput scaling (3 s)")
    axis.grid(True, which="major", alpha=0.27)
    axis.legend(ncol=2)
    figure.savefig(path, dpi=210)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output = output_directory(args)
    env_counts = sorted(int(value) for value in args.env_counts)
    methods = [str(value) for value in args.methods]
    rows: list[dict[str, Any]] = []
    prior_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if bool(args.resume):
        prior_summary = load_json(output / "summary.json")
        if prior_summary is not None:
            prior_rows = {
                (str(row["method"]), int(row["num_envs"])): row
                for row in prior_summary.get("results", [])
                if isinstance(row, dict) and row.get("method") is not None and row.get("num_envs") is not None
            }

    for method in methods:
        oom_seen = False
        for num_envs in env_counts:
            result_path, log_path = result_paths(output, method=method, num_envs=num_envs)
            existing_result = load_json(result_path) if bool(args.resume) else None
            if existing_result is not None:
                row = success_row(method, existing_result, result_path, log_path)
                rows.append(row)
                print(
                    f"[resume] method={method} envs={num_envs} "
                    f"vram={row['process_peak_vram_gib']:.3f} GiB",
                    flush=True,
                )
                continue

            prior_row = prior_rows.get((method, num_envs))
            prior_status = None if prior_row is None else str(prior_row.get("status"))
            retry_prior = prior_status == "timeout" and bool(args.retry_timeouts)
            if prior_row is not None and prior_status != "success" and not retry_prior:
                rows.append(prior_row)
                oom_seen = oom_seen or prior_status in ("oom", "inferred_oom")
                print(f"[preserve] method={method} envs={num_envs} status={prior_status}", flush=True)
                continue

            if oom_seen:
                rows.append(
                    failure_row(
                        method,
                        num_envs,
                        status="inferred_oom",
                        result_path=result_path,
                        log_path=log_path,
                        return_code=None,
                    )
                )
                print(f"[skip-oom] method={method} envs={num_envs}", flush=True)
                continue

            command = child_command(args, method=method, num_envs=num_envs, result_path=result_path)
            return_code, timed_out = run_child(
                command,
                log_path=log_path,
                timeout_s=float(args.child_timeout_seconds),
            )
            result = load_json(result_path)
            if return_code == 0 and result is not None:
                row = success_row(method, result, result_path, log_path)
                rows.append(row)
                print(
                    f"[success] method={method} envs={num_envs} "
                    f"vram={row['process_peak_vram_gib']:.3f} GiB "
                    f"throughput={row['env_steps_per_s']:.1f}",
                    flush=True,
                )
                continue

            oom = is_cuda_oom(log_path)
            status = "oom" if oom else "timeout" if timed_out else "failed"
            rows.append(
                failure_row(
                    method,
                    num_envs,
                    status=status,
                    result_path=result_path,
                    log_path=log_path,
                    return_code=return_code,
                )
            )
            print(
                f"[{status}] method={method} envs={num_envs} return_code={return_code} log={log_path}",
                flush=True,
            )
            oom_seen = oom

    rows.sort(key=lambda row: (METHODS.index(str(row["method"])), int(row["num_envs"])))
    write_csv(output / "aggregate.csv", rows)
    plot_vram(output / "tactile_field_vram_scaling.png", rows, env_counts)
    plot_vram(output / "tactile_field_vram_scaling.pdf", rows, env_counts)
    plot_throughput(output / "marker_throughput_scaling.png", rows, env_counts)
    plot_throughput(output / "marker_throughput_scaling.pdf", rows, env_counts)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "methods": methods,
            "env_counts": env_counts,
            "warmup_steps": int(args.warmup_steps),
            "measure_seconds": float(args.measure_seconds),
            "device": str(args.device),
            "sequential_isolated_processes": True,
            "stop_method_after_first_cuda_oom": True,
            "oom_plot_value": "device_total_vram_gib",
            "retry_timeouts": bool(args.retry_timeouts),
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
        f"[done] statuses={status_counts} plot={output / 'tactile_field_vram_scaling.png'} "
        f"summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
