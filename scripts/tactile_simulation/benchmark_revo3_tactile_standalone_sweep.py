#!/usr/bin/env python3
"""Sweep standalone Revo3 tactile simulation across parallel environment counts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_revo3_tactile_standalone.py"
DEFAULT_ENV_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and plot standalone Revo3 tactile scaling from 1 to 512 environments."
    )
    parser.add_argument("--implementation", choices=("ours", "tacmap", "fots", "tacsl", "hydroshear"), default="tacmap")
    parser.add_argument("--env-counts", nargs="+", type=int, default=DEFAULT_ENV_COUNTS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    measurement = parser.add_mutually_exclusive_group()
    measurement.add_argument("--measure-seconds", type=float, default=None)
    measurement.add_argument("--measure-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-finger", default="index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resume", action="store_true", help="Reuse valid child JSON files in an existing output directory.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one argument token to every standalone child process; repeat as needed.",
    )
    args = parser.parse_args()
    if args.measure_seconds is None and args.measure_steps is None:
        args.measure_seconds = 3.0
    return args


def validate_args(args: argparse.Namespace) -> None:
    counts = [int(value) for value in args.env_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("--env-counts must contain positive integers")
    if len(counts) != len(set(counts)):
        raise ValueError("--env-counts must not contain duplicates")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.measure_seconds is not None and (
        not math.isfinite(float(args.measure_seconds)) or float(args.measure_seconds) <= 0.0
    ):
        raise ValueError("--measure-seconds must be finite and positive")
    if args.measure_steps is not None and int(args.measure_steps) <= 0:
        raise ValueError("--measure-steps must be positive")
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"Standalone benchmark is missing: {BENCHMARK}")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python}")


def output_directory(args: argparse.Namespace) -> Path:
    path = Path(args.output).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.exists() and any(path.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"Output directory is not empty: {path}; use --resume or a new directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def child_command(args: argparse.Namespace, *, num_envs: int, result_path: Path) -> list[str]:
    command = [
        str(Path(args.python).resolve()),
        "-u",
        str(BENCHMARK),
        "--implementation",
        str(args.implementation),
        "--num-envs",
        str(int(num_envs)),
        "--focus-finger",
        str(args.focus_finger),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--device",
        str(args.device),
        "--output",
        str(result_path),
    ]
    if args.measure_seconds is not None:
        command.extend(("--measure-seconds", f"{float(args.measure_seconds):.9g}"))
    else:
        command.extend(("--measure-steps", str(int(args.measure_steps))))
    command.extend(str(value) for value in args.extra_arg)
    return command


def run_child(command: list[str], log_path: Path) -> int:
    print(f"[run] {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return int(process.wait())


def result_row(result: dict[str, Any], *, repeat: int, result_path: Path) -> dict[str, Any]:
    configuration = result["configuration"]
    metrics = result["metrics"]
    gpu = result["gpu"]
    nvml = gpu.get("nvml") or {}
    process_peak = nvml.get("process_current_peak_bytes")
    allocated_peak = gpu.get("peak_allocated_bytes")
    reserved_peak = gpu.get("peak_reserved_bytes")
    divisor = float(1024**3)
    return {
        "num_envs": int(configuration["num_envs"]),
        "repeat": int(repeat),
        "mean_step_ms": float(metrics["mean_step_s"]) * 1000.0,
        "p95_step_ms": float(metrics["p95_step_s"]) * 1000.0,
        "batch_hz": float(metrics["sim_hz"]),
        "env_steps_per_s": float(metrics["env_steps_per_s"]),
        "real_time_factor": float(metrics["real_time_factor"]),
        "process_peak_vram_gib": None if process_peak is None else float(process_peak) / divisor,
        "torch_peak_allocated_gib": None if allocated_peak is None else float(allocated_peak) / divisor,
        "torch_peak_reserved_gib": None if reserved_peak is None else float(reserved_peak) / divisor,
        "gpu": gpu.get("name"),
        "result": str(result_path),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "mean_step_ms",
        "p95_step_ms",
        "batch_hz",
        "env_steps_per_s",
        "real_time_factor",
        "process_peak_vram_gib",
        "torch_peak_allocated_gib",
        "torch_peak_reserved_gib",
    )
    aggregates: list[dict[str, Any]] = []
    for num_envs in sorted({int(row["num_envs"]) for row in rows}):
        group = [row for row in rows if int(row["num_envs"]) == num_envs]
        aggregate: dict[str, Any] = {"num_envs": num_envs, "successful_repeats": len(group)}
        for field in fields:
            values = [float(row[field]) for row in group if row[field] is not None]
            aggregate[f"{field}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, aggregates: list[dict[str, Any]], failed_counts: list[int]) -> None:
    if not aggregates:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["num_envs"]) for row in aggregates]
    panels = (
        ("batch_hz", "Batch frequency (Hz)"),
        ("env_steps_per_s", "Throughput (env-steps/s)"),
        ("process_peak_vram_gib", "Peak process VRAM (GiB)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for axis, (field, ylabel) in zip(axes, panels, strict=True):
        y = [row.get(f"{field}_mean") for row in aggregates]
        error = [row.get(f"{field}_std") or 0.0 for row in aggregates]
        valid = [(xi, float(yi), float(ei)) for xi, yi, ei in zip(x, y, error, strict=True) if yi is not None]
        if valid:
            vx, vy, ve = zip(*valid, strict=True)
            axis.errorbar(vx, vy, yerr=ve, marker="o", linewidth=2.0, capsize=3)
        axis.set_xscale("log", base=2)
        axis.set_xticks(DEFAULT_ENV_COUNTS)
        axis.set_xticklabels([str(value) for value in DEFAULT_ENV_COUNTS], rotation=45)
        axis.set_xlabel("Parallel environments")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
        for failed in sorted(set(failed_counts)):
            axis.axvline(failed, color="tab:red", linestyle=":", alpha=0.25)
    figure.suptitle("Standalone Revo3 tactile scaling")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = output_directory(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for num_envs in (int(value) for value in args.env_counts):
        env_dir = output_dir / f"envs-{num_envs:04d}"
        env_dir.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, int(args.repeats) + 1):
            result_path = env_dir / f"repeat-{repeat:02d}.json"
            log_path = env_dir / f"repeat-{repeat:02d}.log"
            if bool(args.resume) and result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    rows.append(result_row(result, repeat=repeat, result_path=result_path))
                    print(f"[reuse] {result_path}", flush=True)
                    continue
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            command = child_command(args, num_envs=num_envs, result_path=result_path)
            return_code = run_child(command, log_path)
            if return_code != 0 or not result_path.is_file():
                failures.append(
                    {
                        "num_envs": num_envs,
                        "repeat": repeat,
                        "return_code": return_code,
                        "log": str(log_path),
                        "command": command,
                    }
                )
                print(f"[failed] envs={num_envs} repeat={repeat} return_code={return_code}", flush=True)
                if bool(args.fail_fast):
                    break
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(result_row(result, repeat=repeat, result_path=result_path))
        if failures and bool(args.fail_fast):
            break

    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "aggregate.csv", aggregates)
    plot_curves(
        output_dir / "scaling_curves.png",
        aggregates,
        [int(item["num_envs"]) for item in failures],
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "implementation": str(args.implementation),
            "env_counts": [int(value) for value in args.env_counts],
            "repeats": int(args.repeats),
            "warmup_steps": int(args.warmup_steps),
            "measure_seconds": (
                None if args.measure_seconds is None else float(args.measure_seconds)
            ),
            "measure_steps": None if args.measure_steps is None else int(args.measure_steps),
            "device": str(args.device),
        },
        "runs": rows,
        "aggregate": aggregates,
        "failures": failures,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[done] successful={len(rows)} failed={len(failures)} "
        f"plot={output_dir / 'scaling_curves.png'} summary={summary_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
