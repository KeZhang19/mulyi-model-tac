#!/usr/bin/env python3
"""Run repeatable headless scaling benchmarks for the full Revo3 tactile stack.

Each child process uses the same environment and scripted press path as
``visualize_rl_tactile_obs.py``.  The child performs untimed warm-up steps,
then measures the complete policy-observation update with display-only image
assembly disabled.  Running every environment count in a fresh process keeps
CUDA peak-memory measurements independent.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUALIZER = REPO_ROOT / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "revo3_tactile_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full Revo3 Pressure + Depth + RGB + Marker observations headlessly."
    )
    parser.add_argument("--env-counts", type=int, nargs="+", default=(1, 4, 8, 16))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--measure-steps", type=int, default=1000)
    parser.add_argument("--task", default="BrainCo-Dexsuite-Revo3-Right-Lift-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-finger", default="index")
    parser.add_argument("--presser", default="square_4")
    parser.add_argument("--presser-force-n", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--sync-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize CUDA around each measured control step (recommended).",
    )
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one extra argument token to every visualizer child; repeat as needed.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.env_counts or any(int(value) <= 0 for value in args.env_counts):
        raise ValueError("--env-counts must contain positive integers")
    if len(set(int(value) for value in args.env_counts)) != len(args.env_counts):
        raise ValueError("--env-counts must not contain duplicates")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if int(args.measure_steps) <= 0:
        raise ValueError("--measure-steps must be positive")
    if not VISUALIZER.is_file():
        raise FileNotFoundError(f"Visualizer not found: {VISUALIZER}")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_OUTPUT_ROOT / timestamp
    else:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Benchmark output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def child_command(
    args: argparse.Namespace,
    *,
    num_envs: int,
    result_path: Path,
) -> list[str]:
    command = [
        str(Path(args.python).resolve()),
        "-u",
        str(VISUALIZER),
        "--task",
        str(args.task),
        "--device",
        str(args.device),
        "--num_envs",
        str(int(num_envs)),
        "--focus-finger",
        str(args.focus_finger),
        "--presser",
        str(args.presser),
        "--target",
        "tacmap",
        "--press-start-offset",
        "0.035",
        "--press-motion-actor",
        "object",
        "--presser-body-mode",
        "dynamic_axis",
        "--presser-force-n",
        f"{float(args.presser_force_n):.9g}",
        "--presser-axis-max-travel",
        "0.5",
        "--enable-presser-collision",
        "--robot-hold-mode",
        "hard",
        "--tacmap-contact-shell",
        "0",
        "--tacmap-resize-mode",
        "surface",
        "--focus-visuotactile-only",
        "--headless",
        "--no-local-ui",
        "--print-every",
        "0",
        "--benchmark-warmup-steps",
        str(int(args.warmup_steps)),
        "--benchmark-steps",
        str(int(args.measure_steps)),
        "--benchmark-output",
        str(result_path),
        "--benchmark-sync-cuda" if bool(args.sync_cuda) else "--no-benchmark-sync-cuda",
    ]
    command.extend(str(value) for value in args.extra_arg)
    return command


def run_child(command: list[str], log_path: Path) -> int:
    print(f"[RUN] {shlex.join(command)}", flush=True)
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


def result_row(result: dict[str, Any], *, repeat: int, path: Path) -> dict[str, Any]:
    configuration = result["configuration"]
    metrics = result["metrics"]
    gpu = result["gpu"]
    peak_allocated = gpu.get("peak_allocated_bytes")
    peak_reserved = gpu.get("peak_reserved_bytes")
    return {
        "num_envs": int(configuration["num_envs"]),
        "repeat": int(repeat),
        "mean_step_ms": float(metrics["mean_step_s"]) * 1000.0,
        "median_step_ms": float(metrics["median_step_s"]) * 1000.0,
        "p95_step_ms": float(metrics["p95_step_s"]) * 1000.0,
        "p99_step_ms": float(metrics["p99_step_s"]) * 1000.0,
        "sim_hz": float(metrics["sim_hz"]),
        "env_steps_per_s": float(metrics["env_steps_per_s"]),
        "real_time_factor": float(metrics["real_time_factor"]),
        "peak_allocated_gb": None if peak_allocated is None else float(peak_allocated) / 1.0e9,
        "peak_reserved_gb": None if peak_reserved is None else float(peak_reserved) / 1.0e9,
        "gpu": gpu.get("name"),
        "result": str(path),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for num_envs in sorted({int(row["num_envs"]) for row in rows}):
        group = [row for row in rows if int(row["num_envs"]) == num_envs]
        aggregate: dict[str, Any] = {"num_envs": num_envs, "successful_repeats": len(group)}
        for field in (
            "mean_step_ms",
            "p95_step_ms",
            "sim_hz",
            "env_steps_per_s",
            "real_time_factor",
            "peak_allocated_gb",
            "peak_reserved_gb",
        ):
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


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = resolve_output_dir(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for num_envs in (int(value) for value in args.env_counts):
        env_dir = output_dir / f"envs-{num_envs:04d}"
        env_dir.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, int(args.repeats) + 1):
            result_path = env_dir / f"repeat-{repeat:02d}.json"
            log_path = env_dir / f"repeat-{repeat:02d}.log"
            command = child_command(args, num_envs=num_envs, result_path=result_path)
            return_code = run_child(command, log_path)
            if return_code != 0 or not result_path.is_file():
                failure = {
                    "num_envs": num_envs,
                    "repeat": repeat,
                    "return_code": return_code,
                    "log": str(log_path),
                    "command": command,
                }
                failures.append(failure)
                print(f"[FAIL] envs={num_envs} repeat={repeat} return_code={return_code}", flush=True)
                if args.fail_fast:
                    break
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(result_row(result, repeat=repeat, path=result_path))
        if failures and args.fail_fast:
            break

    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "aggregate.csv", aggregates)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "env_counts": [int(value) for value in args.env_counts],
            "repeats": int(args.repeats),
            "warmup_steps": int(args.warmup_steps),
            "measure_steps": int(args.measure_steps),
            "sync_cuda": bool(args.sync_cuda),
            "task": str(args.task),
            "device": str(args.device),
        },
        "runs": rows,
        "aggregate": aggregates,
        "failures": failures,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[DONE] successful={len(rows)} failed={len(failures)} summary={summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
