#!/usr/bin/env python3
"""Run a sequential three-second Depth comparison: Sharpa-TacMap vs OURS."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARPA_WORKER = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_sharpa_tacmap_depth.py"
OURS_WORKER = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_revo3_tactile_standalone.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Sharpa-TacMap and Revo3 OURS Depth for three seconds.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sharpa-root", type=Path, default=Path("/home/abc/sharpa-tacmap"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if float(args.measure_seconds) <= 0.0:
        raise ValueError("--measure-seconds must be positive")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python}")
    if not (Path(args.sharpa_root).expanduser() / "run.py").is_file():
        raise FileNotFoundError(f"Sharpa-TacMap repository is missing: {args.sharpa_root}")


def output_directory(args: argparse.Namespace) -> Path:
    if args.output is None:
        path = REPO_ROOT / "outputs" / "depth_sharpa_vs_ours" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        requested = args.output.expanduser()
        path = requested if requested.is_absolute() else REPO_ROOT / requested
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def stream_child(command: list[str], *, cwd: Path, log_path: Path) -> int:
    print(f"[run] {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
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


def child_commands(args: argparse.Namespace, output: Path) -> list[tuple[str, list[str], Path, Path]]:
    python = str(Path(args.python).resolve())
    common = [
        "--num-envs",
        str(int(args.num_envs)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--measure-seconds",
        f"{float(args.measure_seconds):.9g}",
        "--device",
        str(args.device),
    ]
    sharpa_json = output / "sharpa_tacmap_depth.json"
    ours_json = output / "ours_depth.json"
    return [
        (
            "sharpa_tacmap",
            [
                python,
                "-u",
                str(SHARPA_WORKER),
                "--sharpa-root",
                str(Path(args.sharpa_root).expanduser().resolve()),
                *common,
                "--output",
                str(sharpa_json),
                "--headless",
            ],
            Path(args.sharpa_root).expanduser().resolve(),
            sharpa_json,
        ),
        (
            "ours",
            [
                python,
                "-u",
                str(OURS_WORKER),
                "--implementation",
                "ours",
                "--mode",
                "depth",
                *common,
                "--output",
                str(ours_json),
            ],
            REPO_ROOT,
            ours_json,
        ),
    ]


def gib(value: Any) -> float | None:
    return None if value is None else float(value) / float(1024**3)


def result_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    cfg = result["configuration"]
    metrics = result["metrics"]
    gpu = result["gpu"]
    nvml = gpu.get("nvml") or {}
    observations = result.get("observations") or {}
    depth = observations.get("depth") or observations.get("depth_m") or {}
    return {
        "implementation": name,
        "num_envs": int(cfg["num_envs"]),
        "measured_steps": int(cfg["measured_steps"]),
        "measured_time_s": float(metrics["total_time_s"]),
        "physics_dt_s": float(cfg["physics_dt_s"]),
        "decimation": int(cfg["decimation"]),
        "control_dt_s": float(cfg["control_dt_s"]),
        "depth_shape": "x".join(str(value) for value in depth.get("shape", [])),
        "mean_step_ms": float(metrics["mean_step_s"]) * 1000.0,
        "batch_hz": float(metrics["sim_hz"]),
        "env_steps_per_s": float(metrics["env_steps_per_s"]),
        "real_time_factor": float(metrics["real_time_factor"]),
        "process_peak_vram_gib": gib(nvml.get("process_current_peak_bytes")),
        "torch_peak_allocated_gib": gib(gpu.get("peak_allocated_bytes")),
        "torch_peak_reserved_gib": gib(gpu.get("peak_reserved_bytes")),
        "gpu": gpu.get("name"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["implementation"]) for row in rows]
    panels = (
        ("batch_hz", "Batch frequency (Hz)"),
        ("env_steps_per_s", "Throughput (env-steps/s)"),
        ("process_peak_vram_gib", "Peak process VRAM (GiB)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for axis, (field, title) in zip(axes, panels, strict=True):
        values = [0.0 if row[field] is None else float(row[field]) for row in rows]
        bars = axis.bar(labels, values, color=("#4C78A8", "#F58518"))
        axis.bar_label(bars, fmt="%.3g", padding=3)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Depth comparison: {rows[0]['num_envs']} parallel environments")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output = output_directory(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name, command, cwd, result_path in child_commands(args, output):
        log_path = output / f"{name}.log"
        return_code = stream_child(command, cwd=cwd, log_path=log_path)
        if return_code != 0 or not result_path.is_file():
            failures.append(
                {"implementation": name, "return_code": return_code, "command": command, "log": str(log_path)}
            )
            continue
        rows.append(result_row(name, json.loads(result_path.read_text(encoding="utf-8"))))

    if rows:
        write_csv(output / "comparison.csv", rows)
    if len(rows) == 2:
        plot(output / "depth_comparison.png", rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "num_envs": int(args.num_envs),
            "warmup_steps": int(args.warmup_steps),
            "measure_seconds": float(args.measure_seconds),
            "device": str(args.device),
            "sequential_children": True,
            "sharpa_root": str(Path(args.sharpa_root).expanduser().resolve()),
        },
        "results": rows,
        "failures": failures,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[done] successful={len(rows)} failed={len(failures)} output={output} summary={summary_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
