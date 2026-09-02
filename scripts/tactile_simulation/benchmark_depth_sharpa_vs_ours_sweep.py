#!/usr/bin/env python3
"""Sweep parallel environment counts for the Sharpa-TacMap/OURS Depth comparison."""

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
COMPARISON = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_depth_sharpa_vs_ours.py"
IMPLEMENTATION_COLORS = {"sharpa_tacmap": "#F58518", "ours": "#4C78A8"}
IMPLEMENTATION_LABELS = {"sharpa_tacmap": "TacMap", "ours": "Ours"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Depth VRAM scaling in isolated processes for Sharpa-TacMap and OURS."
    )
    parser.add_argument(
        "--env-counts",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sharpa-root", type=Path, default=Path("/home/abc/sharpa-tacmap"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-environment summaries from an existing output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.env_counts or any(int(value) <= 0 for value in args.env_counts):
        raise ValueError("--env-counts must contain positive integers")
    if len(set(int(value) for value in args.env_counts)) != len(args.env_counts):
        raise ValueError("--env-counts must not contain duplicates")
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
        path = REPO_ROOT / "outputs" / "depth_sharpa_vs_ours" / (
            "vram_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        requested = args.output.expanduser()
        path = requested if requested.is_absolute() else REPO_ROOT / requested
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"Output directory is not empty: {path}. Use --resume or a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def stream_child(command: list[str], *, log_path: Path) -> int:
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


def comparison_command(args: argparse.Namespace, *, num_envs: int, output: Path) -> list[str]:
    return [
        str(Path(args.python).resolve()),
        "-u",
        str(COMPARISON),
        "--num-envs",
        str(int(num_envs)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--measure-seconds",
        f"{float(args.measure_seconds):.9g}",
        "--device",
        str(args.device),
        "--sharpa-root",
        str(Path(args.sharpa_root).expanduser().resolve()),
        "--python",
        str(Path(args.python).resolve()),
        "--output",
        str(output),
    ]


def load_point_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_vram(
    path: Path,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    env_counts: list[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    plotted_values: list[float] = []
    for implementation in ("sharpa_tacmap", "ours"):
        selected = sorted(
            (row for row in rows if row.get("implementation") == implementation),
            key=lambda row: int(row["num_envs"]),
        )
        xs = [int(row["num_envs"]) for row in selected if row.get("process_peak_vram_gib") is not None]
        ys = [float(row["process_peak_vram_gib"]) for row in selected if row.get("process_peak_vram_gib") is not None]
        plotted_values.extend(ys)
        if xs:
            axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=IMPLEMENTATION_COLORS[implementation],
                label=IMPLEMENTATION_LABELS[implementation],
            )

    top = max(plotted_values, default=1.0)
    for failure in failures:
        num_envs = int(failure["num_envs"])
        implementation = str(failure["implementation"])
        axis.scatter(
            [num_envs],
            [top * 1.04],
            marker="x",
            s=65,
            linewidths=2,
            color=IMPLEMENTATION_COLORS.get(implementation, "#777777"),
            zorder=5,
        )
        axis.annotate("failed", (num_envs, top * 1.04), xytext=(0, 7), textcoords="offset points", ha="center")

    axis.set_xscale("log", base=2)
    axis.set_xticks(env_counts)
    axis.set_xticklabels([str(value) for value in env_counts])
    axis.set_xlabel("Parallel environments")
    axis.set_ylabel("Peak process VRAM (GiB)")
    axis.set_title("Depth-only VRAM scaling")
    axis.grid(True, which="major", alpha=0.28)
    axis.legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_throughput(path: Path, rows: list[dict[str, Any]], env_counts: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    for implementation in ("sharpa_tacmap", "ours"):
        selected = sorted(
            (row for row in rows if row.get("implementation") == implementation),
            key=lambda row: int(row["num_envs"]),
        )
        if selected:
            axis.plot(
                [int(row["num_envs"]) for row in selected],
                [float(row["env_steps_per_s"]) for row in selected],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=IMPLEMENTATION_COLORS[implementation],
                label=implementation,
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(env_counts)
    axis.set_xticklabels([str(value) for value in env_counts])
    axis.set_xlabel("Parallel environments")
    axis.set_ylabel("Throughput (env-steps/s)")
    axis.set_title("Depth throughput scaling")
    axis.grid(True, which="major", alpha=0.28)
    axis.legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output = output_directory(args)
    env_counts = sorted(int(value) for value in args.env_counts)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for num_envs in env_counts:
        point_output = output / f"envs-{num_envs:04d}"
        point_summary_path = point_output / "summary.json"
        point_summary = load_point_summary(point_summary_path) if bool(args.resume) else None
        if point_summary is None:
            point_output.mkdir(parents=True, exist_ok=True)
            command = comparison_command(args, num_envs=num_envs, output=point_output)
            return_code = stream_child(command, log_path=output / f"envs-{num_envs:04d}.log")
            point_summary = load_point_summary(point_summary_path)
        else:
            return_code = 0 if not point_summary.get("failures") else 1
            print(f"[resume] envs={num_envs} summary={point_summary_path}", flush=True)

        if point_summary is None:
            failures.extend(
                {"num_envs": num_envs, "implementation": name, "return_code": return_code, "reason": "no summary"}
                for name in ("sharpa_tacmap", "ours")
            )
            continue

        rows.extend(point_summary.get("results") or [])
        point_failures = point_summary.get("failures") or []
        failures.extend({"num_envs": num_envs, **failure} for failure in point_failures)
        print(
            f"[point] envs={num_envs} successful={len(point_summary.get('results') or [])} "
            f"failed={len(point_failures)} return_code={return_code}",
            flush=True,
        )

    rows.sort(key=lambda row: (int(row["num_envs"]), str(row["implementation"])))
    failures.sort(key=lambda row: (int(row["num_envs"]), str(row["implementation"])))
    write_csv(output / "aggregate.csv", rows)
    plot_vram(output / "vram_scaling.png", rows, failures, env_counts)
    plot_vram(output / "depth_only_vram_scaling.pdf", rows, failures, env_counts)
    plot_vram(output / "depth_only_vram_scaling.png", rows, failures, env_counts)
    plot_throughput(output / "throughput_scaling.png", rows, env_counts)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "env_counts": env_counts,
            "warmup_steps": int(args.warmup_steps),
            "measure_seconds": float(args.measure_seconds),
            "device": str(args.device),
            "sequential_isolated_processes": True,
            "sharpa_root": str(Path(args.sharpa_root).expanduser().resolve()),
        },
        "results": rows,
        "failures": failures,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[done] successful={len(rows)} failed={len(failures)} "
        f"vram_plot={output / 'depth_only_vram_scaling.png'} summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
