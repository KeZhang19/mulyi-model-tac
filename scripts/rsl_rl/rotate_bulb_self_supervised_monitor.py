#!/usr/bin/env python3
"""Record self-supervised progress signals for a running Rotate-Bulb job.

The monitor derives phase labels from the simulator's own logged metrics.  It
does not modify policy weights or claim to train the frozen tactile encoder.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_log(path: Path) -> dict:
    """Recover the latest scalar block when no supervisor state file exists."""
    try:
        lines = path.read_text(errors="replace").splitlines()[-4000:]
    except FileNotFoundError:
        return {}
    state: dict = {"status": "running", "latest_metrics": {}}
    for line in lines:
        match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
        if match:
            state["iteration"] = int(match.group(1))
            state["max_iterations"] = int(match.group(2))
        match = re.match(r"\s*(Mean reward|Episode_Reward/[^:]+|Metrics/object_pose/[^:]+|Success rate|Episode_Termination/[^:]+):\s*(.*)", line)
        if match:
            state["latest_metrics"][match.group(1)] = match.group(2).strip()
    return state


def derive(state: dict) -> dict:
    metrics = state.get("latest_metrics", {})
    def number(key: str):
        value = metrics.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    turns = number("Metrics/object_pose/unscrew_turns_max")
    contact = number("Episode_Reward/any_finger_contact")
    success = number("Metrics/object_pose/success")
    failure = number("Episode_Reward/failure")
    if success is None:
        success = number("Success rate")
        if success is not None and success > 1:
            success /= 100.0
    return {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": state.get("status"),
        "iteration": state.get("iteration", state.get("completed_iterations")),
        "mean_reward": number("Mean reward"),
        "turns_max": turns,
        "turns_final": number("Metrics/object_pose/unscrew_turns_final"),
        "reached_1_turn": number("Metrics/object_pose/unscrew_reached_1_turn"),
        "reached_2_turns": number("Metrics/object_pose/unscrew_reached_2_turns"),
        "reached_3_turns": number("Metrics/object_pose/unscrew_reached_3_turns"),
        "contact_reward": contact,
        "success": success,
        "failure": failure,
        "released": number("Metrics/object_pose/released"),
        "alert": (
            "no_success_after_rotation"
            if turns is not None and turns >= 2.0 and (success or 0.0) == 0.0
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = read_state(args.state)
        if not state and args.log is not None:
            state = read_log(args.log)
        record = derive(state)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        args.output.with_suffix(".latest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        time.sleep(max(args.interval, 5.0))


if __name__ == "__main__":
    main()
