"""Launch one Isaac Sim rank with an isolated Kit data root."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import time


torchrun_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", str(torchrun_local_rank)))
visible_gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
physical_gpu = int(visible_gpu_ids[torchrun_local_rank].strip())

# Isaac Sim 5.1 USD Runtime only accepts logical cuda:0. Isolate each rank to
# one physical GPU and preserve only the global distributed rank.
os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
os.environ["LOCAL_RANK"] = "0"

rank_root = Path(os.environ["BULB_TEACHER_PORTABLE_ROOT"]) / f"rank{global_rank}"
rank_root.mkdir(parents=True, exist_ok=False)
for name in ("cache", "config", "data", "tmp"):
    (rank_root / name).mkdir()
os.environ["XDG_CACHE_HOME"] = str(rank_root / "cache")
os.environ["XDG_CONFIG_HOME"] = str(rank_root / "config")
os.environ["XDG_DATA_HOME"] = str(rank_root / "data")
os.environ["TMPDIR"] = str(rank_root / "tmp")

stagger_seconds = float(os.environ.get("BULB_TEACHER_RANK_STAGGER_SECONDS", "20"))
time.sleep(global_rank * stagger_seconds)

from isaaclab.app import AppLauncher


original_resolve_device_settings = AppLauncher._resolve_device_settings


def resolve_device_settings_with_split_gpu(self, launcher_args):
    original_resolve_device_settings(self, launcher_args)
    launcher_args["active_gpu"] = physical_gpu
    launcher_args["physics_gpu"] = 0
    self.device_id = 0
    self.local_rank = 0


AppLauncher._resolve_device_settings = resolve_device_settings_with_split_gpu

print(
    "[RANK_BOOT] "
    f"global_rank={global_rank} physical_gpu={physical_gpu} "
    f"logical_device=cuda:0 portable_root={rank_root}",
    flush=True,
)

kit_args = (
    f"--portable-root {rank_root} "
    "--/renderer/multiGpu/enabled=false "
    "--/app/renderer/multiGpu/enabled=false "
    "--/renderer/gpuEnumeration/glInterop/enabled=false"
)
sys.argv.extend(["--kit_args", kit_args])
train_script = os.environ["BULB_TEACHER_TRAIN_SCRIPT"]
sys.path.insert(0, os.path.dirname(train_script))
sys.argv[0] = train_script
runpy.run_path(train_script, run_name="__main__")
