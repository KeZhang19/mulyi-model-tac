from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Copy a USD and disable instanceable flags on all prims.")
parser.add_argument("--src", required=True, help="Input USD path.")
parser.add_argument("--dst", required=True, help="Output USD path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd  # noqa: E402


src = Path(args.src).expanduser().resolve()
dst = Path(args.dst).expanduser().resolve()
dst.parent.mkdir(parents=True, exist_ok=True)

stage = Usd.Stage.Open(str(src))
if stage is None:
    raise RuntimeError(f"Could not open USD: {src}")

stage.GetRootLayer().Export(str(dst))
stage = Usd.Stage.Open(str(dst))
if stage is None:
    raise RuntimeError(f"Could not open copied USD: {dst}")

total = 0
changed = 0
stack = [stage.GetDefaultPrim()]
while stack:
    prim = stack.pop(0)
    if not prim or not prim.IsValid():
        continue
    total += 1
    if prim.IsInstance() or prim.IsInstanceable():
        prim.SetInstanceable(False)
        changed += 1
    stack.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))

stage.GetRootLayer().Save()
print(f"[INFO] saved {dst}")
print(f"[INFO] prims: {total}, instanceable disabled: {changed}")

simulation_app.close()
