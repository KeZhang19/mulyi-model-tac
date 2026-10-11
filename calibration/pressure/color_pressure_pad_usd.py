#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind a visible material to pressure-pad visuals in a converted USD.")
    parser.add_argument("usd", type=Path)
    parser.add_argument("--urdf", type=Path, required=True, help="URDF containing pressure_pad declarations.")
    parser.add_argument("--rgba", default="0 0.85 1 1", help="Pressure pad RGBA color, e.g. '0 0.85 1 1'.")
    parser.add_argument("--material-path", default="/Looks/pressure_pad_cyan")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    usd_path = args.usd.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    rgba = _parse_rgba(args.rgba)
    pressure_links, visual_links = _pressure_and_visual_links_from_urdf(urdf_path)
    pxr, app = _load_usd_modules()
    Gf, Sdf, Usd, UsdShade = pxr

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Could not open USD: {usd_path}")

    material = _define_preview_material(Gf, Sdf, UsdShade, stage, Sdf.Path(args.material_path), rgba)
    bound: list[str] = []
    missing: list[str] = []
    for link_name in visual_links:
        link_prim = _find_prim_by_name(stage, link_name)
        if link_prim is None:
            missing.append(link_name)
            continue
        visual_prim = link_prim.GetChild("visuals") if link_prim.GetChild("visuals").IsValid() else link_prim
        UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound.append(str(visual_prim.GetPath()))

    stage.GetRootLayer().Save()
    payload = {
        "usd": str(usd_path),
        "urdf": str(urdf_path),
        "material_path": str(material.GetPath()),
        "rgba": list(rgba),
        "pressure_pad_count": len(pressure_links),
        "colored_visual_link_count": len(visual_links),
        "bound_visual_prims": bound,
        "missing_visual_links": missing,
        "passed": len(missing) == 0 and len(bound) == len(visual_links),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(text + "\n", encoding="utf-8")
    if app is not None:
        app.close()
    return 0 if payload["passed"] else 2


def _load_usd_modules() -> tuple[tuple[Any, Any, Any, Any], Any | None]:
    try:
        from pxr import Gf, Sdf, Usd, UsdShade

        return (Gf, Sdf, Usd, UsdShade), None
    except ModuleNotFoundError:  # pragma: no cover - requires Isaac/Kit Python.
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True).app
        from pxr import Gf, Sdf, Usd, UsdShade

        return (Gf, Sdf, Usd, UsdShade), app


def _pressure_and_visual_links_from_urdf(path: Path) -> tuple[list[str], list[str]]:
    root = ET.parse(path).getroot()
    links: list[str] = []
    for link in root.findall("link"):
        if link.find("pressure_pad") is not None:
            link_name = str(link.attrib.get("name", "")).strip()
            if link_name:
                links.append(link_name)
    pressure_links = sorted(set(links))
    available = {str(link.attrib.get("name", "")) for link in root.findall("link")}
    visual_links: list[str] = []
    seen: set[str] = set()
    for link_name in pressure_links:
        candidates = [link_name]
        if link_name.endswith("_touch_link"):
            prefix = link_name[: -len("_touch_link")]
            candidates.extend([f"{prefix}_rubber_link", f"{prefix}_tubber_link"])
        for candidate in candidates:
            if candidate in available and candidate not in seen:
                visual_links.append(candidate)
                seen.add(candidate)
    return pressure_links, visual_links


def _parse_rgba(text: str) -> tuple[float, float, float, float]:
    values = str(text).replace(",", " ").split()
    if len(values) != 4:
        raise ValueError("--rgba must contain four floats")
    rgba = tuple(float(value) for value in values)
    if any(value < 0.0 or value > 1.0 for value in rgba):
        raise ValueError("--rgba values must be in [0, 1]")
    return rgba


def _define_preview_material(
    Gf: Any,
    Sdf: Any,
    UsdShade: Any,
    stage: Any,
    material_path: Any,
    rgba: tuple[float, float, float, float],
) -> Any:
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(rgba[0], rgba[1], rgba[2]))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _find_prim_by_name(stage: Any, name: str) -> Any | None:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


if __name__ == "__main__":
    raise SystemExit(main())
