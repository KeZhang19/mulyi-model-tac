#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import (  # noqa: E402
    load_pressure_pad_specs_from_urdf,
    load_pressure_touch_links_from_urdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a URDF copy with default grid pressure_pad metadata on existing toucher/touch links. "
            "The pressure_pad tag does not create a separate pad link."
        )
    )
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--point-distance", type=float, default=0.0015)
    parser.add_argument("--normal-axis", type=int, default=2)
    parser.add_argument("--normal-offset", type=float, default=0.0)
    parser.add_argument("--normal-sign", type=float, default=1.0)
    parser.add_argument("--stiffness", type=float, default=5000.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--max-force", type=float, default=10.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--taxel-area", type=float, default=1.0)
    parser.add_argument(
        "--pressure-pad-color",
        default="0 0.85 1 1",
        help="RGBA visual color applied to the toucher links carrying pressure_pad metadata.",
    )
    parser.add_argument("--pressure-pad-material-name", default="pressure_pad_cyan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    out_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else urdf_path.with_name(f"{urdf_path.stem}.pressurepad{urdf_path.suffix}")
    )
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_by_name = {str(link.attrib.get("name", "")): link for link in root.findall("link")}
    pressure_links = load_pressure_touch_links_from_urdf(urdf_path)
    existing_links = {spec.link_name for spec in load_pressure_pad_specs_from_urdf(urdf_path, require_files=False)}
    missing_links = [link for link in pressure_links if link not in existing_links]
    attrs = _pad_attrs(args)
    added: list[str] = []
    skipped_missing_link: list[str] = []
    for link_name in missing_links:
        link = link_by_name.get(link_name)
        if link is None:
            skipped_missing_link.append(link_name)
            continue
        ET.SubElement(
            link,
            "pressure_pad",
            {
                "name": f"{link_name}_pressure_pad",
                **attrs,
            },
        )
        added.append(link_name)

    colored_links = _color_pressure_pad_links(
        root,
        sorted(set(pressure_links) | existing_links),
        rgba=_parse_rgba(args.pressure_pad_color),
        material_name=str(args.pressure_pad_material_name),
    )

    ET.indent(tree, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(
        json.dumps(
            {
                "input": str(urdf_path),
                "output": str(out_path),
                "pressure_touch_links": pressure_links,
                "existing_pressure_pad_links": sorted(existing_links),
                "added_pressure_pad_links": added,
                "colored_pressure_pad_links": colored_links,
                "skipped_missing_link_elements": skipped_missing_link,
                "rows": int(args.rows),
                "cols": int(args.cols),
                "taxel_count": int(args.rows) * int(args.cols),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _pad_attrs(args: argparse.Namespace) -> dict[str, str]:
    rows = int(args.rows)
    cols = int(args.cols)
    return {
        "rows": str(rows),
        "cols": str(cols),
        "taxel_count": str(rows * cols),
        "point_distance": f"{float(args.point_distance):.9g}",
        "normal_axis": str(int(args.normal_axis)),
        "normal_offset": f"{float(args.normal_offset):.9g}",
        "normal_sign": f"{float(args.normal_sign):.9g}",
        "origin_semantics": "pad_surface",
        "stiffness": f"{float(args.stiffness):.9g}",
        "damping": f"{float(args.damping):.9g}",
        "max_force": f"{float(args.max_force):.9g}",
        "gain": f"{float(args.gain):.9g}",
        "bias": f"{float(args.bias):.9g}",
        "gamma": f"{float(args.gamma):.9g}",
        "threshold": f"{float(args.threshold):.9g}",
        "area": f"{float(args.taxel_area):.9g}",
    }


def _parse_rgba(text: str) -> str:
    values = str(text).replace(",", " ").split()
    if len(values) != 4:
        raise ValueError("--pressure-pad-color must contain four RGBA floats")
    rgba = [float(value) for value in values]
    if any(value < 0.0 or value > 1.0 for value in rgba):
        raise ValueError("--pressure-pad-color values must be in [0, 1]")
    return " ".join(f"{value:.9g}" for value in rgba)


def _color_pressure_pad_links(
    root: ET.Element,
    link_names: list[str],
    *,
    rgba: str,
    material_name: str,
) -> list[str]:
    colored: list[str] = []
    for link_name in _pressure_pad_visual_links(root, link_names):
        link = root.find(f"./link[@name='{link_name}']")
        if link is None:
            continue
        visuals = link.findall("visual")
        if not visuals:
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            visuals = [visual]
        for visual in visuals:
            material = visual.find("material")
            if material is None:
                material = ET.SubElement(visual, "material")
            material.set("name", material_name)
            color = material.find("color")
            if color is None:
                color = ET.SubElement(material, "color")
            color.set("rgba", rgba)
        colored.append(link_name)
    return colored


def _pressure_pad_visual_links(root: ET.Element, link_names: list[str]) -> list[str]:
    available = {str(link.attrib.get("name", "")) for link in root.findall("link")}
    out: list[str] = []
    seen: set[str] = set()
    for link_name in link_names:
        candidates = [link_name]
        if link_name.endswith("_touch_link"):
            prefix = link_name[: -len("_touch_link")]
            candidates.extend([f"{prefix}_rubber_link", f"{prefix}_tubber_link"])
        for candidate in candidates:
            if candidate in available and candidate not in seen:
                out.append(candidate)
                seen.add(candidate)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
