#!/usr/bin/env python3
"""Visualize restoration or clean latent-only decoding on test samples.

The training script deliberately reports scalar losses only.  This companion
script keeps the visual comparison auditable: all Depth panels share one
physical color scale, and Marker examples can be selected by actual motion
magnitude instead of accidentally showing only near-zero arrows.  The
``clean-latent`` mode disables every dynamic decoder bypass and therefore
visualizes what is recoverable from the shared latent itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation import (  # noqa: E402
    CrossModalTactileNetworkCfg,
    RobustCrossModalTactileNetwork,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    EpisodeSplits,
    NpzTactileDataset,
    TactileNormalization,
    degrade_tactile_inputs,
    load_complete_manifest,
)


MODALITIES = ("rgb", "depth", "marker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/revo3_index_sweep_parallel_v1"))
    parser.add_argument("--resume", type=Path, required=True, help="Checkpoint to visualize.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-backend", choices=("auto", "mmap", "npz"), default="auto")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("restoration", "clean-latent"),
        default="restoration",
        help=(
            "Restore one degraded modality, or encode fully clean inputs and "
            "decode all modalities from the shared latent without dynamic bypasses."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=("high-motion", "high-depth", "mixed"),
        default="mixed",
        help="Choose frames with visible marker motion, strong Depth, or both.",
    )
    parser.add_argument("--severity", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--marker-display-scale", type=float, default=5.0)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if not 0.0 <= args.severity <= 1.0:
        parser.error("--severity must be in [0, 1]")
    if args.marker_display_scale <= 0.0:
        parser.error("--marker-display-scale must be positive")
    return args


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path.expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Unsupported checkpoint: {path}")
    return payload


def _sample_score(sample: dict[str, torch.Tensor]) -> tuple[float, float, float]:
    valid = sample["marker_valid"]
    motion_px = torch.linalg.vector_norm(sample["marker"][:, 2:4] * 5.0, dim=-1)
    marker_score = float(motion_px.masked_select(valid).max()) if bool(valid.any()) else 0.0
    depth_score = float(sample["depth"].max())
    contact = float(bool(sample["contact"]))
    return marker_score, depth_score, contact


def _select_indices(
    dataset: NpzTactileDataset,
    *,
    count: int,
    selection: str,
) -> list[int]:
    scored: list[tuple[int, int, int, int, float, float, float]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        marker_score, depth_score, contact = _sample_score(sample)
        scored.append(
            (
                index,
                int(sample["episode_id"]),
                int(sample["episode_step"]),
                int(sample["sweep_stage"]),
                marker_score,
                depth_score,
                contact,
            )
        )
    if selection == "high-motion":
        scored.sort(key=lambda item: (item[4], item[5]), reverse=True)
    elif selection == "high-depth":
        scored.sort(key=lambda item: (item[5], item[4]), reverse=True)
    else:
        # Mixed selection gives the visual report both a marker-rich frame and
        # a shallow/strong Depth frame, while avoiding adjacent steps from one
        # episode dominating the page.
        scored.sort(key=lambda item: (item[4] + 8.0 * item[5], item[4], item[5]), reverse=True)
    selected: list[int] = []
    selected_episodes: set[int] = set()
    for item in scored:
        if item[1] in selected_episodes:
            continue
        selected.append(item[0])
        selected_episodes.add(item[1])
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected.extend(item[0] for item in scored if item[0] not in selected)
        selected = selected[:count]
    return selected


def _rgb_image(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _depth_image(value: torch.Tensor, normalization: TactileNormalization) -> np.ndarray:
    return value.detach().cpu().squeeze(0).numpy() * float(normalization.depth_scale_m) * 1000.0


def _draw_marker(
    axis: Any,
    image: np.ndarray,
    marker: torch.Tensor,
    valid: torch.Tensor,
    *,
    width: int,
    height: int,
    display_scale: float,
    title: str,
) -> None:
    axis.imshow(image)
    points = marker.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    points = points[valid_np]
    if points.size:
        x0 = points[:, 0] * max(1, width - 1)
        y0 = points[:, 1] * max(1, height - 1)
        dx = points[:, 2] * 5.0 * display_scale
        dy = points[:, 3] * 5.0 * display_scale
        axis.quiver(
            x0,
            y0,
            dx,
            dy,
            color="#ff7f0e",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        axis.scatter(x0, y0, s=5, c="black", linewidths=0.3)
    axis.set_title(title, fontsize=9)
    axis.set_xlim(0, width - 1)
    axis.set_ylim(height - 1, 0)
    axis.axis("off")


def _safe_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else 0.0


def _render_modality(
    modality: str,
    *,
    batch: dict[str, torch.Tensor],
    degraded: dict[str, torch.Tensor],
    restored: dict[str, torch.Tensor],
    normalization: TactileNormalization,
    output_path: Path,
    marker_display_scale: float,
    input_label: str = "Degraded",
    result_label: str = "Restored",
    figure_prefix: str = "Restoration test | degraded modality",
) -> dict[str, Any]:
    count = int(batch["rgb"].shape[0])
    height, width = normalization.image_height, normalization.image_width
    fig, axes = plt.subplots(count, 3, figsize=(13.5, max(3.4, 3.2 * count)), squeeze=False)
    labels = (input_label, result_label, "Clean target")
    errors: list[float] = []
    for row in range(count):
        metadata = (
            f"ep={int(batch['episode_id'][row])} step={int(batch['episode_step'][row])} "
            f"force={float(batch['target_force_n'][row]):g}N"
        )
        if modality == "rgb":
            values = (
                degraded["rgb"][row],
                restored["restored_rgb"][row],
                batch["rgb"][row],
            )
            errors.append(float((values[1] - values[2]).abs().mean()))
            for column, value in enumerate(values):
                axes[row, column].imshow(_rgb_image(value))
                axes[row, column].set_title(
                    f"{labels[column]}\n{metadata}" if column == 0 else labels[column], fontsize=9
                )
                axes[row, column].axis("off")
        elif modality == "depth":
            values = (
                degraded["depth"][row],
                restored["restored_depth"][row],
                batch["depth"][row],
            )
            errors.append(float((values[1] - values[2]).abs().mean() * 1000.0 * normalization.depth_scale_m))
            for column, value in enumerate(values):
                image = _depth_image(value, normalization)
                axes[row, column].imshow(
                    image, cmap="magma", vmin=0.0, vmax=float(normalization.depth_scale_m) * 1000.0
                )
                axes[row, column].set_title(
                    f"{labels[column]}\n{metadata}" if column == 0 else labels[column], fontsize=9
                )
                axes[row, column].axis("off")
        else:
            clean_motion = batch["marker"][row, :, 2:4]
            restored_motion = restored["restored_marker_motion"][row]
            valid = batch["marker_valid"][row]
            errors.append(
                _safe_mean(
                    torch.linalg.vector_norm((restored_motion - clean_motion) * 5.0, dim=-1),
                    valid,
                )
            )
            marker_values = (
                degraded["marker"][row],
                torch.cat((batch["marker"][row, :, :2], restored_motion), dim=-1),
                batch["marker"][row],
            )
            for column, value in enumerate(marker_values):
                _draw_marker(
                    axes[row, column],
                    _rgb_image(batch["rgb"][row]),
                    value,
                    valid,
                    width=width,
                    height=height,
                    display_scale=marker_display_scale,
                    title=f"{labels[column]}\n{metadata}" if column == 0 else labels[column],
                )
    metric_name = {"rgb": "RGB MAE", "depth": "Depth MAE (mm)", "marker": "Marker EPE (px)"}[modality]
    fig.suptitle(
        f"{figure_prefix}: {modality.upper()} | {metric_name}={float(np.mean(errors)):.4g}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"metric": metric_name, "errors": errors}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.resume)
    dataset_root, manifest = load_complete_manifest(args.dataset)
    checkpoint_model_cfg = dict(checkpoint["model_cfg"])
    checkpoint_model_cfg.setdefault("rgb_depth_spatial_skip", False)
    checkpoint_model_cfg.setdefault("depth_rgb_spatial_skip", False)
    checkpoint_model_cfg.setdefault("marker_image_spatial_context", False)
    checkpoint_model_cfg.setdefault("rgb_reference_max_delta", 0.25)
    normalization = TactileNormalization(**checkpoint["normalization"])
    splits = EpisodeSplits.from_json(checkpoint["episode_splits"])
    dataset = NpzTactileDataset(
        dataset_root,
        episode_ids=splits.test,
        normalization=normalization,
        backend=args.dataset_backend,
    )
    selected_indices = _select_indices(
        dataset, count=int(args.num_samples), selection=str(args.selection)
    )
    batch = default_collate([dataset[index] for index in selected_indices])
    batch = {name: value.to(device) if value.is_floating_point() else value.to(device) for name, value in batch.items()}
    model = RobustCrossModalTactileNetwork(
        CrossModalTactileNetworkCfg(**checkpoint_model_cfg)
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "checkpoint": str(args.resume.expanduser().resolve()),
        "mode": str(args.mode),
        "selection": str(args.selection),
        "severity": float(args.severity) if args.mode == "restoration" else None,
        "selected_samples": [],
        "tests": {},
    }
    for row, index in enumerate(selected_indices):
        summary["selected_samples"].append(
            {
                "dataset_index": int(index),
                "episode_id": int(batch["episode_id"][row]),
                "episode_step": int(batch["episode_step"][row]),
                "target_force_n": float(batch["target_force_n"][row]),
                "marker_max_motion_px": float(
                    torch.linalg.vector_norm(batch["marker"][row, :, 2:4] * 5.0, dim=-1)
                    .masked_select(batch["marker_valid"][row])
                    .max()
                ),
                "depth_max_mm": float(batch["depth"][row].max() * normalization.depth_scale_m * 1000.0),
            }
        )

    with torch.no_grad():
        if args.mode == "clean-latent":
            latent = model.encode(
                rgb=batch["rgb"],
                depth=batch["depth"],
                marker=batch["marker"],
                marker_valid_mask=batch["marker_valid"],
            )
            decoded = model.decode_from_latent(
                latent,
                rgb_reference=batch["rgb_reference"],
                marker_positions=batch["marker"][..., :2],
                marker_valid_mask=batch["marker_valid"],
            )
            clean_inputs = {
                "rgb": batch["rgb"],
                "depth": batch["depth"],
                "marker": batch["marker"],
            }
            latent_reconstruction = {
                "restored_rgb": decoded["rgb_recon"],
                "restored_depth": decoded["depth_recon"],
                "restored_marker_motion": decoded["marker_recon"],
            }
            for modality in MODALITIES:
                summary["tests"][modality] = _render_modality(
                    modality,
                    batch=batch,
                    degraded=clean_inputs,
                    restored=latent_reconstruction,
                    normalization=normalization,
                    output_path=output_dir / f"clean_latent_{modality}.png",
                    marker_display_scale=float(args.marker_display_scale),
                    input_label="Clean input",
                    result_label="Latent-only reconstruction",
                    figure_prefix="Clean input -> shared latent -> latent-only decoder",
                )
        else:
            for modality_index, modality in enumerate(MODALITIES):
                generator = torch.Generator(device=device).manual_seed(
                    int(args.seed) + modality_index
                )
                degraded, _, _, _ = degrade_tactile_inputs(
                    batch,
                    normalization=normalization,
                    generator=generator,
                    degraded_modality=modality_index,
                    min_severity=float(args.severity),
                    max_severity=float(args.severity),
                )
                restored = model.restore_degraded_observation(
                    rgb=degraded["rgb"],
                    rgb_reference=batch["rgb_reference"],
                    depth=degraded["depth"],
                    marker=degraded["marker"],
                    marker_valid_mask=degraded["marker_valid"],
                    quality_threshold=float(
                        checkpoint_model_cfg.get("reliability_gate_threshold", 0.9)
                    ),
                )
                summary["tests"][modality] = _render_modality(
                    modality,
                    batch=batch,
                    degraded=degraded,
                    restored=restored,
                    normalization=normalization,
                    output_path=output_dir / f"{modality}_restoration.png",
                    marker_display_scale=float(args.marker_display_scale),
                )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
