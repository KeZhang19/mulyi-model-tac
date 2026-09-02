#!/usr/bin/env python3
"""Detect and label black marker centers in Vitai reference images."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def detect_markers(
    image: np.ndarray, min_y: float, max_y: float
) -> list[tuple[float, float, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_response = cv2.subtract(cv2.GaussianBlur(gray, (0, 0), 7), gray)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 5
    params.maxThreshold = 100
    params.thresholdStep = 2
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 4
    params.maxArea = 600
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.filterByConvexity = True
    params.minConvexity = 0.45
    params.filterByInertia = True
    params.minInertiaRatio = 0.15
    params.minDistBetweenBlobs = 5

    height, width = gray.shape
    return [
        (float(point.pt[0]), float(point.pt[1]), float(point.size))
        for point in cv2.SimpleBlobDetector_create(params).detect(dark_response)
        if 0 <= point.pt[0] < width
        and min_y < point.pt[1] < min(height, max_y)
        and not (point.pt[0] < 40 and point.pt[1] < 30)
    ]


def save_annotation(
    image: np.ndarray,
    markers: list[tuple[float, float, float]],
    output_path: Path,
) -> None:
    scale = 4
    annotated = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    for index, (u_px, v_px, diameter_px) in enumerate(markers, start=2):
        center = (round(u_px * scale), round(v_px * scale))
        radius = max(8, round(diameter_px * scale / 2 + 3))
        cv2.circle(annotated, center, radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(
            annotated,
            center,
            (0, 255, 255),
            cv2.MARKER_CROSS,
            9,
            2,
            cv2.LINE_AA,
        )
        label = f"M{index:03d}"
        label_position = (center[0] + 6, center[1] - 6)
        cv2.putText(
            annotated,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write {output_path}")


def main() -> None:
    default_dataset = (
        Path(__file__).resolve().parents[1] / "vitai_4Fingers-320*240"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=default_dataset)
    parser.add_argument(
        "--references",
        nargs="+",
        default=["reference_004.png", "reference_025.png", "reference_050.png"],
    )
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--min-y", type=float, default=15.0)
    parser.add_argument("--max-y", type=float, default=235.0)
    args = parser.parse_args()

    image_paths = sorted(args.dataset.glob("reference_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No reference images found in {args.dataset}")
    images = [cv2.imread(str(path)) for path in image_paths]
    if any(image is None for image in images):
        raise RuntimeError("Failed to read one or more reference images")
    if len({image.shape for image in images}) != 1:
        raise ValueError("Reference images do not have a common image size")

    median_image = np.median(np.stack(images), axis=0).astype(np.uint8)
    reference_markers = sorted(
        detect_markers(median_image, args.min_y, args.max_y),
        key=lambda point: (point[1], point[0]),
    )
    if len(reference_markers) != args.expected_count:
        raise RuntimeError(
            f"Detected {len(reference_markers)} median markers; "
            f"expected {args.expected_count}"
        )

    output_dir = args.dataset / "marker_annotations"
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "reference_median.png"), median_image)
    save_annotation(
        median_image,
        reference_markers,
        output_dir / "reference_median_markers.png",
    )

    rows = [
        ("reference_median.png", index, *point)
        for index, point in enumerate(reference_markers, start=2)
    ]
    for name in args.references:
        image = cv2.imread(str(args.dataset / name))
        if image is None:
            raise FileNotFoundError(args.dataset / name)
        save_annotation(
            image,
            reference_markers,
            output_dir / f"{Path(name).stem}_markers.png",
        )

    csv_path = output_dir / "marker_centers.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "marker_id", "u_px", "v_px", "diameter_px"])
        writer.writerows(
            (name, f"M{index:03d}", f"{u_px:.3f}", f"{v_px:.3f}", f"{size:.3f}")
            for name, index, u_px, v_px, size in rows
        )
    print(
        f"Wrote {len(args.references) + 1} annotations and "
        f"{len(rows)} marker rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
