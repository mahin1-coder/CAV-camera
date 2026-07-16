#!/usr/bin/env python3
"""
Build a YOLO-format training dataset from saved CAV frames and detections.

Input:
  - frame images saved by the app
  - logs/detections.jsonl rows containing frame_id, class_name, confidence, xyxy

Output follows the Ultralytics detection format:
  dataset/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    data.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "traffic cone",
    "construction barrel",
]


@dataclass(frozen=True)
class Box:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CAV logs + saved frames into an Ultralytics YOLO dataset."
    )
    parser.add_argument("--frames", default="datasets/raw_frames", help="Directory containing saved frame images.")
    parser.add_argument("--detections", default="logs/detections.jsonl", help="Detection JSONL file.")
    parser.add_argument("--out", default="datasets/cav_yolo", help="Output YOLO dataset directory.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Classes to include in this dataset.")
    parser.add_argument("--class-config", default=None, help="YAML file with a top-level classes list.")
    parser.add_argument("--min-conf", type=float, default=0.55, help="Minimum pseudo-label confidence.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of frames assigned to validation.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic train/val split seed.")
    parser.add_argument("--dedupe-iou", type=float, default=0.75, help="Drop same-class boxes above this IoU.")
    parser.add_argument("--copy-empty", action="store_true", help="Copy images with no accepted labels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames)
    detections_path = Path(args.detections)
    out_dir = Path(args.out)

    if not frames_dir.exists():
        raise SystemExit(f"Frame directory not found: {frames_dir}")
    if not detections_path.exists():
        raise SystemExit(f"Detection log not found: {detections_path}")

    classes = _load_classes(args.class_config) if args.class_config else args.classes
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    images = _index_images(frames_dir)
    detections = _load_detections(detections_path, class_to_id, args.min_conf, args.dedupe_iou)

    frame_ids = sorted(set(images) & set(detections))
    if args.copy_empty:
        frame_ids = sorted(set(images))
    if not frame_ids:
        raise SystemExit("No matching frames and detections found. Save frames while logging detections first.")

    rng = random.Random(args.seed)
    rng.shuffle(frame_ids)
    val_count = max(1, int(len(frame_ids) * args.val_ratio)) if len(frame_ids) > 1 else 0
    val_ids = set(frame_ids[:val_count])

    _prepare_dirs(out_dir)
    written_images = 0
    written_labels = 0

    for frame_id in frame_ids:
        split = "val" if frame_id in val_ids else "train"
        image_path = images[frame_id]
        boxes = detections.get(frame_id, [])
        if not boxes and not args.copy_empty:
            continue

        copied = _copy_image(image_path, out_dir / "images" / split)
        label_path = out_dir / "labels" / split / f"{copied.stem}.txt"
        rows = _to_yolo_rows(copied, boxes, class_to_id)
        label_path.write_text("".join(rows), encoding="utf-8")
        written_images += 1
        written_labels += len(rows)

    _write_data_yaml(out_dir, classes)
    _write_manifest(out_dir, args, classes, written_images, written_labels)
    print(f"[Dataset] images={written_images} labels={written_labels} out={out_dir.resolve()}")
    print(f"[Dataset] train with: yolo detect train data={out_dir / 'data.yaml'} model=yolo11n.pt imgsz=640")
    return 0


def _index_images(frames_dir: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in sorted(frames_dir.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame_id = _extract_frame_id(path)
        if frame_id is not None:
            images[frame_id] = path
    return images


def _load_classes(path: str) -> list[str]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        raise SystemExit(f"No top-level classes list found in {path}")
    return [str(name).strip().lower() for name in classes]


def _extract_frame_id(path: Path) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    if not digits:
        return None
    return int(digits[-1])


def _load_detections(
    path: Path,
    class_to_id: dict[str, int],
    min_conf: float,
    dedupe_iou: float,
) -> dict[int, list[Box]]:
    by_frame: dict[int, list[Box]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            class_name = str(row.get("class_name", "")).strip().lower()
            if class_name not in class_to_id:
                continue
            confidence = float(row.get("confidence") or 0.0)
            if confidence < min_conf:
                continue
            box = Box(
                class_name=class_name,
                confidence=confidence,
                x1=int(row["x1"]),
                y1=int(row["y1"]),
                x2=int(row["x2"]),
                y2=int(row["y2"]),
            )
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue
            by_frame[int(row["frame_id"])].append(box)

    return {frame_id: _dedupe_boxes(boxes, dedupe_iou) for frame_id, boxes in by_frame.items()}


def _dedupe_boxes(boxes: list[Box], threshold: float) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if any(box.class_name == other.class_name and _iou(box, other) >= threshold for other in kept):
            continue
        kept.append(box)
    return kept


def _iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    return inter / max(1, area_a + area_b - inter)


def _prepare_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_image(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _to_yolo_rows(image_path: Path, boxes: list[Box], class_to_id: dict[str, int]) -> list[str]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read copied image: {image_path}")
    height, width = image.shape[:2]
    rows: list[str] = []
    for box in boxes:
        x1 = max(0, min(width - 1, box.x1))
        y1 = max(0, min(height - 1, box.y1))
        x2 = max(0, min(width - 1, box.x2))
        y2 = max(0, min(height - 1, box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        rows.append(f"{class_to_id[box.class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return rows


def _write_data_yaml(out_dir: Path, classes: list[str]) -> None:
    import yaml

    data: dict[str, Any] = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    classes: list[str],
    image_count: int,
    label_count: int,
) -> None:
    import yaml

    manifest: dict[str, Any] = {
        "frames": str(Path(args.frames).resolve()),
        "detections": str(Path(args.detections).resolve()),
        "classes": classes,
        "min_conf": args.min_conf,
        "val_ratio": args.val_ratio,
        "dedupe_iou": args.dedupe_iou,
        "seed": args.seed,
        "image_count": image_count,
        "label_count": label_count,
    }
    with (out_dir / "manifest.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
