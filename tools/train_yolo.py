#!/usr/bin/env python3
"""
Train and evaluate a CAV-specific YOLO detector.

This is a thin, reproducible wrapper around Ultralytics so the project has one
known training command instead of a pile of ad-hoc terminal history.
"""

from __future__ import annotations

import argparse
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the CAV dataset.")
    parser.add_argument("--data", default="datasets/cav_yolo/data.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base weights or previous best.pt.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics choose batch size.")
    parser.add_argument("--device", default=None, help="Examples: cpu, mps, cuda:0.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--project", default="runs/cav", help="Training output parent directory.")
    parser.add_argument("--name", default="yolo_cav", help="Training run name.")
    parser.add_argument("--validate-only", action="store_true", help="Run validation without training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml not found: {data_path}. Build it with tools/build_yolo_dataset.py first.")

    _validate_dataset_yaml(data_path)
    from ultralytics import YOLO

    model = YOLO(args.model)

    if args.validate_only:
        metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)
        print(metrics)
        return 0

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    print(results)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        print(f"[Train] Best weights: {best.resolve()}")
        print(f"[Train] Use in configs/config.yaml: model.name: \"{best}\"")
    return 0


def _validate_dataset_yaml(path: Path) -> None:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    missing = []
    for rel in (data["train"], data["val"]):
        if not (root / rel).exists():
            missing.append(str(root / rel))
    if missing:
        raise SystemExit("Missing dataset directories:\n" + "\n".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
