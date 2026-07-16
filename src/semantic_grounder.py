"""
Optional vision-language grounding for open-vocabulary scene understanding.

This module is intentionally lazy-loaded. The normal YOLO detector remains the
real-time path, while LocateAnything can be enabled for slower, higher-semantic
queries such as "traffic cone", "road sign text", or "pedestrian near the curb".
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from src.detector import Detection


_BOX_RE = re.compile(
    r"(?:<ref>(?P<label>.*?)</ref>)?"
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)


class SemanticGrounder:
    """
    Optional LocateAnything-backed open-vocabulary detector.

    The backend is only imported when ``enabled: true``. If the dependency is
    missing, the app keeps running with YOLO only and prints setup guidance.
    """

    def __init__(self, cfg: dict[str, Any], distance_cfg: dict[str, Any] | None = None) -> None:
        self.enabled = bool(cfg.get("enabled", False))
        self.backend = cfg.get("backend", "locate_anything")
        self.model = cfg.get("model", "nvidia/LocateAnything-3B")
        self.queries = [str(q) for q in cfg.get("queries", []) if str(q).strip()]
        self.every_n_frames = max(1, int(cfg.get("every_n_frames", 30)))
        self.cache_results = bool(cfg.get("cache_results", True))
        self.duplicate_iou = float(cfg.get("duplicate_iou", 0.55))
        self.default_confidence = float(cfg.get("default_confidence", 0.50))
        self.use_batch_runtime = bool(cfg.get("use_batch_runtime", False))
        self.attn = cfg.get("attn")
        self.vision_attn = cfg.get("vision_attn")
        self.scheduler = cfg.get("scheduler")

        distance_cfg = distance_cfg or {}
        self._focal_length = float(distance_cfg.get("focal_length_px", 700.0))
        self._known_widths = distance_cfg.get("known_widths_m", {})
        self._worker: Any | None = None
        self._load_error: str | None = None
        self._cached: list[Detection] = []
        self._class_ids: dict[str, int] = {}

        if self.enabled:
            self._load_backend()

    @property
    def available(self) -> bool:
        return self.enabled and self._worker is not None

    def run(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Run semantic grounding on schedule and return cached/new boxes."""
        if not self.available or not self.queries:
            return []
        if frame_id % self.every_n_frames != 0:
            return list(self._cached) if self.cache_results else []

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image

            image = Image.fromarray(rgb)
            answer = self._query_backend(image)
            detections = self._parse_answer(answer, frame.shape[1], frame.shape[0])
            self._cached = detections
            return list(detections)
        except Exception as exc:  # noqa: BLE001
            print(f"[SemanticGrounder] Inference skipped: {exc}")
            return list(self._cached) if self.cache_results else []

    def merge(self, yolo_detections: list[Detection], semantic_detections: list[Detection]) -> list[Detection]:
        """Append semantic boxes unless they duplicate an existing YOLO box."""
        merged = list(yolo_detections)
        for sem in semantic_detections:
            duplicate = any(
                sem.class_name == det.class_name and _iou(sem, det) >= self.duplicate_iou
                for det in yolo_detections
            )
            if not duplicate:
                merged.append(sem)
        merged.sort(key=lambda d: d.confidence, reverse=True)
        return merged

    def _load_backend(self) -> None:
        if self.backend != "locate_anything":
            self._load_error = f"Unsupported backend: {self.backend}"
            print(f"[SemanticGrounder] {self._load_error}")
            return
        try:
            from locateanything_worker import LocateAnythingWorker

            kwargs: dict[str, Any] = {}
            if self.use_batch_runtime:
                kwargs["use_batch_runtime"] = True
            if self.attn:
                kwargs["attn"] = self.attn
            if self.vision_attn:
                kwargs["vision_attn"] = self.vision_attn
            if self.scheduler:
                kwargs["scheduler"] = self.scheduler
            self._worker = LocateAnythingWorker(self.model, **kwargs)
            print(
                f"[SemanticGrounder] LocateAnything ready | "
                f"queries={len(self.queries)} | every={self.every_n_frames} frames"
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            print(
                "[SemanticGrounder] LocateAnything disabled. Install NVlabs/Eagle "
                "Embodied and make locateanything_worker importable to enable it. "
                f"Reason: {exc}"
            )

    def _query_backend(self, image: Any) -> str:
        assert self._worker is not None
        result = self._worker.detect(image, self.queries)
        if isinstance(result, dict):
            return str(result.get("answer", ""))
        return str(result)

    def _parse_answer(self, answer: str, frame_w: int, frame_h: int) -> list[Detection]:
        detections: list[Detection] = []
        for match in _BOX_RE.finditer(answer):
            label = _clean_label(match.group("label")) or "semantic object"
            coords = [int(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
            if coords[0] == coords[2] or coords[1] == coords[3]:
                continue

            x1, y1, x2, y2 = _scale_box(coords, frame_w, frame_h)
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    self._class_id(label),
                    label,
                    self.default_confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                    None,
                    self._focal_length,
                    self._known_widths,
                )
            )
        return detections

    def _class_id(self, label: str) -> int:
        if label not in self._class_ids:
            self._class_ids[label] = -1000 - len(self._class_ids)
        return self._class_ids[label]


def _clean_label(label: str | None) -> str:
    if not label:
        return ""
    return re.sub(r"\s+", " ", label.strip().lower())


def _scale_box(coords: list[int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [max(0, min(1000, v)) for v in coords]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return (
        int(round(left / 1000 * (frame_w - 1))),
        int(round(top / 1000 * (frame_h - 1))),
        int(round(right / 1000 * (frame_w - 1))),
        int(round(bottom / 1000 * (frame_h - 1))),
    )


def _iou(a: Detection, b: Detection) -> float:
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
