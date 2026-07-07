"""
Object detection module — wraps Ultralytics YOLO with ByteTrack/BoT-SORT.

Responsibilities
----------------
* Load a YOLO model once at startup.
* Run inference + tracking on individual BGR frames via ``Tracker``.
* Filter results to the classes of interest from the pipeline config.
* Provide a monocular distance estimate (placeholder).
* Annotate frames with bounding boxes, track IDs, and distance labels.

Upgrade notes (v2)
------------------
* Config-dict driven — no more hardcoded constants from src/config.py.
* Integrates src.tracker.Tracker for ByteTrack / BoT-SORT.
* Bounding-box colours and class names loaded from config.
* show_track_id / show_distance flags respected from config.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.tracker import Tracker


# ── Detection data class ──────────────────────────────────────────────────────

class Detection:
    """
    Immutable data container for one detected object in a single frame.

    Attributes
    ----------
    class_id, class_name, confidence : object identity
    x1, y1, x2, y2                  : bounding box (pixel coords)
    width, height, cx, cy            : derived geometry
    track_id                         : persistent ID from ByteTrack (or None)
    estimated_distance_m             : monocular distance estimate (or None)
    """

    __slots__ = (
        "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "width", "height", "cx", "cy",
        "estimated_distance_m",
        "track_id",
    )

    def __init__(
        self,
        class_id:   int,
        class_name: str,
        confidence: float,
        x1: int, y1: int, x2: int, y2: int,
        track_id:   int | None = None,
        focal_length_px: float = 700.0,
        known_widths: dict[str, float] | None = None,
    ) -> None:
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.width  = x2 - x1
        self.height = y2 - y1
        self.cx     = (x1 + x2) // 2
        self.cy     = (y1 + y2) // 2
        self.track_id = track_id
        self.estimated_distance_m = _estimate_distance(
            class_name, self.width, focal_length_px, known_widths or {}
        )

    def __repr__(self) -> str:
        dist = (
            f"{self.estimated_distance_m:.1f} m"
            if self.estimated_distance_m is not None
            else "N/A"
        )
        tid = f"#{self.track_id} " if self.track_id is not None else ""
        return (
            f"Detection({tid}{self.class_name} "
            f"conf={self.confidence:.2f} dist={dist})"
        )


# ── Distance estimation ───────────────────────────────────────────────────────

def _estimate_distance(
    class_name:      str,
    bbox_width_px:   int,
    focal_length_px: float,
    known_widths:    dict[str, float],
) -> float | None:
    """
    Monocular distance estimate using the apparent-width formula.

        distance_m = (known_width_m × focal_length_px) / bbox_width_px

    ⚠️  PLACEHOLDER — first-order approximation only.
        For production accuracy: calibrate the focal length with a
        checkerboard pattern, or use stereo vision / a depth sensor.

    Returns None when the class width is unknown or the bbox is degenerate.
    """
    known_width = known_widths.get(class_name)
    if known_width is None or bbox_width_px <= 0:
        return None
    return round((known_width * focal_length_px) / bbox_width_px, 2)


# ── Detector ──────────────────────────────────────────────────────────────────

class Detector:
    """
    YOLO-based object detector with integrated ByteTrack tracking.

    Parameters
    ----------
    cfg : dict
        The full pipeline config dict (all sections).
    """

    _DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)

    def __init__(self, cfg: dict[str, Any]) -> None:
        model_cfg   = cfg["model"]
        cls_cfg     = cfg["classes"]
        dist_cfg    = cfg.get("distance", {})
        disp_cfg    = cfg.get("display", {})

        self._model_name    = model_cfg["name"]
        self._conf          = model_cfg["confidence"]
        self._iou           = model_cfg["iou"]
        self._device        = model_cfg.get("device", "") or ""
        self._imgsz         = model_cfg.get("imgsz", 640)
        self._max_det       = int(model_cfg.get("max_det", 100))
        self._agnostic_nms  = bool(model_cfg.get("agnostic_nms", False))
        self._warmup        = bool(model_cfg.get("warmup", True))
        self._class_ids     = cls_cfg["ids"] or None   # None = detect all
        self._class_names: dict[int, str] = {
            int(k): v for k, v in (cls_cfg.get("names") or {}).items()
        }
        self._colors: dict[str, tuple[int, int, int]] = {
            k: tuple(v)  # type: ignore[arg-type]
            for k, v in (cls_cfg.get("colors") or {}).items()
        }
        self._focal_length  = dist_cfg.get("focal_length_px", 700.0)
        self._known_widths  = dist_cfg.get("known_widths_m", {})
        self._show_track_id = disp_cfg.get("show_track_id", True)
        self._show_distance = disp_cfg.get("show_distance", True)
        self._font_scale    = disp_cfg.get("font_scale",    0.55)
        self._box_thickness = disp_cfg.get("box_thickness", 2)
        self.last_inference_ms = 0.0

        print(f"[Detector] Loading model: {self._model_name} …")
        self._model   = YOLO(self._model_name)
        self._tracker = Tracker(self._model, model_cfg.get("tracker", "bytetrack.yaml"))
        # Merge YOLO's built-in names so every class ID has a label
        for cid, cname in self._model.names.items():
            self._class_names.setdefault(int(cid), cname)
        active = "ALL" if self._class_ids is None else str(len(self._class_ids))
        print(
            f"[Detector] Ready.  Classes: {active} | "
            f"tracker: {model_cfg.get('tracker', 'bytetrack.yaml')}"
        )
        if self._warmup:
            self._warmup_model()

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference + tracking on a single BGR frame.

        Returns a list of Detection objects filtered to the configured
        classes of interest, sorted by descending confidence.
        """
        if frame is None or frame.size == 0:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Detector.detect expects a non-empty BGR image with 3 channels.")

        frame_h, frame_w = frame.shape[:2]
        started = time.perf_counter()
        results = self._tracker.run(
            frame,
            conf=self._conf,
            iou=self._iou,
            classes=self._class_ids,
            device=self._device,
            imgsz=self._imgsz,
            max_det=self._max_det,
            agnostic_nms=self._agnostic_nms,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                x1, y1, x2, y2 = self._clip_xyxy(box.xyxy[0].tolist(), frame_w, frame_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                track_id: int | None = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                class_name = self._class_names.get(cls_id, f"cls_{cls_id}")
                detections.append(Detection(
                    cls_id, class_name, conf,
                    x1, y1, x2, y2,
                    track_id,
                    self._focal_length,
                    self._known_widths,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return detections

    def _warmup_model(self) -> None:
        """Run one tiny inference so first live frame does not pay setup cost."""
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model.predict(
                source=dummy,
                conf=self._conf,
                iou=self._iou,
                classes=self._class_ids,
                device=self._device or None,
                imgsz=self._imgsz,
                max_det=1,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Detector] Warmup skipped: {exc}")

    @staticmethod
    def _clip_xyxy(values: list[float], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Clamp a YOLO xyxy box to valid image coordinates."""
        x1, y1, x2, y2 = values
        return (
            max(0, min(frame_w - 1, int(round(x1)))),
            max(0, min(frame_h - 1, int(round(y1)))),
            max(0, min(frame_w - 1, int(round(x2)))),
            max(0, min(frame_h - 1, int(round(y2)))),
        )

    # ── Frame annotation ──────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame:      np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto *frame* in-place.

        Returns the same frame object for convenience.
        """
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_thick = max(1, self._box_thickness - 1)

        for det in detections:
            color = self._colors.get(det.class_name, self._DEFAULT_COLOR)

            # Bounding box
            cv2.rectangle(
                frame,
                (det.x1, det.y1), (det.x2, det.y2),
                color, self._box_thickness,
            )

            # Label parts
            parts: list[str] = []
            if self._show_track_id and det.track_id is not None:
                parts.append(f"#{det.track_id}")
            parts.append(f"{det.class_name} {det.confidence:.2f}")
            if self._show_distance and det.estimated_distance_m is not None:
                parts.append(f"~{det.estimated_distance_m:.1f}m")
            label = "  ".join(parts)

            # Label background
            (tw, th), bl = cv2.getTextSize(label, font, self._font_scale, font_thick)
            ly = max(det.y1 - 6, th + 4)
            cv2.rectangle(
                frame,
                (det.x1, ly - th - bl),
                (det.x1 + tw + 4, ly + bl),
                color, cv2.FILLED,
            )
            cv2.putText(
                frame, label,
                (det.x1 + 2, ly - 2),
                font, self._font_scale,
                (0, 0, 0), font_thick, cv2.LINE_AA,
            )

        return frame
