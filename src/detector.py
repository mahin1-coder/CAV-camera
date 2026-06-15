"""
Object detection module — wraps Ultralytics YOLO.

Responsibilities
----------------
* Load a YOLO model once at startup.
* Run inference on individual BGR frames.
* Filter results to the classes of interest defined in config.
* Provide a monocular distance estimate (placeholder).
* Annotate frames with bounding boxes, labels, and distance.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    BOX_THICKNESS,
    CLASS_COLORS,
    CLASSES_OF_INTEREST,
    CONFIDENCE_THRESHOLD,
    DEFAULT_COLOR,
    FOCAL_LENGTH_PX,
    FONT_SCALE,
    FONT_THICKNESS,
    IOU_THRESHOLD,
    KNOWN_OBJECT_WIDTHS_M,
    MODEL_NAME,
)


# ── Data class ────────────────────────────────────────────────────────────────

class Detection:
    """Immutable data object representing one detected object in a frame."""

    __slots__ = (
        "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "width", "height", "cx", "cy",
        "estimated_distance_m",
        "track_id",
    )

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        track_id: int | None = None,
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
        self.estimated_distance_m = _estimate_distance(class_name, self.width)

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


# ── Distance estimation helper ────────────────────────────────────────────────

def _estimate_distance(class_name: str, bbox_width_px: int) -> float | None:
    """
    Monocular distance estimate using the apparent-width formula.

        distance_m = (known_width_m × focal_length_px) / bbox_width_px

    ⚠️  PLACEHOLDER — this is a first-order approximation only.
        Accuracy depends on:
          • Correct focal length (calibrate with a checkerboard).
          • Objects being viewed roughly head-on (not at an angle).
        For production use, replace with stereo vision or a depth sensor.

    Returns None when the class width is unknown or the bbox is degenerate.
    """
    known_width = KNOWN_OBJECT_WIDTHS_M.get(class_name)
    if known_width is None or bbox_width_px <= 0:
        return None
    return round((known_width * FOCAL_LENGTH_PX) / bbox_width_px, 2)


# ── Detector class ────────────────────────────────────────────────────────────

class Detector:
    """
    YOLO-based object detector.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier (e.g. ``"yolo11n.pt"``).
        The weights are downloaded automatically on first use.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        print(f"[Detector] Loading model: {model_name} …")
        self._model      = YOLO(model_name)
        self._model_name = model_name
        print(
            f"[Detector] Ready.  Tracking classes: "
            f"{list(CLASSES_OF_INTEREST.values())}"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (as returned by ``cv2.VideoCapture.read``).

        Returns
        -------
        list[Detection]
            Detections filtered to ``CLASSES_OF_INTEREST``, sorted by
            descending confidence.
        """
        results = self._model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=list(CLASSES_OF_INTEREST.keys()),
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id     = int(box.cls[0].item())
                conf       = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                # track_id is populated when calling .track(); None for .predict()
                track_id: int | None = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                class_name = CLASSES_OF_INTEREST.get(cls_id, f"cls_{cls_id}")
                detections.append(
                    Detection(cls_id, class_name, conf, x1, y1, x2, y2, track_id)
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── Annotation ────────────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels directly onto *frame* (in-place).

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate.
        detections : list[Detection]
            Detections to render.

        Returns
        -------
        np.ndarray
            The annotated frame (same object as *frame*).
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

            # ── Bounding box ──────────────────────────────────────────────────
            cv2.rectangle(
                frame,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                BOX_THICKNESS,
            )

            # ── Label text ───────────────────────────────────────────────────
            parts = []
            if det.track_id is not None:
                parts.append(f"#{det.track_id}")
            parts.append(f"{det.class_name}  {det.confidence:.2f}")
            if det.estimated_distance_m is not None:
                parts.append(f"~{det.estimated_distance_m:.1f} m")
            label = "  ".join(parts)

            # ── Label background ─────────────────────────────────────────────
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, FONT_SCALE, FONT_THICKNESS
            )
            label_y = max(det.y1 - 6, text_h + 4)
            cv2.rectangle(
                frame,
                (det.x1, label_y - text_h - baseline),
                (det.x1 + text_w + 4, label_y + baseline),
                color,
                cv2.FILLED,
            )

            # ── Label text (black for contrast) ──────────────────────────────
            cv2.putText(
                frame,
                label,
                (det.x1 + 2, label_y - 2),
                font,
                FONT_SCALE,
                (0, 0, 0),
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame
