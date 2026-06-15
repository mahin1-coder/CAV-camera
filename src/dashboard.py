"""
Multi-panel CAV perception dashboard renderer.

Layout  (1280 × 720 canvas)
─────────────────────────────────────────────────────────────
  ┌──────────────┬──────────────────────────┬──────────────┐
  │  RAW CAMERA  │                          │   OBJECT     │
  │   + FPS      │    SEMANTIC WORLD MAP    │  DETECTION   │
  │  (320 × 354) │       (640 × 720)        │  (320 × 354) │
  ├──────────────┤                          ├──────────────┤
  │ SLAM/FEATURES│                          │   TRACKED    │
  │  (or DEPTH)  │                          │  PREDICTION  │
  │  (320 × 354) │                          │  (320 × 354) │
  └──────────────┴──────────────────────────┴──────────────┘

All panels are composited on a dark (#0c0c0c) background.
Coloured borders, white panel labels, and per-panel overlays
are drawn last so they always appear on top.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detector import Detection
from src.decision_engine import DrivingDecision

# ── Layout constants ──────────────────────────────────────────────────────────

CANVAS_W  = 1280
CANVAS_H  = 720
LEFT_W    = 320
RIGHT_W   = 320
CENTER_W  = CANVAS_W - LEFT_W - RIGHT_W   # 640
PANEL_H   = (CANVAS_H - 12) // 2          # 354  (12 px gap)
GAP       = CANVAS_H - 2 * PANEL_H        # 12

LEFT_X    = 0
CENTER_X  = LEFT_W
RIGHT_X   = LEFT_W + CENTER_W

TOP_Y     = 0
BOT_Y     = PANEL_H + GAP

# ── Colours ───────────────────────────────────────────────────────────────────

BG_COLOR      = (12,  12,  12)
BORDER_COLOR  = (45,  45,  45)
LABEL_COLOR   = (210, 210, 210)
FPS_COLOR     = (0,   220,   0)
DIM_COLOR     = (80,   80,  80)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":        (0,   255,   0),
    "bicycle":       (0,   165, 255),
    "car":           (0,     0, 255),
    "motorcycle":    (255,   0, 255),
    "bus":           (255, 165,   0),
    "truck":         (0,   255, 255),
    "traffic light": (255, 255,   0),
    "stop sign":     (0,     0, 200),
}
_DEFAULT_COLOR: tuple[int, int, int] = (160, 160, 160)

ACTION_COLORS: dict[str, tuple[int, int, int]] = {
    "NOMINAL":   (0,   200,  60),
    "CAUTION":   (0,   165, 255),
    "SLOW_DOWN": (0,   165, 255),
    "STOP":      (30,   30, 255),
    "WAIT":      (30,   30, 255),
    "PROCEED":   (0,   240,  80),
}

_TRAIL_MAXLEN = 25


class Dashboard:
    """
    Assembles a multi-panel research-demo dashboard from per-frame data.

    The Dashboard is stateful: it maintains per-track trail histories for
    the Tracked-Prediction panel.

    Parameters
    ----------
    depth_available : bool
        When True the bottom-left panel shows a depth colourmap;
        otherwise it shows the SLAM/features view.
    """

    def __init__(self, depth_available: bool = False) -> None:
        self._depth_available = depth_available
        # Trail history in *frame* pixel space (for the tracking panel)
        self._trail_history: dict[int, collections.deque[tuple[int, int]]] = {}

    # ── Panel constructors ────────────────────────────────────────────────────

    def _make_slam_panel(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a SLAM-style feature-point / edge view.

        Uses ORB keypoints drawn as bright dots on a Canny edge background.
        Both are rendered in green to give the classic "SLAM terminal" look.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)

        out = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
        out[:, :, 1] = edges   # green channel only

        try:
            orb  = cv2.ORB_create(nfeatures=200, fastThreshold=12)
            kps  = orb.detect(gray)
            for kp in kps:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(out, (x, y), 2, (0, 255, 80), -1, cv2.LINE_AA)
        except Exception:  # noqa: BLE001
            pass

        return out

    def _make_depth_panel(
        self,
        frame: np.ndarray,
        depth_u8: np.ndarray | None,
    ) -> np.ndarray:
        """Return an INFERNO depth colourmap, or fall back to SLAM panel."""
        if depth_u8 is None:
            return self._make_slam_panel(frame)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

    def _make_detection_panel(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Camera frame annotated with YOLO bounding boxes and class labels."""
        out = frame.copy()
        for det in detections:
            color = CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            parts: list[str] = []
            if det.track_id is not None:
                parts.append(f"#{det.track_id}")
            parts.append(f"{det.class_name} {det.confidence:.2f}")
            if det.estimated_distance_m is not None:
                parts.append(f"~{det.estimated_distance_m:.1f}m")
            label = "  ".join(parts)

            ty = max(det.y1 - 5, 14)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(out, (det.x1, ty - th - 2), (det.x1 + tw + 2, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(out, label, (det.x1 + 1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return out

    def _make_tracking_panel(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Darkened camera frame with:
          - Coloured bounding boxes per track
          - Fading colour trail (last N positions)
          - Dashed linear-extrapolation prediction
        """
        out = (frame * 0.30).astype(np.uint8)

        for det in detections:
            color = CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 1)

            if det.track_id is not None:
                trail = self._trail_history.setdefault(
                    det.track_id,
                    collections.deque(maxlen=_TRAIL_MAXLEN),
                )
                trail.append((det.cx, det.cy))
                pts = list(trail)
                n   = len(pts)

                # Fading trail
                for i in range(1, n):
                    alpha = i / n
                    tc    = tuple(int(c * alpha) for c in color)
                    cv2.line(out, pts[i - 1], pts[i], tc, 1, cv2.LINE_AA)

                # Prediction dots
                if n >= 2:
                    dx = pts[-1][0] - pts[-2][0]
                    dy = pts[-1][1] - pts[-2][1]
                    for step in range(1, 9):
                        px = pts[-1][0] + dx * step * 2
                        py = pts[-1][1] + dy * step * 2
                        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                            r = max(1, 3 - step // 3)
                            cv2.circle(out, (px, py), r, color, -1, cv2.LINE_AA)

                # Track ID label
                cv2.putText(
                    out, f"T{det.track_id}",
                    (det.cx - 8, det.cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
                )

        # Prune stale trails
        active = {d.track_id for d in detections if d.track_id is not None}
        for tid in [t for t in self._trail_history if t not in active]:
            del self._trail_history[tid]

        return out

    # ── Main render ───────────────────────────────────────────────────────────

    def render(
        self,
        raw_frame:    np.ndarray,
        bev_frame:    np.ndarray,
        detections:   list[Detection],
        decision:     DrivingDecision | None,
        fps:          float,
        frame_id:     int,
        depth_u8:     np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Assemble and return the full 1280 × 720 dashboard image.

        Parameters
        ----------
        raw_frame   : original BGR camera frame
        bev_frame   : 640 × 720 BEV world-map image from BEVMapper
        detections  : current-frame Detection list
        decision    : DrivingDecision from DecisionEngine (or None)
        fps         : rolling-average frame rate
        frame_id    : monotonic frame counter
        depth_u8    : optional 1-channel depth map from DepthEstimator
        """
        # ── Build side panels from full-resolution source frame ───────────
        bot_left = (
            self._make_depth_panel(raw_frame, depth_u8)
            if self._depth_available
            else self._make_slam_panel(raw_frame)
        )
        det_frame = self._make_detection_panel(raw_frame, detections)
        trk_frame = self._make_tracking_panel(raw_frame, detections)

        # ── Resize panels to their dashboard slots ────────────────────────
        def rsz(src: np.ndarray, w: int, h: int) -> np.ndarray:
            return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

        p_raw  = rsz(raw_frame,  LEFT_W,   PANEL_H)
        p_bot  = rsz(bot_left,   LEFT_W,   PANEL_H)
        p_det  = rsz(det_frame,  RIGHT_W,  PANEL_H)
        p_trk  = rsz(trk_frame,  RIGHT_W,  PANEL_H)
        p_bev  = rsz(bev_frame,  CENTER_W, CANVAS_H)

        # ── Assemble canvas ───────────────────────────────────────────────
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        canvas[TOP_Y:TOP_Y + PANEL_H, LEFT_X:LEFT_X + LEFT_W]       = p_raw
        canvas[BOT_Y:BOT_Y + PANEL_H, LEFT_X:LEFT_X + LEFT_W]       = p_bot
        canvas[0:CANVAS_H,            CENTER_X:CENTER_X + CENTER_W]  = p_bev
        canvas[TOP_Y:TOP_Y + PANEL_H, RIGHT_X:RIGHT_X + RIGHT_W]    = p_det
        canvas[BOT_Y:BOT_Y + PANEL_H, RIGHT_X:RIGHT_X + RIGHT_W]    = p_trk

        # ── Panel borders ─────────────────────────────────────────────────
        cv2.line(canvas, (CENTER_X, 0),   (CENTER_X, CANVAS_H),   BORDER_COLOR, 2)
        cv2.line(canvas, (RIGHT_X,  0),   (RIGHT_X,  CANVAS_H),   BORDER_COLOR, 2)
        cv2.line(canvas, (CANVAS_W - 1, 0), (CANVAS_W - 1, CANVAS_H), BORDER_COLOR, 2)
        cv2.line(canvas, (0, BOT_Y - 1), (LEFT_W, BOT_Y - 1),    BORDER_COLOR, 2)
        cv2.line(canvas, (RIGHT_X, BOT_Y - 1), (CANVAS_W, BOT_Y - 1), BORDER_COLOR, 2)

        # ── Panel labels ──────────────────────────────────────────────────
        bot_label = "DEPTH (MiDaS)" if self._depth_available else "SLAM / FEATURES"
        self._put_panel_label(canvas, "RAW CAMERA",      LEFT_X + 6,  TOP_Y + 15)
        self._put_panel_label(canvas, bot_label,         LEFT_X + 6,  BOT_Y + 15)
        self._put_panel_label(canvas, "OBJECT DETECTION", RIGHT_X + 6, TOP_Y + 15)
        self._put_panel_label(canvas, "TRACKED PREDICTION", RIGHT_X + 6, BOT_Y + 15)

        # ── FPS overlay on raw-camera panel ───────────────────────────────
        cv2.putText(
            canvas, f"FPS: {fps:5.1f}",
            (LEFT_X + 6, TOP_Y + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            FPS_COLOR, 1, cv2.LINE_AA,
        )

        # ── Frame counter (bottom-right corner) ───────────────────────────
        cv2.putText(
            canvas, f"#{frame_id:07d}",
            (CANVAS_W - 90, CANVAS_H - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36,
            DIM_COLOR, 1,
        )

        # ── Decision / alert on detection panel ───────────────────────────
        action = decision.suggested_action if decision else "NOMINAL"
        acolor = ACTION_COLORS.get(action, _DEFAULT_COLOR)
        cv2.putText(
            canvas, f"Action: {action}",
            (RIGHT_X + 4, TOP_Y + PANEL_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            acolor, 2, cv2.LINE_AA,
        )

        if decision and decision.alerts:
            for i, alert in enumerate(decision.alerts[:4]):
                cv2.putText(
                    canvas, alert,
                    (RIGHT_X + 4, BOT_Y + 30 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (0, 165, 255), 1, cv2.LINE_AA,
                )

        # ── Object count on tracking panel ────────────────────────────────
        cv2.putText(
            canvas,
            f"Tracks: {len([d for d in detections if d.track_id is not None])}",
            (RIGHT_X + 4, BOT_Y + PANEL_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            (150, 220, 150), 1, cv2.LINE_AA,
        )

        return canvas

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _put_panel_label(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
    ) -> None:
        """Draw a small white panel title with a semi-transparent background."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(canvas, (x - 2, y - th - 2), (x + tw + 2, y + 2),
                      (0, 0, 0), -1)
        cv2.putText(
            canvas, text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            LABEL_COLOR, 1, cv2.LINE_AA,
        )


def save_dashboard(frame: np.ndarray, output_dir: str = "outputs") -> Path:
    """Save the dashboard image as outputs/cav_<timestamp>.jpg."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts   = int(time.time() * 1000)
    path = out / f"cav_{ts}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path
