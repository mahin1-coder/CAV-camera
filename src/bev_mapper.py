"""
Bird's-Eye-View (BEV) world-map renderer.

Produces a pseudo-3D top-down representation of the scene using:
  - Estimated object distances (from Detection.estimated_distance_m)
  - Lateral position derived from the object's horizontal bbox centre
  - Optional MiDaS depth map (when DepthEstimator is active)

This is a *simulated* BEV — not real SLAM.  It projects detections
onto a top-down canvas using a simple perspective model; no point-cloud
or LIDAR data is used.

Visual conventions
------------------
* Ego vehicle: cyan triangle at the bottom centre of the canvas.
* Detected objects: filled coloured rectangles scaled by distance
  (further = smaller, to reinforce depth cue).
* Tracks: fading colour trail + dashed prediction line.
* Distance grid: horizontal lines at 5 m, 10 m, 15 m, 20 m.
* FOV cone: two lines radiating from the ego to represent the camera cone.
"""

from __future__ import annotations

import collections
from typing import Any

import cv2
import numpy as np

from src.detector import Detection

# ── Constants ─────────────────────────────────────────────────────────────────

CANVAS_W = 640
CANVAS_H = 720

_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":        (0,   255,   0),
    "bicycle":       (0,   165, 255),
    "car":           (0,     0, 255),
    "motorcycle":    (255,   0, 255),
    "bus":           (255, 165,   0),
    "truck":         (0,   255, 255),
    "traffic light": (255, 255,   0),
    "stop sign":     (0,     0, 200),
}

_ACTION_COLORS: dict[str, tuple[int, int, int]] = {
    "NOMINAL":   (0, 200,  60),
    "CAUTION":   (0, 165, 255),
    "SLOW_DOWN": (0, 165, 255),
    "STOP":      (30, 30, 255),
    "WAIT":      (30, 30, 255),
    "PROCEED":   (0, 240,  80),
}

_DEFAULT_COLOR: tuple[int, int, int] = (160, 160, 160)
_EGO_Y_OFFSET = 55          # pixels from canvas bottom to ego vehicle centre
_TRAIL_MAXLEN  = 25


class BEVMapper:
    """
    Renders a pseudo-3D / BEV world map from a list of Detection objects.

    Parameters
    ----------
    cfg : dict
        The ``bev`` section of the pipeline config.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._max_range_m:  float = cfg.get("max_range_m",  20.0)
        self._half_fov_m:   float = cfg.get("half_fov_m",   10.0)
        # Persistent per-track trail in BEV canvas coordinates
        self._trails: dict[int, collections.deque[tuple[int, int]]] = {}
        # Pre-render static background (grid + FOV lines)
        self._bg = self._build_background()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        action: str = "NOMINAL",
    ) -> np.ndarray:
        """
        Build and return a BGR BEV canvas for this frame.

        Parameters
        ----------
        detections : list[Detection]
            Current-frame detections (may be empty).
        frame_w, frame_h : int
            Original camera frame dimensions (used for lateral mapping).
        action : str
            Current driving decision action label.
        """
        canvas = self._bg.copy()

        ego_x = CANVAS_W // 2
        ego_y = CANVAS_H - _EGO_Y_OFFSET

        for det in detections:
            cx, cy = self._project(det, frame_w, ego_x, ego_y)
            if not (5 < cx < CANVAS_W - 5 and 5 < cy < CANVAS_H - 5):
                continue

            color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)

            # ── Trail ────────────────────────────────────────────────────────
            if det.track_id is not None:
                trail = self._trails.setdefault(
                    det.track_id,
                    collections.deque(maxlen=_TRAIL_MAXLEN),
                )
                trail.append((cx, cy))
                pts = list(trail)
                n   = len(pts)
                for i in range(1, n):
                    alpha = i / n
                    tc    = tuple(int(c * alpha) for c in color)
                    cv2.line(canvas, pts[i - 1], pts[i], tc, 1, cv2.LINE_AA)

                # ── Prediction (linear extrapolation) ─────────────────────
                if n >= 2:
                    dx = pts[-1][0] - pts[-2][0]
                    dy = pts[-1][1] - pts[-2][1]
                    for step in range(1, 7):
                        px = pts[-1][0] + dx * step * 2
                        py = pts[-1][1] + dy * step * 2
                        if 0 <= px < CANVAS_W and 0 <= py < CANVAS_H:
                            r = max(1, 3 - step // 2)
                            cv2.circle(canvas, (px, py), r, color, -1, cv2.LINE_AA)

            # ── Object box (scaled by distance) ───────────────────────────
            dist = det.estimated_distance_m or 5.0
            half = max(4, int(18 * (1.0 - dist / self._max_range_m) + 4))
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # Short class label
            lbl = det.class_name[:3].upper()
            if det.track_id is not None:
                lbl += f"#{det.track_id}"
            cv2.putText(
                canvas, lbl,
                (x2 + 3, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                color, 1, cv2.LINE_AA,
            )
            if det.estimated_distance_m is not None:
                cv2.putText(
                    canvas, f"{det.estimated_distance_m:.1f}m",
                    (cx - 10, y2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                    (180, 180, 180), 1, cv2.LINE_AA,
                )

        # ── Ego vehicle (cyan triangle) ───────────────────────────────────
        pts_ego = np.array(
            [[ego_x, ego_y - 18], [ego_x - 10, ego_y + 10], [ego_x + 10, ego_y + 10]],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts_ego], (0, 220, 255))
        cv2.polylines(canvas, [pts_ego], True, (255, 255, 255), 1)
        cv2.putText(
            canvas, "EGO",
            (ego_x - 11, ego_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
            (0, 220, 255), 1, cv2.LINE_AA,
        )

        # ── Action overlay ────────────────────────────────────────────────
        acolor = _ACTION_COLORS.get(action, _DEFAULT_COLOR)
        cv2.putText(
            canvas, f"Action: {action}",
            (CANVAS_W // 2 - 65, CANVAS_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            acolor, 2, cv2.LINE_AA,
        )

        return canvas

    def reset_stale_trails(self, active_ids: set[int]) -> None:
        """Remove trails for tracks that have disappeared."""
        stale = [tid for tid in self._trails if tid not in active_ids]
        for tid in stale:
            del self._trails[tid]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _project(
        self,
        det: Detection,
        frame_w: int,
        ego_x: int,
        ego_y: int,
    ) -> tuple[int, int]:
        """Map a Detection to BEV canvas (cx, cy) coordinates."""
        dist = det.estimated_distance_m if det.estimated_distance_m else 5.0
        dist = max(0.5, min(dist, self._max_range_m))

        # Lateral: normalise bbox centre to [-1, 1] from frame centre
        lat_norm = (det.cx / frame_w - 0.5) * 2.0
        # Lateral in metres, widens slightly with distance
        lat_m = lat_norm * self._half_fov_m * (0.4 + 0.6 * dist / self._max_range_m)

        usable_h = CANVAS_H - _EGO_Y_OFFSET - 15
        px_per_m_lat  = (CANVAS_W / 2) / self._half_fov_m
        px_per_m_long = usable_h / self._max_range_m

        canvas_x = int(ego_x + lat_m * px_per_m_lat)
        canvas_y = int(ego_y - dist * px_per_m_long)
        return canvas_x, canvas_y

    def _build_background(self) -> np.ndarray:
        """Pre-render the static BEV background (grid, FOV cone, labels)."""
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (8, 14, 8)   # very dark green tint

        ego_x = CANVAS_W // 2
        ego_y = CANVAS_H - _EGO_Y_OFFSET
        usable_h = CANVAS_H - _EGO_Y_OFFSET - 15

        # Distance grid lines
        grid_color  = (28, 48, 28)
        label_color = (55, 100, 55)
        for d in [5, 10, 15, 20]:
            gy = int(ego_y - d / self._max_range_m * usable_h)
            if 0 <= gy < CANVAS_H:
                cv2.line(canvas, (0, gy), (CANVAS_W, gy), grid_color, 1)
                cv2.putText(
                    canvas, f"{d} m",
                    (4, gy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    label_color, 1,
                )

        # Vertical centre line
        cv2.line(canvas, (ego_x, 0), (ego_x, CANVAS_H), grid_color, 1)

        # FOV cone
        cone_color = (18, 42, 18)
        cv2.line(canvas, (ego_x, ego_y), (20, 10),            cone_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (ego_x, ego_y), (CANVAS_W - 20, 10), cone_color, 1, cv2.LINE_AA)

        # Panel title
        cv2.putText(
            canvas, "SEMANTIC WORLD MAP",
            (CANVAS_W // 2 - 88, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44,
            (70, 140, 70), 1, cv2.LINE_AA,
        )

        return canvas
