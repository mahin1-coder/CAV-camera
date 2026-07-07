"""
Object tracker module — wraps Ultralytics built-in ByteTrack / BoT-SORT.

Ultralytics YOLO ships ByteTrack and BoT-SORT out of the box.
Calling ``model.track(frame, tracker="bytetrack.yaml")`` returns results
that include persistent ``box.id`` values — no extra package needed.

This module provides a thin ``Tracker`` class that:
  * Holds the tracker configuration name.
  * Calls ``model.track()`` and returns the raw Ultralytics results.
  * Exposes a ``reset()`` method to clear tracker state between runs.

Usage
-----
The ``Detector`` class in ``detector.py`` owns one ``Tracker`` instance
and delegates inference to it.  You do not call ``Tracker`` directly from
``main.py``.

TODO (future work)
------------------
* Custom re-identification model for BoT-SORT.
* Per-class track confidence thresholds.
* Track-history buffer for trajectory visualisation.
"""

from __future__ import annotations

from typing import Any

from ultralytics import YOLO


class Tracker:
    """
    Thin wrapper around Ultralytics' built-in multi-object tracker.

    Parameters
    ----------
    model : YOLO
        The loaded Ultralytics YOLO model instance.
    tracker_cfg : str | None
        Tracker configuration name: ``"bytetrack.yaml"`` (default) or
        ``"botsort.yaml"``.  Pass ``None`` to disable tracking and fall
        back to plain ``model.predict()``.
    """

    def __init__(
        self,
        model:       YOLO,
        tracker_cfg: str | None = "bytetrack.yaml",
    ) -> None:
        self._model       = model
        self._tracker_cfg = tracker_cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        frame:       Any,          # np.ndarray  BGR image
        conf:        float = 0.35,
        iou:         float = 0.45,
        classes:     list[int] | None = None,
        device:      str = "",
        imgsz:       int | tuple[int, int] | None = None,
        max_det:     int = 100,
        agnostic_nms: bool = False,
    ) -> list:
        """
        Run inference (with tracking when enabled) on a single BGR frame.

        Returns the raw Ultralytics ``Results`` list.
        """
        kwargs: dict[str, Any] = dict(
            source=frame,
            conf=conf,
            iou=iou,
            classes=classes,
            device=device or None,
            imgsz=imgsz,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )

        if self._tracker_cfg:
            return self._model.track(
                **kwargs,
                tracker=self._tracker_cfg,
                persist=True,   # keep track IDs across consecutive frames
            )
        else:
            return self._model.predict(**kwargs)

    def reset(self) -> None:
        """
        Reset the tracker state (clears all track IDs).

        Call this when restarting the pipeline or switching cameras.
        """
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            try:
                self._model.predictor.trackers = None
            except AttributeError:
                pass
