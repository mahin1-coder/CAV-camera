"""
V2X (Vehicle-to-Everything) simulation module.

Simulates three SAE J2735-inspired message types printed as JSON to stdout.

Message types
-------------
BSM   Basic Safety Message
      Ego vehicle status + list of perceived nearby objects.

DOM   Detected-Object Message  (custom extension)
      Per-detection payload with class, confidence, distance, track ID.
      Suitable for sharing perception results with other vehicles or an RSU.

ISM   Intersection State Message  (SPaT stub)
      Simulated signal-phase-and-timing data from a Road-Side Unit (RSU).

Architecture
------------
V2XSimulator runs a background daemon thread that broadcasts at the
configured interval.  The main loop calls update() once per frame to
provide the latest detections.

TODO (future work)
------------------
* UDP multicast broadcast (e.g. 239.255.0.1:49152) for multi-vehicle LAN sim.
* Parse incoming BSMs from other vehicles on the same LAN.
* Connect to a real DSRC / C-V2X radio SDK (e.g. Cohda Wireless MK5).
* Fuse received V2X data with local YOLO detections in the decision engine.
* Implement ETSI ITS-G5 / USDOT DSRC message encoding.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.detector import Detection


class V2XSimulator:
    """
    Simulated V2X broadcast daemon.

    Parameters
    ----------
    cfg : dict
        The ``v2x`` section of the pipeline config.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._enabled    = cfg.get("enabled",              True)
        self._interval   = cfg.get("broadcast_interval_s", 1.0)
        self._vehicle_id = cfg.get("vehicle_id",           "EGO_001")

        self._latest_detections: list[Detection] = []
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background broadcast thread."""
        if not self._enabled:
            print("[V2X] Simulator disabled (v2x.enabled=false in config).")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="v2x-sim"
        )
        self._thread.start()
        print(
            f"[V2X] Simulator started | id={self._vehicle_id} | "
            f"interval={self._interval:.1f}s"
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        print("[V2X] Simulator stopped.")

    def update(self, detections: list[Detection]) -> None:
        """
        Provide the latest detections for inclusion in the next broadcast.
        Call this once per frame from the main pipeline loop.
        """
        with self._lock:
            self._latest_detections = list(detections)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._emit_bsm()
            self._emit_dom()
            self._emit_ism()
            self._stop_event.wait(self._interval)

    # ── Message builders ──────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _emit_bsm(self) -> None:
        """Basic Safety Message — ego vehicle status + perceived objects."""
        with self._lock:
            objects = [
                {
                    "class":      d.class_name,
                    "confidence": round(d.confidence, 3),
                    "track_id":   d.track_id,
                    "distance_m": d.estimated_distance_m,
                    "bbox_cx":    d.cx,
                    "bbox_cy":    d.cy,
                }
                for d in self._latest_detections
            ]
        bsm: dict[str, Any] = {
            "msg":           "BSM",
            "vehicle_id":    self._vehicle_id,
            "timestamp_utc": self._now(),
            "speed_mps":     None,    # placeholder — requires GPS/IMU
            "heading_deg":   None,    # placeholder — requires GPS/IMU
            "perceived":     objects,
            "object_count":  len(objects),
            "note":          "SIMULATED",
        }
        print(f"[V2X][BSM] {json.dumps(bsm, separators=(',', ':'))}")

    def _emit_dom(self) -> None:
        """Detected-Object Message — one message per detected object."""
        with self._lock:
            dets = list(self._latest_detections)
        for d in dets:
            dom: dict[str, Any] = {
                "msg":           "DOM",
                "vehicle_id":    self._vehicle_id,
                "timestamp_utc": self._now(),
                "track_id":      d.track_id,
                "class_id":      d.class_id,
                "class_name":    d.class_name,
                "confidence":    round(d.confidence, 3),
                "bbox":          [d.x1, d.y1, d.x2, d.y2],
                "distance_m":    d.estimated_distance_m,
                "note":          "SIMULATED",
            }
            print(f"[V2X][DOM] {json.dumps(dom, separators=(',', ':'))}")

    def _emit_ism(self) -> None:
        """Intersection State Message — SPaT stub from simulated RSU."""
        ism: dict[str, Any] = {
            "msg":            "ISM",
            "source_rsu":     "SIM_RSU_001",
            "timestamp_utc":  self._now(),
            "intersections": [
                {
                    "id":           "INT_042",
                    "signal_state": "UNKNOWN",
                    "note":         "Real SPaT not yet implemented",
                }
            ],
        }
        print(f"[V2X][ISM] {json.dumps(ism, separators=(',', ':'))}")
