"""
V2X Simulator — placeholder for Vehicle-to-Everything messaging.

What is V2X?
------------
V2X (Vehicle-to-Everything) is the umbrella term for wireless communication
between vehicles and other road entities:
  • V2V  — Vehicle-to-Vehicle
  • V2I  — Vehicle-to-Infrastructure (traffic lights, road-side units)
  • V2P  — Vehicle-to-Pedestrian
  • V2N  — Vehicle-to-Network (cloud back-end)

Standards used in practice: DSRC (IEEE 802.11p / WAVE) and C-V2X (3GPP).

Current implementation
----------------------
Runs a background daemon thread that periodically prints JSON-serialised
simulated messages to stdout.  Two message types are generated:

  BSM   (Basic Safety Message, SAE J2735)
        Contains the ego vehicle ID and a snapshot of perceived nearby objects
        derived from the latest YOLO detections.

  SPaT  (Signal Phase and Timing, SAE J2735)
        Stub for receiving traffic-light phase information from a Road-Side
        Unit (RSU).  The signal_state field is always "UNKNOWN" until a real
        RSU interface is connected.

TODO (future work)
------------------
* Broadcast BSMs over UDP multicast (e.g. 239.255.0.1:49152) for local
  multi-vehicle simulation.
* Parse incoming BSMs from other simulated vehicles on the same LAN.
* Integrate with SUMO or CARLA for a full traffic simulation environment.
* Connect to a real DSRC / C-V2X radio SDK (e.g. Cohda Wireless MK5).
* Fuse received V2X data with local YOLO detections in the decision engine.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.config import (
    V2X_BROADCAST_INTERVAL_S,
    V2X_EGO_VEHICLE_ID,
    V2X_ENABLED,
)
from src.detector import Detection


class V2XSimulator:
    """
    Simulated V2X broadcast daemon.

    Usage
    -----
    ::

        sim = V2XSimulator()
        sim.start()
        # each frame:
        sim.update_detections(detections)
        # on shutdown:
        sim.stop()
    """

    def __init__(self) -> None:
        self._enabled    = V2X_ENABLED
        self._interval   = V2X_BROADCAST_INTERVAL_S
        self._vehicle_id = V2X_EGO_VEHICLE_ID

        self._latest_detections: list[Detection] = []
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background broadcast thread."""
        if not self._enabled:
            print("[V2X] Simulator is disabled (set V2X_ENABLED=True in config).")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._broadcast_loop,
            daemon=True,
            name="v2x-simulator",
        )
        self._thread.start()
        print(
            f"[V2X] Simulator started | vehicle_id={self._vehicle_id} | "
            f"interval={self._interval:.1f}s"
        )

    def stop(self) -> None:
        """Signal the broadcast thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        print("[V2X] Simulator stopped.")

    def update_detections(self, detections: list[Detection]) -> None:
        """
        Provide the latest detections so they appear in the next BSM.

        Call this once per frame from the main pipeline loop.
        """
        with self._lock:
            self._latest_detections = list(detections)

    # ── Background loop ───────────────────────────────────────────────────────

    def _broadcast_loop(self) -> None:
        while not self._stop_event.is_set():
            self._emit_bsm()
            self._emit_spat_stub()
            self._stop_event.wait(self._interval)

    # ── Message builders ──────────────────────────────────────────────────────

    def _emit_bsm(self) -> None:
        """Build and print a simulated BSM (SAE J2735-inspired)."""
        with self._lock:
            perceived: list[dict[str, Any]] = [
                {
                    "class_name":            det.class_name,
                    "confidence":            round(det.confidence, 3),
                    "estimated_distance_m":  det.estimated_distance_m,
                    "bbox_center_px":        [det.cx, det.cy],
                    "track_id":              det.track_id,
                }
                for det in self._latest_detections
            ]

        bsm: dict[str, Any] = {
            "msg_type":          "BSM",
            "vehicle_id":        self._vehicle_id,
            "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
            "perceived_objects": perceived,
            "object_count":      len(perceived),
            "note":              "SIMULATED — not a real DSRC/C-V2X transmission",
        }
        print(f"[V2X][BSM]  {json.dumps(bsm, separators=(',', ':'))}")

    def _emit_spat_stub(self) -> None:
        """Build and print a stub SPaT message (SAE J2735-inspired)."""
        spat: dict[str, Any] = {
            "msg_type":       "SPaT",
            "source_rsu":     "SIM_RSU_001",
            "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
            "intersections": [
                {
                    "intersection_id": "INT_042",
                    "signal_state":    "UNKNOWN",
                    "note":            "Real SPaT parsing not yet implemented",
                }
            ],
        }
        print(f"[V2X][SPaT] {json.dumps(spat, separators=(',', ':'))}")
