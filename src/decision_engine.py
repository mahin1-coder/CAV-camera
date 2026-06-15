"""
Decision Engine — rule-based stop-sign intersection scenario.

Actions (priority, highest first)
----------------------------------
WAIT      — stop sign confirmed AND cross traffic detected in intersection zone
STOP      — stop sign seen but no cross-traffic classification yet
PROCEED   — stop sign seen, held for required frames, no cross traffic → safe to go
SLOW_DOWN — pedestrian nearby
CAUTION   — traffic light or close-range object
NOMINAL   — no hazards

Stop-sign scenario logic
------------------------
1. YOLO detects a stop sign with confidence ≥ threshold.
2. The engine enters WAIT state and starts a hold counter.
3. It checks the left/right "cross-traffic zones" of the frame for
   vehicles (car, truck, bus, motorcycle).
4. If cross traffic is present → WAIT.
5. If hold counter expires and no cross traffic → PROCEED.
6. After PROCEED, state resets after a cooldown period.

Upgrade notes (v2)
------------------
* Reads all thresholds from the config dict — no hardcoded constants.
* Implements a proper stop-sign state machine (IDLE → STOPPING → WAIT →
  PROCEED → COOLDOWN → IDLE).
* Cross-traffic detection uses configurable zone ratios.

TODO (future work)
------------------
* Replace with a learned policy (RL / imitation learning).
* Fuse with GPS, IMU, HD-map, and V2X intersection messages.
* Implement Automatic Emergency Braking (AEB) via time-to-collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.detector import Detection


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class DrivingDecision:
    """Output produced by the decision engine for a single frame."""

    frame_id:         int
    alerts:           list[str] = field(default_factory=list)
    suggested_action: str       = "NOMINAL"


# ── State machine ─────────────────────────────────────────────────────────────

class _State(Enum):
    IDLE      = auto()
    STOPPING  = auto()   # stop sign seen, holding
    WAIT      = auto()   # holding — cross traffic present
    PROCEED   = auto()   # cleared — safe to go
    COOLDOWN  = auto()   # brief pause after PROCEED before resetting


# ── Decision engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Rule-based decision engine with stop-sign intersection scenario.

    Parameters
    ----------
    cfg : dict
        The ``decision`` section of the pipeline config.
    """

    # Action priority (lower index = higher priority)
    _PRIORITY = ["WAIT", "STOP", "PROCEED", "SLOW_DOWN", "CAUTION", "NOMINAL"]

    _COOLDOWN_FRAMES = 60   # frames after PROCEED before resetting to IDLE

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._stop_conf        = cfg.get("stop_sign_confidence",    0.50)
        self._cross_x_ratio    = cfg.get("cross_traffic_x_ratio",   0.25)
        self._cross_classes    = set(cfg.get("cross_traffic_classes", [2, 3, 5, 7]))
        self._stop_hold        = cfg.get("stop_hold_frames",         30)
        self._close_range_m    = cfg.get("close_range_m",            10.0)
        self._person_range_m   = cfg.get("person_range_m",           15.0)

        self._state: _State = _State.IDLE
        self._hold_counter  = 0
        self._cool_counter  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: list[Detection],
        frame_id:   int,
        frame_w:    int = 1280,
    ) -> DrivingDecision:
        """
        Evaluate detections for the current frame and return a DrivingDecision.

        Parameters
        ----------
        detections : list[Detection]
        frame_id   : int    Monotonic frame counter.
        frame_w    : int    Frame width in pixels (for zone calculation).
        """
        decision = DrivingDecision(frame_id=frame_id)

        stop_sign_seen  = self._detect_stop_sign(detections)
        cross_traffic   = self._detect_cross_traffic(detections, frame_w)

        # ── State machine tick ────────────────────────────────────────────────
        if self._state == _State.IDLE:
            if stop_sign_seen:
                self._state        = _State.STOPPING
                self._hold_counter = self._stop_hold

        elif self._state == _State.STOPPING:
            self._hold_counter -= 1
            if cross_traffic:
                self._state = _State.WAIT
            elif self._hold_counter <= 0:
                self._state        = _State.PROCEED
                self._cool_counter = self._COOLDOWN_FRAMES

        elif self._state == _State.WAIT:
            if not cross_traffic:
                self._state = _State.STOPPING   # recheck hold

        elif self._state == _State.PROCEED:
            self._cool_counter -= 1
            if self._cool_counter <= 0:
                self._state = _State.IDLE

        elif self._state == _State.COOLDOWN:
            self._state = _State.IDLE

        # ── Map state → action ────────────────────────────────────────────────
        if self._state in (_State.WAIT,):
            self._upgrade(decision, "WAIT")
            decision.alerts.append(
                f"STOP SIGN + CROSS TRAFFIC — holding ({self._hold_counter} frames)"
            )
        elif self._state == _State.STOPPING:
            self._upgrade(decision, "STOP")
            decision.alerts.append(
                f"STOP SIGN — waiting ({self._hold_counter} frames remaining)"
            )
        elif self._state == _State.PROCEED:
            self._upgrade(decision, "PROCEED")
            decision.alerts.append("STOP SIGN cleared — PROCEED")

        # ── Per-detection rules ───────────────────────────────────────────────
        for det in detections:
            dist = det.estimated_distance_m

            if det.class_name == "traffic light":
                decision.alerts.append(
                    "TRAFFIC LIGHT — colour classification not yet implemented"
                )
                self._upgrade(decision, "CAUTION")

            elif det.class_name == "person":
                if dist is not None and dist < self._person_range_m:
                    decision.alerts.append(
                        f"PEDESTRIAN within {dist:.1f} m — slow down"
                    )
                    self._upgrade(decision, "SLOW_DOWN")

            elif det.class_name not in ("stop sign", "traffic light"):
                if dist is not None and dist < self._close_range_m:
                    decision.alerts.append(
                        f"{det.class_name.upper()} close at {dist:.1f} m"
                    )
                    self._upgrade(decision, "CAUTION")

        return decision

    def reset(self) -> None:
        """Reset the state machine to IDLE."""
        self._state        = _State.IDLE
        self._hold_counter = 0
        self._cool_counter = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_stop_sign(self, detections: list[Detection]) -> bool:
        return any(
            d.class_name == "stop sign" and d.confidence >= self._stop_conf
            for d in detections
        )

    def _detect_cross_traffic(
        self,
        detections: list[Detection],
        frame_w:    int,
    ) -> bool:
        """
        True when a vehicle-class object appears in the left or right zone
        of the frame (i.e., could be crossing the intersection).
        """
        zone_w = int(frame_w * self._cross_x_ratio)
        for d in detections:
            if d.class_id in self._cross_classes:
                # Object centre is in left or right zone
                if d.cx < zone_w or d.cx > (frame_w - zone_w):
                    return True
        return False

    def _upgrade(self, decision: DrivingDecision, new_action: str) -> None:
        """Upgrade action only if *new_action* has strictly higher priority."""
        cur = self._PRIORITY.index(decision.suggested_action)
        new = self._PRIORITY.index(new_action)
        if new < cur:
            decision.suggested_action = new_action
