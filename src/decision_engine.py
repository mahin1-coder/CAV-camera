"""
Decision Engine — rule-based driving decision placeholder.

Current behaviour
-----------------
Applies a small set of hard-coded safety rules to the detected objects and
returns a DrivingDecision that the main loop can display on screen.

Suggested action values
-----------------------
NOMINAL    — no hazards detected
CAUTION    — a potentially relevant object is present
SLOW_DOWN  — a pedestrian or close object detected within threshold distance
STOP       — a stop sign or very close object detected

TODO (future work)
------------------
* Replace rule table with a learned policy (RL, imitation learning).
* Fuse detections with GPS, IMU, and HD-map data.
* Generate low-level control signals (steer angle, throttle, brake).
* Implement Automatic Emergency Braking (AEB) using time-to-collision.
* Integrate with a path planner (e.g. ROS 2 Nav2, CARLA agent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.detector import Detection


@dataclass
class DrivingDecision:
    """Output produced by the decision engine for a single frame."""

    frame_id:         int
    alerts:           list[str] = field(default_factory=list)
    suggested_action: str       = "NOMINAL"


class DecisionEngine:
    """
    Lightweight rule-based decision engine.

    The engine inspects every Detection and updates a DrivingDecision
    according to a simple priority table:
        STOP  >  SLOW_DOWN  >  CAUTION  >  NOMINAL

    Parameters
    ----------
    close_range_m  : float  Any detected object closer than this (m) triggers
                            at least CAUTION.
    person_range_m : float  A pedestrian closer than this (m) triggers
                            SLOW_DOWN.
    """

    # Action priority (lower index = higher priority)
    _PRIORITY = ["STOP", "SLOW_DOWN", "CAUTION", "NOMINAL"]

    def __init__(
        self,
        close_range_m:  float = 10.0,
        person_range_m: float = 15.0,
    ) -> None:
        self._close_range_m  = close_range_m
        self._person_range_m = person_range_m

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: list[Detection],
        frame_id:   int,
    ) -> DrivingDecision:
        """
        Evaluate a list of detections and return a DrivingDecision.

        Parameters
        ----------
        detections : list[Detection]  Filtered detections from the current frame.
        frame_id   : int              Current frame index.

        Returns
        -------
        DrivingDecision
        """
        decision = DrivingDecision(frame_id=frame_id)

        for det in detections:
            dist = det.estimated_distance_m  # may be None

            # ── Stop sign ────────────────────────────────────────────────────
            if det.class_name == "stop sign" and det.confidence >= 0.50:
                decision.alerts.append(
                    f"STOP SIGN detected  conf={det.confidence:.2f}"
                )
                self._upgrade(decision, "STOP")

            # ── Traffic light ────────────────────────────────────────────────
            elif det.class_name == "traffic light":
                decision.alerts.append(
                    "TRAFFIC LIGHT detected "
                    "(colour classification not yet implemented)"
                )
                self._upgrade(decision, "CAUTION")

            # ── Pedestrian proximity ─────────────────────────────────────────
            elif det.class_name == "person":
                if dist is not None and dist < self._person_range_m:
                    decision.alerts.append(
                        f"PEDESTRIAN within {dist:.1f} m — reduce speed"
                    )
                    self._upgrade(decision, "SLOW_DOWN")

            # ── Generic close-range object ───────────────────────────────────
            elif dist is not None and dist < self._close_range_m:
                decision.alerts.append(
                    f"{det.class_name.upper()} close at {dist:.1f} m"
                )
                self._upgrade(decision, "CAUTION")

        return decision

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _upgrade(self, decision: DrivingDecision, new_action: str) -> None:
        """
        Upgrade decision.suggested_action only if *new_action* has higher
        priority than the current value (lower index in _PRIORITY list).
        """
        current_idx = self._PRIORITY.index(decision.suggested_action)
        new_idx     = self._PRIORITY.index(new_action)
        if new_idx < current_idx:
            decision.suggested_action = new_action
