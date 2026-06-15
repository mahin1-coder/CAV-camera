"""
CAV Camera Perception Pipeline — entry point.
=============================================

Real-time object detection and situational awareness for
Connected Autonomous Vehicles (CAV) using a USB camera and YOLO.

Usage
-----
    python main.py                        # default USB camera (index 0)
    python main.py --camera 1             # external USB camera
    python main.py --model yolov8n.pt     # alternative YOLO model
    python main.py --no-v2x --no-log      # skip V2X sim and CSV logging

Keyboard shortcuts
------------------
    q  /  ESC — quit the pipeline
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.camera import Camera
from src.config import (
    CAMERA_INDEX,
    DISPLAY_WINDOW_NAME,
    FPS_POSITION,
    LOG_EVERY_N_FRAMES,
    MODEL_NAME,
)
from src.decision_engine import DecisionEngine
from src.detector import Detector
from src.logger import DetectionLogger
from src.v2x_simulator import V2XSimulator


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAV Camera Perception Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        metavar="INDEX",
        help="USB camera device index (0 = first / built-in camera).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        metavar="MODEL",
        help="Ultralytics YOLO model name or path  (e.g. yolo11n.pt).",
    )
    parser.add_argument(
        "--no-v2x",
        action="store_true",
        help="Disable the V2X broadcast simulator.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable CSV detection logging.",
    )
    return parser


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _draw_fps(frame, fps: float) -> None:
    """Render the current FPS in the top-left corner."""
    cv2.putText(
        frame,
        f"FPS: {fps:5.1f}",
        FPS_POSITION,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_decision(frame, action: str, alerts: list[str]) -> None:
    """Render the decision-engine output on the frame."""
    h, w   = frame.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX

    color_map: dict[str, tuple[int, int, int]] = {
        "NOMINAL":   (0, 200,   0),
        "CAUTION":   (0, 200, 255),
        "SLOW_DOWN": (0, 165, 255),
        "STOP":      (0,   0, 220),
    }
    color = color_map.get(action, (200, 200, 200))

    # Action label — top-right
    action_text = f"Action: {action}"
    (lw, _), _ = cv2.getTextSize(action_text, font, 0.65, 2)
    cv2.putText(
        frame, action_text,
        (w - lw - 10, 30),
        font, 0.65, color, 2, cv2.LINE_AA,
    )

    # Alert lines — bottom-left (up to 4 most recent)
    for i, alert in enumerate(alerts[:4]):
        cv2.putText(
            frame,
            f"! {alert[:70]}",
            (10, h - 15 - i * 22),
            font, 0.46, (0, 200, 255), 1, cv2.LINE_AA,
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> int:
    args = _build_arg_parser().parse_args()

    # ── Camera ────────────────────────────────────────────────────────────────
    camera = Camera(index=args.camera)
    try:
        camera.open()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    # ── Detector ──────────────────────────────────────────────────────────────
    detector = Detector(model_name=args.model)

    # ── Decision engine ───────────────────────────────────────────────────────
    decision_engine = DecisionEngine()

    # ── Logger (optional) ─────────────────────────────────────────────────────
    logger: DetectionLogger | None = None
    if not args.no_log:
        logger = DetectionLogger()

    # ── V2X simulator (optional) ──────────────────────────────────────────────
    v2x: V2XSimulator | None = None
    if not args.no_v2x:
        v2x = V2XSimulator()
        v2x.start()

    # ── Print startup summary ─────────────────────────────────────────────────
    print(
        f"\n{'─' * 55}\n"
        f"  CAV Camera Perception Pipeline\n"
        f"{'─' * 55}\n"
        f"  Camera index : {args.camera}\n"
        f"  YOLO model   : {args.model}\n"
        f"  Logging      : {'disabled' if args.no_log else 'enabled'}\n"
        f"  V2X sim      : {'disabled' if args.no_v2x else 'enabled'}\n"
        f"  Press  q  or  ESC  to quit\n"
        f"{'─' * 55}\n"
    )

    # ── Capture loop ──────────────────────────────────────────────────────────
    frame_id   = 0
    fps        = 0.0
    prev_time  = time.perf_counter()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[Main] Frame capture failed — camera disconnected?")
                break

            # Object detection
            detections = detector.detect(frame)

            # Decision logic
            decision = decision_engine.evaluate(detections, frame_id)

            # V2X update (non-blocking; runs on its own thread)
            if v2x is not None:
                v2x.update_detections(detections)

            # CSV logging (throttled by LOG_EVERY_N_FRAMES)
            if logger is not None and frame_id % LOG_EVERY_N_FRAMES == 0:
                logger.log(detections, frame_id)

            # Frame annotation
            detector.annotate_frame(frame, detections)
            _draw_fps(frame, fps)
            _draw_decision(frame, decision.suggested_action, decision.alerts)

            # Display
            cv2.imshow(DISPLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # q or ESC
                print("[Main] Quit signal received.")
                break

            # FPS update
            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — shutting down.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        if logger is not None:
            logger.close()
        if v2x is not None:
            v2x.stop()

    print("[Main] Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
