"""
CAV Camera Perception Pipeline — entry point  (v2 research upgrade)
====================================================================

Real-time object detection, tracking, and situational awareness for
Connected Autonomous Vehicles (CAV) using a USB camera and YOLO11.

Usage
-----
    python main.py                                   # defaults from config.yaml
    python main.py --camera 1                        # external USB camera
    python main.py --model yolov8n.pt                # alternative YOLO model
    python main.py --conf 0.40                       # confidence threshold
    python main.py --no-v2x --no-log                 # disable V2X and logging
    python main.py --save-frames                     # save every output frame

Keyboard shortcuts
------------------
    q / ESC    — quit
    s          — save current frame screenshot to outputs/
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src.camera          import Camera
from src.decision_engine import DecisionEngine
from src.detector        import Detector
from src.logger          import DetectionLogger
from src.utils           import (
    FPSCounter,
    draw_decision,
    draw_fps,
    draw_info_bar,
    load_config,
    merge_cli,
    save_frame,
)
from src.v2x             import V2XSimulator


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CAV Camera Perception Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera",       type=int,   default=None,
                   help="USB camera index (overrides config.yaml).")
    p.add_argument("--model",        type=str,   default=None,
                   help="YOLO model name / path (overrides config.yaml).")
    p.add_argument("--conf",         type=float, default=None,
                   help="Detection confidence threshold (overrides config.yaml).")
    p.add_argument("--no-v2x",       action="store_true",
                   help="Disable the V2X broadcast simulator.")
    p.add_argument("--no-log",       action="store_true",
                   help="Disable CSV + JSON detection logging.")
    p.add_argument("--save-frames",  action="store_true",
                   help="Save every annotated frame to outputs/.")
    p.add_argument("--config",       type=str,   default="configs/config.yaml",
                   help="Path to YAML config file.")
    return p


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> int:
    args = _build_parser().parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    cfg = merge_cli(cfg, args.camera, args.model, args.conf)

    # ── Subsystem init ────────────────────────────────────────────────────────
    camera = Camera(cfg["camera"])
    try:
        camera.open()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    detector        = Detector(cfg)
    decision_engine = DecisionEngine(cfg.get("decision", {}))
    fps_counter     = FPSCounter(window=30)

    logger: DetectionLogger | None = None
    if not args.no_log and cfg.get("logging", {}).get("enabled", True):
        logger = DetectionLogger(cfg.get("logging", {}))

    v2x: V2XSimulator | None = None
    if not args.no_v2x and cfg.get("v2x", {}).get("enabled", True):
        v2x = V2XSimulator(cfg.get("v2x", {}))
        v2x.start()

    output_cfg   = cfg.get("output", {})
    save_every   = args.save_frames or output_cfg.get("save_frames", False)
    save_on_det  = output_cfg.get("save_on_detection", False)
    output_dir   = output_cfg.get("output_dir", "outputs")
    log_n        = cfg.get("logging", {}).get("log_every_n_frames", 1)
    win_name     = cfg.get("display", {}).get("window_name", "CAV Perception Pipeline")

    # ── Startup banner ────────────────────────────────────────────────────────
    sep = "─" * 58
    print(
        f"\n{sep}\n"
        f"  CAV Camera Perception Pipeline  (v2)\n"
        f"{sep}\n"
        f"  Camera  : index={cfg['camera']['index']}  "
        f"{cfg['camera']['width']}x{cfg['camera']['height']}@{cfg['camera']['fps']}\n"
        f"  Model   : {cfg['model']['name']}  "
        f"conf={cfg['model']['confidence']}  "
        f"tracker={cfg['model'].get('tracker','bytetrack.yaml')}\n"
        f"  Logging : {'disabled' if args.no_log else 'CSV + JSONL'}\n"
        f"  V2X sim : {'disabled' if args.no_v2x else 'enabled (BSM/DOM/ISM)'}\n"
        f"  Press  q / ESC  to quit   |   s  to screenshot\n"
        f"{sep}\n"
    )

    # ── Capture loop ──────────────────────────────────────────────────────────
    frame_id = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[Main] Frame capture failed — camera disconnected?")
                break

            h, w = frame.shape[:2]

            # Detection + tracking
            detections = detector.detect(frame)

            # Decision engine
            decision = decision_engine.evaluate(detections, frame_id, frame_w=w)

            # V2X (non-blocking background thread)
            if v2x is not None:
                v2x.update(detections)

            # Logging
            if logger is not None and frame_id % log_n == 0:
                logger.log(detections, frame_id)

            # Annotate frame
            detector.annotate_frame(frame, detections)

            fps = fps_counter.tick()
            if cfg.get("display", {}).get("show_fps", True):
                draw_fps(frame, fps)

            draw_decision(frame, decision.suggested_action, decision.alerts)
            draw_info_bar(
                frame,
                f"frame={frame_id}  objects={len(detections)}  "
                f"model={cfg['model']['name']}",
            )

            # Frame saving
            if save_every or (save_on_det and detections):
                saved = save_frame(frame, output_dir, prefix="cav")
                # Only print occasionally to avoid log spam
                if frame_id % 30 == 0:
                    print(f"[Main] Frame saved → {saved}")

            # Display
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):          # q or ESC
                print("[Main] Quit signal received.")
                break
            elif key == ord("s"):              # manual screenshot
                saved = save_frame(frame, output_dir, prefix="screenshot")
                print(f"[Main] Screenshot → {saved}")

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
