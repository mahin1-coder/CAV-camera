"""
CAV Camera Perception Dashboard — entry point  (v3)
====================================================

Advanced multi-panel perception dashboard for Connected Autonomous Vehicles.

Panels
------
  Top-left     RAW CAMERA + FPS overlay
  Bottom-left  SLAM / FEATURES  (ORB keypoints + Canny edges)
               or DEPTH colourmap when MiDaS is available
  Centre       SEMANTIC WORLD MAP (pseudo-3D BEV, updated every frame)
  Top-right    OBJECT DETECTION overlay (YOLO11 + ByteTrack)
  Bottom-right TRACKED PREDICTION (trail + linear extrapolation)

Usage
-----
    python main.py                             # defaults from config.yaml
    python main.py --camera 1                  # external USB camera
    python main.py --conf 0.4                  # confidence threshold
    python main.py --model yolo11n.pt          # alternative YOLO model
    python main.py --no-v2x --no-log           # disable V2X and logging
    python main.py --save-output               # auto-save every dashboard frame
    python main.py --config path/to/cfg.yaml   # custom config

Keyboard controls
-----------------
    q / ESC    quit
    s          save current dashboard image to outputs/cav_<ts>.jpg
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src.bev_mapper        import BEVMapper
from src.camera            import Camera
from src.nuscenes_loader   import NuScenesLoader
from src.dashboard         import Dashboard, save_dashboard
from src.decision_engine import DecisionEngine
from src.depth           import DepthEstimator
from src.detector        import Detector
from src.logger          import DetectionLogger
from src.utils           import FPSCounter, load_config, merge_cli
from src.v2x             import V2XSimulator


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CAV Camera Perception Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera",      type=int,   default=None,
                   help="USB camera device index (overrides config.yaml).")
    p.add_argument("--model",       type=str,   default=None,
                   help="YOLO model name/path (overrides config.yaml).")
    p.add_argument("--conf",        type=float, default=None,
                   help="Detection confidence threshold (overrides config.yaml).")
    p.add_argument("--no-v2x",      action="store_true",
                   help="Disable V2X broadcast simulator.")
    p.add_argument("--no-log",      action="store_true",
                   help="Disable CSV + JSON detection logging.")
    p.add_argument("--save-output", action="store_true",
                   help="Save every dashboard frame to outputs/.")
    p.add_argument("--config",      type=str,   default="configs/config.yaml",
                   help="Path to YAML config file.")
    # ── nuScenes flags ────────────────────────────────────────────────────────
    p.add_argument("--source",      type=str,   default="camera",
                   choices=["camera", "nuscenes"],
                   help="Input source: 'camera' (USB) or 'nuscenes' (dataset).")
    p.add_argument("--nuscenes-data",    type=str, default=None,
                   help="Path to nuScenes dataset root (overrides config.yaml).")
    p.add_argument("--nuscenes-version", type=str, default=None,
                   help="nuScenes version string, e.g. v1.0-mini.")
    p.add_argument("--nuscenes-scene",   type=int, default=None,
                   help="Scene index to play (0-based).")
    p.add_argument("--nuscenes-camera",  type=str, default=None,
                   help="Camera channel, e.g. CAM_FRONT, CAM_BACK.")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _build_parser().parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    cfg = merge_cli(cfg, args.camera, args.model, args.conf)

    # ── Input source: USB camera or nuScenes dataset ─────────────────────────
    use_nuscenes = (args.source == "nuscenes")

    if use_nuscenes:
        ns_cfg = dict(cfg.get("nuscenes", {}))
        if args.nuscenes_data:    ns_cfg["dataroot"]  = args.nuscenes_data
        if args.nuscenes_version: ns_cfg["version"]   = args.nuscenes_version
        if args.nuscenes_scene is not None: ns_cfg["scene_idx"] = args.nuscenes_scene
        if args.nuscenes_camera:  ns_cfg["camera"]    = args.nuscenes_camera
        source = NuScenesLoader(ns_cfg)
    else:
        source = Camera(cfg["camera"])

    try:
        source.open()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    # ── Detector (YOLO + ByteTrack) ───────────────────────────────────────────
    detector = Detector(cfg)

    # ── Depth estimator (MiDaS, optional) ────────────────────────────────────
    depth_est = DepthEstimator(cfg.get("depth", {}))

    # ── BEV world-map renderer ────────────────────────────────────────────────
    bev_mapper = BEVMapper(cfg.get("bev", {}))

    # ── Dashboard assembler ───────────────────────────────────────────────────
    dashboard = Dashboard(depth_available=depth_est.available)

    # ── Decision engine ───────────────────────────────────────────────────────
    decision_engine = DecisionEngine(cfg.get("decision", {}))

    # ── Optional subsystems ───────────────────────────────────────────────────
    logger: DetectionLogger | None = None
    if not args.no_log and cfg.get("logging", {}).get("enabled", True):
        logger = DetectionLogger(cfg.get("logging", {}))

    v2x: V2XSimulator | None = None
    if not args.no_v2x and cfg.get("v2x", {}).get("enabled", True):
        v2x = V2XSimulator(cfg.get("v2x", {}))
        v2x.start()

    # ── Runtime options ───────────────────────────────────────────────────────
    output_cfg  = cfg.get("output", {})
    save_every  = args.save_output or output_cfg.get("save_frames", False)
    output_dir  = output_cfg.get("output_dir", "outputs")
    log_n       = cfg.get("logging", {}).get("log_every_n_frames", 1)
    win_name    = cfg.get("display", {}).get("window_name", "CAV Perception Dashboard")

    fps_counter = FPSCounter(window=30)

    # ── Banner ────────────────────────────────────────────────────────────────
    sep = "─" * 62
    depth_status = "MiDaS_small" if depth_est.available else "bbox-distance fallback"
    if use_nuscenes:
        src_str = (f"nuScenes {source._version}  scene={source._scene_idx} "
                   f"'{source.scene_name}'  {source.total_frames} frames  "
                   f"ch={source._channel}")
    else:
        src_str = (f"USB index={cfg['camera']['index']}  "
                   f"{cfg['camera']['width']}x{cfg['camera']['height']}"
                   f"@{cfg['camera']['fps']}")
    print(
        f"\n{sep}\n"
        f"  CAV Camera Perception Dashboard  (v3)\n"
        f"{sep}\n"
        f"  Source  : {src_str}\n"
        f"  Model   : {cfg['model']['name']}  "
        f"conf={cfg['model']['confidence']}  "
        f"tracker={cfg['model'].get('tracker', 'bytetrack.yaml')}\n"
        f"  Depth   : {depth_status}\n"
        f"  Logging : {'CSV + JSONL' if logger else 'disabled'}\n"
        f"  V2X sim : {'enabled' if v2x else 'disabled'}\n"
        f"  Press  q / ESC  to quit   |   s  to screenshot\n"
        f"{sep}\n"
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_id = 0
    depth_u8 = None   # last computed depth map

    while True:
        # ── Capture ───────────────────────────────────────────────────────────
        ret, frame = source.read()
        if not ret or frame is None:
            if use_nuscenes:
                print("[Main] nuScenes scene complete.")
            else:
                print("[Main] Frame capture failed — camera disconnected?")
            break

        fps = fps_counter.tick()
        fh, fw = frame.shape[:2]

        # ── Detection ─────────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Depth (every N frames) ────────────────────────────────────────────
        if depth_est.should_run():
            new_depth = depth_est.estimate(frame)
            if new_depth is not None:
                depth_u8 = new_depth

        # ── Decision ──────────────────────────────────────────────────────────
        decision = decision_engine.evaluate(detections, frame_id, frame_w=fw)

        # ── BEV world map ─────────────────────────────────────────────────────
        bev_frame = bev_mapper.render(
            detections, fw, fh,
            action=decision.suggested_action,
        )
        active_ids = {d.track_id for d in detections if d.track_id is not None}
        bev_mapper.reset_stale_trails(active_ids)

        # ── Dashboard ─────────────────────────────────────────────────────────
        dash = dashboard.render(
            raw_frame  = frame,
            bev_frame  = bev_frame,
            detections = detections,
            decision   = decision,
            fps        = fps,
            frame_id   = frame_id,
            depth_u8   = depth_u8,
        )

        # ── Logging ───────────────────────────────────────────────────────────
        if logger and frame_id % log_n == 0:
            logger.log(detections, frame_id)

        # ── V2X ───────────────────────────────────────────────────────────────
        if v2x:
            v2x.update(detections)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(win_name, dash)

        # ── Auto-save ─────────────────────────────────────────────────────────
        if save_every:
            save_dashboard(dash, output_dir)

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = save_dashboard(dash, output_dir)
            print(f"[Main] Screenshot saved → {path}")

        frame_id += 1

    # ── Shutdown ──────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    source.release()
    if logger:
        logger.close()
    if v2x:
        v2x.stop()
    print("[Main] Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
