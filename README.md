# CAV Camera Perception Dashboard

Multi-panel autonomous vehicle perception dashboard. USB camera in, research-demo dashboard out.

---

## What it shows

```
┌──────────────┬──────────────────────────┬──────────────┐
│  RAW CAMERA  │                          │   OBJECT     │
│   + FPS      │    SEMANTIC WORLD MAP    │  DETECTION   │
│              │    (pseudo-3D BEV)       │  + Action    │
├──────────────┤                          ├──────────────┤
│ SLAM/FEATURES│                          │   TRACKED    │
│  (or DEPTH)  │                          │  PREDICTION  │
└──────────────┴──────────────────────────┴──────────────┘
```

- **Raw camera** with FPS counter
- **SLAM / Features** panel: ORB keypoints on Canny edges (green, SLAM aesthetic). Replaced by MiDaS INFERNO depth colourmap when depth model loads successfully.
- **Semantic World Map**: pseudo-3D bird's-eye-view. Detected objects projected onto a top-down canvas using estimated distance + lateral position. Per-track trails and linear prediction arrows included. Ego vehicle shown as cyan triangle.
- **Object Detection**: YOLO11 boxes + track IDs + distance labels
- **Tracked Prediction**: darkened frame with fading colour trails and dashed linear extrapolation per track

Action states: `NOMINAL` / `CAUTION` / `SLOW_DOWN` / `STOP` / `WAIT` / `PROCEED`

---

## Setup

```bash
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**macOS only (one-time):** System Settings → Privacy & Security → Camera → enable Terminal.

---

## Running

```bash
python main.py --camera 0 --model yolo11n.pt --conf 0.35
```

Other options:

```bash
python main.py --camera 1            # different camera index
python main.py --no-v2x --no-log     # detection + dashboard only
python main.py --save-output         # auto-save every frame to outputs/
python main.py --conf 0.45           # stricter confidence
```

Keys: `q` / `ESC` to quit, `s` to screenshot.

Screenshots save to `outputs/cav_<timestamp>.jpg`.

---

## Depth estimation

On first run the pipeline tries to download and load **MiDaS_small** (~30 MB via torch.hub). If it loads, the bottom-left panel shows a depth colourmap and the banner prints `MiDaS_small`. If it fails for any reason (no internet, GPU issues, etc.) it falls back silently to bounding-box-based distance — the dashboard still runs normally.

To skip MiDaS entirely, set `depth.enabled: false` in `configs/config.yaml`.

---

## Project layout

```
├── main.py
├── configs/
│   └── config.yaml          # all tuneable parameters
├── src/
│   ├── camera.py            # USB capture
│   ├── detector.py          # YOLO11 + ByteTrack
│   ├── tracker.py           # ByteTrack/BoT-SORT wrapper
│   ├── depth.py             # MiDaS depth (with bbox fallback)
│   ├── bev_mapper.py        # pseudo-3D BEV world map
│   ├── dashboard.py         # multi-panel compositor
│   ├── decision_engine.py   # stop-sign state machine
│   ├── v2x.py               # V2X BSM/DOM/ISM simulation
│   ├── logger.py            # CSV + JSONL writer
│   └── utils.py             # FPS counter, config loader, etc.
├── logs/                    # detections.csv + detections.jsonl
└── outputs/                 # saved dashboard screenshots
```

---

## Configuration

All parameters in `configs/config.yaml`. Common ones:

```yaml
model:
  confidence: 0.35           # detection threshold
  tracker: "bytetrack.yaml"  # or "botsort.yaml"

depth:
  enabled: true              # false = always use bbox-distance fallback
  every_n_frames: 5          # run depth only every Nth frame (saves CPU)

bev:
  max_range_m: 20.0          # furthest distance shown on world map
```

---

## Ubuntu / Jetson

```bash
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --camera 0
```

Camera permission issue: `sudo usermod -aG video $USER` then log out and back in.

For Jetson: install the NVIDIA PyTorch wheel for your JetPack version **before** `pip install -r requirements.txt` to get CUDA depth inference.

---

## Logs

Two files written simultaneously:
- `logs/detections.csv` — one row per detection per frame
- `logs/detections.jsonl` — same data as JSON Lines

Both flush cleanly on quit.

---

## Notes

- The BEV world map is **not real SLAM**. It uses estimated distance (bounding-box width formula or MiDaS) and the object's horizontal position in the frame to project onto the top-down canvas. Accurate only when camera is calibrated.
- V2X messages (BSM/DOM/ISM) are simulated to stdout — not real DSRC/C-V2X radio.
- Distance estimates are approximate. For real-world use, calibrate the focal length with a checkerboard or use stereo/LiDAR.

---
