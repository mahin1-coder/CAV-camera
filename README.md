# CAV Camera Perception Pipeline  (v2)

> Research-grade real-time perception system for **Connected Autonomous Vehicles (CAV)**  
> USB camera · YOLO11 · ByteTrack · Stop-sign decision engine · V2X simulation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-orange)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green)](https://opencv.org)

---

## Overview

| Feature | Status |
|---|---|
| USB camera capture (Mac / Ubuntu / Jetson) | ✅ |
| YOLO11 object detection | ✅ |
| ByteTrack persistent object tracking | ✅ |
| Live annotated display with FPS counter | ✅ |
| Detection confidence filtering | ✅ |
| CSV + JSON Lines logging | ✅ |
| Screenshot / frame saving (`s` key) | ✅ |
| Monocular distance estimation | ⚠️ Placeholder |
| Stop-sign intersection decision engine (WAIT/PROCEED) | ✅ |
| V2X simulation — BSM, DOM, ISM messages | ✅ |
| YAML config system | ✅ |
| CLI arguments | ✅ |

### Detected classes
`person` · `bicycle` · `car` · `motorcycle` · `bus` · `truck` · `traffic light` · `stop sign`

---

## Project Structure

```
cav-camera-perception/
├── main.py                   ← Entry point
├── requirements.txt
├── .gitignore
├── README.md
├── configs/
│   └── config.yaml           ← All tunable parameters
├── logs/
│   ├── detections.csv        ← Created at runtime
│   └── detections.jsonl      ← JSON Lines log
├── outputs/                  ← Saved frames / screenshots
└── src/
    ├── __init__.py
    ├── camera.py             ← USB camera handler
    ├── detector.py           ← YOLO inference + frame annotation
    ├── tracker.py            ← ByteTrack / BoT-SORT wrapper
    ├── decision_engine.py    ← Stop-sign state machine + safety rules
    ├── v2x.py                ← V2X message simulation (BSM/DOM/ISM)
    ├── logger.py             ← Thread-safe CSV + JSONL logger
    └── utils.py              ← FPSCounter, overlays, config loader
```

---

## Quick Start (macOS)

### 1. Clone
```bash
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera
```

### 2. Virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> `ultralytics` auto-downloads `yolo11n.pt` (~6 MB) on first run.

### 4. Grant camera permission (macOS only, one-time)
**System Settings → Privacy & Security → Camera → enable Terminal**

### 5. Run
```bash
python main.py
```

---

## CLI Reference

```
python main.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--camera INDEX` | `0` | USB camera device index |
| `--model MODEL` | `yolo11n.pt` | YOLO model name or path |
| `--conf FLOAT` | `0.35` | Detection confidence threshold |
| `--no-v2x` | off | Disable V2X broadcast simulator |
| `--no-log` | off | Disable CSV + JSON logging |
| `--save-frames` | off | Save every annotated frame to `outputs/` |
| `--config PATH` | `configs/config.yaml` | Path to YAML config |

### Examples
```bash
python main.py --camera 0 --model yolo11n.pt --conf 0.35
python main.py --camera 1                        # external USB camera
python main.py --no-v2x --no-log                 # minimal mode
python main.py --save-frames                     # record all frames
```

### Keyboard shortcuts
| Key | Action |
|---|---|
| `q` / `ESC` | Quit |
| `s` | Save screenshot to `outputs/` |

---

## Configuration

All parameters live in **`configs/config.yaml`** — no code changes needed for common adjustments:

```yaml
camera:
  index: 0          # USB camera index
  width: 1280
  height: 720
  fps: 30

model:
  name: "yolo11n.pt"
  confidence: 0.35
  tracker: "bytetrack.yaml"   # or "botsort.yaml"

decision:
  stop_sign_confidence: 0.50
  stop_hold_frames: 30         # frames to hold at stop sign
  cross_traffic_x_ratio: 0.25  # side-zone fraction for cross-traffic detection
```

---

## Decision Engine — Stop-Sign Scenario

The engine implements a state machine:

```
IDLE → STOPPING → WAIT (cross traffic) → STOPPING → PROCEED → IDLE
                ↘ PROCEED (no traffic, hold expired) ↗
```

| Action | Trigger |
|---|---|
| `STOP` | Stop sign detected, evaluating |
| `WAIT` | Stop sign + vehicle in left/right cross-traffic zone |
| `PROCEED` | Hold timer expired, no cross traffic |
| `SLOW_DOWN` | Pedestrian within 15 m |
| `CAUTION` | Traffic light / object within 10 m |
| `NOMINAL` | No hazards |

---

## V2X Simulation

Three message types are printed as JSON to stdout every second:

| Type | Description |
|---|---|
| `BSM` | Basic Safety Message — ego vehicle status + all perceived objects |
| `DOM` | Detected-Object Message — one message per detected object |
| `ISM` | Intersection State Message — simulated SPaT from Road-Side Unit |

---

## Logs

| File | Format | Description |
|---|---|---|
| `logs/detections.csv` | CSV (Pandas) | Structured detection rows |
| `logs/detections.jsonl` | JSON Lines | One JSON object per detection row |

---

## Ubuntu / Jetson Setup

```bash
# 1. Clone
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Check camera
ls /dev/video*
sudo usermod -aG video $USER   # log out and back in if permission denied

# 5. Run
python main.py --camera 0
```

### Jetson CUDA notes
Install the NVIDIA-provided PyTorch wheel **before** `pip install -r requirements.txt`:
```bash
# Example for JetPack 6.x — check https://forums.developer.nvidia.com/t/pytorch-for-jetson
pip install torch torchvision --index-url <NVIDIA_JETSON_WHEEL_URL>
pip install -r requirements.txt
```

---

## Push to GitHub

```bash
git add .
git commit -m "Upgrade CAV perception pipeline"
git branch -M main
git remote add origin https://github.com/mahin1-coder/CAV-camera.git
git push -u origin main
```

---

## Architecture

```
USB Camera
    │  BGR frame
    ▼
camera.py ──────────────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            │
detector.py  (YOLO11 + ByteTrack)                               │
    │  list[Detection]                                           │
    ├──▶ logger.py          → logs/detections.csv + .jsonl      │
    ├──▶ decision_engine.py → DrivingDecision (state machine)   │
    ├──▶ v2x.py             → BSM / DOM / ISM (stdout JSON)     │
    └──▶ annotate_frame()   ────────────────────────────────────▶│
                                                                 │
                                                        cv2.imshow()
```

---

## Roadmap

- [ ] Camera calibration for accurate focal length
- [ ] Traffic-light colour classification (green / yellow / red)
- [ ] Depth model / stereo vision for real distance estimation
- [ ] ROS 2 integration (Humble / Iron)
- [ ] UDP multicast V2X simulation (multi-vehicle LAN)
- [ ] CARLA / SUMO simulator integration
- [ ] Jetson TensorRT export (`yolo11n.engine`)
- [ ] AEB (Automatic Emergency Braking) via time-to-collision

---

## Requirements

- Python **3.10+**
- [OpenCV](https://opencv.org/) `>=4.9`
- [Ultralytics](https://github.com/ultralytics/ultralytics) `>=8.3` (includes ByteTrack)
- [NumPy](https://numpy.org/) `>=1.26`
- [Pandas](https://pandas.pydata.org/) `>=2.2`
- [PyYAML](https://pyyaml.org/) `>=6.0`

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@misc{cav-camera-perception,
  title  = {CAV Camera Perception Pipeline},
  year   = {2026},
  url    = {https://github.com/mahin1-coder/CAV-camera}
}
```
> using a USB camera, YOLO11, and a modular Python pipeline.

---

## Overview

This project implements a real-time perception stack that can run on a **MacBook** during development and be deployed to **Ubuntu** or **NVIDIA Jetson** hardware.

| Feature | Status |
|---|---|
| USB camera capture | ✅ Implemented |
| YOLO11 object detection | ✅ Implemented |
| Live annotated video display | ✅ Implemented |
| FPS overlay | ✅ Implemented |
| Detection CSV logging | ✅ Implemented |
| Monocular distance estimation | ⚠️ Placeholder (needs camera calibration) |
| Object tracking (ID assignment) | ⚠️ Placeholder (YOLO `.track()` hook ready) |
| Rule-based decision engine | ✅ Implemented (basic rules) |
| V2V / V2I message simulation | ✅ Implemented (stdout simulation) |

### Detected classes
`person` · `bicycle` · `car` · `motorcycle` · `bus` · `truck` · `traffic light` · `stop sign`

---

## Project Structure

```
cav-camera-perception/
├── main.py                  # Entry point — run this
├── requirements.txt         # Python dependencies
├── .gitignore
├── logs/
│   └── detections.csv       # Created at runtime
└── src/
    ├── __init__.py
    ├── config.py            # All tunable parameters in one place
    ├── camera.py            # USB camera handler
    ├── detector.py          # YOLO inference + frame annotation
    ├── logger.py            # Thread-safe CSV logger (Pandas)
    ├── decision_engine.py   # Rule-based driving decision placeholder
    └── v2x_simulator.py     # V2V / V2I message simulation placeholder
```

---

## Quick Start (macOS)

### 1. Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd cav-camera-perception
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** `ultralytics` will automatically download `yolo11n.pt` (~6 MB) on the first run.

### 4. Run the pipeline
```bash
python main.py
```

Press **`q`** or **`ESC`** to quit.

---

## CLI Options

```
python main.py [--camera INDEX] [--model MODEL] [--no-v2x] [--no-log]
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | USB camera device index |
| `--model` | `yolo11n.pt` | YOLO model name or path |
| `--no-v2x` | off | Disable V2X broadcast simulator |
| `--no-log` | off | Disable CSV detection logging |

### Examples
```bash
python main.py --camera 1                  # Use external USB camera
python main.py --model yolov8n.pt          # Use YOLOv8 nano instead
python main.py --no-v2x --no-log           # Minimal mode (detection + display only)
```

---

## Ubuntu / Jetson Setup

```bash
# 1. Clone
git clone <YOUR_GITHUB_REPO_URL>
cd cav-camera-perception

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify camera is visible
ls /dev/video*
# If permission denied:
sudo usermod -aG video $USER   # log out, log back in

# 5. Run
python main.py
```

### Jetson-specific notes
- Use **JetPack 5.x / 6.x** with Python 3.10+.
- Install PyTorch from the [NVIDIA Jetson PyTorch wheel index](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) **before** running `pip install -r requirements.txt` to get CUDA-accelerated inference.
- For CSI cameras (IMX219 etc.) change the camera backend in `src/camera.py` to use a GStreamer pipeline via `cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)`.

---

## Configuration

All tunable parameters live in **`src/config.py`** — no changes to any other file are needed for common adjustments:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | USB camera index |
| `MODEL_NAME` | `yolo11n.pt` | YOLO model |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `FOCAL_LENGTH_PX` | `700.0` | For distance estimation (calibrate per camera) |
| `LOG_EVERY_N_FRAMES` | `1` | Log frequency (increase to reduce I/O) |
| `V2X_BROADCAST_INTERVAL_S` | `1.0` | V2X message interval |

---

## Detection Log

Detections are saved to `logs/detections.csv` with the following columns:

| Column | Description |
|---|---|
| `timestamp_utc` | ISO-8601 UTC timestamp |
| `frame_id` | Monotonic frame counter |
| `track_id` | Object tracking ID (if enabled) |
| `class_name` | Detected object class |
| `confidence` | Detection confidence [0–1] |
| `x1,y1,x2,y2` | Bounding box pixel coordinates |
| `estimated_distance_m` | Monocular distance estimate (placeholder) |

---

## Architecture

```
USB Camera
    │
    ▼
 camera.py  ──────────────────────────────────────────────────────┐
    │  BGR frame                                                   │
    ▼                                                              │
 detector.py (YOLO11)                                             │
    │  list[Detection]                                             │
    ├──▶ logger.py       →  logs/detections.csv                   │
    ├──▶ decision_engine.py  →  DrivingDecision (NOMINAL/STOP/…)  │
    ├──▶ v2x_simulator.py    →  BSM / SPaT (stdout)               │
    └──▶ annotate_frame()  ──────────────────────────────────────▶│
                                                                   │
                                                          cv2.imshow()
```

---

## Roadmap

- [ ] Camera calibration script (checkerboard) for accurate focal length
- [ ] Replace monocular distance estimate with YOLOv8-Pose or a depth model
- [ ] Enable YOLO `.track()` for persistent object tracking IDs
- [ ] Traffic-light colour classification (green / yellow / red)
- [ ] ROS 2 integration (Humble / Iron)
- [ ] UDP multicast V2X simulation (multi-vehicle LAN demo)
- [ ] CARLA simulator integration
- [ ] Jetson TensorRT export (`yolo11n.engine`)

---

## Requirements

- Python **3.10+**
- [OpenCV](https://opencv.org/) `>=4.9`
- [Ultralytics](https://github.com/ultralytics/ultralytics) `>=8.3`
- [NumPy](https://numpy.org/) `>=1.26`
- [Pandas](https://pandas.pydata.org/) `>=2.2`

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial CAV camera perception pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this project in academic work, please cite it as:

```
@misc{cav-camera-perception,
  title  = {CAV Camera Perception Pipeline},
  year   = {2024},
  url    = {<YOUR_GITHUB_REPO_URL>}
}
```
