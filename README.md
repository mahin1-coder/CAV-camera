# CAV Camera Perception Pipeline

> Real-time object detection and situational awareness for **Connected Autonomous Vehicles (CAV)**  
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
