# CAV Camera Perception Pipeline

Real-time perception stack for Connected Autonomous Vehicles. Runs on a MacBook for development and deploys to Ubuntu or NVIDIA Jetson for real hardware.

Plugs in a USB camera, runs YOLO11 detection with ByteTrack tracking, makes basic stop-sign driving decisions, and simulates V2X messages — all in one Python process.

---

## What it does

- Detects people, cars, bikes, trucks, traffic lights, and stop signs via YOLO11
- Tracks objects across frames with ByteTrack (persistent IDs)
- Estimates rough monocular distance from bounding box height
- Runs a stop-sign state machine: IDLE → STOPPING → WAIT → PROCEED
- Broadcasts simulated V2X messages every second (BSM, DOM, ISM) as JSON to stdout
- Logs every detection to `logs/detections.csv` and `logs/detections.jsonl`
- Shows a live annotated window with FPS, decision state, and object labels
- Press `s` to screenshot at any time

---

## Setup

Tested on macOS (Apple Silicon) and Ubuntu 22.04. Python 3.10+.

```bash
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, `ultralytics` auto-downloads `yolo11n.pt` (~6 MB).

**macOS only:** you need to grant Terminal camera access once — System Settings → Privacy & Security → Camera → enable Terminal.

---

## Running

```bash
python main.py
```

Common options:

```bash
python main.py --camera 1          # use a different camera index
python main.py --conf 0.4          # raise confidence threshold
python main.py --no-v2x --no-log   # just detection + display
python main.py --save-frames       # save every frame to outputs/
```

Press `q` or `ESC` to quit. Press `s` to save a screenshot.

---

## Project layout

```
├── main.py                  # entry point
├── configs/
│   └── config.yaml          # all tunable parameters
├── src/
│   ├── camera.py            # USB camera capture
│   ├── detector.py          # YOLO inference + annotation
│   ├── tracker.py           # ByteTrack wrapper
│   ├── decision_engine.py   # stop-sign state machine
│   ├── v2x.py               # V2X BSM/DOM/ISM simulation
│   ├── logger.py            # CSV + JSONL writer
│   └── utils.py             # FPS counter, overlays, config loader
├── logs/                    # created at runtime
└── outputs/                 # screenshots and saved frames
```

---

## Configuration

Everything lives in `configs/config.yaml`. Commonly changed values:

```yaml
camera:
  index: 0
  width: 1280
  height: 720

model:
  confidence: 0.35
  tracker: "bytetrack.yaml"   # swap to "botsort.yaml" if preferred

decision:
  stop_hold_frames: 30        # how long to hold at a stop sign
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

If the camera is permission-denied: `sudo usermod -aG video $USER` then log out and back in.

For Jetson, install the NVIDIA PyTorch wheel for your JetPack version before running `pip install -r requirements.txt`, so you get CUDA inference instead of CPU.

---

## Logs

Detections are written to two files at the same time:

- `logs/detections.csv` — standard CSV, easy to open in Excel or pandas
- `logs/detections.jsonl` — one JSON object per line, good for streaming/processing

Both are flushed and closed cleanly on quit.

---

## What's next

- Camera calibration for accurate distance (currently a rough estimate)
- Traffic light color classification (right now it just detects the object)
- ROS 2 node wrapper
- TensorRT export for Jetson
- Actual UDP multicast for V2X instead of stdout simulation

---

## Requirements

- Python 3.10+
- opencv-python >= 4.9
- ultralytics >= 8.3
- numpy >= 1.26
- pandas >= 2.2
- pyyaml >= 6.0

---

## License

MIT
