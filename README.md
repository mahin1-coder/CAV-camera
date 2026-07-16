# CAV Camera Perception Dashboard

A real-time perception dashboard I built for connected and autonomous vehicle (CAV) research. Plug in a USB camera and it runs a full multi-panel pipeline — object detection, tracking, depth estimation, a bird's-eye-view world map, and V2X message simulation, all in one window.

---

## What it looks like

```
┌──────────────┬──────────────────────────┬──────────────┐
│  RAW CAMERA  │                          │   DETECTION  │
│   + FPS      │    BEV WORLD MAP         │   YOLO11     │
│              │    (pseudo-3D)           │   + Action   │
├──────────────┤                          ├──────────────┤
│ SLAM/DEPTH   │                          │   TRACKING   │
│  FEATURES    │                          │   + TRAILS   │
└──────────────┴──────────────────────────┴──────────────┘
```

**Top-left** — raw camera feed with live FPS  
**Bottom-left** — ORB keypoints on Canny edges (SLAM-style), switches to MiDaS depth colormap if the depth model loads  
**Centre** — bird's-eye-view world map: objects projected top-down with distance, trails, and prediction arrows  
**Top-right** — YOLO11 detections with track IDs and estimated distances  
**Bottom-right** — tracking panel with fading trails and linear motion extrapolation  

Optional semantic grounding can add open-vocabulary boxes from LocateAnything for prompts like `traffic cone`, `road sign text`, or `pedestrian near the curb`.

Decision banner shows: `NOMINAL` / `CAUTION` / `SLOW_DOWN` / `STOP` / `WAIT` / `PROCEED`

---

## Getting started

> I use Conda on macOS Apple Silicon — here's the setup that worked for me:

```bash
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera
conda create -n cav-perception python=3.11 -y
conda activate cav-perception
pip install -r requirements.txt
```

If you prefer a plain venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**macOS camera permission:** System Settings → Privacy & Security → Camera → enable Terminal (one-time).

---

## Running

```bash
python main.py --camera 0
```

Other flags:

| Flag | What it does |
|---|---|
| `--camera 1` | use a different camera index |
| `--conf 0.45` | stricter detection confidence |
| `--no-v2x` | skip V2X message output |
| `--no-log` | don't write CSV/JSONL logs |
| `--save-output` | auto-save every frame to `outputs/` |

Keys: **`q`** or **`ESC`** to quit, **`s`** to save a screenshot.  
Screenshots go to `outputs/cav_<timestamp>.jpg`.

---

## Open-vocabulary grounding

YOLO/ByteTrack stays as the real-time detector. The optional LocateAnything layer is for semantic queries that fixed COCO labels do not cover, such as construction objects, sign text, or attribute-based prompts.

LocateAnything is disabled by default because it is a large vision-language model. To enable it, install NVlabs/Eagle Embodied so `locateanything_worker` is importable, then update:

```yaml
# configs/config.yaml
locate_anything:
  enabled: true
  every_n_frames: 30
  queries:
    - "traffic cone"
    - "construction barrel"
    - "road sign text"
    - "pedestrian near the curb"
```

The app will keep running with YOLO only if LocateAnything is not installed or fails to load.

---

## Making the detector better

The project now has a full improvement loop: capture real frames, use YOLO plus optional LocateAnything to produce pseudo-labels, convert those labels into Ultralytics YOLO format, fine-tune, evaluate, then deploy the best weights back into `configs/config.yaml`.

1. Collect frames and detections:

```bash
python main.py --camera 0 --capture-frames
```

This writes raw frames to `datasets/raw_frames/` and detections to `logs/detections.jsonl`.

2. Build a YOLO dataset:

```bash
python tools/build_yolo_dataset.py \
  --frames datasets/raw_frames \
  --detections logs/detections.jsonl \
  --out datasets/cav_yolo \
  --min-conf 0.55
```

3. Fine-tune YOLO:

```bash
python tools/train_yolo.py \
  --data datasets/cav_yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --imgsz 640
```

4. Deploy the best model:

```yaml
# configs/config.yaml
model:
  name: "runs/cav/yolo_cav/weights/best.pt"
```

Use this loop repeatedly. Good real-world data from your camera usually improves the project more than swapping model names every week.

---

## Depth estimation

On first run it downloads **MiDaS_small** (~30 MB via `torch.hub`). If it loads successfully, the bottom-left panel shows a depth colormap instead of the edge/keypoint view. If the download fails or MiDaS errors out, the pipeline falls back to bounding-box-based distance estimation — nothing breaks, you just lose the colormap.

To skip MiDaS entirely:
```yaml
# configs/config.yaml
depth:
  enabled: false
```

---

## Project structure

```
├── main.py                      # entry point, main loop
├── configs/
│   └── config.yaml              # every tunable parameter lives here
├── src/
│   ├── camera.py                # USB capture wrapper
│   ├── detector.py              # YOLO11n + ByteTrack
│   ├── semantic_grounder.py     # optional LocateAnything open-vocabulary boxes
│   ├── tracker.py               # tracker wrapper
│   ├── depth.py                 # MiDaS depth + bbox fallback
│   ├── bev_mapper.py            # pseudo-3D bird's-eye-view
│   ├── dashboard.py             # 5-panel compositor
│   ├── decision_engine.py       # simple state machine (NOMINAL → STOP)
│   ├── v2x.py                   # simulated BSM / DOM / ISM messages
│   ├── logger.py                # CSV + JSONL writer
│   └── utils.py                 # FPS counter, config loader
├── tools/
│   ├── build_yolo_dataset.py    # logs + frames → YOLO dataset
│   └── train_yolo.py            # fine-tune/evaluate YOLO
├── logs/                        # detections.csv + detections.jsonl
└── outputs/                     # saved screenshots
```

---

## Config

Everything tunable is in `configs/config.yaml`. The main ones:

```yaml
model:
  confidence: 0.35           # lower = more detections, more false positives
  tracker: "bytetrack.yaml"

classes:
  ids: null                  # null = all 80 COCO classes

depth:
  enabled: true
  every_n_frames: 5          # only run MiDaS every 5 frames to save compute

bev:
  max_range_m: 20.0          # how far the world map shows
```

---

## Logs

Every detection is written to two files at the same time:
- `logs/detections.csv` — easy to open in Excel / pandas
- `logs/detections.jsonl` — one JSON object per line, good for downstream processing

Both are flushed cleanly when you quit.

---

## Ubuntu / Jetson

```bash
git clone https://github.com/mahin1-coder/CAV-camera.git
cd CAV-camera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --camera 0
```

Camera permission: `sudo usermod -aG video $USER` then re-login.

Jetson: install the NVIDIA PyTorch wheel for your JetPack version *before* running pip, so you get CUDA inference on the depth model.

---

## Honest caveats

- The BEV map is **not real SLAM**. It projects objects using estimated distance + horizontal frame position. It looks like SLAM but there's no loop closure, no pose graph — it's a visualisation tool.
- V2X messages (BSM, DOM, ISM) are printed to stdout for demo purposes. There's no actual DSRC or C-V2X radio involved.
- Distance numbers are rough. For anything real, calibrate your focal length with a checkerboard or swap in a stereo camera / LiDAR.
