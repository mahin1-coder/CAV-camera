"""
Configuration settings for the CAV Camera Perception Pipeline.

Modify these values to adapt the pipeline to different hardware setups
without touching any other source files.
"""

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0      # USB camera device index (0 = default / built-in)
FRAME_WIDTH   = 1280   # Preferred capture width  (camera may cap lower)
FRAME_HEIGHT  = 720    # Preferred capture height
TARGET_FPS    = 30     # Preferred capture frame-rate

# ── YOLO Model ────────────────────────────────────────────────────────────────
MODEL_NAME             = "yolo11n.pt"  # Auto-downloaded from Ultralytics on first run
CONFIDENCE_THRESHOLD   = 0.45          # Minimum detection confidence [0–1]
IOU_THRESHOLD          = 0.45          # NMS IoU threshold [0–1]

# ── Classes of Interest (COCO class IDs used by YOLO) ─────────────────────────
# Only these classes will be forwarded to downstream modules.
CLASSES_OF_INTEREST: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
}

# ── Bounding-box colour palette (BGR) ─────────────────────────────────────────
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":        (0,   255,   0),    # Green
    "bicycle":       (0,   165, 255),    # Orange
    "car":           (0,     0, 255),    # Red
    "motorcycle":    (255,   0, 255),    # Magenta
    "bus":           (255, 165,   0),    # Blue-orange
    "truck":         (0,   255, 255),    # Yellow
    "traffic light": (255, 255,   0),    # Cyan
    "stop sign":     (0,     0, 200),    # Dark red
}
DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)  # Grey fallback

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR           = "logs"
LOG_FILE          = "detections.csv"
LOG_EVERY_N_FRAMES = 1   # 1 = log every frame; increase to reduce I/O load

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_WINDOW_NAME = "CAV Perception Pipeline"
FONT_SCALE          = 0.55
FONT_THICKNESS      = 2
BOX_THICKNESS       = 2
FPS_POSITION        = (10, 30)  # Pixel coordinates (x, y) for FPS overlay

# ── Distance Estimation (monocular placeholder) ───────────────────────────────
# ⚠️  These are rough approximations.
#     Calibrate FOCAL_LENGTH_PX with a checkerboard pattern for real use.
#     For production accuracy use stereo vision or a depth sensor.
FOCAL_LENGTH_PX = 700.0   # Estimated focal length in pixels

KNOWN_OBJECT_WIDTHS_M: dict[str, float] = {
    "person":        0.50,
    "bicycle":       0.60,
    "car":           1.80,
    "motorcycle":    0.80,
    "bus":           2.50,
    "truck":         2.50,
    "traffic light": 0.30,
    "stop sign":     0.75,
}

# ── V2X Simulation ────────────────────────────────────────────────────────────
V2X_ENABLED               = True
V2X_BROADCAST_INTERVAL_S  = 1.0        # Seconds between simulated broadcasts
V2X_EGO_VEHICLE_ID        = "EGO_001"  # Identifier for the ego vehicle
