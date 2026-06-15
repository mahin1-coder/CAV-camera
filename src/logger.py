"""
Detection logger — persists detections to CSV and JSON Lines.

Design (v2 upgrade)
-------------------
* Dual output: CSV (Pandas) + JSON Lines (.jsonl) — one JSON object per line.
* Rows buffered in memory, auto-flushed every FLUSH_EVERY rows or on close().
* Thread-safe via threading.Lock.
* Config-dict driven — no imports from src/config.py.
* JSON Lines format is ideal for streaming log analysis and ML pipelines.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.detector import Detection


# Column order written to CSV
_CSV_COLUMNS = [
    "timestamp_utc",
    "frame_id",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1", "y1", "x2", "y2",
    "width_px", "height_px",
    "cx", "cy",
    "estimated_distance_m",
]


class DetectionLogger:
    """
    Thread-safe dual-format logger (CSV + JSON Lines) for object detections.

    Parameters
    ----------
    cfg : dict
        The ``logging`` section of the pipeline config, e.g.::

            {
                "enabled": true,
                "log_dir": "logs",
                "csv_file": "detections.csv",
                "json_file": "detections.jsonl",
                "log_every_n_frames": 1
            }
    """

    FLUSH_EVERY: int = 30  # Auto-flush after this many buffered rows

    def __init__(self, cfg: dict[str, Any]) -> None:
        log_dir  = cfg.get("log_dir",  "logs")
        csv_file = cfg.get("csv_file", "detections.csv")
        json_file= cfg.get("json_file","detections.jsonl")

        self._csv_path  = Path(log_dir) / csv_file
        self._json_path = Path(log_dir) / json_file
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._buffer: list[dict] = []
        self._lock   = threading.Lock()
        self._csv_header = not self._csv_path.exists()

        # Open JSON Lines file in append mode
        self._json_fh = self._json_path.open("a", buffering=1)  # line-buffered

        print(f"[Logger] CSV  → {self._csv_path.resolve()}")
        print(f"[Logger] JSONL→ {self._json_path.resolve()}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, detections: list[Detection], frame_id: int) -> None:
        """
        Buffer detection rows for *frame_id*.

        Auto-flushes to disk when the buffer reaches FLUSH_EVERY rows.
        Each detection is also immediately written as a JSON Line.

        Parameters
        ----------
        detections : list[Detection]
        frame_id   : int
        """
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        rows: list[dict] = []
        for det in detections:
            row: dict[str, Any] = {
                "timestamp_utc":        now,
                "frame_id":             frame_id,
                "track_id":             det.track_id,
                "class_id":             det.class_id,
                "class_name":           det.class_name,
                "confidence":           round(det.confidence, 4),
                "x1": det.x1, "y1": det.y1,
                "x2": det.x2, "y2": det.y2,
                "width_px":             det.width,
                "height_px":            det.height,
                "cx":                   det.cx,
                "cy":                   det.cy,
                "estimated_distance_m": det.estimated_distance_m,
            }
            rows.append(row)
            # Write to JSON Lines immediately (line-buffered, no extra lock needed
            # because GIL protects single write calls on CPython)
            self._json_fh.write(json.dumps(row, separators=(",", ":")) + "\n")

        with self._lock:
            self._buffer.extend(rows)
            if len(self._buffer) >= self.FLUSH_EVERY:
                self._flush_locked()

    def flush(self) -> None:
        """Force-flush all buffered CSV rows to disk immediately."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining rows, close files, and report paths."""
        self.flush()
        self._json_fh.flush()
        self._json_fh.close()
        print(f"[Logger] CSV  saved → {self._csv_path.resolve()}")
        print(f"[Logger] JSONL saved → {self._json_path.resolve()}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Write buffered rows to CSV.  Must be called while _lock is held."""
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer, columns=_CSV_COLUMNS)
        df.to_csv(self._csv_path, mode="a", header=self._csv_header, index=False)
        self._csv_header = False
        self._buffer.clear()
