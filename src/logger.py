"""
Detection logger — persists detections to a CSV file.

Design
------
* Rows are buffered in memory and flushed to disk every FLUSH_EVERY rows,
  or explicitly via flush() / close().  This keeps real-time I/O overhead low.
* Thread-safe: a threading.Lock guards all buffer mutations so the logger
  can be called from background threads without data races.
* Pandas is used for structured DataFrame construction before each CSV write,
  enabling straightforward post-processing with standard data-science tools.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import LOG_DIR, LOG_FILE
from src.detector import Detection


# Column order written to CSV
_CSV_COLUMNS = [
    "timestamp_utc",
    "frame_id",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width_px",
    "height_px",
    "cx",
    "cy",
    "estimated_distance_m",
]


class DetectionLogger:
    """
    Thread-safe CSV logger for object detections.

    Parameters
    ----------
    log_dir  : str  Directory in which to write the CSV file.
    log_file : str  CSV filename (created or appended inside *log_dir*).
    """

    FLUSH_EVERY: int = 30  # Auto-flush after this many buffered rows

    def __init__(
        self,
        log_dir: str  = LOG_DIR,
        log_file: str = LOG_FILE,
    ) -> None:
        self._path  = Path(log_dir) / log_file
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._buffer: list[dict] = []
        self._lock   = threading.Lock()
        # Write the header row only when creating a new file
        self._write_header = not self._path.exists()

        print(f"[Logger] Detection log → {self._path.resolve()}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, detections: list[Detection], frame_id: int) -> None:
        """
        Buffer detection rows for the given frame.

        If the internal buffer reaches FLUSH_EVERY rows the data is
        automatically flushed to disk.

        Parameters
        ----------
        detections : list[Detection]  Detections from the current frame.
        frame_id   : int              Monotonically increasing frame counter.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        rows = [
            {
                "timestamp_utc":          now,
                "frame_id":               frame_id,
                "track_id":               det.track_id,
                "class_id":               det.class_id,
                "class_name":             det.class_name,
                "confidence":             round(det.confidence, 4),
                "x1":                     det.x1,
                "y1":                     det.y1,
                "x2":                     det.x2,
                "y2":                     det.y2,
                "width_px":               det.width,
                "height_px":              det.height,
                "cx":                     det.cx,
                "cy":                     det.cy,
                "estimated_distance_m":   det.estimated_distance_m,
            }
            for det in detections
        ]

        with self._lock:
            self._buffer.extend(rows)
            if len(self._buffer) >= self.FLUSH_EVERY:
                self._flush_locked()

    def flush(self) -> None:
        """Force-flush all buffered rows to disk immediately."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining rows and report the final file path."""
        self.flush()
        print(f"[Logger] Session log saved → {self._path.resolve()}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Write buffered rows to CSV.  Must be called while _lock is held."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer, columns=_CSV_COLUMNS)
        df.to_csv(
            self._path,
            mode="a",
            header=self._write_header,
            index=False,
        )
        self._write_header = False   # Only write CSV header once
        self._buffer.clear()
