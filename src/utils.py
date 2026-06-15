"""
Utility helpers for the CAV perception pipeline.

Provides
--------
FPSCounter      — rolling-average FPS tracker
save_frame()    — write a BGR frame to the outputs directory
draw_fps()      — render FPS overlay onto a frame
draw_decision() — render decision-engine action and alerts onto a frame
draw_info_bar() — render a small status line at the bottom of the frame
load_config()   — load and validate configs/config.yaml
merge_cli()     — apply CLI overrides on top of loaded config
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


# ── FPS counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Rolling-average FPS counter.

    Parameters
    ----------
    window : int
        Number of recent frame durations to average over.
    """

    def __init__(self, window: int = 30) -> None:
        self._times: collections.deque[float] = collections.deque(maxlen=window)
        self._last  = time.perf_counter()

    def tick(self) -> float:
        """Record one frame and return the current FPS estimate."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def fps(self) -> float:
        """Most recent FPS without updating the counter."""
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ── Frame saving ──────────────────────────────────────────────────────────────

def save_frame(
    frame: np.ndarray,
    output_dir: str,
    prefix: str = "frame",
) -> Path:
    """
    Save *frame* as a JPEG to *output_dir*.

    The filename is ``<prefix>_<timestamp_ms>.jpg``.

    Returns the path of the saved file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts        = int(time.time() * 1000)
    file_path = out_path / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(file_path), frame)
    return file_path


# ── On-screen overlay helpers ─────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_ACTION_COLORS: dict[str, tuple[int, int, int]] = {
    "NOMINAL":   (0, 200,   0),
    "CAUTION":   (0, 200, 255),
    "SLOW_DOWN": (0, 165, 255),
    "WAIT":      (0,   0, 220),
    "STOP":      (0,   0, 220),
    "PROCEED":   (0, 220,   0),
}


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render the FPS counter in the top-left corner of *frame*."""
    cv2.putText(
        frame, f"FPS: {fps:5.1f}",
        (10, 30), _FONT, 0.70, (0, 255, 0), 2, cv2.LINE_AA,
    )


def draw_decision(
    frame: np.ndarray,
    action: str,
    alerts: list[str],
) -> None:
    """
    Render the decision-engine output onto *frame*.

    * Action label — top-right corner, colour-coded by severity.
    * Alert lines  — bottom-left, up to 5 most recent.
    """
    h, w  = frame.shape[:2]
    color = _ACTION_COLORS.get(action, (200, 200, 200))

    label = f"Action: {action}"
    (lw, _), _ = cv2.getTextSize(label, _FONT, 0.65, 2)
    cv2.putText(frame, label, (w - lw - 10, 30), _FONT, 0.65, color, 2, cv2.LINE_AA)

    for i, alert in enumerate(alerts[:5]):
        cv2.putText(
            frame, f"! {alert[:72]}",
            (10, h - 15 - i * 22),
            _FONT, 0.44, (0, 200, 255), 1, cv2.LINE_AA,
        )


def draw_info_bar(frame: np.ndarray, text: str) -> None:
    """Render a small grey status line at the very bottom of *frame*."""
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 5), _FONT, 0.38, (160, 160, 160), 1, cv2.LINE_AA)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the config file, relative to the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path.resolve()}\n"
            "Make sure you run  python main.py  from the project root directory."
        )
    with cfg_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def merge_cli(
    cfg: dict[str, Any],
    camera: int | None,
    model:  str | None,
    conf:   float | None,
) -> dict[str, Any]:
    """
    Apply CLI argument overrides on top of a loaded config dict.

    Only non-None values are applied; omitted CLI flags leave the
    config value untouched.
    """
    if camera is not None:
        cfg["camera"]["index"] = camera
    if model is not None:
        cfg["model"]["name"] = model
    if conf is not None:
        cfg["model"]["confidence"] = conf
    return cfg
