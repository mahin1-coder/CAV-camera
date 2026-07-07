"""
nuScenes dataset loader — drop-in replacement for Camera.

Iterates through a nuScenes scene and returns frames from one camera channel
(default: CAM_FRONT) one by one via the same read() interface used by Camera,
so the rest of the pipeline needs no changes.

Usage
-----
    loader = NuScenesLoader(cfg)
    loader.open()
    while True:
        ret, frame = loader.read()
        if not ret:
            break          # end of scene
    loader.release()

Config keys (under ``nuscenes`` in config.yaml)
------------------------------------------------
    dataroot  : str   path to the nuScenes dataset root
    version   : str   "v1.0-mini" | "v1.0-trainval" | "v1.0-test"
    scene_idx : int   which scene to play (0-based index into nusc.scene list)
    camera    : str   sensor channel, e.g. "CAM_FRONT", "CAM_FRONT_LEFT", etc.
    loop      : bool  restart from the beginning when the scene ends (default False)
    fps       : float playback target (purely informational — no sleep)
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


_VALID_CAMERAS = {
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
}


class NuScenesLoader:
    """
    Iterates through nuScenes samples and returns BGR frames via read().

    Parameters
    ----------
    cfg : dict
        The ``nuscenes`` section of the pipeline config.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._dataroot  = cfg.get("dataroot", "")
        self._version   = cfg.get("version",   "v1.0-mini")
        self._scene_idx = int(cfg.get("scene_idx", 0))
        self._channel   = cfg.get("camera",    "CAM_FRONT")
        self._loop      = bool(cfg.get("loop", False))

        if self._channel not in _VALID_CAMERAS:
            print(f"[nuScenes] Unknown camera '{self._channel}', falling back to CAM_FRONT.")
            self._channel = "CAM_FRONT"

        self._nusc       = None
        self._tokens: list[str] = []   # ordered sample tokens for the scene
        self._idx        = 0
        self.width       = 1600
        self.height      = 900
        self.fps         = float(cfg.get("fps", 12.0))

    # ── Public interface (matches Camera) ─────────────────────────────────────

    def open(self) -> None:
        """Load the nuScenes metadata and build the sample token list."""
        if not self._dataroot or not os.path.isdir(self._dataroot):
            raise RuntimeError(
                f"[nuScenes] dataroot not found: '{self._dataroot}'\n"
                f"  → Download nuScenes mini from https://www.nuscenes.org/nuscenes\n"
                f"  → Then set nuscenes.dataroot in configs/config.yaml"
            )

        try:
            from nuscenes.nuscenes import NuScenes  # lazy import
        except ImportError:
            raise RuntimeError(
                "[nuScenes] nuscenes-devkit not installed.\n"
                "  → Run: pip install nuscenes-devkit"
            )

        print(f"[nuScenes] Loading {self._version} from {self._dataroot} …")
        self._nusc = NuScenes(
            version  = self._version,
            dataroot = self._dataroot,
            verbose  = False,
        )

        if self._scene_idx >= len(self._nusc.scene):
            raise RuntimeError(
                f"[nuScenes] scene_idx={self._scene_idx} out of range "
                f"(dataset has {len(self._nusc.scene)} scenes)."
            )

        scene = self._nusc.scene[self._scene_idx]
        print(
            f"[nuScenes] Scene {self._scene_idx}: '{scene['name']}' "
            f"| {scene['nbr_samples']} samples | channel={self._channel}"
        )

        # Build ordered list of sample tokens
        token = scene["first_sample_token"]
        while token:
            self._tokens.append(token)
            sample = self._nusc.get("sample", token)
            token  = sample["next"]

        self._idx = 0

        # Read one frame to get actual image dimensions
        frame = self._load_frame(self._tokens[0])
        if frame is not None:
            self.height, self.width = frame.shape[:2]

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Return the next frame as (True, bgr_array).
        Returns (False, None) when the scene is exhausted (and loop=False).
        """
        if self._nusc is None:
            return False, None

        if self._idx >= len(self._tokens):
            if self._loop:
                self._idx = 0
            else:
                return False, None

        frame = self._load_frame(self._tokens[self._idx])
        self._idx += 1

        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        print(f"[nuScenes] Loader released. ({self._idx} / {len(self._tokens)} frames played)")
        self._nusc   = None
        self._tokens = []
        self._idx    = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def scene_name(self) -> str:
        if self._nusc and self._tokens:
            return self._nusc.scene[self._scene_idx]["name"]
        return "unknown"

    @property
    def total_frames(self) -> int:
        return len(self._tokens)

    @property
    def current_frame(self) -> int:
        return self._idx

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_frame(self, sample_token: str) -> np.ndarray | None:
        """Load the camera image for a given sample token."""
        sample    = self._nusc.get("sample", sample_token)
        cam_token = sample["data"].get(self._channel)
        if cam_token is None:
            print(f"[nuScenes] Channel {self._channel} not found in sample.")
            return None

        cam_data  = self._nusc.get("sample_data", cam_token)
        img_path  = os.path.join(self._dataroot, cam_data["filename"])

        if not os.path.isfile(img_path):
            print(f"[nuScenes] Image not found: {img_path}")
            return None

        frame = cv2.imread(img_path)
        return frame
