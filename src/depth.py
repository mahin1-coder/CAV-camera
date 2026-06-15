"""
Depth estimation module for the CAV perception pipeline.

Tries to load MiDaS_small via torch.hub (requires internet on first run).
If loading fails for any reason, falls back silently to bounding-box-based
distance estimation (already present in Detection objects).

The public API is always available — callers check ``estimator.available``
to decide whether to use the depth map or skip it.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Known real-world widths (metres) used for the bbox-distance fallback.
_KNOWN_WIDTHS_M: dict[str, float] = {
    "person":        0.50,
    "bicycle":       0.60,
    "car":           1.80,
    "motorcycle":    0.70,
    "bus":           2.50,
    "truck":         2.50,
    "traffic light": 0.30,
    "stop sign":     0.60,
}


class DepthEstimator:
    """
    Wraps MiDaS_small for per-frame monocular depth estimation.

    Falls back gracefully to bbox-height-based distance when the model
    cannot be loaded (no GPU, no internet, torch not installed, etc.).

    Parameters
    ----------
    cfg : dict
        The ``depth`` section of the pipeline config.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.available: bool = False
        self._model           = None
        self._transform       = None
        self._device: str     = "cpu"
        self._every_n: int    = cfg.get("every_n_frames", 5)
        self._frame_count     = 0

        if not cfg.get("enabled", True):
            print("[Depth] Disabled by config.")
            return

        try:
            import torch  # noqa: PLC0415

            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            print("[Depth] Loading MiDaS_small …  (downloads ~30 MB on first run)")
            model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                trust_repo=True,
                verbose=False,
            )
            transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
                verbose=False,
            )
            model.to(device).eval()

            self._model     = model
            self._transform = transforms.small_transform
            self._device    = device
            self.available  = True
            print(f"[Depth] MiDaS_small ready on {device}.")

        except Exception as exc:  # noqa: BLE001
            print(f"[Depth] MiDaS unavailable ({exc}). Bbox-distance fallback active.")

    # ── Public API ────────────────────────────────────────────────────────────

    def should_run(self) -> bool:
        """Return True on the frames when depth should be estimated."""
        self._frame_count += 1
        return self.available and (self._frame_count % self._every_n == 0)

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Run MiDaS on *frame* and return a uint8 depth map (0 = far, 255 = near).

        Returns ``None`` if the model is unavailable or inference fails.
        """
        if not self.available:
            return None
        try:
            import torch  # noqa: PLC0415

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self._device)

            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

            mn, mx = pred.min(), pred.max()
            if mx > mn:
                depth_u8 = ((pred - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                depth_u8 = np.zeros(frame.shape[:2], dtype=np.uint8)
            return depth_u8

        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def colorize(depth_u8: np.ndarray) -> np.ndarray:
        """Apply INFERNO colormap to a grayscale depth map → BGR image."""
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

    @staticmethod
    def bbox_distance(
        class_name: str,
        bbox_width_px: int,
        focal_px: float = 700.0,
    ) -> float | None:
        """
        Monocular distance estimate from bounding-box width.

            d = (known_width_m × focal_px) / bbox_width_px

        Returns ``None`` when the class is unknown or the bbox is degenerate.
        """
        w = _KNOWN_WIDTHS_M.get(class_name)
        if w is None or bbox_width_px <= 0:
            return None
        return round((w * focal_px) / bbox_width_px, 2)
