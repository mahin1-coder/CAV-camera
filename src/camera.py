"""
Camera module — USB camera capture via OpenCV.

Handles device initialisation, frame reading, and graceful shutdown.
Prints actionable troubleshooting steps when the camera cannot be opened.
"""

from __future__ import annotations

import cv2

from src.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class Camera:
    """Manages a USB camera using cv2.VideoCapture."""

    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self.index = index
        self._cap: cv2.VideoCapture | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the camera device.

        Raises:
            RuntimeError: If the device cannot be opened.  A detailed
                          troubleshooting guide is printed before raising.
        """
        self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            self._print_troubleshooting()
            raise RuntimeError(
                f"Could not open camera at index {self.index}. "
                "See the troubleshooting guide printed above."
            )

        # Request preferred resolution and frame-rate.
        # The camera driver may silently cap these to hardware limits.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        print(
            f"[Camera] Opened  index={self.index} | "
            f"resolution={actual_w}x{actual_h} | "
            f"fps_cap={actual_fps:.1f}"
        )

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """
        Capture the next frame.

        Returns:
            (True, frame)   on success.
            (False, None)   if the camera is closed or capture fails.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[Camera] Released.")

    @property
    def is_opened(self) -> bool:
        """True when the camera device is open and ready."""
        return self._cap is not None and self._cap.isOpened()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _print_troubleshooting(self) -> None:
        sep = "=" * 62
        print(f"\n{sep}")
        print("  CAMERA TROUBLESHOOTING")
        print(sep)
        print(f"  Failed to open camera at index {self.index}.\n")
        print("  1. Confirm the USB camera is physically connected.")
        print("  2. Try a different camera index:")
        print("       python main.py --camera 1")
        print("  3. macOS — grant camera permission:")
        print("       System Settings → Privacy & Security → Camera")
        print("  4. List available indices (run in Python):")
        print("       python -c \"import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]\"")
        print("  5. Ubuntu / Jetson — check device nodes:")
        print("       ls /dev/video*")
        print("       sudo usermod -aG video $USER   # then log out and back in")
        print("  6. Make sure no other app (Zoom, OBS, etc.) is using the camera.")
        print(f"{sep}\n")
