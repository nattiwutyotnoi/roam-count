"""Camera source abstraction (Phase 0 foundation).

Source is chosen from config only -- no code change needed to switch between:
  * webcam index   : 0, 1, ...            (also accepts the string "0")
  * RTSP / HTTP    : "rtsp://phone-ip/..."  (phone streaming)
  * video file     : "clips/walk_01.mp4"

Kept isolated from the main loop so the mobile port (CameraX / AVFoundation)
can replace this one class 1:1 without touching detection/counting logic
(portability guardrail, plan section 2.3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Streaming URL schemes that must NOT be treated as local files.
_STREAM_SCHEMES = ("rtsp://", "http://", "https://", "udp://", "tcp://")


class CameraSource:
    def __init__(self, source, width=None, height=None, fps_request=None):
        self.source = self._normalize(source)
        self.width = width
        self.height = height
        self.fps_request = fps_request
        self.cap: cv2.VideoCapture | None = None

    @staticmethod
    def _normalize(source):
        """"0" -> 0 (webcam index); URLs and file paths stay as str."""
        if isinstance(source, int):
            return source
        s = str(source).strip()
        return int(s) if s.isdigit() else s

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and not self.source.lower().startswith(_STREAM_SCHEMES)

    def open(self) -> "CameraSource":
        # Fail fast with a clear message when a file path simply doesn't exist,
        # instead of the generic "cannot open camera source".
        if self.is_file and not Path(self.source).exists():
            raise FileNotFoundError(
                f"Video file not found: {self.source!r}. "
                "Pass an existing file to --source, or omit --source to use the "
                "webcam configured in config.json."
            )

        # On Windows the default MSMF backend is slow to open a webcam and often
        # ignores resolution requests; DirectShow is far more reliable there.
        if self.is_webcam and sys.platform == "win32":
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.is_webcam:
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps_request:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_request)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source!r}")
        return self

    def reconnect(self) -> bool:
        """Release and reopen the source. Returns True on success, False on
        failure (does not raise) -- for auto-reconnect when a live camera drops."""
        self.release()
        try:
            self.open()
            return True
        except Exception:
            self.release()
            return False

    def read(self):
        """Return the next BGR frame, or None on end-of-stream / read failure."""
        if self.cap is None:
            raise RuntimeError("CameraSource.read() called before open()")
        ok, frame = self.cap.read()
        return frame if ok else None

    @property
    def info(self) -> dict:
        assert self.cap is not None
        return {
            "source": self.source,
            "kind": "webcam" if self.is_webcam else ("file" if self.is_file else "stream"),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(self.cap.get(cv2.CAP_PROP_FPS), 2),
        }

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
