import math
import queue
import threading
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Handle pypylon import gracefully for test environments
try:
    from pypylon import pylon
except ImportError:
    pylon = None

# ---------------------------------------------------------------------------
# ArUco setup (module-level, created once)
# ---------------------------------------------------------------------------
_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
_aruco_detector = cv2.aruco.ArucoDetector(
    _aruco_dict, cv2.aruco.DetectorParameters()
)


def detect_and_annotate(frame: np.ndarray) -> np.ndarray:
    """Detect DICT_6X6 ArUco markers and draw green corner dots on a copy of the frame."""
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, _, _ = _aruco_detector.detectMarkers(gray)
    for marker_corners in corners:
        for pt in marker_corners[0].astype(int):
            cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
    return frame


def parse_camera_settings(gain_str: str, exposure_str: str) -> tuple[float, float]:
    """Parse and validate gain and exposure strings. Raises ValueError on bad input."""
    try:
        gain = float(gain_str)
    except ValueError:
        raise ValueError(f"gain must be numeric, got '{gain_str}'")
    if not math.isfinite(gain):
        raise ValueError(f"gain must be a finite number, got '{gain_str}'")
    if gain < 0:
        raise ValueError(f"gain must be >= 0, got {gain}")
    try:
        exposure = float(exposure_str)
    except ValueError:
        raise ValueError(f"exposure must be numeric, got '{exposure_str}'")
    if not math.isfinite(exposure):
        raise ValueError(f"exposure must be a finite number, got '{exposure_str}'")
    if exposure < 0:
        raise ValueError(f"exposure must be >= 0, got {exposure}")
    return gain, exposure


# ---------------------------------------------------------------------------
# Camera grab thread
# ---------------------------------------------------------------------------
class CameraThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        super().__init__(daemon=True)
        self._queue = frame_queue
        self._stop_event = threading.Event()
        self.camera: pylon.InstantCamera | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            while not self._stop_event.is_set() and self.camera.IsGrabbing():
                grab = self.camera.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )
                if grab.GrabSucceeded():
                    img = converter.Convert(grab)
                    frame = img.GetArray().copy()
                    annotated = detect_and_annotate(frame)
                    try:
                        self._queue.put_nowait(annotated)
                    except queue.Full:
                        pass
                grab.Release()
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.camera is not None:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()

    def stop(self) -> None:
        self._stop_event.set()
