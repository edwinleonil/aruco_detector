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
    """Detect DICT_6X6 ArUco markers and draw green corner dots in-place."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, _, _ = _aruco_detector.detectMarkers(gray)
    for marker_corners in corners:
        for pt in marker_corners[0].astype(int):
            cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
    return frame
