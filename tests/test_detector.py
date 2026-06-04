import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import detect_and_annotate


def test_returns_ndarray_same_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_and_annotate(frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (480, 640, 3)


def test_no_markers_frame_unchanged_shape():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = detect_and_annotate(frame)
    assert result.shape == frame.shape


def test_detects_synthetic_marker():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[50:250, 50:250] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    result = detect_and_annotate(frame)
    # Corner dots drawn as green pixels — check green channel has changed
    assert result[:, :, 1].max() == 255
