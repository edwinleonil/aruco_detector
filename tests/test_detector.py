import numpy as np
import cv2
from main import detect_and_annotate


def test_returns_ndarray_same_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_and_annotate(frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (480, 640, 3)


def test_no_markers_frame_unchanged_shape():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    original = frame.copy()
    result = detect_and_annotate(frame)
    assert result.shape == frame.shape
    assert np.array_equal(result, original), "pixels changed on a frame with no markers"


def test_detects_synthetic_marker():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
    # White background provides the required quiet zone for detection
    frame = np.full((400, 400, 3), 255, dtype=np.uint8)
    frame[100:300, 100:300] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    original = frame.copy()
    result = detect_and_annotate(frame)
    assert not np.array_equal(result, original), "no green dots were drawn — detection may have failed"
    green_mask = (result[:, :, 0] == 0) & (result[:, :, 1] == 255) & (result[:, :, 2] == 0)
    assert green_mask.any(), "no pure-green corner dots found in output"
