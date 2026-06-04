# Basler ArUco Detector UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Tkinter app that streams live video from a Basler camera, detects ArUco 6x6 markers with corner-point overlay, and lets the user control gain/exposure.

**Architecture:** Single `main.py` containing pure utility functions at the top (testable), a `CameraThread` class, and a `App(tk.Tk)` class. A background thread grabs frames, runs detection, and pushes annotated frames into a `queue.Queue(maxsize=1)`; the main thread polls via `after(30ms)` and updates a `Canvas`.

**Tech Stack:** Python 3.11+, pypylon, opencv-contrib-python, pillow, numpy, tkinter (stdlib), uv + pyproject.toml

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata + dependencies |
| `.python-version` | Pin Python version for uv |
| `.gitignore` | Ignore `.venv`, `__pycache__`, etc. |
| `main.py` | Full application (utilities + thread + UI + entry point) |
| `tests/test_detector.py` | Unit tests for `detect_and_annotate()` |
| `tests/test_validation.py` | Unit tests for `parse_camera_settings()` |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "aruco-detector"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-contrib-python>=4.9",
    "pypylon>=3.0",
    "pillow>=10.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create `.python-version`**

```
3.11
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
dist/
*.egg-info/
```

- [ ] **Step 4: Install dependencies**

```bash
uv sync --extra dev
```

Expected: uv creates `.venv/` and installs all packages. No errors.

- [ ] **Step 5: Verify opencv-contrib has aruco**

```bash
uv run python -c "import cv2; print(cv2.aruco.DICT_6X6_250)"
```

Expected output: `8` (the enum integer value).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore
git commit -m "feat: project scaffold with uv and pyproject.toml"
```

---

## Task 2: ArUco Detection Function (TDD)

**Files:**
- Create: `main.py` (start with utilities only)
- Create: `tests/__init__.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 2: Write failing tests in `tests/test_detector.py`**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_detector.py -v
```

Expected: `ImportError` — `main` does not exist yet.

- [ ] **Step 4: Create `main.py` with the detection utility**

```python
import queue
import threading
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from pypylon import pylon

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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_detector.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/__init__.py tests/test_detector.py
git commit -m "feat: ArUco detection utility with tests"
```

---

## Task 3: Camera Settings Validation (TDD)

**Files:**
- Modify: `main.py` — append `parse_camera_settings()` function
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing tests in `tests/test_validation.py`**

```python
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import parse_camera_settings


def test_valid_values():
    gain, exposure = parse_camera_settings("2.0", "5000")
    assert gain == 2.0
    assert exposure == 5000.0


def test_integer_strings_accepted():
    gain, exposure = parse_camera_settings("3", "10000")
    assert gain == 3.0
    assert exposure == 10000.0


def test_non_numeric_gain_raises():
    with pytest.raises(ValueError, match="gain"):
        parse_camera_settings("abc", "5000")


def test_non_numeric_exposure_raises():
    with pytest.raises(ValueError, match="exposure"):
        parse_camera_settings("1.0", "xyz")


def test_negative_values_raise():
    with pytest.raises(ValueError):
        parse_camera_settings("-1", "5000")
    with pytest.raises(ValueError):
        parse_camera_settings("1.0", "-100")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_validation.py -v
```

Expected: `ImportError` — `parse_camera_settings` not defined yet.

- [ ] **Step 3: Add `parse_camera_settings` to `main.py`**

Append after `detect_and_annotate`:

```python
def parse_camera_settings(gain_str: str, exposure_str: str) -> tuple[float, float]:
    """Parse and validate gain and exposure strings. Raises ValueError on bad input."""
    try:
        gain = float(gain_str)
    except ValueError:
        raise ValueError(f"gain must be numeric, got '{gain_str}'")
    try:
        exposure = float(exposure_str)
    except ValueError:
        raise ValueError(f"exposure must be numeric, got '{exposure_str}'")
    if gain < 0:
        raise ValueError(f"gain must be >= 0, got {gain}")
    if exposure < 0:
        raise ValueError(f"exposure must be >= 0, got {exposure}")
    return gain, exposure
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_validation.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_validation.py
git commit -m "feat: camera settings validation with tests"
```

---

## Task 4: Camera Grab Thread

**Files:**
- Modify: `main.py` — append `CameraThread` class

- [ ] **Step 1: Append `CameraThread` to `main.py`**

```python
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
```

- [ ] **Step 2: Run all existing tests to confirm nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: 8 tests PASSED.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: camera grab thread with pypylon"
```

---

## Task 5: Tkinter App Class

**Files:**
- Modify: `main.py` — append `App` class

- [ ] **Step 1: Append `App` class to `main.py`**

```python
# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------
class App(tk.Tk):
    _POLL_MS = 30

    def __init__(self):
        super().__init__()
        self.title("ArUco Detector")
        self.geometry("860x680")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._camera_thread: CameraThread | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._poll_frames()

    # ---- Layout ----

    def _build_ui(self) -> None:
        # Top bar
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=6, pady=4)

        self._btn_connect = tk.Button(top, text="Connect", command=self._connect)
        self._btn_connect.pack(side=tk.LEFT, padx=2)

        self._btn_disconnect = tk.Button(
            top, text="Disconnect", state=tk.DISABLED, command=self._disconnect
        )
        self._btn_disconnect.pack(side=tk.LEFT, padx=2)

        self._status_var = tk.StringVar(value="Status: Idle")
        tk.Label(top, textvariable=self._status_var, anchor=tk.W).pack(
            side=tk.LEFT, padx=10
        )

        # Canvas (live feed)
        self._canvas = tk.Canvas(self, bg="black")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        # Bottom settings bar
        bottom = tk.Frame(self)
        bottom.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(bottom, text="Gain:").pack(side=tk.LEFT)
        self._gain_var = tk.StringVar(value="1.0")
        tk.Entry(bottom, textvariable=self._gain_var, width=8).pack(
            side=tk.LEFT, padx=2
        )

        tk.Label(bottom, text="Exposure (µs):").pack(side=tk.LEFT, padx=(12, 0))
        self._exposure_var = tk.StringVar(value="10000")
        tk.Entry(bottom, textvariable=self._exposure_var, width=10).pack(
            side=tk.LEFT, padx=2
        )

        tk.Button(bottom, text="Apply Settings", command=self._apply_settings).pack(
            side=tk.LEFT, padx=12
        )

    # ---- Camera control ----

    def _connect(self) -> None:
        self._frame_queue = queue.Queue(maxsize=1)
        self._camera_thread = CameraThread(self._frame_queue)
        self._camera_thread.start()
        # Allow up to 1 s for the thread to either confirm running or report error
        self._camera_thread.join(timeout=1.0)
        if not self._camera_thread.is_alive():
            err = self._camera_thread.error or "camera failed to start"
            self._status_var.set(f"Error: {err}")
            self._camera_thread = None
            return
        self._btn_connect.config(state=tk.DISABLED)
        self._btn_disconnect.config(state=tk.NORMAL)
        self._status_var.set("Status: Connected")

    def _disconnect(self) -> None:
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread.join(timeout=3.0)
            self._camera_thread = None
        self._btn_connect.config(state=tk.NORMAL)
        self._btn_disconnect.config(state=tk.DISABLED)
        self._status_var.set("Status: Idle")

    def _apply_settings(self) -> None:
        if self._camera_thread is None or self._camera_thread.camera is None:
            self._status_var.set("Error: camera not connected")
            return
        try:
            gain, exposure = parse_camera_settings(
                self._gain_var.get().strip(), self._exposure_var.get().strip()
            )
        except ValueError as exc:
            self._status_var.set(f"Error: {exc}")
            return
        cam = self._camera_thread.camera
        gain = max(cam.Gain.Min, min(cam.Gain.Max, gain))
        exposure = max(cam.ExposureTime.Min, min(cam.ExposureTime.Max, exposure))
        cam.Gain.Value = gain
        cam.ExposureTime.Value = exposure
        self._status_var.set(f"Applied: gain={gain:.2f}  exposure={exposure:.0f} µs")

    # ---- Frame polling ----

    def _poll_frames(self) -> None:
        # Check if thread died with an error
        if (
            self._camera_thread is not None
            and not self._camera_thread.is_alive()
            and self._camera_thread.error
        ):
            self._status_var.set(f"Error: {self._camera_thread.error}")
            self._camera_thread = None
            self._btn_connect.config(state=tk.NORMAL)
            self._btn_disconnect.config(state=tk.DISABLED)

        try:
            frame = self._frame_queue.get_nowait()
            h, w = frame.shape[:2]
            cw = self._canvas.winfo_width() or 840
            ch = self._canvas.winfo_height() or 540
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((nw, nh), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)
        except queue.Empty:
            pass

        self.after(self._POLL_MS, self._poll_frames)

    def _on_close(self) -> None:
        self._disconnect()
        self.destroy()
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: 8 tests PASSED.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: Tkinter App class with live canvas and settings panel"
```

---

## Task 6: Entry Point

**Files:**
- Modify: `main.py` — append entry point block

- [ ] **Step 1: Append entry point to `main.py`**

```python
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
```

- [ ] **Step 2: Run all tests one final time**

```bash
uv run pytest tests/ -v
```

Expected: 8 tests PASSED.

- [ ] **Step 3: Smoke-test the app launches (no camera required)**

```bash
uv run python main.py
```

Expected: Window opens with "Status: Idle". Clicking Connect without a camera should show an error in the status bar, not crash. Close the window — no errors.

- [ ] **Step 4: Final commit**

```bash
git add main.py
git commit -m "feat: entry point — app is fully runnable"
```
