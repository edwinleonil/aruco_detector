# Basler ArUco Detector UI — Design Spec
Date: 2026-06-04

## Overview

A single-file Tkinter desktop application (`main.py`) that streams live video from a Basler camera, detects ArUco 6x6 markers, and overlays corner points on the feed. The user can control gain and exposure via numeric entry fields and manually connect/disconnect the camera.

---

## Architecture

**Single file: `main.py`**

Three logical sections within the file:

1. **Camera + detection layer** — pypylon grab loop and ArUco detection, runs in a background thread
2. **UI layer** — Tkinter `App` class, manages layout, canvas updates, and user input
3. **Entry point** — `if __name__ == "__main__"` block that launches the app

No external config files. No submodules. Dependencies managed via `pyproject.toml` and `uv`.

---

## Layout

```
+-----------------------------------------------+
|  [Connect]  [Disconnect]   Status: Idle        |
+-----------------------------------------------+
|                                               |
|           Live Video Feed (Canvas)            |
|       (ArUco corner dots drawn on frame)      |
|                                               |
+-----------------------------------------------+
|  Gain:  [_____]   Exposure (µs):  [________]  |
|                     [Apply Settings]           |
+-----------------------------------------------+
```

- **Top bar**: Connect / Disconnect buttons + status label (e.g. "Connected", "Idle", "Error: no camera found")
- **Canvas**: fills available space, frame is resized to fit canvas dimensions while preserving aspect ratio
- **Bottom panel**: `Entry` for gain, `Entry` for exposure in microseconds, one `Apply Settings` button that writes both values to the camera simultaneously

---

## Threading & Data Flow

```
Background thread                    Main thread (Tkinter)
─────────────────                    ─────────────────────
camera.RetrieveResult()
→ convert to numpy array
→ run ArUco detection
→ draw corner dots
→ queue.put(frame)          ──→      after(30ms) polls queue
                                     → PhotoImage update
                                     → canvas.create_image()
```

- Background thread: started on Connect, stopped on Disconnect via a `threading.Event` stop flag
- Queue: `queue.Queue(maxsize=1)` — old frames are dropped if the UI hasn't consumed them yet (non-blocking put with `put_nowait`, catching `queue.Full`)
- Main thread only: all Tkinter operations, camera settings writes
- Apply Settings: validates entry values are numeric and in range before writing to `camera.ExposureTime.Value` and `camera.Gain.Value`
- Connect disables the Connect button and enables Disconnect; Disconnect reverses this

---

## ArUco Detection

- Dictionary: `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)`
- Detector: `cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())`
- Input: grayscale conversion of the grabbed frame
- Output: `corners` list — each entry is a `(1, 4, 2)` array of (x, y) pixel coordinates
- Overlay: `cv2.circle` at each of the 4 corners per detected marker, radius=5, color green (0, 255, 0), filled
- Runs inside the background thread; main thread receives the fully annotated frame

---

## Project Setup

**Package manager:** `uv`  
**Config:** `pyproject.toml`  
**Python:** 3.11+

```
aruco_detector/
├── pyproject.toml
├── main.py
├── .python-version
└── docs/
    └── superpowers/specs/
        └── 2026-06-04-basler-aruco-ui-design.md
```

**Dependencies (pyproject.toml):**
- `opencv-contrib-python` — ArUco detection via `cv2.aruco`
- `pypylon` — Basler Pylon SDK Python bindings
- `pillow` — `ImageTk.PhotoImage` for Tkinter canvas rendering
- `numpy` — frame array manipulation

**Run:** `uv run python main.py`

---

## Error Handling

- No camera found on Connect: show "Error: no camera found" in status label, re-enable Connect button
- Camera disconnects mid-stream: background thread catches the exception, sets stop flag, updates status to "Disconnected"
- Invalid gain/exposure input: show validation error in status label, do not write to camera
- Gain/exposure out of range: clamp to camera's reported min/max before writing

---

## Out of Scope

- Image saving
- Pose estimation / 3D data
- Multi-camera selection
- Camera calibration
