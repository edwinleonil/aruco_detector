import math
import queue
import threading
import tkinter as tk
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
            if not self._stop_event.is_set():
                self.error = str(exc)
        finally:
            if self.camera is not None:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()

    def stop(self) -> None:
        self._stop_event.set()


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
        self._status_var.set(f"Applied: gain={gain:.2f} exposure={exposure:.0f} µs")

    # ---- Frame polling ----

    def _poll_frames(self) -> None:
        # Check if thread died with an error
        if self._camera_thread is not None and not self._camera_thread.is_alive():
            err = self._camera_thread.error or "camera stopped unexpectedly"
            self._status_var.set(f"Error: {err}")
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
            self._canvas.delete("all")
            self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)
        except queue.Empty:
            pass

        self.after(self._POLL_MS, self._poll_frames)

    def _on_close(self) -> None:
        self._disconnect()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
