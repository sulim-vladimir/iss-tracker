"""Threaded capture + detection for ZWO ASI cameras (and a simulated camera)."""

import threading
import time

import numpy as np

from .detect import detect


class Camera:
    def __init__(self, name, cam_cfg, clock):
        self.name = name
        self.cfg = cam_cfg
        self.clock = clock
        self.bayer = bool(cam_cfg.get("bayer"))
        self.width = cam_cfg["width"] // cam_cfg["bin"]
        self.height = cam_cfg["height"] // cam_cfg["bin"]
        self.exposure_ms = cam_cfg["exposure_ms"]
        self.gate = None
        self.sinks = []
        self.fps = 0.0
        self._frame = None
        self._det = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _open(self):
        pass

    def _close(self):
        pass

    def _grab(self):
        raise NotImplementedError

    def start(self):
        self._open()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._close()

    def _run(self):
        n, t_fps = 0, time.monotonic()
        while not self._stop.is_set():
            img, t = self._grab()
            if img is None:
                continue
            det = detect(img, self.cfg["detect_sigma"], self.cfg["detect_min_area"], self.bayer, self.gate)
            if det:
                det.t = t
            with self._lock:
                self._frame, self._det = img, det
                self._seq += 1
            for sink in self.sinks:
                sink(img, t)
            n += 1
            el = time.monotonic() - t_fps
            if el >= 1.0:
                self.fps, n, t_fps = n / el, 0, time.monotonic()

    def latest(self):
        with self._lock:
            return self._frame, self._det, self._seq


class AsiCamera(Camera):
    _sdk_ready = False

    def __init__(self, name, cam_cfg, clock, sdk_lib):
        super().__init__(name, cam_cfg, clock)
        self.sdk_lib = sdk_lib
        self.cam = None

    def _open(self):
        import zwoasi as asi

        if not AsiCamera._sdk_ready:
            asi.init(self.sdk_lib)
            AsiCamera._sdk_ready = True
        names = asi.list_cameras()
        match = [i for i, n in enumerate(names) if self.cfg["name_match"] in n]
        if not match:
            raise RuntimeError(f"camera '{self.cfg['name_match']}' not found; connected: {names}")
        cam = asi.Camera(match[0])
        cam.stop_video_capture()
        cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, self.cfg["usb_bandwidth"])
        cam.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1)
        cam.set_image_type(asi.ASI_IMG_RAW8)
        cam.set_roi(width=self.width, height=self.height, bins=self.cfg["bin"])
        self.width, self.height = cam.get_roi()[2:4]
        cam.set_control_value(asi.ASI_GAIN, int(self.cfg["gain"]))
        cam.set_control_value(asi.ASI_EXPOSURE, int(self.exposure_ms * 1000))
        cam.start_video_capture()
        self.cam, self._asi = cam, asi

    def set_exposure(self, ms):
        self.exposure_ms = ms
        self.cam.set_control_value(self._asi.ASI_EXPOSURE, int(ms * 1000))

    def set_gain(self, gain):
        self.cam.set_control_value(self._asi.ASI_GAIN, int(gain))

    def _grab(self):
        try:
            img = self.cam.capture_video_frame(timeout=int(self.exposure_ms * 2 + 500))
        except self._asi.ZWO_IOError:
            return None, None
        t = self.clock.now() - self.exposure_ms / 2000.0 - self.cfg["latency_s"]
        return img, t

    def _close(self):
        if self.cam:
            self.cam.stop_video_capture()
            self.cam.close()


class SimCamera(Camera):
    """Renders the ISS as a Gaussian blob where the simulated world says it is."""

    def __init__(self, name, cam_cfg, clock, world, fps=30.0, blob_sigma=None):
        super().__init__(name, cam_cfg, clock)
        self.world = world
        self.period = 1.0 / fps
        self.blob_sigma = blob_sigma or (1.5 if not self.bayer else 6.0)
        rng = np.random.default_rng(1)
        self._noise = [np.clip(rng.normal(20, 3, (self.height, self.width)), 0, 255).astype(np.uint8)
                       for _ in range(4)]
        self._k = 0
        self._next = time.monotonic()

    def _grab(self):
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.period, time.monotonic())
        t = self.clock.now()
        img = self._noise[self._k].copy()
        self._k = (self._k + 1) % len(self._noise)
        p = self.world.pixel(self.name, t)
        if p is not None:
            s = self.blob_sigma
            r = int(4 * s) + 1
            x0, y0 = int(round(p[0])), int(round(p[1]))
            xs, ys = np.arange(x0 - r, x0 + r + 1), np.arange(y0 - r, y0 + r + 1)
            gx = np.exp(-((xs - p[0]) ** 2) / (2 * s * s))
            gy = np.exp(-((ys - p[1]) ** 2) / (2 * s * s))
            patch = 180.0 * np.outer(gy, gx)
            xa, xb = max(0, x0 - r), min(self.width, x0 + r + 1)
            ya, yb = max(0, y0 - r), min(self.height, y0 + r + 1)
            if xa < xb and ya < yb:
                sub = patch[ya - (y0 - r):yb - (y0 - r), xa - (x0 - r):xb - (x0 - r)]
                img[ya:yb, xa:xb] = np.clip(img[ya:yb, xa:xb] + sub, 0, 255).astype(np.uint8)
        return img, t
