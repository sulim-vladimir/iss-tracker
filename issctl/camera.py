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
        self.gain = cam_cfg["gain"]
        self.exposure_unit = "ms"   # some V4L2 drivers only expose a raw register
        self.gate = None
        self.follow = False   # keep the gate on whatever was picked, frame to frame
        self.manual = False   # picked by the user: the tracker must not move the gate
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

    def set_exposure(self, ms):
        self.exposure_ms = max(0.001, float(ms))

    def set_gain(self, gain):
        self.gain = max(0, int(gain))

    def select(self, x, y, radius=None):
        """Lock onto the object near (x, y) instead of simply the brightest one."""
        self.gate = (float(x), float(y), float(radius or max(20.0, 0.03 * self.width)))
        self.follow = True
        self.manual = True

    def clear_selection(self):
        self.gate = None
        self.follow = False
        self.manual = False

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
            gate = self.gate  # snapshot: a click may replace it while we are detecting
            det = detect(img, self.cfg["detect_sigma"], self.cfg["detect_min_area"], self.bayer, gate,
                         max_area=self.cfg.get("detect_max_area", 0),
                         edge_margin=self.cfg.get("detect_edge_margin", 0))
            if det:
                det.t = t
                if self.follow and gate is not None and self.gate is gate:
                    self.gate = (det.x, det.y, gate[2])  # stay on the object we were given
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


SDK_CANDIDATES = [
    "/usr/local/lib/libASICamera2.so",
    "/usr/lib/libASICamera2.so",
    "/usr/lib/x86_64-linux-gnu/libASICamera2.so",
    "/usr/lib/aarch64-linux-gnu/libASICamera2.so",   # Raspberry Pi OS 64-bit
    "/opt/FireCapture_v2.7/libASICamera2.so",
]


def find_sdk(configured=None):
    """The ZWO SDK ships with FireCapture and INDI as well, so look around before giving up."""
    import glob
    import os

    seen = []
    for path in [configured, *SDK_CANDIDATES, *sorted(glob.glob("/opt/FireCapture*/libASICamera2.so"))]:
        if path and path not in seen:
            seen.append(path)
            if os.path.exists(path):
                return path
    raise RuntimeError("libASICamera2.so not found - install the ZWO SDK (or point [cameras] "
                       "sdk_lib at it). Looked in: " + ", ".join(seen))


def import_zwoasi():
    """zwoasi tries to load the SDK from the linker path when imported and logs a scary
    'ASI SDK library not found' warning if it is not there. We load it explicitly by path a moment
    later, so drop that one message instead of letting it worry people."""
    import logging

    class _Quiet(logging.Filter):
        def filter(self, record):
            return "ASI SDK library not found" not in record.getMessage()

    quiet = _Quiet()
    logging.getLogger().addFilter(quiet)
    try:
        import zwoasi as asi
    finally:
        logging.getLogger().removeFilter(quiet)
    return asi


class AsiCamera(Camera):
    _sdk_ready = False

    def __init__(self, name, cam_cfg, clock, sdk_lib):
        super().__init__(name, cam_cfg, clock)
        self.sdk_lib = sdk_lib
        self.cam = None

    def _open(self):
        lib = find_sdk(self.sdk_lib) if not AsiCamera._sdk_ready else None
        asi = import_zwoasi()

        if not AsiCamera._sdk_ready:
            if lib != self.sdk_lib:
                print(f"using ZWO SDK at {lib}")
            asi.init(lib)
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
        super().set_exposure(ms)
        self.cam.set_control_value(self._asi.ASI_EXPOSURE, int(self.exposure_ms * 1000))

    def set_gain(self, gain):
        super().set_gain(gain)
        self.cam.set_control_value(self._asi.ASI_GAIN, self.gain)

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


V4L2_QUERYCTRL = 0xC0445624
V4L2_S_CTRL = 0xC008561C
V4L2_NEXT_CTRL = 0x80000000
_QUERY_FMT = "II32sIIIIi"


def v4l2_controls(device):
    """{slug: (id, min, max)} for a V4L2 device, straight from the driver - no v4l-utils needed."""
    import fcntl
    import struct

    out = {}
    with open(device, "rb", buffering=0) as fd:
        cid = V4L2_NEXT_CTRL
        while True:
            buf = bytearray(struct.pack(_QUERY_FMT, cid, 0, b"\0" * 32, 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(fd, V4L2_QUERYCTRL, buf, True)
            except OSError:
                break
            i, kind, name, lo, hi, step, dflt, flags = struct.unpack(_QUERY_FMT, bytes(buf))
            slug = name.split(b"\0")[0].decode().lower().replace(", ", "_").replace(" ", "_")
            if kind != 6:  # skip control-class headers
                out[slug] = (i, lo, hi)
            cid = i | V4L2_NEXT_CTRL
    return out


class V4l2Camera(Camera):
    """Any V4L2 device: a Philips SPC900NC (pwc driver), a UVC webcam, a capture stick.

    Controls are set through V4L2 ioctls, so nothing extra needs installing. Names differ per
    driver (pwc has "exposure" as a raw 0-255 register and "gain_automatic"; UVC has
    "exposure_time_absolute" in 100 us units), so they come from the config.
    """

    def __init__(self, name, cam_cfg, clock):
        super().__init__(name, cam_cfg, clock)
        self.device = cam_cfg.get("device", "/dev/video0")
        self.cap = None
        self.controls = {}
        if not cam_cfg.get("v4l2_exposure_unit_ms", 0.1):
            self.exposure_unit = "raw"   # pwc: 0-255 register, no millisecond mapping

    def _set_ctrl(self, slug, value):
        import fcntl
        import struct

        info = self.controls.get(slug)
        if not info:
            return False
        cid, lo, hi = info
        value = int(max(lo, min(hi, round(value))))
        try:
            with open(self.device, "rb", buffering=0) as fd:
                fcntl.ioctl(fd, V4L2_S_CTRL, struct.pack("Ii", cid, value), True)
            return True
        except OSError as e:
            print(f"{self.name}: cannot set {slug}={value}: {e}")
            return False

    def _open(self):
        import cv2

        try:
            self.controls = v4l2_controls(self.device)
        except OSError as e:
            print(f"{self.name}: cannot list controls on {self.device}: {e}")
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.device}")
        if self.cfg.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.cfg.get("fps"):
            cap.set(cv2.CAP_PROP_FPS, self.cfg["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # we want the newest frame, not a queue
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        self.cap, self._cv2 = cap, cv2
        for cmd in self.cfg.get("v4l2_manual", []):   # e.g. "gain_automatic=0"
            slug, _, value = cmd.partition("=")
            self._set_ctrl(slug, float(value or 0))
        self.set_exposure(self.exposure_ms)
        self.set_gain(self.gain)

    def set_exposure(self, ms):
        super().set_exposure(ms)
        if not self.cap:
            return
        unit = self.cfg.get("v4l2_exposure_unit_ms", 0.1)
        # unit = 0 means the driver exposes a raw register (pwc): pass the number through
        raw = self.exposure_ms if not unit else self.exposure_ms / unit
        if not self._set_ctrl(self.cfg.get("v4l2_exposure_ctrl", "exposure"), raw):
            self.cap.set(self._cv2.CAP_PROP_EXPOSURE, raw)

    def set_gain(self, gain):
        super().set_gain(gain)
        if not self.cap:
            return
        if not self._set_ctrl(self.cfg.get("v4l2_gain_ctrl", "gain"), self.gain):
            self.cap.set(self._cv2.CAP_PROP_GAIN, self.gain)

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, None
        t = self.clock.now() - self.exposure_ms / 2000.0 - self.cfg["latency_s"]
        if frame.ndim == 3:   # colour sensor: detection only needs brightness
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        return frame, t

    def _close(self):
        if self.cap:
            self.cap.release()


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
        if hasattr(self.world, "blobs"):
            blobs = self.world.blobs(self.name, t)
        else:
            p = self.world.pixel(self.name, t)
            blobs = [(p, 180.0)] if p is not None else []
        for p, amplitude in blobs:
            self._draw(img, p, amplitude)
        return img, t

    def _draw(self, img, p, amplitude):
        s = self.blob_sigma
        r = int(4 * s) + 1
        x0, y0 = int(round(p[0])), int(round(p[1]))
        xs, ys = np.arange(x0 - r, x0 + r + 1), np.arange(y0 - r, y0 + r + 1)
        gx = np.exp(-((xs - p[0]) ** 2) / (2 * s * s))
        gy = np.exp(-((ys - p[1]) ** 2) / (2 * s * s))
        patch = amplitude * np.outer(gy, gx)
        xa, xb = max(0, x0 - r), min(self.width, x0 + r + 1)
        ya, yb = max(0, y0 - r), min(self.height, y0 + r + 1)
        if xa < xb and ya < yb:
            sub = patch[ya - (y0 - r):yb - (y0 - r), xa - (x0 - r):xb - (x0 - r)]
            img[ya:yb, xa:xb] = np.clip(img[ya:yb, xa:xb] + sub, 0, 255).astype(np.uint8)
