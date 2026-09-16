"""Background SER video writer (readable by AutoStakkert!, SER Player, PIPP)."""

import queue
import struct
import threading
import time

SER_COLOR = {None: 0, "RGGB": 8, "GRBG": 9, "GBRG": 10, "BGGR": 11}
UNIX_TO_SER_TICKS = 62135596800  # seconds from 0001-01-01 to 1970-01-01


def _ticks(unix):
    return int((unix + UNIX_TO_SER_TICKS) * 1e7)


class RecordControl:
    """Start/stop recording on demand. Each start opens a new SER file; recording is also
    suspended automatically whenever the ISS is not visible (shadow, obstruction)."""

    def __init__(self, cam, out_dir, bayer=None, telescope="", instrument="", prefix="iss"):
        self.cam = cam
        self.out_dir = out_dir
        self.bayer, self.telescope, self.instrument, self.prefix = bayer, telescope, instrument, prefix
        self.writer = None
        self.enabled = False
        self.visible = True
        self.last = None  # summary of the most recently closed file

    @property
    def available(self):
        return self.cam is not None

    def set_enabled(self, on):
        if not self.available:
            return
        if on and self.writer is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"{self.prefix}-{time.strftime('%Y%m%d-%H%M%S')}.ser"
            self.writer = SerWriter(path, self.cam.width, self.cam.height, bayer=self.bayer,
                                    telescope=self.telescope, instrument=self.instrument)
            self.cam.sinks.append(self.writer)
        self.enabled = bool(on)
        self._apply()
        if not on:
            self.close()

    def set_visible(self, visible):
        self.visible = bool(visible)
        self._apply()

    def _apply(self):
        if self.writer:
            self.writer.active = self.enabled and self.visible

    def state(self):
        if not self.available:
            return None
        w = self.writer
        return {"available": True, "recording": self.enabled,
                "paused": self.enabled and not self.visible,
                "frames": w.frames if w else 0, "dropped": w.dropped if w else 0,
                "path": w.path.name if w else None}

    def close(self):
        if self.writer:
            if self.writer in self.cam.sinks:
                self.cam.sinks.remove(self.writer)
            self.writer.close()
            self.last = {"path": self.writer.path, "frames": self.writer.frames,
                         "dropped": self.writer.dropped}
            self.writer = None


class SerWriter:
    def __init__(self, path, width, height, bayer=None, telescope="", instrument=""):
        self.path = path
        self.width, self.height = width, height
        self.frames = 0
        self.dropped = 0
        self.active = False
        self._stamps = []
        self._q = queue.Queue(maxsize=512)
        self._f = open(path, "wb")
        self._header = [SER_COLOR[bayer], width, height, telescope, instrument]
        self._write_header()
        self._thread = threading.Thread(target=self._run, name="ser-writer", daemon=True)
        self._thread.start()

    def _write_header(self):
        color, w, h, tel, inst = self._header
        now = time.time()
        hdr = (b"LUCAM-RECORDER"
               + struct.pack("<7i", 0, color, 0, w, h, 8, self.frames)
               + b"issctl".ljust(40, b"\0") + inst.encode()[:40].ljust(40, b"\0")
               + tel.encode()[:40].ljust(40, b"\0")
               + struct.pack("<qq", _ticks(now + (time.localtime().tm_gmtoff or 0)), _ticks(now)))
        self._f.seek(0)
        self._f.write(hdr)
        self._f.seek(0, 2)

    def __call__(self, img, t):
        if not self.active:
            return
        if img.shape != (self.height, self.width):
            return
        try:
            self._q.put_nowait((img, t))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            img, t = item
            self._f.write(img.tobytes())
            self._stamps.append(_ticks(t))
            self.frames += 1

    def close(self):
        self.active = False
        self._q.put(None)
        self._thread.join()
        for s in self._stamps:
            self._f.write(struct.pack("<q", s))
        self._write_header()
        self._f.close()
