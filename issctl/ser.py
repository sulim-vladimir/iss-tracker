"""Background SER video writer (readable by AutoStakkert!, SER Player, PIPP)."""

import queue
import struct
import threading
import time

SER_COLOR = {None: 0, "RGGB": 8, "GRBG": 9, "GBRG": 10, "BGGR": 11}
UNIX_TO_SER_TICKS = 62135596800  # seconds from 0001-01-01 to 1970-01-01


def _ticks(unix):
    return int((unix + UNIX_TO_SER_TICKS) * 1e7)


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
