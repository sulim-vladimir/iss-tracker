import time


class Clock:
    """Wall clock in unix seconds. Simulation can shift it to a pass and speed it up."""

    def __init__(self, start_unix=None, speed=1.0):
        self._real0 = time.monotonic()
        self._unix0 = time.time() if start_unix is None else start_unix
        self.speed = speed

    def now(self):
        return self._unix0 + (time.monotonic() - self._real0) * self.speed

    def sleep(self, seconds):
        if seconds > 0:
            time.sleep(seconds / self.speed)
