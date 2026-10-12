import time


class Clock:
    """Wall clock in unix seconds. Simulation can shift it to a pass and speed it up."""

    def __init__(self, start_unix=None, speed=1.0):
        self._real0 = time.monotonic()
        self._unix0 = time.time() if start_unix is None else start_unix
        self.speed = speed
        # real time reads the wall clock itself: monotonic stops while the machine is suspended,
        # and a console that slept 21 h planned passes and stamped sessions 21 h in the past
        self._wall = start_unix is None and speed == 1.0

    def now(self):
        if self._wall:
            return time.time()
        return self._unix0 + (time.monotonic() - self._real0) * self.speed

    def sleep(self, seconds):
        if seconds > 0:
            time.sleep(seconds / self.speed)
