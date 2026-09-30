"""First lock in pass mode: the brightest blob anywhere, trusted outright, was a star.

On the real rig (2026-09-30) the tracker took a star as a rocket body while still slewing to the
pass, 90 s before it rose, and drove its estimate 60 deg off. A first lock now needs the pass to
have started, the mount to have arrived, the blob to be near the prediction and to hold still in
the frame while the stars drift past."""

import numpy as np
import pytest

from issctl.calib import ideal_calibration
from issctl.clock import Clock
from issctl.config import load_config
from issctl.control import FreeRun, Tracker
from issctl.detect import Detection
from issctl.mount import HOME, SimMount

START = HOME + np.array([30.0, -40.0])


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class _Pass:
    """A pass that has the mount exactly where it should be, moving at a satellite's rate."""

    side = "east_looking"

    def __init__(self, t_start, where=START, rate=(0.3, 0.1)):
        self.t_start, self.t_end, self.where, self.rate = t_start, t_start + 600.0, where, rate

    def at(self, tq):
        return np.array(self.where, dtype=float), np.array(self.rate, dtype=float)

    def illum_at(self, tq):
        return 1.0

    def open_at(self, tq):
        return 1.0


class _Cam:
    width, height, gate, manual = 1280, 960, None, False

    def select(self, x, y, radius=None):
        self.gate, self.manual = (x, y, 40.0), True

    def clear_selection(self):
        self.gate, self.manual = None, False

    def latest(self):
        return None, None, 0


def rig(cfg, traj_start=0.0, **kw):
    clock = Clock(speed=10.0)
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    cal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=20.0, dec_cal=50.0)
    cal["axis2_cal"] = float(START[1])
    traj = _Pass(clock.now() + traj_start, **kw)
    t = Tracker(cfg, {"cameras": {"guide": cal}}, mount, {"guide": _Cam()}, clock, traj)
    return t, clock, cal


def feed(t, clock, xy_at, seconds, every=0.2):
    """Detections every `every` s for `seconds`; xy_at(k) gives the pixel of the k-th."""
    for k in range(int(seconds / every)):
        x, y = xy_at(k)
        t._vision("guide", Detection(x, y, 500.0, 20, clock.now()))
        clock.sleep(every)


def test_a_still_blob_near_the_prediction_is_acquired_after_it_has_held(cfg):
    t, clock, cal = rig(cfg)
    b = cal["boresight"]
    feed(t, clock, lambda k: (b[0] + 20.0, b[1] - 10.0), 0.6)
    assert not np.isfinite(t.last_good)                  # not yet: it has not held long enough
    feed(t, clock, lambda k: (b[0] + 20.0 + 0.5 * (k % 2), b[1] - 10.0), 1.0)
    assert np.isfinite(t.last_good) and t.source == "guide"


def test_a_star_drifting_through_is_never_taken(cfg):
    """With the mount following a satellite at 0.3 deg/s, a star crosses the guide at ~20 px/s."""
    t, clock, cal = rig(cfg)
    b = cal["boresight"]
    feed(t, clock, lambda k: (b[0] - 100.0 + 4.0 * k, b[1] + 1.6 * k), 5.0)
    assert not np.isfinite(t.last_good)


def test_nothing_is_taken_before_the_pass_starts(cfg):
    t, clock, cal = rig(cfg, traj_start=100.0)
    b = cal["boresight"]
    feed(t, clock, lambda k: (b[0], b[1]), 3.0)
    assert not np.isfinite(t.last_good)


def test_nothing_is_taken_while_the_mount_is_still_slewing(cfg):
    t, clock, cal = rig(cfg, where=START + np.array([5.0, 3.0]))   # 5 deg to go
    b = cal["boresight"]
    feed(t, clock, lambda k: (b[0], b[1]), 3.0)
    assert not np.isfinite(t.last_good)


def test_nothing_further_than_the_acquisition_radius(cfg):
    t, clock, cal = rig(cfg)
    b = cal["boresight"]
    px_per_deg = float(np.linalg.norm(np.array(cal["J"])[:, 1]))
    feed(t, clock, lambda k: (b[0] + 4.0 * px_per_deg, b[1]), 3.0)     # 4 deg off, holding still
    assert not np.isfinite(t.last_good)


def test_the_pre_lock_search_is_a_circle_round_the_boresight(cfg):
    t, clock, cal = rig(cfg)
    t._search_gate("guide", clock.now())
    gx, gy, gr = t.cams["guide"].gate
    assert (gx, gy) == tuple(cal["boresight"])
    px_per_deg = float(np.linalg.norm(np.array(cal["J"])[:, 1]))
    assert gr == pytest.approx(3.0 * px_per_deg, rel=0.01)


def test_a_hand_pick_and_servo_mode_are_not_held_back(cfg):
    t, clock, cal = rig(cfg, traj_start=100.0)          # before the pass, even
    t.select("guide", 700.0, 500.0)
    feed(t, clock, lambda k: (700.0, 500.0), 0.2)
    assert np.isfinite(t.last_good)
    clock = Clock()
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    servo = Tracker(cfg, {"cameras": {"guide": cal}}, mount, {"guide": _Cam()}, clock, FreeRun())
    servo.step()
    servo._vision("guide", Detection(700.0, 500.0, 500.0, 20, clock.now()))
    assert np.isfinite(servo.last_good)
