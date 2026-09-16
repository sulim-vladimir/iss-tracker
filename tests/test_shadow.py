"""Controller behaviour around Earth's shadow, with a scripted trajectory and fake camera."""

import numpy as np
import pytest

from issctl.calib import ideal_calibration, jacobian
from issctl.config import load_config
from issctl.control import Tracker
from issctl.detect import Detection
from issctl.mount import SimMount
from issctl.predict import Trajectory

T0 = 1_700_000_000.0
DARK_UNTIL = T0 + 100.0


class FakeClock:
    speed = 1.0

    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 0.0)


class StubCam:
    def __init__(self, name, cam_cfg):
        self.name = name
        self.width = cam_cfg["width"]
        self.height = cam_cfg["height"]
        self.gate = None
        self._det = None
        self._seq = 0

    def emit(self, x, y, t):
        self._det = Detection(x, y, 1000.0, 9, t)
        self._seq += 1

    def latest(self):
        return None, self._det, self._seq


def make_trajectory():
    t = np.arange(T0, T0 + 200.0, 0.5)
    a1 = 10.0 + 0.2 * (t - T0)          # 0.2 deg/s on axis1
    a2 = np.full_like(t, 45.0)
    lit = np.where(t < DARK_UNTIL, 0.0, 1.0)
    return Trajectory(t, a1, a2, np.full_like(t, 40.0), "east_looking", T0, T0 + 200.0, lit)


@pytest.fixture
def rig():
    cfg = load_config()
    traj = make_trajectory()
    clock = FakeClock(T0 + 50.0)
    state = {"index": [0.0, 90.0], "cameras": {"guide": ideal_calibration(cfg["cameras"]["guide"])}}
    mount = SimMount(cfg, state, clock, start=traj.at(T0 + 50.0)[0])
    cam = StubCam("guide", cfg["cameras"]["guide"])
    tracker = Tracker(cfg, state, mount, {"guide": cam}, clock, traj, log=lambda *a: None)
    return cfg, traj, clock, mount, cam, tracker, state


def pixel_for_offset(state, axis2, offset_deg):
    """Pixel where an object sitting offset_deg away from the pointing would appear."""
    cal = state["cameras"]["guide"]
    return np.array(cal["boresight"]) - jacobian(cal, axis2) @ np.asarray(offset_deg)


def run_steps(tracker, mount, clock, n=4, dt=0.05):
    for _ in range(n):
        mount.query()
        tracker.step()
        clock.sleep(dt)


def test_ignores_detections_in_shadow(rig):
    cfg, traj, clock, mount, cam, tracker, state = rig
    run_steps(tracker, mount, clock, 2)
    assert tracker.source == "shadow"

    # a star drifts through the frame while the ISS is dark
    star = pixel_for_offset(state, mount.position()[1], [0.4, -0.3])
    cam.emit(star[0], star[1], clock.now())
    run_steps(tracker, mount, clock, 3)

    assert tracker.source == "shadow"
    assert np.allclose(tracker.cross, 0.0)
    assert tracker.time_offset == 0.0
    assert tracker.last_seen["guide"] == -np.inf


def test_reacquires_after_shadow_exit(rig):
    cfg, traj, clock, mount, cam, tracker, state = rig
    run_steps(tracker, mount, clock, 2)
    assert tracker.source == "shadow"

    clock.t = DARK_UNTIL + 5.0          # ISS comes back into sunlight
    mount.query()
    tracker.step()
    assert tracker.source == "predict"
    # searching again, centred on the boresight; on this narrow guide field that is the whole frame
    assert cam.gate is not None
    assert cam.gate[:2] == tuple(state["cameras"]["guide"]["boresight"])
    assert cam.gate[2] >= 0.5 * cam.height

    offset = np.array([0.05, -0.04])
    px = pixel_for_offset(state, mount.position()[1], offset)
    cam.emit(px[0], px[1], clock.now())
    run_steps(tracker, mount, clock, 2)

    assert tracker.source == "guide"
    assert tracker.last_seen["guide"] == pytest.approx(clock.now(), abs=0.5)
    assert np.linalg.norm(tracker.cross) > 0.0
    assert cam.gate is not None


def test_rejects_outlier_once_locked(rig):
    cfg, traj, clock, mount, cam, tracker, state = rig
    clock.t = DARK_UNTIL + 5.0
    mount.mech = traj.at(clock.now())[0] - mount.index   # mount already near the ISS, as when locked
    px = pixel_for_offset(state, mount.position()[1], [0.02, 0.01])
    cam.emit(px[0], px[1], clock.now())
    run_steps(tracker, mount, clock, 2)
    assert tracker.source == "guide"
    cross_locked = tracker.cross.copy()

    # a much brighter star appears 3 degrees away: far beyond max_offset_jump_arcmin
    star = pixel_for_offset(state, mount.position()[1], [3.0, 2.0])
    cam.emit(star[0], star[1], clock.now())
    run_steps(tracker, mount, clock, 2)

    assert tracker.rejected >= 1
    assert np.allclose(tracker.cross, cross_locked, atol=0.02)


def test_jump_allowance_grows_while_blind(rig):
    cfg, traj, clock, mount, cam, tracker, state = rig
    base = cfg["tracking"]["max_offset_jump_arcmin"]
    assert tracker._jump_allowance(0.0) == base                    # locked: tight
    assert tracker._jump_allowance(10.0) > base                    # coasting: wider
    assert tracker._jump_allowance(1e6) == cfg["tracking"]["max_reacquire_arcmin"]  # but bounded
