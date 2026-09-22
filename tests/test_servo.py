"""Servo mode: tracking with no orbit, no site and no alignment - only the cameras.

These exist because each one is a bug that got through once: a reference seeded from a stale
pose sent the mount to the home position, a run with nothing in the field never ended, and a
search gate that opened slower than the give-up timeout turned a one-second glitch into a lost
pass.
"""

import numpy as np
import pytest

from issctl.clock import Clock
from issctl.config import load_config
from issctl.control import FreeRun, Tracker
from issctl.mount import HOME, SimMount


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def tracker_for(cfg, mount, reference):
    return Tracker(cfg, {"cameras": {}}, mount, {}, Clock(), reference)


def test_free_run_holds_still_until_something_is_seen(cfg):
    """With no detections the reference never moves, so neither does the mount."""
    clock = Clock()
    start = HOME + [20.0, -30.0]
    mount = SimMount(cfg, {}, clock, start=start)
    t = tracker_for(cfg, mount, FreeRun())
    for _ in range(50):
        t.step()
    assert np.all(np.abs(mount.position() - start) < 0.01)


def test_free_run_seeds_from_a_fresh_reading_not_the_placeholder(cfg):
    """The driver reports the home pose until it has queried once.

    Seeding from that stale value anchored the whole run to the wrong place and the loop then
    slewed the tube there - away from the target the user had just pointed it at.
    """
    clock = Clock()
    start = HOME + [35.0, -20.0]
    mount = SimMount(cfg, {}, clock, start=start)
    assert np.allclose(mount.last[1], HOME)        # the placeholder, before any query
    ref = FreeRun()
    tracker_for(cfg, mount, ref).step()
    assert np.all(np.abs(ref.seed - start) < 0.01)


def test_servo_uses_its_own_estimator_gains(cfg):
    """cross_alpha/cross_beta smooth a residual; in servo mode they carry the whole motion."""
    clock = Clock()
    mount = SimMount(cfg, {}, clock)
    pass_mode = tracker_for(cfg, mount, _FakeTrajectory())
    servo = tracker_for(cfg, mount, FreeRun())
    assert servo.alpha > pass_mode.alpha and servo.beta > pass_mode.beta
    assert servo.growth > pass_mode.growth


def test_the_gate_opens_before_the_run_gives_up(cfg):
    """A servo run must not abandon a target its own search gate has not finished looking for."""
    clock = Clock()
    mount = SimMount(cfg, {}, clock)
    t = tracker_for(cfg, mount, FreeRun())
    full = t.tr["max_reacquire_arcmin"]
    assert t._jump_allowance(t.give_up_s) >= full, (
        "the gate is still opening when the run gives up: a brief loss becomes a lost pass")


def test_limit_guard_refuses_to_drive_past_a_stop(cfg):
    """Nothing in servo mode knows where the mount points, so this is the only collision check."""
    clock = Clock()
    mount = SimMount(cfg, {}, clock)
    t = tracker_for(cfg, mount, FreeRun())
    lim = cfg["mount"]["axis1_hour_limit"]
    lo, hi = cfg["mount"].get("axis2_limits", [-10.0, 190.0])
    at_limit = np.array([lim, hi])
    assert np.allclose(t._limit_guard([0.5, 0.5], at_limit), [0.0, 0.0])   # further out: refused
    assert np.allclose(t._limit_guard([-0.5, -0.5], at_limit), [-0.5, -0.5])  # back in: allowed
    assert np.allclose(t._limit_guard([0.5, 0.5], np.array([0.0, 90.0])), [0.5, 0.5])
    assert np.allclose(t._limit_guard([-0.5, -0.5], np.array([-lim, lo])), [0.0, 0.0])


def test_reanchoring_does_not_move_the_target(cfg):
    """The seed follows the estimate so axis1 wrapping stays unambiguous - silently."""
    clock = Clock()
    mount = SimMount(cfg, {}, clock)
    ref = FreeRun()
    t = tracker_for(cfg, mount, ref)
    t.step()
    t.cross = np.array([40.0, -5.0])
    t.t_update = clock.now()
    before = t.target(t.t_update)[0].copy()
    t._reanchor()
    assert np.max(np.abs(t.cross)) < 1e-9
    assert np.allclose(t.target(t.t_update)[0], before)


class _FakeTrajectory:
    """Just enough of a planned pass for the tracker to configure itself."""

    t_start, t_end, side = 0.0, 1e9, "east_looking"

    def at(self, tq):
        return np.zeros(2), np.zeros(2)

    def illum_at(self, tq):
        return 1.0

    def open_at(self, tq):
        return 1.0
