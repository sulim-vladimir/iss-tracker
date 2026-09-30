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


def test_main_only_confirms_what_the_guide_has(cfg):
    """After a guide lock, main held the mount on a star for half a minute on the rig: any blob
    in its 7' field passed the 25' jump check. Now it must agree with the estimate to a few
    arcmin - a star elsewhere in the main field never takes over, the target itself does."""
    from issctl.calib import axes_offset_from_pixel, jacobian

    clock = Clock(speed=10.0)
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    gcal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=20.0, dec_cal=50.0)
    mcal = ideal_calibration(cfg["cameras"]["main"], rotation_deg=-60.0, dec_cal=50.0)
    for c in (gcal, mcal):
        c["axis2_cal"] = float(START[1])
    main = _Cam()
    main.width, main.height = 1936, 1096
    t = Tracker(cfg, {"cameras": {"guide": gcal, "main": mcal}, "main_steers": True}, mount,
                {"guide": _Cam(), "main": main}, clock, _Pass(clock.now()))
    b = gcal["boresight"]
    feed(t, clock, lambda k: (b[0], b[1]), 2.0)                     # guide locks on the target
    assert t.source == "guide"

    def main_px(arcmin):                  # where something `arcmin` from the target shows in main
        axis2 = mount.position()[1]
        return np.array(mcal["boresight"]) - jacobian(mcal, axis2) @ np.array([0.0, arcmin / 60.0])

    star = main_px(5.0)
    for _ in range(6):
        t._vision("main", Detection(star[0], star[1], 900.0, 30, clock.now()))
        clock.sleep(0.1)
    assert t.source == "guide" and t.main_streak == 0

    target = main_px(1.0)
    for _ in range(3):
        t._vision("main", Detection(target[0], target[1], 900.0, 30, clock.now()))
        clock.sleep(0.1)
    assert t.source == "main"


class _Moving(_Pass):
    """A satellite 20 deg from the mount, running away along axis1 at 0.5 deg/s."""

    def at(self, tq):
        return START + np.array([20.0 + 0.5 * (tq - self.t0), 0.0]), np.array([0.5, 0.0])


def test_a_pass_already_up_is_met_ahead_not_chased(cfg):
    """Tracking a satellite that was already up: the mount aimed at where it was NOW, arrived
    late and trailed it for the whole pass. It must wait where it can get to first."""
    clock = Clock(speed=10.0)
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    traj = _Moving(clock.now() - 60.0)          # started a minute ago
    traj.t0 = clock.now()
    t = Tracker(cfg, {"cameras": {}}, mount, {}, clock, traj)
    now = clock.now()
    meet = t._intercept(now)
    p, _ = traj.at(meet)
    assert meet > now and t._slew_time(mount.position(), p) + 3.0 <= meet - now + 1.0
    t.hold_until = meet
    ref, vel = t.target(now + 1.0)
    assert np.allclose(ref, p) and np.allclose(vel, 0.0)      # waiting there, still
    ref, vel = t.target(meet + 1.0)
    assert np.allclose(vel, [0.5, 0.0])                        # then following at its rate
    # a pass that has not started yet is simply met at its start
    later = _Pass(clock.now() + 600.0)
    assert Tracker(cfg, {"cameras": {}}, mount, {}, clock, later)._intercept(now) == later.t_start


def test_alt_az_goes_through_the_star_alignment(cfg):
    """The target dot and the logged alt/az used an ideal mount: 5 deg off the mount's own
    reading on the rig, which goes through the model."""
    from issctl import align
    from issctl import geometry as geo
    from issctl.sim import misalignment

    model = misalignment(50.0, (3.0, -2.0), 20.0)
    state = {"alignment": {"model": dict(model.to_dict(), n_points=4)}, "cameras": {}}
    clock = Clock()
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    t = Tracker(cfg, state, mount, {}, clock, _Pass(clock.now()))
    alt, az = t.altaz(START)
    ha, dec = align.pointing_hadec(state, START)
    want = geo.hadec_to_altaz(ha, dec, t.lat)
    assert alt == pytest.approx(float(want[0]), abs=1e-6) and az == pytest.approx(float(want[1]), abs=1e-6)
    ideal = geo.hadec_to_altaz(*geo.axes_to_hadec(*START), t.lat)
    assert abs(alt - float(ideal[0])) + abs(az - float(ideal[1])) > 1.0      # and that matters


def test_it_waits_where_the_satellite_comes_into_sunlight(cfg):
    """The mount went to where the pass became trackable - with the satellite still in Earth's
    shadow. It must wait where it lights up."""
    clock = Clock(speed=10.0)
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    now = clock.now()
    traj = _Moving(now + 30.0)                   # rises in 30 s...
    traj.t0 = now
    lights = now + 120.0                         # ...but only lights up 90 s after that
    traj.illum_at = lambda tq: 1.0 if tq >= lights else 0.0
    t = Tracker(cfg, {"cameras": {}}, mount, {}, clock, traj)
    meet = t._intercept(now)
    assert meet >= lights and meet < lights + 2.0


def test_main_noise_wandering_about_never_takes_over(cfg):
    """On the rig, main's 'detections' jumped (66,161) -> (1294,1080) -> (763,17) -> ... - noise
    in a 10 ms frame with the satellite not in view - took over, and drove the mount off at
    1 deg/s in Dec. A handoff now needs a steady blob near the centre, and main in charge may
    not jump."""
    clock = Clock(speed=10.0)
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    gcal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=20.0, dec_cal=50.0)
    mcal = ideal_calibration(cfg["cameras"]["main"], rotation_deg=-60.0, dec_cal=50.0)
    for c in (gcal, mcal):
        c["axis2_cal"] = float(START[1])
    main = _Cam()
    main.width, main.height = 1936, 1096
    t = Tracker(cfg, {"cameras": {"guide": gcal, "main": mcal}, "main_steers": True}, mount,
                {"guide": _Cam(), "main": main}, clock, _Pass(clock.now()))
    b = gcal["boresight"]
    feed(t, clock, lambda k: (b[0], b[1]), 2.0)
    assert t.source == "guide"
    t.step()
    assert main.gate is not None and main.gate[2] < 300            # main looks near its centre

    c = np.array(mcal["boresight"])
    rng = np.random.default_rng(0)
    for _ in range(12):                                              # noise, wandering about
        x, y = c + rng.uniform(-200, 200, 2)
        t._vision("main", Detection(x, y, 300.0, 25, clock.now()))
        clock.sleep(0.1)
    assert t.source == "guide"

    for k in range(3):                                               # the target, steady
        t._vision("main", Detection(c[0] + 10 + k, c[1] - 5, 900.0, 40, clock.now()))
        clock.sleep(0.1)
    assert t.source == "main"
    before = t.cross.copy()
    t._vision("main", Detection(c[0] + 180, c[1] + 150, 900.0, 40, clock.now()))   # a jump
    assert np.allclose(t.cross, before)


def test_by_default_main_does_not_steer(cfg):
    """Guide only unless switched on: main watches and records."""
    clock = Clock(speed=10.0)
    mount = SimMount(cfg, {}, clock, start=START)
    mount.query()
    gcal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=20.0, dec_cal=50.0)
    mcal = ideal_calibration(cfg["cameras"]["main"], rotation_deg=-60.0, dec_cal=50.0)
    for c in (gcal, mcal):
        c["axis2_cal"] = float(START[1])
    main = _Cam()
    main.width, main.height = 1936, 1096
    t = Tracker(cfg, {"cameras": {"guide": gcal, "main": mcal}}, mount,
                {"guide": _Cam(), "main": main}, clock, _Pass(clock.now()))
    assert not t.main_steers
    b = gcal["boresight"]
    feed(t, clock, lambda k: (b[0], b[1]), 2.0)
    c = mcal["boresight"]
    for k in range(6):                                  # a perfect, steady target in main
        t._vision("main", Detection(c[0] + 3, c[1] - 2, 900.0, 40, clock.now()))
        clock.sleep(0.1)
    assert t.source == "guide" and t.main_streak == 0
