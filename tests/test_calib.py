"""Calibration sanity checks - especially the scale diagnostic, which catches a wrong gear ratio."""

import numpy as np
import pytest

from issctl import geometry as geo

from issctl.calib import (axis1_plausible, axis1_stretch, boresight_from_picks, centring_move,
                          ideal_calibration, image_jog_rates, jacobian, measured_cos_dec,
                          pixels_per_deg, scale_check)

GUIDE = {"pixel_um": 5.6, "bin": 1, "focal_length_mm": 16, "width": 640, "height": 480}
MAIN = {"pixel_um": 2.9, "bin": 1, "focal_length_mm": 750, "width": 1936, "height": 1096}


def test_scale_check_accepts_a_correct_calibration():
    for cfg in (GUIDE, MAIN):
        J = np.array(ideal_calibration(cfg, rotation_deg=17.0)["J"])
        expected, measured, factor = scale_check(J, cfg, dec_cal=0.0)
        assert np.allclose(measured, expected, rtol=1e-6)
        assert np.allclose(factor, 1.0, rtol=1e-6)


def test_scale_check_reports_an_undermoving_axis():
    """The real symptom seen on the mount: both cameras measured ~3x too few px per degree."""
    short = 3.0
    J = np.array(ideal_calibration(GUIDE, rotation_deg=-4.3)["J"]) / short
    expected, measured, factor = scale_check(J, GUIDE, dec_cal=0.0)
    assert np.allclose(measured, expected / short, rtol=1e-6)
    assert np.allclose(factor, short, rtol=1e-6)      # multiply gear_ratio by 3
    assert abs(expected - 49.9) < 0.5                  # 16 mm, 5.6 um


def test_scale_check_handles_per_axis_differences():
    J = np.array(ideal_calibration(MAIN)["J"])
    J[:, 0] /= 2.0            # axis1 geared differently from axis2
    _, _, factor = scale_check(J, MAIN, dec_cal=0.0)
    assert np.allclose(factor, [2.0, 1.0], rtol=1e-6)


def test_scale_check_undoes_the_cos_dec_scaling():
    """Axis1 image motion shrinks with cos(dec); the check must not blame the gearing for it."""
    cfg, dec = GUIDE, 60.0
    J = np.array(ideal_calibration(cfg)["J"])
    J[:, 0] *= np.cos(np.radians(dec))
    _, _, factor = scale_check(J, cfg, dec_cal=dec)
    assert np.allclose(factor, 1.0, rtol=1e-6)


def test_image_jog_maps_each_arrow_to_its_own_direction():
    """Both arrow pairs must move the image along different directions, whatever the rotation."""
    cal = ideal_calibration(GUIDE, rotation_deg=25.0, dec_cal=0.0)
    right = image_jog_rates(cal, axis2=40.0, jog=(1, 0), speed=0.1)
    up = image_jog_rates(cal, axis2=40.0, jog=(0, 1), speed=0.1)
    J = jacobian(cal, 40.0)
    dx, dy = J @ right, J @ up
    assert dx[0] > 0 and abs(dx[1]) < 0.05 * abs(dx[0])      # right: +x in the image only
    assert dy[1] < 0 and abs(dy[0]) < 0.05 * abs(dy[1])      # up: -y in the image only
    assert max(abs(right)) == pytest.approx(0.1) and max(abs(up)) == pytest.approx(0.1)


def test_image_jog_gives_up_near_the_pole():
    """At axis2 = 90 the RA axis only rotates the field, so there is no sensible mapping."""
    cal = ideal_calibration(GUIDE, rotation_deg=25.0, dec_cal=0.0)
    assert image_jog_rates(cal, axis2=90.0, jog=(1, 0), speed=0.1) is None
    assert image_jog_rates(cal, axis2=88.0, jog=(0, 1), speed=0.1) is None
    assert image_jog_rates(cal, axis2=60.0, jog=(1, 0), speed=0.1) is not None


def test_scale_check_is_blind_at_the_pole():
    """Calibrated at dec 90 the clamps cancel, so the check cannot see an axis1 problem at all -
    which is why calibration warns you to move away from the pole instead."""
    cal = ideal_calibration(GUIDE)
    J = jacobian(cal, axis2=90.0)          # what the tracker would use if calibrated at home
    _, _, factor = scale_check(J, GUIDE, dec_cal=90.0)
    assert np.allclose(factor, 1.0, rtol=1e-6)


def test_centring_move_brings_the_target_to_the_boresight():
    cal = ideal_calibration(GUIDE, rotation_deg=31.0)
    for axis2 in (35.0, 130.0):
        offset = np.array([0.6, -0.4])                       # where the object sits, in axis degrees
        px = np.array(cal["boresight"]) - jacobian(cal, axis2) @ offset
        move = centring_move(cal, axis2, px)
        assert np.allclose(move, offset, rtol=1e-6)          # moving by it lands on the boresight


def test_centring_move_refuses_absurd_answers():
    """A misdetection or bad calibration must not trigger a huge slew."""
    cal = ideal_calibration(GUIDE)
    px = np.array(cal["boresight"]) - jacobian(cal, 40.0) @ np.array([50.0, 0.0])
    assert centring_move(cal, 40.0, px) is None
    assert centring_move(cal, 40.0, px, max_deg=90) is not None


def test_centring_move_can_aim_at_the_frame_centre():
    """The boresight is where the main camera looks; the frame centre is a different place."""
    cal = ideal_calibration(GUIDE, rotation_deg=8.0)
    cal["boresight"] = [280.0, 210.0]                    # a real guide/main offset
    centre = [(GUIDE["width"] - 1) / 2, (GUIDE["height"] - 1) / 2]
    px = np.array([350.0, 180.0])                        # where the object currently sits
    to_bore = centring_move(cal, 40.0, px)
    to_frame = centring_move(cal, 40.0, px, target_px=centre)
    assert not np.allclose(to_bore, to_frame)
    J = jacobian(cal, 40.0)
    assert np.allclose(px + J @ to_bore, cal["boresight"], atol=1e-6)
    assert np.allclose(px + J @ to_frame, centre, atol=1e-6)


def test_measure_backlash_flags_slack_it_cannot_resolve():
    """A step smaller than the slack tells you only 'at least this much' - say so."""
    from issctl.calib import measure_backlash
    from issctl.camera import SimCamera
    from issctl.clock import Clock
    from issctl.config import load_config
    from issctl.mount import SimMount
    from issctl.sim import CalibWorld

    cfg = load_config()
    clock = Clock()
    mount = SimMount(cfg, {"index": [0.0, 90.0]}, clock, start=[20.0, 40.0], backlash=[0.0, 0.5])
    mount.query()
    cam = SimCamera("guide", cfg["cameras"]["guide"], clock, CalibWorld(cfg, mount, decoys=())).start()
    try:
        cal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=12.0)
        measured, saturated = measure_backlash(mount, cam, cal, axis=1, step_deg=0.2,
                                               log=lambda *a: None)
    finally:
        cam.stop()
    assert saturated and measured >= 0.15           # "at least the step", not the true 0.5


def test_measure_backlash_recovers_simulated_lost_motion():
    """A simulated mount with known slack must measure as having that much."""
    from issctl.calib import measure_backlash
    from issctl.clock import Clock
    from issctl.config import load_config
    from issctl.sim import CalibWorld
    from issctl.camera import SimCamera
    from issctl.mount import SimMount

    cfg = load_config()
    slack = 0.05                                    # 3 arcmin of lost motion on axis2
    clock = Clock()
    mount = SimMount(cfg, {"index": [0.0, 90.0]}, clock, start=[20.0, 40.0], backlash=[0.0, slack])
    mount.query()
    world = CalibWorld(cfg, mount, decoys=())
    cam = SimCamera("guide", cfg["cameras"]["guide"], clock, world).start()
    try:
        cal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=12.0)
        measured, saturated = measure_backlash(mount, cam, cal, axis=1, step_deg=0.4,
                                               log=lambda *a: None)
    finally:
        cam.stop()
    assert abs(measured - slack) < 0.02             # within ~1 arcmin
    assert not saturated                            # 0.4 deg of travel resolves 0.05 deg of slack


def test_orthogonalise_squares_up_a_skewed_matrix():
    from issctl.calib import axes_angle, orthogonalise

    cal = ideal_calibration(MAIN, rotation_deg=-148.2)
    J = np.array(cal["J"])
    tilt = np.radians(11.0)                                   # tilt the Dec column by 11 deg
    rot = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
    skew = J.copy()
    skew[:, 1] = rot @ J[:, 1]
    assert abs(axes_angle(skew) - 90) == pytest.approx(11.0, abs=0.01)
    fixed = orthogonalise(skew)
    assert abs(axes_angle(fixed) - 90) < 1e-6
    # column lengths are preserved, and each direction moves by about half the error
    for col in (0, 1):
        assert np.linalg.norm(fixed[:, col]) == pytest.approx(np.linalg.norm(skew[:, col]))
        turn = abs(np.degrees(np.arctan2(fixed[1, col], fixed[0, col])
                              - np.arctan2(skew[1, col], skew[0, col])))
        assert turn == pytest.approx(5.5, abs=0.1)            # half the error each


def test_calibrate_against_reference_cancels_backlash():
    """The narrow camera is calibrated from the wide one, so mount slack cancels in the ratio."""
    from issctl.calib import calibrate_against
    from issctl.camera import SimCamera
    from issctl.clock import Clock
    from issctl.config import load_config
    from issctl.mount import SimMount
    from issctl.sim import CalibWorld

    cfg = load_config()
    clock = Clock()
    # slack far bigger than the narrow camera's own calibration step would be
    mount = SimMount(cfg, {"index": [0.0, 90.0]}, clock, start=[20.0, 40.0], backlash=[0.03, 0.03])
    mount.query()
    world = CalibWorld(cfg, mount, decoys=())
    guide = SimCamera("guide", cfg["cameras"]["guide"], clock, world).start()
    main = SimCamera("main", cfg["cameras"]["main"], clock, world).start()
    try:
        J = calibrate_against(mount, main, guide, world.true_cal["guide"], step_deg=0.02,
                              log=lambda *a: None)
    finally:
        guide.stop()
        main.stop()
    truth = np.array(world.true_cal["main"]["J"])
    assert np.allclose(np.linalg.norm(J[:, 1]), np.linalg.norm(truth[:, 1]), rtol=0.1)
    angle = np.degrees(np.arctan2(J[1, 0], J[0, 0]) - np.arctan2(truth[1, 0], truth[0, 0]))
    assert abs((angle + 180) % 360 - 180) < 5


def test_calibration_order_puts_the_narrow_camera_first_when_it_can():
    """With a stored reference the narrow camera goes first: its small moves keep every target in
    frame. Without one the wide camera must go first, whatever that costs."""
    from issctl.calib import calibration_order

    cams = {"guide": GUIDE, "main": MAIN}
    order, ref = calibration_order(cams, existing={"guide": {"J": [[1, 0], [0, 1]]}})
    assert order == ["main", "guide"] and ref == "guide"

    order, ref = calibration_order(cams, existing={})
    assert order == ["guide", "main"] and ref == "guide"

    # a stale entry without a matrix is not a reference
    order, _ = calibration_order(cams, existing={"guide": {"boresight": [1, 2]}})
    assert order == ["guide", "main"]


# ---- the axis1 column has to obey geometry, whatever the tracker reported ----------------

def test_measured_cos_dec_recovers_the_declination_from_the_image_alone():
    """The ratio of the two columns is an estimate of cos(dec) that owes nothing to the mount."""
    for dec_cal in (0.0, 30.0, 58.0, 81.5):
        for cfg in (GUIDE, MAIN):
            cal = ideal_calibration(cfg, rotation_deg=-22.0, dec_cal=dec_cal)
            assert measured_cos_dec(cal) == pytest.approx(np.cos(np.radians(dec_cal)), rel=1e-6)
            assert axis1_plausible(cal)


def test_a_mount_that_does_not_know_where_it_is_does_not_block_centring():
    """Regression: the guard used to compare the measurement against the mount's claim and refuse.

    On a hand-pushed mount the mount is the one that is wrong. Its matrix is still exactly right
    where it was measured, because jacobian() only ever applies the CHANGE in declination.
    """
    real = ideal_calibration(GUIDE, rotation_deg=12.0, dec_cal=24.6)
    cal = dict(real, dec_cal=83.0)               # what an unsynced counter claimed at the time
    assert measured_cos_dec(cal) == pytest.approx(0.909, abs=0.005)
    assert axis1_plausible(cal)                  # 0.909 is a possible cos(dec); nothing is wrong
    axis2 = geo.dec_to_axis2(83.0) if hasattr(geo, "dec_to_axis2") else 83.0
    assert axis1_stretch(cal, axis2) == pytest.approx(1.0, abs=0.01)   # it has not slewed
    assert image_jog_rates(cal, axis2, (1, 0), 0.1) is not None


def test_a_column_longer_than_axis2_is_impossible_at_any_declination():
    """cos(dec) cannot exceed 1. This is the main camera's 7.2x: axis1 rotated the field about a
    point outside its 0.2 deg frame and the tracker read the swing as a shift."""
    cal = ideal_calibration(MAIN, dec_cal=81.5)
    J = np.array(cal["J"])
    J[:, 0] *= 7.2 / measured_cos_dec(cal)
    bad = dict(cal, J=J.tolist())
    assert not axis1_plausible(bad)
    assert image_jog_rates(bad, 40.0, (1, 0), 0.1) is None


def test_stretching_a_pole_measured_column_is_what_made_centring_run_away():
    """Regression for a real failure: guide centring diverged 70 -> 194 -> 481 px.

    The axis2 half of every correction was right; the axis1 half asked for 3.7 deg where the
    error was worth 1, because the matrix was measured at dec 81.5 and then used far away from
    there. The column itself is plausible - it is the extrapolation that is not.
    """
    dec_cal, axis2_now = 81.5, 20.0
    truth = ideal_calibration(GUIDE, dec_cal=dec_cal)
    J = np.array(truth["J"])
    J[:, 0] *= 0.27                                   # what the tracker actually reported there
    cal = dict(truth, J=J.tolist())

    assert axis1_plausible(cal)                       # nothing impossible about it on its own
    assert axis1_stretch(cal, axis2_now) > 3.0        # but it is being stretched ~6x
    assert axis1_stretch(cal, geo.dec_to_axis2(dec_cal)
                         if hasattr(geo, "dec_to_axis2") else dec_cal) < 1.2

    offset = np.array([1.0, 0.0])                     # one degree of axis1 from the boresight
    px = np.array(cal["boresight"]) - jacobian(truth, axis2_now) @ offset
    move = centring_move(cal, axis2_now, px, max_deg=90)
    assert move[0] == pytest.approx(1 / 0.27, rel=0.02)   # ~3.7x too far, exactly as logged
    assert image_jog_rates(cal, axis2_now, (1, 0), 0.1) is None    # arrows fall back to raw axes


# ---- a boresight you set by hand --------------------------------------------------------

def test_boresight_from_picks_is_exactly_the_guide_pick_when_main_is_centred():
    """No matrix is involved in that case, which is why it is the recipe to give the user."""
    main = ideal_calibration(MAIN, rotation_deg=-148.0, dec_cal=40.0)
    guide = ideal_calibration(GUIDE, rotation_deg=31.0, dec_cal=40.0)
    b, carried = boresight_from_picks(main, guide, main["boresight"], [312.0, 197.0])
    assert np.allclose(b, [312.0, 197.0])
    assert carried == pytest.approx(0.0)


def test_boresight_from_picks_carries_an_off_centre_pick_across():
    main = ideal_calibration(MAIN, rotation_deg=-148.0, dec_cal=40.0)
    guide = ideal_calibration(GUIDE, rotation_deg=31.0, dec_cal=40.0)
    axis2 = 50.0
    # one object, seen in both: put it a little off the main boresight and work out both pixels
    off = np.array([0.05, -0.03])
    px_main = np.array(main["boresight"]) - jacobian(main, axis2) @ off
    px_guide = np.array([300.0, 200.0]) - jacobian(guide, axis2) @ off
    b, carried = boresight_from_picks(main, guide, px_main, px_guide)
    assert np.allclose(b, [300.0, 200.0], atol=1e-6)     # back to where main's centre looks
    assert carried > 1.0                                 # and it says how much came from the maths


def test_boresight_from_picks_ignores_where_the_mount_thinks_it_is():
    """The whole point for a hand-pushed mount: the axis-to-sky factor is shared by both cameras
    and cancels in the ratio, so an unsynced (or plain wrong) declination changes nothing.

    The two matrices are deliberately given DIFFERENT dec_cal here - calibrating one camera alone
    later leaves them that way - which is exactly when normalising both before dividing matters.
    """
    main = ideal_calibration(MAIN, rotation_deg=-148.0, dec_cal=37.0)
    guide = ideal_calibration(GUIDE, rotation_deg=31.0, dec_cal=61.0)
    pm, pg = [820.0, 610.0], [301.0, 244.0]
    first = boresight_from_picks(main, guide, pm, pg)
    assert first[1] > 0                                  # the picks really do disagree
    for shifted in (dict(main, dec_cal=0.0), dict(main, dec_cal=-80.0)):
        # re-express main's own matrix at a different declination: same camera, same physics
        J = np.array(main["J"], dtype=float)
        J[:, 0] *= (np.cos(np.radians(shifted["dec_cal"])) / np.cos(np.radians(37.0)))
        again = boresight_from_picks(dict(shifted, J=J.tolist()), guide, pm, pg)
        assert np.allclose(again[0], first[0])
        assert again[1] == pytest.approx(first[1])


def test_measure_waits_for_a_slow_camera():
    """At a star exposure of 1 s the camera gives one frame a second. measure() asked for ten
    frames in five seconds, so every measurement failed and calibration slewed to 'steer back'
    a target that had never left the field."""
    import threading
    import time

    from issctl.calib import measure

    class Slow:
        fps, manual, gate = 4.0, True, None

        def __init__(self):
            self.seq = 0
            threading.Thread(target=self._run, daemon=True).start()

        def _run(self):
            for _ in range(40):
                time.sleep(0.25)
                self.seq += 1

        def latest(self):
            det = type("D", (), {"x": 10.0, "y": 20.0})()
            return None, det, self.seq

    assert np.allclose(measure(Slow(), n=10, timeout=1.0), [10.0, 20.0])


def test_axis1_column_flips_sign_across_the_meridian():
    """Measured on one side of the pier, used on the other: the tube turns the other way under
    the same axis1 step, and centring without the sign flip ran away (330', then 602')."""
    from issctl.model import PointingModel
    from issctl.solve import SimSolution

    cam = {"pixel_um": 3.75, "bin": 1, "focal_length_mm": 15.5, "width": 1280, "height": 960}
    true = ideal_calibration(cam, 20.0)
    model = PointingModel()
    centre = np.array([639.5, 479.5])

    def measured(a1, a2, eps=0.05):
        sol = SimSolution(true, *model.camera_frame(a1, a2), 0.0, 1280, 960)
        seen = sol.hadec(centre)
        cols = []
        for d in ([eps, 0.0], [0.0, eps]):
            moved = SimSolution(true, *model.camera_frame(a1 + d[0], a2 + d[1]), 0.0, 1280, 960)
            cols.append((moved.pixel_hadec(*seen) - centre) / eps)
        return np.column_stack(cols)

    stored = {"J": measured(20.0, 40.0).tolist(), "dec_cal": 40.0, "axis2_cal": 40.0}
    for a1, a2 in ((-160.0, 140.0), (-150.0, 125.0), (30.0, 55.0)):
        assert np.allclose(jacobian(stored, a2), measured(a1, a2), atol=1.0), (a1, a2)
    # a calibration from before axis2_cal was kept behaves exactly as it always did
    legacy = {"J": stored["J"], "dec_cal": 40.0}
    assert np.allclose(jacobian(legacy, 140.0), np.array(stored["J"]), atol=1e-9)


def test_boresight_click_snaps_to_the_bright_spot_nearby():
    from issctl.detect import snap

    img = np.random.default_rng(0).normal(20, 3, (480, 640)).clip(0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:480, 0:640]
    for (cx, cy), amp in (((300.4, 200.7), 180.0), ((360.0, 260.0), 250.0)):
        img = np.clip(img + amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 4.5), 0, 255).astype(np.uint8)
    # clicked 6 px off the fainter star: that one, not the brighter one further away
    x, y = snap(img, 305.0, 196.0, radius=25)
    assert abs(x - 300.4) < 0.3 and abs(y - 200.7) < 0.3
    assert snap(img, 100.0, 100.0, radius=25) is None      # empty sky: nothing to snap to


def _frame_with(blobs, shape=(1096, 1936), seed=3):
    rng = np.random.default_rng(seed)
    img = rng.normal(20, 3, shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (cx, cy), amp, s in blobs:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_detect_in_a_gate_matches_the_whole_frame():
    """Only a window round the gate is processed now; the answer must not change."""
    from issctl.detect import detect

    img = _frame_with([((901.3, 503.7), 150.0, 2.5)])
    for bayer in (False, True):
        full = detect(img, 5.0, 20, bayer)
        for gate in ((900.0, 500.0, 60.0), (903.0, 499.0, 25.0)):   # an odd centre: Bayer phase
            win = detect(img, 5.0, 20, bayer, gate)
            assert abs(win.x - full.x) < 0.2 and abs(win.y - full.y) < 0.2, (bayer, gate)
        assert abs(full.x - 901.3) < 0.3 and abs(full.y - 503.7) < 0.3


def test_detect_keeps_to_its_gate_even_next_to_something_brighter():
    from issctl.detect import detect

    img = _frame_with([((600.0, 400.0), 60.0, 2.0), ((700.0, 400.0), 250.0, 3.0)])
    got = detect(img, 5.0, 3, False, (600.0, 400.0, 40.0))
    assert abs(got.x - 600.0) < 0.3
    assert detect(img, 5.0, 3, False).x == pytest.approx(700.0, abs=0.3)      # no gate: brightest
    assert detect(img, 5.0, 3, False, (300.0, 300.0, 40.0)) is None           # empty gate


def test_detect_edge_margin_is_the_frame_edge_not_the_window_edge():
    from issctl.detect import detect

    img = _frame_with([((12.0, 500.0), 200.0, 2.0)])
    assert detect(img, 6.0, 12, False, edge_margin=20) is None      # (12 px: no noise blobs)
    assert detect(img, 6.0, 12, False, (12.0, 500.0, 40.0), edge_margin=20) is None
    assert detect(img, 6.0, 12, False, (12.0, 500.0, 40.0)) is not None
