"""Calibration sanity checks - especially the scale diagnostic, which catches a wrong gear ratio."""

import numpy as np
import pytest

from issctl.calib import (centring_move, ideal_calibration, image_jog_rates, jacobian,
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


# ---- calibrating with no point source in view -------------------------------------------------

class _ScriptedCam:
    """A camera showing whatever frame the test last handed it, with a fresh sequence number."""

    def __init__(self, frame, bayer=False):
        self.frame, self.bayer = frame, bayer
        self.height, self.width = frame.shape[:2]
        self.seq = 0

    def latest(self):
        self.seq += 1
        return self.frame, None, self.seq

    def show(self, frame):
        self.frame = frame


def _windows(shape=(600, 800)):
    """Blurry bright rectangles - lit windows through a defocused scope, with no point source."""
    import cv2

    rng = np.random.default_rng(0)
    img = np.zeros(shape, np.float32)
    for x, y, w, h in [(180, 90, 140, 300), (420, 70, 150, 320), (240, 450, 100, 110)]:
        img[y:y + h, x:x + w] = 200
    return cv2.GaussianBlur(img, (0, 0), 15) + 12 + rng.normal(0, 3, shape).astype(np.float32)


def _shifted(img, dx, dy):
    import cv2

    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]))


def test_scene_tracker_measures_a_shift_with_nothing_point_like_in_frame():
    """The whole reason this exists: a scene full of structure but no blob to detect."""
    from issctl.calib import FeatureTracker
    from issctl.detect import detect

    scene = _windows()
    # Blob mode does not fail loudly here - it "finds" a target. That is the trap: the detection
    # is a whole lit window, thousands of pixels of it, and its centroid wanders with the edges
    # as they drift out of frame. A shift measured from that is worse than no shift at all.
    blob = detect(scene, sigma=6.0, min_area=20, edge_margin=3)
    assert blob is not None and blob.area > 1000, blob
    cam = _ScriptedCam(scene)
    t = FeatureTracker(cam)
    assert t.reset(), "found nothing to follow"
    cam.show(_shifted(scene, 37, -21))
    moved = t.measure(n=5) - t.origin
    assert np.allclose(moved, [37, -21], atol=0.5), moved


# There is deliberately no simulator test of scene calibration. CalibWorld renders point
# sources, not scenery: goodFeaturesToTrack finds no corners in it at all, so FeatureTracker
# correctly refuses and there is nothing to measure. The scene tracker is exercised against
# synthetic textured frames above; its accuracy on real scenery was measured separately on a
# main-camera capture (best 5 corners: 0.27 px median error over one calibration step).


def test_every_tracker_is_reachable_and_declares_itself():
    """The browser selector sends these strings straight through to make_tracker."""
    from issctl.calib import TRACKERS, BlobTracker, FeatureTracker, make_tracker

    cam = _ScriptedCam(_windows())
    want = {"blob": BlobTracker, "scene": FeatureTracker}
    assert set(TRACKERS) == set(want)
    for mode, cls in want.items():
        t = make_tracker(cam, mode)
        assert isinstance(t, cls), (mode, type(t))
        assert t.mode == mode
    assert make_tracker(cam, "nonsense").mode == "blob"    # unknown falls back, never crashes
    assert BlobTracker(cam).absolute and not FeatureTracker(cam).absolute


def test_feature_tracker_follows_only_the_region_you_click():
    """Clicking picks one patch of scenery, so depth differences across the frame cannot mix in."""
    from issctl.calib import FeatureTracker

    scene = _windows()
    cam = _ScriptedCam(scene)
    t = FeatureTracker(cam, region=(250, 150, 90))
    assert t.reset()
    pts = t.points.reshape(-1, 2)
    d = np.linalg.norm(pts - [250, 150], axis=1)
    assert d.max() <= 90 + 1, f"corners escaped the clicked circle: {d.max():.1f} px"


def test_feature_tracker_reports_losing_the_scene_instead_of_inventing_a_shift():
    """Optical flow answers confidently for points it has lost - the round trip catches that."""
    from issctl.calib import FeatureTracker

    rng = np.random.default_rng(3)
    scene = _windows()
    cam = _ScriptedCam(scene)
    t = FeatureTracker(cam)
    assert t.reset()
    cam.show(rng.normal(60, 20, scene.shape).astype(np.float32))   # scene replaced by noise
    assert t.measure(n=5) is None
    assert t.response < 0.5
