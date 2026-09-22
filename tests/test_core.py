import time

import numpy as np
import pytest

from issctl import geometry as geo
from issctl.calib import axes_offset_from_pixel, ideal_calibration, jacobian
from issctl.clock import Clock
from issctl.config import load_config
from issctl.detect import detect
from issctl.mount import SimMount
from issctl.predict import (SIM_TLE, Site, find_passes, illumination, make_satellite, plan_pass,
                            shadow_events, sun_vector, time_to_unix, unix_to_time)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_altaz_hadec_roundtrip():
    rng = np.random.default_rng(0)
    for lat in (50.45, -33.9, 10.0):
        alt = rng.uniform(-10, 89, 200)
        az = rng.uniform(0, 360, 200)
        ha, dec = geo.altaz_to_hadec(alt, az, lat)
        alt2, az2 = geo.hadec_to_altaz(ha, dec, lat)
        assert np.allclose(alt, alt2, atol=1e-9)
        assert np.allclose(geo.wrap180(az - az2), 0, atol=1e-7)


def test_hadec_known_points():
    lat = 50.0
    ha, dec = geo.altaz_to_hadec(90.0 - lat, 180.0, lat)   # meridian, equator
    assert abs(ha) < 1e-9 and abs(dec) < 1e-9
    ha, dec = geo.altaz_to_hadec(0.0, 90.0, lat)           # east point
    assert abs(ha + 90) < 1e-9 and abs(dec) < 1e-9
    ha, dec = geo.altaz_to_hadec(lat, 0.0, lat)            # pole
    assert abs(dec - 90) < 1e-9


def test_mount_axes_roundtrip():
    for side in geo.SIDES:
        ha = np.array([-150, -60, 0, 45, 170.0])
        dec = np.array([-20, 10, 45, 60, 80.0])
        a1, a2 = geo.hadec_to_axes(ha, dec, side)
        ha2, dec2 = geo.axes_to_hadec(a1, a2)
        assert np.allclose(geo.wrap180(ha - ha2), 0) and np.allclose(dec, dec2)


def test_calibration_inverse():
    cal = ideal_calibration({"pixel_um": 3.75, "bin": 1, "focal_length_mm": 180, "width": 1280, "height": 960},
                            rotation_deg=25)
    for a2 in (30.0, 120.0):
        d = np.array([0.1, -0.2])
        px = np.array(cal["boresight"]) + jacobian(cal, a2) @ d
        assert np.allclose(axes_offset_from_pixel(cal, a2, px), -d)


def test_detect_blob():
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(30, 4, (960, 1280)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:960, 0:1280]
    img = np.clip(img + 150 * np.exp(-((xx - 811.3) ** 2 + (yy - 402.7) ** 2) / (2 * 2.0 ** 2)), 0, 255).astype(np.uint8)
    img[100, 100] = 255  # hot pixel
    d = detect(img, sigma=6, min_area=3)
    assert d is not None and abs(d.x - 811.3) < 0.7 and abs(d.y - 402.7) < 0.7
    assert detect(img, gate=(100, 100, 50)) is None


def test_detect_bayer_extended():
    rng = np.random.default_rng(4)
    img = np.clip(rng.normal(20, 3, (1096, 1936)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:1096, 0:1936]
    img = np.clip(img + 120 * np.exp(-((xx - 500.0) ** 2 + (yy - 700.0) ** 2) / (2 * 15.0 ** 2)), 0, 255).astype(np.uint8)
    d = detect(img, sigma=5, min_area=20, bayer=True)
    assert d is not None and abs(d.x - 500) < 3 and abs(d.y - 700) < 3


def test_sim_mount_move(cfg):
    clock = Clock()
    m = SimMount(cfg, {}, clock)
    m.query()
    target = m.position() + [1.0, -0.5]
    end = m.move_to(target, timeout=10)
    assert np.all(np.abs(end - target) < 0.005)


def test_illumination_geometry(cfg):
    sat = make_satellite(SIM_TLE)
    t0 = time_to_unix(sat.epoch)
    t = t0 + np.arange(0, 5600, 10.0)  # one full orbit
    lit = illumination(sat, t)
    assert np.all((lit >= 0) & (lit <= 1))
    assert lit.max() > 0.99 and lit.min() < 0.01           # a low orbit has day and night
    assert 0.25 < np.mean(lit > 0.5) < 0.85                 # roughly a third in shadow
    # sunward of Earth the satellite is always fully lit
    r = sat.at(unix_to_time(t)).position.km.T
    u_sun, _ = sun_vector(t)
    assert np.all(lit[np.sum(r * u_sun, axis=1) > 0] == 1.0)
    # entering/leaving shadow takes a few seconds, not instants
    fine = t0 + np.arange(0, 5600, 0.5)
    lf = illumination(sat, fine)
    partial = np.sum((lf > 0.02) & (lf < 0.98)) * 0.5
    assert 2 < partial < 120


def test_shadow_events():
    t = np.arange(0, 100, 1.0)
    lit = np.where((t > 30) & (t < 70), 0.0, 1.0)
    ev = shadow_events(t, lit)
    assert [e[1] for e in ev] == ["enters", "leaves"]
    assert abs(ev[0][0] - 31) < 1.5 and abs(ev[1][0] - 70) < 1.5


def test_pass_planning(cfg):
    site = Site(cfg)
    sat = make_satellite(SIM_TLE)
    t0 = time_to_unix(sat.epoch)
    passes = find_passes(sat, site, t0, 36)
    assert passes
    p = max(passes, key=lambda p: p["max_alt"])
    traj, rep = plan_pass(sat, site, cfg["mount"], p["rise"], p["set"])
    assert rep["tracked_s"] > 0
    pos, vel = traj.at((traj.t_start + traj.t_end) / 2)
    assert np.all(np.isfinite(pos)) and np.all(np.abs(vel) < 3)


def test_pose_choice_prefers_legal_poses(cfg):
    mount_cfg = dict(cfg["mount"], axis1_hour_limit=120, axis2_limits=[-10, 190])
    # a target reachable both ways: with no current position, take the lower counterweight
    best, options = geo.choose_pose(-60.0, 30.0, mount_cfg)
    assert best["ok"] and abs(best["axes"][0]) == min(abs(o["axes"][0]) for o in options)

    # tightening the Dec travel rules out the pose that swings past the pole
    tight = dict(mount_cfg, axis2_limits=[0, 95])
    best, options = geo.choose_pose(112.0, 19.0, tight)
    assert best is None or best["axes"][1] <= 95


def test_pose_choice_avoids_a_pointless_flip(cfg):
    """From a pose near axis2=46 (Capella side), staying on that side must win over flipping."""
    mount_cfg = dict(cfg["mount"], axis1_hour_limit=170, axis2_limits=[-10, 190])
    current = [-22.9, 46.1]
    best, _ = geo.choose_pose(-100.0, 40.0, mount_cfg, current=current)
    assert abs(best["axes"][1] - current[1]) < 90     # no swing through the pole
    assert best["travel"] == min(
        max(abs(geo.wrap180(o["axes"][0] - current[0])), abs(o["axes"][1] - current[1]))
        for o in geo.choose_pose(-100.0, 40.0, mount_cfg, current=current)[1] if o["ok"])


def test_position_survives_a_restart(cfg, tmp_path):
    """The Uno's counters reset when the port opens, so the position has to come from disk."""
    from issctl.cli import remember_position, restore_position
    from issctl.config import load_state, save_state

    clock = Clock()
    state_path = tmp_path / "state.json"
    mount = SimMount(cfg, {}, clock)
    mount.query()
    mount.move_to(mount.position() + [12.0, -20.0], timeout=30)
    parked = mount.position()
    state = {}
    remember_position(state, state_path, mount)
    assert state["position"] == pytest.approx(list(parked), abs=1e-6)

    # a new session: fresh mount, counters at zero, state read back from disk
    reloaded = load_state(state_path)
    restarted = SimMount(cfg, {}, clock)
    assert not np.allclose(restarted.position(), parked)     # would otherwise claim it is at home
    restore_position(reloaded, restarted, log=lambda *a: None)
    assert np.allclose(restarted.position(), parked, atol=1e-6)


def test_restore_position_is_a_no_op_without_a_saved_one(cfg):
    from issctl.cli import restore_position

    mount = SimMount(cfg, {}, Clock())
    before = mount.position()
    restore_position({}, mount, log=lambda *a: None)
    assert np.allclose(mount.position(), before)


def test_target_field_accepts_several_notations(cfg):
    """Names, decimal RA/Dec, sexagesimal and a fixed alt/az direction all resolve."""
    from issctl.predict import target_hadec

    site = Site(cfg)
    t = 1_780_000_000.0
    named = target_hadec("vega", site, t)
    for text in ("18:36:56 +38:47:01", "18h36m56s +38d47m01s"):
        got = target_hadec(text, site, t)
        assert abs(got[0] - named[0]) < 0.05 and abs(got[1] - named[1]) < 0.05

    decimal = target_hadec("18.6 38.8", site, t)
    assert abs(decimal[0] - named[0]) < 0.5      # 18.6 h is a rounded Vega
    assert target_hadec("18.6, 38.8", site, t) == decimal

    alt, az = 30.0, 180.0                        # a fixed direction, e.g. a landmark
    ha, dec, got_alt, got_az = target_hadec(f"altaz {alt} {az}", site, t)
    assert (got_alt, got_az) == (alt, az)
    assert np.allclose(geo.hadec_to_altaz(ha, dec, site.lat), (alt, az), atol=1e-6)

    with pytest.raises(ValueError):
        target_hadec("nonsense", site, t)


def _frame_with_blob(x, y, sigma=3.0, amp=150, shape=(1096, 1936), seed=7):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(20, 3, shape), 0, 255)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_detect_rejects_blobs_on_the_frame_edge():
    """Vignetting and out-of-focus scenery light up the border; that is not a target."""
    img = _frame_with_blob(1930, 1090)                      # bottom-right corner, like the real case
    assert detect(img, sigma=5, min_area=20) is not None    # found without the guard
    assert detect(img, sigma=5, min_area=20, edge_margin=8) is None


def test_detect_rejects_sprawling_regions():
    img = _frame_with_blob(900, 500, sigma=80, amp=200)     # a big soft glow, not a point source
    assert detect(img, sigma=5, min_area=20) is not None
    assert detect(img, sigma=5, min_area=20, max_area=20000) is None
    # a real target of sensible size still passes both guards
    img = _frame_with_blob(900, 500, sigma=6)
    d = detect(img, sigma=5, min_area=20, max_area=20000, edge_margin=8)
    assert d is not None and abs(d.x - 900) < 2 and abs(d.y - 500) < 2
