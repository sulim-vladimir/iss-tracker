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
