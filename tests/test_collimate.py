import math

import numpy as np
import pytest

from issctl.clock import Clock
from issctl.collimate import (Collimator, SimCollimationCamera, coma_tolerance_mm, fit_circle,
                              prepare, robust_fit, rotation_axis, snap)
from issctl.config import load_config

CCFG = {"focal_mm": 750.0, "aperture_mm": 150.0, "lens_mm": 12.0}


def sim():
    cfg = load_config()
    return SimCollimationCamera("guide", cfg["cameras"]["guide"], Clock())


def radii(cam):
    r_s = cam.r_s
    return {"S": r_s, "P": 0.965 * r_s, "D": 0.14 * r_s, "C": 0.04 * r_s}


def rough_clicks(c, r, angles=(20, 140, 260), err=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return [(c[0] + (r + rng.uniform(-err, err)) * math.cos(math.radians(a)),
             c[1] + (r + rng.uniform(-err, err)) * math.sin(math.radians(a))) for a in angles]


def test_fit_circle_exact_and_robust():
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts = np.column_stack([100 + 40 * np.cos(th), 80 + 40 * np.sin(th)])
    assert fit_circle(pts) == pytest.approx((100, 80, 40), abs=1e-6)
    bad = pts.copy()
    bad[:6] += 15.0                       # a clip / a vane: points off the circle
    (cx, cy, r), keep, _ = robust_fit(bad)
    assert (cx, cy, r) == pytest.approx((100, 80, 40), abs=0.05)
    assert keep.sum() == 54


@pytest.mark.parametrize("name", ["S", "P", "D", "C"])
def test_snap_finds_each_circle_from_three_rough_clicks(name):
    """Clicked a few px off the edge, the circle still lands on it - through the clips, the
    vanes, and the primary's edge hiding behind the secondary's on one side."""
    cam = sim()
    img = prepare(cam.frame())
    truth = cam.truth()[name]
    r = radii(cam)[name]
    err = min(3.0, 0.2 * r)
    circle, pol, q, locked = snap(img, rough_clicks(truth, r, err=err))
    assert locked
    # S and P nearly touch on one side here, and lean ~0.7 px towards each other there
    assert math.hypot(circle[0] - truth[0], circle[1] - truth[1]) < (0.9 if name in "SP" else 0.6)
    assert circle[2] == pytest.approx(r, abs=1.5)


def test_rotation_axis_from_a_180_deg_turn():
    cam = sim()
    a = cam.frame()
    cam.sim_action("sim_turn", {})
    b = cam.frame()
    (x, y), resp = rotation_axis(a, b)
    assert resp > 0.2
    assert (x, y) == pytest.approx(cam.axis, abs=0.5)
    # the frame centre is clearly wrong here, so the test means something
    assert math.hypot(cam.axis[0] - (cam.width - 1) / 2, cam.axis[1] - (cam.height - 1) / 2) > 10


def pick(coll, cam, name):
    truth = cam.truth()[name]
    r = radii(cam)[name]
    coll.action("pick", {"what": name})
    for x, y in rough_clicks(truth, r, err=min(3.0, 0.2 * r)):
        coll.action("click", {"fx": x / cam.width, "fy": y / cam.height})
    coll.action("done", {})


def test_collimator_readings_follow_the_screws(tmp_path):
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1          # one frame, no capture thread
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    for n in "SPDC":
        pick(coll, cam, n)
    # the turn test puts the axis where it really is, and the circles where they went
    coll.action("turn_first", {})
    cam.sim_action("sim_turn", {})
    cam._frame, cam._seq = cam.frame(), 2
    coll.action("turn_second", {})
    coll.update()
    assert coll.axis == pytest.approx(cam.axis, abs=0.5)
    t = cam.truth()
    steps = {s["key"]: s for s in coll.steps()}
    # S and P each lean ~0.7 px towards the other where the primary's edge runs into the
    # secondary's and the two merge in the blur: 0.4% of the radius, a quarter of "good"
    for key, a, b, tol in (("focuser", "axis", "S", 1.0), ("secondary", "S", "P", 1.5),
                           ("primary", "C", "D", 0.5)):
        want = t[b] - t[a]
        assert (steps[key]["dx"], steps[key]["dy"]) == pytest.approx(tuple(want), abs=tol), key
    assert steps["primary"]["grade"] == "adjust"
    # turning the primary's screws moves D onto C, and the reading follows the picture
    for _ in range(12):
        cam.sim_action("sim_nudge", {"what": "primary"})
    cam._frame, cam._seq = cam.frame(), 3
    coll.update()
    assert {s["key"]: s for s in coll.steps()}["primary"]["px"] < 1.0
    # and it all comes back after a restart
    again = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    assert again.axis == pytest.approx(coll.axis)
    assert set(again.circles) == set("SPDC")


def test_coma_tolerance():
    assert coma_tolerance_mm(750, 150) == pytest.approx(0.89, abs=0.01)


def test_auto_exposure_brings_a_clipped_picture_back(tmp_path):
    """Started at the console's night setting, the daytime tube is all white - and an all-white
    frame stretched for display is all black, with no edge to find."""
    cam = sim()
    cam.set_exposure(506.0)
    cam.start()
    try:
        coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
        coll._auto_exposure()
        assert coll.clipped < 0.002
        assert 0.6 < coll.peak < 0.9, coll.msg
        assert cam.exposure_ms < 506.0
        assert coll.auto is None
    finally:
        cam.stop()


def test_a_click_far_from_a_locked_edge_is_refused(tmp_path):
    """Done not pressed, the next circle's clicks used to land in this one and drag it away."""
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    coll.action("pick", {"what": "P"})
    for x, y in rough_clicks(cam.truth()["P"], radii(cam)["P"]):
        coll.action("click", {"fx": x / cam.width, "fy": y / cam.height})
    r = coll.circles["P"]["r"]
    t = cam.truth()["D"]
    coll.action("click", {"fx": t[0] / cam.width, "fy": t[1] / cam.height})    # meant for D
    assert "not used" in coll.msg
    assert coll.circles["P"]["r"] == pytest.approx(r, abs=1.0)


def test_readings_need_the_measured_axis_and_fall_back_to_p_on_the_axis(tmp_path):
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    pick(coll, cam, "P")
    steps = {s["key"]: s for s in coll.steps()}
    assert not steps["focuser"]["have"] and not steps["secondary"]["have"]
    assert "turn test" in steps["secondary"]["text"]      # never against the frame centre
    coll.axis = tuple(cam.axis)
    sec = {s["key"]: s for s in coll.steps()}["secondary"]
    assert sec["have"] and sec["ref_name"] == "axis"      # no S: P on the axis instead
    want = cam.truth()["P"] - cam.truth()["axis"]
    assert (sec["dx"], sec["dy"]) == pytest.approx(tuple(want), abs=0.7)
