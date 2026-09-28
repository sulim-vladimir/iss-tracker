"""Calibrating by the stars: plate solving, the pointing model it feeds, and the star boresight."""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from issctl import align
from issctl import geometry as geo
from issctl import predict as pr
from issctl.calib import ideal_calibration
from issctl.clock import Clock
from issctl.config import load_config
from issctl.model import angle_arcsec
from issctl.model import unit as sky_unit
from issctl.mount import SimMount
from issctl.sim import CalibWorld, misalignment
from issctl.solve import AstrometrySolver, SimSolution, SimSolver, missing_indexes

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def cfg():
    return load_config(Path(__file__).parent.parent / "config.example.toml")


@pytest.fixture(scope="module")
def site(cfg):
    return pr.Site(cfg)


class FrameCam:
    """Just enough of a camera for solve_camera: a frame counter and a clock."""

    def __init__(self, cam_cfg, clock):
        self.name, self.cfg, self.clock = "guide", cam_cfg, clock
        self.width, self.height = cam_cfg["width"], cam_cfg["height"]
        self.exposure_ms, self.gain, self.exposure_unit = 1000.0, 100, "ms"
        self.seq = 0

    def latest_frame(self):
        self.seq += 1       # every look is a new frame: the simulated solver ignores its pixels
        return np.zeros((2, 2), np.uint8), self.clock.now(), self.seq

    def set_exposure(self, ms):
        self.exposure_ms = ms

    def set_gain(self, g):
        self.gain = g


def rig(cfg, pointing_error=(7.0, -4.0), azimuth_error=90.0, start=(20.0, 40.0)):
    """A tripod turned 90 deg from north with counters that are degrees out: the balcony."""
    clock = Clock(speed=20.0)
    mount = SimMount(cfg, {}, clock, start=list(start))
    mount.query()
    world = CalibWorld(cfg, mount, pointing_error=pointing_error,
                       model=misalignment(50.0, (2.0, -1.0), azimuth_error))
    cam = FrameCam(cfg["cameras"]["guide"], clock)
    return mount, world, cam, SimSolver(world, cam)


def true_vector(world, mount, axes):
    """Where these counters really point. The tube sits at axis + index0 while the counters read
    mech + index: a re-index moves only the second, a push by hand only the first."""
    return world.sky_model.forward(*(np.asarray(axes) + world.pointing_error
                                     + (mount.axis - mount.mech) + (mount.index0 - mount.index)))


# ---- coordinates ----

def test_catalogue_and_local_frame_round_trip(site):
    t = 1790000000.0
    ha, dec = pr.radec_to_hadec(200.9814, 54.9254, site, t)
    ra, de = pr.hadec_to_radec(ha, dec, site, t)
    assert abs(ra - 200.9814) * 3600 < 0.5 and abs(de - 54.9254) * 3600 < 0.5
    # and it is the same answer the named-target path gives
    ha2, dec2, _, _ = pr.target_hadec("Mizar", site, t)
    assert abs(ha - ha2) < 1e-6 and abs(dec - dec2) < 1e-6


def test_unknown_star_suggests_close_names(site):
    with pytest.raises(ValueError, match="mizar"):
        pr.target_hadec("Mizzar", site, 1790000000.0)


def test_sim_solution_maps_pixels_and_sky_both_ways(cfg):
    cal = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=25.0)
    p = sky_unit(30.0, 40.0)
    e1 = np.cross([0, 0, 1.0], p)
    e1 /= np.linalg.norm(e1)
    sol = SimSolution(cal, p, e1, np.cross(p, e1), 0.0, 1280, 960)
    for px in ([639.5, 479.5], [10.0, 20.0], [1200.0, 900.0]):
        assert np.allclose(sol.pixel_hadec(*sol.hadec(px)), px, atol=1e-6)
    assert sol.scale_arcsec() == pytest.approx(48.34, rel=0.01)


# ---- the star calibration ----

def test_star_calibration_measures_J_and_aligns_a_turned_tripod(cfg):
    mount, world, cam, solver = rig(cfg)
    state, warnings = {}, []
    cal = align.calibrate_on_stars(mount, cam, solver, state, log=lambda *a: None,
                                   warnings=warnings, slew_rate=3.0, settle_s=0.0)
    phys = mount.physical_at(mount.clock.now()) + world.pointing_error
    J_true = np.array(world.true_cal["guide"]["J"])
    J_true[:, 0] *= np.cos(np.radians(geo.axis2_to_dec(phys[1])))
    assert np.allclose(cal["J"], J_true, atol=0.03 * np.abs(J_true).max())
    assert not warnings
    # the Dec index went into the counters: axis2 now reads the mount's true declination
    assert mount.position()[1] == pytest.approx(phys[1], abs=0.05)
    assert cal["dec_cal"] == pytest.approx(geo.axis2_to_dec(phys[1]), abs=0.05)
    # four points within a few degrees, yet the model holds across the sky
    model = align.current_model(state)
    for a in ([-60, 30], [60, 80], [0, 10], [90, 120], [-30, 150]):
        err = angle_arcsec(model.forward(*a), true_vector(world, mount, a)) / 3600
        assert err < 0.2, (a, err)


def test_sync_after_a_hand_push_keeps_the_alignment(cfg):
    mount, world, cam, solver = rig(cfg)
    state = {}
    align.calibrate_on_stars(mount, cam, solver, state, log=lambda *a: None, slew_rate=3.0,
                             settle_s=0.0)
    n = len(align.points(state))
    # pushed by hand: the tube moves, the counters do not
    mount.axis += np.array([25.0, -12.0])
    mount.query()
    b = [(cam.width - 1) / 2, (cam.height - 1) / 2]
    sol = solver.solve(None, mount.clock.now())
    here = mount.position()
    with pytest.raises(ValueError, match="sync on stars"):
        align.add_point(state, here, *sol.hadec(b), sol.t)
    align.sync_to(state, mount, *sol.hadec(b))
    assert len(align.points(state)) == n
    now = mount.position()
    err = angle_arcsec(align.current_model(state).forward(*now), true_vector(world, mount, now))
    assert err / 3600 < 0.05
    # and now a new point is accepted
    sol = solver.solve(None, mount.clock.now())
    assert align.add_point(state, mount.position(), *sol.hadec(b), sol.t) < 0.05


def test_star_calibration_refuses_stale_counters(cfg):
    mount, world, cam, solver = rig(cfg)
    state = {}
    align.calibrate_on_stars(mount, cam, solver, state, log=lambda *a: None, slew_rate=3.0,
                             settle_s=0.0)
    mount.axis += np.array([0.0, 10.0])
    mount.query()
    with pytest.raises(RuntimeError, match="sync on stars"):
        align.calibrate_on_stars(mount, cam, solver, state, log=lambda *a: None, slew_rate=3.0,
                                 settle_s=0.0)


def test_sync_without_a_model_drops_a_lone_point(cfg):
    mount, world, cam, solver = rig(cfg)
    state = {}
    align.add_point(state, mount.position(), 10.0, 20.0, 0.0)
    align.sync_to(state, mount, 12.0, 21.0)
    assert align.points(state) == []


def test_boresight_from_an_identified_star(cfg):
    mount, world, cam, solver = rig(cfg, pointing_error=(0.3, -0.2))
    sol = solver.solve(None, mount.clock.now())
    guide = dict(world.true_cal["guide"])
    main = dict(world.true_cal["main"])
    truth = np.array(guide["boresight"])
    guide["boresight"] = (truth + [40.0, -25.0]).tolist()     # a stale, hand-set boresight
    px_main = np.array(world.pixel("main", mount.clock.now()))  # the target star, in main
    bore, label, miss, _ = align.boresight_on_star(sol, main, guide, px_main)
    assert label == "target"
    assert np.allclose(bore, truth, atol=1.0)


def test_planner_uses_the_model(cfg, site):
    """plan_pass took a model and ignored it - a turned tripod was planned as if aligned."""
    sat = pr.make_satellite(pr.SIM_TLE)
    p = pr.find_passes(sat, site, pr.time_to_unix(sat.epoch), 24)[0]
    model = misalignment(site.lat, (0.0, 0.0), 90.0)
    ideal, _ = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"])
    turned, _ = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], model=model)
    t = 0.5 * (turned.t_start + turned.t_end)
    a = turned.at(t)[0]
    ha, dec, _, _ = pr.sat_hadec(sat, site, t)
    assert angle_arcsec(model.forward(*a), sky_unit(ha, dec)) < 60
    assert np.max(np.abs(a - ideal.at(t)[0])) > 10


# ---- a real frame ----

@pytest.mark.skipif(not shutil.which("solve-field"), reason="astrometry.net not installed")
def test_real_guide_frame_solves(cfg, site):
    """ASI120MM Mini + 16 mm lens, 1 s, from the balcony: the Big Dipper's handle."""
    import cv2

    cam_cfg = cfg["cameras"]["guide"]
    if missing_indexes(cam_cfg):
        pytest.skip("index files missing - issctl solve-setup")
    img = cv2.imread(str(DATA / "guide-alkaid.png"), cv2.IMREAD_UNCHANGED)
    t0 = time.monotonic()
    sol = AstrometrySolver(cam_cfg, site).solve(img, 1790000000.0)
    assert time.monotonic() - t0 < 30
    ra, dec = sol.radec(sol.centre())
    assert abs(ra - 216.06) < 0.05 and abs(dec - 53.01) < 0.05
    # the lens is really 15.5 mm, not the nominal 16
    assert sol.scale_arcsec() == pytest.approx(49.8, abs=0.3)
    named = {label: px for px, label, _ in sol.catalog()}
    assert np.hypot(*(named["alkaid"] - [1039, 187])) < 3


def test_a_sync_relabels_the_position_history(cfg):
    """A frame exposed just before a sync must still be paired with the right counters: the
    history used to keep the old labels, and 'sync on stars' followed by 'calibrate on stars'
    fitted its first point across the whole correction (75" rms, axes 86 deg apart)."""
    clock = Clock()
    mount = SimMount(cfg, {}, clock, start=[20.0, 40.0])
    mount.query()
    t_frame = clock.now()
    time.sleep(0.01)
    mount.query()
    mount.sync(-30.0, 10.0)
    assert np.allclose(mount.position_at(t_frame), mount.position(), atol=1e-6)
