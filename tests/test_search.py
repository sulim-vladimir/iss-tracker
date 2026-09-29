"""Spiral search for a star outside the main camera's few arcminutes."""

import numpy as np
import pytest

from issctl.calib import pixels_per_deg
from issctl.camera import SimCamera
from issctl.clock import Clock
from issctl.config import load_config
from issctl.mount import SimMount
from issctl.search import Search, search_step_deg, spiral_offsets
from issctl.sim import CalibWorld


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_spiral_covers_the_square_without_gaps_or_repeats():
    pts = np.array(list(spiral_offsets(1.0, 3.0)))
    assert len(pts) == 49 and tuple(pts[0]) == (0.0, 0.0)
    assert len({tuple(p) for p in pts}) == 49
    assert np.all(np.abs(pts) <= 3.0)
    # each stop is one step from the last: the mount never jumps across the pattern
    assert np.allclose(np.max(np.abs(np.diff(pts, axis=0)), axis=1), 1.0)


def test_step_fits_the_frame_at_any_rotation(cfg):
    main = cfg["cameras"]["main"]
    step = search_step_deg(main)
    short = min(main["width"], main["height"]) / pixels_per_deg(main)
    assert step * np.sqrt(2) <= short


def _rig(cfg, offset, hot=None):
    clock = Clock(speed=5.0)
    mount = SimMount(cfg, {}, clock, start=[20.0, 40.0])
    mount.query()
    world = CalibWorld(cfg, mount, offset=offset, decoys=())
    if hot is not None:
        # a bright spot fixed on the sensor - brighter than the star, and it never moves
        blobs = world.blobs
        world.blobs = lambda cam, t: blobs(cam, t) + ([(np.array(hot), 250.0)] if cam == "main" else [])
    cam = SimCamera("main", cfg["cameras"]["main"], clock, world, fps=10.0).start()
    return mount, world, cam


def test_finds_a_star_outside_the_field(cfg):
    mount, world, cam = _rig(cfg, offset=(0.3, -0.25))
    assert world.pixel("main", mount.clock.now()) is None          # not in view to begin with
    logs = []
    try:
        hit = Search(mount, cam, log=logs.append, settle_s=0.0, slew_rate=3.0).run()
        seen = world.pixel("main", mount.clock.now())
    finally:
        cam.stop()
    assert hit is not None, logs
    assert seen is not None                                         # left pointing at it
    assert np.hypot(*(np.array(hit["px"]) - seen)) < 5


def test_ignores_a_hot_spot_that_does_not_move(cfg):
    mount, world, cam = _rig(cfg, offset=(0.3, -0.25), hot=(300.0, 250.0))
    logs = []
    try:
        hit = Search(mount, cam, log=logs.append, settle_s=0.0, slew_rate=3.0).run()
        seen = world.pixel("main", mount.clock.now())
    finally:
        cam.stop()
    assert any("does not move with the mount" in m for m in logs), logs
    assert hit is not None and seen is not None
    assert np.hypot(*(np.array(hit["px"]) - seen)) < 5


def test_gives_up_and_returns_when_there_is_nothing(cfg):
    mount, world, cam = _rig(cfg, offset=(3.0, 3.0))
    start = mount.position().copy()
    try:
        hit = Search(mount, cam, log=lambda *a: None, radius_deg=0.2, settle_s=0.0,
                     slew_rate=3.0).run()
    finally:
        cam.stop()
    assert hit is None
    assert np.allclose(mount.position(), start, atol=0.01)
