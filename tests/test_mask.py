import numpy as np

from issctl.mask import SkyMask, segments


def test_empty_mask_sees_everything():
    m = SkyMask()
    assert m.empty
    assert np.all(m.visible([5.0, 40.0, 80.0], [0.0, 120.0, 300.0]))


def test_opening_limits_azimuth_and_altitude():
    m = SkyMask(openings=[[100, 260, 18, 80]])
    assert not m.empty
    alt = np.array([30.0, 30.0, 10.0, 85.0, 30.0])
    az = np.array([180.0, 90.0, 180.0, 180.0, 260.0])
    assert list(m.visible(alt, az)) == [True, False, False, False, True]


def test_azimuth_range_wraps_through_north():
    m = SkyMask(openings=[[350, 20, 10, 70]])
    alt = np.full(4, 30.0)
    assert list(m.visible(alt, [355.0, 5.0, 30.0, 180.0])) == [True, True, False, False]


def test_blocker_subtracts_from_opening():
    m = SkyMask(openings=[[100, 260, 10, 80]], blockers=[[168, 176, 0, 90]])
    alt = np.full(3, 40.0)
    assert list(m.visible(alt, [160.0, 172.0, 200.0])) == [True, False, True]


def test_segments_finds_runs():
    t = np.arange(0, 20, 1.0)
    ok = np.zeros(20, dtype=bool)
    ok[2:9] = True     # 6 s run
    ok[12:14] = True   # 1 s run, below min_length
    assert segments(t, ok, min_length=2.0) == [(2.0, 8.0)]
    assert len(segments(t, ok, min_length=0.5)) == 2
