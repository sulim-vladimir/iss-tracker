import numpy as np
import pytest

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


def test_a_frame_line_clicked_in_the_guide_image_lands_on_the_sky():
    """Two clicks along a window-frame edge in the guide image become a line on the sky chart:
    the frame centre is where the alignment says the mount points, and a straight edge in the
    image is a great circle on the sky."""
    from issctl import align
    from issctl import geometry as geo
    from issctl.calib import ideal_calibration
    from issctl.config import load_config
    from issctl.mask import frame_line
    from issctl.model import PointingModel

    cfg = load_config()
    g = cfg["cameras"]["guide"]
    frame = (g["width"] // g["bin"], g["height"] // g["bin"])
    lat = cfg["site"]["latitude"]
    state = {"alignment": {"model": dict(PointingModel().to_dict(), n_points=4)},
             "cameras": {"guide": ideal_calibration(g, rotation_deg=30.0, dec_cal=40.0)}}
    axes = np.array([20.0, 60.0])
    centre = ((frame[0] - 1) / 2.0, (frame[1] - 1) / 2.0)
    line = frame_line(state, axes, centre, (centre[0] + 400.0, centre[1] - 250.0), lat, frame)
    ha, dec = align.pointing_hadec(state, axes)
    alt, az = geo.hadec_to_altaz(ha, dec, lat)
    assert line[0] == pytest.approx([az % 360.0, alt], abs=1e-3)
    assert len(line) == 12 and line[-1] != line[0]
    v = np.array([[np.cos(np.radians(h)) * np.cos(np.radians(a)),
                   np.cos(np.radians(h)) * np.sin(np.radians(a)), np.sin(np.radians(h))]
                  for a, h in line])
    normal = np.cross(v[0], v[-1])
    assert np.max(np.abs(v @ normal / np.linalg.norm(normal))) < np.radians(0.2)   # ~a great circle
    with pytest.raises(ValueError, match="alignment"):
        frame_line({"cameras": state["cameras"]}, axes, centre, centre, lat, frame)
