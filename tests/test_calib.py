"""Calibration sanity checks - especially the scale diagnostic, which catches a wrong gear ratio."""

import numpy as np
import pytest

from issctl.calib import ideal_calibration, image_jog_rates, jacobian, pixels_per_deg, scale_check

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
