"""Camera-to-mount calibration.

For each camera we store J (2x2): image shift in pixels per +1 deg of (axis1, axis2),
measured at declination dec_cal, and the boresight pixel (where the main camera's centre
lands). Axis1 image motion scales with cos(dec) and is the same for both pier sides.
"""

import time

import numpy as np

from . import geometry as geo


def pixels_per_deg(cam_cfg):
    arcsec_per_px = 206.265 * cam_cfg["pixel_um"] * cam_cfg["bin"] / cam_cfg["focal_length_mm"]
    return 3600.0 / arcsec_per_px


def ideal_calibration(cam_cfg, rotation_deg=0.0, dec_cal=0.0, parity=1):
    s = pixels_per_deg(cam_cfg)
    c, n = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
    J = s * np.array([[c, -n * parity], [n, c * parity]])
    w, h = cam_cfg["width"] // cam_cfg["bin"], cam_cfg["height"] // cam_cfg["bin"]
    return {"J": J.tolist(), "dec_cal": dec_cal, "boresight": [(w - 1) / 2, (h - 1) / 2]}


def cal_px_per_deg(cal):  # noqa: D401
    """Image scale in pixels per degree on the sky (axis2 is a pure sky rotation)."""
    return float(np.linalg.norm(np.array(cal["J"], dtype=float)[:, 1]))


def jacobian(cal, axis2):
    J = np.array(cal["J"], dtype=float)
    dec = float(geo.axis2_to_dec(axis2))
    k = np.cos(np.radians(dec)) / max(np.cos(np.radians(cal["dec_cal"])), 0.05)
    J[:, 0] *= np.sign(k) * max(abs(k), 0.05)
    return J


def axes_offset_from_pixel(cal, axis2, px):
    """Axis move (deg) that brings an object at pixel px onto the boresight."""
    return np.linalg.solve(jacobian(cal, axis2), np.asarray(cal["boresight"]) - np.asarray(px))


def measure(cam, n=10, timeout=5.0):
    if not getattr(cam, "manual", False):
        cam.gate = None   # but never throw away a target the user picked by hand
    _, _, last = cam.latest()
    pts, deadline = [], time.monotonic() + timeout
    while len(pts) < n and time.monotonic() < deadline:
        _, det, seq = cam.latest()
        if seq != last:
            last = seq
            if det:
                pts.append((det.x, det.y))
        time.sleep(0.005)
    return np.mean(pts, axis=0) if len(pts) >= max(3, n // 2) else None


def default_step(cam, fraction=0.2):
    """Move enough to shift the target ~20% of the frame height in THIS camera.

    A single step cannot serve both: with a 16 mm guide lens (74 px/deg) and the main camera at
    750 mm (4500 px/deg), 0.08 deg is 375 px in the main frame but only 6 px in the guide.
    """
    return float(np.clip(fraction * cam.height / pixels_per_deg(cam.cfg), 0.02, 3.0))


def calibrate_cameras(mount, cams, steps=None, track_rate=None, log=print, abort=None, warnings=None):
    """Needs one bright target visible in every camera (centre it in the main camera first).

    Each camera is calibrated with its own step size, so wide and narrow fields both get a
    well-measured shift.
    """
    steps = dict(steps or {})
    start = mount.position()
    for name, cam in cams.items():
        if measure(cam) is None:
            raise RuntimeError(f"no target detected in {name} camera")
        steps.setdefault(name, default_step(cam))
    def check_abort():
        if abort and abort():
            raise RuntimeError("calibration aborted")

    cols = {name: [None, None] for name in cams}
    for name, cam in cams.items():
        step_deg = steps[name]
        for axis in (0, 1):
            check_abort()
            d = np.zeros(2)
            d[axis] = step_deg
            mount.move_to(start - d, track_rate=track_rate, abort=abort)
            mount.move_to(start, track_rate=track_rate, abort=abort)  # approach from + side
            time.sleep(0.5)
            base = measure(cam)
            mount.move_to(start + d, track_rate=track_rate, abort=abort)
            time.sleep(0.5)
            moved = measure(cam)
            check_abort()
            if base is None or moved is None:
                raise RuntimeError(f"lost the target in {name} while moving axis{axis + 1} "
                                   f"by {step_deg:.3f} deg")
            cols[name][axis] = (moved - base) / step_deg
            log(f"{name}: axis{axis + 1} +{step_deg:.3f} deg -> {(moved - base).round(1)} px")
            mount.move_to(start - d, track_rate=track_rate, abort=abort)
            mount.move_to(start, track_rate=track_rate, abort=abort)
    time.sleep(0.5)
    final = {n: measure(c) for n, c in cams.items()}
    dec_cal = float(geo.axis2_to_dec(mount.position()[1]))

    result = {}
    for n, cam in cams.items():
        J = np.column_stack(cols[n])
        result[n] = {"J": J.tolist(), "dec_cal": dec_cal,
                     "boresight": [(cam.width - 1) / 2, (cam.height - 1) / 2]}
        s = np.linalg.svd(J, compute_uv=False)
        log(f"{n}: scale {s.round(1)} px/deg, rotation {np.degrees(np.arctan2(J[1, 0], J[0, 0])):.1f} deg")
    if "main" in cams and "guide" in cams and final["main"] is not None and final["guide"] is not None:
        m = result["main"]
        dth = np.linalg.solve(np.array(m["J"]), np.array(m["boresight"]) - final["main"])
        boresight = final["guide"] + np.array(result["guide"]["J"]) @ dth
        result["guide"]["boresight"] = boresight.tolist()
        # Both cameras must have measured the SAME object for this to mean anything. If the guide
        # locked onto a different (often brighter) light, the boresight lands far from the centre.
        centre = np.array([(cams["guide"].width - 1) / 2, (cams["guide"].height - 1) / 2])
        off_deg = float(np.linalg.norm(boresight - centre)) / cal_px_per_deg(result["guide"])
        log(f"guide boresight (main camera centre) at {np.round(boresight, 1)}, "
            f"{off_deg:.2f} deg from frame centre")
        if off_deg > 2.0 and warnings is not None:
            warnings.append(f"boresight {off_deg:.1f}° off centre - did both cameras see the "
                            f"same object? Click the target in each image, then calibrate again.")
    return result
