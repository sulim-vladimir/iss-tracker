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
    cam.gate = None
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


def calibrate_cameras(mount, cams, step_deg=0.08, track_rate=None, log=print):
    """Needs a bright star visible in every camera (centre it in the main camera first)."""
    start = mount.position()
    for name, cam in cams.items():
        if measure(cam) is None:
            raise RuntimeError(f"no star detected in {name} camera")
    cols = {name: [None, None] for name in cams}
    for axis in (0, 1):
        d = np.zeros(2)
        d[axis] = step_deg
        mount.move_to(start - d, track_rate=track_rate)
        mount.move_to(start, track_rate=track_rate)       # approach from + side: backlash taken up
        time.sleep(0.5)
        base = {n: measure(c) for n, c in cams.items()}
        mount.move_to(start + d, track_rate=track_rate)
        time.sleep(0.5)
        moved = {n: measure(c) for n, c in cams.items()}
        for n in cams:
            if base[n] is None or moved[n] is None:
                raise RuntimeError(f"lost star in {n} camera while moving axis{axis + 1}")
            cols[n][axis] = (moved[n] - base[n]) / step_deg
            log(f"{n}: axis{axis + 1} +{step_deg} deg -> {cols[n][axis].round(1)} px")
        mount.move_to(start - d, track_rate=track_rate)
        mount.move_to(start, track_rate=track_rate)
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
        result["guide"]["boresight"] = (final["guide"] + np.array(result["guide"]["J"]) @ dth).tolist()
        log(f"guide boresight (main camera centre) at {np.round(result['guide']['boresight'], 1)}")
    return result
