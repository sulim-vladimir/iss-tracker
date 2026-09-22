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


def image_jog_rates(cal, axis2, jog, speed, min_cos_dec=0.15):
    """Axis rates that move the target in the image the way the arrows were pressed.

    jog is (right, up) in screen terms. Returns None when this cannot work: near the pole axis1
    only rotates the field instead of shifting it, so a screen direction has no sensible mapping
    and the caller should drive the axes directly.
    """
    dec = float(geo.axis2_to_dec(axis2))
    if abs(np.cos(np.radians(dec))) < min_cos_dec:
        return None
    want_px = np.array([jog[0], -jog[1]], dtype=float)   # screen up is -y in image coordinates
    d = np.linalg.solve(jacobian(cal, axis2), want_px)
    peak = float(np.max(np.abs(d)))
    return d / peak * speed if peak > 1e-9 else np.zeros(2)


def centring_move(cal, axis2, px, max_deg=20.0, target_px=None):
    """Axis move that brings the object at px onto target_px (the boresight by default).

    A wild answer means the calibration or the detection is wrong, and slewing tens of degrees
    because of a misdetected pixel is worse than doing nothing.
    """
    want = np.asarray(cal["boresight"] if target_px is None else target_px, dtype=float)
    d = np.linalg.solve(jacobian(cal, axis2), want - np.asarray(px, dtype=float))
    return None if np.max(np.abs(d)) > max_deg else d


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


def default_step(cam, fraction=0.15):
    """Move enough to shift the target ~20% of the frame height in THIS camera.

    A single step cannot serve both: with a 16 mm guide lens (74 px/deg) and the main camera at
    750 mm (4500 px/deg), 0.08 deg is 375 px in the main frame but only 6 px in the guide.
    """
    return float(np.clip(fraction * cam.height / pixels_per_deg(cam.cfg), 0.02, 3.0))


def implied_focal_length(px_per_deg, cam_cfg):
    """Focal length that would give this image scale, in mm.

    Only as good as its assumptions: that the axis really turned what it was told, and that the
    target is far enough away. Rotating the mount also translates the camera by its radius r from
    the axis, so a target at distance d shifts by a factor (1 +/- r/d) - several percent indoors.
    """
    return px_per_deg * cam_cfg["pixel_um"] * cam_cfg["bin"] / 1000.0 / np.tan(np.radians(1.0))


def axes_angle(J):
    """Angle between the two measured axis directions, in degrees. Should be 90."""
    J = np.asarray(J, dtype=float)
    a0 = np.degrees(np.arctan2(J[1, 0], J[0, 0]))
    a1 = np.degrees(np.arctan2(J[1, 1], J[0, 1]))
    return float(abs((a1 - a0 + 180) % 360 - 180))


def orthogonalise(J):
    """Force the axis directions perpendicular, keeping each column's length.

    RA and Dec are perpendicular on the mount, so a measured deviation is error - creep, slack
    releasing or shimmer during the (necessarily small) steps. Splitting the error between the two
    columns is better than believing a skewed matrix.
    """
    J = np.asarray(J, dtype=float).copy()
    a0 = np.arctan2(J[1, 0], J[0, 0])
    a1 = np.arctan2(J[1, 1], J[0, 1])
    sep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    fix = (np.sign(sep) * np.pi / 2 - sep) / 2          # share the correction between them
    for col, angle, turn in ((0, a0, -fix), (1, a1, +fix)):
        length = float(np.linalg.norm(J[:, col]))
        J[:, col] = length * np.array([np.cos(angle + turn), np.sin(angle + turn)])
    return J


def scale_check(J, cam_cfg, dec_cal):
    """Compare measured pixels-per-axis-degree with what the optics imply.

    The optics fix how many pixels one degree of SKY is worth, so a mismatch means the axis did not
    turn as far as commanded: steps_per_deg is wrong. Returns (expected, measured per axis,
    correction factor per axis) - multiply gear_ratio by the factor.
    """
    J = np.asarray(J, dtype=float)
    expected = pixels_per_deg(cam_cfg)
    cos_dec = max(np.cos(np.radians(dec_cal)), 0.05)
    measured = np.array([np.linalg.norm(J[:, 0]) / cos_dec, np.linalg.norm(J[:, 1])])
    return expected, measured, expected / np.maximum(measured, 1e-9)


def measure_backlash(mount, cam, cal, axis, step_deg=None, track_rate=None, log=print,
                     abort=None, slew_rate=0.3):
    """How much command an axis swallows before it actually turns, in degrees.

    Take up the slack in one direction, then reverse by a known amount and see how far the image
    really moved. The difference is lost motion: backlash, belt stretch and worm end-float together.
    """
    scale = pixels_per_deg(cam.cfg)
    step_deg = step_deg or float(np.clip(0.3 * cam.height / scale, 0.02, 1.0))
    d = np.zeros(2)
    d[axis] = step_deg
    start = mount.position()
    mount.move_to(start + d, track_rate=track_rate, abort=abort, max_rate=slew_rate)  # slack taken up +
    time.sleep(0.5)
    before = measure(cam)
    mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)      # now reverse
    time.sleep(0.5)
    after = measure(cam)
    if before is None or after is None:
        raise RuntimeError(f"lost the target in {cam.name} while measuring backlash")
    moved_px = float(np.linalg.norm(after - before))
    expected_px = scale * step_deg * max(np.cos(np.radians(geo.axis2_to_dec(start[1]))), 0.05) \
        if axis == 0 else scale * step_deg
    lost = max(0.0, (expected_px - moved_px) / max(scale, 1e-9))
    saturated = moved_px < 0.1 * expected_px       # the axis barely moved: slack ate the whole step
    log(f"axis{axis + 1} on {cam.name}: reversed {step_deg:.3f} deg, image moved {moved_px:.0f} px "
        f"of {expected_px:.0f} -> lost motion {lost * 60:.1f} arcmin"
        + (" (AT LEAST: the slack exceeds what this camera can measure in one step)"
           if saturated else ""))
    return lost, saturated


def bring_into_view(mount, cam, ref_cam, ref_cal, track_rate=None, log=print, abort=None,
                    slew_rate=0.3, tries=3):
    """Put the target back where a narrow camera can see it, using a wide one that still has it.

    Big moves (calibrating the wide field, measuring backlash) do not repeat to within a narrow
    field - the mount's slack is larger than that. The wide camera always knows where the target
    is, so it can steer it back onto the boresight, which is where the narrow camera looks.
    """
    for _ in range(tries):
        if measure(cam, n=3, timeout=2.0) is not None:
            return True
        px = measure(ref_cam, n=5, timeout=3.0)
        if px is None:
            return False
        d = centring_move(ref_cal, mount.position()[1], px)
        if d is None:
            return False
        log(f"{cam.name}: lost the target, steering it back with {ref_cam.name}")
        mount.move_to(mount.position() + d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        if abort and abort():
            return False
        time.sleep(0.5)
    return measure(cam, n=3, timeout=2.0) is not None


def calibrate_against(mount, cam, ref_cam, ref_cal, step_deg=None, ramp_steps=3, track_rate=None,
                      log=print, abort=None, slew_rate=0.3, measure_frames=10):
    """Calibrate a narrow camera by comparing it with an already-calibrated wide one.

    A narrow field cannot take a step big enough to beat the mount's backlash, so measuring it
    against the axes is hopeless. But both cameras see the SAME motion, whatever the axes actually
    did, so the ratio between their image shifts is clean: backlash, stiction and scale error all
    cancel. With S_cam and S_ref the measured shifts per commanded degree, and G the reference
    matrix, the answer is J = S_cam . S_ref^-1 . G.
    """
    scale = pixels_per_deg(cam.cfg)
    step_deg = step_deg or float(np.clip(0.2 * cam.height / scale, 0.005, 0.5))
    start = mount.position()
    s_cam, s_ref = [], []
    for axis in (0, 1):
        d = np.zeros(2)
        d[axis] = step_deg
        mount.move_to(start - d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        angles, here, there = [], [], []
        for k in range(ramp_steps + 1):
            if abort and abort():
                raise RuntimeError("calibration aborted")
            if k:
                mount.move_to(start + d * k, track_rate=track_rate, abort=abort, max_rate=slew_rate)
            time.sleep(0.4)
            a, b = measure(cam, n=measure_frames), measure(ref_cam, n=measure_frames)
            if a is None or b is None:
                raise RuntimeError(f"lost the target in {cam.name if a is None else ref_cam.name} "
                                   f"after {k} steps of {step_deg:.3f} deg")
            angles.append(k * step_deg)
            here.append(a)
            there.append(b)
        A = np.vstack([np.array(angles), np.ones(len(angles))]).T
        s_cam.append(np.linalg.lstsq(A, np.array(here), rcond=None)[0][0])
        s_ref.append(np.linalg.lstsq(A, np.array(there), rcond=None)[0][0])
        log(f"{cam.name}: axis{axis + 1} moved {np.linalg.norm(s_cam[-1]) * angles[-1]:.0f} px "
            f"while {ref_cam.name} moved {np.linalg.norm(s_ref[-1]) * angles[-1]:.1f} px")
        mount.move_to(start - d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)
    S_cam, S_ref = np.column_stack(s_cam), np.column_stack(s_ref)
    if abs(np.linalg.det(S_ref)) < 1e-9:
        raise RuntimeError(f"{ref_cam.name} barely moved - use a bigger step")
    return S_cam @ np.linalg.inv(S_ref) @ np.array(ref_cal["J"], dtype=float)


def calibrate_cameras(mount, cams, steps=None, track_rate=None, log=print, abort=None,
                      warnings=None, slew_rate=0.5, only=None, existing=None,
                      ramp_steps=3, measure_frames=10):
    """Needs one bright target visible in every camera (centre it in the main camera first).

    Each camera is calibrated with its own step size, so wide and narrow fields both get a
    well-measured shift. Moves are deliberately slow (slew_rate): a skipped step during calibration
    silently corrupts the measurement, because the counter keeps counting.

    ramp_steps controls how far it walks: more steps take longer but average away noise, and the
    total travel must stay inside the frame.

    only=[names] calibrates just those cameras - useful when one of them could not see the target
    the first time round. The boresight still needs both cameras to see the target now, but the
    other camera's stored matrix (existing) is reused, so you never have to redo the wide field.
    """
    steps = dict(steps or {})
    start = mount.position()
    # Measure every camera BEFORE moving anything: this is the only moment all of them are looking
    # at the same pose, so it is the only reliable basis for the guide->main boresight. Calibrating
    # a wide guide needs degrees of motion, which throws the target far outside the main frame, and
    # backlash means the mount does not come back precisely enough to re-measure afterwards.
    usable, missing, at_start = {}, [], {}
    for name, cam in cams.items():
        at_start[name] = measure(cam)
        if at_start[name] is None:
            missing.append(name)
            continue
        usable[name] = cam
        steps.setdefault(name, default_step(cam))
    if not usable:
        raise RuntimeError(f"no target detected in {' or '.join(cams)} - adjust exposure/gain, "
                           f"focus, or click the target in the image")
    if missing:
        log(f"skipping {', '.join(missing)}: no target detected there")
        if warnings is not None:
            warnings.append(f"{', '.join(missing)} not calibrated (no target detected) - without the "
                            f"main camera there is no boresight, so the handoff is not set up")
    # Widest field first: a narrow camera cannot out-step the mount's backlash, so it is calibrated
    # against the wide one afterwards, where the slack cancels in the ratio.
    ordered = sorted(usable, key=lambda n: pixels_per_deg(usable[n].cfg))
    cams = {n: usable[n] for n in ordered if only is None or n in only}
    if not cams:
        raise RuntimeError(f"{' and '.join(only)}: no target detected there")
    log(f"calibrating {', '.join(cams)}")
    def check_abort():
        if abort and abort():
            raise RuntimeError("calibration aborted")

    cols = {name: [None, None] for name in cams}
    reference = ordered[0] if ordered else None            # the widest field we have
    for name, cam in cams.items():
        ref_cal = None
        if reference and reference != name:
            stored = (existing or {}).get(reference) or {}
            fresh = cols.get(reference)
            if fresh and all(c is not None for c in fresh):
                ref_wide = usable[reference]
                ref_cal = {"J": np.column_stack(fresh),
                           "dec_cal": float(geo.axis2_to_dec(mount.position()[1])),
                           # steering the target back needs to know where the narrow camera looks
                           "boresight": stored.get("boresight",
                                                   [(ref_wide.width - 1) / 2,
                                                    (ref_wide.height - 1) / 2])}
            else:
                ref_cal = stored or None
        if ref_cal and reference in usable and measure(usable[reference]) is not None:
            if not bring_into_view(mount, cam, usable[reference], ref_cal, track_rate=track_rate,
                                   log=log, abort=abort, slew_rate=slew_rate):
                log(f"{name}: target not in view and could not be recovered - skipped")
                if warnings is not None:
                    warnings.append(f"{name} not calibrated: the target never came back into its "
                                    f"field. Centre it ('send to main') and calibrate {name} alone.")
                continue
            log(f"{name}: calibrating against {reference} - the mount's slack cancels in the ratio")
            J = calibrate_against(mount, cam, usable[reference], ref_cal, track_rate=track_rate,
                                  log=log, abort=abort, slew_rate=slew_rate,
                                  measure_frames=measure_frames, ramp_steps=ramp_steps)
            cols[name] = [J[:, 0], J[:, 1]]
            continue
        step_deg = steps[name]
        for axis in (0, 1):
            check_abort()
            d = np.zeros(2)
            d[axis] = step_deg
            # Take up the slack in the + direction, then walk the same way in equal steps and fit a
            # line. Every measurement after the first is backlash-free, several points average the
            # noise, and the fit residual shows whether the axis moved smoothly at all.
            mount.move_to(start - d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
            mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)
            angles, points = [], []
            for k in range(ramp_steps + 1):
                check_abort()
                if k:
                    mount.move_to(start + d * k, track_rate=track_rate, abort=abort,
                                  max_rate=slew_rate)
                time.sleep(0.4)
                seen = measure(cam, n=measure_frames)
                if seen is None:
                    raise RuntimeError(f"lost the target in {name} after {k} steps of "
                                       f"{step_deg:.3f} deg on axis{axis + 1}")
                angles.append(k * step_deg)
                points.append(seen)
            A = np.vstack([np.array(angles), np.ones(len(angles))]).T
            fit, *_ = np.linalg.lstsq(A, np.array(points), rcond=None)
            slope = fit[0]
            resid = float(np.sqrt(np.mean((np.array(points) - A @ fit) ** 2)))
            cols[name][axis] = slope
            travel = float(np.linalg.norm(slope)) * angles[-1]
            log(f"{name}: axis{axis + 1} {ramp_steps} x {step_deg:.3f} deg -> {travel:.0f} px "
                f"({np.linalg.norm(slope):.0f} px/deg, fit residual {resid:.1f} px)")
            if resid > 0.15 * travel and warnings is not None:
                warnings.append(f"{name} axis{axis + 1}: the steps were not consistent "
                                f"(residual {resid:.0f} px of {travel:.0f}) - backlash, skipping "
                                f"or a moving target")
            mount.move_to(start - d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
            mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)
    dec_cal = float(geo.axis2_to_dec(mount.position()[1]))
    if abs(np.cos(np.radians(dec_cal))) < 0.5 and warnings is not None:
        # Near the pole axis1 rotates the field instead of shifting it, so its column is
        # meaningless and cos(dec_cal) rescaling blows up everywhere else.
        warnings.append(f"calibrated at dec {dec_cal:.0f}° - too close to the pole for axis1 to "
                        f"move the image. Point below dec 60° (well away from home) and redo it.")

    result = {}
    scale_factors = []
    for n, cam in cams.items():
        J = np.column_stack(cols[n])
        skew = axes_angle(J)
        if abs(skew - 90) > 1.0:
            J = orthogonalise(J)
            log(f"{n}: measured axes {skew:.1f} deg apart, squared up to 90")
        if abs(skew - 90) > 5.0 and warnings is not None:
            warnings.append(f"{n}: axes measured {skew:.1f} deg apart instead of 90 - the steps "
                            f"were noisy (creep, slack, shimmer). Redo on a steady distant target.")
        result[n] = {"J": J.tolist(), "dec_cal": dec_cal,
                     "boresight": [(cam.width - 1) / 2, (cam.height - 1) / 2]}
        sv = np.linalg.svd(J, compute_uv=False)
        log(f"{n}: {3600 / max(sv.mean(), 1e-9):.2f} arcsec/px (scale {sv.round(1)} px/deg), "
            f"rotation {np.degrees(np.arctan2(J[1, 0], J[0, 0])):.1f} deg")

        expected, measured, factor = scale_check(J, cam.cfg, dec_cal)
        log(f"{n}: optics say {3600 / expected:.2f} arcsec/px; measured "
            f"{np.round(3600 / np.maximum(measured, 1e-9), 2)} arcsec/px "
            f"-> axis moved {1 / factor[0]:.2f}x / {1 / factor[1]:.2f}x of what was commanded; "
            f"= focal length {implied_focal_length(measured[1], cam.cfg):.0f} mm if the axes moved "
            f"exactly as commanded and the target is far away (config says "
            f"{cam.cfg['focal_length_mm']:g})")
        scale_factors.append(factor)

    if scale_factors and warnings is not None:
        factor = np.mean(scale_factors, axis=0)
        if np.any(np.abs(factor - 1) > 0.15):
            warnings.append(
                f"scale mismatch x{factor[0]:.2f}/{factor[1]:.2f}: EITHER focal_length_mm is wrong "
                f"(see the implied focal length above) OR the axes under/over-move, in which case "
                f"multiply gear_ratio by that factor. Check with axis-scale, and calibrate on a "
                f"DISTANT target.")
    # The boresight needs a matrix for each camera - freshly measured or from a previous run - and
    # both cameras looking at the same object right now.
    existing = existing or {}
    jm = result.get("main", existing.get("main"))
    jg = result.get("guide", existing.get("guide"))
    if jm and jg and at_start.get("main") is not None and at_start.get("guide") is not None:
        result.setdefault("guide", dict(jg))
        dth = np.linalg.solve(np.array(jm["J"]), np.array(jm["boresight"]) - at_start["main"])
        boresight = at_start["guide"] + np.array(jg["J"]) @ dth
        result["guide"]["boresight"] = boresight.tolist()
        # Both cameras must have measured the SAME object for this to mean anything. If the guide
        # locked onto a different (often brighter) light, the boresight lands far from the centre.
        guide_cam = usable["guide"]
        centre = np.array([(guide_cam.width - 1) / 2, (guide_cam.height - 1) / 2])
        off_deg = float(np.linalg.norm(boresight - centre)) / cal_px_per_deg(result["guide"])
        log(f"guide boresight (main camera centre) at {np.round(boresight, 1)}, "
            f"{off_deg:.2f} deg from frame centre")
        if off_deg > 2.0 and warnings is not None:
            warnings.append(f"boresight {off_deg:.1f}° off centre - did both cameras see the "
                            f"same object? Click the target in each image, then calibrate again.")
    return result
