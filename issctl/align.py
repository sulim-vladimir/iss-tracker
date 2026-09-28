"""Alignment from the stars: what the mount's step counters mean on the sky.

A plate solve pairs the counters at the moment of exposure with where the tube really pointed.
Enough such pairs fix the pointing model (model.py): the orientation of the polar axis, however
the tripod stands, plus the Dec index and cone. That model is what lets goto and pass mode work
on a mount that was never polar aligned - and it needs no alignment star to be centred by hand,
because the solve reads the pointing straight off the frame.

Two ways the counters and the sky part company, handled differently:

* the tripod is where it was, but the counters stopped meaning anything - the mount was pushed
  by hand, lost steps, was restarted without its saved position. The MODEL is still right, only
  the counters are off: `sync_to` re-indexes them through the model and every earlier point stays
  valid. This is what "sync" means from now on.
* the tripod itself moved. Nothing earlier is valid: clear the alignment and start again.

A new point that disagrees with the model by more than RESYNC_DEG is one of those two, and is
refused with a message saying so, rather than silently dragging the model off.

The model follows the GUIDE FRAME CENTRE, not the boresight. The centre is a fixed direction on
the tube; the boresight is a calibration result that may be wrong, and a model built on a wrong
one would have to be thrown away when it is corrected. Where the main camera looks is a small
offset from here, which "centre by solve" and the tracker apply.
"""

import time

import numpy as np

from . import geometry as geo
from .calib import (axes_angle, boresight_from_picks, cal_px_per_deg, orthogonalise,
                    pixels_per_deg)
from .model import PointingModel, angle_arcsec, fit_model
from .model import unit as sky_unit
from .solve import SolveError, field_deg, solve_camera

RESYNC_DEG = 3.0          # a point this far from the model means the counters no longer apply
CONE_SPREAD_DEG = 30.0    # below this sky spread, cone and polar axis cannot be told apart


def points(state):
    return state.setdefault("alignment", {}).setdefault("points", [])


def current_model(state):
    """The fitted model, or None while there is not enough to fit one worth using."""
    m = (state.get("alignment") or {}).get("model")
    return PointingModel.from_dict(m) if m and m.get("n_points", 0) >= 2 else None


def pointing_hadec(state, axes):
    """Where the tube points, through the model when there is one."""
    model = current_model(state)
    if model is None:
        ha, dec = geo.axes_to_hadec(*axes)
    else:
        ha, dec = model.axes_to_hadec(*axes)
    return float(ha), float(dec)


def hadec_to_axes_fn(state):
    """(ha, dec, side) -> axes, through the model when there is one. For goto and the planner."""
    model = current_model(state)
    return geo.hadec_to_axes if model is None else model.hadec_to_axes


def clear(state):
    state["alignment"] = {"points": []}


def disagreement_deg(state, axes, ha, dec):
    """How far the current model puts these counters from where the sky says they point."""
    model = current_model(state)
    if model is None:
        return None
    v = model.forward(*axes)
    return float(angle_arcsec(v, sky_unit(ha, dec)) / 3600.0)


def _free(vectors, n_dirs=0):
    n = len(vectors)
    if n < 2:
        return None
    if n_dirs >= 2:
        return [0, 1, 2, 3, 4]      # the turning axes pin the orientation and cone down directly
    v = np.asarray(vectors)
    spread = float(np.degrees(np.arccos(np.clip(np.min(v @ v.T), -1, 1))))
    # Cone needs points spread across the sky: from a few degrees it trades off against the polar
    # axis direction and the fit wanders. The four points of one star calibration are like that.
    return [0, 1, 2, 3] if n < 3 or spread < CONE_SPREAD_DEG else [0, 1, 2, 3, 4]


def refit(state, mount=None, cams_state=None):
    """Fit the model to the stored points. With a mount, also move the fitted Dec index into the
    counters, so axis2 then reads the true declination of the mount frame - the one quantity the
    camera matrices depend on (axis1's image motion scales with its cosine).

    Returns (model, dec_shift_deg)."""
    pts = points(state)
    if not pts:
        state["alignment"].pop("model", None)
        return None, 0.0
    axes = np.array([p["axes"] for p in pts], dtype=float)
    vectors = sky_unit(np.array([p["ha"] for p in pts]), np.array([p["dec"] for p in pts]))
    prior = PointingModel.from_dict((state.get("alignment") or {}).get("model"))
    dirs = [(d["kind"], d["axis1"], d["v"], d["weight"]) for d in state["alignment"].get("axis_dirs", [])]
    model = fit_model(axes, vectors, prior=prior, free=_free(vectors, len(dirs)), axis_dirs=dirs)
    shift = 0.0
    if mount is not None and len(pts) >= 2 and abs(model.d2) > 1e-6:
        shift = model.d2
        east = mount.position()[1] <= 90.0
        mount.index = mount.index + [0.0, shift]     # through the setter: relabels the history
        mount.query()
        for p in pts:
            p["axes"][1] += shift
        # A camera matrix measured before this carries the old counters' idea of its dec.
        for cal in (cams_state or {}).values():
            if "dec_cal" in cal:
                cal["dec_cal"] = float(cal["dec_cal"] + (shift if east else -shift))
        residuals = getattr(model, "residuals_arcsec", None)
        model = PointingModel(model.rotvec_deg, 0.0, model.cone, model.rms_arcsec, model.n_points)
        model.residuals_arcsec = residuals
    state["alignment"]["model"] = model.to_dict()
    return model, shift


def add_point(state, axes, ha, dec, t, source="solve"):
    """Store one (counters, sky) pair. Refuses one the model says the counters cannot explain."""
    off = disagreement_deg(state, axes, ha, dec)
    if off is not None and off > RESYNC_DEG:
        raise ValueError(
            f"the counters and the sky disagree by {off:.1f} deg, far more than the model "
            f"allows. Either the mount was moved without the counters (pushed by hand, lost "
            f"steps): 'sync on stars' re-indexes them and keeps the alignment. Or the tripod "
            f"was moved: 'clear alignment' and start again.")
    points(state).append({"axes": [float(a) for a in axes], "ha": float(ha), "dec": float(dec),
                          "t": float(t), "source": source})
    return off


def sync_to(state, mount, ha, dec):
    """Make the current counters read (ha, dec). Returns the correction, in axis degrees.

    With a model, the counters are re-indexed THROUGH it, so the tripod's orientation and every
    stored point stay valid - this is the right answer to "I pushed the mount by hand". Without
    one there is nothing to preserve: the ideal-mount sync is used and any lone point, taken in
    the old counters, is dropped."""
    model = current_model(state)
    if model is None:
        if points(state):
            clear(state)
        return mount.sync(ha, dec)
    cur = mount.query()
    best = None
    for side in geo.SIDES:
        a1, a2 = model.hadec_to_axes(ha, dec, side)
        want = np.array([cur[0] + geo.wrap180(float(a1) - cur[0]), float(a2)])
        delta = want - cur
        # A hand push can swing the tube past the pole, so the counters' side proves nothing:
        # the side that needs the smaller correction is the one the tube is on.
        if best is None or np.max(np.abs(delta)) < np.max(np.abs(best)):
            best = delta
    mount.index = mount.index + best
    mount.query()
    return best


def describe(state):
    pts = points(state)
    model = current_model(state)
    if model is None:
        return (f"{len(pts)} star point{'s' * (len(pts) != 1)}, no model yet - "
                f"'calibrate on stars' or add stars in different parts of the sky")
    pole = model.R @ np.array([0.0, 0.0, 1.0])
    ha, dec = np.degrees(np.arctan2(pole[1], pole[0])), np.degrees(np.arcsin(pole[2]))
    return (f"{model.describe()} | polar axis points at ha {ha:+.1f} dec {dec:+.1f}")


# ---- camera calibration from solves ----

def turning_axis(s0, s1, n=7, margin=0.1):
    """The rotation that carried the tube from frame s0 to frame s1, from the whole field.

    Every pixel is rigidly attached to the tube, so pixel i looked along u_i before and w_i after,
    with w_i = M u_i for the one rotation M the axis made. Its axis is the mount axis itself - in
    the sky frame, wherever the tripod points it - and its angle is how far the axis REALLY turned.

    Returns (unit axis, angle deg, weight): the weight is how strongly the fit may trust the axis,
    relative to one solved pointing - a small turn seen over a small field locates its axis only
    roughly, and that is what the fit is told."""
    from .model import _kabsch

    xs = np.linspace(margin, 1 - margin, n) * (s0.width - 1)
    ys = np.linspace(margin, 1 - margin, n) * (s0.height - 1)
    px = [np.array([x, y]) for y in ys for x in xs]
    u = np.array([s0.vector(p) for p in px])
    w = np.array([s1.vector(p) for p in px])
    M = _kabsch(u, w)
    rv = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    angle = float(np.degrees(np.arctan2(np.linalg.norm(rv) / 2, (np.trace(M) - 1) / 2)))
    axis = rv / max(np.linalg.norm(rv), 1e-12)
    c = u.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((u - c / np.linalg.norm(c)) ** 2, axis=1))))
    return axis, angle, np.radians(angle) * spread

def _solve_at(mount, cam, solver, log):
    try:
        sol = solve_camera(cam, solver, log=log, after_move=True)
    except SolveError as e:
        # haze drifting through, a gust: one bad frame should not throw away the whole run
        log(f"{cam.name}: {e} - trying one more frame")
        sol = solve_camera(cam, solver, log=log)
    return sol, np.asarray(mount.position_at(sol.t), dtype=float)


def calibrate_on_stars(mount, cam, solver, state, step_deg=None, track_rate=None, log=print,
                       abort=None, slew_rate=0.5, warnings=None, settle_s=0.5):
    """Measure the guide camera's matrix by solving frames either side of a move on each axis.

    The ramp of calibrate_cameras, with the blob replaced by the whole sky: each frame is
    solved on its own, so there is nothing to lose between steps, the move can be degrees, and
    no target has to stay in view. The pixel where a fixed direction lands in each frame gives
    J directly, and the four solved frames double as alignment points.

    Returns the camera calibration dict (J, dec_cal, boresight).
    """
    def check_abort():
        if abort and abort():
            raise RuntimeError("calibration aborted")

    cal_old = (state.get("cameras") or {}).get(cam.name) or {}
    b = np.array([(cam.width - 1) / 2, (cam.height - 1) / 2])     # see the module docstring
    # Small by default: from a balcony the window frame leaves little room, and each frame is
    # solved on its own, so the stars need not stay in view - only the precision scales with it.
    step = float(step_deg or cam.cfg.get("star_cal_step_deg") or 0.06 * field_deg(cam.cfg)[0])
    start = mount.position()

    sol, axes = _solve_at(mount, cam, solver, log)
    off = disagreement_deg(state, axes, *sol.hadec(b))
    if off is not None and off > RESYNC_DEG:
        raise RuntimeError(
            f"the model puts this pointing {off:.1f} deg from where the stars say it is - the "
            f"counters are stale. 'sync on stars' first (or 'clear alignment' if the tripod "
            f"moved), then calibrate.")

    moves, shots, dirs = [], [], []
    for axis in (0, 1):
        d = np.zeros(2)
        d[axis] = step
        check_abort()
        # Take up the slack in the + direction on BOTH axes, so every solve of the start pose
        # finds the gears the same way round. Preloading only the axis about to move left the
        # other one wherever the last move put it - 10' of Dec slack on the real mount.
        mount.move_to(start - step, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        time.sleep(settle_s)
        s0, a0 = _solve_at(mount, cam, solver, log)
        check_abort()
        mount.move_to(start + d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        time.sleep(settle_s)
        s1, a1 = _solve_at(mount, cam, solver, log)
        # the direction that sat on the boresight before the move: where is it now?
        moves.append((s1.pixel_hadec(*s0.hadec(b)) - b, a1 - a0))
        shots += [(s0, a0), (s1, a1)]
        v, turned, weight = turning_axis(s0, s1)
        commanded = float((a1 - a0)[axis])
        dirs.append({"kind": ("ra", "dec")[axis], "axis1": float(a0[0]), "v": v.tolist(),
                     "weight": weight})
        log(f"{cam.name}: axis{axis + 1} really turned {turned:.3f} deg for {commanded:.3f} "
            f"commanded ({turned / commanded:.3f}x)")
        if warnings is not None and abs(turned / commanded - 1) > 0.03:
            warnings.append(f"axis{axis + 1} turned {turned / commanded:.3f}x what the counters "
                            f"said (backlash if the move was small, otherwise gear_ratio: "
                            f"multiply it by {commanded / turned:.4f})")
        log(f"{cam.name}: axis{axis + 1} +{step:.2f} deg moved the sky "
            f"{np.linalg.norm(moves[-1][0]):.0f} px")
        mount.move_to(start - step, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)

    dp = np.column_stack([m[0] for m in moves])
    da = np.column_stack([m[1] for m in moves])
    J = dp @ np.linalg.inv(da)
    skew = axes_angle(J)
    if abs(skew - 90) > 1.0:
        log(f"{cam.name}: axes measured {skew:.1f} deg apart, squared up to 90")
        J = orthogonalise(J)
    if abs(skew - 90) > 5.0 and warnings is not None:
        warnings.append(f"{cam.name}: axes measured {skew:.1f} deg apart on the stars - an axis "
                        f"slipped or stuck during the moves. Redo it.")

    # the same starting pose solved twice: how well the mount comes back to the same place
    rep = angle_arcsec(shots[0][0].vector(b), shots[2][0].vector(b)) / 3600.0
    log(f"{cam.name}: returning to the start differed by {rep * 60:.1f}' on the sky")

    for s, a in shots:
        add_point(state, a, *s.hadec(b), s.t, source=f"starcal-{cam.name}")
    state["alignment"].setdefault("axis_dirs", []).extend(dirs)
    model, shift = refit(state, mount, state.get("cameras"))
    if shift:
        log(f"Dec index corrected by {shift:+.2f} deg - axis2 now reads the true declination")
    log(f"alignment: {describe(state)}")

    dec_cal = float(geo.axis2_to_dec(mount.position()[1]))
    # From the solves' own scale: J also carries how far the axes really turned, and on a
    # mount whose RA gives short measure that reads as a shorter lens.
    arcsec = float(np.median([s.scale_arcsec() for s, _ in shots]))
    scale = 3600.0 / arcsec
    focal = cam.cfg["focal_length_mm"] * scale / pixels_per_deg(cam.cfg)
    log(f"{cam.name}: {3600 / scale:.2f}\"/px = focal length {focal:.2f} mm "
        f"(config says {cam.cfg['focal_length_mm']:g})")
    if warnings is not None and abs(focal / cam.cfg["focal_length_mm"] - 1) > 0.02:
        warnings.append(f"{cam.name}: the stars say the focal length is {focal:.1f} mm, not "
                        f"{cam.cfg['focal_length_mm']:g} - set focal_length_mm = {focal:.1f} in "
                        f"config.toml. Unlike an indoor calibration this is not muddled by "
                        f"target distance or gearing.")
    # J is measured at the centre; the boresight is carried forward untouched
    return {"J": J.tolist(), "dec_cal": dec_cal,
            "boresight": list(cal_old.get("boresight", b.tolist())), "source": "stars"}


def boresight_on_star(sol, cal_main, cal_guide, px_main, tol_deg=1.5):
    """Guide boresight from a star seen in the main camera, identified by the solve.

    The star's guide pixel comes from the catalogue through the solve, not from a blob detector,
    so it cannot be confused with a brighter neighbour; and a star is at infinity, so there is
    no parallax between the cameras. Which catalogue star it is: the one nearest to where the
    CURRENT boresight says the main camera's star should appear.

    Returns (boresight, label, miss_deg, carried_px)."""
    old = np.asarray(cal_guide["boresight"], dtype=float)
    pred = old - boresight_from_picks(cal_main, cal_guide, px_main, [0.0, 0.0])[0]
    stars = sol.catalog()
    if not stars:
        raise ValueError("no catalogue stars in the solved field")
    ppd = cal_px_per_deg(cal_guide)
    ranked = sorted(stars, key=lambda s: np.hypot(*(s[0] - pred)))
    px, label, _ = ranked[0]
    miss = float(np.hypot(*(px - pred)) / ppd)
    if miss > tol_deg:
        raise ValueError(f"no catalogue star within {tol_deg:g} deg of where the main camera's "
                         f"star should be (nearest: {label}, {miss:.1f} deg). Centre a bright "
                         f"star in the main camera first - 'goto' one, then 'centre by solve'.")
    if len(ranked) > 1:
        other = float(np.hypot(*(ranked[1][0] - pred)) / ppd)
        if other < 2 * max(miss, 0.1):
            raise ValueError(f"ambiguous: {label} ({miss:.2f} deg) and {ranked[1][1]} "
                             f"({other:.2f} deg) are both near where the main camera's star "
                             f"should be. Centre a star with no close neighbour.")
    bore, carried = boresight_from_picks(cal_main, cal_guide, px_main, px)
    return bore, label, miss, carried
