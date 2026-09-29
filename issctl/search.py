"""Spiral search: find a star in the main camera when it is not in the field.

The main camera sees a few arcminutes - 12.9' x 7.3' with the Barlow - so a goto, or a guide
boresight that has never been measured, easily leaves the star just outside it. The search walks a
square spiral of sky offsets round where it started, one main-field step at a time, and looks at a
fresh main frame at each stop.

Three things make it trustworthy on the real mount:

* every stop is approached from the same side (move_to's `approach`), so the ~10' of Dec slack -
  more than one step - cannot leave holes in the pattern;
* the step fits the main field whatever way the camera is turned: a square of side
  short/sqrt(2) fits inside the frame at any rotation, and the spiral is laid out in axes;
* a detection only counts once the star has MOVED with the mount: a nudge on axis2 must carry it
  across the frame by about the expected number of pixels. A hot pixel, a reflection or a glint
  stays put. Anything that stays put is remembered and ignored for the rest of the search.

Once the star is in, `calibrate_on_star` measures the main camera's matrix from the same kind of
moves, in main's own pixels. Calibrating it against the guide does not work: the moves that keep
a star inside 7' shift the guide image by 1-4 px, and a ratio against that is 15-20% out (seen on
the rig, 2026-09-29).
"""

import time

import numpy as np

from . import geometry as geo
from .calib import SkyAnchor, axes_angle, axes_offset_from_pixel, orthogonalise, pixels_per_deg
from .detect import detect
from .solve import _fresh_timed

SLACK_DEG = 0.2          # approach allowance when the backlash has not been measured
STATIC_PX = 6.0          # a blob this close to one seen at another stop has not moved


def search_step_deg(cam_cfg, fill=0.9):
    """The largest square step that stays inside the frame at any camera rotation."""
    short = min(cam_cfg["width"], cam_cfg["height"]) / pixels_per_deg(cam_cfg)
    return fill * short / np.sqrt(2.0)


def spiral_offsets(step, radius):
    """(u, v) sky offsets in degrees, a square spiral out from (0, 0) to `radius`."""
    rings = int(np.ceil(radius / step - 1e-9))
    yield 0.0, 0.0
    x = y = 0
    for ring in range(1, rings + 1):
        x += 1                                       # step out to the next ring, then go round it
        yield x * step, y * step
        for dx, dy, n in ((0, 1, 2 * ring - 1), (-1, 0, 2 * ring), (0, -1, 2 * ring), (1, 0, 2 * ring)):
            for _ in range(n):
                x, y = x + dx, y + dy
                yield x * step, y * step


def _axes_offset(u, v, axis2):
    """Sky offset (u along axis1's motion, v along axis2) as an axis move: axis1 moves the sky
    only cos(dec) as far, and not at all at the pole."""
    c = max(abs(float(np.cos(np.radians(geo.axis2_to_dec(axis2))))), 0.2)
    return np.array([u / c, v])


def _find(cam, frame, ignore):
    """Brightest star-like blob that is not a known static one, full-resolution pixels."""
    cfg = cam.cfg
    img = np.array(frame, copy=True)
    for x, y in ignore:
        r = int(4 * STATIC_PX)
        ya, xa = max(0, int(y) - r), max(0, int(x) - r)
        img[ya:int(y) + r + 1, xa:int(x) + r + 1] = np.median(img)
    det = detect(img, sigma=cfg.get("detect_sigma", 6.0), min_area=cfg.get("detect_min_area", 3),
                 bayer=bool(cfg.get("bayer")), max_area=cfg.get("detect_max_area", 20000),
                 edge_margin=cfg.get("detect_edge_margin", 8))
    return None if det is None else np.array([det.x, det.y])


class Search:
    def __init__(self, mount, cam, track_rate=None, log=print, abort=None, radius_deg=0.5,
                 step_deg=None, slack_deg=None, slew_rate=0.5, settle_s=0.3):
        self.mount, self.cam, self.log = mount, cam, log
        self.abort = abort or (lambda: False)
        self.track_rate = track_rate
        self.radius = float(radius_deg)
        self.step = float(step_deg or search_step_deg(cam.cfg))
        slack = SLACK_DEG if slack_deg is None else slack_deg
        self.approach = np.maximum(np.broadcast_to(np.asarray(slack, dtype=float), (2,)), 0.02)
        self.slew_rate, self.settle_s = slew_rate, settle_s
        self.static = []       # blobs that did not move with the mount

    def _go(self, anchor, d):
        self.mount.move_to(anchor(d), track_rate=self.track_rate, abort=self.abort,
                           max_rate=self.slew_rate, approach=self.approach)
        time.sleep(self.settle_s)

    def _look(self):
        """A frame exposed entirely after the move, and what is in it."""
        exp_s = float(getattr(self.cam, "exposure_ms", 0.0) or 0.0) / 1000.0
        not_before = self.mount.clock.now() + 0.5 * exp_s
        frame, _ = _fresh_timed(self.cam, timeout=10.0 + 3 * exp_s, not_before=not_before)
        return _find(self.cam, frame, self.static)

    def _moves_with_mount(self, anchor, d, px):
        """Nudge axis2 and check the blob travels about as far as the optics say it should."""
        ppd = pixels_per_deg(self.cam.cfg)
        nudge = 0.2 * self.cam.height / ppd
        expected = nudge * ppd
        for sign in (1.0, -1.0):      # the other way if the first one pushed it off the frame
            self._go(anchor, d + np.array([0.0, sign * nudge]))
            after = self._look()
            if after is None:
                continue
            moved = float(np.hypot(*(after - px)))
            if 0.6 * expected < moved < 1.5 * expected:
                self._go(anchor, d)
                back = self._look()
                return back if back is not None and np.hypot(*(back - px)) < 0.3 * expected else px
            if moved < STATIC_PX:
                break
        self._go(anchor, d)
        return None

    def run(self):
        """Returns a dict describing the hit (the mount left pointing at it), or None."""
        anchor = SkyAnchor(self.mount, self.track_rate)
        axis2 = float(self.mount.position()[1])
        offsets = list(spiral_offsets(self.step, self.radius))
        self.log(f"spiral search in {self.cam.name}: {len(offsets)} stops of "
                 f"{self.step * 60:.1f}', out to {self.radius * 60:.0f}'")
        for k, (u, v) in enumerate(offsets):
            if self.abort():
                self.log("spiral search aborted - the mount stays where it is")
                return None
            d = _axes_offset(u, v, axis2)
            self._go(anchor, d)
            px = self._look()
            if px is None:
                if k % 10 == 9:
                    self.log(f"spiral search: {k + 1}/{len(offsets)} stops, nothing yet")
                continue
            self.log(f"spiral search: something at ({px[0]:.0f}, {px[1]:.0f}) at stop {k + 1}, "
                     f"{u * 60:+.1f}' {v * 60:+.1f}' - checking it moves with the mount")
            confirmed = self._moves_with_mount(anchor, d, px)
            if confirmed is None:
                self.static.append(px)
                self.log(f"spiral search: ({px[0]:.0f}, {px[1]:.0f}) does not move with the mount "
                         f"- a hot pixel or reflection, ignoring it")
                continue
            self.log(f"spiral search: FOUND at stop {k + 1}/{len(offsets)}, "
                     f"{u * 60:+.1f}' {v * 60:+.1f}' from the start, at pixel "
                     f"({confirmed[0]:.0f}, {confirmed[1]:.0f})")
            return {"stop": k + 1, "offset_arcmin": [u * 60, v * 60], "axes_offset": d.tolist(),
                    "px": confirmed.tolist()}
        self._go(anchor, np.zeros(2))
        self.log(f"spiral search: nothing within {self.radius * 60:.0f}' - back at the start. "
                 f"Check the star is in the guide frame near the boresight, or widen the search")
        return None


def _look_mean(search, n):
    """The star's position averaged over n fresh frames, or None if any frame missed it."""
    pts = [search._look() for _ in range(n)]
    return None if any(p is None for p in pts) else np.mean(pts, axis=0)


def _take_up_slack(s, anchor, cam, log):
    """Arrive at the start from below on both axes, the way every later move will.

    The star was centred with the gears wherever the last move left them; taking up the slack
    shifts the pointing by as much as the slack itself - 10' on the rig's Dec, more than the whole
    main field. If the star has gone, walk UP (never down, which would undo it) until it is back,
    and measure from there. After a spiral search nothing moves: it already left the slack
    taken up this way."""
    s._go(anchor, -1.2 * s.approach)
    s._go(anchor, np.zeros(2))
    if s._look() is not None:
        return anchor
    hop = 0.4 * min(cam.width, cam.height) / pixels_per_deg(cam.cfg)
    for k in range(1, int(np.ceil(1.5 * s.approach[1] / hop)) + 1):
        s._go(anchor, np.array([0.0, k * hop]))
        if s._look() is not None:
            log(f"{cam.name}: the Dec slack moved the star {k * hop * 60:.1f}' - found it again")
            return SkyAnchor(s.mount, s.track_rate)
    raise RuntimeError(f"taking up the slack lost the star in {cam.name} - run 'spiral search in "
                       f"main' (it leaves the slack taken up the same way), then calibrate")


def calibrate_on_star(mount, cam, track_rate=None, log=print, abort=None, slack_deg=None,
                      existing=None, step_deg=None, frames=3, slew_rate=0.3, settle_s=0.5,
                      warnings=None):
    """The main camera's matrix from moving a star it can see, in its own pixels.

    Each axis goes to -s, 0 and +s about where it started, always arriving from the same side, so
    the Dec slack drops out; a line through the three star positions is that axis's column.
    Returns the calibration dict (J, dec_cal, axis2_cal, boresight), the star left centred on the
    boresight."""
    warnings = [] if warnings is None else warnings
    s = Search(mount, cam, track_rate=track_rate, log=log, abort=abort, slack_deg=slack_deg,
               slew_rate=slew_rate, settle_s=settle_s)
    ppd = pixels_per_deg(cam.cfg)
    step = float(step_deg or 0.25 * cam.height / ppd)       # a quarter frame: ~270 px at 1500 mm
    anchor = SkyAnchor(mount, track_rate)
    axis2 = float(mount.position()[1])

    def at(d):
        if s.abort():
            raise RuntimeError("calibration aborted")
        s._go(anchor, np.asarray(d, dtype=float))
        p = _look_mean(s, frames)
        if p is None:
            raise RuntimeError(f"lost the star in {cam.name} - centre it and try again "
                               f"(it has to stay in view for {step * 60:.1f}' either way)")
        return p

    anchor = _take_up_slack(s, anchor, cam, log)
    start = at([0.0, 0.0])
    log(f"{cam.name}: star at ({start[0]:.0f}, {start[1]:.0f}); moving each axis "
        f"{step * 60:.1f}' either way")
    cols, resid = [], []
    for axis in (0, 1):
        offs = np.array([-step, 0.0, step])
        pts = []
        for o in offs:
            d = np.zeros(2)
            d[axis] = o
            pts.append(at(d))
        pts = np.array(pts)
        slope = np.polyfit(offs, pts, 1)[0]                 # px per degree, both coordinates
        fit = pts - (np.outer(offs, slope) + (pts - np.outer(offs, slope)).mean(axis=0))
        cols.append(slope)
        resid.append(float(np.max(np.hypot(*fit.T))))
        log(f"{cam.name}: axis{axis + 1} {np.linalg.norm(slope):.0f} px/deg, "
            f"straight to {resid[-1]:.1f} px")
    J = np.column_stack(cols)
    back = at([0.0, 0.0])
    skew = axes_angle(J)
    cos_img = float(np.linalg.norm(J[:, 0]) / np.linalg.norm(J[:, 1]))
    cos_mount = abs(float(np.cos(np.radians(geo.axis2_to_dec(axis2)))))
    scale = float(np.linalg.norm(J[:, 1]) / ppd)
    log(f"{cam.name}: axes {skew:.1f} deg apart; {3600 / np.linalg.norm(J[:, 1]):.3f}\"/px "
        f"(optics say {3600 / ppd:.3f}, x{scale:.3f}); axis1/axis2 {cos_img:.3f} vs cos(dec) "
        f"{cos_mount:.3f}; came back within {np.hypot(*(back - start)):.1f} px")

    if abs(skew - 90) > 10:
        raise RuntimeError(f"axes measured {skew:.0f} deg apart - an axis slipped or the star was "
                           f"lost for a moment. Nothing saved; redo it.")
    if abs(skew - 90) > 1:
        J = orthogonalise(J)
    if abs(skew - 90) > 3:
        warnings.append(f"{cam.name}: axes {skew:.1f} deg apart - squared up, but redo it if "
                        f"you can")
    if abs(scale - 1) > 0.05:
        warnings.append(f"{cam.name}: image scale x{scale:.3f} of what focal_length_mm = "
                        f"{cam.cfg['focal_length_mm']:g} says - {cam.cfg['focal_length_mm'] * scale:.0f}"
                        f" mm? (Barlow spacing changes it)")
    if abs(cos_img - cos_mount) > 0.1:
        warnings.append(f"{cam.name}: axis1 moves the image {cos_img:.2f}x as far as axis2, the "
                        f"counters say cos(dec) = {cos_mount:.2f} - sync on stars, then redo it")
    if max(resid) > 15 or np.hypot(*(back - start)) > 30:
        warnings.append(f"{cam.name}: the moves were not clean (off a straight line by "
                        f"{max(resid):.0f} px, came back {np.hypot(*(back - start)):.0f} px off) "
                        f"- wind, a loose clutch or a snagged cable")

    # The aim point in main is its frame centre. The whole main field is ~9 x 16 guide pixels,
    # so a hand-set offset there buys nothing the guide could ever resolve; what matters is that
    # the GUIDE boresight marks where this centre looks.
    bore = [(cam.width - 1) / 2, (cam.height - 1) / 2]
    cal = {"J": J.tolist(), "dec_cal": float(geo.axis2_to_dec(axis2)), "axis2_cal": axis2,
           "boresight": bore, "source": "star"}
    # centre it: two moves, each arriving from the same side like the rest
    here = back
    for _ in range(2):
        d = axes_offset_from_pixel(cal, axis2, here)
        if np.hypot(*(here - bore)) < 10:
            break
        s._go(anchor, d)
        anchor = SkyAnchor(mount, track_rate)
        found = _look_mean(s, 1)
        if found is None:
            break
        here = found
    log(f"{cam.name}: calibrated on the star; it is {np.hypot(*(here - np.array(bore))):.0f} px "
        f"from the boresight")
    return cal
