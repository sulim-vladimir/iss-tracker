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
"""

import time

import numpy as np

from . import geometry as geo
from .calib import SkyAnchor, pixels_per_deg
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
