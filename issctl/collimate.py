"""Newtonian collimation with a camera in the focuser: an electronic Cheshire.

The ASI120MM Mini has a 1.25" body, so with a short M12 lens it slides into the eyepiece holder
and looks down the focuser at the secondary. One picture holds four circles, and collimating is
making three pairs of them concentric:

  S  the secondary mirror's outline        S on the focuser axis  secondary under the focuser
  P  the primary's edge, in the secondary  P on S                 secondary tilt (its 3 screws)
  D  the primary's centre donut            D on C                 primary tilt (its screws)
  C  the camera's reflection in the primary (the dark spot of its lens, or a white collar)

The focuser axis is not the sensor centre - the camera never sits dead centre in the holder - so
it is measured: turn the camera 180 deg in the holder, and whatever stayed put is on the axis.
D on C needs no reference at all: the primary sends the camera's own reflection straight back.

A circle is picked by clicking three or more points on its edge, then follows that edge frame to
frame: rays from the centre, the strongest step of the same sign along each, a robust fit. Blur
does not matter - a blurred circle still has a sharp centre - which is why one lens can serve the
secondary's edge at 15 cm and the donut at 80 cm.
"""

import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .camera import Camera

FEATURES = {"S": "secondary outline", "P": "primary edge", "D": "centre donut",
            "C": "camera reflection"}
PICK_HINTS = {
    "S": " The secondary's own outline, round everything else - NOT the dark disk in the middle,"
         " which is its reflection. It shows only with white paper behind the secondary.",
    "P": " The rim of the bright disk, between the clips.",
    "D": " The donut's outer edge, in the middle of the dark disk.",
    "C": " The dark spot of the lens, in the white card's reflection.",
}
COLOURS = {"S": (255, 170, 60), "P": (80, 200, 255), "D": (90, 90, 255), "C": (230, 110, 230)}
GRADE_COLOURS = {"good": (80, 200, 80), "close": (40, 200, 230), "adjust": (70, 70, 235)}
ARROW_GAIN = 5.0
EDGE_SCALE = 0.005          # slope measured over this share of a circle's radius (1-3 px)          # offsets are drawn this many times longer than they are
LAMBDA_MM = 550e-6


# ---------------------------------------------------------------- geometry

def fit_circle(pts):
    """Least-squares circle through points: an algebraic start, then a few geometric steps
    (the algebraic fit alone pulls the radius in on a short, noisy arc)."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = a / 2, b / 2
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    for _ in range(8):
        dx, dy = x - cx, y - cy
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        step, *_ = np.linalg.lstsq(J, -(d - r), rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-4:
            break
    return float(cx), float(cy), float(abs(r))


def robust_fit(pts, min_sigma=0.5):
    """Circle fit that throws out what is not on the circle - mirror clips, spider vanes, a
    reflection crossing the edge. Returns ((cx, cy, r), inlier mask, rms) or None."""
    pts = np.asarray(pts, float)
    keep = np.ones(len(pts), bool)
    circle = None
    # Loose first, then tight: on the real primary a tenth of the rays stopped on the clips,
    # 22 px inside the rim, all on one side. They pulled the first fit 2 px off, a 3-sigma cut
    # round that threw out the rim itself, and the fit fell apart.
    floors = (12.0, 6.0, 3.0, 0.0, 0.0, 0.0)
    for floor in floors:
        if keep.sum() < 3:
            return None
        circle = fit_circle(pts[keep])
        res = np.hypot(pts[:, 0] - circle[0], pts[:, 1] - circle[1]) - circle[2]
        k = res[keep]
        sigma = max(1.4826 * float(np.median(np.abs(k - np.median(k)))), min_sigma)
        new = np.abs(res) < max(3.0 * sigma, floor)
        if floor == 0.0 and (new == keep).all():
            break
        keep = new
    if keep.sum() < 3:
        return None
    rms = float(np.sqrt(np.mean((np.hypot(pts[keep, 0] - circle[0], pts[keep, 1] - circle[1])
                                 - circle[2]) ** 2)))
    return circle, keep, rms


def prepare(frame):
    """Float image scaled 0..1 and slightly smoothed, for edge finding."""
    img = np.asarray(frame, np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.percentile(img[::4, ::4], (1, 99.5))
    img = (img - float(lo)) / max(float(hi - lo), 1e-6)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def edge_points(img, cx, cy, r, band, polarity=0, n=240, near=None):
    """Where each of n rays from (cx, cy) crosses an edge within r +- band.

    polarity +1 takes only edges that get brighter outwards, -1 darker, 0 either. near (px)
    favours steps close to r over stronger ones further off: the primary's edge sits a few px
    inside the secondary's, and is the stronger of the two. Returns the points, their strength
    and the sign of each step, for the rays that found a clear one."""
    h, w = img.shape
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.arange(max(r - band, 1.0), r + band + 0.5, 0.5)
    X = (cx + np.outer(np.cos(th), rs)).astype(np.float32)
    Y = (cy + np.outer(np.sin(th), rs)).astype(np.float32)
    inside = (X >= 0) & (X <= w - 1) & (Y >= 0) & (Y <= h - 1)
    prof = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # The slope measured over a few px, not between neighbouring samples: a soft edge (the rim a
    # little out of focus, 8 px from bright to dark) has a flat-topped slope whose steepest
    # sample wandered with the noise - the primary's edge jumped 4 px between two fits, frame
    # by frame. Smoothed along the ray, its peak sits still.
    if EDGE_SCALE:
        sigma = float(np.clip(EDGE_SCALE * r, 1.0, 3.0)) / (rs[1] - rs[0])     # in samples
        prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=sigma, sigmaY=1e-3)
    d = np.gradient(prof, axis=1)
    score = d * polarity if polarity else np.abs(d)
    if near:
        score = score * np.exp(-0.5 * ((rs - r) / near) ** 2)
    score = np.where(inside, score, -np.inf)
    score[:, [0, -1]] = -np.inf                 # a peak on the band's edge is not a crossing
    k = np.argmax(score, axis=1)
    rows = np.arange(n)
    best = score[rows, k]
    usable = np.isfinite(best) & (k > 0) & (k < len(rs) - 1)
    # sub-sample peak: a parabola through the best step and its two neighbours
    km, kp = np.clip(k - 1, 0, len(rs) - 1), np.clip(k + 1, 0, len(rs) - 1)
    s0, s1, s2 = score[rows, km], best, score[rows, kp]
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(usable, 0.5 * (s0 - s2) / (s0 - 2 * s1 + s2), 0.0)
    off = np.clip(np.nan_to_num(off), -0.5, 0.5)
    radius = rs[k] + off * (rs[1] - rs[0])
    if usable.any():
        top = float(np.percentile(best[usable], 80))
        usable &= best > max(0.3 * top, 0.004)
    pts = np.column_stack([cx + np.cos(th) * radius, cy + np.sin(th) * radius])
    return pts[usable], best[usable], np.sign(d[rows, k])[usable]


def follow(img, circle, band=None, polarity=0, min_inliers=12, near=None):
    """Move a circle onto its edge in a new frame. Returns (circle, polarity, quality) or None.

    quality is the share of rays that ended on the circle: low when the edge is faint, half out
    of the frame, or covered."""
    cx, cy, r = circle
    band = band if band is not None else min(max(0.08 * r, 4.0), 25.0)
    if near is None and polarity:
        near = max(0.5 * band, 6.0)   # following: an edge where the circle was beats one further off
    pts, _, signs = edge_points(img, cx, cy, r, band, polarity, near=near)
    if len(pts) < min_inliers:
        return None
    fit = robust_fit(pts)
    if fit is None:
        return None
    new, keep, rms = fit
    if keep.sum() < min_inliers or abs(new[2] - r) > band:
        return None
    if not polarity:
        polarity = int(np.sign(np.median(signs[keep]))) or 1
    return new, polarity, float(keep.sum()) / 240.0


def snap(img, clicks):
    """A circle from points clicked on its edge, moved onto the edge itself.

    Returns (circle, polarity, quality, locked): when no clear edge is found, the circle through
    the clicks is kept, unlocked, so the reading can still be used by eye."""
    clicks = np.asarray(clicks, float)
    c0 = fit_circle(clicks)
    if not np.isfinite(c0).all() or c0[2] < 3:
        return None
    scatter = float(np.abs(np.hypot(clicks[:, 0] - c0[0], clicks[:, 1] - c0[1]) - c0[2]).max())
    band = min(max(scatter + 6.0, 0.06 * c0[2], 6.0), max(0.3 * c0[2], 6.0))
    got = follow(img, c0, band=band, near=max(scatter + 2.0, 3.0))   # the edge you clicked
    if got is None:
        return c0, 0, 0.0, False
    circle, pol, q = got
    again = follow(img, circle, polarity=pol)     # tighter band, now the sign is known
    if again is not None:
        circle, pol, q = again
    return circle, pol, q, True


def rotation_axis(a, b):
    """Where the focuser axis falls in the image, from frames taken before and after turning the
    camera 180 deg in the holder.

    Points on the axis keep their pixel whatever the camera's turn, so b is a turned through
    180 deg about that pixel. Turned about the frame centre instead, a lands shifted from b by
    twice the axis' offset from that centre. Returns ((x, y), response): a low response means
    the two frames did not match - the turn was not 180 deg, or the light changed."""
    fa, fb = prepare(a), prepare(b)
    h, w = fa.shape
    turned = np.ascontiguousarray(fa[::-1, ::-1])
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(turned, np.ascontiguousarray(fb), win)
    return ((w - 1) / 2 + dx / 2, (h - 1) / 2 + dy / 2), float(response)


def coma_tolerance_mm(focal_mm, aperture_mm):
    """How far the primary's axis may miss the focuser axis at the focal plane before coma at the
    field centre grows to the size of the Airy disk: tangential coma 3 theta / (16 F^2) against
    2.44 lambda / D, i.e. 0.0072 F^3 mm (0.9 mm at f/5)."""
    F = focal_mm / aperture_mm
    return 2.44 * LAMBDA_MM * 16.0 * F ** 3 / 3.0


def direction(dx, dy):
    """Which way, in the picture (x right, y down), an offset points."""
    parts = []
    if abs(dx) >= 0.5:
        parts.append(f"{abs(dx):.1f} px {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 0.5:
        parts.append(f"{abs(dy):.1f} px {'down' if dy > 0 else 'up'}")
    return " and ".join(parts) or "on it"


# ---------------------------------------------------------------- the live tool

class Collimator:
    """The collimation page's state: picked circles, the axis, the readings."""

    def __init__(self, cam, ccfg, path=None, sim=False):
        self.cam, self.path, self.sim = cam, path, sim
        saved = {}
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text())
            except ValueError:
                saved = {}
        self.lens_mm = float(saved.get("lens_mm") or ccfg.get("lens_mm")
                             or cam.cfg.get("focal_length_mm") or 12.0)
        self.focal_mm = float(ccfg.get("focal_mm", 750.0))
        self.aperture_mm = float(ccfg.get("aperture_mm", 150.0))
        self.pixel_um = float(cam.cfg.get("pixel_um", 3.75)) * cam.cfg.get("bin", 1)
        self.axis = tuple(saved["axis"]) if saved.get("axis") else None
        self.circles = {k: dict(v, locked=False) for k, v in (saved.get("circles") or {}).items()
                        if k in FEATURES}
        self.tracking = True
        self.picking, self.clicks = None, []
        self.turn = None                  # first frame of the 180 deg turn test
        self.msg = "Pick the circles: S, P, D and C." if not self.circles else ""
        self._seq = None
        self._img = None
        self._lock = threading.RLock()
        self.saved_camera = saved.get("camera") or {}    # exposure/gain last used here
        self.peak = None           # brightest part of the frame, 0..1 of full scale
        self.clipped = 0.0         # share of pixels at full scale
        self.auto = None           # what automatic exposure is doing, while it runs

    # -- persistence
    def save(self):
        if self.path is None:
            return
        data = {"lens_mm": self.lens_mm,
                "camera": {"exposure_ms": self.cam.exposure_ms, "gain": self.cam.gain}, "axis": list(self.axis) if self.axis else None,
                "circles": {k: {f: v[f] for f in ("x", "y", "r", "pol")}
                            for k, v in self.circles.items()},
                "at": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)

    # -- frames
    def update(self):
        """Follow every picked circle into the newest frame (once per frame, however many
        browsers are watching). Returns the frame."""
        frame, _, seq = self.cam.latest()
        if frame is None:
            return None
        with self._lock:
            if seq == self._seq:
                return frame
            self._seq = seq
            self._img = prepare(frame)
            full = 255.0 if frame.dtype == np.uint8 else float(np.iinfo(frame.dtype).max) \
                if np.issubdtype(frame.dtype, np.integer) else float(frame.max() or 1)
            sub = np.asarray(frame)[::4, ::4]
            self.peak = float(np.percentile(sub, 99.5)) / full
            self.clipped = float(np.mean(sub >= 0.98 * full))
            if self.tracking:
                for c in self.circles.values():
                    circle = (c["x"], c["y"], c["r"])
                    got = follow(self._img, circle, polarity=c["pol"])
                    if got is None:
                        # a brisk screw turn moves a small circle further than its band in one
                        # frame: look wider, with the edge's sign to keep off its neighbours
                        got = follow(self._img, circle, band=min(max(0.5 * c["r"], 8.0), 40.0),
                                     polarity=c["pol"])
                    if got is None:
                        c["locked"] = False
                        continue
                    # a big jump is caught only part of the way - the rays on the far side miss
                    # the edge - so go again from there until it stops moving
                    for _ in range(4):
                        moved = math.hypot(got[0][0] - circle[0], got[0][1] - circle[1])
                        circle = got[0]
                        if moved < 0.2:
                            break
                        got = follow(self._img, circle, polarity=c["pol"]) or got
                    (c["x"], c["y"], c["r"]), c["pol"], c["q"] = got
                    c["locked"] = True
        return frame

    def axis_or_centre(self):
        if self.axis:
            return self.axis, True
        return ((self.cam.width - 1) / 2, (self.cam.height - 1) / 2), False

    # -- browser actions
    def action(self, name, q):
        with self._lock:
            if name == "pick" and q.get("what") in FEATURES:
                self.picking, self.clicks = q["what"], []
                self.msg = (f"Click 3 or more points on the edge of {FEATURES[self.picking]} "
                            f"({self.picking}), spread round it, then Done."
                            + PICK_HINTS.get(self.picking, ""))
            elif name == "click":
                self._click(float(q["fx"]) * self.cam.width, float(q["fy"]) * self.cam.height)
            elif name == "done":
                if self.picking and self.picking in self.circles:
                    self.save()
                self.picking, self.clicks = None, []
                self.msg = ""
            elif name == "forget" and q.get("what") in FEATURES:
                self.circles.pop(q["what"], None)
                self.save()
            elif name == "tracking":
                self.tracking = q.get("on") not in ("0", "false")
            elif name == "lens":
                self.lens_mm = max(0.5, float(q["mm"]))
                self.save()
            elif name == "turn_first":
                self._turn_first()
            elif name == "turn_second":
                self._turn_second()
            elif name == "save_frame":
                self._save_frame()
            elif name == "auto_exposure":
                if self.auto is None:
                    self.auto = "starting"
                    threading.Thread(target=self._auto_exposure, name="collimate-exposure",
                                     daemon=True).start()
            elif name == "axis_reset":
                self.axis, self.turn = None, None
                self.msg = "Axis forgotten: the frame centre stands in for it."
                self.save()
            elif self.sim and name.startswith("sim_"):
                self.msg = self.cam.sim_action(name, q) or ""
            return self.state()

    def _save_frame(self):
        """The raw frame as the camera gave it (no stretch, no drawing), for a closer look."""
        frame, _, _ = self.cam.latest()
        if frame is None:
            self.msg = "No frame to save."
            return
        folder = (self.path.parent.parent if self.path else Path(".")) / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"collimation-{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(str(f), np.asarray(frame))
        self.msg = f"Saved {f}"

    def _auto_exposure(self, target=0.75, tries=10):
        """Set the exposure so the brightest parts of the picture sit at ~3/4 of full scale:
        edges need contrast, and a clipped edge has none. Gain goes to its lowest first: this is
        daylight, and the cleanest picture is the one with the least gain."""
        cam = self.cam
        try:
            if cam.gain:
                cam.set_gain(0)
            mine = (cam.exposure_ms, cam.gain)
            for _ in range(tries):
                # judge a frame exposed after the last change. The ASI keeps handing over frames
                # taken before it for a while: at 28 fps two frames on were still the old
                # exposure, and Auto quartered it down to nothing on frames that were clipped
                _, _, seq0 = cam.latest()
                t0 = time.monotonic()
                while True:
                    if (cam.exposure_ms, cam.gain) != mine:
                        # set by hand meanwhile: the hand wins, and the wait sized from the
                        # old exposure must not hold Auto up for half a minute
                        self.msg = "Auto exposure stopped: exposure or gain was set by hand."
                        return
                    el, exp_s = time.monotonic() - t0, cam.exposure_ms / 1000.0
                    if cam.latest()[2] >= seq0 + 5 and el >= max(0.4, 3 * exp_s):
                        break
                    if el > 5 * exp_s + 3.0:
                        break
                    time.sleep(0.02)
                if cam.latest()[2] < seq0 + 5:
                    self.msg = "Auto exposure: no frames from the camera."
                    return
                self._seq = None
                self.update()
                peak, clipped = self.peak, self.clipped
                if clipped > 0.002 or peak >= 0.97:
                    factor = 0.25                # clipped: how far over is unknown, so a big step
                elif peak < 0.02:
                    factor = 8.0
                else:
                    factor = min(max(target / peak, 0.25), 8.0)
                if abs(factor - 1) < 0.15:
                    self.msg = f"Auto exposure: {cam.exposure_ms:.3g} ms, brightest at {100 * peak:.0f}%."
                    self.save()
                    return
                ms = min(max(cam.exposure_ms * factor, 0.032), 4000.0)
                if ms == cam.exposure_ms:
                    self.msg = (f"Auto exposure stopped at the camera's limit, {ms:.3g} ms: "
                                + ("raise the gain or add light." if factor > 1 else "lower the gain."))
                    return
                self.auto = f"{ms:.3g} ms"
                cam.set_exposure(ms)
                mine = (cam.exposure_ms, cam.gain)
            self.msg = "Auto exposure did not settle - is the light changing?"
        except Exception as e:
            self.msg = f"Auto exposure failed: {e}"
        finally:
            self.auto = None

    def _click(self, x, y):
        if not self.picking:
            self.msg = "Choose which circle first (S, P, D or C), then click its edge."
            return
        c = self.circles.get(self.picking)
        if c and c.get("locked") and len(self.clicks) >= 3:
            off = abs(math.hypot(x - c["x"], y - c["y"]) - c["r"])
            if off > max(15.0, 0.15 * c["r"]):
                # seen for real: the next circle's clicks landing in this one, Done not pressed,
                # and a locked primary edge dragged into a 629 px circle through the middle
                self.msg = (f"That click is {off:.0f} px off the edge of {FEATURES[self.picking]} - "
                            "not used. Press Done, then the next circle's button.")
                return
        self.clicks.append((x, y))
        if len(self.clicks) < 3:
            self.msg = f"{FEATURES[self.picking]}: {len(self.clicks)} point(s), need 3 or more."
            return
        if self._img is None:
            self.update()
        got = snap(self._img, self.clicks)
        if got is None:
            self.msg = "Those points do not make a circle - spread them round the edge."
            return
        (x0, y0, r), pol, q, locked = got
        self.circles[self.picking] = {"x": x0, "y": y0, "r": r, "pol": pol, "q": q,
                                      "locked": locked}
        self.msg = (f"{FEATURES[self.picking]}: r = {r:.0f} px, "
                    + ("locked onto the edge" if locked else
                       "no clear edge there - kept where you clicked, not followed")
                    + ". More clicks refine it; Done when happy.")

    def _turn_first(self):
        frame, _, _ = self.cam.latest()
        if frame is None:
            self.msg = "No frame yet."
            return
        self.turn = frame.copy()
        self.msg = ("Now turn the camera 180 deg in the holder (use a mark on its body), "
                    "tighten the screw, and press Second frame.")

    def _turn_second(self):
        frame, _, _ = self.cam.latest()
        if self.turn is None or frame is None:
            self.msg = "Take the first frame first."
            return
        (ax, ay), resp = rotation_axis(self.turn, frame)
        self.turn = None
        if resp < 0.05:
            self.msg = (f"The two frames do not match (response {resp:.2f}): was the turn 180 deg "
                        "and the light steady? Try again.")
            return
        self.axis = (ax, ay)
        # the picked circles turned with the camera: put them where they are now
        for c in self.circles.values():
            c["x"], c["y"] = 2 * ax - c["x"], 2 * ay - c["y"]
        cx, cy = (self.cam.width - 1) / 2, (self.cam.height - 1) / 2
        self.msg = (f"Focuser axis at ({ax:.1f}, {ay:.1f}): {math.hypot(ax - cx, ay - cy):.1f} px "
                    f"from the frame centre (match {resp:.2f}).")
        self.save()

    # -- readings
    def steps(self):
        """The three readings: one centre against another (or against the focuser axis).

        Each carries its offset in its own unit (vx, vy, value) with its good/close limits, so
        the page can draw it as a target, and both centres in pixels for the arrows."""
        axis, measured = self.axis_or_centre()
        axis = axis if measured else None   # the frame centre is no stand-in for a reading:
        #                                     the M12 holder alone can put it 140 px off
        tol = coma_tolerance_mm(self.focal_mm, self.aperture_mm)
        S, P, D, C = (self._xy(k) for k in "SPDC")
        r = {k: v["r"] for k, v in self.circles.items()}
        # the camera sees D and its own reflection 2e/f apart, e = how far the primary's axis
        # passes from the camera (the focuser axis) at the focal plane
        mm_per_px = self.pixel_um / 1000.0 / self.lens_mm * self.focal_mm / 2.0
        out = []

        def step(key, title, mirror, screws, ref, ref_name, tgt, tgt_name, unit, scale,
                 good, close, explain):
            item = {"key": key, "title": title, "mirror": mirror, "screws": screws,
                    "have": False, "unit": unit, "good": good, "close": close,
                    "ref_name": ref_name, "tgt_name": tgt_name}
            needs = [n for n, v in ((ref_name, ref), (tgt_name, tgt)) if v is None]
            if needs:
                item["text"] = "needs " + " and ".join(
                    "the focuser axis (turn test)" if n == "axis" else f"circle {n}"
                    for n in needs)
                out.append(item)
                return
            dx, dy = tgt[0] - ref[0], tgt[1] - ref[1]
            px = math.hypot(dx, dy)
            value = px * scale
            item.update(have=True, dx=round(dx, 2), dy=round(dy, 2), px=round(px, 2),
                        vx=round(dx * scale, 3), vy=round(dy * scale, 3), value=round(value, 3),
                        ref_xy=[round(ref[0], 1), round(ref[1], 1)],
                        tgt_xy=[round(tgt[0], 1), round(tgt[1], 1)],
                        grade="good" if value <= good else "close" if value <= close else "adjust",
                        locked=all(self.circles.get(n, {}).get("locked", True)
                                   for n in (ref_name, tgt_name) if n in FEATURES),
                        text=explain(value, px),
                        where=f"{tgt_name} is {direction(dx, dy)} of {ref_name}")
            out.append(item)

        step("focuser", "1. Secondary under the focuser", "secondary",
             "centre bolt and turn (rarely needed)", axis, "axis", S, "S", "%",
             100.0 / r["S"] if S else 0, 3.0, 8.0,
             lambda v, px: f"{v:.1f}% of the secondary's radius ({px:.1f} px)")
        # P on S, or - while the secondary's outline is not picked - P on the focuser axis:
        # either way the question is whether the focuser looks at the primary's centre
        ref2, name2 = (S, "S") if S else (axis, "axis")
        step("secondary", "2. Secondary tilt", "secondary", "3 tilt screws",
             ref2, name2, P, "P", "%", 100.0 / r["P"] if P else 0, 1.5, 4.0,
             lambda v, px: f"{v:.1f}% of the primary's radius ({px:.1f} px)")
        step("primary", "3. Primary tilt", "primary", "collimation screws",
             C, "C", D, "D", "mm", mm_per_px, 0.5 * tol, tol,
             lambda v, px: f"axes {v:.2f} mm apart at focus ({px:.1f} px); "
                           f"coma = Airy disk at {tol:.2f} mm")
        return out

    def _xy(self, k):
        v = self.circles.get(k)
        return (v["x"], v["y"]) if v else None

    def state(self):
        self.update()          # the readings must follow the screws with no picture open too
        (ax, ay), measured = self.axis_or_centre()
        return {"picking": self.picking, "clicks": len(self.clicks), "tracking": self.tracking,
                "lens_mm": self.lens_mm, "focal_mm": self.focal_mm,
                "aperture_mm": self.aperture_mm, "axis": [round(ax, 1), round(ay, 1)],
                "axis_measured": measured, "turn_pending": self.turn is not None,
                "circles": {k: {"x": round(v["x"], 2), "y": round(v["y"], 2),
                                "r": round(v["r"], 1), "locked": v.get("locked", False)}
                            for k, v in self.circles.items()},
                "steps": self.steps(), "msg": self.msg, "sim": self.sim, "auto": self.auto,
                "peak": None if self.peak is None else round(100 * self.peak, 1),
                "clipped": round(100 * self.clipped, 2),
                "cam": {"name": self.cam.name, "exposure_ms": self.cam.exposure_ms,
                        "gain": self.cam.gain, "fps": round(self.cam.fps, 1)}}

    # -- picture
    def render(self, max_width=1280):
        frame = self.update()
        if frame is None:
            return None
        img = np.asarray(frame)
        if img.ndim == 2:
            lo, hi = np.percentile(img[::4, ::4], (0.5, 99.8))
            img = np.clip((img.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        k = min(1.0, max_width / img.shape[1])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        with self._lock:
            self._draw(img, k)
        return img

    def _draw(self, img, k):
        def p(x, y):
            return int(round(x * k)), int(round(y * k))

        (ax, ay), measured = self.axis_or_centre()
        axis_col = (0, 220, 0) if measured else (160, 160, 160)
        cv2.drawMarker(img, p(ax, ay), axis_col, cv2.MARKER_CROSS, 48, 2 if measured else 1)
        cv2.putText(img, "axis" if measured else "frame centre", (p(ax, ay)[0] + 8, p(ax, ay)[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_col, 1, cv2.LINE_AA)
        for name, c in self.circles.items():
            col = COLOURS[name]
            thick = 3 if name == self.picking else 2
            style = cv2.LINE_AA if c.get("locked", False) else cv2.LINE_4
            cv2.circle(img, p(c["x"], c["y"]), max(1, int(round(c["r"] * k))), col, thick, style)
            cv2.drawMarker(img, p(c["x"], c["y"]), col, cv2.MARKER_TILTED_CROSS, 10, 1)
            lx, ly = p(c["x"] + 0.71 * c["r"], c["y"] - 0.71 * c["r"])
            cv2.putText(img, name if c.get("locked", False) else name + "?", (lx + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        for n, st in enumerate(self.steps(), 1):
            if not st["have"]:
                continue
            ref, tgt = st["ref_xy"], st["tgt_xy"]
            end = (ref[0] + ARROW_GAIN * (tgt[0] - ref[0]), ref[1] + ARROW_GAIN * (tgt[1] - ref[1]))
            col = GRADE_COLOURS[st["grade"]]
            if math.hypot(end[0] - ref[0], end[1] - ref[1]) >= 3:
                cv2.arrowedLine(img, p(*ref), p(*end), col, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(img, str(n), (p(*end)[0] + 6, p(*end)[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        for x, y in self.clicks:
            cv2.circle(img, p(x, y), 4, (0, 255, 255), -1)


# ---------------------------------------------------------------- simulator

class SimCollimationCamera(Camera):
    """A synthetic view down a slightly miscollimated Newtonian's focuser, with a white card
    opposite the focuser: secondary outline, the primary in it with its clips, the secondary's
    reflection with the vanes, the camera's collar and lens, and the donut. The camera can be
    turned in its holder and the screws nudged, from the page."""

    def __init__(self, name, cam_cfg, clock, fps=10.0, seed=2):
        super().__init__(name, cam_cfg, clock)
        self.period = 1.0 / fps
        self.rng = np.random.default_rng(seed)
        w, h = self.width, self.height
        self.axis = (0.5 * w + 11.3, 0.5 * h - 7.6)    # the camera sits a little off-centre
        self.turned = 0.0
        self.secondary = np.array([4.0, -3.0])         # S centre - axis
        self.tilt2 = np.array([9.0, 5.5])              # P centre - S centre
        self.tilt1 = np.array([-7.0, 4.0])             # D centre - C centre
        self.r_s = 0.31 * h
        self._next = time.monotonic()

    def scene(self, turned=None):
        turned = self.turned if turned is None else turned
        h, w = self.height, self.width
        ax = np.array(self.axis)
        S = ax + self.secondary
        P = S + self.tilt2
        C = ax + 0.5 * self.secondary            # the camera's reflection, near the axis
        D = C + self.tilt1
        r_s = self.r_s
        r_p = 0.965 * r_s
        img = np.full((h, w), 140, np.uint8)     # the white card behind the secondary
        disk = lambda c, r: cv2.circle(np.zeros((h, w), np.uint8), (int(round(c[0] * 16)),
                                       int(round(c[1] * 16))), int(round(r * 16)), 1, -1,
                                       cv2.LINE_AA, 4).astype(bool)
        in_s = disk(S, r_s)
        img[in_s] = 38                           # tube wall seen in the secondary
        prim = np.full((h, w), 175, np.uint8)    # the lit primary
        for a in (np.radians(30), np.radians(150), np.radians(270)):   # mirror clips
            u = np.array([math.cos(a), math.sin(a)])
            v = np.array([-u[1], u[0]])
            pts = np.array([P + u * (r_p + 4) + v * 16, P + u * (r_p - 12) + v * 16,
                            P + u * (r_p - 12) - v * 16, P + u * (r_p + 4) - v * 16])
            cv2.fillPoly(prim, [np.round(pts).astype(np.int32)], 40)
        sr = 0.36 * r_s                          # the secondary's reflection, round C
        cv2.circle(prim, tuple(np.round(C).astype(int)), int(sr), 30, -1, cv2.LINE_AA)
        for a in np.radians([45, 135, 225, 315]):   # spider vanes
            u = np.array([math.cos(a), math.sin(a)])
            cv2.line(prim, tuple(np.round(C + u * sr).astype(int)),
                     tuple(np.round(C + u * 2 * r_p).astype(int)), 30, 3, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(C).astype(int)), int(0.27 * r_s), 150, -1, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(D).astype(int)), int(0.14 * r_s), 235, -1, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(D).astype(int)), int(0.065 * r_s), 150, -1, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(C).astype(int)), int(0.04 * r_s), 12, -1, cv2.LINE_AA)
        m = in_s & disk(P, r_p)
        img[m] = prim[m]
        if turned:
            R = cv2.getRotationMatrix2D((float(ax[0]), float(ax[1])), turned, 1.0)
            img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
        return img

    def truth(self):
        """Where the circles really are in the picture (for tests): axis, S, P, C, D."""
        ax = np.array(self.axis)
        S = ax + self.secondary
        C = ax + 0.5 * self.secondary
        out = {"axis": ax, "S": S, "P": S + self.tilt2, "C": C, "D": C + self.tilt1}
        if self.turned:
            a = math.radians(-self.turned)       # warpAffine turns the picture counterclockwise
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            out = {k: ax + R @ (v - ax) for k, v in out.items()}
        return out

    def frame(self):
        img = self.scene() * (self.exposure_ms / 10.0) * (1 + self.gain / 100.0) / 2.0
        img = img + self.rng.normal(0, 2.0, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _grab(self):
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.period, time.monotonic())
        return self.frame(), self.clock.now()

    def sim_action(self, name, q):
        if name == "sim_turn":
            self.turned = (self.turned + 180.0) % 360.0
            return "Simulated camera turned 180 deg in the holder."
        if name == "sim_nudge":
            which = q.get("what")
            attr = {"secondary": "tilt2", "primary": "tilt1"}.get(which)
            if attr is None:
                return None
            v = getattr(self, attr)
            # a screw turn takes off about a third, and pushes a little sideways
            side = np.array([-v[1], v[0]]) * self.rng.normal(0, 0.15)
            setattr(self, attr, v * 0.65 + side)
            return f"Simulated {which} screw turned."
        return None
