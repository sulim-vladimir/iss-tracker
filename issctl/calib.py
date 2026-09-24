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
    # A matrix is what you MEASURED at dec_cal, and at dec_cal one axis1 degree is worth only
    # cos(dec_cal) degrees of sky. jacobian() undoes this again at the declination in use.
    J[:, 0] *= max(abs(np.cos(np.radians(dec_cal))), 0.05)
    w, h = cam_cfg["width"] // cam_cfg["bin"], cam_cfg["height"] // cam_cfg["bin"]
    return {"J": J.tolist(), "dec_cal": dec_cal, "boresight": [(w - 1) / 2, (h - 1) / 2]}


def cal_px_per_deg(cal):  # noqa: D401
    """Image scale in pixels per degree on the sky (axis2 is a pure sky rotation)."""
    return float(np.linalg.norm(np.array(cal["J"], dtype=float)[:, 1]))


def axis1_factor(cal, axis2):
    """What the stored axis1 column must be multiplied by to be valid at this axis2.

    Only the CHANGE in declination since the matrix was measured, never the absolute value. Both
    ends come from the same step counter, so a mount that has no idea where it is cancels itself
    out and this is exactly 1 until you slew.
    """
    dec = float(geo.axis2_to_dec(axis2))
    k = np.cos(np.radians(dec)) / max(np.cos(np.radians(cal.get("dec_cal", 0.0))), 0.05)
    return float(np.sign(k) * max(abs(k), 0.05))


def jacobian(cal, axis2):
    J = np.array(cal["J"], dtype=float)
    J[:, 0] *= axis1_factor(cal, axis2)
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
    if not axis1_plausible(cal) or axis1_stretch(cal, axis2) > 3.0:
        return None     # axis1 cannot be trusted here - fall back to driving the axes directly
    if axis1_worth_now(cal, axis2) < min_cos_dec:
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


class BlobTracker:
    """Where the target is, from the detected blob. Needs something point-like in view."""

    mode = "blob"
    absolute = True          # positions mean the same thing in every camera

    def __init__(self, cam):
        self.cam = cam

    def reset(self):
        pass

    def measure(self, n=10, timeout=5.0):
        return measure(self.cam, n, timeout)

    def describe(self):
        return "no target detected"


def prepare_frame(img, bayer=False, gain=None):
    """8-bit grey for the flow tracker, binned 2x2 if the sensor is Bayer.

    A Bayer frame's colour checkerboard is a strong gradient at every pixel, which is exactly what
    a gradient-following tracker would lock onto. Binning removes it; the shift then comes back in
    binned pixels, so the caller scales it. `gain` fixes the brightness normalisation across a
    run - rescaling each frame by its own peak would turn a passing headlight into apparent
    motion of everything else.
    """
    g = np.asarray(img)
    if g.ndim == 3:
        g = g.mean(axis=2)
    g = g.astype(np.float32)
    scale = 1
    if bayer:
        h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
        g = g[:h:2, :w:2] + g[1:h:2, :w:2] + g[:h:2, 1:w:2] + g[1:h:2, 1:w:2]
        scale = 2
    if gain is None:
        gain = 255.0 / max(float(g.max()), 1e-6)
    return np.clip(g * gain, 0, 255).astype(np.uint8), scale, gain


SELECT_BLUR = 3.0       # smoothing applied ONLY when choosing corners, never when tracking them
SELECT_BLOCK = 11       # gradient window for the corner score; 3 is far too small on a grainy frame


def pick_corners(img, max_points=15, mask=None):
    """Choose corners to follow, on a smoothed copy of the frame.

    A high-gain frame is grainy, and grain has a sharper local gradient than a real edge spread
    over tens of pixels - so the raw score picks noise over structure, which is what put the
    markers in the middle of a blank bright wall. Smoothing first fixes that: measured on a
    focused frame, corners landing on structure that survives heavy smoothing went from 9/15 to
    12/15. The smoothing is for the CHOICE only; tracking runs on the sharp image, because that
    is where the sub-pixel accuracy lives.
    """
    import cv2

    soft = cv2.GaussianBlur(img, (0, 0), SELECT_BLUR)
    return cv2.goodFeaturesToTrack(soft, maxCorners=max_points, qualityLevel=0.01,
                                   minDistance=8, mask=mask, blockSize=SELECT_BLOCK)


def scene_corners(frame, bayer=False, max_points=15, region=None):
    """The corners scene calibration would follow in this frame, in full-resolution pixels.

    The preview draws these, so what you see on the image is what actually gets tracked - not a
    separate guess at it. Returns an empty array when the frame has too little structure, which
    is the honest answer and the one worth seeing before you press the button.
    """
    import cv2

    img, scale, _ = prepare_frame(frame, bayer)
    mask = None
    if region:
        x, y, r = region
        mask = np.zeros(img.shape, np.uint8)
        cv2.circle(mask, (int(x / scale), int(y / scale)), max(int(r / scale), 8), 255, -1)
    pts = pick_corners(img, max_points, mask)
    return np.zeros((0, 2)) if pts is None else pts.reshape(-1, 2) * scale


class FeatureTracker:
    """How far the scene has moved, from a handful of corners tracked with optical flow.

    For a camera with no point source in view. Blob detection needs a star or a lamp; from a
    balcony there may be none, just lit windows or scenery. Lucas-Kanade follows image gradients
    directly, so it works on structure that is soft and low contrast - which phase correlation
    also does, but this is roughly twice as accurate on a real frame and, unlike correlation, it
    can tell a field that ROTATED from one that shifted.

    Measured on a real main-camera frame (blurry lit windows, 0.4"/px), error in sensor pixels
    over one calibration step: best single corner 0.45, best 5 0.27, best 20 0.26, all 300 0.37.
    More is not better - weak corners drag the answer down - so it keeps only the strongest few.
    A deliberately weak corner gave 3.76 px and lost lock 2 times in 12, which is the whole case
    for using several rather than one.

    `region` (x, y, radius) restricts it to one patch. Nothing in the UI sets it: it exists for
    the day a scene with real depth needs it - the camera swings on a ~0.3 m radius, so objects
    at different distances shift by different amounts - but it costs accuracy, so the whole
    frame is the default.

    What it CANNOT do is tell two cameras they are looking at the same thing - each tracker's
    origin is its own reference frame. That is enough for J and not enough for the boresight.
    """

    mode = "scene"
    absolute = False         # positions are relative to this camera's own reference frame

    def __init__(self, cam, origin=None, region=None, max_points=15, fb_tolerance=2.0,
                 min_points=3, min_tracked=0.4):
        self.cam = cam
        self.origin = np.array(origin if origin is not None else
                               [(cam.width - 1) / 2, (cam.height - 1) / 2], dtype=float)
        self.region = region
        self.max_points = max_points
        self.fb_tolerance = fb_tolerance
        self.min_points = min_points
        self.min_tracked = min_tracked
        self.ref = None
        self.points = None
        self.response = None
        self.reason = None

    def _prepare(self, img):
        gain = None if self.ref is None else self._gain
        out, scale, self._gain = prepare_frame(img, getattr(self.cam, "bayer", False), gain)
        return out, scale

    def _mask(self, shape):
        if not self.region:
            return None
        import cv2

        x, y, r = self.region
        s = 2 if getattr(self.cam, "bayer", False) else 1
        m = np.zeros(shape, np.uint8)
        cv2.circle(m, (int(x / s), int(y / s)), max(int(r / s), 8), 255, -1)
        return m

    def describe(self):
        """Why the last attempt went the way it did - the SDK-style silence is no use here."""
        if self.ref is None:
            return self.reason or "no reference frame"
        n = 0 if self.points is None else len(self.points)
        kept = "" if self.response is None else f", {self.response * 100:.0f}% of them held"
        return f"following {n} corners{kept}"

    def reset(self, timeout=5.0):
        """Take the reference frame and choose the corners to follow."""
        import cv2

        frame = _fresh_frame(self.cam, timeout)
        if frame is None:
            self.reason = "no frame arrived from the camera"
            return False
        self.ref = None                       # so _prepare recomputes the normalisation
        ref, self._scale = self._prepare(frame)
        pts = pick_corners(ref, self.max_points, self._mask(ref.shape))
        if pts is None or len(pts) < self.min_points:
            self.ref, self.points = None, None
            self.reason = (f"only {0 if pts is None else len(pts)} corners to follow"
                           + (" inside the region you clicked" if self.region else " in the frame")
                           + " - too little structure, or it needs focusing")
            return False
        self.ref, self.points = ref, pts
        self.reason = None
        return True

    def _flow(self, cur):
        """Displacement of the corners, keeping only those that survive a round trip.

        Optical flow does not fail loudly: it will report a confident answer for points it has
        completely lost. Tracking back to the reference and discarding whatever fails to return
        to where it started is what turns that into an honest measurement.
        """
        import cv2

        lk = dict(winSize=(21, 21), maxLevel=4)
        fwd, st1, _ = cv2.calcOpticalFlowPyrLK(self.ref, cur, self.points, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, self.ref, fwd, None, **lk)
        fb = np.linalg.norm((back - self.points).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_tolerance)
        if ok.sum() < self.min_points or ok.mean() < self.min_tracked:
            self.reason = (f"only {int(ok.sum())} of {len(self.points)} corners survived the "
                           f"round trip - the scene moved too far, or changed")
            return None, float(ok.mean())
        d = (fwd[ok] - self.points[ok]).reshape(-1, 2)
        return np.median(d, axis=0), float(ok.mean())

    def measure(self, n=10, timeout=5.0):
        if self.ref is None and not self.reset(timeout):
            return None
        shifts, kept, deadline = [], [], time.monotonic() + timeout
        last = self.cam.latest()[2]
        while len(shifts) < n and time.monotonic() < deadline:
            frame, _, seq = self.cam.latest()
            if seq != last and frame is not None:
                last = seq
                cur, scale = self._prepare(frame)
                if cur.shape != self.ref.shape:
                    return None
                d, frac = self._flow(cur)
                kept.append(frac)
                if d is not None:
                    shifts.append(d * scale)
            time.sleep(0.005)
        self.response = float(np.median(kept)) if kept else 0.0
        if len(shifts) < max(3, n // 2):
            return None
        return self.origin + np.median(shifts, axis=0)


def _fresh_frame(cam, timeout=5.0):
    """The next frame to arrive, not whatever is sitting in the buffer from before a move."""
    last = cam.latest()[2]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame, _, seq = cam.latest()
        if seq != last and frame is not None:
            return frame
        time.sleep(0.005)
    return None


TRACKERS = {
    "blob":  "a point source: a star, a planet, a distant lamp",
    "scene": "corners of the scenery, followed by optical flow (needs some structure)",
}


def make_tracker(cam, mode="blob", origin=None):
    """A tracker for this camera.

    Scene mode follows the whole frame. It used to take the clicked target's gate as a region to
    restrict itself to, which was wrong twice over: there is no persistent "scene mode" for a
    click to belong to - the mode is an argument to one press of one button - and the gate is a
    DETECTION gate that re-centres on whatever blob is found inside it, so on scenery it wanders
    off the spot that was picked. `goodFeaturesToTrack` already returns the strongest corners in
    the frame, which measured better than any hand-picked patch anyway (0.26 px against 0.45 for
    a single chosen corner), so there is nothing to choose.
    """
    if mode == "scene":
        return FeatureTracker(cam, origin)
    return BlobTracker(cam)


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


def default_step(cam, ramp_steps=3, fraction=0.35):
    """Per-step angle for a ramp that keeps the target inside THIS camera's frame.

    The whole ramp must fit: starting near the centre, the target may travel about 40% of the
    frame height before it leaves. One size cannot serve both cameras - with a 16 mm guide lens
    (50 px/deg) and the main at 1500 mm (9000 px/deg) the same angle is 6 px in one and 1000 in
    the other.
    """
    total = fraction * cam.height / pixels_per_deg(cam.cfg)
    return float(np.clip(total / max(ramp_steps, 1), 0.002, 3.0))


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


def measured_cos_dec(cal):
    """cos(dec) as the IMAGE measured it: how far axis1 shifted the image against axis2.

    One axis1 degree shifts the image cos(dec) times as far as one axis2 degree - geometry, not
    calibration, so no focal length, gear ratio or camera rotation can change it. That makes the
    ratio of the two columns an estimate of cos(dec) that owes nothing to the mount's step
    counters, which is the only estimate available on a mount that was pushed round by hand.
    """
    J = np.asarray(cal["J"], dtype=float)
    return float(np.linalg.norm(J[:, 0]) / max(np.linalg.norm(J[:, 1]), 1e-9))


def axis1_plausible(cal, tol=1.15):
    """False when the axis1 column is impossible rather than merely surprising.

    cos(dec) cannot exceed 1, so an axis1 column LONGER than the axis2 one cannot be a
    translation at any declination. That happens when axis1 rotated the field about a point
    outside the frame - near the pole in a narrow camera - and the tracker reported the swing as
    a shift. A column shorter than axis2 is always possible: it is just a higher declination.
    """
    return measured_cos_dec(cal) <= tol


def axis1_worth_now(cal, axis2):
    """cos(dec) as it applies to the axis1 column at this pose, from the image where possible.

    The image's own estimate at calibration time, carried forward by the only part the mount is
    still good for. Near zero means axis1 rotates the field instead of shifting it, so a screen
    direction has no sensible mapping onto the axes.
    """
    return measured_cos_dec(cal) * abs(axis1_factor(cal, axis2))


def axis1_stretch(cal, axis2):
    """How far jacobian() is stretching the axis1 column from where it was measured.

    Turning pixels into axis degrees never needs the absolute declination - only the CHANGE in it
    since the matrix was measured. Both ends of that ratio come from the same step counter, so a
    mount that is not synced cancels itself out: this is exactly 1 when you have not slewed since
    calibrating, however wrong the counter is. It grows as you work further from where you
    calibrated, and cos() near the pole makes it grow very fast - which is what turns a small
    error in a near-pole axis1 column into a centring loop that walks away.
    """
    f = abs(axis1_factor(cal, axis2))
    return float(max(f, 1.0 / f))


def _unskewed(cal):
    """The camera matrix with the cos(dec_cal) taken back out of the axis1 column.

    Not a pointing: just both cameras expressed against the same yardstick so they can be
    compared. Which yardstick does not matter, as long as it is the same one.
    """
    J = np.asarray(cal["J"], dtype=float).copy()
    J[:, 0] /= max(abs(np.cos(np.radians(cal.get("dec_cal", 0.0)))), 0.05)
    return J


def boresight_from_picks(cal_main, cal_guide, px_main, px_guide):
    """Guide pixel that the main camera's boresight looks at, from one object picked in both.

    Only a point source lets the SOFTWARE decide that two cameras are looking at the same thing.
    A person looking at two pictures needs no point source: any corner you can recognise will do.

    Where the mount is pointing does not enter into it, and this is worth spelling out because it
    is the one calibration answer that is free of it. Each camera's matrix is J = P . A, with P the
    fixed pixels-per-degree-of-sky of that camera and A = diag(cos dec, 1) the axis-to-sky factor
    that both cameras share at any instant. The guide pixel per main pixel is therefore
    J_guide . J_main^-1 = P_guide . A . A^-1 . P_main^-1 = P_guide . P_main^-1, and A has gone.
    So a hand-pointed mount with no idea where it is answers this exactly as well as a synced one.

    Returns (boresight, carried_px). Exact when the object sits on the main boresight, and
    carried_px is then 0: nothing had to be carried across and neither matrix was consulted.
    Otherwise the offset travels through both matrices and their errors come with it, which is
    what carried_px measures.
    """
    aim = np.asarray(cal_main["boresight"], dtype=float)
    dth = np.linalg.solve(_unskewed(cal_main), aim - np.asarray(px_main, dtype=float))
    carried = _unskewed(cal_guide) @ dth
    return np.asarray(px_guide, dtype=float) + carried, float(np.linalg.norm(carried))


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
                      log=print, abort=None, slew_rate=0.3, measure_frames=10,
                      tracker=None, ref_tracker=None):
    """Calibrate a narrow camera by comparing it with an already-calibrated wide one.

    A narrow field cannot take a step big enough to beat the mount's backlash, so measuring it
    against the axes is hopeless. But both cameras see the SAME motion, whatever the axes actually
    did, so the ratio between their image shifts is clean: backlash, stiction and scale error all
    cancel. With S_cam and S_ref the measured shifts per commanded degree, and G the reference
    matrix, the answer is J = S_cam . S_ref^-1 . G.
    """
    step_deg = step_deg or default_step(cam, ramp_steps)
    tracker = tracker or BlobTracker(cam)
    ref_tracker = ref_tracker or BlobTracker(ref_cam)
    start = mount.position()
    s_cam, s_ref = [], []
    for axis in (0, 1):
        d = np.zeros(2)
        d[axis] = step_deg
        mount.move_to(start - d, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        mount.move_to(start, track_rate=track_rate, abort=abort, max_rate=slew_rate)
        time.sleep(0.4)
        tracker.reset()
        ref_tracker.reset()
        angles, here, there = [], [], []
        for k in range(ramp_steps + 1):
            if abort and abort():
                raise RuntimeError("calibration aborted")
            if k:
                mount.move_to(start + d * k, track_rate=track_rate, abort=abort, max_rate=slew_rate)
            time.sleep(0.4)
            a = tracker.measure(n=measure_frames)
            b = ref_tracker.measure(n=measure_frames)
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
        raise RuntimeError(f"{ref_cam.name} hardly moved during the ramp - the mount's slack is "
                           f"eating steps of {step_deg:.3f} deg; reduce the backlash first")
    return S_cam @ np.linalg.inv(S_ref) @ np.array(ref_cal["J"], dtype=float)


def calibration_order(cam_cfgs, existing=None):
    """Which camera to calibrate first, and which is the reference.

    A narrow camera cannot out-step the mount's backlash, so it is calibrated against the wide one,
    where the slack cancels in the ratio. That reference may be a STORED calibration - a camera's
    matrix only changes if the camera itself moves - and then the narrow camera goes FIRST, because
    its small moves keep the target inside every frame. Only without a stored reference must the
    wide camera go first, and the target then has to be steered back into the narrow field.
    """
    if not cam_cfgs:
        return [], None
    widest = min(cam_cfgs, key=lambda n: pixels_per_deg(cam_cfgs[n]))
    have_reference = bool(((existing or {}).get(widest) or {}).get("J"))
    order = sorted(cam_cfgs, key=lambda n: pixels_per_deg(cam_cfgs[n]), reverse=have_reference)
    return order, widest


def calibrate_cameras(mount, cams, steps=None, track_rate=None, log=print, abort=None,
                      warnings=None, slew_rate=0.5, only=None, existing=None,
                      ramp_steps=3, measure_frames=10, mode="blob"):
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
    trackers = {n: make_tracker(c, mode) for n, c in cams.items()}
    start = mount.position()
    # Measure every camera BEFORE moving anything: this is the only moment all of them are looking
    # at the same pose, so it is the only reliable basis for the guide->main boresight. Calibrating
    # a wide guide needs degrees of motion, which throws the target far outside the main frame, and
    # backlash means the mount does not come back precisely enough to re-measure afterwards.
    usable, missing, at_start = {}, [], {}
    for name, cam in cams.items():
        at_start[name] = trackers[name].measure()
        log(f"{name}: {trackers[name].describe()}")
        if at_start[name] is None:
            missing.append(name)
            continue
        usable[name] = cam
        steps.setdefault(name, default_step(cam, ramp_steps))
    if not usable:
        raise RuntimeError(
            f"nothing to measure in {' or '.join(cams)} - "
            + ("too little structure to follow - focus it, check the exposure, or click a "
               "region with more in it" if mode != "blob" else
               "adjust exposure/gain, focus, or click the target in the image. With no point "
               "source at all (a wall of lit windows, daylight scenery) calibrate on the scene "
               "instead, which follows the scenery rather than a target"))
    if missing:
        log(f"skipping {', '.join(missing)}: no target detected there")
        if warnings is not None:
            warnings.append(f"{', '.join(missing)} not calibrated (no target detected) - without the "
                            f"main camera there is no boresight, so the handoff is not set up")
    ordered, widest = calibration_order({n: c.cfg for n, c in usable.items()}, existing)
    cams = {n: usable[n] for n in ordered if only is None or n in only}
    if not cams:
        raise RuntimeError("; ".join(f"{n}: {trackers[n].describe()}" for n in only))
    log(f"calibrating {', '.join(cams)}")
    def check_abort():
        if abort and abort():
            raise RuntimeError("calibration aborted")

    cols = {name: [None, None] for name in cams}
    reference = widest if len(usable) > 1 else None
    log(f"order: {', '.join(cams)}"
        + (f" ({reference} is the reference)" if reference else ""))
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
        if ref_cal and reference in usable and trackers[reference].measure() is not None:
            # Steering the target back by pixel needs a position in a frame both cameras
            # share, which only a point source gives. There is nothing to recover otherwise:
            # the scenery is the target and it cannot leave.
            if trackers[name].absolute and not bring_into_view(
                    mount, cam, usable[reference], ref_cal, track_rate=track_rate,
                    log=log, abort=abort, slew_rate=slew_rate):
                log(f"{name}: target not in view and could not be recovered - skipped")
                if warnings is not None:
                    warnings.append(f"{name} not calibrated: the target never came back into its "
                                    f"field. Centre it ('send to main') and calibrate {name} alone.")
                continue
            log(f"{name}: calibrating against {reference} - the mount's slack cancels in the ratio")
            J = calibrate_against(mount, cam, usable[reference], ref_cal, track_rate=track_rate,
                                  log=log, abort=abort, slew_rate=slew_rate,
                                  measure_frames=measure_frames, ramp_steps=ramp_steps,
                                  tracker=trackers[name], ref_tracker=trackers[reference])
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
            time.sleep(0.4)
            # The reference frame is taken here, at the foot of the ramp with the slack already
            # taken up, so every point in the fit is measured against the same starting scene.
            trackers[name].reset()
            angles, points = [], []
            for k in range(ramp_steps + 1):
                check_abort()
                if k:
                    mount.move_to(start + d * k, track_rate=track_rate, abort=abort,
                                  max_rate=slew_rate)
                time.sleep(0.4)
                seen = trackers[name].measure(n=measure_frames)
                if seen is None:
                    raise RuntimeError(
                        f"lost the target in {name} after {k} steps of {step_deg:.3f} deg on "
                        f"axis{axis + 1}"
                        + ("" if mode == "blob" else
                           " - too little of the scene survived the move; use a smaller step, "
                           "or pick a region with more structure"))
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
        # Carry the stored boresight forward rather than resetting it to the frame centre. This
        # run may not be able to measure one - scene and phase modes never can, and blob mode
        # cannot when only one camera saw the target - and a boresight measured on a real point
        # source weeks ago is worth far more than a guess made now. The cross-camera block below
        # overwrites it on the runs that do measure it.
        prior = ((existing or {}).get(n) or {}).get("boresight")
        result[n] = {"J": J.tolist(), "dec_cal": dec_cal,
                     "boresight": list(prior) if prior is not None else
                     [(cam.width - 1) / 2, (cam.height - 1) / 2]}
        sv = np.linalg.svd(J, compute_uv=False)
        log(f"{n}: {3600 / max(sv.mean(), 1e-9):.2f} arcsec/px (scale {sv.round(1)} px/deg), "
            f"rotation {np.degrees(np.arctan2(J[1, 0], J[0, 0])):.1f} deg")

        ratio = measured_cos_dec(result[n])
        ok = axis1_plausible(result[n])
        if not ok and warnings is not None:
            warnings.append(
                f"{n}: axis1 moved the image {ratio:.2f}x as far as axis2, which no declination "
                f"allows - cos(dec) cannot exceed 1. That column is noise: axis1 rotated the "
                f"field about a point outside the frame and the tracker read the swing as a "
                f"shift. Centring and the image-frame arrows are off for {n} until it is redone "
                f"away from the pole.")
        elif warnings is not None and abs(ratio - np.cos(np.radians(dec_cal))) > 0.15:
            warnings.append(
                f"{n}: the image says axis1 was worth cos(dec) = {ratio:.2f}, i.e. dec "
                f"{np.degrees(np.arccos(min(ratio, 1.0))):.0f}deg, but the mount said dec "
                f"{dec_cal:.0f}deg. If the mount is not synced that is expected and harmless - "
                f"centring works as long as you do not slew far before using it. If it IS synced, "
                f"axis1 was measured too near the pole to mean anything.")
        expected, measured, factor = scale_check(J, cam.cfg, dec_cal)
        log(f"{n}: optics say {3600 / expected:.2f} arcsec/px; measured "
            f"{np.round(3600 / np.maximum(measured, 1e-9), 2)} arcsec/px "
            f"-> axis moved {1 / factor[0]:.2f}x / {1 / factor[1]:.2f}x of what was commanded; "
            f"= focal length {implied_focal_length(measured[1], cam.cfg):.0f} mm if the axes moved "
            f"exactly as commanded and the target is far away (config says "
            f"{cam.cfg['focal_length_mm']:g})")
        # A column that failed the geometry test above says nothing about gears or focal length,
        # and "multiply gear_ratio by 7.23" is advice that would wreck a working mount.
        scale_factors.append(factor if ok else [np.nan, factor[1]])

    if scale_factors and warnings is not None:
        factor = np.nanmean(scale_factors, axis=0)
        if np.any(np.abs(np.nan_to_num(factor, nan=1.0) - 1) > 0.15):
            axes = (f"x{factor[0]:.2f}/{factor[1]:.2f}" if np.isfinite(factor[0])
                    else f"x{factor[1]:.2f} on axis2 (axis1 not measurable here)")
            warnings.append(
                f"scale mismatch {axes}: EITHER focal_length_mm is wrong "
                f"(see the implied focal length above) OR the axes under/over-move, in which case "
                f"multiply gear_ratio by that factor. Check with axis-scale, and calibrate on a "
                f"DISTANT target.")
    # The boresight needs a matrix for each camera - freshly measured or from a previous run - and
    # both cameras looking at the same object right now.
    existing = existing or {}
    jm = result.get("main", existing.get("main"))
    jg = result.get("guide", existing.get("guide"))
    if mode != "blob" and jm and jg:
        log(f"{mode} mode: J measured, boresight kept as it was - following a camera's own "
            f"scenery says nothing about where the OTHER camera is looking")
        if warnings is not None:
            warnings.append(
                "boresight not measured: only a point source can tell that two cameras are looking at "
                "the same thing. The guide->main handoff still uses the stored boresight, so "
                "redo that part on a point source (a distant lamp, a planet, the Moon) at 1 km "
                "or more - closer than that the parallax between the two cameras exceeds the "
                "main camera's field.")
    elif jm and jg and at_start.get("main") is not None and at_start.get("guide") is not None:
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
