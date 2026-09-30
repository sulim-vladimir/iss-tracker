"""Plate solving: where the guide camera points, read off the stars themselves.

Everything else in the calibration path has to trust the mount's step counters, and on a mount
that is pushed by hand and never synced those counters are fiction. A solved frame is not: it
gives the true direction of every pixel, whatever the counters say and whichever way the tripod
faces. Stars are also at infinity, so there is no parallax between the two cameras.

astrometry.net's `solve-field` does the work, offline, against the Tycho-2 "4100" index files
in data/astrometry (`issctl solve-setup` fetches the ones this camera's field needs). Only the
guide camera is solved: at 16 mm it sees ~17 deg and hundreds of stars, whereas the main camera's
7' field holds a handful at best and would need gigabytes of index.

A Solution answers in the mount's own frame - local hour angle and declination, refracted, at
the moment the frame was exposed - because that is what the pointing model and the axes work in.
The catalogue (J2000) coordinates are kept for display.
"""

import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import predict as pr
from .calib import pixels_per_deg
from .config import ROOT
from .model import to_hadec
from .model import unit as sky_unit

INDEX_DIR = ROOT / "data" / "astrometry"
INDEX_URL = "http://data.astrometry.net/4100/index-{n}.fits"
# Tycho-2 index files: the range of quad sizes each one holds, in arcminutes. astrometry.net
# wants quads between ~10% and 100% of the field width.
INDEX_QUADS_ARCMIN = {
    4107: (22, 30), 4108: (30, 42), 4109: (42, 60), 4110: (60, 85), 4111: (85, 120),
    4112: (120, 170), 4113: (170, 240), 4114: (240, 340), 4115: (340, 480), 4116: (480, 680),
    4117: (680, 1000), 4118: (1000, 1400), 4119: (1400, 2000),
}


class SolveError(RuntimeError):
    pass


def field_deg(cam_cfg):
    """Field width and height in degrees, from the optics in the config."""
    s = pixels_per_deg(cam_cfg)
    return cam_cfg["width"] // cam_cfg["bin"] / s, cam_cfg["height"] // cam_cfg["bin"] / s


def needed_indexes(cam_cfg):
    w = field_deg(cam_cfg)[0] * 60
    return [n for n, (lo, hi) in INDEX_QUADS_ARCMIN.items() if hi >= 0.1 * w and lo <= w]


def missing_indexes(cam_cfg, index_dir=INDEX_DIR):
    return [n for n in needed_indexes(cam_cfg) if not (Path(index_dir) / f"index-{n}.fits").exists()]


def fetch_indexes(cam_cfg, index_dir=INDEX_DIR, log=print):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    for n in missing_indexes(cam_cfg, index_dir):
        url = INDEX_URL.format(n=n)
        log(f"fetching {url}")
        part = index_dir / f"index-{n}.fits.part"
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.rename(index_dir / f"index-{n}.fits")
    return needed_indexes(cam_cfg)


# ---- solutions ----

class Solution:
    """Where every pixel of one solved frame points. Subclasses supply hadec/pixel_hadec."""

    t = None          # unix time the frame was exposed
    width = height = 0

    def hadec(self, px):
        raise NotImplementedError

    def pixel_hadec(self, ha, dec):
        raise NotImplementedError

    def radec(self, px):
        return None

    def catalog(self):
        """Known stars in the frame: [(pixel, label, magnitude)], brightest first."""
        return []

    def vector(self, px):
        return sky_unit(*self.hadec(px))

    def centre(self):
        return np.array([(self.width - 1) / 2, (self.height - 1) / 2])

    def scale_arcsec(self):
        c = self.centre()
        a, b = self.vector(c), self.vector(c + [10.0, 0.0])
        return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) * 3600 / 10.0)

    def north_angle(self, px=None):
        """Image direction of celestial north at px, degrees from image 'up' (-y), + clockwise."""
        px = self.centre() if px is None else np.asarray(px, dtype=float)
        ha, dec = self.hadec(px)
        d = self.pixel_hadec(ha, min(float(dec) + 0.2, 89.9)) - px
        return float(np.degrees(np.arctan2(d[0], -d[1])))

    def describe(self, px=None):
        px = self.centre() if px is None else px
        ha, dec = self.hadec(px)
        rd = self.radec(px)
        where = (f"RA {rd[0] / 15:.3f}h Dec {rd[1]:+.2f} (J2000), " if rd is not None else "")
        return (f"{where}HA {float(ha):+.2f} Dec {float(dec):+.2f} now, "
                f"{self.scale_arcsec():.1f}\"/px, north {self.north_angle(px):+.0f} deg")


class WcsSolution(Solution):
    """A real solve: astrometry.net's WCS (with its lens-distortion polynomial) plus the site."""

    def __init__(self, header, site, t, width, height, stars=()):
        import warnings

        from astropy.wcs import WCS, FITSFixedWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            self.wcs = WCS(header)
        self.site, self.t = site, t
        self.width, self.height = width, height
        self.stars = list(stars)    # (ra, dec, mag, label)

    def radec(self, px):
        ra, dec = self.wcs.all_pix2world([px[0]], [px[1]], 0)
        return float(ra[0]), float(dec[0])

    def pixel_radec(self, ra, dec):
        x, y = self.wcs.all_world2pix([ra], [dec], 0)
        return np.array([float(x[0]), float(y[0])])

    def hadec(self, px):
        return pr.radec_to_hadec(*self.radec(px), self.site, self.t)

    def pixel_hadec(self, ha, dec):
        return self.pixel_radec(*pr.hadec_to_radec(ha, dec, self.site, self.t))

    def catalog(self):
        # Only stars near the field: projecting one from the far side of the sky sends the
        # distortion polynomial's inverse off to infinity.
        c = sky_unit(*self.radec(self.centre()))
        reach = np.cos(np.radians(0.6 * self.scale_arcsec() / 3600 * np.hypot(self.width, self.height)))
        out = []
        for ra, dec, mag, label in sorted(self.stars, key=lambda s: s[2]):
            if sky_unit(ra, dec) @ c < reach:
                continue
            try:
                px = self.pixel_radec(ra, dec)
            except Exception:       # astropy's NoConvergence, at the very edge of the lens
                continue
            if np.all(np.isfinite(px)) and 0 <= px[0] < self.width and 0 <= px[1] < self.height:
                # a named star is also in Tycho-2: keep the name, drop the duplicate
                if not any(np.hypot(*(px - q)) < 3.0 for q, _, _ in out):
                    out.append((px, label, mag))
        return out


class SimSolution(Solution):
    """What a solve would return in the simulator: the truth, as a gnomonic projection."""

    def __init__(self, cal, p_hat, e1, e2, t, width, height, stars=()):
        self.J0 = np.asarray(cal["J"], dtype=float)
        self.b = np.asarray(cal["boresight"], dtype=float)
        self.p, self.e1, self.e2 = (np.asarray(v, dtype=float) for v in (p_hat, e1, e2))
        self.t, self.width, self.height = t, width, height
        self.stars = list(stars)    # (unit vector, label, mag)

    def hadec(self, px):
        off = np.radians(-np.linalg.solve(self.J0, np.asarray(px, dtype=float) - self.b))
        v = self.p + np.tan(off[0]) * self.e1 + np.tan(off[1]) * self.e2
        ha, dec = to_hadec(v / np.linalg.norm(v))
        return float(ha), float(dec)

    def pixel_hadec(self, ha, dec):
        v = sky_unit(ha, dec)
        w = v / (v @ self.p)
        off = np.degrees(np.arctan([w @ self.e1, w @ self.e2]))
        return self.b - self.J0 @ off

    def catalog(self):
        out = []
        for v, label, mag in sorted(self.stars, key=lambda s: s[2]):
            if v @ self.p <= 0:
                continue
            px = self.pixel_hadec(*to_hadec(v))
            if 0 <= px[0] < self.width and 0 <= px[1] < self.height:
                out.append((px, label, mag))
        return out


# ---- solvers ----

def _mono(img):
    g = np.asarray(img, dtype=np.float32)
    return g.mean(axis=2) if g.ndim == 3 else g


def find_stars(img, max_stars=80, kernel=15, sigma=4.0, peak_sigma=6.0, max_area=1000, edge=4,
               tile=64, busy=2.0, lively_max=0.04):
    """Star positions, brightest first, as (x, y, flux) - and NOT the lit building next to them.

    solve-field's own extractor ranks sources by brightness, and from a balcony the brightest
    things in the frame are window corners and the edge of a wall: they crowd out the stars and a
    field with eight good stars in it never solves. So only what a star looks like is kept:
    small (a top-hat with a kernel wider than a star leaves only small things standing), compact
    and roughly round, and alone - a window corner sits in a lot of other structure.
    """
    import cv2

    # A star covers several pixels and the noise of a 1 s, 8-bit guide frame does not: smoothing
    # by about a star's width first is what lets the faint ones stand out. Without it, six
    # frames in a row near Mizar and Thuban (tests/data/guide-mizar.png) never solved.
    g = cv2.GaussianBlur(_mono(img), (0, 0), 1.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    h, w = g.shape
    # Noise judged tile by tile. Over the whole frame, the texture of a building raises the
    # estimate until the faint stars in the clear part drop below it; and a tile far busier
    # than the quietest ones is scenery, where nothing that passes for a star is one.
    ty, tx = max(1, h // tile), max(1, w // tile)
    tiles = th[:ty * tile, :tx * tile].reshape(ty, tile, tx, tile).swapaxes(1, 2).reshape(ty, tx, -1)
    tmed = np.median(tiles, axis=2)
    tnoise = 1.4826 * np.median(np.abs(tiles - tmed[..., None]), axis=2) + 1e-3
    sky_noise = float(np.percentile(tnoise, 25))
    base = float(np.median(tmed))
    # Clear sky above 3 sigma is noise and the odd star; a lit facade is edges everywhere.
    lively = np.mean(tiles > base + 3.0 * sky_noise, axis=2)
    quiet = tnoise < busy * sky_noise
    sky = quiet & (lively < lively_max)
    lit = th > base + 3.0 * sky_noise
    mask = (th > base + sigma * sky_noise).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if area < 2 or area > max_area:
            continue
        cy, cx = min(ty - 1, (y0 + bh // 2) // tile), min(tx - 1, (x0 + bw // 2) // tile)
        if not sky[cy, cx] and not (quiet[cy, cx] and _alone_in_tile(lit, cy, cx, tile, stats[i],
                                                                        lively_max)):
            continue
        if max(bw, bh) > 2.5 * min(bw, bh) or area < 0.35 * bw * bh:
            continue          # a line or a smear: an edge, a wire, a trail
        if x0 < edge or y0 < edge or x0 + bw > w - edge or y0 + bh > h - edge:
            continue
        r = int(max(bw, bh)) + 6
        xa, xb, ya, yb = max(0, x0 - r), min(w, x0 + bw + r), max(0, y0 - r), min(h, y0 + bh + r)
        ring = mask[ya:yb, xa:xb].astype(bool) & (lab[ya:yb, xa:xb] != i)
        if ring.mean() > 0.03:
            continue          # other structure all around it: part of the scenery
        sel = lab[y0:y0 + bh, x0:x0 + bw] == i
        wts = th[y0:y0 + bh, x0:x0 + bw] * sel
        # Traced at `sigma`, kept only if it PEAKS well above it: at 4 sigma over a million
        # pixels a few dozen noise blobs always get through, and on a thin night they outnumber
        # the stars and the solve fails.
        if float(wts.max()) < base + peak_sigma * sky_noise:
            continue
        flux = float(wts.sum())
        yy, xx = np.mgrid[y0:y0 + bh, x0:x0 + bw]
        out.append((float((xx * wts).sum() / flux), float((yy * wts).sum() / flux), flux))
    out.sort(key=lambda s: -s[2])
    return out[:max_stars]


def _alone_in_tile(lit, cy, cx, tile, stat, lively_max):
    """A bright star lights enough of its own tile to pass for scenery - on a real frame that
    threw out the two brightest stars and the solve failed. Count the tile again without the
    star's own neighbourhood: scenery is still lit, a star on clear sky is not."""
    x0, y0, bw, bh, _ = stat
    r = 2 * int(max(bw, bh)) + 6
    ya, xa = cy * tile, cx * tile
    t = lit[ya:ya + tile, xa:xa + tile].copy()
    t[max(0, y0 - r - ya):max(0, y0 + bh + r - ya), max(0, x0 - r - xa):max(0, x0 + bw + r - xa)] = False
    return float(t.mean()) < lively_max


class AstrometrySolver:
    """solve-field, run on one frame at a time, blind or with a hint of where to look."""

    def __init__(self, cam_cfg, site, index_dir=INDEX_DIR, timeout_s=None, downsample=None):
        self.cfg, self.site = cam_cfg, site
        self.index_dir = Path(index_dir)
        self.timeout_s = timeout_s or cam_cfg.get("solve_timeout_s", 60)
        self.downsample = downsample or cam_cfg.get("solve_downsample", 2)
        self.last = None

    def check(self):
        """Raise with advice if this machine cannot solve at all."""
        if not shutil.which("solve-field"):
            raise SolveError("solve-field is not installed - `sudo apt install astrometry.net`")
        missing = missing_indexes(self.cfg, self.index_dir)
        if missing:
            raise SolveError(f"index files {missing} missing from {self.index_dir} - run "
                             f"`issctl solve-setup` (needs internet once), or copy them over "
                             f"from a machine that has them")

    def _run(self, work, hint):
        s = pixels_per_deg(self.cfg) / 3600.0     # px per arcsec
        cfg_file = work / "engine.cfg"
        cfg_file.write_text(f"add_path {self.index_dir}\nautoindex\ninparallel\n"
                            f"cpulimit {int(self.timeout_s)}\n")
        cmd = ["solve-field", "--config", str(cfg_file), "--overwrite", "--no-plots",
               "--no-remove-lines", "--uniformize", "0", "--no-verify", "--crpix-center",
               "--scale-units", "arcsecperpix",
               # the lens's real focal length is only nominal - and finding it out is a bonus
               "--scale-low", f"{0.7 / s:.2f}", "--scale-high", f"{1.4 / s:.2f}",
               "--downsample", str(self.downsample), "--cpulimit", str(int(self.timeout_s)),
               "--new-fits", "none", "--index-xyls", "none", "--match", "none",
               "--solved", "none", "--corr", "none", "--tag-all",
               "--dir", str(work), "--temp-dir", str(work),
               "--width", str(self.shape[1]), "--height", str(self.shape[0]),
               "--x-column", "X", "--y-column", "Y", "--sort-column", "FLUX",
               str(work / "frame.xy")]
        if hint is not None:
            cmd += ["--ra", f"{hint[0]:.4f}", "--dec", f"{hint[1]:.4f}", "--radius", "30"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s + 30)

    def solve(self, img, t, hint=None):
        """Solve one frame taken at unix time t. hint=(ra, dec) narrows the search; a solve
        that fails with a hint is retried blind, because the mount may have moved a long way."""
        from astropy.io import fits

        self.check()
        g = _mono(img)
        self.shape = g.shape
        t0 = time.monotonic()
        stars = find_stars(g)
        if len(stars) < 6:
            self._keep_failed(g)
            raise SolveError(f"only {len(stars)} stars found - lengthen the exposure (0.5-2 s), "
                             f"focus, or point at a clearer patch of sky")
        with tempfile.TemporaryDirectory(prefix="issctl-solve-") as tmp:
            work = Path(tmp)
            xy = np.array(stars)
            fits.BinTableHDU.from_columns([
                fits.Column("X", "E", array=xy[:, 0] + 1),      # FITS pixels count from 1
                fits.Column("Y", "E", array=xy[:, 1] + 1),
                fits.Column("FLUX", "E", array=xy[:, 2])]).writeto(work / "frame.xy")
            for h in ([hint, None] if hint is not None else [None]):
                res = self._run(work, h)
                if (work / "frame.wcs").exists():
                    break
            else:
                errors = [line.strip() for line in (res.stdout + res.stderr).splitlines()
                          if "error" in line.lower()][-2:]
                saved = self._keep_failed(g)
                raise SolveError(
                    f"no solution after {time.monotonic() - t0:.0f}s from {len(stars)} stars"
                    + (f" ({' | '.join(errors)})" if errors else "")
                    + ". Too few stars, or too little sky in the frame: lengthen the exposure "
                    f"(0.5-2 s), focus, or point at a clearer patch"
                    + (f". Frame kept as {saved}" if saved else ""))
            header = fits.getheader(work / "frame.wcs")
            catalog = self._stars(work / "frame.rdls")
        sol = WcsSolution(header, self.site, t, g.shape[1], g.shape[0], catalog)
        sol.n_stars = len(stars)
        sol.detected = stars            # (x, y, flux) as detected - for brightness()
        sol.elapsed_s = time.monotonic() - t0
        self.last = sol
        return sol

    @staticmethod
    def _keep_failed(g):
        """A frame that would not solve is worth keeping: it is the only way to find out why."""
        try:
            import cv2

            logs = ROOT / "logs"
            logs.mkdir(exist_ok=True)
            path = logs / f"solve-failed-{time.strftime('%Y%m%d-%H%M%S')}.png"
            cv2.imwrite(str(path), np.clip(g, 0, 255).astype(np.uint8))
            return path.name
        except Exception:
            return None

    @staticmethod
    def _stars(rdls):
        """Catalogue stars in the field (Tycho-2 from the index), plus the named bright stars:
        Tycho-2 leaves out the very brightest, which are exactly the ones you would centre."""
        from astropy.io import fits

        out = [(ra, dec, -2.0, name) for name, (ra, dec) in pr.STARS.items()]
        if rdls.exists():
            data = fits.getdata(rdls, 1)
            mags = data["MAG"] if "MAG" in data.columns.names else np.full(len(data), 9.0)
            out += [(float(r), float(d), float(m), f"mag {m:.1f}")
                    for r, d, m in zip(data["RA"], data["DEC"], mags)]
        return out


class SimSolver:
    """Solves the simulator's frames by asking the simulated world where the camera points."""

    def __init__(self, world, cam, noise_arcsec=2.0, seed=3):
        self.world, self.cam = world, cam
        self.noise = np.radians(noise_arcsec / 3600.0)
        self.rng = np.random.default_rng(seed)
        self.last = None

    def check(self):
        pass

    def solve(self, img, t, hint=None):
        p, e1, e2 = self.world.sky_truth(self.cam.name, t)
        if p is None:
            raise SolveError("simulated sky: nothing to solve")
        tilt = self.rng.normal(0.0, self.noise, 2)
        p = p + tilt[0] * e1 + tilt[1] * e2
        p /= np.linalg.norm(p)
        sol = SimSolution(self.world.true_cal[self.cam.name], p, e1, e2, t, self.cam.width,
                          self.cam.height, self.world.catalog())
        sol.elapsed_s = 0.0
        self.last = sol
        return sol


def make_solver(cam, site, world=None):
    return SimSolver(world, cam) if world is not None else AstrometrySolver(cam.cfg, site)


# ---- taking a frame to solve ----

def _fresh_timed(cam, skip=0, timeout=10.0, not_before=None):
    """The next frame to arrive after `skip` more, with the time it was exposed - and, with
    not_before (clock time), none that arrived before it."""
    deadline = time.monotonic() + timeout
    last = cam.latest_frame()[2]
    seen = 0
    while time.monotonic() < deadline:
        frame, t, seq = cam.latest_frame()
        if seq != last and frame is not None:
            last = seq
            if seen >= skip and (not_before is None or t is None or t >= not_before):
                return frame, t
            seen += 1
        time.sleep(0.01)
    raise SolveError(f"{cam.name}: no frame within {timeout:.0f}s")


def solve_camera(cam, solver, log=print, after_move=False):
    """Grab a fresh frame at the solve exposure, solve it, and put the exposure back.

    after_move: the frame in flight when a move ended was partly exposed DURING the move, and
    its stars are trails - on the real mount one read 48.4"/px against 49.9 for every other and
    threw a whole calibration off. Skip it.

    The ISS wants a few milliseconds; stars want the better part of a second. Only a camera whose
    exposure is in real milliseconds is switched - a raw V4L2 register has no fixed meaning.
    """
    want = cam.cfg.get("solve_exposure_ms") or 0
    saved = (cam.exposure_ms, cam.gain)
    switch = want > 0 and cam.exposure_unit == "ms" and abs(cam.exposure_ms - want) > 1e-6
    try:
        if switch:
            cam.set_exposure(want)
            if cam.cfg.get("solve_gain"):
                cam.set_gain(cam.cfg["solve_gain"])
        # After a change, frames already queued in the camera were still exposed the old way -
        # skipping one is not enough: a guide at 200 ms has several in flight, and a solve of one
        # of those is a black frame of noise. Only one exposed wholly after the switch will do.
        exp_s = max(want, cam.exposure_ms) / 1000.0
        img, t = _fresh_timed(cam, skip=1 if (switch or after_move) else 0,
                              timeout=10.0 + 4 * exp_s,
                              not_before=cam.clock.now() + 1.2 * exp_s if switch else None)
    finally:
        if switch:
            cam.set_exposure(saved[0])
            cam.set_gain(saved[1])
    hint = None
    last = getattr(solver, "last", None)
    if last is not None and last.radec(last.centre()) is not None:
        hint = last.radec(last.centre())
    sol = solver.solve(img, t, hint=hint)
    log(f"{cam.name} solved in {sol.elapsed_s:.1f}s"
        + (f" from {sol.n_stars} stars" if getattr(sol, "n_stars", None) else "")
        + f": {sol.describe()}")
    return sol


def brightness(sol, px, near_px=12.0, match_px=3.0):
    """How bright is the thing at pixel px of a solved frame? Two answers, when they exist:

    * measured: its flux against the frame's own zero point, fitted from every detected star the
      solve matched to Tycho-2 - so it works for what no catalogue has, a satellite included;
    * catalogue: the Tycho-2 (or named) star sitting there.

    Returns a dict: px (the blob used), mag, mag_err, n_ref (stars behind the zero point),
    catalog_label, catalog_mag, catalog_px. Missing answers are None."""
    stars = np.array(getattr(sol, "detected", None) or np.zeros((0, 3)), dtype=float).reshape(-1, 3)
    cat = [(np.asarray(p, dtype=float), label, float(m)) for p, label, m in sol.catalog()]
    tycho = [c for c in cat if c[2] > -1.5]          # the named bright stars carry no magnitude
    out = {"px": None, "mag": None, "mag_err": None, "n_ref": 0,
           "catalog_label": None, "catalog_mag": None, "catalog_px": None}
    # the frame's zero point: catalogue magnitude + 2.5 log10(measured flux), robustly
    zps = []
    for x, y, f in stars:
        if f <= 0 or not tycho:
            continue
        d = [np.hypot(*(c[0] - (x, y))) for c in tycho]
        k = int(np.argmin(d))
        if d[k] <= match_px:
            zps.append(tycho[k][2] + 2.5 * np.log10(f))
    zp = err = None
    if len(zps) >= 5:
        zps = np.array(zps)
        zp = float(np.median(zps))
        err = float(1.4826 * np.median(np.abs(zps - zp)))
        out["n_ref"] = len(zps)
    px = np.asarray(px, dtype=float)
    blob = None
    if len(stars):
        d = np.hypot(stars[:, 0] - px[0], stars[:, 1] - px[1])
        k = int(np.argmin(d))
        if d[k] <= near_px:
            blob = stars[k]
            out["px"] = [float(blob[0]), float(blob[1])]
            if zp is not None and blob[2] > 0:
                out["mag"] = round(zp - 2.5 * float(np.log10(blob[2])), 2)
                out["mag_err"] = round(err, 2)
    at = blob[:2] if blob is not None else px
    if cat:
        d = [np.hypot(*(c[0] - at)) for c in cat]
        k = int(np.argmin(d))
        if d[k] <= (match_px if blob is not None else near_px):
            label, m = cat[k][1], cat[k][2]
            named = [c for c in cat if c[2] <= -1.5 and np.hypot(*(c[0] - cat[k][0])) <= match_px]
            out["catalog_label"] = named[0][1] if named else label
            out["catalog_mag"] = None if m <= -1.5 else round(m, 1)
            out["catalog_px"] = [float(cat[k][0][0]), float(cat[k][0][1])]
    return out
