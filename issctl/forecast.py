"""Coming up: which bright satellites will cross the guide field, or the reachable sky, soon.

The identify catalogues run forward. A satellite is listed when it is sunlit while the sky here
is dark, and its brightness is estimated from McCants' standard magnitudes (qs.mag: the magnitude
at 1000 km and half phase) with a diffusely reflecting sphere for the phase - good to a
magnitude or so, and a tumbling rocket body does what it likes.

What the public catalogues lack are most old rocket bodies, often the brightest things up there:
the full catalogue needs a Space-Track account. The answer is only as complete as that.
"""

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import geometry as geo
from . import identify as idf
from . import predict as pr

MAGS_URL = "https://www.mmccants.org/programs/qsmag.zip"
MAGS_MAX_AGE_H = 24.0 * 7
DARK_SUN_ALT = -6.0        # civil twilight: brighter than that, the guide sees sky, not satellites
STEP_S = 10.0


def refresh_mags(directory=idf.CATALOG_DIR, log=print, offline=False):
    path = Path(directory) / "qs.mag"
    if offline or (path.exists() and time.time() - path.stat().st_mtime < MAGS_MAX_AGE_H * 3600):
        return path if path.exists() else None
    try:
        req = urllib.request.Request(MAGS_URL, headers={"User-Agent": "issctl"})
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())) as z:
            data = z.read(next(n for n in z.namelist() if n.lower().endswith(".mag")))
        Path(directory).mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        log("catalogue qs.mag: updated")
    except Exception as e:
        log(f"catalogue qs.mag: could not update ({e})")
    return path if path.exists() else None


def load_mags(path):
    """Catalogue id (no leading zeros) -> standard magnitude."""
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            out[line[:5].strip().lstrip("0")] = float(line[33:37])
        except ValueError:
            continue
    return out


def candidates(tles, mags, max_std_mag=8.0):
    """The catalogue objects with a known standard magnitude, bright enough to bother with."""
    keep = []
    for name, sid, l1, l2 in tles:
        m = mags.get(sid.lstrip("0"))
        if m is not None and m <= max_std_mag:
            keep.append((name, sid, l1, l2, m))
    built = idf.build([k[:4] for k in keep])
    std = {k[1]: k[4] for k in keep}
    return [(name, sid, sat, std[sid]) for name, sid, sat in built]


def _lit(r, u_sun, d_sun):
    """Sunlit fraction for geocentric positions r (N,3), the ISS's own shadow model."""
    along = np.sum(r * u_sun, axis=1)
    perp = np.linalg.norm(r - along[:, None] * u_sun, axis=1)
    ell = np.abs(along)
    r_umbra = pr.R_ATMOS * (1.0 - ell / (pr.R_ATMOS * d_sun / (pr.R_SUN - pr.R_ATMOS)))
    r_pen = pr.R_ATMOS * (1.0 + ell / (pr.R_ATMOS * d_sun / (pr.R_SUN + pr.R_ATMOS)))
    frac = np.clip((perp - r_umbra) / np.maximum(r_pen - r_umbra, 1e-6), 0.0, 1.0)
    return np.where(along < 0, frac, 1.0)


def apparent_mag(std, range_km, phase_rad):
    """Standard magnitude (1000 km, 90 deg phase) seen at this range and phase angle."""
    f = (np.sin(phase_rad) + (np.pi - phase_rad) * np.cos(phase_rad)) / np.pi
    return std + 5 * np.log10(range_km / 1000.0) - 2.5 * np.log10(np.maximum(f * np.pi, 1e-4))


def field_track(site, alt, az, t0, times, follow_stars):
    """Unit alt/az vectors of the field centre at `times`: fixed on the stars when the mount
    tracks sidereally, otherwise fixed where it points."""
    if not follow_stars:
        return np.tile(idf._altaz_unit(np.array(alt), np.array(az)), (len(times), 1))
    ha, dec = geo.altaz_to_hadec(alt, az, site.lat)
    ha = ha + (times - t0) * (360.0 / 86164.0905)
    a, z = geo.hadec_to_altaz(ha, np.full_like(ha, dec), site.lat)
    return idf._altaz_unit(np.asarray(a), np.asarray(z))


def forecast(sats, site, t0, minutes=60.0, field=None, radius_deg=6.5, mask=None,
             min_alt=None, max_mag=7.0, step_s=STEP_S):
    """Sunlit, dark-sky appearances in the next `minutes`.

    field: unit alt/az vectors of the field centre per time step (field_track), for crossings of
    that field; None for passes anywhere above min_alt and inside the sky mask.
    Returns dicts sorted by start time: name, id, start, end, peak (time of best magnitude),
    mag, alt, az (at peak), sep (closest to the field centre, field mode), range_km."""
    from skyfield.api import load

    t = t0 + np.arange(0.0, minutes * 60.0, step_s)
    ts = load.timescale()
    when = ts.from_datetimes([idf._utc(x) for x in t])
    u_sun, d_sun = pr.sun_vector(t)
    here = site.topos.at(when).position.km.T
    sun_alt = _sun_alt(site, t)
    dark = sun_alt < DARK_SUN_ALT
    if not dark.any():
        return []
    out = []
    for name, sid, sat, std in sats:
        try:
            geo_r = sat.at(when).position.km.T
            a, z, d = (sat - site.topos).at(when).altaz()
        except Exception:
            continue
        alt, az, rng = a.degrees, z.degrees, d.km
        ok = dark & (alt > (min_alt if min_alt is not None else 0.0))
        if field is not None:
            sep = np.degrees(np.arccos(np.clip(np.sum(idf._altaz_unit(alt, az) * field, -1),
                                               -1.0, 1.0)))
            ok &= sep < radius_deg
        else:
            sep = np.full_like(alt, np.nan)
        if mask is not None and ok.any():
            ok &= np.asarray(mask.visible(alt, az), dtype=bool)
        if not ok.any():
            continue
        k = np.where(ok)[0]
        lit = _lit(geo_r[k], u_sun[k], d_sun[k]) > 0.5
        k = k[lit]
        if not len(k):
            continue
        to_obs = here[k] - geo_r[k]
        to_sun = u_sun[k] * d_sun[k][:, None] - geo_r[k]
        cosb = np.sum(to_obs * to_sun, 1) / (np.linalg.norm(to_obs, axis=1) *
                                             np.linalg.norm(to_sun, axis=1))
        mag = apparent_mag(std, rng[k], np.arccos(np.clip(cosb, -1.0, 1.0)))
        j = int(np.argmin(mag))
        if mag[j] > max_mag:
            continue
        # one entry per appearance: split where consecutive steps are not consecutive
        runs = np.split(np.arange(len(k)), np.where(np.diff(k) > 1)[0] + 1)
        for run in runs:
            jj = run[int(np.argmin(mag[run]))]
            if mag[jj] > max_mag:
                continue
            out.append({"name": name, "id": sid, "start": float(t[k[run[0]]]),
                        "end": float(t[k[run[-1]]] + step_s), "peak": float(t[k[jj]]),
                        "mag": round(float(mag[jj]), 1), "alt": round(float(alt[k[jj]]), 1),
                        "az": round(float(az[k[jj]]), 1),
                        "sep": None if field is None else round(float(np.min(sep[k[run]])), 1),
                        "range_km": round(float(rng[k[jj]])),
                        # where it will be, for the sky chart: [az, alt, t] every step
                        "path": [[round(float(az[i]), 1), round(float(alt[i]), 1), round(float(t[i]))]
                                 for i in k[run]]})
    out.sort(key=lambda r: r["start"])
    return out


def _sun_alt(site, t):
    """Sun altitude here, every 5 minutes and interpolated between: it moves a degree in four."""
    from astropy.coordinates import get_sun

    coarse = np.arange(t[0], t[-1] + 300.0, 300.0)
    alt, _ = pr._astropy_altaz(lambda tt, loc: get_sun(tt), site, coarse)
    return np.interp(t, coarse, np.atleast_1d(alt))


class Forecaster:
    """Holds the built catalogue between requests: building it is the slow part."""

    def __init__(self, site, catalog_dir=idf.CATALOG_DIR, log=print):
        self.site, self.catalog_dir, self.log = site, catalog_dir, log
        self.sats, self.built_at = None, 0.0

    def _ensure(self, offline=False):
        if self.sats is not None and time.time() - self.built_at < 12 * 3600:
            return
        tles = idf.load_tles(idf.refresh_catalogs(self.catalog_dir, log=self.log, offline=offline))
        path = refresh_mags(self.catalog_dir, log=self.log, offline=offline)
        if not tles or path is None:
            raise RuntimeError("no catalogue or magnitudes yet - connect to the internet once")
        self.sats = candidates(tles, load_mags(path))
        self.built_at = time.time()

    def run(self, t0, **kw):
        self._ensure()
        return forecast(self.sats, self.site, t0, **kw)
