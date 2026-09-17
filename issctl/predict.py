"""ISS / star / planet positions in hour angle + declination, and per-pass mount trajectories."""

import time
import urllib.request

import numpy as np
from scipy.interpolate import CubicSpline
from skyfield.api import EarthSatellite, load, wgs84

from . import geometry as geo
from . import mask as mask_mod
from .config import ROOT

# Historical ISS TLE (2008) used only by the simulator.
SIM_TLE = (
    "ISS (ZARYA)",
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)

# Bright stars for sync/calibration: name -> (RA deg, Dec deg), J2000.
STARS = {
    "sirius": (101.2871, -16.7161), "arcturus": (213.9154, 19.1825),
    "vega": (279.2347, 38.7837), "capella": (79.1723, 45.9980),
    "rigel": (78.6345, -8.2016), "procyon": (114.8255, 5.2250),
    "betelgeuse": (88.7929, 7.4071), "altair": (297.6958, 8.8683),
    "aldebaran": (68.9802, 16.5093), "antares": (247.3519, -26.4320),
    "spica": (201.2983, -11.1613), "pollux": (116.3290, 28.0262),
    "fomalhaut": (344.4128, -29.6222), "deneb": (310.3580, 45.2803),
    "regulus": (152.0930, 11.9672), "castor": (113.6495, 31.8883),
    "polaris": (37.9546, 89.2641), "dubhe": (165.9320, 61.7510),
    "alkaid": (206.8852, 49.3133), "mirfak": (51.0807, 49.8612),
    "alpheratz": (2.0969, 29.0904),
}
BODIES = ("moon", "mercury", "venus", "mars", "jupiter", "saturn")

_ts = None


def timescale():
    global _ts
    if _ts is None:
        _ts = load.timescale(builtin=True)
    return _ts


def unix_to_time(t):
    return timescale().utc(1970, 1, 1, 0, 0, np.asarray(t, dtype=float))


def time_to_unix(t):
    return t.utc_datetime().timestamp()


class Site:
    def __init__(self, cfg):
        s = cfg["site"]
        self.lat = s["latitude"]
        self.lon = s["longitude"]
        self.elevation = s["elevation_m"]
        self.temperature_c = s["temperature_c"]
        self.pressure_mbar = s["pressure_mbar"]
        self.min_altitude = s["min_altitude"]
        self.topos = wgs84.latlon(self.lat, self.lon, elevation_m=self.elevation)


def get_tle(cfg, offline=False):
    path = ROOT / cfg["tle"]["cache"]
    fresh = path.exists() and time.time() - path.stat().st_mtime < cfg["tle"]["max_age_hours"] * 3600
    if not fresh and not offline:
        try:
            with urllib.request.urlopen(cfg["tle"]["url"], timeout=15) as r:
                lines = [l.strip() for l in r.read().decode().splitlines() if l.strip()]
            if len(lines) >= 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(lines[:3]) + "\n")
            else:
                print("TLE download returned unexpected content, using cache")
        except OSError as e:
            print(f"TLE download failed ({e}), using cache")
    if not path.exists():
        raise RuntimeError(f"no TLE cached at {path} - run once with network access "
                           f"(without --offline) to fetch one from {cfg['tle']['url']}")
    name, l1, l2 = path.read_text().splitlines()[:3]
    return name.strip(), l1, l2


def make_satellite(tle):
    name, l1, l2 = tle
    return EarthSatellite(l1, l2, name, timescale())


def tle_age_days(sat, t_unix):
    return (t_unix - time_to_unix(sat.epoch)) / 86400.0


def sat_hadec(sat, site, t_unix):
    """Refracted apparent position -> (ha, dec, alt, az) in degrees; works on arrays."""
    topo = (sat - site.topos).at(unix_to_time(t_unix))
    alt, az, _ = topo.altaz(temperature_C=site.temperature_c, pressure_mbar=site.pressure_mbar)
    ha, dec = geo.altaz_to_hadec(alt.degrees, az.degrees, site.lat)
    return ha, dec, alt.degrees, az.degrees


# ---- stars / planets via astropy (bundled ephemeris, no downloads) ----

def _init_astropy():
    """No network in the field, and dates past the bundled Earth-orientation table are fine:
    the degraded accuracy is at the arcsecond level, far below our pointing errors."""
    import warnings

    from astropy.utils import iers
    from astropy.utils.exceptions import AstropyWarning

    iers.conf.auto_download = False
    warnings.filterwarnings("ignore", message=".*polar motions.*", category=AstropyWarning)
    warnings.filterwarnings("ignore", message=".*IERS table.*", category=AstropyWarning)
    try:
        iers.conf.iers_degraded_accuracy = "ignore"  # option(error, warn, ignore)
    except (AttributeError, TypeError):  # older astropy, or a different option set
        pass


def _astropy_altaz(coord_fn, site, t_unix):
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation
    from astropy.time import Time

    _init_astropy()
    loc = EarthLocation(lat=site.lat * u.deg, lon=site.lon * u.deg, height=site.elevation * u.m)
    t = Time(t_unix, format="unix")
    frame = AltAz(obstime=t, location=loc, pressure=site.pressure_mbar * u.hPa,
                  temperature=site.temperature_c * u.deg_C, relative_humidity=0.5,
                  obswl=0.55 * u.micron)
    aa = coord_fn(t, loc).transform_to(frame)
    return aa.alt.deg, aa.az.deg


def target_hadec(name, site, t_unix):
    """name: star from STARS, a body from BODIES, or 'RA_hours Dec_deg'. Returns ha, dec, alt, az."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord, get_body

    key = name.strip().lower()
    if key in STARS:
        ra, dec = STARS[key]
        fn = lambda t, loc: SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    elif key in BODIES:
        fn = lambda t, loc: get_body(key, t, loc)
    else:
        try:
            ra_h, dec_d = (float(x) for x in key.replace(",", " ").split())
        except ValueError:
            raise ValueError(f"unknown target '{name}'")
        fn = lambda t, loc: SkyCoord(ra=ra_h * 15 * u.deg, dec=dec_d * u.deg, frame="icrs")
    alt, az = _astropy_altaz(fn, site, t_unix)
    ha, dec = geo.altaz_to_hadec(alt, az, site.lat)
    return float(ha), float(dec), float(alt), float(az)


R_EARTH = 6378.137
R_ATMOS = R_EARTH + 80.0   # opaque/absorbing shell: the ISS fades before geometric umbra
R_SUN = 696000.0


def sun_vector(t_unix):
    """Unit vector to the Sun and its distance (km), geocentric, vectorised."""
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.time import Time

    _init_astropy()

    s = get_sun(Time(np.atleast_1d(t_unix), format="unix")).cartesian.xyz.to(u.km).value.T
    d = np.linalg.norm(s, axis=-1)
    return s / d[:, None], d


def illumination(sat, t_unix):
    """Sunlit fraction: 1 in full sun, 0 in umbra, in between inside the penumbra."""
    t = np.atleast_1d(np.asarray(t_unix, dtype=float))
    r = np.atleast_2d(sat.at(unix_to_time(t)).position.km.T)
    u_sun, d_sun = sun_vector(t)
    along = np.sum(r * u_sun, axis=1)
    perp = np.linalg.norm(r - along[:, None] * u_sun, axis=1)
    behind = along < 0
    ell = np.abs(along)
    r_umbra = R_ATMOS * (1.0 - ell / (R_ATMOS * d_sun / (R_SUN - R_ATMOS)))
    r_penumbra = R_ATMOS * (1.0 + ell / (R_ATMOS * d_sun / (R_SUN + R_ATMOS)))
    frac = np.clip((perp - r_umbra) / np.maximum(r_penumbra - r_umbra, 1e-6), 0.0, 1.0)
    out = np.where(behind, frac, 1.0)
    return out if np.ndim(t_unix) else float(out[0])


def sun_state(sat, site, t_unix):
    """(sunlit fraction, sun altitude at site) at one instant."""
    from astropy.coordinates import get_sun

    sun_alt, _ = _astropy_altaz(lambda tt, loc: get_sun(tt), site, t_unix)
    return float(illumination(sat, t_unix)), float(sun_alt)


def pass_horizon(sat, site, p, margin=900.0):
    """Rise and set at altitude 0 for a pass, rather than at the tracking minimum altitude."""
    times, events = sat.find_events(site.topos, unix_to_time(p["culm"] - margin),
                                    unix_to_time(p["culm"] + margin), altitude_degrees=0.0)
    ev = [(time_to_unix(t), int(e)) for t, e in zip(times, events)]
    rise = max((t for t, e in ev if e == 0 and t <= p["culm"]), default=p["rise"])
    set_ = min((t for t, e in ev if e == 2 and t >= p["culm"]), default=p["set"])
    return rise, set_


def find_passes(sat, site, t0_unix, hours):
    times, events = sat.find_events(site.topos, unix_to_time(t0_unix),
                                    unix_to_time(t0_unix + hours * 3600),
                                    altitude_degrees=site.min_altitude)
    passes, cur = [], None
    for ti, ev in zip(times, events):
        u = time_to_unix(ti)
        if ev == 0:
            cur = {"rise": u}
        elif ev == 1 and cur is not None:
            cur["culm"] = u
        elif ev == 2 and cur is not None and "culm" in cur:
            cur["set"] = u
            cur["max_alt"] = float(sat_hadec(sat, site, cur["culm"])[2])
            passes.append(cur)
            cur = None
    return passes


# ---- mount trajectory ----

def steps_per_deg(axis_cfg):
    return (axis_cfg["motor_steps"] * axis_cfg["microsteps"] * axis_cfg["gear_ratio"]
            * axis_cfg["worm_teeth"] / 360.0)


def axis_rate_limits(mount_cfg):
    return np.array([
        min(mount_cfg["max_rate_deg_s"], mount_cfg["firmware_max_step_rate"] / steps_per_deg(mount_cfg[k]))
        for k in ("axis1", "axis2")
    ])


class Trajectory:
    """Mechanical axis angles vs unix time for one pass, one pier side."""

    def __init__(self, t, a1, a2, alt, side, t_start, t_end, lit=None, open_sky=None):
        self.t, self.a1, self.a2, self.alt, self.side = t, a1, a2, alt, side
        self.t_start, self.t_end = t_start, t_end
        self.lit = np.ones_like(t) if lit is None else lit
        self.open_sky = np.ones_like(t) if open_sky is None else np.asarray(open_sky, dtype=float)
        self._s = [CubicSpline(t, a1), CubicSpline(t, a2)]
        self._d = [s.derivative() for s in self._s]

    def illum_at(self, tq):
        return float(np.interp(tq, self.t, self.lit))

    def open_at(self, tq):
        """1 where the sky is clear of known obstructions, 0 behind a building/frame."""
        return float(np.interp(tq, self.t, self.open_sky))

    def at(self, tq):
        """Position and velocity (deg, deg/s). Outside the sampled span: endpoint, zero velocity."""
        if tq <= self.t[0] or tq >= self.t[-1]:
            tq = min(max(tq, self.t[0]), self.t[-1])
            return np.array([self._s[0](tq), self._s[1](tq)]), np.zeros(2)
        return (np.array([self._s[0](tq), self._s[1](tq)]),
                np.array([self._d[0](tq), self._d[1](tq)]))


def _longest_run(mask):
    best, start, best_len = (0, 0), None, 0
    for i, ok in enumerate(np.append(mask, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best, best_len = (start, i - 1), i - start
            start = None
    return best, best_len


def shadow_events(t, lit, threshold=0.5):
    """Times where the ISS crosses into/out of shadow: [(unix, 'enters'|'leaves'), ...]."""
    above = lit > threshold
    idx = np.nonzero(np.diff(above.astype(int)))[0]
    return [(float(t[i + 1]), "leaves" if above[i + 1] else "enters") for i in idx]


def plan_pass(sat, site, mount_cfg, rise, set_, dt=0.25, margin=30.0, model=None, mask=None):
    """Choose the pier side that keeps the ISS trackable, sunlit and unobstructed for longest."""
    t = np.arange(rise - margin, set_ + margin, dt)
    ha, dec, alt, az = sat_hadec(sat, site, t)
    lit = illumination(sat, t)
    open_sky = np.ones(len(t), dtype=bool) if mask is None or mask.empty else mask.visible(alt, az)
    vmax = axis_rate_limits(mount_cfg)
    lim = mount_cfg["axis1_hour_limit"]
    best = None
    for side in geo.SIDES:
        a1, a2 = geo.hadec_to_axes(ha, dec, side)
        a1 = np.degrees(np.unwrap(np.radians(a1)))
        a1 -= 360.0 * np.round(np.median(a1[alt >= site.min_altitude]) / 360.0) if np.any(alt >= site.min_altitude) else 0
        v1, v2 = np.gradient(a1, t), np.gradient(a2, t)
        a2_lo, a2_hi = mount_cfg.get("axis2_limits", [-10.0, 190.0])
        ok = ((alt >= site.min_altitude) & (np.abs(a1) <= lim) & (a2 >= a2_lo) & (a2 <= a2_hi)
              & (np.abs(v1) <= vmax[0]) & (np.abs(v2) <= vmax[1]))
        (i0, i1), n = _longest_run(ok)
        vis = alt >= site.min_altitude
        tracked = np.zeros_like(ok)
        if n > 1:
            tracked[i0:i1 + 1] = True
        report = {
            "side": side,
            "track_start": float(t[i0]), "track_end": float(t[i1]),
            "tracked_s": float(t[i1] - t[i0]) if n > 1 else 0.0,
            "useful_s": float(np.sum(tracked & (lit > 0.5) & open_sky) * dt),
            "sunlit_s": float(np.sum(vis & (lit > 0.5)) * dt),
            "blocked_s": float(np.sum(vis & (lit > 0.5) & ~open_sky) * dt),
            "windows": mask_mod.segments(t, tracked & (lit > 0.5) & open_sky),
            "shadow": shadow_events(t[vis], lit[vis]),
            "visible_s": float(vis.sum() * dt),
            "max_rate": [float(np.abs(v1[vis]).max()), float(np.abs(v2[vis]).max())],
            "rate_limit": vmax.tolist(),
            "axis1_range": [float(a1[vis].min()), float(a1[vis].max())],
            "limited_by": [name for name, bad in (
                ("axis1_limit", np.any(vis & (np.abs(a1) > lim))),
                ("axis2_limit", np.any(vis & ((a2 < a2_lo) | (a2 > a2_hi)))),
                ("axis1_rate", np.any(vis & (np.abs(v1) > vmax[0]))),
                ("axis2_rate", np.any(vis & (np.abs(v2) > vmax[1]))),
            ) if bad],
        }
        key = (report["useful_s"], report["tracked_s"])
        if best is None or key > (best[0]["useful_s"], best[0]["tracked_s"]):
            best = (report, a1, a2)
    report, a1, a2 = best
    traj = Trajectory(t, a1, a2, alt, report["side"], report["track_start"], report["track_end"],
                      lit, open_sky.astype(float))
    return traj, report
