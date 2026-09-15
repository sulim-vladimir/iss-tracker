"""ISS / star / planet positions in hour angle + declination, and per-pass mount trajectories."""

import time
import urllib.request

import numpy as np
from scipy.interpolate import CubicSpline
from skyfield.api import EarthSatellite, load, wgs84

from . import geometry as geo
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
        raise RuntimeError(f"no TLE available at {path}")
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

def _astropy_altaz(coord_fn, site, t_unix):
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
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


def sun_state(sat, site, t_unix):
    """(ISS sunlit, sun altitude at site) at one instant."""
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.time import Time

    t = Time(t_unix, format="unix")
    s = get_sun(t).cartesian.xyz.to(u.km).value
    u_sun = s / np.linalg.norm(s)
    r = sat.at(unix_to_time(t_unix)).position.km
    along = r @ u_sun
    sunlit = along > 0 or np.linalg.norm(r - along * u_sun) > 6371.0
    sun_alt, _ = _astropy_altaz(lambda tt, loc: get_sun(tt), site, t_unix)
    return bool(sunlit), float(sun_alt)


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

    def __init__(self, t, a1, a2, alt, side, t_start, t_end):
        self.t, self.a1, self.a2, self.alt, self.side = t, a1, a2, alt, side
        self.t_start, self.t_end = t_start, t_end
        self._s = [CubicSpline(t, a1), CubicSpline(t, a2)]
        self._d = [s.derivative() for s in self._s]

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


def plan_pass(sat, site, mount_cfg, rise, set_, dt=0.25, margin=30.0):
    """Choose the pier side that tracks the longest continuous part of the pass."""
    t = np.arange(rise - margin, set_ + margin, dt)
    ha, dec, alt, _ = sat_hadec(sat, site, t)
    vmax = axis_rate_limits(mount_cfg)
    lim = mount_cfg["axis1_hour_limit"]
    best = None
    for side in geo.SIDES:
        a1, a2 = geo.hadec_to_axes(ha, dec, side)
        a1 = np.degrees(np.unwrap(np.radians(a1)))
        a1 -= 360.0 * np.round(np.median(a1[alt >= site.min_altitude]) / 360.0) if np.any(alt >= site.min_altitude) else 0
        v1, v2 = np.gradient(a1, t), np.gradient(a2, t)
        ok = (alt >= site.min_altitude) & (np.abs(a1) <= lim) & (np.abs(v1) <= vmax[0]) & (np.abs(v2) <= vmax[1])
        (i0, i1), n = _longest_run(ok)
        vis = alt >= site.min_altitude
        report = {
            "side": side,
            "track_start": float(t[i0]), "track_end": float(t[i1]),
            "tracked_s": float(t[i1] - t[i0]) if n > 1 else 0.0,
            "visible_s": float(vis.sum() * dt),
            "max_rate": [float(np.abs(v1[vis]).max()), float(np.abs(v2[vis]).max())],
            "rate_limit": vmax.tolist(),
            "axis1_range": [float(a1[vis].min()), float(a1[vis].max())],
            "limited_by": [name for name, bad in (
                ("axis1_limit", np.any(vis & (np.abs(a1) > lim))),
                ("axis1_rate", np.any(vis & (np.abs(v1) > vmax[0]))),
                ("axis2_rate", np.any(vis & (np.abs(v2) > vmax[1]))),
            ) if bad],
        }
        if best is None or report["tracked_s"] > best[0]["tracked_s"]:
            best = (report, a1, a2)
    report, a1, a2 = best
    traj = Trajectory(t, a1, a2, alt, report["side"], report["track_start"], report["track_end"])
    return traj, report
