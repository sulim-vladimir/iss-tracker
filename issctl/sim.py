"""Simulated sky: the true ISS differs from the tracker's model by a TLE timing error,
a cross-track offset and a mount pointing (index/polar) error."""

import numpy as np
from scipy.spatial.transform import Rotation

from . import geometry as geo
from .calib import ideal_calibration, jacobian
from .model import PointingModel, angle_arcsec
from .model import unit as sky_unit
from .predict import sat_hadec


def pointing_at(mount, t):
    """Where the telescope really points: the simulated mount can lag its counters (backlash)."""
    return getattr(mount, "physical_at", mount.position_at)(t)


class SimWorld:
    def __init__(self, cfg, sat, site, mount, time_error_s=1.5, cross_error=(0.05, -0.04),
                 pointing_error=(0.35, -0.25), rotations=None, traj=None, shadow_threshold=0.3,
                 mask=None, clouds=(), polar_error_deg=(0.0, 0.0), azimuth_error_deg=0.0):
        self.sat, self.site, self.mount = sat, site, mount
        # A real misalignment rotates the whole sky mapping - quite different from a constant
        # offset added to the axis readings.
        self.model = misalignment(site.lat, polar_error_deg, azimuth_error_deg)
        self.traj, self.shadow_threshold = traj, shadow_threshold
        self.mask, self.clouds = mask, list(clouds)
        self.time_error = time_error_s
        self.cross_error = np.array(cross_error)
        self.pointing_error = np.array(pointing_error)
        rotations = rotations or {"guide": 12.0, "main": -7.0}
        self.true_cal = {n: ideal_calibration(cfg["cameras"][n], rotations[n]) for n in ("guide", "main")}

    def calibration_estimate(self, scale_error=1.03, rot_error_deg=2.0):
        """What a (slightly imperfect) calibration run would store."""
        out = {}
        r = np.radians(rot_error_deg)
        R = scale_error * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        for n, cal in self.true_cal.items():
            out[n] = dict(cal, J=(R @ np.array(cal["J"])).tolist())
        return out

    def _pointing_at(self, t):
        return pointing_at(self.mount, t)

    def iss_axes(self, t, near):
        ha, dec, alt, az = sat_hadec(self.sat, self.site, t + self.time_error)
        side = "east_looking" if near[1] <= 90.0 else "west_looking"
        p = np.array(geo.hadec_to_axes(ha, dec, side), dtype=float) + self.cross_error
        p[0] = near[0] + geo.wrap180(p[0] - near[0])
        return p, alt, az

    def obscured(self, t, alt, az):
        """Hidden by a mapped obstruction, or by a cloud the tracker knows nothing about."""
        if self.mask is not None and not self.mask.empty and not bool(self.mask.visible(alt, az)):
            return True
        return any(a <= t <= b for a, b in self.clouds)

    def _pixel_tilted(self, cam, t, m):
        ha, dec, alt, az = sat_hadec(self.sat, self.site, t + self.time_error)
        if alt < 0:
            return None
        if self.traj is not None and self.traj.illum_at(t + self.time_error) < self.shadow_threshold:
            return None
        if self.obscured(t, alt, az):
            return None
        p_hat, e1, e2 = self.model.sky_axes(m[0], m[1])
        return _sky_pixel(self.true_cal[cam], p_hat, e1, e2, sky_unit(ha, dec))

    def pixel(self, cam, t):
        m = self._pointing_at(t)
        if m is None:
            return None
        if self.model is not None:
            return self._pixel_tilted(cam, t, m)
        pointing = m + self.pointing_error
        p, alt, az = self.iss_axes(t, pointing)
        if alt < 0:
            return None
        if self.traj is not None and self.traj.illum_at(t + self.time_error) < self.shadow_threshold:
            return None  # in Earth's shadow the ISS is simply not there
        if self.obscured(t, alt, az):
            return None
        cal = self.true_cal[cam]
        px = np.array(cal["boresight"]) + jacobian(cal, pointing[1]) @ (pointing - p)
        w, h = (cal["boresight"][0] + 0.5) * 2, (cal["boresight"][1] + 0.5) * 2
        if not (0 <= px[0] < w and 0 <= px[1] < h):
            return None
        return px


class CalibWorld:
    """One fixed bright target at a constant mount position - a distant light, as recommended for
    calibration. Lets the calibration routine be exercised without a sky."""

    def __init__(self, cfg, mount, offset=(0.02, -0.015), rotations=None, pointing_error=(0.3, -0.2),
                 decoys=((1.5, 0.8, 235.0), (-2.0, -1.2, 205.0))):
        self.mount = mount
        self.pointing_error = np.asarray(pointing_error, dtype=float)
        # offset is measured from where the telescope really points, i.e. as the user would see it
        self.axes_target = mount.position() + self.pointing_error + np.asarray(offset, dtype=float)
        # other lights in the field, deliberately brighter than the one we actually want
        self.decoys = [(self.axes_target + np.array([dx, dy]), amp) for dx, dy, amp in decoys]
        rotations = rotations or {"guide": 12.0, "main": -7.0}
        self.true_cal = {n: ideal_calibration(cfg["cameras"][n], rotations[n]) for n in rotations}

    def _project(self, cam, pointing, axes):
        cal = self.true_cal[cam]
        px = np.array(cal["boresight"]) + jacobian(cal, pointing[1]) @ (pointing - axes)
        w, h = (cal["boresight"][0] + 0.5) * 2, (cal["boresight"][1] + 0.5) * 2
        return px if (0 <= px[0] < w and 0 <= px[1] < h) else None

    def pixel(self, cam, t):
        m = pointing_at(self.mount, t)
        if m is None:
            return None
        return self._project(cam, m + self.pointing_error, self.axes_target)

    def blobs(self, cam, t):
        m = pointing_at(self.mount, t)
        if m is None:
            return []
        pointing = m + self.pointing_error
        out = []
        for axes, amp in [(self.axes_target, 150.0)] + self.decoys:
            px = self._project(cam, pointing, axes)
            if px is not None:
                out.append((px, amp))
        return out


def misalignment(lat, polar_error_deg=(0.0, 0.0), azimuth_error_deg=0.0):
    """Mount whose polar axis is tilted and/or swung in azimuth. Azimuth error is a rotation
    about the local vertical, which for an unaligned tripod is the big one."""
    R = Rotation.identity()
    if azimuth_error_deg:
        R = Rotation.from_rotvec(np.radians(azimuth_error_deg) * sky_unit(0.0, lat)) * R
    if any(polar_error_deg):
        R = Rotation.from_rotvec(np.radians([polar_error_deg[0], polar_error_deg[1], 0.0])) * R
    if R.magnitude() < 1e-12:
        return None
    return PointingModel(rotvec_deg=np.degrees(R.as_rotvec()))


def _sky_pixel(cal, p_hat, e1, e2, v):
    """Where a sky direction lands on a camera whose boresight looks along p_hat."""
    d = np.asarray(v) - p_hat
    off = np.degrees([float(np.dot(d, e1)), float(np.dot(d, e2))])
    px = np.array(cal["boresight"]) - jacobian(cal, 0.0) @ off
    w, h = (cal["boresight"][0] + 0.5) * 2, (cal["boresight"][1] + 0.5) * 2
    return px if (0 <= px[0] < w and 0 <= px[1] < h) else None


def random_clouds(t0, t1, n, seed=0, min_len=3.0, max_len=12.0):
    """n opaque intervals at random times: clouds the tracker cannot predict."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        length = rng.uniform(min_len, max_len)
        start = rng.uniform(t0, max(t0, t1 - length))
        out.append((float(start), float(start + length)))
    return sorted(out)


def evaluate(world, cfg, log_path):
    """True on-sky pointing error from a tracking log (the simulator knows the truth)."""
    import csv

    from .calib import pixels_per_deg

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty log")
        return None
    t = np.array([float(r["t"]) for r in rows])
    axes = np.array([[float(r["a1"]), float(r["a2"])] for r in rows])
    pos = axes + world.pointing_error
    src = np.array([r["source"] for r in rows])
    ha, dec, alt, _ = sat_hadec(world.sat, world.site, t + world.time_error)
    if world.model is not None:
        pointed = world.model.forward(axes[:, 0], axes[:, 1])
        sky = angle_arcsec(pointed, sky_unit(ha, dec)) / 60.0
        return _report(sky, alt, src, cfg, len(t))
    err = np.zeros((len(t), 2))
    for i in range(len(t)):
        side = "east_looking" if pos[i, 1] <= 90.0 else "west_looking"
        p = np.array(geo.hadec_to_axes(ha[i], dec[i], side), dtype=float) + world.cross_error
        d = pos[i] - p
        d[0] = geo.wrap180(d[0])
        err[i] = d * geo.sky_metric(pos[i, 1])
    sky = np.hypot(err[:, 0], err[:, 1]) * 60  # arcmin
    return _report(sky, alt, src, cfg, len(t))


def _report(sky, alt, src, cfg, n):
    from .calib import pixels_per_deg
    main = cfg["cameras"]["main"]
    half_h = main["height"] / 2 / pixels_per_deg(main) * 60
    up = alt > cfg["site"]["min_altitude"]
    print(f"samples {n}; time in control: " +
          ", ".join(f"{s} {np.mean(src[up] == s) * 100:.0f}%"
                    for s in ("predict", "guide", "main", "shadow", "blocked")))
    for label, sel in (("guide", up & (src == "guide")), ("main", up & (src == "main")),
                       ("shadow", up & (src == "shadow")), ("blocked", up & (src == "blocked")),
                       ("predict", up & (src == "predict"))):
        if sel.any():
            s = sky[sel]
            print(f"true error on {label:5s}: median {np.median(s) * 60:5.1f}\"  95% {np.percentile(s, 95) * 60:6.1f}\"  "
                  f"inside main half-height ({half_h:.1f}') {np.mean(s < half_h) * 100:.0f}%")
    return sky
