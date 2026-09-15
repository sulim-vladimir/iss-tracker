"""Simulated sky: the true ISS differs from the tracker's model by a TLE timing error,
a cross-track offset and a mount pointing (index/polar) error."""

import numpy as np

from . import geometry as geo
from .calib import ideal_calibration, jacobian
from .predict import sat_hadec


class SimWorld:
    def __init__(self, cfg, sat, site, mount, time_error_s=1.5, cross_error=(0.05, -0.04),
                 pointing_error=(0.35, -0.25), rotations=None):
        self.sat, self.site, self.mount = sat, site, mount
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

    def iss_axes(self, t, near):
        ha, dec, alt, _ = sat_hadec(self.sat, self.site, t + self.time_error)
        side = "east_looking" if near[1] <= 90.0 else "west_looking"
        p = np.array(geo.hadec_to_axes(ha, dec, side), dtype=float) + self.cross_error
        p[0] = near[0] + geo.wrap180(p[0] - near[0])
        return p, alt

    def pixel(self, cam, t):
        m = self.mount.position_at(t)
        if m is None:
            return None
        pointing = m + self.pointing_error
        p, alt = self.iss_axes(t, pointing)
        if alt < 0:
            return None
        cal = self.true_cal[cam]
        px = np.array(cal["boresight"]) + jacobian(cal, pointing[1]) @ (pointing - p)
        w, h = (cal["boresight"][0] + 0.5) * 2, (cal["boresight"][1] + 0.5) * 2
        if not (0 <= px[0] < w and 0 <= px[1] < h):
            return None
        return px


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
    pos = np.array([[float(r["a1"]), float(r["a2"])] for r in rows]) + world.pointing_error
    src = np.array([r["source"] for r in rows])
    ha, dec, alt, _ = sat_hadec(world.sat, world.site, t + world.time_error)
    err = np.zeros((len(t), 2))
    for i in range(len(t)):
        side = "east_looking" if pos[i, 1] <= 90.0 else "west_looking"
        p = np.array(geo.hadec_to_axes(ha[i], dec[i], side), dtype=float) + world.cross_error
        d = pos[i] - p
        d[0] = geo.wrap180(d[0])
        err[i] = d * geo.sky_metric(pos[i, 1])
    sky = np.hypot(err[:, 0], err[:, 1]) * 60  # arcmin
    main = cfg["cameras"]["main"]
    half_h = main["height"] / 2 / pixels_per_deg(main) * 60
    up = alt > cfg["site"]["min_altitude"]
    print(f"samples {len(t)}; time in control: " +
          ", ".join(f"{s} {np.mean(src[up] == s) * 100:.0f}%" for s in ("predict", "guide", "main")))
    for label, sel in (("guide", up & (src == "guide")), ("main", up & (src == "main"))):
        if sel.any():
            s = sky[sel]
            print(f"true error on {label:5s}: median {np.median(s) * 60:5.1f}\"  95% {np.percentile(s, 95) * 60:6.1f}\"  "
                  f"inside main half-height ({half_h:.1f}') {np.mean(s < half_h) * 100:.0f}%")
    return sky
