"""Pointing model for a mount that is not polar aligned.

The mount is treated as a two-axis gimbal with unknown orientation. In the mount's own frame
(z = RA axis) the tube direction for mechanical angles (axis1, axis2) is the ideal German
equatorial pointing, tilted by the cone error c along the Dec axis direction n:

    u = unit(ha_m = axis1 - 90, dec_m = axis2 + d2)
    n = (-sin ha_m, cos ha_m, 0)
    p = cos(c) u + sin(c) n

and the sky direction in the local hour-angle/declination frame is v = R p.

Parameters: R (rotation vector, 3: polar axis altitude/azimuth error + RA index), d2 (Dec index),
cone. With R = I and d2 = cone = 0 this is exactly the ideal mount in geometry.py.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import geometry as geo

Z = np.array([0.0, 0.0, 1.0])


def unit(ha, dec):
    h, d = np.radians(ha), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(h), np.cos(d) * np.sin(h), np.sin(d)], axis=-1)


def to_hadec(v):
    v = np.asarray(v, dtype=float)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])),
            np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0))))


def angle_arcsec(v1, v2):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1 * v2, axis=-1))) * 3600


class PointingModel:
    def __init__(self, rotvec_deg=(0.0, 0.0, 0.0), d2=0.0, cone=0.0, rms_arcsec=None, n_points=0):
        self.rotvec_deg = np.asarray(rotvec_deg, dtype=float)
        self.d2 = float(d2)
        self.cone = float(cone)
        self.rms_arcsec = rms_arcsec
        self.n_points = n_points
        self.R = Rotation.from_rotvec(np.radians(self.rotvec_deg)).as_matrix()

    # ---- persistence ----
    def to_dict(self):
        return {"rotvec_deg": self.rotvec_deg.tolist(), "d2": self.d2, "cone": self.cone,
                "rms_arcsec": self.rms_arcsec, "n_points": self.n_points}

    @classmethod
    def from_dict(cls, d):
        return cls(**d) if d else cls()

    @classmethod
    def from_state(cls, state):
        return cls.from_dict(state.get("alignment", {}).get("model"))

    def describe(self):
        pole = self.R @ Z
        tilt = np.degrees(np.arccos(np.clip(pole @ Z, -1, 1)))
        rms = f"{self.rms_arcsec:.0f}\"" if self.rms_arcsec is not None else "-"
        return (f"{self.n_points} pts rms {rms} | RA axis {tilt:.2f} deg from pole, "
                f"dec index {self.d2:+.2f}, cone {self.cone:+.2f}")

    # ---- kinematics ----
    def mount_dec(self, a2):
        """Declination in the mount frame (for image-scale factors)."""
        return geo.axis2_to_dec(np.asarray(a2) + self.d2)

    def mount_vectors(self, a1, a2):
        h = np.radians(np.asarray(a1, dtype=float) - 90.0)
        u = unit(np.asarray(a1, dtype=float) - 90.0, np.asarray(a2, dtype=float) + self.d2)
        n = np.stack([-np.sin(h), np.cos(h), np.zeros_like(h)], axis=-1)
        c = np.radians(self.cone)
        return np.cos(c) * u + np.sin(c) * n, n

    def forward(self, a1, a2):
        p, _ = self.mount_vectors(a1, a2)
        return p @ self.R.T

    def axes_to_hadec(self, a1, a2):
        return to_hadec(self.forward(a1, a2))

    def sky_axes(self, a1, a2):
        """Sky-frame unit vectors of the pointing and of its motion per +axis1 / +axis2."""
        p, n = self.mount_vectors(a1, a2)
        e1 = np.cross(Z, p)
        e2 = np.cross(p, n)
        norm = lambda e: e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)
        return p @ self.R.T, norm(e1) @ self.R.T, norm(e2) @ self.R.T

    def hadec_to_axes(self, ha, dec, side, iters=5):
        pm = unit(ha, dec) @ self.R  # R^T v
        hm, dm = to_hadec(pm)
        if side == "east_looking":
            a1, a2 = hm + 90.0, dm - self.d2
        elif side == "west_looking":
            a1, a2 = hm - 90.0, 180.0 - dm - self.d2
        else:
            raise ValueError(side)
        a1 = geo.wrap180(a1)
        a2 = np.asarray(a2, dtype=float)
        for _ in range(iters):  # Gauss-Newton on the exact kinematics (cone)
            p, n = self.mount_vectors(a1, a2)
            r = pm - p
            j1, j2 = np.cross(Z, p), np.cross(p, n)
            a11 = np.sum(j1 * j1, -1) + 1e-12
            a12 = np.sum(j1 * j2, -1)
            a22 = np.sum(j2 * j2, -1) + 1e-12
            b1, b2 = np.sum(j1 * r, -1), np.sum(j2 * r, -1)
            det = a11 * a22 - a12 * a12
            a1 = a1 + np.degrees((a22 * b1 - a12 * b2) / det)
            a2 = a2 + np.degrees((a11 * b2 - a12 * b1) / det)
        return a1, a2

    def best_side(self, ha, dec, limit=None):
        """Side that keeps the counterweight lowest (smallest |axis1|)."""
        best = None
        for side in geo.SIDES:
            a1, a2 = self.hadec_to_axes(ha, dec, side)
            if best is None or abs(float(a1)) < abs(best[1]):
                best = (side, float(a1), float(a2))
        if limit is not None and abs(best[1]) > limit:
            return None
        return best


# ---- fitting ----

PARAMS = ("rx", "ry", "rz", "d2", "cone")


def _kabsch(p, v):
    U, _, Vt = np.linalg.svd(v.T @ p)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def fit_model(axes, vectors, prior=None):
    """axes (N,2) mechanical degrees, vectors (N,3) observed sky unit vectors (HA/Dec frame).

    Free parameters grow with the number of points: 1 -> RA index + Dec index,
    2 -> full orientation + Dec index, >=3 -> + cone.
    """
    axes = np.atleast_2d(np.asarray(axes, dtype=float))
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    n = len(axes)
    if n == 0:
        return PointingModel()
    free = {1: [2, 3], 2: [0, 1, 2, 3]}.get(n, [0, 1, 2, 3, 4])
    prior = prior or PointingModel()

    def params(model):
        return np.array([*model.rotvec_deg, model.d2, model.cone])

    def build(x, base):
        full = base.copy()
        full[free] = x
        return PointingModel(full[:3], full[3], full[4])

    def resid(x, base):
        m = build(x, base)
        r = (m.forward(axes[:, 0], axes[:, 1]) - vectors).ravel()
        reg = 1e-4 * np.radians(np.array([m.d2, m.cone]))  # weak pull toward 0 for near-degenerate sets
        return np.concatenate([r, reg])

    starts = [params(prior)]
    for rz in (0.0, 90.0, 180.0, -90.0):
        starts.append(np.array([0.0, 0.0, rz, 0.0, 0.0]))
    if n >= 2:
        for d2 in (0.0, prior.d2):
            ideal = PointingModel(d2=d2)
            R = _kabsch(ideal.forward(axes[:, 0], axes[:, 1]), vectors)
            starts.append(np.array([*np.degrees(Rotation.from_matrix(R).as_rotvec()), d2, 0.0]))

    best = None
    for s in starts:
        sol = least_squares(resid, s[free], args=(s,), method="lm" if len(s[free]) * 1 <= 3 * n else "trf",
                            x_scale=1.0, diff_step=1e-6)
        if best is None or sol.cost < best[0].cost:
            best = (sol, s)
    sol, base = best
    model = build(sol.x, base)
    err = angle_arcsec(model.forward(axes[:, 0], axes[:, 1]), vectors)
    model.rms_arcsec = float(np.sqrt(np.mean(err ** 2)))
    model.n_points = n
    model.residuals_arcsec = err
    return model
