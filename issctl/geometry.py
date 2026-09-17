"""Coordinate transforms and German-equatorial mount kinematics.

Mechanical axis angles (degrees):
  axis1: RA axis, 0 = counterweight straight down, + = sidereal tracking direction.
  axis2: Dec axis, 90 = tube parallel to polar axis (home).

Two pier sides reach any HA/Dec:
  "east_looking": HA = axis1 - 90,  Dec = axis2          (tube west of pier, HA < 0 with cw down)
  "west_looking": HA = axis1 + 90,  Dec = 180 - axis2    (tube east of pier, HA > 0 with cw down)
"""

import numpy as np

SIDES = ("east_looking", "west_looking")


def altaz_to_hadec(alt, az, lat):
    """Alt/az (az from north through east) -> hour angle (+west) and declination. Degrees."""
    a, A, p = np.radians(alt), np.radians(az), np.radians(lat)
    x = np.cos(p) * np.sin(a) - np.sin(p) * np.cos(a) * np.cos(A)
    y = -np.cos(a) * np.sin(A)
    z = np.sin(p) * np.sin(a) + np.cos(p) * np.cos(a) * np.cos(A)
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1, 1)))


def hadec_to_altaz(ha, dec, lat):
    h, d, p = np.radians(ha), np.radians(dec), np.radians(lat)
    x = np.cos(d) * np.cos(h)  # toward meridian/equator point
    y = np.cos(d) * np.sin(h)  # toward west point
    z = np.sin(d)              # toward pole
    north = -np.sin(p) * x + np.cos(p) * z
    east = -y
    up = np.cos(p) * x + np.sin(p) * z
    alt = np.degrees(np.arcsin(np.clip(up, -1, 1)))
    az = np.degrees(np.arctan2(east, north)) % 360
    return alt, az


COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass(az):
    return COMPASS[int((float(az) % 360) / 22.5 + 0.5) % 16]


def wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def hadec_to_axes(ha, dec, side):
    if side == "east_looking":
        return wrap180(np.asarray(ha) + 90.0), np.asarray(dec, dtype=float)
    if side == "west_looking":
        return wrap180(np.asarray(ha) - 90.0), 180.0 - np.asarray(dec, dtype=float)
    raise ValueError(side)


def axes_to_hadec(a1, a2):
    """Pointing from mechanical angles; the side is implied by axis2 (<=90 east_looking)."""
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    east = a2 <= 90.0
    ha = np.where(east, a1 - 90.0, a1 + 90.0)
    dec = np.where(east, a2, 180.0 - a2)
    return wrap180(ha), dec


def axis2_to_dec(a2):
    a2 = np.asarray(a2, dtype=float)
    return np.where(a2 <= 90.0, a2, 180.0 - a2)


def pose_within_limits(a1, a2, mount_cfg):
    """Can the mount hold this pose? axis1 is limited from counterweight-down, axis2 by how far
    the tube may swing past the pole before it meets the tripod."""
    lo, hi = mount_cfg.get("axis2_limits", [-10.0, 190.0])
    return bool(abs(float(a1)) <= mount_cfg["axis1_hour_limit"] and lo <= float(a2) <= hi)


def choose_pose(ha, dec, mount_cfg, current=None):
    """Pick the pier side for a target: legal poses first, then the shortest move.

    Minimising |axis1| alone will happily fling the tube far past the pole, and the flip back can
    be a 115 deg Dec swing. Preferring the nearest legal pose avoids pointless meridian flips.
    """
    options = []
    for side in SIDES:
        a1, a2 = hadec_to_axes(ha, dec, side)
        a1, a2 = float(a1), float(a2)
        travel = 0.0 if current is None else max(abs(wrap180(a1 - current[0])), abs(a2 - current[1]))
        options.append({"side": side, "axes": [a1, a2], "travel": travel,
                        "ok": pose_within_limits(a1, a2, mount_cfg)})
    legal = [o for o in options if o["ok"]]
    if not legal:
        return None, options
    best = min(legal, key=(lambda o: o["travel"]) if current is not None
               else (lambda o: abs(o["axes"][0])))
    return best, options


def sky_metric(a2):
    """Weights to turn (d_axis1, d_axis2) into on-sky degrees: axis1 scales by cos(dec)."""
    return np.array([np.cos(np.radians(axis2_to_dec(a2))), 1.0])
