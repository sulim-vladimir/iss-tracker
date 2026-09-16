"""Static sky obstructions: what a balcony, window frame or roofline hides.

The sky is described by rectangles in azimuth/altitude degrees:
  openings  - the sky is only usable inside one of these (empty = the whole sky)
  blockers  - subtracted from the openings (a window frame post, a mast, a tree)
Azimuth ranges may wrap through north, e.g. [350, 20].
"""

import numpy as np


def _in_az(az, lo, hi):
    az = np.asarray(az) % 360.0
    lo, hi = lo % 360.0, hi % 360.0
    return (lo <= az) & (az <= hi) if lo <= hi else (az >= lo) | (az <= hi)


def _inside(alt, az, rect):
    az_min, az_max, alt_min, alt_max = rect
    return _in_az(az, az_min, az_max) & (np.asarray(alt) >= alt_min) & (np.asarray(alt) <= alt_max)


class SkyMask:
    def __init__(self, openings=(), blockers=()):
        self.openings = [list(map(float, r)) for r in openings]
        self.blockers = [list(map(float, r)) for r in blockers]

    @classmethod
    def from_config(cls, cfg):
        sky = cfg.get("site", {}).get("sky", {})
        return cls(sky.get("openings", ()), sky.get("blockers", ()))

    @property
    def empty(self):
        return not self.openings and not self.blockers

    def visible(self, alt, az):
        """True where the sky can actually be seen."""
        alt = np.asarray(alt, dtype=float)
        az = np.asarray(az, dtype=float)
        if self.openings:
            vis = np.zeros(alt.shape, dtype=bool)
            for r in self.openings:
                vis |= _inside(alt, az, r)
        else:
            vis = np.ones(alt.shape, dtype=bool)
        for r in self.blockers:
            vis &= ~_inside(alt, az, r)
        return vis

    def describe(self):
        if self.empty:
            return "whole sky"
        parts = [f"open az {r[0]:.0f}-{r[1]:.0f} alt {r[2]:.0f}-{r[3]:.0f}" for r in self.openings]
        parts += [f"blocked az {r[0]:.0f}-{r[1]:.0f} alt {r[2]:.0f}-{r[3]:.0f}" for r in self.blockers]
        return "; ".join(parts)


def segments(t, ok, min_length=2.0):
    """Contiguous [start, end] runs where ok is True, longer than min_length seconds."""
    out = []
    start = None
    for i, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if t[i - 1] - t[start] >= min_length:
                out.append((float(t[start]), float(t[i - 1])))
            start = None
    return out
