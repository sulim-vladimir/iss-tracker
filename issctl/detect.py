"""Bright-blob detection for the ISS (and calibration stars)."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    flux: float
    area: int
    t: float = 0.0


def detect(img, sigma=6.0, min_area=3, bayer=False, gate=None, max_width=1300,
           max_area=0, edge_margin=0):
    """Brightest compact blob above sigma*noise. gate=(x, y, radius) restricts the search.

    max_area rejects sprawling regions (a lit wall, a cloud edge) and edge_margin rejects blobs
    touching the frame border, where vignetting and out-of-focus scenery produce gradients that
    look like detections. Both are in full-resolution pixels; 0 disables them.

    Coordinates are in full-resolution pixels of the input image.
    """
    if bayer:
        h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        g = img[:h, :w].astype(np.float32)
        g = g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2]
        scale = 2
    else:
        g = img.astype(np.float32)
        scale = 1
    while g.shape[1] > max_width:
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        scale *= 2

    h, w = g.shape
    coarse = cv2.resize(g, (max(w // 16, 4), max(h // 16, 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.blur(coarse, (5, 5)), (w, h), interpolation=cv2.INTER_LINEAR)
    r = g - bg
    sub = r[::3, ::3]
    med = float(np.median(sub))
    noise = 1.4826 * float(np.median(np.abs(sub - med))) + 1e-3
    mask = (r > med + sigma * noise).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    lab = labels.ravel()
    wts = np.where(lab > 0, r.ravel() - med, 0.0)
    ys, xs = np.divmod(np.arange(lab.size), w)
    flux = np.bincount(lab, weights=wts, minlength=n)
    cx = np.bincount(lab, weights=wts * xs, minlength=n) / np.maximum(flux, 1e-9)
    cy = np.bincount(lab, weights=wts * ys, minlength=n) / np.maximum(flux, 1e-9)
    fx = (cx + 0.5) * scale - 0.5
    fy = (cy + 0.5) * scale - 0.5

    areas = stats[:, cv2.CC_STAT_AREA] * scale * scale
    valid = areas >= max(1, min_area)
    valid[0] = False
    if max_area:
        valid &= areas <= max_area
    if edge_margin:
        m = max(1, int(round(edge_margin / scale)))
        left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        right = left + stats[:, cv2.CC_STAT_WIDTH]
        bottom = top + stats[:, cv2.CC_STAT_HEIGHT]
        valid &= (left >= m) & (top >= m) & (right <= w - m) & (bottom <= h - m)
    if gate is not None:
        gx, gy, gr = gate
        valid &= (fx - gx) ** 2 + (fy - gy) ** 2 <= gr * gr
    if not valid.any():
        return None
    i = int(np.argmax(np.where(valid, flux, -np.inf)))
    return Detection(float(fx[i]), float(fy[i]), float(flux[i]), int(areas[i]))
