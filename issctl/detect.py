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
           max_area=0, edge_margin=0, smooth=0.0):
    """Brightest compact blob above sigma*noise. gate=(x, y, radius) restricts the search.

    max_area rejects sprawling regions (a lit wall, a cloud edge) and edge_margin rejects blobs
    touching the frame border, where vignetting and out-of-focus scenery produce gradients that
    look like detections. Both are in full-resolution pixels; 0 disables them.

    smooth: Gaussian sigma in (binned) pixels applied first. A faint star or satellite spreads
    over a few pixels and the noise does not, so smoothing by about a star's width lifts it
    above the threshold: on real guide frames, 1 px and 5 sigma found 22 of 23 mag 5-6 stars
    that the unsmoothed 6 sigma found 4 of.

    Coordinates are in full-resolution pixels of the input image.

    This runs on every frame of both cameras, so it is written for the Pi: 93 ms per main frame
    capped the main camera below 10 fps. With a gate only a window round it is processed, and
    the centroids are summed over the blobs' own pixels rather than over every pixel.
    """
    h0, w0 = img.shape[:2]
    ox = oy = 0
    if gate is not None:
        # The window: the gate plus room for the background estimate, on even pixels so a Bayer
        # crop keeps its colour phase. Only worth it when it is a real saving.
        m = int(gate[2]) + 64
        x0, x1 = max(0, int(gate[0]) - m) & ~1, min(w0, int(gate[0]) + m + 2)
        y0, y1 = max(0, int(gate[1]) - m) & ~1, min(h0, int(gate[1]) + m + 2)
        if (x1 - x0) * (y1 - y0) < 0.25 * w0 * h0 and x1 - x0 >= 64 and y1 - y0 >= 64:
            img, ox, oy = img[y0:y1, x0:x1], x0, y0
    if bayer:
        h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        # the 2x2 sum, as an area resize of the float image - a third of the time of adding the
        # four strided planes
        g = cv2.resize(img[:h, :w].astype(np.float32), (w // 2, h // 2),
                       interpolation=cv2.INTER_AREA) * 4.0
        scale = 2
    else:
        g = img.astype(np.float32)
        scale = 1
    while g.shape[1] > max_width:
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        scale *= 2
    if smooth and smooth > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(smooth))

    h, w = g.shape
    coarse = cv2.resize(g, (max(w // 16, 4), max(h // 16, 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.blur(coarse, (5, 5)), (w, h), interpolation=cv2.INTER_LINEAR)
    r = g - bg
    sub = r[::6, ::6]           # 1 pixel in 36 is still tens of thousands: plenty for two medians
    med = float(np.median(sub))
    noise = 1.4826 * float(np.median(np.abs(sub - med))) + 1e-3
    mask = (r > med + sigma * noise).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[:, cv2.CC_STAT_AREA] * scale * scale
    valid = areas >= max(1, min_area)
    valid[0] = False
    if max_area:
        valid &= areas <= max_area
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    if edge_margin:
        # the edge of the FRAME, not of the window
        em = max(1, int(round(edge_margin / scale)))
        fl, ft = left + ox // scale, top + oy // scale
        fw, fh = (w0 // 2 * 2 if bayer else w0) // scale, (h0 // 2 * 2 if bayer else h0) // scale
        valid &= (fl >= em) & (ft >= em) & (fl + bw <= fw - em) & (ft + bh <= fh - em)
    if not valid.any():
        return None
    cand = np.nonzero(valid)[0]
    if len(cand) <= 32:
        # The usual case: a handful of blobs, each summed inside its own box.
        best = None
        for i in cand:
            xa, ya, bw_i, bh_i = int(left[i]), int(top[i]), int(bw[i]), int(bh[i])
            wts = np.where(labels[ya:ya + bh_i, xa:xa + bw_i] == i,
                           r[ya:ya + bh_i, xa:xa + bw_i] - med, 0.0)
            f = float(wts.sum())
            if f <= 0:
                continue
            yy, xx = np.mgrid[ya:ya + bh_i, xa:xa + bw_i]
            x = (float((wts * xx).sum()) / f + 0.5) * scale - 0.5 + ox
            y = (float((wts * yy).sum()) / f + 0.5) * scale - 0.5 + oy
            if gate is not None and (x - gate[0]) ** 2 + (y - gate[1]) ** 2 > gate[2] ** 2:
                continue
            if best is None or f > best[0]:
                best = (f, x, y, int(areas[i]))
        return None if best is None else Detection(best[1], best[2], best[0], best[3])
    # Many blobs (a noisy frame, scenery): flux and weighted centroid of all of them at once,
    # from their own pixels only.
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    wts = r[ys, xs] - med
    flux = np.bincount(lab, weights=wts, minlength=n)
    safe = np.maximum(flux, 1e-9)
    fx = (np.bincount(lab, weights=wts * xs, minlength=n) / safe + 0.5) * scale - 0.5 + ox
    fy = (np.bincount(lab, weights=wts * ys, minlength=n) / safe + 0.5) * scale - 0.5 + oy
    valid &= flux > 0
    if gate is not None:
        valid &= (fx - gate[0]) ** 2 + (fy - gate[1]) ** 2 <= gate[2] ** 2
    if not valid.any():
        return None
    i = int(np.argmax(np.where(valid, flux, -np.inf)))
    return Detection(float(fx[i]), float(fy[i]), float(flux[i]), int(areas[i]))


def snap(img, x, y, radius, sigma=6.0, min_area=3, bayer=False, smooth=0.0):
    """The centroid of the brightest blob within `radius` of a click, or None.

    A click on a star in a scaled-down browser image lands a few pixels off it, and one screen
    pixel is two or three sensor pixels; the star's own centroid is good to a fraction of one.
    That difference is the whole of what a boresight is for."""
    det = detect(img, sigma=sigma, min_area=min_area, bayer=bayer, gate=(x, y, radius),
                 smooth=smooth)
    return None if det is None else (det.x, det.y)
