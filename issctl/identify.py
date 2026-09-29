"""What was that? Name the satellite a servo (or pass) session followed.

The session CSV has the mount counters and, whenever a camera saw the object, where in the image
it was. Through the star alignment and the guide matrix that is the object's own sky track, and
every catalogued satellite is compared against it: the one that flew the same path at the same
moments is the answer.

Two catalogues, because the interesting ones are often not in the public one:

* CelesTrak "active" plus "visual" (the brightest objects, rocket bodies included);
* Mike McCants' amateur-observed orbits of classified satellites. The first real identification
  (2026-09-29) was NOSS 3-8 (B), which only this one has - the public catalogue's best was 6 deg
  off.

The mount's own alt/az columns are NOT used: the tracker writes them for an ideal, polar-aligned
mount, several degrees off on a tripod 3.6 deg from the pole.
"""

import csv
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import align
from . import geometry as geo
from .calib import jacobian
from .config import ROOT

CATALOG_DIR = ROOT / "data" / "catalog"
SOURCES = {
    "active.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "visual.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
    "classfd.tle": "https://www.mmccants.org/tles/classfd.zip",
}
MAX_AGE_H = 24.0
SAMPLES = 24          # points along the track: plenty to tell a formation pair apart
MATCH_DEG = 0.5       # median separation below which a match is called confident
LIKELY_DEG = 1.5      # ...and "probably", if nothing else comes within twice as close


def refresh_catalogs(directory=CATALOG_DIR, max_age_h=MAX_AGE_H, log=print, offline=False):
    """Download what is missing or older than max_age_h; keep the old copy when offline."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = directory / name
        if offline or (path.exists() and time.time() - path.stat().st_mtime < max_age_h * 3600):
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "issctl"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    data = z.read(next(n for n in z.namelist() if n.endswith(".tle")))
            if b"\n1 " not in data:
                raise ValueError("no orbits in the reply")
            path.write_bytes(data)
            log(f"catalogue {name}: updated")
        except Exception as e:
            log(f"catalogue {name}: could not update ({e})"
                + (" - using the copy from before" if path.exists() else ""))
    return [directory / n for n in SOURCES if (directory / n).exists()]


def load_tles(paths):
    """(name, catalogue id, line1, line2), one per object: a later file never duplicates an id.
    Ids are kept as text - classified catalogues use letters in them."""
    out, seen = [], set()
    for path in paths:
        lines = [ln.rstrip() for ln in Path(path).read_text(errors="replace").splitlines()]
        for i in range(len(lines) - 1):
            l1, l2 = lines[i], lines[i + 1]
            if not (l1.startswith("1 ") and l2.startswith("2 ")):
                continue
            sid = l1[2:7].strip()
            if sid in seen:
                continue
            seen.add(sid)
            prev = lines[i - 1].strip() if i > 0 else ""
            name = prev if prev and not prev.startswith(("1 ", "2 ")) else sid
            out.append((name[2:] if name.startswith("0 ") else name, sid, l1, l2))
    return out


def session_state(csv_path, state):
    """The alignment and camera matrices in force when the session ran: the sidecar written at
    the start if there is one, otherwise today's (and a note saying so)."""
    side = Path(csv_path).with_suffix(".json")
    if side.exists():
        return json.loads(side.read_text()), None
    return state, "no record of the alignment at the time - using the current one"


def write_session_state(csv_path, state):
    """Called at session start, so a later re-alignment cannot skew this session's answer."""
    keep = {"alignment": {"model": (state.get("alignment") or {}).get("model")},
            "cameras": state.get("cameras", {})}
    Path(csv_path).with_suffix(".json").write_text(json.dumps(keep))


def read_track(csv_path, state, frame=(1280, 960), samples=SAMPLES):
    """(unix times, sky unit vectors in the local HA/Dec frame) of the OBJECT, from the frames
    where a camera saw it. The guide pixel is carried onto the guide frame centre - the direction
    the alignment describes - through the guide matrix."""
    rows = list(csv.DictReader(open(csv_path)))
    guide = (state.get("cameras") or {}).get("guide")
    centre = np.array([(frame[0] - 1) / 2.0, (frame[1] - 1) / 2.0])
    seen, last = [], None
    for r in rows:
        if r["source"] not in ("guide", "main") or r["det_x"] in ("", "nan"):
            continue
        key = (r["source"], r["det_x"], r["det_y"])
        if key == last:
            continue            # the CSV repeats the last detection between frames
        last = key
        axes = np.array([float(r["a1"]), float(r["a2"])])
        if guide:
            J = jacobian(guide, axes[1])
            if r["source"] == "guide":
                px = np.array([float(r["det_x"]), float(r["det_y"])])
            else:           # main holds it on the main boresight = the guide boresight
                px = np.asarray(guide["boresight"], dtype=float)
            axes = axes + np.linalg.solve(J, centre - px)
        seen.append((float(r["t"]), axes))
    if len(seen) < 3:
        raise ValueError(f"{Path(csv_path).name}: the cameras saw the object in fewer than 3 "
                         f"frames - nothing to identify")
    pick = np.unique(np.linspace(0, len(seen) - 1, min(samples, len(seen))).round().astype(int))
    t = np.array([seen[i][0] for i in pick])
    v = np.array([align.sky_unit(*align.pointing_hadec(state, seen[i][1])) for i in pick])
    return t, v


def identify(t, v, tles, site, top=5):
    """Rank catalogue objects by how closely they flew the track (median separation, deg)."""
    from skyfield.api import EarthSatellite, load

    ts = load.timescale()
    when = ts.from_datetimes([_utc(x) for x in t])
    here = site.topos
    alt, az = geo.hadec_to_altaz(*_hadec(v), site.lat)      # skyfield answers in alt/az
    track = _altaz_unit(np.asarray(alt), np.asarray(az))
    res = []
    for name, sid, l1, l2 in tles:
        try:
            sat = EarthSatellite(l1, l2, name, ts)
            a, z, d = (sat - here).at(when).altaz()
        except Exception:
            continue
        w = _altaz_unit(a.degrees, z.degrees)
        sep = np.degrees(np.arccos(np.clip(np.sum(w * track, axis=-1), -1.0, 1.0)))
        res.append({"name": name, "id": sid, "median_deg": float(np.median(sep)),
                    "max_deg": float(sep.max()), "range_km": float(np.mean(d.km)),
                    "alt": [float(a.degrees[0]), float(a.degrees[-1])],
                    "az": [float(z.degrees[0]), float(z.degrees[-1])]})
    res.sort(key=lambda r: r["median_deg"])
    return res[:top]


def _hadec(v):
    v = np.asarray(v, dtype=float)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])),
            np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0))))


def _altaz_unit(alt, az):
    a, z = np.radians(alt), np.radians(az)
    return np.stack([np.cos(a) * np.cos(z), np.cos(a) * np.sin(z), np.sin(a)], axis=-1)


def _utc(x):
    import datetime
    return datetime.datetime.fromtimestamp(float(x), tz=datetime.timezone.utc)


def describe(matches, note=None):
    """One line for the console, then the runners-up."""
    if not matches:
        return "what was that: no catalogue to compare with"
    best = matches[0]
    runner = matches[1]["median_deg"] if len(matches) > 1 else np.inf
    clear = runner > 2 * best["median_deg"]
    where = (f"{best['name']} (catalogue {best['id']}), {best['median_deg']:.2f} deg from the "
             f"track, {best['range_km']:.0f} km away")
    if best["median_deg"] < MATCH_DEG:
        lines = [f"what was that: {where}"
                 + ("" if clear else " - but a close neighbour matches almost as well")]
    elif best["median_deg"] < LIKELY_DEG and clear:
        lines = [f"what was that: probably {where} - a loose fit (few detections, or the "
                 f"alignment is off)"]
    else:
        lines = [f"what was that: nothing in the catalogues flew this path - the nearest, "
                 f"{best['name']}, stayed {best['median_deg']:.1f} deg away. An aircraft, or an "
                 f"object no catalogue here carries"]
    lines += [f"  {m['name']} ({m['id']}): {m['median_deg']:.2f} deg, max {m['max_deg']:.2f}, "
              f"{m['range_km']:.0f} km" for m in matches[1:4]]
    if note:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def latest_session(logs=ROOT / "logs"):
    runs = sorted(Path(logs).glob("servo-*.csv")) + sorted(Path(logs).glob("track-*.csv"))
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def what_was_that(csv_path, state, site, frame=(1280, 960), log=print, offline=False,
                  catalog_dir=CATALOG_DIR):
    """The whole job: catalogues, track, ranking. Returns (matches, text)."""
    used, note = session_state(csv_path, state)
    t, v = read_track(csv_path, used, frame)
    tles = load_tles(refresh_catalogs(catalog_dir, log=log, offline=offline))
    if not tles:
        raise RuntimeError("no satellite catalogue - connect to the internet once")
    matches = identify(t, v, tles, site)
    return matches, describe(matches, note)
