"""Identify: name the satellite a servo (or pass) session followed.

The session CSV has the mount counters and, whenever a camera saw the object, where in the image
it was. Through the star alignment and the guide matrix that is the object's own sky track, and
every catalogued satellite is compared against it: the one that flew the same path at the same
moments is the answer.

Two catalogues, because the interesting ones are often not in the public one:

* CelesTrak "active" plus "visual" (the brightest objects, rocket bodies included);
* bright.tle: every other object McCants rates bright and still in orbit - dead satellites and
  rocket bodies, which "active" leaves out (forecast.refresh_bright builds it);
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
# Built by forecast.refresh_bright, not downloaded here: it needs qs.mag to know what to fetch.
# Read last, so the curated files above win when an object is in both.
EXTRA = ("bright.tle",)
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
    return [directory / n for n in (*SOURCES, *EXTRA) if (directory / n).exists()]


def load_tles(paths):
    """(name, catalogue id, line1, line2), one per object: a later file never duplicates an id.
    Ids are kept as text - classified catalogues use letters in them."""
    return parse_tles(Path(path).read_text(errors="replace") for path in paths)


def parse_tles(texts):
    """load_tles on text already in memory: one string, or several (earlier ones win)."""
    out, seen = [], set()
    for text in ([texts] if isinstance(texts, str) else texts):
        lines = [ln.rstrip() for ln in text.splitlines()]
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


def find_satellite(query, tles):
    """Catalogue entries matching a NORAD number or a name: an exact number or name wins, else
    every name containing the text (case ignored). Returns a list of (name, id, line1, line2)."""
    q = str(query).strip()
    if q.isdigit():
        return [t for t in tles if t[1].lstrip("0") == q.lstrip("0")]
    low = q.lower()
    exact = [t for t in tles if t[0].lower() == low or t[1].lower() == low]
    return exact or [t for t in tles if low in t[0].lower()]


def session_state(csv_path, state):
    """The alignment and camera matrices in force when the session ran: the sidecar written at
    the start if there is one, otherwise today's (and a note saying so)."""
    side = Path(csv_path).with_suffix(".json")
    if side.exists():
        return json.loads(side.read_text()), None
    return state, "no record of the alignment at the time - using the current one"


def write_session_state(csv_path, state, name=None):
    """Called at session start, so a later re-alignment cannot skew this session's answer.
    `name`: what a pass session set out to track, for the history list."""
    keep = {"alignment": {"model": (state.get("alignment") or {}).get("model")},
            "cameras": state.get("cameras", {}), "name": name}
    Path(csv_path).with_suffix(".json").write_text(json.dumps(keep))


def _result_path(csv_path):
    return Path(csv_path).with_suffix(".id.json")


def save_result(csv_path, matches, text):
    """Keep what Identify said about a session, for the history list."""
    kind = verdict(matches)
    best = matches[0]["name"] if matches else None
    _result_path(csv_path).write_text(json.dumps({
        "text": text.splitlines()[0] if text else "", "verdict": kind,
        "best": best if kind else None, "at": time.time()}))


def _first_last_t(path):
    """Start time and duration of a session log without reading all of it."""
    with open(path, "rb") as f:
        f.readline()
        first = f.readline()
        if not first:
            return None, 0.0
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4096))
        last = f.read().splitlines()[-1]
    try:
        t0, t1 = float(first.split(b",")[0]), float(last.split(b",")[0])
    except ValueError:
        return None, 0.0
    return t0, max(0.0, t1 - t0)


def list_sessions(logs=None, limit=20):
    """The latest real sessions (not simulations), newest first: file, kind (servo/track),
    start (unix), duration (s), name (what a pass tracked), result (a saved identification)."""
    import re

    logs = Path(logs) if logs is not None else ROOT / "logs"
    runs = [p for p in list(logs.glob("servo-*.csv")) + list(logs.glob("track-*.csv"))
            if re.fullmatch(r"(servo|track)-\d{8}-\d{6}\.csv", p.name)]
    out = []
    for p in sorted(runs, key=lambda p: p.name[6:], reverse=True)[:limit]:
        t0, dur = _first_last_t(p)
        if t0 is None:
            continue
        side, res = p.with_suffix(".json"), _result_path(p)
        name = json.loads(side.read_text()).get("name") if side.exists() else None
        result = json.loads(res.read_text()) if res.exists() else None
        out.append({"file": p.name, "kind": p.name.split("-")[0], "start": t0,
                    "duration": round(dur, 1), "name": name, "result": result})
    return out


def object_axes(state, axes, px, source, frame=(1280, 960)):
    """Counters that would put the OBJECT on the guide frame centre - the direction the
    alignment describes. A guide detection is carried there through the guide matrix; main holds
    it on its centre, which the guide boresight marks."""
    axes = np.asarray(axes, dtype=float)
    guide = (state.get("cameras") or {}).get("guide")
    if not guide:
        return axes
    centre = np.array([(frame[0] - 1) / 2.0, (frame[1] - 1) / 2.0])
    px = np.asarray(px if source == "guide" else guide["boresight"], dtype=float)
    return axes + np.linalg.solve(jacobian(guide, axes[1]), centre - px)


def sky_vectors(state, axes_list):
    return np.array([align.sky_unit(*align.pointing_hadec(state, a)) for a in axes_list])


def read_track(csv_path, state, frame=(1280, 960), samples=SAMPLES):
    """(unix times, sky unit vectors in the local HA/Dec frame) of the OBJECT, from the frames
    where a camera saw it."""
    rows = list(csv.DictReader(open(csv_path)))
    seen, last = [], None
    for r in rows:
        if r["source"] not in ("guide", "main") or r["det_x"] in ("", "nan"):
            continue
        key = (r["source"], r["det_x"], r["det_y"])
        if key == last:
            continue            # the CSV repeats the last detection between frames
        last = key
        px = (float(r["det_x"]), float(r["det_y"]))
        seen.append((float(r["t"]), object_axes(state, (float(r["a1"]), float(r["a2"])), px,
                                                r["source"], frame)))
    if len(seen) < 3:
        raise ValueError(f"{Path(csv_path).name}: the cameras saw the object in fewer than 3 "
                         f"frames - nothing to identify")
    pick = np.unique(np.linspace(0, len(seen) - 1, min(samples, len(seen))).round().astype(int))
    return (np.array([seen[i][0] for i in pick]), sky_vectors(state, [seen[i][1] for i in pick]))


def build(tles):
    """Catalogue lines as skyfield satellites, once: constructing 16 000 of them is most of the
    time a ranking takes."""
    from skyfield.api import EarthSatellite, load

    ts = load.timescale()
    out = []
    for name, sid, l1, l2 in tles:
        try:
            out.append((name, sid, EarthSatellite(l1, l2, name, ts)))
        except Exception:
            continue
    return out


def identify(t, v, tles, site, top=5):
    """Rank catalogue objects by how closely they flew the track (median separation, deg)."""
    return rank(t, v, build(tles), site, top)


def rank(t, v, sats, site, top=5):
    """identify() over satellites already built."""
    from skyfield.api import load

    ts = load.timescale()
    when = ts.from_datetimes([_utc(x) for x in t])
    alt, az = geo.hadec_to_altaz(*_hadec(v), site.lat)      # skyfield answers in alt/az
    track = _altaz_unit(np.asarray(alt), np.asarray(az))
    res = []
    for name, sid, sat in sats:
        try:
            a, z, d = (sat - site.topos).at(when).altaz()
        except Exception:
            continue
        w = _altaz_unit(a.degrees, z.degrees)
        sep = np.degrees(np.arccos(np.clip(np.sum(w * track, axis=-1), -1.0, 1.0)))
        res.append({"name": name, "id": sid, "median_deg": float(np.median(sep)),
                    "max_deg": float(sep.max()), "range_km": float(np.mean(d.km)),
                    "alt": [float(a.degrees[0]), float(a.degrees[-1])],
                    "az": [float(z.degrees[0]), float(z.degrees[-1])], "sat": (name, sid, sat)})
    res.sort(key=lambda r: r["median_deg"])
    return res[:top]


def verdict(matches):
    """'sure', 'probably' or None for the best of a ranking - the rule describe() words."""
    if not matches:
        return None
    best = matches[0]
    runner = matches[1]["median_deg"] if len(matches) > 1 else np.inf
    clear = runner > 2 * best["median_deg"]
    if best["median_deg"] < MATCH_DEG and clear:
        return "sure"
    if best["median_deg"] < LIKELY_DEG and clear:
        return "probably"
    return None


class LiveIdentifier:
    """Names the object while it is being followed, for the guide caption.

    Fed each new detection; every few seconds it ranks the catalogue against the last WINDOW_S of
    them. The first ranking takes the whole catalogue, a few seconds on the Pi; after that only
    the leading candidates are re-ranked, which is fast, until the fit gets worse and the whole
    catalogue is searched again. The catalogue is built once and reused for every session."""

    WINDOW_S = 30.0
    EVERY_S = 5.0
    CANDIDATES = 40

    def __init__(self, site, frame=(1280, 960), catalog_dir=CATALOG_DIR, log=print):
        self.site, self.frame, self.catalog_dir, self.log = site, frame, catalog_dir, log
        self.sats = None
        self.reset()

    HOLD_S = 15.0     # keep the last name this long through one ranking that fits nothing

    def reset(self):
        self.samples, self.candidates, self.label, self.last_rank = [], None, "", -1e9
        self.named_at = -1e9

    def add(self, t, state, axes, px, source):
        if not (state.get("alignment") or {}).get("model"):
            return          # without the star alignment the counters say nothing about the sky
        self.samples.append((float(t), object_axes(state, axes, px, source, self.frame), state))
        cut = float(t) - self.WINDOW_S
        self.samples = [x for x in self.samples if x[0] >= cut]

    def update(self, now):
        """Rank if it is time and there is enough to rank. Returns the label."""
        if now - self.last_rank < self.EVERY_S or len(self.samples) < 4 \
                or self.samples[-1][0] - self.samples[0][0] < 3.0:
            return self.label
        self.last_rank = now
        if self.sats is None:
            self.sats = build(load_tles(refresh_catalogs(self.catalog_dir, log=self.log)))
        pick = np.unique(np.linspace(0, len(self.samples) - 1, min(12, len(self.samples)))
                         .round().astype(int))
        t = np.array([self.samples[i][0] for i in pick])
        v = np.array([sky_vectors(self.samples[i][2], [self.samples[i][1]])[0] for i in pick])
        fits = lambda r: bool(r) and r[0]["median_deg"] < LIKELY_DEG
        res = rank(t, v, self.candidates or self.sats, self.site, top=self.CANDIDATES)
        if self.candidates is not None and not fits(res):
            res = rank(t, v, self.sats, self.site, top=self.CANDIDATES)     # lost it: search all
        self.candidates = [r["sat"] for r in res]
        kind = verdict(res)
        if kind == "sure":
            label = res[0]["name"]
        elif kind == "probably":
            label = f"probably {res[0]['name']}"
        elif fits(res) and len(res) > 1 and res[1]["median_deg"] < LIKELY_DEG:
            label = f"{res[0]['name']} or {res[1]['name']}"     # a formation pair, too close to split yet
        else:
            label = ""
        if label:
            self.label, self.named_at = label, now
        elif now - self.named_at > self.HOLD_S:
            self.label = ""
        return self.label


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
        return "identify: no catalogue to compare with"
    best = matches[0]
    runner = matches[1]["median_deg"] if len(matches) > 1 else np.inf
    clear = runner > 2 * best["median_deg"]
    where = (f"{best['name']} (catalogue {best['id']}), {best['median_deg']:.2f} deg from the "
             f"track, {best['range_km']:.0f} km away")
    if best["median_deg"] < MATCH_DEG:
        lines = [f"identify: {where}"
                 + ("" if clear else " - but a close neighbour matches almost as well")]
    elif best["median_deg"] < LIKELY_DEG and clear:
        lines = [f"identify: probably {where} - a loose fit (few detections, or the "
                 f"alignment is off)"]
    else:
        lines = [f"identify: nothing in the catalogues flew this path - the nearest, "
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
    text = describe(matches, note)
    try:
        save_result(csv_path, matches, text)
    except OSError:
        pass
    return matches, text
