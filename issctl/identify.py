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
# Ranking the whole catalogue one satellite at a time took ~30 s on the Pi and held Python the
# whole time: the 50 Hz control loop stalled for up to 1.5 s at a go while a session was being
# named, with the last rate command latched - the mount lurched back and forth (USA 327,
# 2026-10-12). A vectorised SGP4 pass over all of them picks the nearest few for the exact ranking.
PREFILTER_ABOVE = 300     # catalogue size from which the quick pass runs
PREFILTER_KEEP = 200      # ...and how many of the nearest it keeps
PREFILTER_CHUNK = 1500    # satellites per step; the control loop gets Python back in between


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
    best = matches[0] if matches and kind else None
    _result_path(csv_path).write_text(json.dumps({
        "text": text.splitlines()[0] if text else "", "verdict": kind,
        "best": best["name"] if best else None, "best_id": best["id"] if best else None,
        "at": time.time()}))


def session_summary(csv_path):
    """What a session did, from its log, for History: how much of the time after the first lock a
    camera steered (shadow and obstructions left out), how far from the green cross the guide
    held the target (median, arcmin), how often it was lost, and how often the control loop ran
    late. Cached next to the log (<log>.sum.json) once the log has stopped growing."""
    p = Path(csv_path)
    st = p.stat()
    cache = p.with_suffix(".sum.json")
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("size") == st.st_size:
                return c
        except (OSError, ValueError):
            pass
    rows = list(csv.DictReader(open(p)))
    if len(rows) < 2:
        return None
    t = np.array([float(r["t"]) for r in rows])
    src = np.array([r["source"] for r in rows])
    step = np.diff(t, append=t[-1])
    locked = np.isin(src, ("guide", "main"))
    out = {"size": st.st_size, "late": int((np.diff(t) > 0.25).sum()),
           "late_max": round(float(np.diff(t).max()), 2)}
    if locked.any():
        first = int(np.argmax(locked))
        seen = ~np.isin(src[first:], ("shadow", "blocked"))
        span = float(step[first:][seen].sum())
        out["locked"] = round(float(step[first:][locked[first:]].sum()) / span, 3) if span > 0 else None
        out["lost"] = int(np.sum(locked[:-1] & (src[1:] == "predict")))
    side = p.with_suffix(".json")
    guide = (json.loads(side.read_text()).get("cameras") or {}).get("guide") if side.exists() else None
    if guide:
        b, scale = np.asarray(guide["boresight"], dtype=float), float(np.linalg.norm(
            np.asarray(guide["J"], dtype=float)[:, 1]))
        offs, last = [], None
        for r in rows:
            if r["source"] != "guide" or r["det_x"] in ("", "nan"):
                continue
            key = (r["det_x"], r["det_y"])
            if key != last:            # the log repeats the last detection between frames
                last = key
                offs.append(np.hypot(float(r["det_x"]) - b[0], float(r["det_y"]) - b[1]) / scale * 60)
        if offs:
            out["off_arcmin"] = round(float(np.median(offs)), 1)
    if time.time() - st.st_mtime > 30:
        try:
            cache.write_text(json.dumps(out))
        except OSError:
            pass
    return out


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


def observed(logs=None, tles=None):
    """{catalogue id: unix time last seen} of every satellite a session was identified as, for
    marking Coming up. Results saved before the id was kept have only the name: `tles` (the
    catalogue) looks those up by exact name, and without it they are skipped."""
    logs = Path(logs) if logs is not None else ROOT / "logs"
    by_name = None
    out = {}
    for res in logs.glob("*.id.json"):
        try:
            r = json.loads(res.read_text())
        except (OSError, ValueError):
            continue
        if r.get("verdict") not in ("sure", "probably") or not r.get("best"):
            continue
        sid = r.get("best_id")
        if not sid and tles is not None:
            if by_name is None:
                by_name = {}
                for name, i, _, _ in tles:
                    by_name.setdefault(name.strip().lower(), i)
            sid = by_name.get(r["best"].strip().lower())
        if not sid:
            continue
        t0, _ = _first_last_t(res.with_name(res.name.replace(".id.json", ".csv")))
        t = t0 if t0 is not None else float(r.get("at") or 0.0)
        key = str(sid).lstrip("0")
        out[key] = max(out.get(key, 0.0), t)
    return out


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
        try:
            summary = session_summary(p)
        except Exception:
            summary = None             # a damaged log must not take the whole list down
        out.append({"file": p.name, "kind": p.name.split("-")[0], "start": t0,
                    "duration": round(dur, 1), "name": name, "result": result, "summary": summary})
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
    if len(sats) > PREFILTER_ABOVE:
        sats = [sats[i] for i in nearest(t, track, sats, site, max(PREFILTER_KEEP, top))]
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


def nearest(t, track, sats, site, keep):
    """Indices of the `keep` satellites whose median separation from the track (alt/az unit
    vectors at unix times t) is smallest - all of them at once with SGP4 arrays, TEME turned to
    the Earth by GMST. Geometric and approximate (no polar motion, UT1 = UTC): it only chooses
    which few rank() then measures properly."""
    from sgp4.api import SatrecArray

    t = np.asarray(t, dtype=float)
    jd_all = t / 86400.0 + 2440587.5
    jd = np.floor(jd_all - 0.5) + 0.5
    fr = jd_all - jd
    du = jd_all - 2451545.0
    tc = du / 36525.0
    gmst = np.radians((280.46061837 + 360.98564736629 * du + 0.000387933 * tc ** 2
                       - tc ** 3 / 38710000.0) % 360.0)
    lat, lon = np.radians(site.lat), np.radians(site.lon)
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    nrad = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = getattr(site, "elevation", 0.0) / 1000.0
    here = np.array([(nrad + h) * np.cos(lat) * np.cos(lon), (nrad + h) * np.cos(lat) * np.sin(lon),
                     (nrad * (1 - e2) + h) * np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    c, s = np.cos(gmst), np.sin(gmst)
    med = np.full(len(sats), np.inf)
    for k in range(0, len(sats), PREFILTER_CHUNK):
        part = sats[k:k + PREFILTER_CHUNK]
        err, r, _ = SatrecArray([sat.model for _, _, sat in part]).sgp4(jd, fr)   # TEME, km
        x = c * r[..., 0] + s * r[..., 1] - here[0]
        y = -s * r[..., 0] + c * r[..., 1] - here[1]
        z = r[..., 2] - here[2]
        d = np.stack([x, y, z], axis=-1)
        enu = np.stack([d @ north, d @ east, d @ up], axis=-1)      # _altaz_unit's order
        enu /= np.linalg.norm(enu, axis=-1, keepdims=True)
        sep = np.degrees(np.arccos(np.clip(np.sum(enu * track, axis=-1), -1.0, 1.0)))
        sep[err != 0] = np.inf
        med[k:k + len(part)] = np.median(sep, axis=1)
        time.sleep(0)                       # let the control loop run between chunks
    med[~np.isfinite(med)] = np.inf
    return np.argsort(med)[:keep]


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
        self.best = None      # (name, catalogue id, satellite) while the label names one object

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
        best = res[0]["sat"] if kind in ("sure", "probably") else None
        if kind == "sure":
            label = res[0]["name"]
        elif kind == "probably":
            label = f"probably {res[0]['name']}"
        elif fits(res) and len(res) > 1 and res[1]["median_deg"] < LIKELY_DEG:
            label = f"{res[0]['name']} or {res[1]['name']}"     # a formation pair, too close to split yet
        else:
            label = ""
        if label:
            self.label, self.named_at, self.best = label, now, best
        elif now - self.named_at > self.HOLD_S:
            self.label, self.best = "", None
        return self.label


def sky_path(sat, site, t0, t1, step=10.0):
    """[[az, alt, unix t], ...] of a satellite above the horizon between t0 and t1, for the sky
    chart - the same form as a Coming-up row's path."""
    from . import predict as pr
    t = np.arange(t0, t1 + step, step)
    alt, az, _ = (sat - site.topos).at(pr.unix_to_time(t)).altaz()
    a, z = alt.degrees, az.degrees
    return [[round(float(z[i]), 2), round(float(a[i]), 2), round(float(t[i]), 1)]
            for i in range(len(t)) if a[i] > 0.0]


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
