"""Favourites: the satellites worth coming back to, and their next visible passes.

A small database in data/favorites.json, one entry per catalogue id. An entry comes from a History
session that Identify named ("Add to favourites"), or from a name or NORAD number typed in. Each
keeps the sessions it was seen in and, when no catalogue rates it, a standard magnitude of its own
- from a Brightness measurement taken during the session, else the default - which Coming up uses
too, so a favourite nobody rated still shows up there.

Passes are the forecast's (forecast.py), run over days instead of an hour and only for these:
sunlit while the sky here is dark, above min_altitude and inside the sky openings.
"""

import json
import time
from pathlib import Path

from . import forecast as fc
from . import identify as idf
from .config import ROOT

PATH = ROOT / "data" / "favorites.json"
HOURS = 48.0


def _key(sid):
    return str(sid).strip().lstrip("0")


def load(path=None):
    """{catalogue id: entry}. A missing or unreadable file is an empty database."""
    path = Path(path) if path is not None else PATH
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {_key(e["id"]): e for e in data.get("favorites", []) if e.get("id")}


def save(favs, path=None):
    path = Path(path) if path is not None else PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(favs.values(), key=lambda e: e.get("added", 0.0))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"favorites": rows}, indent=1))
    tmp.replace(path)


def add(favs, sid, name, std_mag=None, mag_from=None, session=None, now=None):
    """Add a satellite, or note another session of one already there. Returns the entry.
    `std_mag` is kept only for an object no catalogue rates; an existing rating is not replaced
    by the default."""
    k = _key(sid)
    e = favs.get(k)
    if e is None:
        e = favs[k] = {"id": k, "name": name, "added": float(now if now is not None else time.time()),
                       "std_mag": None, "mag_from": None, "sessions": []}
    e["name"] = name or e["name"]
    if std_mag is not None and (e.get("std_mag") is None or mag_from != "default"):
        e["std_mag"], e["mag_from"] = round(float(std_mag), 1), mag_from
    if session and session not in e["sessions"]:
        e["sessions"].append(session)
    return e


def remove(favs, query):
    """Remove by catalogue id or exact name (case ignored). Returns the entry, or None."""
    q = str(query).strip()
    k = _key(q) if q.isdigit() else next(
        (k for k, e in favs.items() if e["name"].strip().lower() == q.lower()), None)
    return favs.pop(k, None) if k else None


def ratings(favs):
    """{id: standard magnitude} of the favourites that carry their own, for the Forecaster."""
    return {k: e["std_mag"] for k, e in favs.items() if e.get("std_mag") is not None}


def migrate(favs, state, tles=None):
    """The ratings History's old "Add to Coming up" kept in state.json become favourites (it was
    the same gesture). Returns True when state changed and needs saving."""
    old = state.pop("std_mags", None)
    if not old:
        return False
    names = {_key(i): n for n, i, _, _ in (tles or [])}
    for sid, mag in old.items():
        add(favs, sid, names.get(_key(sid), f"#{_key(sid)}"), mag, "rated from History")
    return True


def satellites(favs, tles, rating, default_std_mag=5.0):
    """(name, id, skyfield satellite, standard magnitude) of every favourite the catalogues have
    an orbit for, ready for forecast.forecast. `rating(id)`: the catalogue's standard magnitude
    or None."""
    want = set(favs)
    keep = [t for t in tles if _key(t[1]) in want]
    out = []
    for name, sid, sat in idf.build(keep):
        k = _key(sid)
        std = rating(k)
        if std is None:
            std = favs[k].get("std_mag")
        out.append((favs[k]["name"] or name, sid, sat,
                    float(std if std is not None else default_std_mag)))
    return out


def passes(sats, site, t0, hours=HOURS, mask=None, min_alt=None, max_mag=99.0):
    """Visible passes of these satellites in the next `hours`, in the forecast's own rows
    (name, id, start, end, peak, mag, alt, az, range_km, path) sorted by start. `alt`/`az` are at
    the brightest moment; `max_alt` is the highest point while it is visible."""
    rows = fc.forecast(sats, site, t0, minutes=hours * 60.0, mask=mask, min_alt=min_alt,
                       max_mag=max_mag)
    for r in rows:
        r["max_alt"] = max(p[1] for p in r["path"]) if r["path"] else r["alt"]
    return rows
