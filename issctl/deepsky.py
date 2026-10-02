"""Deep-sky objects by name for goto, sync and centre by solve: Messier, NGC, IC and common names.

From OpenNGC (github.com/mattiaverga/OpenNGC, CC-BY-SA 4.0 - Mattia Verga): every NGC and IC
object, the Messier numbers, and the addendum for what the NGC lacks (M40, M45, Caldwell
objects, the Horsehead...). Downloaded once into data/catalog/, like the satellite catalogues, so
goto works offline after that.

Anything else - a star the short STARS list does not have, a Sharpless region - goes to CDS's
name resolver (Sesame) instead, which needs the internet.
"""

import csv
import io
import re
import threading
import urllib.request
from pathlib import Path

from .config import ROOT

CATALOG_DIR = ROOT / "data" / "catalog"
BASE = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/"
FILES = ("NGC.csv", "addendum.csv")
TYPES = {"*": "star", "**": "double star", "*Ass": "association of stars", "OCl": "open cluster",
         "GCl": "globular cluster", "Cl+N": "cluster with nebula", "G": "galaxy",
         "GPair": "galaxy pair", "GTrpl": "galaxy triplet", "GGroup": "group of galaxies",
         "PN": "planetary nebula", "HII": "HII region", "DrkN": "dark nebula",
         "EmN": "emission nebula", "Neb": "nebula", "RfN": "reflection nebula",
         "SNR": "supernova remnant", "Nova": "nova", "Dup": "duplicate entry"}

_index = None
_lock = threading.Lock()


def key(text):
    """'M 31', 'Messier 31', 'm031' -> 'm31'; 'NGC 224', 'NGC0224' -> 'ngc224'; names lowercased
    with everything but letters and digits dropped."""
    k = re.sub(r"[^a-z0-9]", "", str(text).lower())
    k = re.sub(r"^messier(?=\d)", "m", k)
    return re.sub(r"^([a-z]+)0+(?=\d)", r"\1", k)


def fetch(directory=None, log=print, offline=False):
    """The catalogue files, downloading any that are missing. They hardly ever change."""
    directory = Path(directory or CATALOG_DIR)
    for name in FILES:
        path = directory / name
        if path.exists() or offline:
            continue
        try:
            req = urllib.request.Request(BASE + name, headers={"User-Agent": "issctl"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if b"Name;Type;RA;Dec" not in data[:200]:
                raise ValueError("not the OpenNGC table")
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            log(f"deep-sky catalogue {name}: downloaded")
        except Exception as e:
            log(f"deep-sky catalogue {name}: could not download ({e})")
    return [directory / n for n in FILES if (directory / n).exists()]


def _dms(text, hours):
    sign = -1.0 if text.strip().startswith("-") else 1.0
    d, m, s = (abs(float(x)) for x in text.strip().lstrip("+-").split(":"))
    return sign * (d + m / 60 + s / 3600) * (15.0 if hours else 1.0)


def build(paths):
    """{designation key: entry} and [(common-name key, entry)]. A real entry beats a 'Dup' one
    for the same key; nonexistent objects and ones without a position are left out."""
    names, common = {}, []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for r in csv.DictReader(io.StringIO(text), delimiter=";"):
            if not r.get("RA") or r.get("Type") == "NonEx":
                continue
            try:
                ra, dec = _dms(r["RA"], True), _dms(r["Dec"], False)
            except ValueError:
                continue
            mag = r.get("V-Mag") or r.get("B-Mag") or ""
            e = {"name": r["Name"], "type": r["Type"], "ra": ra, "dec": dec,
                 "messier": int(r["M"]) if r.get("M") else None,
                 "common": [c.strip() for c in (r.get("Common names") or "").split(",") if c.strip()],
                 "mag": float(mag) if mag else None}
            keys = [key(r["Name"])] + ([f"m{e['messier']}"] if e["messier"] else [])
            for k in keys:
                if k not in names or names[k]["type"] == "Dup":
                    names[k] = e
            common += [(key(c), e) for c in e["common"]]
    return names, common


def _get_index(directory=None, log=print, offline=False):
    global _index
    with _lock:
        if _index is None:
            paths = fetch(directory, log=log, offline=offline)
            if not paths:
                return None
            _index = build(paths)
        return _index


def lookup(text, directory=None, log=print, offline=False):
    """The catalogue entry for a designation or common name, or None if it is not a deep-sky
    object here. A partial common name ('andromeda') is accepted when it fits only one object;
    when it fits several, ValueError names them."""
    idx = _get_index(directory, log=log, offline=offline)
    k = key(text)
    if idx is None or not k:
        return None
    names, common = idx
    if k in names:
        return names[k]
    exact = [e for c, e in common if c == k]
    if exact:
        return exact[0]
    if len(k) < 4:
        return None
    hits = {id(e): e for c, e in common if k in c}
    if len(hits) == 1:
        return next(iter(hits.values()))
    if hits:
        options = sorted({e["common"][0] for e in hits.values()})
        raise ValueError(f"'{text}' fits {len(options)} objects: {', '.join(options[:6])}"
                         + (" ..." if len(options) > 6 else ""))
    return None


def describe(e):
    """'M31 = NGC0224, Andromeda Galaxy - galaxy, mag 3.4'."""
    if e["type"] == "Dup" and e["messier"]:     # M102: listed as another name for M101
        return f"{e['name']} - the same object as M{e['messier']}"
    label = f"M{e['messier']} = {e['name']}" if e["messier"] and e["name"] != f"M{e['messier']}" \
        else e["name"]
    extra = [TYPES.get(e["type"], e["type"])] + ([f"mag {e['mag']:.1f}"] if e["mag"] is not None else [])
    return label + (f", {e['common'][0]}" if e["common"] else "") + " - " + ", ".join(extra)


def resolve_online(text):
    """CDS Sesame, for names no local list has. Needs the internet; (ra_deg, dec_deg) or None."""
    from astropy.coordinates import SkyCoord
    try:
        c = SkyCoord.from_name(text)
    except Exception:
        return None
    return float(c.icrs.ra.deg), float(c.icrs.dec.deg)
