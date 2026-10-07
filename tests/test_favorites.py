"""Favourites: the database of satellites worth coming back to, and their passes."""

from pathlib import Path

import numpy as np
import pytest

from issctl import favorites as fav
from issctl import forecast as fc
from issctl import identify as idf
from issctl import predict as pr
from issctl.config import load_config

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def site():
    return pr.Site(load_config(Path(__file__).parent.parent / "config.example.toml"))


def test_survives_a_restart_and_a_missing_file(tmp_path):
    path = tmp_path / "favorites.json"
    assert fav.load(path) == {}
    favs = {}
    fav.add(favs, "042065", "NOSS 3-8 (B)", 3.0, "measured mag 4.1 at 1400 km",
            session="servo-20261002-230054.csv", now=1.0)
    fav.add(favs, "46984", "SENTINEL-6A", now=2.0)
    fav.save(favs, path)
    back = fav.load(path)
    assert list(back) == ["42065", "46984"]                 # leading zeros do not make a new one
    assert back["42065"]["std_mag"] == 3.0 and back["42065"]["sessions"] == ["servo-20261002-230054.csv"]
    assert fav.ratings(back) == {"42065": 3.0}              # only the ones carrying their own


def test_adding_again_keeps_the_entry_and_a_measured_rating(tmp_path):
    favs = {}
    fav.add(favs, "42065", "NOSS 3-8 (B)", 3.4, "measured mag 4.1", session="a.csv")
    fav.add(favs, "42065", "NOSS 3-8 (B)", 5.0, "default", session="b.csv")
    e = favs["42065"]
    assert e["std_mag"] == 3.4 and e["sessions"] == ["a.csv", "b.csv"]
    fav.add(favs, "42065", "NOSS 3-8 (B)", 2.9, "measured mag 3.6", session="b.csv")
    assert e["std_mag"] == 2.9 and e["sessions"] == ["a.csv", "b.csv"]


def test_remove_by_number_or_name():
    favs = {}
    fav.add(favs, "42058", "NOSS 3-8 (A)")
    fav.add(favs, "42065", "NOSS 3-8 (B)")
    assert fav.remove(favs, "042058")["name"] == "NOSS 3-8 (A)"
    assert fav.remove(favs, "noss 3-8 (b)")["id"] == "42065"
    assert fav.remove(favs, "42065") is None and favs == {}


def test_old_coming_up_ratings_become_favourites():
    """History's "Add to Coming up" kept ratings in state.json; they move, named from the catalogue."""
    tles = idf.load_tles([DATA / "sample.tle"])
    state, favs = {"std_mags": {"42065": 3.1}}, {}
    assert fav.migrate(favs, state, tles)
    assert "std_mags" not in state
    assert favs["42065"]["name"] == "NOSS 3-8 (B)" and fav.ratings(favs) == {"42065": 3.1}
    assert not fav.migrate(favs, state, tles)


def test_satellites_take_the_catalogue_rating_else_their_own_else_the_default():
    tles = idf.load_tles([DATA / "sample.tle"])
    favs = {}
    fav.add(favs, "42058", "NOSS 3-8 (A)")
    fav.add(favs, "42065", "NOSS 3-8 (B)", 3.3, "measured")
    fav.add(favs, "67006", "STARLINK-36156")
    fav.add(favs, "99999", "NOT IN ANY CATALOGUE")
    sats = fav.satellites(favs, tles, {"67006": 5.5}.get, default_std_mag=6.0)
    std = {s[1].lstrip("0"): s[3] for s in sats}
    assert std == {"42058": 6.0, "42065": 3.3, "67006": 5.5}   # no orbit, no entry


def test_passes_days_ahead_carry_the_highest_point(site, monkeypatch):
    """Dark sky and sunshine forced: two days of NOSS 3-8 (B) give several passes, each with a
    path for the chart and its highest point, in time order."""
    monkeypatch.setattr(fc, "_sun_alt", lambda s, t: np.full(len(t), -30.0))
    monkeypatch.setattr(fc, "_lit", lambda r, u, d: np.ones(len(r)))
    tles = idf.load_tles([DATA / "sample.tle"])
    favs = {}
    fav.add(favs, "42065", "NOSS 3-8 (B)")
    sats = fav.satellites(favs, tles, lambda k: 3.0)
    rows = fav.passes(sats, site, 1790640000.0, hours=48, min_alt=10.0)
    assert len(rows) >= 3
    assert [r["start"] for r in rows] == sorted(r["start"] for r in rows)
    for r in rows:
        assert r["max_alt"] >= r["alt"] > 10.0 and r["start"] <= r["peak"] <= r["end"]
        assert r["max_alt"] == max(p[1] for p in r["path"])
