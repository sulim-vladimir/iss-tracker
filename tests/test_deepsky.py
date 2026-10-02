"""Goto by name: Messier, NGC, IC and common names from OpenNGC."""

from pathlib import Path

import pytest

from issctl import deepsky as ds
from issctl import predict as pr
from issctl.config import load_config

SAMPLE = Path(__file__).parent / "data" / "openngc-sample.csv"   # real rows, a handful


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    for name in ds.FILES:                  # both files present: nothing is downloaded
        (tmp_path / name).write_text(SAMPLE.read_text() if name == "NGC.csv" else SAMPLE.read_text().splitlines()[0] + "\n")
    monkeypatch.setattr(ds, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(ds, "_index", None)
    monkeypatch.setattr(ds, "resolve_online", lambda text: None)     # no network in tests
    return tmp_path



@pytest.mark.parametrize("text", ["M31", "m 31", "Messier 31", "M031", "NGC 224", "ngc0224",
                                  "NGC224", "Andromeda Galaxy", "andromeda"])
def test_the_same_object_by_any_of_its_names(catalogue, text):
    e = ds.lookup(text)
    assert e["name"] == "NGC0224" and e["messier"] == 31
    assert e["ra"] == pytest.approx(10.6848, abs=1e-3) and e["dec"] == pytest.approx(41.2691, abs=1e-3)


def test_messier_objects_outside_the_ngc_and_duplicates(catalogue):
    assert ds.lookup("M45")["name"] == "Mel022"                       # the Pleiades, addendum
    assert ds.lookup("Horsehead Nebula")["name"] == "B033"
    assert ds.lookup("M101")["name"] == "NGC5457"                     # the real entry, not M102's Dup
    assert ds.lookup("M102")["type"] == "Dup"                         # still found by its own name
    assert ds.lookup("IC 434")["name"] == "IC0434"
    assert ds.lookup("IC0067") is None                                # nonexistent: never a goto


def test_ambiguous_and_unknown_names(catalogue):
    with pytest.raises(ValueError, match="fits 5 objects"):
        ds.lookup("nebula")                                           # Orion, North America, Flame...
    assert ds.lookup("M99") is None
    assert ds.lookup("xyz") is None


def test_describe(catalogue):
    assert ds.describe(ds.lookup("m31")) == "M31 = NGC0224, Andromeda Galaxy - galaxy, mag 3.4"
    assert ds.describe(ds.lookup("ngc 7000")).startswith("NGC7000, North America Nebula - HII region")


def test_goto_targets_resolve_through_the_catalogue(catalogue):
    site = pr.Site(load_config(Path(__file__).parent.parent / "config.example.toml"))
    ha, dec, alt, az = pr.target_hadec("M42", site, 1.79e9)
    assert dec == pytest.approx(-5.39, abs=0.5)                       # apparent, so not exactly J2000
    assert pr.describe_target("orion nebula").startswith("M42 = NGC1976")
    with pytest.raises(ValueError, match="unknown target"):
        pr.target_hadec("no such thing", site, 1.79e9)
    with pytest.raises(ValueError, match="fits"):
        pr.target_hadec("nebula", site, 1.79e9)
