"""Coming up: bright satellites through the guide field or the visible sky."""

from pathlib import Path

import numpy as np
import pytest

from issctl import forecast as fc
from issctl import identify as idf
from issctl import predict as pr
from issctl.config import load_config

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def site():
    return pr.Site(load_config(Path(__file__).parent.parent / "config.example.toml"))


def test_standard_magnitude_is_1000_km_at_half_phase():
    assert fc.apparent_mag(5.0, 1000.0, np.pi / 2) == pytest.approx(5.0)
    assert fc.apparent_mag(5.0, 2000.0, np.pi / 2) == pytest.approx(5.0 + 5 * np.log10(2))
    # full phase (sun behind the observer) is brighter than half, a thin crescent much fainter
    assert fc.apparent_mag(5.0, 1000.0, 0.0) < 5.0 < fc.apparent_mag(5.0, 1000.0, 2.8)


def test_earth_shadow():
    u, d = np.array([[1.0, 0.0, 0.0]]), np.array([1.496e8])
    assert fc._lit(np.array([[7000.0, 0.0, 0.0]]), u, d)[0] == 1.0        # sunward side
    assert fc._lit(np.array([[-7000.0, 0.0, 0.0]]), u, d)[0] == 0.0       # straight behind
    assert fc._lit(np.array([[-7000.0, 7000.0, 0.0]]), u, d)[0] == 1.0    # behind, but clear of it


def test_lists_what_crosses_the_field_and_nothing_far_away(site, monkeypatch):
    """Dark sky and sunshine are forced: this is about the geometry. The field is laid along
    NOSS 3-8 (B)'s own path, so it and its formation partner a degree away cross it; the
    Starlink elsewhere does not."""
    from skyfield.api import load

    monkeypatch.setattr(fc, "_sun_alt", lambda s, t: np.full(len(t), -30.0))
    monkeypatch.setattr(fc, "_lit", lambda r, u, d: np.ones(len(r)))
    tles = idf.load_tles([DATA / "sample.tle"])
    sats = [(n, i, s, 4.0) for n, i, s in idf.build(tles)]
    noss_b = sats[1][2]
    ts = load.timescale()
    grid = 1790640000.0 + np.arange(0, 2 * 86400, 30.0)
    alt = (noss_b - site.topos).at(ts.from_datetimes([idf._utc(x) for x in grid])).altaz()[0].degrees
    t0 = float(grid[int(np.argmax(alt > 30))]) - 120.0
    t = t0 + np.arange(0.0, 600.0, fc.STEP_S)
    a, z, _ = (noss_b - site.topos).at(ts.from_datetimes([idf._utc(x) for x in t])).altaz()
    field = idf._altaz_unit(a.degrees, z.degrees)
    res = fc.forecast(sats, site, t0, minutes=10, field=field, max_mag=9.0)
    names = {r["name"] for r in res}
    assert {"NOSS 3-8 (B)", "NOSS 3-8 (A)"} <= names and "STARLINK-36156" not in names
    b = next(r for r in res if r["name"] == "NOSS 3-8 (B)")
    assert b["sep"] < 0.1 and b["start"] <= b["peak"] <= b["end"]


def test_field_follows_the_stars_when_tracking(site):
    t = 1790640000.0 + np.array([0.0, 3600.0])
    fixed = fc.field_track(site, 40.0, 350.0, t[0], t, follow_stars=False)
    stars = fc.field_track(site, 40.0, 350.0, t[0], t, follow_stars=True)
    assert np.allclose(fixed[0], fixed[1])
    moved = np.degrees(np.arccos(np.clip(np.dot(stars[0], stars[1]), -1, 1)))
    # an hour of sidereal turn is 15.04 deg of hour angle, times cos(dec) on the sky: small here,
    # because north at 40 deg up is only about 12 deg from the pole
    from issctl import geometry as geo
    dec = float(geo.altaz_to_hadec(40.0, 350.0, site.lat)[1])
    turn = 2 * np.degrees(np.arcsin(np.cos(np.radians(dec)) * np.sin(np.radians(15.041 / 2))))
    assert moved == pytest.approx(turn, abs=0.01)
    assert np.allclose(stars[0], fixed[0])


def test_a_pass_already_in_progress_is_found(site):
    """Picking a satellite that is up right now said "no pass": the search only knew passes by
    their rise, and this one had risen before the search began."""
    from skyfield.api import load

    tles = idf.load_tles([DATA / "sample.tle"])
    sat = idf.build(tles)[1][2]
    t0 = 1790640000.0
    first = pr.find_passes(sat, site, t0, 24)[0]
    for mid in (first["rise"] + 0.3 * (first["culm"] - first["rise"]),     # before the top
                first["culm"] + 0.5 * (first["set"] - first["culm"])):     # after it
        p = pr.find_passes(sat, site, mid, 24)[0]
        assert p["rise"] == pytest.approx(mid) and p["set"] == pytest.approx(first["set"], abs=1)
        assert p["rise"] <= p["culm"] <= p["set"]


def test_the_chart_draws_the_planned_pass_where_it_really_is(site):
    """The plan is kept as axis angles; drawn back through an ideal mount, a tripod 3.7 deg off
    the pole put the pass degrees away from the same satellite's Coming-up path."""
    from issctl.cli import sky_payload
    from issctl.config import load_config
    from issctl.mask import SkyMask
    from issctl.sim import misalignment

    cfg = load_config()
    sat = idf.build(idf.load_tles([DATA / "sample.tle"]))[1][2]
    p = pr.find_passes(sat, site, 1790640000.0, 24)[0]
    model = misalignment(site.lat, (3.0, -2.0), 20.0)
    traj, _ = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], model=model)
    track = sky_payload(cfg, SkyMask(), traj, site)["track"]
    t0 = traj.t_start
    for az, alt, lit, open_, dt in track[::40]:
        true_alt, true_az = pr.sat_hadec(sat, site, t0 + dt)[2:]
        assert abs(alt - true_alt) < 0.05 and abs((az - true_az + 180) % 360 - 180) < 0.1


def _mag_line(sid, flag, year, desig, name, mag):
    """One qs.mag record, columns as McCants writes them."""
    return f"{sid:05d} {flag} {year:02d} {desig:6s} {name:15s}{mag:4.1f}  1.0 0.0 0.0 .50"


def test_bright_orbits_are_fetched_for_what_the_catalogues_lack(tmp_path, monkeypatch):
    """qs.mag rates dead satellites and rocket bodies that CelesTrak's "active" group leaves out.
    refresh_bright asks for each launch year that has one, keeps only those, never refetches
    what a curated catalogue already has, and skips decayed and faint objects."""
    lines = (DATA / "sample.tle").read_text().splitlines()
    (tmp_path / "active.tle").write_text("\n".join(lines[6:9]) + "\n")       # the Starlink
    (tmp_path / "qs.mag").write_text("\n".join([
        "00001 d Desig...  Name.......... Mag.  Sz1 Sz2 Sz3 RCS Comments",
        _mag_line(42058, " ", 17, "11A", "NOSS 3-8 (A)", 4.0),
        _mag_line(42065, " ", 17, "11B", "NOSS 3-8 (B)", 9.5),     # too faint to bother
        _mag_line(23688, "d", 95, "56A", "STS 73", -1.5),          # came down long ago
        _mag_line(67006, " ", 25, "293D", "Starlink", 5.0),        # active.tle has it
    ]) + "\n")
    asked = []

    def fetch(url):
        asked.append(url)
        return (DATA / "sample.tle").read_text()       # a year's reply: more than was asked for

    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    path = fc.refresh_bright(tmp_path, log=lambda *a: None, fetch=fetch)
    assert asked == [fc.BRIGHT_URL.format(year=2017)]
    assert [t[1] for t in idf.load_tles([path])] == ["42058"]
    # the forecast, track and identify all read it, after the curated files
    assert idf.refresh_catalogs(tmp_path, offline=True)[-1] == path
    # fresh: nothing is asked again
    assert fc.refresh_bright(tmp_path, log=lambda *a: None, fetch=fetch) == path and len(asked) == 1


def test_bright_orbits_survive_a_failed_refresh(tmp_path, monkeypatch):
    import os
    lines = (DATA / "sample.tle").read_text().splitlines()
    (tmp_path / "qs.mag").write_text(_mag_line(42058, " ", 17, "11A", "NOSS 3-8 (A)", 4.0) + "\n")
    (tmp_path / "bright.tle").write_text("\n".join(lines[0:3]) + "\n")
    os.utime(tmp_path / "bright.tle", (0, 0))                      # stale
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)

    def offline(url):
        raise OSError("no network")

    path = fc.refresh_bright(tmp_path, log=lambda *a: None, fetch=offline)
    assert [t[1] for t in idf.load_tles([path])] == ["42058"]


def test_configured_magnitudes_bring_unrated_objects_in(tmp_path, monkeypatch, site):
    """qs.mag has no rating for NOSS 3-8, so the pair never appeared in Coming up - even on
    the night it was followed by hand. [forecast] std_mags fills that in."""
    (tmp_path / "qs.mag").write_text(_mag_line(67006, " ", 25, "293D", "Starlink", 5.0) + "\n")
    monkeypatch.setattr(idf, "refresh_catalogs", lambda *a, **k: [DATA / "sample.tle"])
    monkeypatch.setattr(fc, "refresh_mags", lambda *a, **k: tmp_path / "qs.mag")
    monkeypatch.setattr(fc, "refresh_bright", lambda *a, **k: None)
    plain = fc.Forecaster(site, catalog_dir=tmp_path, log=lambda *a: None)
    plain._ensure()
    assert [s[1] for s in plain.sats] == ["67006"]
    rated = fc.Forecaster(site, catalog_dir=tmp_path, log=lambda *a: None,
                          std_mags={"42058": 3.0, "042065": 3.0})
    rated._ensure()
    assert sorted(s[1] for s in rated.sats) == ["42058", "42065", "67006"]


def test_a_measured_magnitude_becomes_a_standard_one(site):
    """History's "Add to Coming up" rates a satellite from a Brightness measurement: the
    inverse of apparent_mag, at the distance and sun angle the orbit gives for that moment."""
    for rng, ph in ((1000.0, np.pi / 2), (2300.0, 1.9), (800.0, 0.6)):
        m = fc.apparent_mag(4.0, rng, ph)
        assert fc.standard_mag(m, rng, ph) == pytest.approx(4.0)
    sat = idf.build(idf.load_tles([DATA / "sample.tle"]))[1][2]
    t = pr.time_to_unix(sat.epoch) + 600.0
    rng, ph = fc.range_phase(sat, site, t)
    assert 300.0 < rng < 15000.0 and 0.0 <= ph <= np.pi


def test_ratings_added_later_are_used_by_the_next_forecast(tmp_path, monkeypatch, site):
    (tmp_path / "qs.mag").write_text(_mag_line(67006, " ", 25, "293D", "Starlink", 5.0) + "\n")
    monkeypatch.setattr(idf, "refresh_catalogs", lambda *a, **k: [DATA / "sample.tle"])
    monkeypatch.setattr(fc, "refresh_mags", lambda *a, **k: tmp_path / "qs.mag")
    monkeypatch.setattr(fc, "refresh_bright", lambda *a, **k: None)
    f = fc.Forecaster(site, catalog_dir=tmp_path, log=lambda *a: None)
    f._ensure()
    assert f.rating("67006") == 5.0 and f.rating("42065") is None
    f.add_rating("042065", 3.2)
    assert f.rating("42065") == 3.2 and f.sats is None          # rebuilt on the next run
    f._ensure()
    assert "42065" in [s[1] for s in f.sats]
