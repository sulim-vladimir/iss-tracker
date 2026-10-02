"""Identify: naming the satellite a session followed, from its log and the catalogues."""

import csv
from pathlib import Path

import numpy as np
import pytest

from issctl import geometry as geo
from issctl import identify as idf
from issctl import predict as pr
from issctl.calib import ideal_calibration, jacobian
from issctl.config import load_config

DATA = Path(__file__).parent / "data"
FRAME = (1280, 960)


@pytest.fixture(scope="module")
def cfg():
    return load_config(Path(__file__).parent.parent / "config.example.toml")


@pytest.fixture(scope="module")
def site(cfg):
    return pr.Site(cfg)


@pytest.fixture(scope="module")
def tles():
    return idf.load_tles([DATA / "sample.tle"])


def test_catalogue_ids_may_contain_letters_and_are_not_duplicated(tmp_path, tles):
    assert [t[0] for t in tles] == ["NOSS 3-8 (A)", "NOSS 3-8 (B)", "STARLINK-36156"]
    odd = tmp_path / "odd.tle"
    odd.write_text("SECRET 1\n1 A0178U 26001A   26270.8 .0 00000-0 00000-0 0 01\n"
                   "2 A0178  63.4 250.9 0172559 358.8 1.1 13.41 06\n")
    both = idf.load_tles([DATA / "sample.tle", DATA / "sample.tle", odd])
    assert len(both) == 4 and both[-1][:2] == ("SECRET 1", "A0178")


def _visible(sat_line, site, above=20.0):
    """A stretch of unix times when this satellite is well up over the example site."""
    from skyfield.api import EarthSatellite, load

    ts = load.timescale()
    sat = EarthSatellite(*sat_line[2:], sat_line[0], ts)
    t0 = 1790640000.0                                    # 2026-09-29, near the orbits' epoch
    grid = t0 + np.arange(0, 2 * 86400, 30.0)
    alt = (sat - site.topos).at(ts.from_datetimes([idf._utc(x) for x in grid])).altaz()[0].degrees
    k = int(np.argmax(alt > above))
    assert alt[k] > above, "no pass in the test window"
    return sat, ts, grid[k] + np.arange(0, 80, 0.5)


def _write_session(path, sat, ts, times, site, guide=None, px=(700.0, 520.0)):
    """A session log of the tube following `sat`: counters from the sky, through an ideal mount.
    With a guide matrix, the object sits at `px` instead of the frame centre, and the counters
    are off by exactly that much."""
    alt, az, _ = (sat - site.topos).at(ts.from_datetimes([idf._utc(x) for x in times])).altaz()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a1", "a2", "alt", "az", "tgt1", "tgt2", "cmd1", "cmd2", "time_offset",
                    "cross1", "cross2", "source", "det_x", "det_y", "lit", "open"])
        for k, t in enumerate(times):
            ha, dec = geo.altaz_to_hadec(alt.degrees[k], az.degrees[k], site.lat)
            axes = np.array([float(v) for v in geo.hadec_to_axes(ha, dec, "east_looking")])
            det = ("", "")
            if guide is not None:
                centre = np.array([(FRAME[0] - 1) / 2, (FRAME[1] - 1) / 2])
                axes = axes - np.linalg.solve(jacobian(guide, axes[1]), centre - np.array(px))
                det = (f"{px[0]:.1f}", f"{px[1] + 0.1 * (k % 3):.1f}")    # a little jitter
            else:
                det = ("640.0", f"{480.0 + 0.1 * (k % 3):.1f}")
            w.writerow([f"{t:.3f}", f"{axes[0]:.5f}", f"{axes[1]:.5f}", 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        "guide", *det, 1, 1])


def test_names_the_satellite_and_not_its_formation_partner(tmp_path, tles, site):
    sat, ts, times = _visible(tles[1], site)
    log = tmp_path / "servo-test.csv"
    _write_session(log, sat, ts, times, site)
    t, v = idf.read_track(log, {}, FRAME)
    matches = idf.identify(t, v, tles, site)
    assert matches[0]["name"] == "NOSS 3-8 (B)" and matches[0]["median_deg"] < 0.05
    assert matches[1]["name"] == "NOSS 3-8 (A)" and matches[1]["median_deg"] > 0.3
    assert idf.describe(matches).startswith("identify: NOSS 3-8 (B)")


def test_the_object_pixel_is_carried_through_the_guide_matrix(tmp_path, tles, site, cfg):
    """The object held at a boresight 60 px off centre: ignoring that costs most of a degree."""
    guide = ideal_calibration(cfg["cameras"]["guide"], rotation_deg=110.0, dec_cal=40.0)
    guide["axis2_cal"] = 40.0
    sat, ts, times = _visible(tles[1], site)
    log = tmp_path / "servo-test.csv"
    _write_session(log, sat, ts, times, site, guide=guide)
    t, v = idf.read_track(log, {"cameras": {"guide": guide}}, FRAME)
    best = idf.identify(t, v, tles, site)[0]
    assert best["name"] == "NOSS 3-8 (B)" and best["median_deg"] < 0.05


def test_uses_the_alignment_recorded_with_the_session(tmp_path):
    log = tmp_path / "servo-x.csv"
    log.write_text("")
    then = {"alignment": {"model": {"rotvec_deg": [1, 2, 3], "d2": 0, "cone": 0,
                                    "rms_arcsec": 5, "n_points": 4}}, "cameras": {}}
    idf.write_session_state(log, then)
    used, note = idf.session_state(log, {"alignment": {}})
    assert note is None and used["alignment"]["model"]["rotvec_deg"] == [1, 2, 3]
    used, note = idf.session_state(tmp_path / "servo-y.csv", {"now": 1})
    assert used == {"now": 1} and "current" in note


def test_says_so_when_nothing_matches(tles, site):
    # a track nowhere near any of the three: straight up, at a time they are all elsewhere
    t = 1790640000.0 + np.arange(0, 60, 5.0)
    v = np.tile([np.cos(np.radians(site.lat)), 0.0, np.sin(np.radians(site.lat))], (len(t), 1))
    text = idf.describe(idf.identify(t, v, tles, site))
    assert "nothing in the catalogues" in text


def test_refuses_a_session_that_saw_nothing(tmp_path):
    log = tmp_path / "servo-empty.csv"
    log.write_text("t,a1,a2,alt,az,tgt1,tgt2,cmd1,cmd2,time_offset,cross1,cross2,source,det_x,"
                   "det_y,lit,open\n1,0,0,0,0,0,0,0,0,0,0,0,predict,nan,nan,1,1\n")
    with pytest.raises(ValueError, match="fewer than 3"):
        idf.read_track(log, {}, FRAME)


def test_names_it_live_while_following(tmp_path, tles, site):
    from issctl.model import PointingModel

    sat, ts, times = _visible(tles[1], site)
    log = tmp_path / "servo-live.csv"
    _write_session(log, sat, ts, times, site)
    rows = list(csv.DictReader(open(log)))
    state = {"alignment": {"model": dict(PointingModel().to_dict(), n_points=4)}}
    live = idf.LiveIdentifier(site, FRAME)
    live.sats = idf.build(tles)                       # no downloading in a test
    labels = []
    for r in rows[::4]:                               # a detection every 2 s
        t = float(r["t"])
        live.add(t, state, (float(r["a1"]), float(r["a2"])),
                 (float(r["det_x"]), float(r["det_y"])), "guide")
        labels.append(live.update(t))
    assert labels[0] == ""                            # too little to go on at first
    assert labels[-1] == "NOSS 3-8 (B)"
    assert live.candidates and len(live.candidates) <= live.CANDIDATES


def test_live_names_nothing_without_a_star_alignment(site):
    live = idf.LiveIdentifier(site, FRAME)
    live.add(0.0, {}, (10.0, 40.0), (640.0, 480.0), "guide")
    assert live.samples == []


def test_find_satellite_by_number_or_name(tles):
    assert [t[0] for t in idf.find_satellite("42065", tles)] == ["NOSS 3-8 (B)"]
    assert [t[0] for t in idf.find_satellite("042065", tles)] == ["NOSS 3-8 (B)"]
    assert [t[0] for t in idf.find_satellite("noss 3-8 (a)", tles)] == ["NOSS 3-8 (A)"]
    assert len(idf.find_satellite("noss", tles)) == 2          # ambiguous: both of the pair
    assert idf.find_satellite("hubble", tles) == []


def test_get_satellite_for_the_iss_and_for_anything_else(monkeypatch, cfg):
    from issctl import cli

    monkeypatch.setattr(idf, "refresh_catalogs", lambda *a, **k: [DATA / "sample.tle"])
    sat, name = cli.get_satellite(cfg, "NOSS 3-8 (B)")
    assert name == "NOSS 3-8 (B)" and sat.model.satnum == 42065
    with pytest.raises(cli.UnknownSatellite, match="matches 2 satellites"):
        cli.get_satellite(cfg, "noss")
    with pytest.raises(cli.UnknownSatellite, match="no satellite called"):
        cli.get_satellite(cfg, "hubble")
    monkeypatch.setattr(cli.pr, "get_tle", lambda cfg, offline=False: cli.pr.SIM_TLE)
    assert cli.get_satellite(cfg, "")[1] == "ISS" and cli.get_satellite(cfg, "25544")[1] == "ISS"


def test_pass_at_picks_the_pass_up_at_that_time():
    from issctl.cli import pass_at

    rows = [({"rise": 100.0, "set": 700.0}, {}, "visible"), ({"rise": 6000.0, "set": 6500.0}, {}, "visible")]
    assert pass_at(rows, 400.0) is rows[0] and pass_at(rows, 6100.0) is rows[1]
    assert pass_at(rows, 3000.0) is None


def test_history_lists_sessions_with_what_they_were(tmp_path, tles, site):
    """The latest sessions, newest first, with the pass name and a saved identification;
    simulation runs are left out."""
    sat, ts, times = _visible(tles[1], site)
    old = tmp_path / "servo-20260929-232446.csv"
    _write_session(old, sat, ts, times, site)
    new = tmp_path / "track-20261001-001841.csv"
    _write_session(new, sat, ts, times[:40], site)
    idf.write_session_state(new, {}, name="TERRA")
    (tmp_path / "track-20261001-002000-sim.csv").write_text(old.read_text())
    rows = idf.list_sessions(tmp_path)
    assert [r["file"] for r in rows] == [new.name, old.name]
    assert rows[0]["kind"] == "track" and rows[0]["name"] == "TERRA" and rows[0]["result"] is None
    assert rows[1]["duration"] == pytest.approx(times[-1] - times[0], abs=0.01)
    t, v = idf.read_track(old, {}, FRAME)
    matches = idf.identify(t, v, tles, site)
    idf.save_result(old, matches, idf.describe(matches))
    got = [r for r in idf.list_sessions(tmp_path) if r["file"] == old.name][0]["result"]
    assert got["best"] == "NOSS 3-8 (B)" and got["verdict"] == "sure"
