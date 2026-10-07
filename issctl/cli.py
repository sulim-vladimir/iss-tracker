"""issctl command line: passes, mount-test, console, track (real hardware or --sim)."""

import argparse
import json
import re
import collections
import datetime
import time
from pathlib import Path

import numpy as np

from . import align
from . import geometry as geo
from . import predict as pr
from .clock import Clock
from .config import ROOT, SIM_STATE_FILE, STATE_FILE, load_config, load_state, save_state
from .identify import write_session_state
from .mask import SkyMask


def fmt_t(u):
    return datetime.datetime.fromtimestamp(u).strftime("%a %d %H:%M:%S")


def describe(rep):
    lim = ",".join(rep["limited_by"]) or "-"
    return (f"side={rep['side']:12s} track {rep['tracked_s']:4.0f}/{rep['visible_s']:4.0f}s "
            f"lit {rep['sunlit_s']:4.0f}s blocked {rep['blocked_s']:4.0f}s usable {rep['useful_s']:4.0f}s "
            f"peak {rep['max_rate'][0]:.2f}/{rep['max_rate'][1]:.2f} deg/s "
            f"axis1 {rep['axis1_range'][0]:+.0f}..{rep['axis1_range'][1]:+.0f} limits: {lim}")


compass = geo.compass


def altaz_at(sat, site, times):
    _, _, alt, az = pr.sat_hadec(sat, site, np.atleast_1d(np.asarray(times, dtype=float)))
    return np.atleast_1d(alt), np.atleast_1d(az)


def sky_path(sat, site, p):
    """Where the pass sits in the sky: rise -> culmination -> set."""
    alt, az = altaz_at(sat, site, [p["rise"], p["culm"], p["set"]])
    return (f"rises az {az[0]:3.0f} {compass(az[0]):3s} -> alt {alt[1]:2.0f} az {az[1]:3.0f} "
            f"{compass(az[1]):3s} -> sets az {az[2]:3.0f} {compass(az[2]):3s}")


def describe_windows(sat, site, rep, t_ref):
    out = []
    for a, b in rep["windows"]:
        alt, az = altaz_at(sat, site, [a, b])
        out.append(f"{a - t_ref:+.0f}..{b - t_ref:+.0f}s (alt {alt[0]:.0f}->{alt[1]:.0f}, "
                   f"az {az[0]:.0f} {compass(az[0])}->{az[1]:.0f} {compass(az[1])})")
    return ", ".join(out)


ISS_NAMES = ("", "iss", "25544", "iss (zarya)", "zarya")


class UnknownSatellite(ValueError):
    """No catalogue entry, or several, for what was asked."""


def get_satellite(cfg, query=None, offline=False, log=print):
    """(skyfield satellite, name) for "ISS" or any catalogue satellite, by name or NORAD number.

    The ISS keeps its own TLE ([tle] url, refreshed every max_age_hours) - the one the tracker has
    always used. Anything else comes from the satellite catalogues (identify.py): CelesTrak's
    active and visual groups and McCants' classified orbits."""
    from . import identify as idf
    if str(query or "").strip().lower() in ISS_NAMES:
        return pr.make_satellite(pr.get_tle(cfg, offline=offline)), "ISS"
    tles = idf.load_tles(idf.refresh_catalogs(log=log, offline=offline))
    found = idf.find_satellite(query, tles)
    if not found:
        raise UnknownSatellite(f"no satellite called '{query}' in the catalogues")
    if len(found) > 1:
        names = ", ".join(f"{n} ({i})" for n, i, _, _ in found[:6])
        raise UnknownSatellite(f"'{query}' matches {len(found)} satellites: {names}"
                         + (" ..." if len(found) > 6 else "") + " - give the number")
    name, sid, l1, l2 = found[0]
    return pr.make_satellite((name, l1, l2)), name


def pass_at(rows, t):
    """The pass from list_passes that is up at time t (a Coming-up entry), or None."""
    for row in rows:
        if row[0]["rise"] - 60 <= t <= row[0]["set"] + 60:
            return row
    return None


def describe_shadow(rep, t_ref=None):
    if not rep["shadow"]:
        return "sunlit throughout" if rep["sunlit_s"] > 0 else "in shadow throughout"
    return ", ".join(f"{what} shadow at " + (f"{t - t_ref:+.0f}s" if t_ref else fmt_t(t))
                     for t, what in rep["shadow"])


def list_passes(cfg, sat, site, t0, hours, mask=None, model=None):
    rows = []
    for p in pr.find_passes(sat, site, t0, hours):
        _, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask,
                              model=model)
        _, sun_alt = pr.sun_state(sat, site, p["culm"])
        if sun_alt >= -4:
            vis = "day"
        elif rep["sunlit_s"] <= 0:
            vis = "shadow"
        elif rep["useful_s"] <= 0:
            vis = "blocked" if rep["blocked_s"] > 0 else "shadow"
        elif rep["useful_s"] < 20:
            vis = "brief"
        else:
            vis = "visible"
        rows.append((p, rep, vis))
    return rows


def cmd_passes(args, cfg):
    site = pr.Site(cfg)
    sat, name = get_satellite(cfg, args.sat, offline=args.offline)
    mask = SkyMask.from_config(cfg)
    now = time.time()
    model = align.current_model(load_state())
    print(f"{name} | TLE age {pr.tle_age_days(sat, now):.1f} days | sky: {mask.describe()} | "
          f"mount: {model.describe() if model else 'no star alignment - assumed polar aligned'}")
    for i, (p, rep, vis) in enumerate(list_passes(cfg, sat, site, now, args.hours, mask, model)):
        print(f"{i:2d} {fmt_t(p['rise'])}  max {p['max_alt']:4.1f}  {vis:7s} {describe(rep)}")
        print(f"   {sky_path(sat, site, p)}"
              + (f" | {describe_shadow(rep)}" if rep["shadow"] else ""))
        if rep["windows"]:
            print(f"   usable: {describe_windows(sat, site, rep, rep['track_start'])}")


def cmd_favorites(args, cfg):
    """Favourites from the terminal: list, add, remove, and their visible passes."""
    from . import favorites as fav
    from . import identify as idf
    from .forecast import Forecaster
    site = pr.Site(cfg)
    fcfg = cfg.get("forecast", {})
    favs = fav.load()
    forecaster = Forecaster(site, std_mags={**(fcfg.get("std_mags") or {}), **fav.ratings(favs)})
    tles = (idf.load_tles(idf.refresh_catalogs(offline=True)) if args.offline
            else forecaster.catalogue())
    if args.add:
        q = "25544" if args.add.strip().lower() in ISS_NAMES else args.add
        found = idf.find_satellite(q, tles)
        if len(found) != 1:
            raise SystemExit(f"'{args.add}': " + ("not in the catalogues" if not found else
                             ", ".join(f"{n} ({i})" for n, i, _, _ in found[:8])
                             + " - give the number"))
        name, sid, _, _ = found[0]
        unrated = forecaster.rating(sid) is None
        fav.add(favs, sid, name, float(fcfg.get("default_std_mag", 5.0)) if unrated else None,
                "default" if unrated else None)
        fav.save(favs)
        print(f"added {name} ({sid})")
    if args.remove:
        e = fav.remove(favs, args.remove)
        if e is None:
            raise SystemExit(f"'{args.remove}' is not a favourite")
        fav.save(favs)
        print(f"removed {e['name']} ({e['id']})")
    seen = idf.observed(tles=tles)
    for k, e in favs.items():
        last = seen.get(k)
        print(f"{e['name']:24s} {k:>6s}  "
              + (f"seen {datetime.datetime.fromtimestamp(last):%d %b}" if last else "not seen yet")
              + (f"  std mag {e['std_mag']:.1f} ({e['mag_from']})" if e.get("std_mag") is not None else ""))
    if not favs:
        print("no favourites yet")
        return
    forecaster.set_ratings({**(fcfg.get("std_mags") or {}), **fav.ratings(favs)})
    sats = fav.satellites(favs, tles, forecaster.rating, float(fcfg.get("default_std_mag", 5.0)))
    rows = fav.passes(sats, site, time.time(), args.hours, mask=SkyMask.from_config(cfg),
                      min_alt=cfg["site"]["min_altitude"])
    print(f"\nvisible passes, next {args.hours:g} h:" if rows else
          f"\nno visible passes in the next {args.hours:g} h")
    for r in rows:
        print(f"  {fmt_t(r['start'])} - {datetime.datetime.fromtimestamp(r['end']):%H:%M}  "
              f"mag {r['mag']:4.1f}  max alt {r['max_alt']:3.0f}  {r['name']} ({r['id']})")


def guide_radius_deg(cfg):
    """Half the guide field's short side: the circle certainly in view, whatever the camera's
    rotation. The sky chart draws it round the pointing; "Through the guide field" counts it."""
    from .solve import field_deg
    return 0.5 * min(field_deg(cfg["cameras"]["guide"]))


def sky_payload(cfg, mask, traj=None, site=None):
    """Static data for the sky chart: the pass track, the mask and the horizon limit."""
    out = {"mask": {"openings": mask.openings, "blockers": mask.blockers},
           "min_alt": cfg["site"]["min_altitude"], "track": [],
           "guide_radius_deg": round(guide_radius_deg(cfg), 2)}
    if traj is not None and site is not None:
        step = max(1, len(traj.t) // 400)
        if getattr(traj, "az", None) is not None:
            alt, az = traj.alt[::step], traj.az[::step]      # the sky path it was planned from
        else:
            a1, a2 = traj.a1[::step], traj.a2[::step]
            model = getattr(traj, "model", None)
            ha, dec = geo.axes_to_hadec(a1, a2) if model is None else model.axes_to_hadec(a1, a2)
            alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
        lit, open_sky = traj.lit[::step], traj.open_sky[::step]
        t = traj.t[::step]
        out["track"] = [[round(float(z), 2), round(float(a), 2), round(float(l), 2), int(o),
                         round(float(tt - traj.t_start), 1)]
                        for z, a, l, o, tt in zip(az, alt, lit, open_sky, t) if a > -5]
    return out


def apply_saved_settings(cams, state, log=print):
    """Re-apply the exposure/gain last used for each camera, so a restart looks the same.

    A camera that has just re-enumerated on USB answers the first control with "General error";
    that used to kill the console on start-up, with the mount left stopped. Skip it and say so."""
    for name, cam in cams.items():
        saved = state.get("camera_settings", {}).get(name, {})
        try:
            if "exposure_ms" in saved:
                cam.set_exposure(saved["exposure_ms"])
            if "gain" in saved:
                cam.set_gain(saved["gain"])
        except Exception as e:
            log(f"{name}: could not restore its saved exposure/gain ({e}) - set them by hand")


def remember_position(state, state_path, mount):
    """Keep the live axis angles on disk so a restart does not need re-homing."""
    state["position"] = [float(v) for v in mount.position()]
    state["position_at"] = time.time()
    state["index"] = mount.index.tolist()
    save_state(state, state_path)


def restore_position(state, mount, log=print):
    pos = state.get("position")
    if not pos:
        return
    mount.restore_position(pos)
    when = state.get("position_at")
    ago = f", saved {datetime.datetime.fromtimestamp(when):%H:%M:%S}" if when else ""
    log(f"position restored: axis1 {pos[0]:+.3f} axis2 {pos[1]:+.3f}{ago} "
        f"- re-home if the mount was moved by hand")


def remember_settings(state, state_path, name, cam):
    state.setdefault("camera_settings", {})[name] = {"exposure_ms": cam.exposure_ms, "gain": cam.gain}
    save_state(state, state_path)


def make_controls(cams, recorder=None, mount_action=None, mount_state=None, estop=None, stopped=None,
                  on_select=None, sky=None, pointing=None, target=None, pass_info=None,
                  on_settings=None, picked=None):
    """Callbacks the preview page uses for exposure, gain and recording."""
    picked = {} if picked is None else picked   # last hand-picked pixel per camera

    def state():
        out = {"cams": {}, "record": recorder.state() if recorder else None,
               "time": f"{datetime.datetime.now():%H:%M:%S}",
               "stopped": bool(stopped()) if stopped else False,
               "pass": pass_info() if pass_info else None}
        for key, fn in (("pointing", pointing), ("target", target)):
            value = fn() if fn else None          # None whenever there is nothing to show
            out[key] = list(value) if value is not None else None
        for n, c in cams.items():
            _, det, _ = c.latest()
            out["cams"][n] = {"fps": c.fps, "exposure_ms": c.exposure_ms, "gain": c.gain,
                              "exposure_unit": getattr(c, "exposure_unit", "ms"),
                              "det": [round(det.x, 1), round(det.y, 1)] if det else None,
                              "manual": c.manual}
        return out

    def exposure(name, ms=None, factor=None):
        cam = cams.get(name)
        if cam:
            cam.set_exposure(float(ms) if ms else cam.exposure_ms * float(factor))
            if on_settings:
                on_settings(name, cam)

    def gain(name, value=None, delta=None):
        cam = cams.get(name)
        if cam:
            cam.set_gain(int(float(value)) if value else cam.gain + int(float(delta)))
            if on_settings:
                on_settings(name, cam)

    def record(on):
        if recorder:
            recorder.set_enabled(on)

    def select(name, fx=None, fy=None, clear=None):
        """Click on the image: track that object rather than whichever is brightest."""
        cam = cams.get(name)
        if not cam:
            return
        if clear or fx is None or fy is None:
            (on_select or (lambda *a: None))(name, None, None)
            cam.clear_selection()
            picked.pop(name, None)
        else:
            x, y = float(fx) * cam.width, float(fy) * cam.height
            # Kept apart from cam.gate: the gate is a DETECTION gate and re-centres itself on
            # whatever blob turns up inside it, which is the last thing you want from a pixel you
            # chose by eye and are about to turn into a boresight.
            picked[name] = (x, y)
            if on_select:
                on_select(name, x, y)
            else:
                cam.select(x, y)

    return {"state": state, "exposure": exposure, "gain": gain,
            "record": record if recorder and recorder.available else None,
            "mount_action": mount_action, "mount_state": mount_state, "estop": estop,
            "select": select, "sky": sky}


def start_preview(cams, state, port, status=None, controls=None):
    from .preview import Preview
    try:
        Preview(cams, state, port, status=status, controls=controls).start()
        print(f"preview on http://localhost:{port}/")
    except OSError as e:
        print(f"preview unavailable on port {port}: {e} (use --port)")


# ---------------------------------------------------------------- hardware

def open_mount(cfg, state, clock):
    from .mount import SerialMount
    return SerialMount(cfg, state, clock)


def open_cameras(cfg, clock):
    from .camera import AsiCamera, V4l2Camera
    cams = {}
    for name in ("guide", "main"):
        cam_cfg = cfg["cameras"][name]
        try:
            if cam_cfg.get("driver", "asi") == "v4l2":
                cams[name] = V4l2Camera(name, cam_cfg, clock).start()
            else:
                cams[name] = AsiCamera(name, cam_cfg, clock, cfg["cameras"]["sdk_lib"]).start()
        except Exception as e:
            print(f"{name} camera unavailable: {e}")
    return cams


def cmd_mount_test(args, cfg):
    clock = Clock()
    mount = open_mount(cfg, load_state(), clock)
    mount.enable(True)
    try:
        for axis in (0, 1):
            for sign in (1, -1):
                r = np.zeros(2)
                r[axis] = sign * args.rate
                p0 = mount.query()
                t_end = time.monotonic() + args.seconds
                while time.monotonic() < t_end:
                    mount.set_rates(*r)
                    time.sleep(0.05)
                mount.stop()
                time.sleep(1.0)
                p1 = mount.query()
                print(f"axis{axis + 1} {r[axis]:+.2f} deg/s for {args.seconds}s: moved {p1 - p0} deg "
                      f"(expected ~{r[axis] * args.seconds:+.2f})")
    finally:
        mount.close()


def set_config_value(path, section, key, value):
    """Rewrite one key inside one [section] of a TOML file, leaving comments alone."""
    import re

    text = Path(path).read_text()
    start = text.index(f"[{section}]")
    end = text.find("\n[", start + 1)
    end = len(text) if end < 0 else end
    block = text[start:end]
    new_block, n = re.subn(rf"(?m)^(\s*{re.escape(key)}\s*=\s*)([^#\n]+)", rf"\g<1>{value} ", block)
    if not n:
        raise KeyError(f"{key} not found in [{section}] of {path}")
    Path(path).write_text(text[:start] + new_block + text[end:])


def cmd_axis_scale(args, cfg):
    """Move one axis a known amount, compare with the angle you measure, fix gear_ratio.

    This is what catches a wrong motor step angle, microstep setting or pulley ratio: symptoms are
    gotos landing short (or long) by a constant factor.
    """
    from .predict import steps_per_deg

    axis_key = f"axis{args.axis}"
    clock = Clock()
    mount = open_mount(cfg, load_state(), clock)
    try:
        mount.enable(True)
        start = mount.query()
        print(f"moving {axis_key} by {args.move:+.1f} deg at {args.rate} deg/s - watch the mount")
        target = start.copy()
        target[args.axis - 1] += args.move
        mount.move_to(target)
        end = mount.position()
        print(f"firmware counted {end[args.axis - 1] - start[args.axis - 1]:+.3f} deg "
              f"({steps_per_deg(cfg['mount'][axis_key]):.0f} steps/deg configured)")
        measured = args.measured
        if measured is None:
            try:
                measured = float(input("measured physical angle, deg (blank to skip): ") or "nan")
            except (EOFError, ValueError):
                measured = float("nan")
        if measured != measured or measured == 0:
            print("no measurement, nothing changed")
            return
        old = cfg["mount"][axis_key]["gear_ratio"]
        new = old * args.move / measured
        print(f"moved {measured:.2f} deg instead of {args.move:.2f}: "
              f"{axis_key} gear_ratio {old:g} -> {new:.4f} "
              f"({steps_per_deg(dict(cfg['mount'][axis_key], gear_ratio=new)):.0f} steps/deg)")
        print("check the obvious causes too: motor step angle (1.8 vs 0.9 deg), microstep jumpers, "
              "pulley teeth - a clean factor of 2 or 3 usually means one of those.")
        if args.write:
            set_config_value(ROOT / "config.toml", f"mount.{axis_key}", "gear_ratio", f"{new:.4f}")
            print(f"config.toml updated - re-home and recalibrate the cameras")
    finally:
        mount.close()


# ---------------------------------------------------------------- console

def cmd_console(args, cfg):
    import curses
    import threading

    from .calib import (axis1_plausible, axis1_stretch, calibrate_cameras, centring_move,
                        image_jog_rates, measure)
    from .mount import SIDEREAL_DEG_S, SimMount
    from .solve import solve_camera

    state_path = SIM_STATE_FILE if args.sim else None
    state = load_state(state_path)
    clock = Clock()
    site = pr.Site(cfg)
    if args.sim:
        from .camera import SimCamera
        from .sim import CalibWorld
        # start away from the pole: at axis2 = 90 the axis1 measurement degenerates (cos dec -> 0)
        mount = SimMount(cfg, state, clock, start=[20.0, 40.0],
                         backlash=[0.0, getattr(args, "sim_backlash", 0.0)])
        mount.query()
        # a fixed "distant light" a little off the boresight, to exercise jogging and calibration,
        # on a tripod turned 30 deg off north that only the star calibration can see
        from .sim import misalignment
        world = CalibWorld(cfg, mount, model=misalignment(site.lat, (1.0, -0.6), 30.0))
        cams = {n: SimCamera(n, cfg["cameras"][n], clock, world).start() for n in ("guide", "main")}
    else:
        world = None
        mount, cams = open_mount(cfg, state, clock), open_cameras(cfg, clock)
        restore_position(state, mount)
    from .solve import make_solver
    solver = make_solver(cams["guide"], site, world) if "guide" in cams else None
    last_solve = {"sol": None, "axes": None}
    apply_saved_settings(cams, state)
    for n in state.get("marks_hidden", []):      # boresight/centre crosses hidden on the page
        if n in cams:
            cams[n].show_marks = False
    def status_lines(name):
        return []   # axis angles and the clock live in the mount panel and the top bar

    speeds = [0.004, 0.02, 0.1, 0.5, 2.0]

    class Messages(dict):
        """Keeps every message, not just the newest, so the browser can show a log.

        ui["msg"] is assigned all over the console's handlers rather than funnelled through
        say(), so the record is taken at the point of assignment - catching them at the call
        sites would mean finding all of them, and missing the next one somebody adds.
        """

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.log = collections.deque(maxlen=300)

        def __setitem__(self, key, value):
            if key == "msg" and value and value != self.get("msg"):
                stamped = value[:2].isdigit() and value[2:3] == ":"
                self.log.append(value if stamped
                                else f"{datetime.datetime.now():%H:%M:%S}  {value}")
            super().__setitem__(key, value)

    ui = Messages({"jog": np.zeros(2), "speed": 2, "tracking": False, "busy": False,
                   "quit": False, "msg": "", "frame": "axes", "abort": threading.Event(),
                   "mode": "console", "motors": True, "jog_rates": np.zeros(2)})
    picked = {}     # the pixel you last clicked in each image, kept for the hand-set boresight

    def jog_frames():
        return ["axes"] + [n for n in cams if n in state.get("cameras", {})]

    def refresh_jog_rates():
        """Work out the jog rates ONCE per key press, like a hand controller.

        Recomputing them every cycle made the mount oscillate: near the pole the image-frame
        mapping switches to raw axes, and as the mount drifted across that boundary the commanded
        direction flipped back and forth.
        """
        j = ui["jog"]
        if not j.any():
            ui["jog_rates"] = np.zeros(2)
            return
        cal = state.get("cameras", {}).get(ui["frame"])
        rates = None if cal is None else image_jog_rates(cal, mount.position()[1], j, speeds[ui["speed"]])
        ui["jog_raw"] = rates is None
        ui["jog_rates"] = j * speeds[ui["speed"]] if rates is None else rates

    fails = {"t": -1e9}

    def keepalive():
        saved = (mount.position().copy(), time.monotonic())
        while not ui["quit"]:
            # One bad cycle must not end the loop: if this thread dies, nothing feeds the
            # firmware's watchdog and the mount stops 0.5 s later, silently, until a restart.
            try:
                # keep the stored position fresh, so a crash or power cut loses at most a few seconds
                now_pos = mount.position()
                if time.monotonic() - saved[1] > 5.0 and np.any(np.abs(now_pos - saved[0]) > 0.01):
                    saved = (now_pos.copy(), time.monotonic())
                    try:
                        persist()
                    except Exception:
                        pass
                if ui["mode"] == "track":
                    pass  # the tracker owns the mount while a pass is running
                elif aborted():
                    mount.set_rates(0.0, 0.0)
                elif not ui["busy"]:
                    r = ui["jog_rates"]
                    if ui["tracking"]:
                        r = r + sidereal()
                    mount.set_rates(*r)
            except Exception as e:
                if time.monotonic() - fails["t"] > 10.0:
                    say(f"mount keepalive: {type(e).__name__}: {e} - retrying")
                    fails["t"] = time.monotonic()
                time.sleep(0.2)
            time.sleep(0.05)

    def persist():
        remember_position(state, state_path, mount)

    rates_cache = {"t": -1e9, "r": np.array([SIDEREAL_DEG_S, 0.0])}

    def sidereal():
        """Tracking rates that hold the stars still: both axes, through the star alignment when
        there is one. Refreshed every few seconds - they change slowly across the sky."""
        now = time.monotonic()
        if now - rates_cache["t"] > 5.0:
            try:
                rates_cache["r"] = align.tracking_rates(state, mount.position(), SIDEREAL_DEG_S)
            except Exception:
                rates_cache["r"] = np.array([SIDEREAL_DEG_S, 0.0])
            rates_cache["t"] = now
        return rates_cache["r"]

    def prompt(scr, text):
        scr.nodelay(False)
        curses.echo()
        curses.curs_set(1)
        h, _ = scr.getmaxyx()
        scr.move(h - 1, 0)
        scr.clrtoeol()
        scr.addstr(h - 1, 0, text)
        s = scr.getstr(h - 1, len(text), 40).decode().strip()
        curses.noecho()
        curses.curs_set(0)
        scr.nodelay(True)
        return s

    def aborted():
        return ui["abort"].is_set()

    def emergency_stop():
        """Cut motion now: drop jog and tracking, abort any running goto/calibration, halt motors."""
        ui["abort"].set()
        if session["tracker"]:
            session["tracker"].stop_requested = True
        ui["jog"][:] = 0
        ui["tracking"] = False
        try:
            mount.estop()
            ui["msg"] = "EMERGENCY STOP - motors halted; re-sync before trusting positions"
        except Exception as e:
            ui["msg"] = f"EMERGENCY STOP failed: {e}"

    def say(text):
        ui["msg"] = f"{datetime.datetime.now():%H:%M:%S}  {text}"

    def busy(fn):
        ui["abort"].clear()
        ui["busy"] = True
        try:
            fn()
        except Exception as e:
            say(f"error: {e}")
        finally:
            ui["busy"] = False

    def goto(name):
        first = True
        what = None if name.strip().lower() in (*pr.STARS, *pr.BODIES) else pr.describe_target(name)
        if what:
            say(f"goto {what}")
        for _ in range(2):  # second pass corrects for sky motion during the slew
            ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
            cur = mount.position()
            best, options = geo.choose_pose(ha, dec, cfg["mount"], current=cur,
                                            to_axes=align.hadec_to_axes_fn(state))
            if best is None:
                say(f"{name} is not reachable: " + ", ".join(
                    f"{o['side']} needs axis1 {o['axes'][0]:+.0f} axis2 {o['axes'][1]:+.0f}"
                    for o in options))
                return
            if first and abs(best["axes"][1] - cur[1]) > 90:
                say(f"{name}: meridian flip, axis2 {cur[1]:+.0f} -> {best['axes'][1]:+.0f} "
                    f"(tube swings past the pole - check clearance)")
                time.sleep(2.0)
            first = False
            mount.move_to(best["axes"], track_rate=sidereal(), abort=aborted,
                          approach=state.get("backlash_deg"))
            if aborted():
                say("goto aborted")
                return
        ui["tracking"] = True
        say(f"at {name} (alt {alt:.1f}), {best['side']}, tracking on")

    def goto_altaz(alt, az):
        """Slew to a point picked on the sky chart and hold it there: an alt/az, not a star, so
        sidereal tracking goes off - the place to wait for a satellite to come through."""
        if alt < 0:
            say(f"alt {alt:.1f}: below the horizon")
            return
        ui["tracking"] = False
        for _ in range(2):  # the second pass corrects for the sky turning during the slew
            ha, dec = geo.altaz_to_hadec(alt, az, site.lat)
            best, options = geo.choose_pose(float(ha), float(dec), cfg["mount"],
                                            current=mount.position(),
                                            to_axes=align.hadec_to_axes_fn(state))
            if best is None:
                say(f"alt {alt:.0f} az {az:.0f} is not reachable: " + ", ".join(
                    f"{o['side']} needs axis1 {o['axes'][0]:+.0f} axis2 {o['axes'][1]:+.0f}"
                    for o in options))
                return
            mount.move_to(best["axes"], abort=aborted, approach=state.get("backlash_deg"))
            if aborted():
                say("goto aborted")
                return
        say(f"at alt {alt:.1f} az {az:.1f} ({geo.compass(az)}), holding still - sidereal off"
            + ("" if alt >= site.min_altitude else f" (below min_altitude {site.min_altitude:g})"))

    def do_sync(name):
        ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
        d = align.sync_to(state, mount, ha, dec)
        persist()
        ui["msg"] = f"synced on {name}: correction {d.round(3)} deg"

    # ---- plate solving ----
    def solve_here():
        """Solve the guide frame now. Returns (solution, counters at exposure, boresight pixel)."""
        if solver is None:
            raise RuntimeError("no guide camera to solve")
        sol = solve_camera(cams["guide"], solver, log=say)
        axes = np.asarray(mount.position_at(sol.t), dtype=float)
        last_solve.update(sol=sol, axes=axes)
        return sol, axes, sol.centre()    # the alignment follows the guide centre (align.py)

    def do_solve():
        sol, axes, b = solve_here()
        ha, dec = sol.hadec(b)
        alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
        thinks = align.pointing_hadec(state, axes)
        off = float(align.angle_arcsec(align.sky_unit(*thinks), align.sky_unit(ha, dec)) / 3600)
        named = [label for _, label, mag in sol.catalog() if not label.startswith("mag")][:4]
        say(f"guide centre at alt {float(alt):.1f} az {float(az):.1f} ({geo.compass(az)}); the "
            f"mount thinks it is {off:.2f} deg from there"
            + (f"; in view: {', '.join(named)}" if named else ""))

    def do_brightness(fx, fy):
        """How bright is what was clicked in the guide image: a plate solve of a fresh frame
        gives both the catalogue star there and the frame's own zero point - which also puts a
        magnitude on things no catalogue has, a satellite included."""
        from .solve import brightness
        g = cams["guide"]
        x, y = float(fx) * g.width, float(fy) * g.height
        sol, _, _ = solve_here()
        b = brightness(sol, (x, y))
        parts = []
        if b["mag"] is not None and not b["catalog_label"]:
            # kept for "Add to Coming up": a satellite measured during a session rates it
            ui.setdefault("brightness_log", []).append({"t": float(sol.t), "mag": float(b["mag"])})
            del ui["brightness_log"][:-50]
        if b["mag"] is not None:
            parts.append(f"measured magnitude {b['mag']:.1f} ± {b['mag_err']:.1f} "
                         f"(against {b['n_ref']} catalogue stars in the frame)")
        elif b["px"] is None:
            parts.append("nothing detected within 12 px of the click")
        else:
            parts.append("too few catalogue stars matched to measure a magnitude")
        if b["catalog_label"]:
            if b["catalog_mag"] is None:                 # a named bright star, no magnitude kept
                parts.append(f"catalogue: {b['catalog_label']}")
            elif b["catalog_label"].startswith("mag"):
                parts.append(f"catalogue (Tycho-2) magnitude {b['catalog_mag']:.1f}")
            else:
                parts.append(f"catalogue: {b['catalog_label']}, magnitude {b['catalog_mag']:.1f}")
        elif b["px"] is not None:
            parts.append("no catalogue star there - a satellite, or fainter than Tycho-2")
        where = f"({b['px'][0]:.0f}, {b['px'][1]:.0f})" if b["px"] else f"({x:.0f}, {y:.0f})"
        say(f"brightness at {where}: " + " · ".join(parts))

    def do_solve_sync():
        sol, axes, b = solve_here()
        d = align.sync_to(state, mount, *sol.hadec(b))
        persist()
        say(f"synced on the stars: counters corrected by {d.round(3)} deg"
            + ("" if align.current_model(state) else
               " (no alignment yet - 'calibrate on stars' next, so goto and passes know how "
               "the tripod stands)"))

    def do_align_add():
        sol, axes, b = solve_here()
        try:
            off = align.add_point(state, axes, *sol.hadec(b), sol.t)
        except ValueError as e:
            say(str(e))
            return
        model, shift = align.refit(state, mount, state.get("cameras"))
        persist()
        say((f"star added ({off:.2f} deg from the model's prediction). " if off is not None
             else "star added. ") + align.describe(state)
            + (f" - Dec index {shift:+.2f} deg moved into the counters" if shift else ""))

    def do_starcal():
        say("calibrating the guide camera on the stars...")
        warnings = []
        cal = align.calibrate_on_stars(mount, cams["guide"], solver, state,
                                       track_rate=sidereal() if ui["tracking"] else None,
                                       log=say, abort=aborted, warnings=warnings)
        state.setdefault("cameras", {})["guide"] = cal
        state["calibrated_at"] = time.time()
        state["calibration_warnings"] = warnings
        persist()
        say("star calibration done" + (f" - {len(warnings)} warning(s)" if warnings else "")
            + ". Next: goto a bright star, 'centre by solve', then 'boresight on star'.")

    def do_star_boresight():
        cal = state.get("cameras", {})
        if not cal.get("main") or not cal.get("guide"):
            say("need both camera matrices first: 'calibrate on stars' for the guide, then "
                "'calibrate on target' in main with a bright star centred")
            return
        px_main = measure(cams["main"])
        if px_main is None:
            say("no star detected in the main camera - centre a bright one there first")
            return
        sol, _, _ = solve_here()
        try:
            bore, label, miss, carried = align.boresight_on_star(sol, cal["main"], cal["guide"],
                                                                 px_main)
        except ValueError as e:
            say(str(e))
            return
        old = np.asarray(cal["guide"]["boresight"], dtype=float)
        cal["guide"]["boresight"] = [float(bore[0]), float(bore[1])]
        persist()
        say(f"boresight on {label}: moved {np.hypot(*(bore - old)):.0f} px to "
            f"{bore.round(1)} (star was {miss:.2f} deg from where the old boresight put it"
            + (f", {carried:.0f} px carried through the matrices" if carried > 1 else "") + ")")

    def do_spiral():
        """Walk a spiral round the current pointing until the main camera sees a star."""
        from .search import Search
        if "main" not in cams:
            say("no main camera")
            return
        radius = float(cfg["cameras"]["main"].get("search_radius_deg", 0.5))
        ui["spiral"] = True
        try:
            hit = Search(mount, cams["main"], track_rate=sidereal() if ui["tracking"] else None,
                         log=say, abort=aborted, radius_deg=radius,
                         slack_deg=state.get("backlash_deg"), auto_stop=False,
                         dwell_s=float(cfg["cameras"]["main"].get("search_dwell_s", 1.5))).run()
        finally:
            ui["spiral"] = False
        if hit is not None:
            say(f"spiral search: stopped at stop {hit['stop']} "
                f"({hit['offset_arcmin'][0]:+.1f}' {hit['offset_arcmin'][1]:+.1f}'). "
                f"If the star is in main: 'calibrate main on star'")

    def do_main_on_star():
        """The main camera's matrix from moving the star it sees - its own pixels, no guide."""
        from .search import calibrate_on_star
        if "main" not in cams:
            say("no main camera")
            return
        say("calibrating main on the star...")
        warnings = []
        cal = calibrate_on_star(mount, cams["main"],
                                track_rate=sidereal() if ui["tracking"] else None, log=say,
                                abort=aborted, slack_deg=state.get("backlash_deg"),
                                existing=state.get("cameras", {}).get("main"), warnings=warnings)
        state.setdefault("cameras", {})["main"] = cal
        state["calibrated_at"] = time.time()
        state["calibration_warnings"] = warnings
        persist()
        say("main calibrated on the star" + (f" - {len(warnings)} warning(s)" if warnings else "")
            + ". The guide boresight is unchanged: if servo hands over to main, it was right")

    def do_solve_centre(name):
        """Put a named target on the boresight using the solved frame instead of the counters:
        exact however badly the mount knows where it is - and exactly as good as the boresight."""
        cal = state.get("cameras", {}).get("guide")
        if not cal:
            say("calibrate the guide first - the move is worked out in its image")
            return
        # Repeat until it is there: on the real mount each move landed ~5% short (the RA drive's
        # short measure plus slack), so a single move left 20' and it took three presses.
        for attempt in range(1, 5):
            sol, _, _ = solve_here()
            ha, dec, _, _ = pr.target_hadec(name, site, sol.t)
            px = sol.pixel_hadec(ha, dec)
            axis2 = mount.position()[1]
            d = centring_move(cal, axis2, px)
            if d is None:
                say(f"{name} is too far from the guide field to centre from here - goto it first")
                return
            off = float(np.hypot(*(d * geo.sky_metric(axis2)))) * 60
            if off < 1.0:
                say(f"{name} is on the boresight ({off:.1f}' off, {attempt - 1} move"
                    f"{'s' * (attempt != 2)})")
                return
            if aborted():
                return
            mount.move_to(mount.position() + d,
                          track_rate=sidereal() if ui["tracking"] else None,
                          abort=aborted, approach=state.get("backlash_deg"))
            say(f"{name}: moved {off:.1f}' towards the boresight")
        say(f"{name}: still not within 1' after 4 moves - the guide matrix may need redoing")

    def go_home():
        """Slew to the home pose: counterweight down, tube along the polar axis. Home as the
        counters know it - after syncs that is within a few degrees of the mechanical one."""
        from .mount import HOME
        ui["tracking"] = False
        say("going home...")
        mount.move_to(HOME.copy(), abort=aborted)
        say("home reached" if not aborted() else "going home aborted")

    def do_cal(only=None, mode="blob"):
        say("calibrating on the target...")
        warnings = []
        res = calibrate_cameras(mount, cams, track_rate=sidereal() if ui["tracking"] else None,
                                log=say, abort=aborted, warnings=warnings, only=only,
                                existing=state.get("cameras"), mode=mode)
        state.setdefault("cameras", {}).update(res)
        state["calibrated_at"] = time.time()
        state["calibration_warnings"] = warnings
        persist()
        say(f"calibration done - {len(warnings)} warning(s), see below" if warnings else
            f"calibration complete, saved to {(state_path or STATE_FILE).name}")
        if mode == "blob":
            recover_main()

    def recover_main():
        """Big moves (guide calibration, backlash) overshoot the main camera's tiny field by more
        than the mount repeats to. The guide can always put the target back."""
        main, guide = cams.get("main"), cams.get("guide")
        if not main or not guide or not state.get("cameras", {}).get("guide"):
            return
        if main.latest()[1] is not None or guide.latest()[1] is None:
            return
        say("main lost the target during the moves - bringing it back with the guide")
        do_centre("guide")

    def do_backlash(name=None):
        """Measure lost motion on both axes.

        The guide sees a wide field, so it can measure slack of any size; the main camera is ~180x
        finer but can only measure slack smaller than its own field, and says so when it cannot.
        """
        from .calib import measure_backlash

        if name not in cams:
            name = "guide" if "guide" in cams and state.get("cameras", {}).get("guide") else "main"
        cam, cal = cams.get(name), state.get("cameras", {}).get(name)
        if cam is None or cal is None:
            say(f"{name}: need a calibrated camera with a target in view")
            return
        track = sidereal() if ui["tracking"] else None
        out, saturated = [], False
        for axis in (0, 1):
            lost, sat = measure_backlash(mount, cam, cal, axis, track_rate=track, log=say,
                                         abort=aborted)
            out.append(lost)
            saturated |= sat
            if aborted():
                say("backlash measurement aborted")
                return
        if saturated:
            say(f"{name} cannot resolve this much slack - measure on the guide first, "
                f"reduce it mechanically, then refine here")
            recover_main()
            return
        state["backlash_deg"] = [round(v, 4) for v in out]
        persist()
        say(f"backlash: axis1 {out[0] * 60:.1f}' axis2 {out[1] * 60:.1f}' "
            f"(measured on {name}, now compensated on goto/centre)")
        recover_main()

    def do_centre(name, where="boresight"):
        """Put the object the camera is showing onto the boresight - or onto the frame centre,
        which is what you want when aligning the guide camera itself."""
        cam, cal = cams.get(name), state.get("cameras", {}).get(name)
        if cam is None:
            say(f"no {name} camera")
            return
        if cal is None:
            say(f"{name} is not calibrated - cannot turn pixels into axis angles")
            return
        # Two different ways for the axis1 column to be unusable, and they need different advice.
        # Neither is "the mount disagrees with the image": on a mount pushed round by hand the
        # mount is the one that is wrong, and the matrix is still perfectly good where it was
        # measured, because jacobian() only ever applies the CHANGE in declination since then.
        if not axis1_plausible(cal):
            say(f"{name}: axis1 moved the image further than axis2, which no declination allows - "
                f"that column is noise, not a measurement. Recalibrate {name} away from the pole.")
            return
        stretch = axis1_stretch(cal, mount.position()[1])
        if stretch > 3.0:
            # This is what made centring walk away before: a small error in the axis1 column,
            # multiplied by cos(dec_now)/cos(dec_cal), overshoots by the same factor every pass.
            say(f"{name}: the mount has moved from dec {cal.get('dec_cal', 0):.0f}deg (where this "
                f"was calibrated) to dec {geo.axis2_to_dec(mount.position()[1]):.0f}deg, so axis1 "
                f"is being stretched {stretch:.0f}x and centring would run away. Recalibrate "
                f"{name} here - or sync the mount if it does not really know where it is.")
            return
        target_px = (None if where == "boresight"
                     else [(cam.width - 1) / 2, (cam.height - 1) / 2])
        aim = np.asarray(cal["boresight"] if target_px is None else target_px, dtype=float)
        track = sidereal() if ui["tracking"] else None
        previous = None
        for _ in range(4):
            px = measure(cam, n=5, timeout=3.0)
            if px is None:
                say(f"nothing detected in the {name} image")
                return
            off_px = float(np.hypot(*(aim - px)))
            if off_px < 3.0:
                break
            if previous is not None and off_px > 0.9 * previous:
                # each move should shrink the error; if it does not, the matrix is wrong for the
                # optics in use - overshooting by 2x just bounces the target about
                say(f"{name}: centring is not converging ({previous:.0f} -> {off_px:.0f} px) - "
                    f"recalibrate this camera, its scale looks wrong for the current optics")
                return
            previous = off_px
            d = centring_move(cal, mount.position()[1], px, target_px=target_px)
            if d is None:
                say(f"{name}: implied move is absurd - check the calibration or pick the target again")
                return
            say(f"centring {name}: {off_px:.0f} px off, moving {d.round(3)} deg")
            # No backlash overshoot here: centring is closed-loop, it measures and corrects again.
            # An open-loop detour of twice the slack just throws the target out of a narrow field.
            mount.move_to(mount.position() + d, track_rate=track, abort=aborted, max_rate=0.5)
            if aborted():
                say("centring aborted")
                return
            time.sleep(0.4)
        say(f"{name} target on the {'frame centre' if target_px else 'boresight'} "
            f"({off_px:.0f} px off)")

    def snap_click(name, x, y):
        """The brightest spot near a click in this camera's latest frame, else the click."""
        from .detect import snap
        cam = cams[name]
        frame = cam.latest()[0]
        radius = max(20.0, 0.03 * cam.width)
        hit = None if frame is None else snap(
            frame, x, y, radius, sigma=cam.cfg.get("detect_sigma_picked", cam.cfg.get("detect_sigma", 6.0)),
            min_area=cam.cfg.get("detect_min_area", 3), bayer=bool(cam.cfg.get("bayer")),
            smooth=cam.cfg.get("detect_smooth", 0.0))
        if hit is None:
            say(f"{name}: nothing bright within {radius:.0f} px of the click - using the click "
                f"itself")
            return x, y
        say(f"{name}: snapped to the bright spot at {hit[0]:.1f},{hit[1]:.1f} "
            f"({np.hypot(hit[0] - x, hit[1] - y):.0f} px from the click)")
        return float(hit[0]), float(hit[1])

    def set_boresight(name, fx, fy):
        """Put this camera's boresight on the pixel that was just clicked.

        The direct way, and the one that needs nothing else to be true: no matrix, no mount
        position, no detection. The guide boresight means "where the main camera is looking", so
        the pixel to click is the object that is sitting in the middle of the main image right
        now. On main it sets the aim point itself - the spot the ISS gets driven to.
        """
        cam, cal = cams.get(name), state.get("cameras", {}).get(name)
        if cam is None:
            say(f"no {name} camera")
            return
        if name == "main":
            say("main's boresight is its frame centre - set the GUIDE boresight instead")
            return
        if cal is None:
            say(f"{name} has no calibration to put a boresight in - calibrate it first")
            return
        x, y = snap_click(name, float(fx) * cam.width, float(fy) * cam.height)
        cal["boresight"] = [float(x), float(y)]
        persist()
        centre = np.array([(cam.width - 1) / 2, (cam.height - 1) / 2])
        from .calib import cal_px_per_deg
        off = float(np.linalg.norm(np.array([x, y]) - centre)) / cal_px_per_deg(cal)
        say(f"{name} boresight set to {x:.1f},{y:.1f} - {off:.2f} deg from the frame centre"
            + (" (the ISS now gets driven to that spot, not the middle)" if name == "main" else
               " - it must be the object that is in the MIDDLE of the main image"))

    def pointing():
        pos = mount.position()
        ha_p, dec_p = align.pointing_hadec(state, pos)
        alt_p, az_p = geo.hadec_to_altaz(ha_p, dec_p, site.lat)
        return pos, float(alt_p), float(az_p)

    def in_background(fn):
        threading.Thread(target=busy, args=(fn,), daemon=True).start()

    def relabel_counters():
        """Home redefines what the counters mean, so star points taken before no longer apply."""
        if align.points(state):
            align.clear(state)
            say("star alignment cleared: home redefines the counters it was measured in")

    STAR_KEYS = {ord("S"): "solve", ord("Y"): "solve_sync", ord("K"): "starcal",
                 ord("A"): "align_add", ord("B"): "star_boresight"}
    STAR_ACTIONS = {"solve": do_solve, "solve_sync": do_solve_sync, "align_add": do_align_add,
                    "starcal": do_starcal, "star_boresight": do_star_boresight,
                    "align_clear": None, "solve_centre": None}

    def mount_action(action, params):
        """Same operations as the curses keys, for the browser panel."""
        if action == "estop":
            return emergency_stop()
        if action == "track" and "on" not in params:
            # "track" with "on" is the sidereal toggle, handled below. Taking every "track" as a
            # pass made the sidereal button start an ISS session - with a usable pass coming,
            # a slew to where it begins.
            return start_tracking(params.get("pass"), params.get("sat"), params.get("at"))
        if action == "servo":
            pick = None
            if params.get("fx") is not None and "guide" in cams:
                g = cams["guide"]
                pick = (float(params["fx"]) * g.width, float(params["fy"]) * g.height)
            elif "guide" in cams and cams["guide"].manual and cams["guide"].gate is not None:
                pick = cams["guide"].gate[:2]    # clicked first: the red circle is on it now
            return start_servo(pick)
        if action == "untrack":
            return stop_tracking()
        if action == "stop":
            # Graceful cancel: drop the jog and, if a goto/sync/calibration is running, ask it to
            # give up. The axes decelerate normally instead of losing steps like the estop does.
            ui["jog"][:] = 0
            refresh_jog_rates()
            if ui["mode"] == "track":
                # A tracker owns the mount and re-commands at control_hz, so stopping the axes
                # from here would only produce a stutter. End the session instead.
                return stop_tracking()
            if ui["busy"]:
                ui["abort"].set()
                say("slew cancelled")
            else:
                mount.stop()
            return
        if ui["mode"] == "track" and action != "main_steers":
            ui["msg"] = "tracking a pass - stop it first"
            return
        if ui["busy"]:
            return
        if aborted() and action not in ("stop", "frame", "speed"):
            ui["abort"].clear()  # any deliberate command clears the latched stop
        if action in ("jog", "goto", "track", "calibrate", "centre", "starcal",
                      "solve_centre", "gohome", "spiral", "maincal", "goto_altaz") and not ui["motors"]:
            mount.enable(True)
            ui["motors"] = True
        if action == "jog":
            axis = int(params.get("axis", 1)) - 1
            ui["jog"][axis] = float(params.get("dir", 0))
            refresh_jog_rates()
        elif action == "speed":
            ui["speed"] = max(0, min(len(speeds) - 1, int(params.get("index", 2))))
            refresh_jog_rates()
        elif action == "frame":
            frame = params.get("frame", "axes")
            if frame in jog_frames():
                ui["frame"] = frame
                refresh_jog_rates()
                ui["msg"] = (f"arrows move the target in the {frame} image" if frame != "axes"
                             else "arrows drive the mount axes directly")
        elif action == "track":
            ui["tracking"] = params.get("on") not in (None, "0", "false")
        elif action == "gohome":
            ui["jog"][:] = 0
            in_background(go_home)
        elif action == "forecast":
            mode = "field" if params.get("mode") == "field" else "sky"
            threading.Thread(target=do_forecast, args=(mode,), daemon=True).start()
        elif action == "identify":
            f = str(params.get("file") or "")
            if f and re.fullmatch(r"(servo|track)-\d{8}-\d{6}\.csv", f) and (ROOT / "logs" / f).exists():
                identify_later(ROOT / "logs" / f)
            else:
                identify_later(None)
        elif action == "history":
            refresh_history()
        elif action == "favorite":
            f, q = str(params.get("file") or ""), params.get("q")
            if re.fullmatch(r"(servo|track)-\d{8}-\d{6}\.csv", f) and (ROOT / "logs" / f).with_suffix(".id.json").exists():
                threading.Thread(target=lambda: _safely(favorite_session, f), daemon=True).start()
            elif q:
                threading.Thread(target=lambda: _safely(favorite_query, q), daemon=True).start()
        elif action == "unfavorite":
            sid = str(params.get("id") or "")
            if sid:
                threading.Thread(target=lambda: _safely(favorite_remove, sid), daemon=True).start()
        elif action == "favorite_passes":
            only = str(params.get("id") or "")     # blank: all of them
            threading.Thread(target=lambda: _safely(do_favorite_passes, only), daemon=True).start()
        elif action == "frame_line":
            # a window-frame edge: two clicks in the guide image, saved as a line on the sky chart
            from .mask import frame_line
            g = cams["guide"]
            try:
                pts = [(float(params[f"fx{i}"]) * g.width, float(params[f"fy{i}"]) * g.height)
                       for i in (1, 2)]
                line = frame_line(state, mount.position(), *pts, site.lat, frame=(g.width, g.height))
            except (KeyError, ValueError) as e:
                say(f"frame line: {e}")
            else:
                state.setdefault("frame_lines", []).append({"pts": line, "at": time.time()})
                state["frame_shown"] = True
                persist()
                (a0, h0), (a1, h1) = line[0], line[-1]
                say(f"frame line {len(state['frame_lines'])} saved: az {a0:.1f} alt {h0:.1f} -> "
                    f"az {a1:.1f} alt {h1:.1f}")
        elif action == "frame_undo":
            if state.get("frame_lines"):
                state["frame_lines"].pop()
                persist()
                say(f"frame line removed, {len(state['frame_lines'])} left")
        elif action == "marks":
            name, on = str(params.get("cam") or ""), params.get("on") not in (None, "0", "false")
            if name in cams:
                cams[name].show_marks = on
                hidden = set(state.get("marks_hidden", [])) - {name} | (set() if on else {name})
                state["marks_hidden"] = sorted(hidden)
                persist()
        elif action == "frame_shown":
            state["frame_shown"] = params.get("on") not in (None, "0", "false")
            persist()
        elif action == "track_identified":
            threading.Thread(target=lambda: _safely(track_identified), daemon=True).start()
        elif action == "main_steers":
            state["main_steers"] = params.get("on") not in (None, "0", "false")
            persist()
            tr = session["tracker"]
            if tr is not None:
                tr.main_steers = state["main_steers"]
                if not tr.main_steers:
                    tr.main_streak = 0
            say("main camera " + ("may take over steering when it has the target steady"
                                  if state["main_steers"] else
                                  "does not steer: guide only (main still shows and records)"))
        elif action == "identify_on":
            state["identify_on"] = params.get("on") not in (None, "0", "false")
            persist()
            say("satellite naming " + ("on: live in the guide caption, and after each session"
                                       if state["identify_on"] else
                                       "off (Identify still works when pressed)"))
        elif action == "brightness":
            if solver is None or "guide" not in cams:
                ui["msg"] = "no guide camera to solve"
            else:
                fx, fy = params.get("fx"), params.get("fy")
                in_background(lambda: do_brightness(fx, fy))
        elif action == "goto_altaz":
            ui["jog"][:] = 0
            alt, az = float(params.get("alt", -90)), float(params.get("az", 0)) % 360.0
            in_background(lambda: goto_altaz(alt, az))
        elif action == "spiral":
            ui["jog"][:] = 0
            in_background(do_spiral)
        elif action == "maincal":
            ui["jog"][:] = 0
            in_background(do_main_on_star)
        elif action == "home":
            ui["jog"][:] = 0
            ui["tracking"] = False
            mount.set_home()
            relabel_counters()
            persist()
            ui["msg"] = "home set (counterweight down, tube at pole)"
        elif action in STAR_ACTIONS:
            if solver is None:
                ui["msg"] = "no guide camera to solve"
            elif action == "align_clear":
                align.clear(state)
                persist()
                ui["msg"] = "star alignment cleared"
            elif action == "solve_centre":
                target = (params.get("target") or "").strip()
                if not target:
                    ui["msg"] = "enter a target first"
                else:
                    in_background(lambda: do_solve_centre(target))
            else:
                in_background(STAR_ACTIONS[action])
        elif action in ("sync", "goto"):
            target = (params.get("target") or "").strip()
            if not target:
                ui["msg"] = "enter a target first"
            else:
                ui["jog"][:] = 0
                in_background(lambda: (do_sync if action == "sync" else goto)(target))
        elif action == "centre":
            if cams:
                ui["jog"][:] = 0
                in_background(lambda: do_centre(params.get("cam", "guide"),
                                                params.get("where", "boresight")))
            else:
                ui["msg"] = "no cameras"
        elif action == "boresight":
            if params.get("fx") is not None and params.get("fy") is not None:
                set_boresight(params.get("cam"), params["fx"], params["fy"])
        elif action == "backlash":
            if cams:
                ui["jog"][:] = 0
                in_background(lambda: do_backlash(params.get("cam")))
            else:
                ui["msg"] = "no cameras"
        elif action == "calibrate":
            if cams:
                ui["jog"][:] = 0
                which = params.get("cam")
                only = [which] if which in cams else None
                in_background(lambda: do_cal(only))
            else:
                ui["msg"] = "no cameras"
        elif action == "motors":
            on = params.get("on") not in (None, "0", "false")
            ui["jog"][:] = 0
            if not on:
                ui["tracking"] = False
            mount.enable(on)
            ui["motors"] = on
            ui["msg"] = ("motors energised" if on else
                         "motors off - no holding torque, re-sync if the mount slips")
        elif action == "mask":
            _, alt_m, az_m = pointing()
            ui["msg"] = f"sky mask point: az {az_m:.1f} alt {alt_m:.1f}"

    # ---- tracking sessions started from the browser, sharing this process's mount and cameras ----
    session = {"tracker": None, "traj": None, "info": None, "thread": None}

    def sky_now():
        return sky_payload(cfg, SkyMask.from_config(cfg), session["traj"], site)

    def target_now():
        tr = session["tracker"]
        return tr.target_altaz() if tr else None

    def pass_now():
        info, tr = session["info"], session["tracker"]
        if not info:
            return None
        return dict(info, now=clock.now(), source=tr.source if tr else "idle")

    def identify_session(path=None):
        """Identify: runs beside everything else - it never touches the mount."""
        from . import identify as idf
        path = path or idf.latest_session()
        if path is None:
            say("identify: no session log yet - follow something first")
            return
        g = cfg["cameras"]["guide"]
        try:
            say(f"identify: comparing {Path(path).name} with the satellite catalogues...")
            _, text = idf.what_was_that(path, state, site, frame=(g["width"], g["height"]),
                                        log=say)
            for line in text.splitlines():
                say(line.strip())
        except Exception as e:
            say(f"identify: {e}")
        refresh_history()

    def refresh_history():
        from .identify import list_sessions
        try:
            rows = list_sessions()
            for r in rows:            # can it be added to the favourites? only what is not one yet
                sid = (r.get("result") or {}).get("best_id")
                r["favorite"] = None if not sid else str(sid).lstrip("0") in favs
            ui["history"] = rows
            from . import identify as idf
            tles = idf.load_tles(idf.refresh_catalogs(offline=True, log=lambda *_: None))
            ui["observed"] = idf.observed(tles=tles)     # Coming up marks what was seen before
            publish_favorites()
        except Exception as e:
            say(f"history: {e}")

    from . import favorites as fav
    from .forecast import Forecaster, field_track
    favs, favs_lock = fav.load(), threading.Lock()
    if state.get("std_mags") and not args.sim:
        # History's old "Add to Coming up" kept its ratings in state.json: they are favourites now
        from . import identify as idf
        if fav.migrate(favs, state, idf.load_tles(idf.refresh_catalogs(offline=True,
                                                                      log=lambda *_: None))):
            fav.save(favs)
            persist()
    cfg_mags = cfg.get("forecast", {}).get("std_mags") or {}
    # ratings for what qs.mag lacks: the config's, plus the favourites' own
    forecaster = Forecaster(site, log=say, std_mags={**cfg_mags, **fav.ratings(favs)})

    def publish_favorites():
        """The favourites for the page, with when each was last seen (History)."""
        seen = ui.get("observed") or {}
        with favs_lock:
            ui["favorites"] = [dict(e, sessions=len(e.get("sessions", [])), last_seen=seen.get(k))
                               for k, e in favs.items()]

    def favorites_changed(rating_changed):
        with favs_lock:
            fav.save(favs)
            if rating_changed:
                forecaster.set_ratings({**cfg_mags, **fav.ratings(favs)})
        publish_favorites()
        refresh_history()
        do_favorite_passes()
        if rating_changed and ui.get("forecast"):
            do_forecast(ui["forecast"].get("mode") or "sky")

    def own_rating(sid, session_path=None):
        """(standard magnitude, how) for an object no catalogue rates, else (None, None). From a
        Brightness measurement taken during the session when there is one (its distance and sun
        angle then from the orbit), otherwise the default."""
        from . import forecast as fc
        from . import identify as idf
        if forecaster.rating(sid) is not None:
            return None, None
        std, how = float(cfg.get("forecast", {}).get("default_std_mag", 5.0)), "default"
        if session_path is not None:
            t0, dur = idf._first_last_t(session_path)
            seen = [b for b in ui.get("brightness_log", []) if t0 - 10 <= b["t"] <= t0 + dur + 10]
            if seen:
                sat, _ = get_satellite(cfg, sid, offline=True, log=say)
                rng, ph = fc.range_phase(sat, site, seen[-1]["t"])
                std = float(fc.standard_mag(seen[-1]["mag"], rng, ph))
                how = f"measured mag {seen[-1]['mag']:.1f} at {rng:.0f} km"
        return round(std, 1), how

    def favorite_session(fname):
        """History's "Add to favourites": the satellite a session was identified as."""
        path = ROOT / "logs" / fname
        res = json.loads(path.with_suffix(".id.json").read_text())
        sid, name = res.get("best_id"), res.get("best")
        if not sid:
            say("add to favourites: identify the session first")
            return
        std, how = own_rating(sid, path)
        with favs_lock:
            fav.add(favs, sid, name, std, how, session=fname)
        say(f"{name}: added to favourites"
            + (f", standard magnitude {std:.1f} ({how}) - Coming up lists it now" if std is not None
               else ""))
        favorites_changed(std is not None)

    def favorite_query(query):
        """The favourites box: a name or NORAD number from the catalogues."""
        from . import identify as idf
        q = str(query or "").strip()
        if not q:
            return
        if q.lower() in ISS_NAMES:
            q = "25544"
        found = idf.find_satellite(q, forecaster.catalogue())
        if len(found) != 1:
            names = ", ".join(f"{n} ({i})" for n, i, _, _ in found[:6])
            say(f"add to favourites: no satellite called '{q}' in the catalogues" if not found else
                f"add to favourites: '{q}' matches {len(found)} satellites: {names}"
                + (" ..." if len(found) > 6 else "") + " - give the number")
            return
        name, sid, _, _ = found[0]
        std, how = own_rating(sid)
        with favs_lock:
            fav.add(favs, sid, name, std, how)
        say(f"{name} ({sid}): added to favourites"
            + (f", standard magnitude {std:.1f} ({how})" if std is not None else ""))
        favorites_changed(std is not None)

    def favorite_remove(sid):
        with favs_lock:
            e = fav.remove(favs, sid)
        if e is None:
            return
        say(f"{e['name']}: removed from favourites")
        favorites_changed(e.get("std_mag") is not None)

    def do_favorite_passes(only=None):
        """The favourites' visible passes over the next [forecast] favorite_hours. Reads only."""
        hours = float(cfg.get("forecast", {}).get("favorite_hours", fav.HOURS))
        prev = ui.get("fav_passes") or {}
        only = only if only is not None else prev.get("only_id")
        with favs_lock:
            if only and str(only).lstrip("0") not in favs:
                only = None                     # that one was removed: all of them again
            pick = {k: e for k, e in favs.items() if not only or k == str(only).lstrip("0")}
        if not pick:
            ui["fav_passes"] = None
            return
        ui["fav_passes"] = dict(prev, busy=True)
        try:
            now = clock.now()
            sats = fav.satellites(pick, forecaster.catalogue(), forecaster.rating,
                                  float(cfg.get("forecast", {}).get("default_std_mag", 5.0)))
            items = fav.passes(sats, site, now, hours, mask=SkyMask.from_config(cfg),
                               min_alt=cfg["site"]["min_altitude"])
            missing = len(pick) - len(sats)
            who = next(iter(pick.values()))["name"] if only else "favourites"
            ui["fav_passes"] = {"at": now, "hours": hours, "items": items, "busy": False,
                                "only_id": str(only).lstrip("0") if only else None,
                                "only": who if only else None, "missing": missing}
            say(f"favourite passes: {len(items)} visible in the next {hours:.0f} h ({who})"
                + (f"; {missing} without an orbit in the catalogues" if missing else ""))
        except Exception as e:
            ui["fav_passes"] = dict(prev, busy=False)
            say(f"favourite passes: {e}")

    def do_forecast(mode):
        """Coming up: bright satellites through the guide field, or anywhere visible from here.
        Reads only - it runs beside everything else and never touches the mount."""
        ui["forecast"] = dict(ui.get("forecast") or {}, busy=True)
        try:
            now = clock.now()
            minutes = float(cfg.get("forecast", {}).get("minutes", 60))
            if mode == "field":
                _, alt, az = pointing()
                times = now + np.arange(0.0, minutes * 60.0, 10.0)
                follow = bool(ui["tracking"])
                items = forecaster.run(now, minutes=minutes, max_mag=float(
                    cfg.get("forecast", {}).get("field_max_mag", 7.5)),
                    field=field_track(site, alt, az, now, times, follow),
                    radius_deg=guide_radius_deg(cfg))
                where = (f"Through the guide field (alt {alt:.0f}° az {az:.0f}°, "
                         f"{'following the stars' if follow else 'fixed'})")
            else:
                items = forecaster.run(now, minutes=minutes, mask=SkyMask.from_config(cfg),
                                       min_alt=cfg["site"]["min_altitude"], max_mag=float(
                                           cfg.get("forecast", {}).get("sky_max_mag", 6.0)))
                where = f"Anywhere above {cfg['site']['min_altitude']:g}°"
            ui["forecast"] = {"mode": mode, "at": now, "where": where, "items": items,
                              "minutes": minutes, "busy": False}
            say(f"coming up: {len(items)} bright pass{'es' * (len(items) != 1)} - {where}, "
                f"next {minutes:.0f} min")
        except Exception as e:
            ui["forecast"] = dict(ui.get("forecast") or {}, busy=False)
            say(f"coming up: {e}")

    def identify_later(path):
        threading.Thread(target=identify_session, args=(path,), daemon=True).start()

    from .identify import LiveIdentifier
    live = LiveIdentifier(site, frame=(cfg["cameras"]["guide"]["width"],
                                       cfg["cameras"]["guide"]["height"]), log=say)

    def _safely(fn, *a):
        try:
            fn(*a)
        except Exception as e:
            say(f"{getattr(fn, '__name__', 'error')}: {e}")

    def name_it_live():
        """Name what is being looked at, for the guide caption and the sky chart: the target of
        a running session, or - with no session - the object picked in the guide image, which the
        camera's circle follows while the mount stands still. Each sample is the detection with
        the counters AT ITS FRAME TIME: at a 1 s exposure the counters now are up to half a
        degree further on. Reads only - it never moves the mount."""
        from . import identify as idf
        key, last_t, failed, path_for = None, -np.inf, False, None
        while True:
            time.sleep(0.2)
            try:
                tr, guide = session["tracker"], cams.get("guide")
                samples = []
                if tr is not None:
                    now_key = ("session", id(tr))
                    src = tr.source
                    t, px = tr.last_det.get(src, -np.inf), tr.last_px.get(src)
                    if src in ("guide", "main") and px is not None and np.isfinite(t):
                        samples = [(t, px, src)]
                elif guide is not None and guide.manual and len(guide.pick_track):
                    track = list(guide.pick_track)
                    now_key = ("pick", track[0][0])
                    samples = [(t, (x, y), "guide") for t, x, y in track]
                else:
                    now_key = None
                if now_key != key:
                    key, last_t, failed, path_for = now_key, -np.inf, False, None
                    live.reset()
                    ui["identified"] = None
                if key is None or not state.get("identify_on", True):
                    live.label, ui["identified"] = "", None
                    continue
                for t, px, src in samples:
                    if t > last_t:
                        last_t = t
                        at = mount.position_at(t)
                        live.add(t, state, at if at is not None else mount.position(), px, src)
                if failed:
                    continue
                live.update(clock.now())
                if live.best is None:
                    ui["identified"], path_for = None, None
                elif path_for is None or path_for[0] != live.best[1] or clock.now() - path_for[1] > 60:
                    name, sid, sat = live.best
                    now = clock.now()
                    ui["identified"] = {"name": name, "id": sid, "label": live.label,
                                        "path": idf.sky_path(sat, site, now - 120, now + 900)}
                    path_for = (sid, now)
                else:
                    ui["identified"]["label"] = live.label
            except Exception as e:
                failed = True
                say(f"live identification stopped: {e}")

    threading.Thread(target=name_it_live, name="live-naming", daemon=True).start()

    def track_identified():
        """Sky chart's Track: follow the object just named on its orbit (pass mode). A running
        Follow session is stopped first - the orbit carries the target through faint spells and
        gaps that the camera alone cannot."""
        ident = ui.get("identified")
        if not ident:
            say("track: nothing identified to track")
            return
        tr, th = session["tracker"], session["thread"]
        if tr is not None:
            if (session["info"] or {}).get("mode") != "servo":
                say("track: a pass is already being tracked")
                return
            stop_tracking()
            if th is not None:
                th.join(timeout=10)
        say(f"track: switching to {ident['name']} on its orbit")
        start_tracking(which=ident["id"], at=clock.now())

    def run_tracker(tracker, label):
        """Hand the mount to a tracker until it finishes, then give it back to the console."""
        session["tracker"] = tracker
        ui["mode"] = "track"
        ui["jog"][:] = 0
        ui["tracking"] = False
        # Recording is yours: Start/Stop recording only. A session neither starts nor stops it,
        # nor pauses it while the target is out of sight.
        tracker.run()
        ui["msg"] = f"{label} stopped" if tracker.stop_requested else f"{label} finished"

    def start_servo(pick=None):
        """Follow whatever the cameras can see, with no orbit and no alignment.

        The mount is left exactly where it is pointing and the reference is frozen there, so
        nothing moves until a detection arrives. `pick` is the guide pixel of the object - clicked
        before pressing Follow (the camera has kept its circle on it since) or after: the session
        starts locked on that object, not on the brightest blob in view.
        """
        if session["thread"] and session["thread"].is_alive():
            return
        from .control import FreeRun, Tracker

        def run_session():
            try:
                if not state.get("cameras"):
                    ui["msg"] = "no camera calibration - calibrate first ('c')"
                    return
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                logs = ROOT / "logs"
                logs.mkdir(exist_ok=True)
                tracker = Tracker(cfg, state, mount, cams, clock,
                                  FreeRun(mount.position()),
                                  log=lambda s: ui.__setitem__("msg", s),
                                  log_path=logs / f"servo-{stamp}.csv")
                write_session_state(logs / f"servo-{stamp}.csv", state)
                session["csv"] = logs / f"servo-{stamp}.csv"
                session["info"] = {"mode": "servo", "start": clock.now(), "end": None,
                                   "rise": None, "max_alt": None, "rise_at": "-",
                                   "starts_at": f"{datetime.datetime.now():%H:%M:%S}"}
                if pick is not None and "guide" in cams:
                    track = list(cams["guide"].pick_track)   # select() below clears it
                    tracker.select("guide", *pick)
                    say(f"following the object at guide pixel ({pick[0]:.0f}, {pick[1]:.0f})")
                    tracker.seed_rate("guide", track)
                else:
                    say("servo mode: point at the target and click it in the guide image")
                run_tracker(tracker, "servo")
            except Exception as e:
                ui["msg"] = f"servo error: {e}"
            finally:
                ui["mode"] = "console"
                session["tracker"] = None
                done = session.pop("csv", None)
                if done is not None and state.get("identify_on", True):
                    identify_later(done)
                elif done is not None:
                    refresh_history()
                recorder.set_gate(True)
                try:
                    mount.stop()
                except Exception:
                    pass

        session["thread"] = threading.Thread(target=run_session, name="servo-session", daemon=True)
        session["thread"].start()

    def start_tracking(index=None, which=None, at=None):
        """Track the next usable pass of a satellite - the ISS unless `which` names another, by
        name or NORAD number. `at` (unix time, a Coming-up entry) picks the pass up at that time."""
        if session["thread"] and session["thread"].is_alive():
            return
        from .control import Tracker

        def run_session():
            try:
                ui["msg"] = "planning pass..."
                sat, name = get_satellite(cfg, which, log=say)
                mask = SkyMask.from_config(cfg)
                model = align.current_model(state)
                # far enough ahead for the pass asked for: Favourites look days ahead
                hours = 24.0 if at in (None, "") else max(24.0, (float(at) - clock.now()) / 3600 + 2)
                rows = list_passes(cfg, sat, site, clock.now() - 60, hours, mask, model)
                if not rows:
                    ui["msg"] = f"{name}: no passes in the next {hours:.0f} h"
                    return
                if at not in (None, ""):
                    row = pass_at(rows, float(at))
                    if row is None:
                        ui["msg"] = f"{name}: no pass at {fmt_t(float(at))}"
                        return
                    p, rep, _ = row
                elif index not in (None, ""):
                    p, rep, _ = rows[int(index)]
                else:
                    cand = [r for r in rows if r[2] == "visible" and r[1]["useful_s"] > 0] or rows
                    p, rep, _ = cand[0]
                traj, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask,
                                         model=model)
                rise, _ = pr.pass_horizon(sat, site, p)
                session["traj"] = traj
                session["info"] = {
                    "name": name, "start": traj.t_start, "end": traj.t_end, "rise": rise,
                    "max_alt": p["max_alt"],
                    "rise_at": f"{datetime.datetime.fromtimestamp(rise):%H:%M:%S}",
                    "starts_at": f"{datetime.datetime.fromtimestamp(traj.t_start):%H:%M:%S}"}
                if rep["useful_s"] <= 0:
                    ui["msg"] = f"{name}: pass at {fmt_t(p['rise'])} is not usable ({describe(rep)})"
                    return
                say(f"{name}: pass rising {fmt_t(p['rise'])}, max alt {p['max_alt']:.0f}, "
                    f"{describe(rep)}")
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                logs = ROOT / "logs"
                logs.mkdir(exist_ok=True)
                tracker = Tracker(cfg, state, mount, cams, clock, traj,
                                  log=lambda s: ui.__setitem__("msg", s),
                                  log_path=logs / f"track-{stamp}.csv", name=name)
                write_session_state(logs / f"track-{stamp}.csv", state, name=name)
                session["csv"] = logs / f"track-{stamp}.csv"
                run_tracker(tracker, "tracking")
            except Exception as e:
                ui["msg"] = f"tracking error: {e}"
            finally:
                ui["mode"] = "console"
                session["tracker"] = None
                done = session.pop("csv", None)
                if done is not None and state.get("identify_on", True):
                    identify_later(done)
                elif done is not None:
                    refresh_history()
                recorder.set_gate(True)
                try:
                    mount.stop()
                except Exception:
                    pass

        session["thread"] = threading.Thread(target=run_session, name="track-session", daemon=True)
        session["thread"].start()

    def stop_tracking():
        tr = session["tracker"]
        if tr:
            tr.stop_requested = True
            ui["msg"] = "stopping tracking..."

    def tracking_note():
        """Which camera the running tracker is steering with, for the camera captions."""
        tr = session["tracker"]
        if tr is None or ui["mode"] != "track":
            return None
        return {"source": tr.source, "main_streak": int(getattr(tr, "main_streak", 0)),
                "handoff": int(tr.tr.get("main_handoff_frames", 0))}

    def click_target(name, x, y):
        """A click on a camera image. While a session runs it goes to the tracker - "this is the
        target, start again on it" - so servo can be started first and clicked after; before
        one it only points the camera's detection at the object, which the session then picks up."""
        tr = session["tracker"]
        if tr is not None and ui["mode"] == "track":
            if x is None:
                tr.clear_selection(name)
            else:
                tr.select(name, x, y)
        elif x is not None:
            cams[name].select(x, y)

    def mount_status():
        pos, alt_s, az_s = pointing()
        cal = {}
        for n, c in state.get("cameras", {}).items():
            J = np.array(c["J"], dtype=float)
            scale = float(np.linalg.norm(J[:, 1]))          # px per degree of sky
            cal[n] = {"scale": round(scale, 1),
                      "arcsec_px": round(3600.0 / scale, 2) if scale > 1e-6 else None,
                      "rotation": round(float(np.degrees(np.arctan2(J[1, 0], J[0, 0]))), 1)}
        return {"log": list(ui.log),
                "axis1": round(float(pos[0]), 4), "axis2": round(float(pos[1]), 4),
                "alt": round(alt_s, 2), "az": round(az_s, 2), "compass": geo.compass(az_s),
                "speeds": speeds, "speed_index": ui["speed"], "tracking": ui["tracking"],
                "frame": ui["frame"] if ui["frame"] in jog_frames() else "axes",
                "frames": jog_frames(), "aborted": aborted(), "mode": ui["mode"],
                "jog_raw": ui.get("jog_raw", False),
                "calibrated_at": state.get("calibrated_at"), "motors": ui["motors"],
                "backlash_deg": state.get("backlash_deg"),
                "position_at": state.get("position_at"),
                "cal_warnings": state.get("calibration_warnings", []),
                "alignment": align.describe(state),
                "session": ((session["info"] or {}).get("mode", "pass")
                            if ui["mode"] == "track" else None),
                "busy": ui["busy"], "msg": ui["msg"], "jog": ui["jog"].tolist(), "cal": cal,
                "sat_label": live.label if state.get("identify_on", True) else "",
                "identified": ui.get("identified") if state.get("identify_on", True) else None,
                "frame_lines": [l["pts"] for l in state.get("frame_lines", [])],
                "observed": ui.get("observed") or {},
                "frame_shown": state.get("frame_shown", True),
                "marks_hidden": state.get("marks_hidden", []),
                "identify_on": state.get("identify_on", True),
                "spiral": bool(ui.get("spiral")),
                "history": ui.get("history"),
                "favorites": ui.get("favorites"), "fav_passes": ui.get("fav_passes"),
                "main_steers": bool(state.get("main_steers", cfg["tracking"].get("main_steers", False))),
                "forecast": ui.get("forecast"), "now": clock.now(),
                "rates": [round(float(r), 4) for r in mount.rate_cmd],
                "track": tracking_note()}

    from .ser import RecordControl
    main_cfg = cfg["cameras"]["main"]
    recorder = RecordControl(cams.get("main"), ROOT / main_cfg["record_dir"],
                             bayer=main_cfg.get("bayer"), telescope=f"{main_cfg['focal_length_mm']}mm",
                             instrument=main_cfg["name_match"])
    if cfg["preview"]["enabled"]:
        start_preview(cams, state, args.port or cfg["preview"]["port"], status=status_lines,
                      controls=make_controls(cams, recorder, mount_action, mount_status,
                                             estop=emergency_stop, stopped=aborted,
                                             on_select=click_target,
                                             on_settings=lambda n, c: remember_settings(state, state_path, n, c),
                                             sky=sky_now, pointing=lambda: pointing()[1:],
                                             target=target_now, pass_info=pass_now,
                                             picked=picked))

    def run(scr):
        curses.curs_set(0)
        scr.nodelay(True)
        threading.Thread(target=keepalive, daemon=True).start()
        mount.enable(True)
        cam_names = list(cams)
        sel = 0
        while True:
            k = scr.getch()
            if k == ord("q"):
                break
            elif k == curses.KEY_RIGHT:
                ui["jog"][0] = 0 if ui["jog"][0] < 0 else 1
                refresh_jog_rates()
            elif k == curses.KEY_LEFT:
                ui["jog"][0] = 0 if ui["jog"][0] > 0 else -1
                refresh_jog_rates()
            elif k == curses.KEY_UP:
                ui["jog"][1] = 0 if ui["jog"][1] < 0 else 1
                refresh_jog_rates()
            elif k == curses.KEY_DOWN:
                ui["jog"][1] = 0 if ui["jog"][1] > 0 else -1
                refresh_jog_rates()
            elif k == ord(" "):
                mount_action("stop", {})
            elif k == ord("X"):
                emergency_stop()
            elif ord("1") <= k <= ord("5"):
                ui["speed"] = k - ord("1")
                refresh_jog_rates()
            elif k == ord("t"):
                ui["tracking"] = not ui["tracking"]
            elif k == ord("H"):
                ui["jog"][:] = 0
                ui["tracking"] = False
                mount.set_home()
                relabel_counters()
                persist()
                ui["msg"] = "home set (counterweight down, tube at pole)"
            elif k == ord("s"):
                name = prompt(scr, "sync to (star/planet or 'RAh Dec'): ")
                if name:
                    busy(lambda: do_sync(name))
            elif k == ord("g"):
                name = prompt(scr, "goto (star/planet, M31, NGC 7000 or 'RAh Dec'): ")
                if name:
                    ui["jog"][:] = 0
                    busy(lambda: goto(name))
            elif k == ord("c") and cams:
                busy(do_cal)
            elif k in STAR_KEYS:
                mount_action(STAR_KEYS[k], {})
            elif k == ord("p"):
                mount_action("track", {})
            elif k == ord("v"):
                mount_action("servo", {})
            elif k == ord("m"):
                mount_action("mask", {})
            elif k == ord("f"):
                frames = jog_frames()
                nxt = frames[(frames.index(ui["frame"]) + 1) % len(frames)] if ui["frame"] in frames else frames[0]
                mount_action("frame", {"frame": nxt})
            elif k == ord("x") and cam_names:
                sel = (sel + 1) % len(cam_names)
            elif k in (ord("-"), ord("=")) and cam_names:
                cam = cams[cam_names[sel]]
                if hasattr(cam, "set_exposure"):
                    cam.set_exposure(cam.exposure_ms * (1.5 if k == ord("=") else 1 / 1.5))
                    remember_settings(state, state_path, cam.name, cam)
                    ui["msg"] = f"{cam.name} exposure {cam.exposure_ms:.2f} {cam.exposure_unit}"
            elif k in (ord("["), ord("]")) and cam_names:
                cam = cams[cam_names[sel]]
                if hasattr(cam, "set_gain"):
                    cam.set_gain(cam.gain + (25 if k == ord("]") else -25))
                    remember_settings(state, state_path, cam.name, cam)
                    ui["msg"] = f"{cam.name} gain {cam.gain}"

            pos = mount.position()
            ha, dec = align.pointing_hadec(state, pos)
            alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
            scr.erase()
            lines = [
                "ISS mount console   q quit | arrows jog (toggle) | space stop jog | X EMERGENCY STOP | 1-5 speed | t sidereal",
                "                    H home | s sync | g goto | c calibrate on target | m mask point",
                "                    f arrow frame",
                "                    stars: S solve | Y sync on stars | K calibrate on stars | "
                "A add star | B boresight on star",
                "                    p track next pass | v servo (follow what the camera sees)",
                "                    x select cam | -/= exposure | [/] gain",
                "",
                f"axis1 {pos[0]:+9.4f}   axis2 {pos[1]:+9.4f}   side {'east_looking' if pos[1] <= 90 else 'west_looking'}",
                f"HA {float(ha):+8.3f}   Dec {float(dec):+8.3f}   Alt {float(alt):6.2f}   Az {float(az):6.2f}",
                f"alignment: {align.describe(state)}",
                f"jog speed {speeds[ui['speed']]} deg/s   tracking {'ON' if ui['tracking'] else 'off'}   "
                f"rates {mount.rate_cmd.round(4)}",
                f"arrows move: {ui['frame']}"
                + ("  (target in image)" if ui["frame"] != "axes" else "  (raw axes)"),
                "",
            ]
            for i, n in enumerate(cam_names):
                _, det, _ = cams[n].latest()
                d = f"det ({det.x:7.1f},{det.y:7.1f}) flux {det.flux:8.0f}" if det else "no detection"
                mark = ">" if i == sel else " "
                lines.append(f"{mark}{n:5s} {cams[n].fps:5.1f} fps  exp {cams[n].exposure_ms:6.2f} ms  "
                             f"gain {cams[n].gain:4.0f}  {d}")
            lines += ["", ui["msg"]]
            for i, line in enumerate(lines):
                try:
                    scr.addstr(i, 0, line)
                except curses.error:
                    pass
            scr.refresh()
            time.sleep(0.05)

    def _terminate(signum, frame):
        raise KeyboardInterrupt   # `kill` shuts down like Ctrl-C: state saved, cameras closed

    import signal
    signal.signal(signal.SIGTERM, _terminate)
    try:
        if args.web:
            threading.Thread(target=keepalive, daemon=True).start()
            mount.enable(True)
            print(f"web console on http://localhost:{args.port or cfg['preview']['port']}/"
                  "  (no terminal UI; Ctrl-C to quit)")
            while True:
                time.sleep(0.5)
        else:
            curses.wrapper(run)
    except KeyboardInterrupt:
        pass
    finally:
        ui["quit"] = True
        time.sleep(0.1)
        recorder.close()
        persist()
        for c in cams.values():
            c.stop()
        mount.close()


# ---------------------------------------------------------------- plate solving

def cmd_solve_setup(args, cfg):
    import shutil

    from .solve import INDEX_DIR, fetch_indexes, field_deg

    cam = cfg["cameras"]["guide"]
    w, h = field_deg(cam)
    got = fetch_indexes(cam)
    print(f"guide field {w:.1f} x {h:.1f} deg: index files {got} in {INDEX_DIR}")
    if not shutil.which("solve-field"):
        print("solve-field is missing: sudo apt install astrometry.net")


def cmd_solve(args, cfg):
    import cv2

    from .solve import AstrometrySolver

    if args.image.lower().endswith((".fits", ".fit")):
        from astropy.io import fits
        img = fits.getdata(args.image)
    else:
        img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"cannot read {args.image}")
        return
    site = pr.Site(cfg)
    sol = AstrometrySolver(cfg["cameras"]["guide"], site).solve(img, time.time())
    print(f"solved in {sol.elapsed_s:.1f}s: {sol.describe()}")
    g = cfg["cameras"]["guide"]
    f = 206.265 * g["pixel_um"] * g["bin"] / sol.scale_arcsec()
    print(f"focal length from the stars: {f:.2f} mm")
    for px, label, mag in sol.catalog()[:10]:
        print(f"  {label:>12s} at {px[0]:7.1f},{px[1]:7.1f}")


# ---------------------------------------------------------------- track

def servo_window(sat, site, p, mount_cfg, model=None, dt=1.0):
    """The longest stretch of a pass a servo run could actually follow.

    Three conditions, and servo mode can check none of them for itself: the ISS must be up,
    sunlit (there is nothing to follow but the target), and in a pose the mount can hold. The
    last one is why this matters even though servo mode needs no alignment - the loop will chase
    the target straight into an axis limit otherwise.
    """
    t = np.arange(p["rise"], p["set"], dt)
    ha, dec, alt, _ = pr.sat_hadec(sat, site, t)
    ok = (alt >= site.min_altitude) & (pr.illumination(sat, t) > 0.5)
    reach = np.zeros(len(t), dtype=bool)
    for side in geo.SIDES:
        a1, a2 = (geo.hadec_to_axes if model is None else model.hadec_to_axes)(ha, dec, side)
        a2_lo, a2_hi = mount_cfg.get("axis2_limits", [-10.0, 190.0])
        reach |= ((np.abs(geo.wrap180(a1)) <= mount_cfg["axis1_hour_limit"])
                  & (np.asarray(a2) >= a2_lo) & (np.asarray(a2) <= a2_hi))
    if (ok & reach).any():
        ok &= reach
    elif not ok.any():
        return None, 0.0, False
    # else: the target is visible but the mount cannot hold every pose on the way. Say so and
    # let the run happen anyway - that is the situation the limit guard exists for.
    (i0, i1), _ = pr._longest_run(ok)
    return float(t[i0]), float(t[i1] - t[i0]), bool(reach[i0:i1 + 1].all())


def cmd_identify(args, cfg):
    from . import identify as idf
    path = Path(args.csv) if args.csv else idf.latest_session()
    if path is None:
        print("no servo-*.csv or track-*.csv in logs/")
        return
    g = cfg["cameras"]["guide"]
    print(f"{path.name}:")
    _, text = idf.what_was_that(path, load_state(), pr.Site(cfg), frame=(g["width"], g["height"]),
                                offline=args.offline)
    print(text)


def cmd_track(args, cfg):
    from .control import Tracker

    state = load_state()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.sim:
        from .camera import SimCamera
        from .mount import HOME, SimMount
        from .sim import SimWorld, misalignment

        site = pr.Site(cfg)
        if args.sat:
            sat, name = get_satellite(cfg, args.sat, offline=True)
        else:
            sat, name = pr.make_satellite(pr.SIM_TLE), "ISS"
        mask = SkyMask.from_config(cfg)
        passes = pr.find_passes(sat, site, pr.time_to_unix(sat.epoch), 48)
        model = misalignment(site.lat, (args.polar_error, -0.6 * args.polar_error),
                             args.azimuth_error)
        if args.pass_index is not None:
            p = passes[args.pass_index]
            traj, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)
        else:  # prefer a pass the ISS is actually visible for, else the highest
            plans = [(p, *pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)) for p in passes]
            if args.servo:
                p, traj, rep = max(plans, key=lambda x: servo_window(sat, site, x[0],
                                                                    cfg["mount"], model)[1])
            else:
                p, traj, rep = max(plans, key=lambda x: (x[2]["useful_s"], x[0]["max_alt"]))
        state = {"index": HOME.tolist()}
        if args.servo:
            # Nobody slews in servo mode: the run begins when a human has the target in the
            # guide field, so start the clock where the ISS is up and sunlit and put the tube on
            # it, off by args.hand_error. With a misaligned mount that pose has nothing to do
            # with the planned trajectory, which is the whole point of the exercise.
            from .sim import aim_axes
            t_begin, follow_s, reachable = servo_window(sat, site, p, cfg["mount"], model)
            if t_begin is None:
                print("no part of this pass is up and sunlit - servo mode has nothing to follow")
                return
            print(f"servo starts at {fmt_t(t_begin)}, {follow_s:.0f}s to follow"
                  + ("" if reachable else
                     " (the mount cannot hold every pose on the way - the limit guard will stop "
                     "the axes before the end)"))
            sim_lead = 2.0
            clock = Clock(start_unix=t_begin - sim_lead, speed=args.speed)
            he = args.hand_error
            start = aim_axes(sat, site, t_begin, cfg["mount"], model, args.time_error,
                             offset=(he, -0.7 * he))
        else:
            sim_lead = args.sim_lead
            clock = Clock(start_unix=traj.t_start - sim_lead, speed=args.speed)
            start = traj.at(traj.t_start)[0] + [1.0, -0.5] if args.sim_lead < 60 else HOME
        mount = SimMount(cfg, state, clock, start=start)
        clouds = []
        if args.clouds:
            from .sim import random_clouds
            clouds = random_clouds(traj.t_start, traj.t_end, args.clouds, seed=args.cloud_seed)
            print("clouds at " + ", ".join(f"{a - traj.t_start:+.0f}..{b - traj.t_start:+.0f}s" for a, b in clouds))
        pe = args.pointing_error
        world = SimWorld(cfg, sat, site, mount, time_error_s=args.time_error, traj=traj,
                         mask=mask, clouds=clouds, pointing_error=(pe, -0.7 * pe),
                         polar_error_deg=(args.polar_error, -0.6 * args.polar_error),
                         azimuth_error_deg=args.azimuth_error)
        state["cameras"] = world.calibration_estimate(scale_error=args.cal_scale_error,
                                                      rot_error_deg=args.cal_rot_error)
        cams = {n: SimCamera(n, cfg["cameras"][n], clock, world).start() for n in ("guide", "main")}
        lead = sim_lead
    else:
        clock = Clock()
        site = pr.Site(cfg)
        sat, name = get_satellite(cfg, args.sat, offline=args.offline)
        mask = SkyMask.from_config(cfg)
        rows = list_passes(cfg, sat, site, time.time() - 60, 24, mask)
        if not rows:
            print("no passes in the next 24 h")
            return
        if args.pass_index is not None:
            p, rep, vis = rows[args.pass_index]
        else:
            cand = [r for r in rows if r[2] == "visible" and r[1]["useful_s"] > 0] or rows
            p, rep, vis = cand[0]
        traj, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)
        if "cameras" not in state:
            print("warning: no camera calibration in data/state.json - run console and press 'c'")
        mount = open_mount(cfg, state, clock)
        restore_position(state, mount)
        cams = open_cameras(cfg, clock)
        apply_saved_settings(cams, state)
        lead = None

    print(f"{name}: pass {fmt_t(p['rise'])} max {p['max_alt']:.1f} deg: {describe(rep)}")
    horizon_rise, horizon_set = pr.pass_horizon(sat, site, p)
    print(f"sky path: {sky_path(sat, site, p)}")
    print(f"above horizon: {fmt_t(horizon_rise)} .. {datetime.datetime.fromtimestamp(horizon_set):%H:%M:%S} "
          f"({horizon_set - horizon_rise:.0f}s), trackable from "
          f"{datetime.datetime.fromtimestamp(traj.t_start):%H:%M:%S}")
    print(f"illumination: {describe_shadow(rep, traj.t_start)} (relative to track start)")
    if rep["windows"]:
        print(f"usable windows: {describe_windows(sat, site, rep, traj.t_start)}")
    if rep["tracked_s"] <= 0 and not args.servo:
        print("pass not trackable with current limits")
        return

    from .ser import RecordControl
    main_cfg = cfg["cameras"]["main"]
    recorder = RecordControl(cams.get("main"), ROOT / main_cfg["record_dir"],
                             bayer=main_cfg.get("bayer"), telescope=f"{main_cfg['focal_length_mm']}mm",
                             instrument=main_cfg["name_match"])
    auto_record = bool(main_cfg.get("record") and not args.sim or args.record)

    log_path = logs / f"{'servo' if args.servo else 'track'}-{stamp}{'-sim' if args.sim else ''}.csv"
    if args.servo:
        # The planned pass is kept for the sky chart and the shadow curve, but it is not what the
        # loop follows: the reference is frozen where the tube is now and the cameras do the rest.
        from .control import FreeRun
        reference = FreeRun(visibility=traj)
        print("servo mode: following the cameras, not the orbit - point at the target and "
              "click it in the guide image")
    else:
        reference = traj
    tracker = Tracker(cfg, state, mount, cams, clock, reference, log_path=log_path, name=name)

    def status_lines(name):
        if name != "guide":
            return []
        now = clock.now()
        return [f"t{now - traj.t_start:+.1f}s | src {tracker.source} | "
                f"TLE dt {tracker.time_offset:+.2f}s | sunlit {tracker.lit:.2f}"]

    def stop_tracking():
        """Abandon the pass and halt the motors - the tracker is driving them at up to 3 deg/s."""
        tracker.stop_requested = True
        try:
            mount.estop()
        except Exception as e:
            print(f"emergency stop failed: {e}")
        print("EMERGENCY STOP - tracking abandoned, motors halted")

    # serve the page even with no cameras: the sky chart, countdown and stop button still matter
    if cfg["preview"]["enabled"] and not args.no_preview:
        start_preview(cams, state, args.port or cfg["preview"]["port"], status=status_lines,
                      controls=make_controls(cams, recorder,
                                             on_settings=lambda n, c: remember_settings(state, None, n, c),
                                             estop=stop_tracking,
                                             stopped=lambda: tracker.stop_requested,
                                             on_select=lambda n, x, y: (tracker.select(n, x, y)
                                                                        if x is not None
                                                                        else tracker.clear_selection(n)),
                                             sky=lambda: sky_payload(cfg, mask, traj, site),
                                             pointing=tracker.altaz, target=tracker.target_altaz,
                                             pass_info=lambda: {
                                                 "now": clock.now(), "start": traj.t_start,
                                                 "end": traj.t_end, "max_alt": p["max_alt"],
                                                 "rise": horizon_rise,
                                                 "rise_at": f"{datetime.datetime.fromtimestamp(horizon_rise):%H:%M:%S}",
                                                 "starts_at": f"{datetime.datetime.fromtimestamp(traj.t_start):%H:%M:%S}",
                                                 "source": tracker.source}))

    def on_start():
        if auto_record and recorder.available:
            recorder.set_enabled(True)  # armed; frames start only once the ISS is trackable and seen
            print("recording armed, waiting for the target")

    def on_end():
        if recorder.writer:
            recorder.set_enabled(False)
            last = recorder.last
            print(f"recorded {last['frames']} frames, dropped {last['dropped']} -> {last['path']}")

    try:
        tracker.run(lead_s=lead, on_start=on_start, on_end=on_end, on_record=recorder.set_gate)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if not args.sim:
            remember_position(state, None, mount)
        for c in cams.values():
            c.stop()
        mount.close()
    print(f"log: {log_path}  (rejected detections: {tracker.rejected})")
    if args.sim:
        from .sim import evaluate
        evaluate(world, cfg, log_path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="issctl")
    ap.add_argument("--config", help="config file (default config.toml, else config.example.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("passes", help="list upcoming passes and whether the mount can follow them")
    p.add_argument("--sat", help="satellite name or NORAD number (default: the ISS)")
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--offline", action="store_true")

    p = sub.add_parser("favorites", help="list favourite satellites and their visible passes")
    p.add_argument("--add", help="satellite name or NORAD number to add")
    p.add_argument("--remove", help="NORAD number or exact name to remove")
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--offline", action="store_true")

    p = sub.add_parser("mount-test", help="run each axis both ways and report measured motion")
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=2.0)

    p = sub.add_parser("axis-scale", help="measure an axis against reality and fix gear_ratio")
    p.add_argument("--axis", type=int, choices=(1, 2), required=True)
    p.add_argument("--move", type=float, default=90.0, help="degrees to command (default 90)")
    p.add_argument("--rate", type=float, default=0.5, help="slew rate, deg/s")
    p.add_argument("--measured", type=float, help="angle you actually measured, deg")
    p.add_argument("--write", action="store_true", help="write the corrected gear_ratio to config.toml")

    p = sub.add_parser("solve-setup", help="fetch the star index files plate solving needs")

    p = sub.add_parser("solve", help="plate-solve an image file (FITS, PNG) from the guide camera")
    p.add_argument("image")

    p = sub.add_parser("console", help="jog, home, sync, goto, calibrate cameras")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--port", type=int, help="preview port (default from config)")
    p.add_argument("--sim-backlash", type=float, default=0.0,
                   help="simulated Dec lost motion in degrees, for testing backlash handling")
    p.add_argument("--web", action="store_true",
                   help="browser only, no terminal UI (handy over SSH or from a phone)")

    p = sub.add_parser("track", help="track a pass of the ISS or any catalogued satellite")
    p.add_argument("--sat", help="satellite name or NORAD number (default: the ISS)")
    p.add_argument("--port", type=int, help="preview port (default from config)")
    p.add_argument("--pass", dest="pass_index", type=int, help="index from 'passes' (default: next visible)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--record", action="store_true", help="record SER (also in --sim)")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--sim", action="store_true", help="simulated mount, cameras and sky")
    p.add_argument("--speed", type=float, default=1.0, help="simulation speed factor")
    p.add_argument("--sim-lead", type=float, default=20.0, help="seconds before track start to begin")
    p.add_argument("--time-error", type=float, default=1.5, help="simulated TLE timing error, s")
    p.add_argument("--polar-error", type=float, default=0.0,
                   help="simulated polar-axis misalignment, deg (a real axis tilt, not an offset)")
    p.add_argument("--servo", action="store_true",
                   help="follow the cameras instead of the orbit: no prediction, no alignment, "
                        "no slew - the tube is pointed by hand and the loop keeps the target "
                        "on the boresight")
    p.add_argument("--hand-error", type=float, default=1.0,
                   help="simulated hand-pointing error at the start of a --servo run, deg")
    p.add_argument("--azimuth-error", type=float, default=0.0,
                   help="simulated mount azimuth error, deg (rotation about the vertical)")
    p.add_argument("--pointing-error", type=float, default=0.35,
                   help="simulated mount pointing error on axis1, deg (axis2 gets -0.7x)")
    p.add_argument("--cal-rot-error", type=float, default=2.0,
                   help="simulated camera-calibration rotation error, deg")
    p.add_argument("--cal-scale-error", type=float, default=1.03,
                   help="simulated camera-calibration scale error (1.0 = perfect)")
    p.add_argument("--clouds", type=int, default=0, help="simulate N unpredicted cloud gaps")
    p.add_argument("--cloud-seed", type=int, default=0)

    p = sub.add_parser("identify", help="name the satellite a session followed")
    p.add_argument("csv", nargs="?", help="session log (default: the newest in logs/)")
    p.add_argument("--offline", action="store_true", help="use the catalogues already downloaded")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    commands = {"passes": cmd_passes, "favorites": cmd_favorites, "mount-test": cmd_mount_test,
                "console": cmd_console, "track": cmd_track, "axis-scale": cmd_axis_scale, "solve-setup": cmd_solve_setup,
                "solve": cmd_solve, "identify": cmd_identify}
    try:
        commands[args.cmd](args, cfg)
    except UnknownSatellite as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
