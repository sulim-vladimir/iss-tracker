"""issctl command line: passes, mount-test, console, track (real hardware or --sim)."""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np

from . import geometry as geo
from . import predict as pr
from .clock import Clock
from .config import ROOT, SIM_STATE_FILE, STATE_FILE, load_config, load_state, save_state
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


def describe_shadow(rep, t_ref=None):
    if not rep["shadow"]:
        return "sunlit throughout" if rep["sunlit_s"] > 0 else "in shadow throughout"
    return ", ".join(f"{what} shadow at " + (f"{t - t_ref:+.0f}s" if t_ref else fmt_t(t))
                     for t, what in rep["shadow"])


def list_passes(cfg, sat, site, t0, hours, mask=None):
    rows = []
    for p in pr.find_passes(sat, site, t0, hours):
        _, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)
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
    sat = pr.make_satellite(pr.get_tle(cfg, offline=args.offline))
    mask = SkyMask.from_config(cfg)
    now = time.time()
    print(f"TLE age {pr.tle_age_days(sat, now):.1f} days | sky: {mask.describe()}")
    for i, (p, rep, vis) in enumerate(list_passes(cfg, sat, site, now, args.hours, mask)):
        print(f"{i:2d} {fmt_t(p['rise'])}  max {p['max_alt']:4.1f}  {vis:7s} {describe(rep)}")
        print(f"   {sky_path(sat, site, p)}"
              + (f" | {describe_shadow(rep)}" if rep["shadow"] else ""))
        if rep["windows"]:
            print(f"   usable: {describe_windows(sat, site, rep, rep['track_start'])}")


def sky_payload(cfg, mask, traj=None, site=None):
    """Static data for the sky chart: the pass track, the mask and the horizon limit."""
    out = {"mask": {"openings": mask.openings, "blockers": mask.blockers},
           "min_alt": cfg["site"]["min_altitude"], "track": []}
    if traj is not None and site is not None:
        step = max(1, len(traj.t) // 400)
        a1, a2 = traj.a1[::step], traj.a2[::step]
        ha, dec = geo.axes_to_hadec(a1, a2)
        alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
        lit, open_sky = traj.lit[::step], traj.open_sky[::step]
        t = traj.t[::step]
        out["track"] = [[round(float(z), 2), round(float(a), 2), round(float(l), 2), int(o),
                         round(float(tt - traj.t_start), 1)]
                        for z, a, l, o, tt in zip(az, alt, lit, open_sky, t) if a > -5]
    return out


def apply_saved_settings(cams, state):
    """Re-apply the exposure/gain last used for each camera, so a restart looks the same."""
    for name, cam in cams.items():
        saved = state.get("camera_settings", {}).get(name, {})
        if "exposure_ms" in saved:
            cam.set_exposure(saved["exposure_ms"])
        if "gain" in saved:
            cam.set_gain(saved["gain"])


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
                  on_settings=None):
    """Callbacks the preview page uses for exposure, gain and recording."""

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
        else:
            x, y = float(fx) * cam.width, float(fy) * cam.height
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

    from .calib import calibrate_cameras, centring_move, image_jog_rates, measure
    from .mount import SIDEREAL_DEG_S, SimMount

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
        # a fixed "distant light" a little off the boresight, to exercise jogging and calibration
        world = CalibWorld(cfg, mount)
        cams = {n: SimCamera(n, cfg["cameras"][n], clock, world).start() for n in ("guide", "main")}
    else:
        mount, cams = open_mount(cfg, state, clock), open_cameras(cfg, clock)
        restore_position(state, mount)
    apply_saved_settings(cams, state)
    def status_lines(name):
        return []   # axis angles and the clock live in the mount panel and the top bar

    speeds = [0.004, 0.02, 0.1, 0.5, 2.0]
    ui = {"jog": np.zeros(2), "speed": 2, "tracking": False, "busy": False, "quit": False,
          "msg": "", "frame": "guide", "abort": threading.Event(), "mode": "console",
          "motors": True, "jog_rates": np.zeros(2)}

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

    def keepalive():
        saved = (mount.position().copy(), time.monotonic())
        while not ui["quit"]:
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
                    r = r + [SIDEREAL_DEG_S, 0.0]
                mount.set_rates(*r)
            time.sleep(0.05)

    def persist():
        remember_position(state, state_path, mount)

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
        for _ in range(2):  # second pass corrects for sky motion during the slew
            ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
            cur = mount.position()
            best, options = geo.choose_pose(ha, dec, cfg["mount"], current=cur)
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
            mount.move_to(best["axes"], track_rate=[SIDEREAL_DEG_S, 0.0], abort=aborted,
                          approach=state.get("backlash_deg"))
            if aborted():
                say("goto aborted")
                return
        ui["tracking"] = True
        say(f"at {name} (alt {alt:.1f}), {best['side']}, tracking on")

    def do_sync(name):
        ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
        d = mount.sync(ha, dec)
        persist()
        ui["msg"] = f"synced on {name}: correction {d.round(3)} deg"

    def do_cal(only=None):
        say("calibrating...")
        warnings = []
        res = calibrate_cameras(mount, cams, track_rate=[SIDEREAL_DEG_S, 0.0] if ui["tracking"] else None,
                                log=say, abort=aborted, warnings=warnings, only=only,
                                existing=state.get("cameras"))
        state.setdefault("cameras", {}).update(res)
        state["calibrated_at"] = time.time()
        state["calibration_warnings"] = warnings
        persist()
        say(f"calibration done - {len(warnings)} warning(s), see below" if warnings else
            f"calibration complete, saved to {(state_path or STATE_FILE).name}")
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
        track = [SIDEREAL_DEG_S, 0.0] if ui["tracking"] else None
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
        target_px = (None if where == "boresight"
                     else [(cam.width - 1) / 2, (cam.height - 1) / 2])
        aim = np.asarray(cal["boresight"] if target_px is None else target_px, dtype=float)
        track = [SIDEREAL_DEG_S, 0.0] if ui["tracking"] else None
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

    def pointing():
        pos = mount.position()
        ha_p, dec_p = geo.axes_to_hadec(*pos)
        alt_p, az_p = geo.hadec_to_altaz(ha_p, dec_p, site.lat)
        return pos, float(alt_p), float(az_p)

    def in_background(fn):
        threading.Thread(target=busy, args=(fn,), daemon=True).start()

    def mount_action(action, params):
        """Same operations as the curses keys, for the browser panel."""
        if action == "estop":
            return emergency_stop()
        if action == "track":
            return start_tracking(params.get("pass"))
        if action == "servo":
            return start_servo()
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
        if ui["mode"] == "track":
            ui["msg"] = "tracking a pass - stop it first"
            return
        if ui["busy"]:
            return
        if aborted() and action not in ("stop", "frame", "speed"):
            ui["abort"].clear()  # any deliberate command clears the latched stop
        if action in ("jog", "goto", "track", "calibrate", "centre") and not ui["motors"]:
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
        elif action == "home":
            ui["jog"][:] = 0
            ui["tracking"] = False
            mount.set_home()
            persist()
            ui["msg"] = "home set (counterweight down, tube at pole)"
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

    def run_tracker(tracker, label):
        """Hand the mount to a tracker until it finishes, then give it back to the console."""
        session["tracker"] = tracker
        ui["mode"] = "track"
        ui["jog"][:] = 0
        ui["tracking"] = False
        if main_cfg.get("record") and recorder.available:
            recorder.set_enabled(True)  # armed only; the gate below decides when to write
        recorder.set_gate(False)
        tracker.run(on_record=recorder.set_gate)
        ui["msg"] = f"{label} stopped" if tracker.stop_requested else f"{label} finished"

    def start_servo():
        """Follow whatever the cameras can see, with no orbit and no alignment.

        The mount is left exactly where it is pointing and the reference is frozen there, so
        nothing moves until a detection arrives. Point at the ISS by hand first, then click it
        in the guide image - that click is what starts the estimate.
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
                session["info"] = {"mode": "servo", "start": clock.now(), "end": None,
                                   "rise": None, "max_alt": None, "rise_at": "-",
                                   "starts_at": f"{datetime.datetime.now():%H:%M:%S}"}
                say("servo mode: point at the target and click it in the guide image")
                run_tracker(tracker, "servo")
            except Exception as e:
                ui["msg"] = f"servo error: {e}"
            finally:
                ui["mode"] = "console"
                session["tracker"] = None
                recorder.set_enabled(False)
                recorder.set_gate(True)
                try:
                    mount.stop()
                except Exception:
                    pass

        session["thread"] = threading.Thread(target=run_session, name="servo-session", daemon=True)
        session["thread"].start()

    def start_tracking(index=None):
        if session["thread"] and session["thread"].is_alive():
            return
        from .control import Tracker

        def run_session():
            try:
                ui["msg"] = "planning pass..."
                sat = pr.make_satellite(pr.get_tle(cfg))
                mask = SkyMask.from_config(cfg)
                rows = list_passes(cfg, sat, site, clock.now() - 60, 24, mask)
                if not rows:
                    ui["msg"] = "no passes in the next 24 h"
                    return
                if index not in (None, ""):
                    p, rep, _ = rows[int(index)]
                else:
                    cand = [r for r in rows if r[2] == "visible" and r[1]["useful_s"] > 0] or rows
                    p, rep, _ = cand[0]
                traj, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)
                rise, _ = pr.pass_horizon(sat, site, p)
                session["traj"] = traj
                session["info"] = {
                    "start": traj.t_start, "end": traj.t_end, "rise": rise,
                    "max_alt": p["max_alt"],
                    "rise_at": f"{datetime.datetime.fromtimestamp(rise):%H:%M:%S}",
                    "starts_at": f"{datetime.datetime.fromtimestamp(traj.t_start):%H:%M:%S}"}
                if rep["useful_s"] <= 0:
                    ui["msg"] = f"pass at {fmt_t(p['rise'])} is not usable ({describe(rep)})"
                    return
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                logs = ROOT / "logs"
                logs.mkdir(exist_ok=True)
                tracker = Tracker(cfg, state, mount, cams, clock, traj,
                                  log=lambda s: ui.__setitem__("msg", s),
                                  log_path=logs / f"track-{stamp}.csv")
                run_tracker(tracker, "tracking")
            except Exception as e:
                ui["msg"] = f"tracking error: {e}"
            finally:
                ui["mode"] = "console"
                session["tracker"] = None
                recorder.set_enabled(False)
                recorder.set_gate(True)  # back to manual control in console mode
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

    def mount_status():
        pos, alt_s, az_s = pointing()
        cal = {}
        for n, c in state.get("cameras", {}).items():
            J = np.array(c["J"], dtype=float)
            scale = float(np.linalg.norm(J[:, 1]))          # px per degree of sky
            cal[n] = {"scale": round(scale, 1),
                      "arcsec_px": round(3600.0 / scale, 2) if scale > 1e-6 else None,
                      "rotation": round(float(np.degrees(np.arctan2(J[1, 0], J[0, 0]))), 1)}
        return {"axis1": round(float(pos[0]), 4), "axis2": round(float(pos[1]), 4),
                "alt": round(alt_s, 2), "az": round(az_s, 2), "compass": geo.compass(az_s),
                "speeds": speeds, "speed_index": ui["speed"], "tracking": ui["tracking"],
                "frame": ui["frame"] if ui["frame"] in jog_frames() else "axes",
                "frames": jog_frames(), "aborted": aborted(), "mode": ui["mode"],
                "jog_raw": ui.get("jog_raw", False),
                "calibrated_at": state.get("calibrated_at"), "motors": ui["motors"],
                "backlash_deg": state.get("backlash_deg"),
                "position_at": state.get("position_at"),
                "cal_warnings": state.get("calibration_warnings", []),
                "busy": ui["busy"], "msg": ui["msg"], "jog": ui["jog"].tolist(), "cal": cal}

    from .ser import RecordControl
    main_cfg = cfg["cameras"]["main"]
    recorder = RecordControl(cams.get("main"), ROOT / main_cfg["record_dir"],
                             bayer=main_cfg.get("bayer"), telescope=f"{main_cfg['focal_length_mm']}mm",
                             instrument=main_cfg["name_match"])
    if cfg["preview"]["enabled"]:
        start_preview(cams, state, args.port or cfg["preview"]["port"], status=status_lines,
                      controls=make_controls(cams, recorder, mount_action, mount_status,
                                             estop=emergency_stop, stopped=aborted,
                                             on_settings=lambda n, c: remember_settings(state, state_path, n, c),
                                             sky=sky_now, pointing=lambda: pointing()[1:],
                                             target=target_now, pass_info=pass_now))

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
                persist()
                ui["msg"] = "home set (counterweight down, tube at pole)"
            elif k == ord("s"):
                name = prompt(scr, "sync to (star/planet or 'RAh Dec'): ")
                if name:
                    busy(lambda: do_sync(name))
            elif k == ord("g"):
                name = prompt(scr, "goto (star/planet or 'RAh Dec'): ")
                if name:
                    ui["jog"][:] = 0
                    busy(lambda: goto(name))
            elif k == ord("c") and cams:
                busy(do_cal)
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
            ha, dec = geo.axes_to_hadec(*pos)
            alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
            scr.erase()
            lines = [
                "ISS mount console   q quit | arrows jog (toggle) | space stop jog | X EMERGENCY STOP | 1-5 speed | t sidereal",
                "                    H home | s sync | g goto | c calibrate | m mask point | f arrow frame",
                "                    p track next pass | v servo (follow what the camera sees)",
                "                    x select cam | -/= exposure | [/] gain",
                "",
                f"axis1 {pos[0]:+9.4f}   axis2 {pos[1]:+9.4f}   side {'east_looking' if pos[1] <= 90 else 'west_looking'}",
                f"HA {float(ha):+8.3f}   Dec {float(dec):+8.3f}   Alt {float(alt):6.2f}   Az {float(az):6.2f}",
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
        sat = pr.make_satellite(pr.SIM_TLE)
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
        sat = pr.make_satellite(pr.get_tle(cfg, offline=args.offline))
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

    print(f"pass {fmt_t(p['rise'])} max {p['max_alt']:.1f} deg: {describe(rep)}")
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
    tracker = Tracker(cfg, state, mount, cams, clock, reference, log_path=log_path)

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

    p = sub.add_parser("console", help="jog, home, sync, goto, calibrate cameras")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--port", type=int, help="preview port (default from config)")
    p.add_argument("--sim-backlash", type=float, default=0.0,
                   help="simulated Dec lost motion in degrees, for testing backlash handling")
    p.add_argument("--web", action="store_true",
                   help="browser only, no terminal UI (handy over SSH or from a phone)")

    p = sub.add_parser("track", help="track a pass")
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

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"passes": cmd_passes, "mount-test": cmd_mount_test, "console": cmd_console,
     "track": cmd_track, "axis-scale": cmd_axis_scale}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
