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


def make_controls(cams, recorder=None, mount_action=None, mount_state=None, estop=None, stopped=None,
                  on_select=None, sky=None, pointing=None, target=None, pass_info=None):
    """Callbacks the preview page uses for exposure, gain and recording."""

    def state():
        out = {"cams": {}, "record": recorder.state() if recorder else None,
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

    def gain(name, value=None, delta=None):
        cam = cams.get(name)
        if cam:
            cam.set_gain(int(float(value)) if value else cam.gain + int(float(delta)))

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


# ---------------------------------------------------------------- console

def cmd_console(args, cfg):
    import curses
    import threading

    from .calib import calibrate_cameras, jacobian
    from .mount import SIDEREAL_DEG_S, SimMount

    state_path = SIM_STATE_FILE if args.sim else None
    state = load_state(state_path)
    clock = Clock()
    site = pr.Site(cfg)
    if args.sim:
        from .camera import SimCamera
        from .sim import CalibWorld
        # start away from the pole: at axis2 = 90 the axis1 measurement degenerates (cos dec -> 0)
        mount = SimMount(cfg, state, clock, start=[20.0, 40.0])
        mount.query()
        # a fixed "distant light" a little off the boresight, to exercise jogging and calibration
        world = CalibWorld(cfg, mount)
        cams = {n: SimCamera(n, cfg["cameras"][n], clock, world).start() for n in ("guide", "main")}
    else:
        mount, cams = open_mount(cfg, state, clock), open_cameras(cfg, clock)
    def status_lines(name):
        if name != "guide":
            return []
        pos = mount.position()
        return [f"{datetime.datetime.now():%H:%M:%S}",
                f"axis1 {pos[0]:+.3f}  axis2 {pos[1]:+.3f}"]

    speeds = [0.004, 0.02, 0.1, 0.5, 2.0]
    ui = {"jog": np.zeros(2), "speed": 2, "tracking": False, "busy": False, "quit": False,
          "msg": "", "frame": "guide", "abort": threading.Event(), "mode": "console",
          "motors": True}

    def jog_frames():
        return ["axes"] + [n for n in cams if n in state.get("cameras", {})]

    def jog_rates():
        """Jog in the frame the user is looking at: arrows move the target in the image.

        A rotated camera makes raw axis jogging confusing, so the calibration matrix converts
        the screen direction (right, up) into the axis rates that produce it.
        """
        j = ui["jog"]
        if not j.any():
            return np.zeros(2)
        cal = state.get("cameras", {}).get(ui["frame"])
        if cal is None:
            return j * speeds[ui["speed"]]
        want_px = np.array([j[0], -j[1]])  # screen up is -y in image coordinates
        d = np.linalg.solve(jacobian(cal, mount.position()[1]), want_px)
        peak = np.max(np.abs(d))
        return d / peak * speeds[ui["speed"]] if peak > 1e-9 else np.zeros(2)

    def keepalive():
        while not ui["quit"]:
            if ui["mode"] == "track":
                pass  # the tracker owns the mount while a pass is running
            elif aborted():
                mount.set_rates(0.0, 0.0)
            elif not ui["busy"]:
                r = jog_rates()
                if ui["tracking"]:
                    r = r + [SIDEREAL_DEG_S, 0.0]
                mount.set_rates(*r)
            time.sleep(0.05)

    def persist():
        state["index"] = mount.index.tolist()
        save_state(state, state_path)

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

    def busy(fn):
        ui["abort"].clear()
        ui["busy"] = True
        try:
            fn()
        except Exception as e:
            ui["msg"] = f"error: {e}"
        finally:
            ui["busy"] = False

    def goto(name):
        for _ in range(2):  # second pass corrects for sky motion during the slew
            ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
            cur = mount.position()
            options = []
            for side in geo.SIDES:
                a1, a2 = geo.hadec_to_axes(ha, dec, side)
                options.append((abs(float(a1)), [float(a1), float(a2)]))
            target = min(options)[1]
            mount.move_to(target, track_rate=[SIDEREAL_DEG_S, 0.0], abort=aborted)
            if aborted():
                ui["msg"] = "goto aborted"
                return
        ui["tracking"] = True
        ui["msg"] = f"at {name} (alt {alt:.1f}), tracking on"

    def do_sync(name):
        ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
        d = mount.sync(ha, dec)
        persist()
        ui["msg"] = f"synced on {name}: correction {d.round(3)} deg"

    def do_cal():
        ui["msg"] = "calibrating..."
        warnings = []
        res = calibrate_cameras(mount, cams, track_rate=[SIDEREAL_DEG_S, 0.0] if ui["tracking"] else None,
                                log=lambda s: ui.__setitem__("msg", s), abort=aborted, warnings=warnings)
        state.setdefault("cameras", {}).update(res)
        state["calibrated_at"] = time.time()
        state["calibration_warnings"] = warnings
        persist()
        ui["msg"] = ("WARNING: " + warnings[0]) if warnings else \
                    f"calibration complete, saved to {(state_path or STATE_FILE).name}"

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
        if action == "untrack":
            return stop_tracking()
        if ui["mode"] == "track":
            ui["msg"] = "tracking a pass - stop it first"
            return
        if ui["busy"] and action != "stop":
            return
        if aborted() and action not in ("stop", "frame", "speed"):
            ui["abort"].clear()  # any deliberate command clears the latched stop
        if action in ("jog", "goto", "track", "calibrate") and not ui["motors"]:
            mount.enable(True)
            ui["motors"] = True
        if action == "jog":
            axis = int(params.get("axis", 1)) - 1
            ui["jog"][axis] = float(params.get("dir", 0))
        elif action == "stop":
            ui["jog"][:] = 0
        elif action == "speed":
            ui["speed"] = max(0, min(len(speeds) - 1, int(params.get("index", 2))))
        elif action == "frame":
            frame = params.get("frame", "axes")
            if frame in jog_frames():
                ui["frame"] = frame
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
        elif action == "calibrate":
            if cams:
                ui["jog"][:] = 0
                in_background(do_cal)
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
                session["tracker"] = tracker
                ui["mode"] = "track"
                ui["jog"][:] = 0
                ui["tracking"] = False
                if main_cfg.get("record") and recorder.available:
                    recorder.set_enabled(True)  # armed only; the gate below decides when to write
                recorder.set_gate(False)
                tracker.run(on_record=recorder.set_gate)
                ui["msg"] = "tracking stopped" if tracker.stop_requested else "pass finished"
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
            cal[n] = {"scale": round(float(np.linalg.norm(J[:, 1])), 1),
                      "rotation": round(float(np.degrees(np.arctan2(J[1, 0], J[0, 0]))), 1)}
        return {"axis1": round(float(pos[0]), 4), "axis2": round(float(pos[1]), 4),
                "alt": round(alt_s, 2), "az": round(az_s, 2), "compass": geo.compass(az_s),
                "speeds": speeds, "speed_index": ui["speed"], "tracking": ui["tracking"],
                "frame": ui["frame"] if ui["frame"] in jog_frames() else "axes",
                "frames": jog_frames(), "aborted": aborted(), "mode": ui["mode"],
                "calibrated_at": state.get("calibrated_at"), "motors": ui["motors"],
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
            elif k == curses.KEY_LEFT:
                ui["jog"][0] = 0 if ui["jog"][0] > 0 else -1
            elif k == curses.KEY_UP:
                ui["jog"][1] = 0 if ui["jog"][1] < 0 else 1
            elif k == curses.KEY_DOWN:
                ui["jog"][1] = 0 if ui["jog"][1] > 0 else -1
            elif k == ord(" "):
                ui["jog"][:] = 0
            elif k == ord("X"):
                emergency_stop()
            elif ord("1") <= k <= ord("5"):
                ui["speed"] = k - ord("1")
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
                    ui["msg"] = f"{cam.name} exposure {cam.exposure_ms:.2f} ms"
            elif k in (ord("["), ord("]")) and cam_names:
                cam = cams[cam_names[sel]]
                if hasattr(cam, "set_gain"):
                    cam.set_gain(cam.gain + (25 if k == ord("]") else -25))
                    ui["msg"] = f"{cam.name} gain {cam.gain}"

            pos = mount.position()
            ha, dec = geo.axes_to_hadec(*pos)
            alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
            scr.erase()
            lines = [
                "ISS mount console   q quit | arrows jog (toggle) | space stop jog | X EMERGENCY STOP | 1-5 speed | t sidereal",
                "                    H home | s sync | g goto | c calibrate | m mask point | f arrow frame",
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

def cmd_track(args, cfg):
    from .control import Tracker

    state = load_state()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.sim:
        from .camera import SimCamera
        from .mount import HOME, SimMount
        from .sim import SimWorld

        site = pr.Site(cfg)
        sat = pr.make_satellite(pr.SIM_TLE)
        mask = SkyMask.from_config(cfg)
        passes = pr.find_passes(sat, site, pr.time_to_unix(sat.epoch), 48)
        if args.pass_index is not None:
            p = passes[args.pass_index]
            traj, rep = pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)
        else:  # prefer a pass the ISS is actually visible for, else the highest
            plans = [(p, *pr.plan_pass(sat, site, cfg["mount"], p["rise"], p["set"], mask=mask)) for p in passes]
            p, traj, rep = max(plans, key=lambda x: (x[2]["useful_s"], x[0]["max_alt"]))
        clock = Clock(start_unix=traj.t_start - args.sim_lead, speed=args.speed)
        state = {"index": HOME.tolist()}
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
        lead = args.sim_lead
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
        cams = open_cameras(cfg, clock)
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
    if rep["tracked_s"] <= 0:
        print("pass not trackable with current limits")
        return

    from .ser import RecordControl
    main_cfg = cfg["cameras"]["main"]
    recorder = RecordControl(cams.get("main"), ROOT / main_cfg["record_dir"],
                             bayer=main_cfg.get("bayer"), telescope=f"{main_cfg['focal_length_mm']}mm",
                             instrument=main_cfg["name_match"])
    auto_record = bool(main_cfg.get("record") and not args.sim or args.record)

    log_path = logs / f"track-{stamp}{'-sim' if args.sim else ''}.csv"
    tracker = Tracker(cfg, state, mount, cams, clock, traj, log_path=log_path)

    def status_lines(name):
        if name != "guide":
            return []
        now = clock.now()
        return [
            f"{datetime.datetime.fromtimestamp(now):%H:%M:%S}  t{now - traj.t_start:+.1f}s",
            f"src {tracker.source} | TLE dt {tracker.time_offset:+.2f}s | sunlit {tracker.lit:.2f}",
        ]

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
                      controls=make_controls(cams, recorder, estop=stop_tracking,
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

    p = sub.add_parser("console", help="jog, home, sync, goto, calibrate cameras")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--port", type=int, help="preview port (default from config)")
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
    {"passes": cmd_passes, "mount-test": cmd_mount_test, "console": cmd_console, "track": cmd_track}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
