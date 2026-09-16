"""issctl command line: passes, mount-test, console, track (real hardware or --sim)."""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np

from . import geometry as geo
from . import predict as pr
from .clock import Clock
from .config import ROOT, load_config, load_state, save_state
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


def start_preview(cams, state, port):
    from .preview import Preview
    try:
        Preview(cams, state, port).start()
        print(f"preview on http://localhost:{port}/")
    except OSError as e:
        print(f"preview unavailable on port {port}: {e} (use --port)")


# ---------------------------------------------------------------- hardware

def open_mount(cfg, state, clock):
    from .mount import SerialMount
    return SerialMount(cfg, state, clock)


def open_cameras(cfg, clock):
    from .camera import AsiCamera
    cams = {}
    for name in ("guide", "main"):
        try:
            cams[name] = AsiCamera(name, cfg["cameras"][name], clock, cfg["cameras"]["sdk_lib"]).start()
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

    from .calib import calibrate_cameras
    from .mount import SIDEREAL_DEG_S, SimMount

    state = load_state()
    clock = Clock()
    site = pr.Site(cfg)
    if args.sim:
        mount, cams = SimMount(cfg, state, clock), {}
    else:
        mount, cams = open_mount(cfg, state, clock), open_cameras(cfg, clock)
    if cams and cfg["preview"]["enabled"]:
        start_preview(cams, state, args.port or cfg["preview"]["port"])

    speeds = [0.004, 0.02, 0.1, 0.5, 2.0]
    ui = {"jog": np.zeros(2), "speed": 2, "tracking": False, "busy": False, "quit": False, "msg": ""}

    def keepalive():
        while not ui["quit"]:
            if not ui["busy"]:
                r = ui["jog"] * speeds[ui["speed"]]
                if ui["tracking"]:
                    r = r + [SIDEREAL_DEG_S, 0.0]
                mount.set_rates(*r)
            time.sleep(0.05)

    def persist():
        state["index"] = mount.index.tolist()
        save_state(state)

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

    def busy(fn):
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
            mount.move_to(target, track_rate=[SIDEREAL_DEG_S, 0.0])
        ui["tracking"] = True
        ui["msg"] = f"at {name} (alt {alt:.1f}), tracking on"

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
                    def do_sync():
                        ha, dec, alt, _ = pr.target_hadec(name, site, clock.now())
                        d = mount.sync(ha, dec)
                        persist()
                        ui["msg"] = f"synced on {name}: correction {d.round(3)} deg"
                    busy(do_sync)
            elif k == ord("g"):
                name = prompt(scr, "goto (star/planet or 'RAh Dec'): ")
                if name:
                    ui["jog"][:] = 0
                    busy(lambda: goto(name))
            elif k == ord("c") and cams:
                def do_cal():
                    ui["msg"] = "calibrating..."
                    res = calibrate_cameras(mount, cams, track_rate=[SIDEREAL_DEG_S, 0.0] if ui["tracking"] else None,
                                            log=lambda s: None)
                    state.setdefault("cameras", {}).update(res)
                    persist()
                    ui["msg"] = "camera calibration saved"
                busy(do_cal)
            elif k == ord("m"):
                pos = mount.position()
                ha_m, dec_m = geo.axes_to_hadec(*pos)
                alt_m, az_m = geo.hadec_to_altaz(ha_m, dec_m, site.lat)
                ui["msg"] = f"sky mask point: az {float(az_m):.1f} alt {float(alt_m):.1f}"
            elif k == ord("x") and cam_names:
                sel = (sel + 1) % len(cam_names)
            elif k in (ord("-"), ord("=")) and cam_names:
                cam = cams[cam_names[sel]]
                if hasattr(cam, "set_exposure"):
                    cam.set_exposure(cam.exposure_ms * (1.5 if k == ord("=") else 1 / 1.5))

            pos = mount.position()
            ha, dec = geo.axes_to_hadec(*pos)
            alt, az = geo.hadec_to_altaz(ha, dec, site.lat)
            scr.erase()
            lines = [
                "ISS mount console   q quit | arrows jog (toggle) | space stop | 1-5 speed | t sidereal",
                "                    H set home | s sync | g goto | c calibrate cams | m mask point | x cam | -/= exposure",
                "",
                f"axis1 {pos[0]:+9.4f}   axis2 {pos[1]:+9.4f}   side {'east_looking' if pos[1] <= 90 else 'west_looking'}",
                f"HA {float(ha):+8.3f}   Dec {float(dec):+8.3f}   Alt {float(alt):6.2f}   Az {float(az):6.2f}",
                f"jog speed {speeds[ui['speed']]} deg/s   tracking {'ON' if ui['tracking'] else 'off'}   "
                f"rates {mount.rate_cmd.round(4)}",
                "",
            ]
            for i, n in enumerate(cam_names):
                _, det, _ = cams[n].latest()
                d = f"det ({det.x:7.1f},{det.y:7.1f}) flux {det.flux:8.0f}" if det else "no detection"
                mark = ">" if i == sel else " "
                lines.append(f"{mark}{n:5s} {cams[n].fps:5.1f} fps  exp {cams[n].exposure_ms:6.2f} ms  {d}")
            lines += ["", ui["msg"]]
            for i, line in enumerate(lines):
                try:
                    scr.addstr(i, 0, line)
                except curses.error:
                    pass
            scr.refresh()
            time.sleep(0.05)

    try:
        curses.wrapper(run)
    finally:
        ui["quit"] = True
        time.sleep(0.1)
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
        world = SimWorld(cfg, sat, site, mount, time_error_s=args.time_error, traj=traj,
                         mask=mask, clouds=clouds)
        state["cameras"] = world.calibration_estimate()
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
    print(f"sky path: {sky_path(sat, site, p)}")
    print(f"illumination: {describe_shadow(rep, traj.t_start)} (relative to track start)")
    if rep["windows"]:
        print(f"usable windows: {describe_windows(sat, site, rep, traj.t_start)}")
    if rep["tracked_s"] <= 0:
        print("pass not trackable with current limits")
        return

    recorder = None
    main_cfg = cfg["cameras"]["main"]
    if "main" in cams and (main_cfg.get("record") and not args.sim or args.record):
        from .ser import SerWriter
        out = ROOT / main_cfg["record_dir"]
        out.mkdir(exist_ok=True)
        recorder = SerWriter(out / f"iss-{stamp}.ser", cams["main"].width, cams["main"].height,
                             bayer=main_cfg.get("bayer"), telescope=f"{main_cfg['focal_length_mm']}mm",
                             instrument=main_cfg["name_match"])
        cams["main"].sinks.append(recorder)

    if cams and cfg["preview"]["enabled"] and not args.no_preview:
        start_preview(cams, state, args.port or cfg["preview"]["port"])

    log_path = logs / f"track-{stamp}{'-sim' if args.sim else ''}.csv"
    tracker = Tracker(cfg, state, mount, cams, clock, traj, log_path=log_path)

    def on_start():
        if recorder:
            recorder.active = True
            print(f"recording {recorder.path}")

    def on_end():
        if recorder:
            recorder.close()
            print(f"recorded {recorder.frames} frames, dropped {recorder.dropped}")

    def on_visibility(visible, reason):
        if recorder:
            recorder.active = visible  # nothing to record while the ISS is dark or hidden

    try:
        tracker.run(lead_s=lead, on_start=on_start, on_end=on_end, on_visibility=on_visibility)
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
    p.add_argument("--clouds", type=int, default=0, help="simulate N unpredicted cloud gaps")
    p.add_argument("--cloud-seed", type=int, default=0)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"passes": cmd_passes, "mount-test": cmd_mount_test, "console": cmd_console, "track": cmd_track}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
