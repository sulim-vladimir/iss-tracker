"""Closed-loop ISS tracker.

Inner loop (control_hz): rate = trajectory feed-forward + Kp * position error, using the
firmware's step counts as position feedback.

Outer loop (every camera detection): the ISS position is converted into mount-axis
coordinates and compared with the prediction. The residual is split into
  - an along-track part  -> time offset into the TLE trajectory (TLE timing error), and
  - a cross-track part   -> alpha-beta filtered axis offset (pointing/polar/TLE error).
The main camera takes over from the guide camera after a few consecutive detections.
"""

import csv
import time

import numpy as np

from . import geometry as geo
from .calib import axes_offset_from_pixel, cal_px_per_deg
from .mount import limit_correction

SHADOW_OFFSET_CLAMP_S = 5.0


class Tracker:
    def __init__(self, cfg, state, mount, cameras, clock, traj, log=print, log_path=None):
        self.cfg, self.mount, self.cams, self.clock, self.traj, self.log = cfg, mount, cameras, clock, traj, log
        m, tr = cfg["mount"], cfg["tracking"]
        self.tr = tr
        self.dt = 1.0 / m["control_hz"]
        self.kp = m["kp_position"]
        self.cmd_latency = m["command_latency_s"]
        self.cal = state.get("cameras", {})
        self.lat = cfg["site"]["latitude"]
        self.time_offset = 0.0
        self.cross = np.zeros(2)
        self.cross_rate = np.zeros(2)
        self.t_update = None
        self.seq = {n: 0 for n in cameras}
        self.last_det = {n: -np.inf for n in cameras}   # last detection that drove the loop
        self.last_seen = {n: -np.inf for n in cameras}  # last detection of any kind
        self.main_streak = 0
        self.source = "predict"
        self.lit = 1.0
        self.open_sky = 1.0
        self.visible = True
        self.last_good = -np.inf
        self.rejected = 0
        self.force_accept = False
        self.last_px = {}
        self.stop_requested = False
        self.on_visibility = None
        self._csv = None
        if log_path:
            self._csv_file = open(log_path, "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["t", "a1", "a2", "alt", "az", "tgt1", "tgt2", "cmd1", "cmd2",
                                "time_offset", "cross1", "cross2", "source", "det_x", "det_y",
                                "lit", "open"])

    def select(self, name, x, y):
        """User pointed at the ISS in a frame: lock onto it and trust the next detection."""
        cam = self.cams.get(name)
        if not cam:
            return
        cam.select(x, y)
        self.force_accept = True
        self.main_streak = 0
        self.log(f"target selected by hand in {name} at ({x:.0f}, {y:.0f})")

    def clear_selection(self, name=None):
        for n, cam in self.cams.items():
            if name in (None, n):
                cam.clear_selection()
        self.log("manual selection cleared, back to automatic")

    def altaz(self, pos=None):
        """Where the telescope is actually pointing, in the sky the user sees."""
        pos = self.mount.last[1] if pos is None else pos
        ha, dec = geo.axes_to_hadec(pos[0], pos[1])
        alt, az = geo.hadec_to_altaz(ha, dec, self.lat)
        return float(alt), float(az)

    def cross_at(self, t):
        if self.t_update is None:
            return self.cross.copy()
        return self.cross + self.cross_rate * (t - self.t_update)

    def target(self, t):
        p, v = self.traj.at(t + self.time_offset)
        return p + self.cross_at(t), v + self.cross_rate

    # ---- vision ----
    def _jump_allowance(self, lost_for):
        """How far from the estimate a detection may sit. Grows while we coast blind (clouds),
        because the prediction drifts, but never far enough to let a random star take over."""
        extra = self.tr["reacquire_growth_arcmin_per_s"] * max(0.0, lost_for - self.tr["lost_timeout_s"])
        return min(self.tr["max_offset_jump_arcmin"] + extra, self.tr["max_reacquire_arcmin"])

    def _search_gate(self, name, now):
        """Restrict the search to where the ISS can plausibly be, instead of the whole frame."""
        cam = self.cams[name]
        cal = self.cal.get(name)
        if cam.manual:
            return  # the user picked the target: leave their choice alone
        if cal is None or not np.isfinite(self.last_good):
            cam.gate = None  # never locked yet: nothing to bound the search with, use the whole frame
            return
        radius = self._jump_allowance(now - self.last_good) / 60.0 * cal_px_per_deg(cal)
        cam.gate = (cal["boresight"][0], cal["boresight"][1],
                    min(radius, 0.5 * max(cam.width, cam.height)))

    def _vision(self, name, det):
        cal = self.cal.get(name)
        if cal is None or not self.visible:
            return  # behind a building or in shadow: anything we detect is not the ISS
        self.last_seen[name] = det.t
        if name == "main":
            self.main_streak += 1
            if self.main_streak < self.tr["main_handoff_frames"]:
                return
        elif self.main_streak >= self.tr["main_handoff_frames"]:
            # Main camera is in charge; keep the guide gate on the boresight (where the ISS must
            # be) so a handback starts from the right place instead of a stale position.
            self._search_gate(name, det.t)
            return
        meas = self.mount.position_at(det.t)
        if meas is None:
            return
        self.last_det[name] = det.t
        self.source = name
        self.last_px[name] = (det.x, det.y)

        iss = meas + axes_offset_from_pixel(cal, meas[1], (det.x, det.y))
        p, v = self.traj.at(det.t + self.time_offset)
        o = iss - p - self.cross_at(det.t)
        o[0] = geo.wrap180(o[0])

        # A detection implying a big jump is another object (star, hot pixel, another satellite).
        jump = float(np.hypot(*(o * geo.sky_metric(meas[1])))) * 60
        if self.force_accept:
            self.force_accept = False  # user pointed at it, so believe it however far off it is
        elif np.isfinite(self.last_good) and jump > self._jump_allowance(det.t - self.last_good):
            self.rejected += 1
            return
        self.last_good = det.t

        g = geo.sky_metric(meas[1]) ** 2
        vv = float(np.sum(g * v * v))
        dt_obs = float(np.sum(g * o * v)) / vv if vv > 1e-6 else 0.0
        lim = self.tr["max_time_offset_s"]
        self.time_offset = float(np.clip(self.time_offset + self.tr["time_gain"] * dt_obs, -lim, lim))
        resid = o - dt_obs * v

        pred = self.cross_at(det.t)
        dt = self.dt if self.t_update is None else max(det.t - self.t_update, 1e-3)
        self.cross = pred + self.tr["cross_alpha"] * resid
        self.cross_rate = self.cross_rate + self.tr["cross_beta"] * resid / dt
        self.t_update = det.t

        if name == "guide" and not self.cams[name].manual:
            cam = self.cams[name]
            cam.gate = (det.x, det.y, 0.15 * cam.width)

    def _check_timeouts(self, now):
        timeout = self.tr["lost_timeout_s"]
        if "main" in self.cams and now - self.last_seen["main"] > timeout and self.main_streak:
            if self.main_streak >= self.tr["main_handoff_frames"]:
                self.log("main camera lost target, back to guide")
            self.main_streak = 0
        if "guide" in self.cams and now - self.last_seen["guide"] > timeout and not self.cams["guide"].manual:
            self._search_gate("guide", now)
        if all(now - t > timeout for t in self.last_seen.values()) and self.source != "predict":
            self.log("target lost, following prediction")
            self.source = "predict"
            self.cross_rate[:] = 0.0

    # ---- control ----
    def step(self):
        now = self.clock.now()
        # The TLE timing error shifts the real shadow entry, but only by seconds. While acquiring,
        # time_offset also absorbs pointing error and can swing far, so clamp its effect here -
        # otherwise a wild estimate could make us declare shadow and stop believing the cameras.
        t_look = now + np.clip(self.time_offset, -SHADOW_OFFSET_CLAMP_S, SHADOW_OFFSET_CLAMP_S)
        lit = self.traj.illum_at(t_look)
        open_sky = self.traj.open_at(t_look)
        visible = lit >= self.tr["shadow_threshold"] and open_sky >= 0.5
        if visible != self.visible:
            reason = "shadow" if lit < self.tr["shadow_threshold"] else "blocked"
            if not visible:
                self.log(f"ISS {'entering Earths shadow' if reason == 'shadow' else 'behind an obstruction'}"
                         " - coasting on prediction")
                self.source = reason
                self.main_streak = 0
                self.cross_rate[:] = 0.0
                for cam in self.cams.values():
                    if not cam.manual:
                        cam.gate = None
            else:
                self.log("ISS should be back in view - looking for it again")
                self.source = "predict"
            if self.on_visibility:
                self.on_visibility(visible, reason)
        self.lit, self.open_sky, self.visible = lit, open_sky, visible
        for name, cam in self.cams.items():
            _, det, seq = cam.latest()
            if seq != self.seq[name]:
                self.seq[name] = seq
                if det is not None:
                    self._vision(name, det)
                elif name == "main":
                    self.main_streak = 0 if self.main_streak < self.tr["main_handoff_frames"] else self.main_streak
        if self.visible:
            self._check_timeouts(now)

        t_meas, meas = self.mount.last
        p_meas, _ = self.target(t_meas)
        err = p_meas - meas
        err[0] = geo.wrap180(err[0])
        _, v = self.target(now + self.cmd_latency)
        tracking = self.traj.t_start <= now + self.time_offset <= self.traj.t_end
        ff = v if tracking else np.zeros(2)
        cmd = ff + limit_correction(self.kp * err, err, self.mount.max_accel)
        pos = self.mount.set_rates(*cmd)
        if self._csv:
            px = self.last_px.get(self.source, (np.nan, np.nan))
            alt, az = self.altaz(pos)
            self._csv.writerow([f"{now:.3f}", f"{pos[0]:.5f}", f"{pos[1]:.5f}",
                                f"{alt:.3f}", f"{az:.3f}", f"{p_meas[0]:.5f}",
                                f"{p_meas[1]:.5f}", f"{cmd[0]:.5f}", f"{cmd[1]:.5f}",
                                f"{self.time_offset:.3f}", f"{self.cross[0]:.5f}", f"{self.cross[1]:.5f}",
                                self.source, f"{px[0]:.1f}", f"{px[1]:.1f}", f"{self.lit:.3f}",
                                f"{self.open_sky:.0f}"])
        return err

    def run(self, lead_s=None, on_start=None, on_end=None, on_visibility=None):
        self.on_visibility = on_visibility
        lead_s = self.tr["lead_s"] if lead_s is None else lead_s
        traj = self.traj
        self.mount.enable(True)
        started = False
        last_report = 0.0
        try:
            while not self.stop_requested:
                now = self.clock.now()
                if now > traj.t_end + 2.0:
                    break
                if now < traj.t_start - lead_s:
                    if now - last_report > 10:
                        self.log(f"waiting: slew starts in {traj.t_start - lead_s - now:.0f} s")
                        last_report = now
                    self.mount.query()
                    self.clock.sleep(0.5)
                    continue
                if not started and now >= traj.t_start - 5.0:
                    started = True
                    if on_start:
                        on_start()
                tick = time.monotonic()
                err = self.step()
                if now - last_report > 2.0:
                    sky = err * geo.sky_metric(self.mount.last[1][1]) * 60
                    alt, az = self.altaz()
                    self.log(f"t{now - traj.t_start:+6.1f}s src={self.source:7s} "
                             f"alt {alt:5.1f} az {az:5.1f} {geo.compass(az):3s} "
                             f"err=({sky[0]:+6.2f},{sky[1]:+6.2f})' "
                             f"dt={self.time_offset:+5.2f}s cross=({self.cross[0] * 60:+5.1f},{self.cross[1] * 60:+5.1f})'")
                    last_report = now
                self.clock.sleep(self.dt - (time.monotonic() - tick) * self.clock.speed)
        finally:
            self.mount.stop()
            if on_end:
                on_end()
            if self._csv:
                self._csv_file.close()
