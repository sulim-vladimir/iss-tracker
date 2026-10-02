"""Closed-loop ISS tracker, in two modes that share one control loop.

Inner loop (control_hz): rate = feed-forward + Kp * position error, using the firmware's step
counts as position feedback.

Outer loop (every camera detection): the ISS position is converted into mount-axis coordinates
and compared with the reference. The residual is split into
  - an along-track part  -> time offset into the TLE trajectory (TLE timing error), and
  - a cross-track part   -> alpha-beta filtered axis offset (pointing/polar/TLE error).
The main camera takes over from the guide camera after a few consecutive detections.

What differs between the modes is only the REFERENCE the residual is measured against:

  pass mode  - a planned Trajectory. The feed-forward comes from the orbit, so the mount is
               already moving at very nearly the right rate before the cameras see anything,
               and the loop only trims. Needs a pointing model good to a few degrees.
  servo mode - a FreeRun: position frozen, velocity zero. cross/cross_rate then stop being a
               correction and become the whole estimate of where the target is and how fast it
               moves, so the feed-forward is derived from the camera instead of the orbit.
               Needs no orbit, no site, no alignment - only the camera calibration.

Servo mode is what makes an unaligned mount usable: point at the ISS by hand, click it, and the
loop keeps it on the boresight. What it cannot do is know in advance that the pass is reachable,
so the mechanical limit guard below is the only thing standing between it and a tripod leg.
"""

import csv
import time

import numpy as np

from . import geometry as geo
from .calib import axes_offset_from_pixel, cal_px_per_deg, jacobian
from .mount import SIDEREAL_DEG_S, limit_correction

SHADOW_OFFSET_CLAMP_S = 5.0


class FreeRun:
    """Reference for servo mode: a target that the tracker knows nothing about.

    `at` returns a frozen position and zero velocity, which turns Tracker.cross/cross_rate into a
    plain alpha-beta estimator of the target's own motion in mount axes. The seed is re-anchored
    as the estimate walks away from it, so wrapping axis1 stays unambiguous over a long run.

    `visibility` may be a planned Trajectory, used ONLY for its shadow and obstruction curves -
    both are functions of time alone and need no alignment, so a servo run can still coast
    through Earth's shadow when a pass has been identified.
    """

    servo = True
    side = None

    def __init__(self, seed=None, visibility=None):
        # Seeded lazily from the mount's own reading at the first control cycle: a position taken
        # at construction time can predate the driver's first query, and anchoring the whole run
        # to a stale pose sends the mount off to wherever that was.
        self.seed = None if seed is None else np.asarray(seed, dtype=float).copy()
        self.visibility = visibility
        self.t_start, self.t_end = -np.inf, np.inf

    def at(self, tq):
        if self.seed is None:
            raise RuntimeError("servo reference used before it was seeded")
        return self.seed.copy(), np.zeros(2)

    def reseed(self, pos):
        self.seed = np.asarray(pos, dtype=float).copy()

    def illum_at(self, tq):
        return 1.0 if self.visibility is None else self.visibility.illum_at(tq)

    def open_at(self, tq):
        return 1.0 if self.visibility is None else self.visibility.open_at(tq)



class Tracker:
    def __init__(self, cfg, state, mount, cameras, clock, traj, log=print, log_path=None,
                 name="ISS"):
        self.name = name          # what is being followed, for the messages
        self.state = state        # the star alignment, to turn axis angles into alt/az
        # Whether the main camera may take over steering. Off by default: the first good real
        # track was guide-only, and every mess since came from main taking over (a star, then
        # noise). Main still shows and records; it just does not steer until switched on.
        self.main_steers = bool((state or {}).get("main_steers", cfg["tracking"].get("main_steers", False)))
        self.cfg, self.mount, self.cams, self.clock, self.traj, self.log = cfg, mount, cameras, clock, traj, log
        m, tr = cfg["mount"], cfg["tracking"]
        self.tr = tr
        self.dt = 1.0 / m["control_hz"]
        self.kp = m["kp_position"]
        self.cmd_latency = m["command_latency_s"]
        self.cal = state.get("cameras", {})
        self.lat = cfg["site"]["latitude"]
        self.servo = bool(getattr(traj, "servo", False))
        # Pass mode's gains filter a slow residual around a trajectory that already carries the
        # motion. In servo mode the same two numbers ARE the motion, so they have to follow the
        # target itself - a filter tuned to smooth a drift lags a 1 deg/s satellite by arcminutes.
        self.alpha = tr.get("servo_alpha", 0.5) if self.servo else tr["cross_alpha"]
        self.beta = tr.get("servo_beta", 0.3) if self.servo else tr["cross_beta"]
        # How fast the search gate opens after a loss. Pass mode can afford a slow ramp: the
        # prediction keeps pointing at the target while it is out of sight. Servo mode coasts on
        # an estimate instead, so its uncertainty grows at roughly the target's own speed - and a
        # gate that opens slower than give_up_s turns a one-second glitch into a lost pass.
        self.growth = tr.get("servo_reacquire_growth_arcmin_per_s", 60.0) if self.servo \
            else tr["reacquire_growth_arcmin_per_s"]
        self.give_up_s = tr.get("servo_give_up_s", 20.0)
        # How far the target may smear across the guide image during one exposure while the
        # servo centres it. Jumping the whole offset at full speed - 2.6 deg in under a second -
        # turned a 1 s exposure into a streak and lost the object every time on the rig.
        self.smear_px = tr.get("servo_smear_px", 20.0)
        self.reseed_deg = tr.get("servo_reseed_deg", 30.0)
        self.at_limit = [False, False]
        self.t0 = None
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
        self.recordable = False
        self.last_good = -np.inf
        self.rejected = 0
        self.force_accept = False
        # Acquisition in pass mode: before the first lock, a detection is only a candidate. See
        # _acquire - the brightest blob anywhere, trusted outright, was a star 60 deg away.
        self.acquire_radius = tr.get("acquire_radius_arcmin", 180.0)
        self.acquire_settle = tr.get("acquire_settle_deg", 1.0)
        self.acquire_still_px = tr.get("acquire_still_px", 6.0)
        self.acquire_s = (tr.get("acquire_min_s", 1.0), tr.get("acquire_max_s", 5.0))
        self.candidates = {n: [] for n in cameras}
        # Main camera: where its last accepted/candidate detection was, and how far the next may
        # be from it - see _vision. Noise in a 10 ms main frame took over the loop otherwise.
        self.main_last_px = None
        self.main_still_px = tr.get("main_still_px", 40.0)
        self.main_step_px = tr.get("main_step_px", 80.0)
        # Pass mode: until this time the reference holds still at the pass's position then, so
        # the mount waits AHEAD of the satellite instead of chasing where it is (see _intercept).
        self.hold_until = None
        self.last_px = {}
        self.stop_requested = False
        self.on_visibility = None
        self.on_record = None
        for name, cam in cameras.items():
            cam.gate_fn = lambda t, n=name: self._predicted_gate(n, t)
        self._csv = None
        if log_path:
            self._csv_file = open(log_path, "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["t", "a1", "a2", "alt", "az", "tgt1", "tgt2", "cmd1", "cmd2",
                                "time_offset", "cross1", "cross2", "source", "det_x", "det_y",
                                "lit", "open"])

    def target_altaz(self):
        """Where the ISS should be right now, per the corrected prediction."""
        return self.altaz(self.target(self.clock.now())[0])

    def select(self, name, x, y):
        """User pointed at the ISS in a frame: lock onto it and trust the next detection."""
        cam = self.cams.get(name)
        if not cam:
            return
        cam.select(x, y)
        self.force_accept = True
        self.main_streak = 0
        # A fresh start on the clicked object, not a correction blended into whatever was being
        # followed: that may have been a star, and half a jump plus a rate kick is a lurch.
        self.last_good = -np.inf
        if self.servo:
            self.cross_rate[:] = 0.0      # in servo mode the rate belonged to the old object
        self.log(f"target selected by hand in {name} at ({x:.0f}, {y:.0f})")

    def seed_rate(self, name, track, window_s=8.0, min_points=3, min_span_s=1.5):
        """Servo: start at the picked object's own speed and direction, measured while the
        camera's circle followed it before Follow was pressed. Otherwise the rate starts at zero
        and is learned frame by frame - slowly, at a 1 s guide exposure - while the mount lags.
        `track` is the camera's pick_track: (t, x, y). Returns the rate (deg/s, axes) or None."""
        cal = self.cal.get(name)
        pts = list(track)
        if not self.servo or cal is None or len(pts) < min_points \
                or self.clock.now() - pts[-1][0] > 5.0:
            return None
        pts = [p for p in pts if p[0] >= pts[-1][0] - window_s]
        span = pts[-1][0] - pts[0][0]
        if len(pts) < min_points or span < min_span_s:
            return None
        rows = []
        for t, x, y in pts:
            m = self.mount.position_at(t)
            if m is None:
                return None
            rows.append((t, *(m + axes_offset_from_pixel(cal, m[1], (x, y)))))
        a = np.array(rows)
        a[:, 1] = a[0, 1] + geo.wrap180(a[:, 1] - a[0, 1])
        rate = np.array([np.polyfit(a[:, 0] - a[0, 0], a[:, k], 1)[0] for k in (1, 2)])
        if not np.all(np.isfinite(rate)) or np.any(np.abs(rate) > np.asarray(self.mount.max_rate)):
            return None
        self.cross_rate = rate
        sky = float(np.hypot(*(rate * geo.sky_metric(a[-1, 2]))))
        self.log(f"measured its motion over {span:.0f} s ({len(pts)} frames): {sky:.2f} deg/s - "
                 f"starting at that rate")
        return rate

    def clear_selection(self, name=None):
        for n, cam in self.cams.items():
            if name in (None, n):
                cam.clear_selection()
        self.log("manual selection cleared, back to automatic")

    def altaz(self, pos=None):
        """Where the telescope is actually pointing, in the sky the user sees - through the star
        alignment. The ideal mount put the target dot on the sky chart, and every alt/az in the
        log, several degrees off on a tripod 3.7 deg from the pole."""
        from . import align
        pos = self.mount.last[1] if pos is None else pos
        ha, dec = align.pointing_hadec(self.state or {}, pos)
        alt, az = geo.hadec_to_altaz(ha, dec, self.lat)
        return float(alt), float(az)

    def cross_at(self, t):
        if self.t_update is None:
            return self.cross.copy()
        return self.cross + self.cross_rate * (t - self.t_update)

    def target(self, t):
        if self.hold_until is not None and t < self.hold_until:
            p, _ = self.traj.at(self.hold_until + self.time_offset)
            return p + self.cross_at(t), np.zeros(2)
        p, v = self.traj.at(t + self.time_offset)
        return p + self.cross_at(t), v + self.cross_rate

    def _slew_time(self, frm, to):
        """Seconds to move between two poses: accelerate, cruise, decelerate, slowest axis."""
        d = np.abs(geo.wrap180(np.asarray(to, dtype=float) - np.asarray(frm, dtype=float)))
        vmax = np.asarray(self.mount.max_rate, dtype=float) * 0.8
        acc = max(float(self.mount.max_accel), 1e-3)
        t = np.where(d > vmax ** 2 / acc, d / vmax + vmax / acc, 2 * np.sqrt(d / acc))
        return float(np.max(t))

    def _intercept(self, now, margin=3.0):
        """When to start following: the first moment of the pass that the satellite can be SEEN
        (sunlit, and not behind a mapped obstruction) and the mount can reach ahead of it.
        Aiming at where it is NOW (a pass already up) had the mount arrive late and trail it for
        the whole pass; starting where the pass became trackable parked it in Earth's shadow."""
        here = self.mount.position()
        t = max(now, self.traj.t_start)
        while t < self.traj.t_end:
            seen = (self.traj.illum_at(t) >= self.tr["shadow_threshold"]
                    and self.traj.open_at(t) >= 0.5)
            if seen:
                p, _ = self.traj.at(t)
                if self._slew_time(here, p) + margin <= t - now:
                    return t
            t += 1.0
        return self.traj.t_start

    # ---- vision ----
    def _jump_allowance(self, lost_for):
        """How far from the estimate a detection may sit. Grows while we coast blind (clouds),
        because the prediction drifts, but never far enough to let a random star take over."""
        extra = self.growth * max(0.0, lost_for - self._lost_timeout("guide"))
        return min(self.tr["max_offset_jump_arcmin"] + extra, self.tr["max_reacquire_arcmin"])

    def _main_allowance(self, lost_for):
        """How far a main detection may sit from the estimate: a little more than the guide can
        place it, opening slowly while nothing has been seen, and never to the whole main field."""
        base = self.tr.get("main_agree_arcmin", 1.5)
        return min(base + 0.5 * max(0.0, lost_for), 3 * base)

    def _search_gate(self, name, now):
        """Restrict the search to where the ISS can plausibly be, instead of the whole frame."""
        cam = self.cams[name]
        cal = self.cal.get(name)
        if cam.manual:
            return  # the user picked the target: leave their choice alone
        if cal is None:
            cam.gate = None
            return
        if not np.isfinite(self.last_good):
            # Never locked. Servo mode has no prediction to bound the search with; pass mode
            # does, and the target cannot be further off it than the acquisition radius.
            cam.gate = None if self.servo else (
                cal["boresight"][0], cal["boresight"][1],
                min(self.acquire_radius / 60.0 * cal_px_per_deg(cal), 0.5 * max(cam.width, cam.height)))
            return
        radius = self._jump_allowance(now - self.last_good) / 60.0 * cal_px_per_deg(cal)
        cam.gate = (cal["boresight"][0], cal["boresight"][1],
                    min(radius, 0.5 * max(cam.width, cam.height)))

    def _predicted_gate(self, name, t):
        """Search circle for a frame taken at t, round where the estimate puts the target in it.

        Only for a target the user picked and the loop has locked: the camera's own gate then
        follows the last detection, which is fine while the mount holds still and useless while
        it moves - the target jumps across the image between frames and the circle stays behind
        on a star. Centred instead from the mount's position at that frame's time."""
        cam, cal = self.cams[name], self.cal.get(name)
        if cal is None or not cam.manual or not np.isfinite(self.last_good) or cam.gate is None:
            return None
        meas = self.mount.position_at(t)
        if meas is None:
            return None
        d = self.target(t)[0] - meas
        d[0] = geo.wrap180(d[0])
        x, y = np.asarray(cal["boresight"], dtype=float) - jacobian(cal, meas[1]) @ d
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        lost = self._jump_allowance(t - self.last_good) / 60.0 * cal_px_per_deg(cal)
        return float(x), float(y), max(float(cam.gate[2]), min(lost, 0.5 * max(cam.width, cam.height)))

    def _vision(self, name, det):
        cal = self.cal.get(name)
        if cal is None or not self.visible:
            return  # behind a building or in shadow: anything we detect is not the ISS
        if name == "main" and not self.main_steers:
            return                  # guide only: main watches and records, it does not steer
        self.last_seen[name] = det.t
        if name != "main" and self.main_streak >= self.tr["main_handoff_frames"]:
            # Main camera is in charge; keep the guide gate on the boresight (where the ISS must
            # be) so a handback starts from the right place instead of a stale position.
            self._search_gate(name, det.t)
            return
        meas = self.mount.position_at(det.t)
        if meas is None:
            return

        iss = meas + axes_offset_from_pixel(cal, meas[1], (det.x, det.y))
        p, v = self.traj.at(det.t + self.time_offset)
        o = iss - p - self.cross_at(det.t)
        o[0] = geo.wrap180(o[0])

        # A detection implying a big jump is another object (star, hot pixel, another satellite).
        jump = float(np.hypot(*(o * geo.sky_metric(meas[1])))) * 60
        if name == "main" and not self.force_accept:
            # Main only confirms what the guide already has. Its whole field is 7', so the
            # general jump allowance (25') let ANY star in it take over after three frames - and
            # on the rig, after a guide lock, main held the mount on a star for half a minute.
            # Then noise in a 10 ms frame, wandering hundreds of px from frame to frame, took over
            # and drove the mount off: so the handoff also needs a STEADY blob, and once main is
            # in charge it may not jump.
            px = np.array([det.x, det.y])
            moved = np.inf if self.main_last_px is None else float(np.hypot(*(px - self.main_last_px)))
            in_charge = self.main_streak >= self.tr["main_handoff_frames"]
            if not np.isfinite(self.last_good) or jump > self._main_allowance(det.t - self.last_good) \
                    or (in_charge and moved > self.main_step_px):
                if not in_charge:
                    self.main_streak, self.main_last_px = 0, None
                self.rejected += 1
                return
            if not in_charge and moved > self.main_still_px:
                self.main_streak = 0                 # a new candidate: start counting again
            self.main_last_px = px
            self.main_streak += 1
            if self.main_streak < self.tr["main_handoff_frames"]:
                return
        if not self.force_accept and not self.servo and not np.isfinite(self.last_good) \
                and not self._acquire(name, det, meas, p, v, jump):
            return
        if self.force_accept:
            self.force_accept = False  # user pointed at it, so believe it however far off it is
        elif np.isfinite(self.last_good) and jump > self._jump_allowance(det.t - self.last_good):
            self.rejected += 1
            return
        # Only now is this camera driving the loop - marking it earlier made the captions and the
        # log say "main" while every main detection was being thrown away.
        self.last_det[name] = det.t
        self.source = name
        self.last_px[name] = (det.x, det.y)
        first_fix = not np.isfinite(self.last_good)
        self.last_good = det.t

        resid = o
        if not self.servo:
            # Split the residual: how far along its own path the target is (a clock/TLE error)
            # and how far off it (pointing). In servo mode there is no path to be along, and
            # v is zero anyway, so the whole residual is positional.
            g = geo.sky_metric(meas[1]) ** 2
            vv = float(np.sum(g * v * v))
            dt_obs = float(np.sum(g * o * v)) / vv if vv > 1e-6 else 0.0
            lim = self.tr["max_time_offset_s"]
            self.time_offset = float(np.clip(self.time_offset + self.tr["time_gain"] * dt_obs, -lim, lim))
            resid = o - dt_obs * v

        pred = self.cross_at(det.t)
        dt = self.dt if self.t_update is None else max(det.t - self.t_update, 1e-3)
        if first_fix:
            # Nothing is known yet, so the measurement IS the estimate. Filtering the first fix
            # would leave the mount crawling toward a target it can already see, and in servo
            # mode that first fix is the only thing anchoring the whole run.
            self.cross = pred + resid
        else:
            self.cross = pred + self.alpha * resid
            self.cross_rate = self.cross_rate + self.beta * resid / dt
        self.t_update = det.t

        if name == "guide" and not self.cams[name].manual:
            cam = self.cams[name]
            cam.gate = (det.x, det.y, 0.15 * cam.width)

    def _acquire(self, name, det, meas, p, v, jump):
        """May this detection be the first lock? Pass mode only, and not for a hand pick.

        The ISS is usually the brightest thing in the frame; nothing else is. Taking the brightest
        blob anywhere, at any time, locked onto a star while the mount was still slewing to a
        rocket body's rise point, and drove the estimate 60 deg off (real rig, 2026-09-30). So:
        * nothing before the pass is trackable, and nothing while the mount is still slewing;
        * nothing further from the prediction than acquire_radius_arcmin;
        * and it must stay put in the frame: the mount follows the prediction, so the satellite
          is nearly still while stars drift by at the satellite's own rate. It must hold within
          acquire_still_px for as long as a star would take to move three times that far."""
        cand = self.candidates[name]
        if det.t < max(self.traj.t_start, self.hold_until or -np.inf) or jump > self.acquire_radius:
            cand.clear()
            return False
        settle = geo.sky_metric(meas[1]) * (p + self.cross_at(det.t) - meas)
        settle[0] = geo.wrap180(settle[0])
        if float(np.hypot(*settle)) > self.acquire_settle:
            cand.clear()                         # still slewing: the frame is a smear of sky
            return False
        cal = self.cal[name]
        star_px_s = float(np.hypot(*(jacobian(cal, meas[1]) @ (np.asarray(v) - [SIDEREAL_DEG_S, 0.0]))))
        need = float(np.clip(3 * self.acquire_still_px / max(star_px_s, 1e-6), *self.acquire_s))
        cand.append((det.t, det.x, det.y))
        # only what is still the same blob counts: drop the history once it has moved on
        while cand and np.hypot(cand[0][1] - det.x, cand[0][2] - det.y) > self.acquire_still_px:
            cand.pop(0)
        if cand[-1][0] - cand[0][0] < need:
            return False
        cand.clear()
        self.log(f"acquired {self.name} in {name}: held still {need:.1f} s, "
                 f"{jump:.0f}' from the prediction")
        return True

    def _lost_timeout(self, name):
        """How long this camera may go without a detection before the target counts as lost:
        lost_timeout_s, or 2.5 frame intervals when its frames come further apart. At a 1 s
        guide exposure one skipped frame already passed 1.5 s, and every frame was reported
        "target lost, coasting" on the rig while the guide held the target on the boresight."""
        base = self.tr["lost_timeout_s"]
        cam = self.cams.get(name)
        if cam is None:
            return base
        period = getattr(cam, "exposure_ms", 0.0) / 1000.0
        fps = getattr(cam, "fps", 0.0) or 0.0
        if fps >= 0.2:                        # slower than that is a stall, not a frame rate
            period = max(period, 1.0 / fps)
        return max(base, min(2.5 * period, 10.0))

    def _check_timeouts(self, now):
        timeout = self._lost_timeout
        if "main" in self.cams and now - self.last_seen["main"] > timeout("main") and self.main_streak:
            self.main_last_px = None
            if self.main_streak >= self.tr["main_handoff_frames"]:
                self.log("main camera lost target, back to guide")
            self.main_streak = 0
        if "guide" in self.cams and now - self.last_seen["guide"] > timeout("guide") and not self.cams["guide"].manual:
            self._search_gate("guide", now)
        if all(now - t > timeout(n) for n, t in self.last_seen.items()) and self.source != "predict":
            self.source = "predict"
            if self.servo:
                # There is no prediction to fall back on - the estimated rate is all we have, so
                # coast on it. Constant velocity holds the target inside the guide field for tens
                # of seconds, which covers a cloud gap or a missed frame or two.
                self.log(f"target lost, coasting at {self.cross_rate.round(3)} deg/s")
            else:
                self.log("target lost, following prediction")
                self.cross_rate[:] = 0.0

    # ---- control ----
    def _smear_limit(self, corr):
        """Slow the centring move so the target stays a dot in the camera that is steering.
        Only the correction: the target's own motion (the feed-forward) is not limited."""
        name = self.source if self.source in self.cams else "guide"
        cam, cal = self.cams.get(name), self.cal.get(name)
        if cam is None or cal is None or getattr(cam, "exposure_ms", None) is None:
            return corr
        exp_s = max(cam.exposure_ms / 1000.0, 1e-3)
        vmax = self.smear_px / (cal_px_per_deg(cal) * exp_s)        # deg/s on the sky
        speed = float(np.hypot(*(corr * geo.sky_metric(self.mount.last[1][1]))))
        return corr * (vmax / speed) if speed > vmax else corr

    def _limit_guard(self, cmd, pos):
        """Refuse to drive an axis further past its mechanical limit.

        In pass mode the planner has already walked the whole trajectory and rejected a pass that
        does not fit. In servo mode nothing has, because nothing knows where the mount points -
        so this is the only thing between the loop and a counterweight meeting a tripod leg.
        """
        m = self.cfg["mount"]
        lo, hi = m.get("axis2_limits", [-10.0, 190.0])
        bounds = ((-m["axis1_hour_limit"], m["axis1_hour_limit"]), (lo, hi))
        cmd = np.asarray(cmd, dtype=float).copy()
        for i, (a, b) in enumerate(bounds):
            outside = (pos[i] <= a and cmd[i] < 0) or (pos[i] >= b and cmd[i] > 0)
            if outside:
                cmd[i] = 0.0
                if not self.at_limit[i]:
                    self.log(f"axis{i + 1} at its limit ({pos[i]:+.1f} deg) - refusing to go further")
            self.at_limit[i] = outside
        return cmd

    def _reanchor(self):
        """Move the frozen servo reference up to the estimate.

        cross is an offset from the seed, and axis1 offsets are wrapped to +/-180. Over a long run
        the estimate walks far enough from a fixed seed for that wrap to become ambiguous, so the
        seed follows it. target() is unchanged by the shift.
        """
        shift = self.cross.copy()
        self.traj.reseed(self.traj.seed + shift)
        self.cross = self.cross - shift

    def step(self):
        now = self.clock.now()
        if self.servo and self.traj.seed is None:
            # A fresh query, not mount.last: until the driver has asked once, `last` is still the
            # placeholder home pose, and anchoring the run there would slew the tube away from
            # the target the user just pointed it at.
            self.traj.reseed(self.mount.query())
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
                self.log(f"{self.name} {'entering Earths shadow' if reason == 'shadow' else 'behind an obstruction'}"
                         " - coasting on prediction")
                self.source = reason
                self.main_streak = 0
                if not self.servo:
                    self.cross_rate[:] = 0.0   # the trajectory carries the motion; in servo mode it is the motion
                for cam in self.cams.values():
                    if not cam.manual:
                        cam.gate = None
            else:
                self.log(f"{self.name} should be back in view - looking for it again")
                self.source = "predict"
            if self.on_visibility:
                self.on_visibility(visible, reason)
        self.lit, self.open_sky, self.visible = lit, open_sky, visible
        main, mcal = self.cams.get("main"), self.cal.get("main")
        if main is not None and mcal is not None and not main.manual and self.main_steers:
            # look only where the target must be: the whole frame's brightest blob, when the
            # target is not in view, is noise at a main exposure
            r = self._main_allowance(now - self.last_good if np.isfinite(self.last_good) else 0.0)
            main.gate = (mcal["boresight"][0], mcal["boresight"][1],
                         min(r / 60.0 * cal_px_per_deg(mcal), 0.5 * max(main.width, main.height)))
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

        # Only worth recording while the pass is actually trackable, the ISS is not in shadow or
        # behind something, and we have it: otherwise the frames are empty sky.
        in_window = self.traj.t_start <= now <= self.traj.t_end
        locked = (now - self.last_good) < self.tr["record_lock_timeout_s"]
        recordable = bool(in_window and self.visible and locked)
        if recordable != self.recordable:
            self.recordable = recordable
            if self.on_record:        # only when a recorder follows it: the console records by hand
                self.log(f"recording {'armed' if recordable else 'held (target not trackable)'}")
                self.on_record(recordable)

        t_meas, meas = self.mount.last
        p_meas, _ = self.target(t_meas)
        err = p_meas - meas
        err[0] = geo.wrap180(err[0])
        _, v = self.target(now + self.cmd_latency)
        tracking = self.traj.t_start <= now + self.time_offset <= self.traj.t_end
        ff = v if tracking else np.zeros(2)
        corr = limit_correction(self.kp * err, err, self.mount.max_accel)
        if self.servo:
            corr = self._smear_limit(corr)
        cmd = ff + corr
        cmd = self._limit_guard(cmd, meas)
        pos = self.mount.set_rates(*cmd)
        if self.servo and np.max(np.abs(self.cross)) > self.reseed_deg:
            self._reanchor()
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

    def run(self, lead_s=None, on_start=None, on_end=None, on_visibility=None, on_record=None):
        self.on_visibility = on_visibility
        self.on_record = on_record
        lead_s = self.tr["lead_s"] if lead_s is None else lead_s
        traj = self.traj
        self.mount.enable(True)
        started = False
        last_report = 0.0
        self.t0 = self.clock.now() if self.servo else traj.t_start
        try:
            while not self.stop_requested:
                now = self.clock.now()
                if now > traj.t_end + 2.0:
                    break
                if self.servo and now - max(self.last_good, self.t0) > self.give_up_s:
                    self.log(f"nothing seen for {self.give_up_s:.0f}s - giving up"
                             if np.isfinite(self.last_good) else
                             f"nothing found in {self.give_up_s:.0f}s - is the target in the "
                             f"guide field? giving up")
                    break
                if now < traj.t_start - lead_s:
                    if now - last_report > 10:
                        self.log(f"waiting: slew starts in {traj.t_start - lead_s - now:.0f} s")
                        last_report = now
                    self.mount.query()
                    self.clock.sleep(0.5)
                    continue
                if not self.servo and self.hold_until is None:
                    self.hold_until = self._intercept(now)
                    if self.hold_until > max(now, traj.t_start) + 1.0:
                        if getattr(traj, "az", None) is not None:
                            alt = float(np.interp(self.hold_until, traj.t, traj.alt))
                            az = float(np.interp(self.hold_until, traj.t, np.unwrap(traj.az, period=360))) % 360
                        else:
                            alt, az = self.altaz(traj.at(self.hold_until)[0])
                        why = ("comes into sunlight" if traj.illum_at(max(now, traj.t_start))
                               < self.tr["shadow_threshold"] else "can be met, ahead of it")
                        self.log(f"{self.name}: waiting at alt {alt:.0f} az {az:.0f}, where it {why}; "
                                 f"following from {self.hold_until - now:.0f} s")
                if not started and now >= traj.t_start - 5.0:
                    started = True
                    if on_start:
                        on_start()
                tick = time.monotonic()
                err = self.step()
                if now - last_report > 2.0:
                    sky = err * geo.sky_metric(self.mount.last[1][1]) * 60
                    alt, az = self.altaz()
                    self.log(f"t{now - self.t0:+6.1f}s src={self.source:7s} "
                             f"alt {alt:5.1f} az {az:5.1f} {geo.compass(az):3s} "
                             f"err=({sky[0]:+6.2f},{sky[1]:+6.2f})' "
                             f"dt={self.time_offset:+5.2f}s cross=({self.cross[0] * 60:+5.1f},{self.cross[1] * 60:+5.1f})'")
                    last_report = now
                self.clock.sleep(self.dt - (time.monotonic() - tick) * self.clock.speed)
        finally:
            self.mount.stop()
            for cam in self.cams.values():
                cam.gate_fn = None
            if self.recordable and self.on_record:
                self.on_record(False)
            if on_end:
                on_end()
            if self._csv:
                self._csv_file.close()
