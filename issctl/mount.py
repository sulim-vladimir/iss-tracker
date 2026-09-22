"""Mount drivers. Angles are mechanical axis degrees (see geometry.py); rates in deg/s."""

import collections
import threading
import time

import numpy as np

from . import geometry as geo
from .predict import axis_rate_limits, steps_per_deg

HOME = np.array([0.0, 90.0])
SIDEREAL_DEG_S = 360.0 / 86164.0905


class Mount:
    def __init__(self, cfg, state, clock):
        m = cfg["mount"]
        self.cfg = m
        self.clock = clock
        self.max_rate = axis_rate_limits(m)
        self.max_accel = m["max_accel_deg_s2"]
        self.index = np.array(state.get("index", HOME.tolist()), dtype=float)
        self.history = collections.deque(maxlen=400)
        self.last = (clock.now(), HOME.copy())
        self.rate_cmd = np.zeros(2)
        self._hlock = threading.Lock()

    # -- to implement --
    def _set_rates_mech(self, r):  # returns (t, mech_deg)
        raise NotImplementedError

    def _query_mech(self):
        raise NotImplementedError

    def _zero_counters(self):
        raise NotImplementedError

    def enable(self, on):
        pass

    def stop(self):
        self.set_rates(0.0, 0.0)

    def estop(self):
        """Immediate stop, no deceleration ramp. May lose steps: re-sync before trusting positions."""
        self.set_rates(0.0, 0.0)

    def close(self):
        pass

    # -- common --
    def _record(self, t, mech):
        pos = mech + self.index
        with self._hlock:
            self.last = (t, pos)
            self.history.append((t, pos[0], pos[1]))
        return pos

    def set_rates(self, r1, r2):
        r = np.clip([r1, r2], -self.max_rate, self.max_rate)
        self.rate_cmd = r
        t, mech = self._set_rates_mech(r)
        return self._record(t, mech)

    def query(self):
        t, mech = self._query_mech()
        return self._record(t, mech)

    def position(self):
        return self.last[1].copy()

    def position_at(self, t):
        with self._hlock:
            if not self.history:
                return None
            h = np.array(self.history)
        return np.array([np.interp(t, h[:, 0], h[:, 1]), np.interp(t, h[:, 0], h[:, 2])])

    def set_home(self):
        """Declare the current pose as home: counterweight down, tube parallel to the polar axis."""
        self._zero_counters()
        self.index = HOME.copy()
        self.query()

    def restore_position(self, pos):
        """Adopt a position saved by an earlier session.

        The Uno resets when the serial port opens, so its counters always start at zero. Without
        this the driver assumes the mount is still at the last homed pose. Only valid if nobody
        moved the tube by hand in between - re-home if in doubt.
        """
        self.query()
        mech = self.last[1] - self.index
        self.index = np.asarray(pos, dtype=float) - mech
        return self.query()

    def sync(self, ha, dec):
        """Correct the index so the current pose reads as (ha, dec). Returns the correction."""
        cur = self.query()
        side = "east_looking" if cur[1] <= 90.0 else "west_looking"
        true = np.array(geo.hadec_to_axes(ha, dec, side), dtype=float)
        true[0] = cur[0] + geo.wrap180(true[0] - cur[0])
        delta = true - cur
        self.index = self.index + delta
        self.query()
        return delta

    def move_to(self, target, tol=0.003, timeout=120.0, track_rate=None, abort=None, max_rate=None,
                approach=None):
        """Accel-aware position loop. track_rate adds a constant feed-forward (e.g. sidereal).

        max_rate caps the slew for this move: stepper motors skip silently when pushed too fast on
        a stiff or unbalanced axis, and the step counter keeps counting as if nothing happened.

        approach=per-axis backlash in degrees makes the final approach come from the + side on
        every move: overshoot, then come back. The slack is then always taken up the same way, so
        where the counters say the telescope is points is where it actually is.
        """
        target = np.array(target, dtype=float)
        if approach is not None:
            # Overshoot only the axes that would otherwise arrive from the - side, and only by a
            # little more than the slack: every axis then finishes its travel in the + direction.
            slack = np.maximum(np.asarray(approach, dtype=float), 0.0)
            over = np.where(target - self.position() < 0, -1.2 * slack, 0.0)
            if np.any(over < 0):
                self.move_to(target + over, tol=tol, timeout=timeout, track_rate=track_rate,
                             abort=abort, max_rate=max_rate)
                if abort and abort():
                    return self.position()
        ff = np.zeros(2) if track_rate is None else np.asarray(track_rate, dtype=float)
        cap = self.max_rate if max_rate is None else np.minimum(self.max_rate, abs(max_rate))
        dt = 1.0 / self.cfg["control_hz"]
        kp = self.cfg["kp_position"]
        t0 = self.clock.now()
        settled = 0
        while self.clock.now() - t0 < timeout:
            if abort and abort():
                break
            now = self.clock.now()
            err = (target + ff * (now - t0)) - self.last[1]
            cmd = limit_correction(kp * err, err, self.max_accel) + ff
            self.set_rates(*np.clip(cmd, -cap, cap))
            if np.all(np.abs(err) < tol):
                settled += 1
                if settled > 5:
                    break
            else:
                settled = 0
            self.clock.sleep(dt)
        self.set_rates(*ff)
        return self.position()


def limit_correction(cmd, err, accel):
    """Cap a position-correction rate so the axis can stop within the remaining error."""
    cap = np.sqrt(2.0 * 0.8 * accel * np.abs(err))
    return np.clip(cmd, -cap, cap)


class SerialMount(Mount):
    def __init__(self, cfg, state, clock):
        import serial

        super().__init__(cfg, state, clock)
        m = cfg["mount"]
        self.spd = np.array([steps_per_deg(m["axis1"]), steps_per_deg(m["axis2"])])
        self.sign = np.array([-1.0 if m[k]["reverse"] else 1.0 for k in ("axis1", "axis2")])
        self.lock = threading.Lock()
        self.ser = serial.Serial(m["port"], m["baud"], timeout=0.3)
        time.sleep(2.0)  # Uno resets when the port opens
        self.ser.reset_input_buffer()
        ver = self._cmd("V", "ISSMOUNT")
        if not ver:
            raise RuntimeError("no ISSMOUNT firmware response on " + m["port"])
        self._cmd("X", "P")
        self._cmd(f"U {m['axis1']['microsteps']} {m['axis2']['microsteps']}")
        acc = self.max_accel * self.spd
        self._cmd(f"A {acc[0]:.1f} {acc[1]:.1f}")
        self._cmd(f"M {m['firmware_max_step_rate']:.0f}")
        self._cmd(f"I {1000 * m['idle_disable_s']:.0f}")   # auto power-down when stopped
        self.query()

    def _cmd(self, line, expect="OK"):
        with self.lock:
            self.ser.write((line + "\n").encode())
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                resp = self.ser.readline().decode(errors="replace").strip()
                if resp.startswith(expect):
                    return resp
                if resp.startswith("ERR"):
                    raise RuntimeError(f"firmware: {resp} (for '{line}')")
        raise TimeoutError(f"no '{expect}' reply to '{line}'")

    def _parse_p(self, resp):
        t = self.clock.now()
        parts = resp.split()
        steps = np.array([float(parts[1]), float(parts[2])])
        return t, steps * self.sign / self.spd

    def _set_rates_mech(self, r):
        s = r * self.spd * self.sign
        return self._parse_p(self._cmd(f"R {s[0]:.3f} {s[1]:.3f}", "P"))

    def _query_mech(self):
        return self._parse_p(self._cmd("Q", "P"))

    def _zero_counters(self):
        self._cmd("Z 0 0")

    def enable(self, on):
        self._cmd(f"E {1 if on else 0}")

    def stop(self):
        self._parse_p(self._cmd("S", "P"))
        self.rate_cmd = np.zeros(2)

    def estop(self):
        self._cmd("X", "P")

    def close(self):
        try:
            self.estop()
        finally:
            self.ser.close()


class SimMount(Mount):
    """Accel-limited kinematic mount with a small command latency."""

    def __init__(self, cfg, state, clock, start=HOME, latency=0.015, backlash=0.0):
        super().__init__(cfg, state, clock)
        self.mech = np.array(start, dtype=float) - self.index
        self.backlash = np.broadcast_to(np.asarray(backlash, dtype=float), (2,)).copy()
        # The step counter lives on the MOTOR side and never sees the slack: mech is what the
        # firmware reports, axis is where the telescope really points, play is the gap between.
        self.axis = self.mech.copy()
        self.play = np.zeros(2)
        self.phys_history = collections.deque(maxlen=400)
        self.rate = np.zeros(2)
        self.target = np.zeros(2)
        self.pending = collections.deque()
        self.latency = latency
        self.t = clock.now()
        self.lock = threading.Lock()

    def _advance(self):
        now = self.clock.now()
        while self.t < now:
            h = min(0.002, now - self.t)
            while self.pending and self.pending[0][0] <= self.t:
                self.target = self.pending.popleft()[1]
            dv = np.clip(self.target - self.rate, -self.max_accel * h, self.max_accel * h)
            step = (self.rate + dv / 2) * h
            self.mech += step                       # the counter always advances
            play = np.clip(self.play + step, 0.0, self.backlash)
            self.axis += step - (play - self.play)  # the telescope lags while slack is taken up
            self.play = play
            self.rate += dv
            self.t += h
        return now

    def physical_at(self, t):
        """Where the telescope really points, which differs from the counters by the lost motion."""
        if not self.phys_history:
            return self.position_at(t)
        h = np.array(self.phys_history)
        return np.array([np.interp(t, h[:, 0], h[:, 1]), np.interp(t, h[:, 0], h[:, 2])])

    def _note_physical(self, t):
        self.phys_history.append((t, *(self.axis + self.index)))

    def _set_rates_mech(self, r):
        with self.lock:
            now = self._advance()
            self.pending.append((now + self.latency, np.array(r)))
            self._note_physical(now)
            return now, self.mech.copy()

    def _query_mech(self):
        with self.lock:
            now = self._advance()
            self._note_physical(now)
            return now, self.mech.copy()

    def _zero_counters(self):
        with self.lock:
            self._advance()
            self.axis -= self.mech
            self.mech = np.zeros(2)

    def estop(self):
        with self.lock:
            self._advance()
            self.pending.clear()
            self.rate = np.zeros(2)
            self.target = np.zeros(2)
        self.rate_cmd = np.zeros(2)
