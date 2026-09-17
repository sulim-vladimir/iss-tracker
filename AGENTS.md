# Working on this project

Handover notes for whoever (human or agent) picks this up next. `README.md` explains what the
system does and how to run it; this file is about the state of play, the traps, and what to do next.

## What this is

Closed-loop ISS tracking on a Sky-Watcher EQ3-2 driven by steppers: an Arduino Uno runs the axes,
a Raspberry Pi 5 (development happens on the owner's Ubuntu laptop) predicts the pass, drives the
rates and corrects from two cameras. Owner: an astrophotographer in Sumy, Ukraine, observing from a
balcony with a restricted sky view.

Hardware: EQ3-2 (RA worm 130 teeth, Dec 65 - confirmed against several sources), Arduino Uno +
2x DRV8825 on the `serialSpeed.ino` pinout, 150/750 Newtonian + ASI290MC (main), ASI120MM Mini or
Philips SPC900NC + 16 mm M12 lens (guide).

## Ground rules that earned their place

* **The simulator is the test bench.** Hardware is often not connected. `track --sim` runs the whole
  chain; `console --sim` gives a simulated mount, a fixed "distant light" and two brighter decoys.
  Anything you change in the control path, exercise there before claiming it works.
* **Run the tests.** `.venv/bin/python -m pytest -q tests` (34 at the time of writing). Several exist
  because a real bug slipped through: the shadow/gate tests, the pole-degeneracy tests, the SER
  writer tests.
* **Never commit the owner's site.** `config.toml` (real coordinates), `data/state.json`,
  `captures/`, `logs/` are gitignored. `config.example.toml` keeps placeholder coordinates.
  Simulation writes `data/state-sim.json` so it can never overwrite real calibration.
* **Config gets defaults from the example file.** `load_config` layers `config.toml` over
  `config.example.toml`, so adding a key to the example is enough - old configs keep working.
* **Be careful with anything that moves the mount.** Ask before slewing, flashing firmware, or
  deleting captures. The browser page carries an emergency stop; the firmware halts the axes if the
  host goes quiet for 0.5 s.

## State of play (2026-09-18)

Verified on real hardware:

* firmware flashed and answering (`ISSMOUNT 1`), ~4-5 ms round trip, protocol as documented in the
  `.ino` header;
* both ZWO cameras enumerate and stream; the SPC900NC works through the V4L2 driver (pwc, raw 0-255
  exposure register, `gain_automatic` must be forced off);
* axis directions: **both axes needed `reverse = true`**. RA was the last to be found because at
  home the tube sits on the polar axis, where turning RA does not move the pointing at all.

Verified only in simulation: acquisition, guide->main handoff (~17" median error, ISS inside the
main field ~100% of the time), shadow coasting, obstruction masks, cloud gaps, SER recording,
pass planning, click-to-select.

**Open: the Dec axis moves about half of what it is told.** RA measures right (scale check x0.94),
Dec short (x2.25). Pulleys and belts are identical on both axes and the worm counts are confirmed,
so the mechanics match the config. Two live hypotheses:

1. the Dec driver is not in 1/16 - the old sketch only ever used 1/2, 1/4 and 1/8, so **M2 has never
   been driven high on this hardware** and a stuck-high M0 would silently give 1/32 (half motion);
2. the Dec motor skips steps (imbalance, driver current, acceleration), which the step counter
   cannot see.

Next step is the A/B: set `microsteps = 8` on both axes (a mode the old sketch proved works), then
`./issctl.sh axis-scale --axis 2 --move 45` with an inclinometer on the tube.

## Not built yet

* **Unaligned mount support.** `issctl/model.py` has a fitted 5-parameter pointing model
  (orientation, Dec index, cone error) that is **not wired into** the planner, mount or tracker.
  The owner wants to rotate the mount ~90 deg in azimuth to fit a north balcony; measured in
  simulation, tracking survives ~5 deg of azimuth error and fails by 30 deg. The maths is verified:
  from 6 points with 5" noise it recovers a 53.66 deg axis tilt and points to 3.6" median.
  Acceptance test exists: `track --sim --azimuth-error 90` should track as well as an aligned mount.
* **Sensors** (accelerometer for tilt, magnetometer for repeatability) - discussed, postponed.
* Backlash compensation; latency tuning from real logs.

## Debugging playbook

Symptoms we have already chased, so you do not chase them again:

| Symptom | Cause found |
|---|---|
| goto lands at the wrong altitude, azimuth roughly right | an axis direction reversed - invisible at home, only shows once the tube is off the pole |
| calibration scale disagrees with the optics | wrong `focal_length_mm` scales BOTH axes equally; a per-axis difference is drivetrain |
| calibration returns nonsense for axis1 | calibrated near the pole: axis1 rotates the field instead of shifting it (warns now) |
| boresight lands degrees off centre | the two cameras locked onto different objects - click the same one in each (warns now) |
| target ends at the frame edge after calibrating | expected: guide calibration needs degrees of motion; the boresight is taken before any move |
| mount oscillates while jogging | jog rates were recomputed every cycle near the image/axes fallback boundary; they are latched per key press now |
| "cannot restart calibration" | it did restart and failed identically; messages are timestamped now |

Useful when diagnosing: calibration prints the implied focal length beside the measured scale, and
`axis-scale` compares a commanded move against a measured one and computes the corrected
`gear_ratio`.

## Layout

`issctl/`: `predict` (TLE, passes, trajectories), `geometry` (alt/az <-> HA/Dec <-> axes, pose
choice), `mount` (serial + simulated), `camera` (ZWO, V4L2, simulated), `detect`, `calib`,
`control` (the tracker), `mask`, `ser`, `preview` (browser UI), `sim`, `model` (unwired), `cli`.
Firmware in `firmware/issmount/`. Tests in `tests/`.
