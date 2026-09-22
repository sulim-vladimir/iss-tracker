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

## Two tracking modes, one loop

`control.py` runs the same loop against two references, and which one it gets is the only
difference between the modes:

* **pass mode** follows a planned `Trajectory`. The feed-forward comes from the orbit, so the
  mount is already moving at nearly the right rate before the cameras see anything. It needs the
  mount's orientation to be right to within a few degrees.
* **servo mode** follows a `FreeRun`: position frozen, velocity zero. `cross`/`cross_rate` then
  stop being a correction and become the whole estimate of where the target is and how fast it
  moves, so the feed-forward comes from the camera. It needs no orbit, no site and no alignment -
  only the camera calibration. Point the tube at the ISS by hand, click it in the guide image,
  and the loop keeps it on the boresight.

Both are reachable from the one app: in the console press `p` for the next pass or `v` for servo
(browser actions `track` and `servo`), and headless via `track` / `track --servo`.

**Servo mode is what makes the north balcony work**, because it does not care which way the
tripod faces. Pass mode on a mount rotated 90 deg in azimuth is 78 DEGREES off; servo mode on the
same setup, when it locks, holds the ISS to a median 5-7" on the main camera, 100% of the time
inside the main field, with ~75% of the run under main-camera control.

**"When it locks" is the open problem.** Servo mode is bimodal: the same command, run four times
(`track --sim --servo --speed 10 --pass 5 --azimuth-error 90`), locked twice and never acquired
at all the other twice - 80% of the run in `predict`, meaning nothing was ever found. There is no
middle outcome, which says this is an ACQUISITION failure, not a tracking one. Suspects, in
order: the first-fix path in `_vision` (in the simulator nothing clicks the target, so there is
no `force_accept` and the first detection has to pass the gate on its own); the initial search
gate when `last_good` is still `-inf`; and a race between the first `mount.query()` seeding the
reference and the first camera frame. Fix this before trusting servo mode on real hardware -
everything below it is measured on the runs that did lock, so it is all conditional on this.

What servo mode gives up: it cannot know in advance that a pass is reachable, sunlit or clear of
the window frame. `servo_window()` in `cli.py` answers that from the TLE when one is available -
illumination and altitude need no alignment - and `Tracker._limit_guard` is the live backstop.

Two things it is genuinely sensitive to, measured:

* **calibration rotation and scale.** 15% scale and 10 deg of rotation together are fine (100% in
  the main field). 30% scale loses the handoff entirely - the guide holds it, the main never sees
  it. The control law tolerates far more than that; what breaks first is the handoff and the
  search gate.
* **its own gains.** Pass mode's `cross_alpha`/`cross_beta` smooth a residual; in servo mode the
  same two numbers carry the entire motion. Sharing them cost a factor of five in accuracy
  (125" -> 23"), which is why `servo_alpha`/`servo_beta` exist.

## Not built yet

* **The fitted pointing model.** `issctl/model.py` has a 5-parameter model (orientation, Dec
  index, cone error) that is still **not wired into** the planner, mount or pass mode. Servo mode
  made it unnecessary for tracking, but it is still what would let the PLANNER say whether a pass
  is reachable before you go outside, and what `sync` should feed instead of shifting the index.
  The maths is verified: from 6 points with 5" noise it recovers a 53.66 deg axis tilt and points
  to 3.6" median. Two well-separated points are enough to fix the orientation, and the ISS itself
  supplies them - but a single-pass fit absorbs the TLE timing error into the orientation, so do
  not persist one as the mount's alignment.
* **Sensors** (accelerometer for tilt, magnetometer for repeatability) - discussed, postponed.
* Backlash compensation; latency tuning from real logs.

## Debugging playbook

Symptoms we have already chased, so you do not chase them again:

| Symptom | Cause found |
|---|---|
| goto lands at the wrong altitude, azimuth roughly right | an axis direction reversed - invisible at home, only shows once the tube is off the pole |
| calibration scale disagrees with the optics | wrong `focal_length_mm` scales BOTH axes equally; a per-axis difference is drivetrain |
| implied focal length looks wrong | it conflates three things: true focal length, axis scale, and target distance (indoors the camera's radius from the axis over the target range is several percent). The telescope IS 750 mm; trust a star-based calibration, not an indoor one |
| calibration returns nonsense for axis1 | calibrated near the pole: axis1 rotates the field instead of shifting it (warns now) |
| boresight lands degrees off centre | the two cameras locked onto different objects - click the same one in each (warns now) |
| target ends at the frame edge after calibrating | expected: guide calibration needs degrees of motion; the boresight is taken before any move |
| mount oscillates while jogging | jog rates were recomputed every cycle near the image/axes fallback boundary; they are latched per key press now |
| "cannot restart calibration" | it did restart and failed identically; messages are timestamped now |
| servo run slews away to the home pose instead of holding | the reference was seeded from `mount.last`, which is the placeholder home position until the driver has queried once. Seed from `mount.query()` |
| servo run never ends when nothing is in the field | the give-up check only fired after a first lock; it now runs from the start of the session |
| servo loses the target after a one-second glitch and never gets it back | the search gate grew at the pass-mode rate and was still opening when `servo_give_up_s` fired. In servo mode it opens at roughly the ISS's own speed |
| servo tracks but the main camera never takes over | calibration scale error. 15% is fine, 30% is not - the handoff, not the control law, is what gives out |

Useful when diagnosing: calibration prints the implied focal length beside the measured scale, and
`axis-scale` compares a commanded move against a measured one and computes the corrected
`gear_ratio`.

## Layout

`issctl/`: `predict` (TLE, passes, trajectories), `geometry` (alt/az <-> HA/Dec <-> axes, pose
choice), `mount` (serial + simulated), `camera` (ZWO, V4L2, simulated), `detect`, `calib`,
`control` (the tracker), `mask`, `ser`, `preview` (browser UI), `sim`, `model` (unwired), `cli`.
Firmware in `firmware/issmount/`. Tests in `tests/`.
