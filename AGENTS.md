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

**On the "bimodal acquisition" noted earlier - it looks like a simulator artefact, not a bug.**
An initial batch of four runs of `track --sim --servo --speed 10 --pass 5 --azimuth-error 90`
had two good and two bad. Re-running the identical command six times later gave six good ones
(70-73% of the run under main-camera control, median 7.5" error, 100% inside the main field), so
the earlier split was not reproducible.

Two things point away from a defect in the acquisition path. The bad runs were never "nothing
found" as first written - they show `guide 13%, main 8%`, so the target WAS acquired and then
lost. And the simulator has no unseeded randomness at all: `SimCamera` noise is `default_rng(1)`
over a fixed pool of frames, `random_clouds` takes a seed. The only thing left that can vary
between identical runs is timing - at `--speed 10` a 50 Hz control loop is 500 Hz of real time,
plus two camera threads, and a host that cannot keep up delivers late frames.

So before chasing this in `_vision`: measure whether the loop is keeping up. `Tracker.run`
computes `self.clock.sleep(self.dt - (time.monotonic() - tick) * self.clock.speed)` and simply
sleeps a negative amount when it overruns - counting those overruns would settle it in one run.
That matters for the Pi too, which is slower than the laptop these numbers came from, and where
the same overrun would happen at `--speed 1`. Verify on hardware at real speed before believing
either result.

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

## Calibrating with no point source (scene mode)

From this balcony there is often nothing point-like to calibrate on - just lit windows, or
daylight scenery. `C` in the console, "on scene" in the browser, `mode="scene"` in
`calibrate_cameras`. `FeatureTracker` replaces blob detection with Shi-Tomasi corners followed by
pyramidal Lucas-Kanade, and measures the same displacement-per-degree the ramp wanted.

Blob mode does not fail loudly on such a scene, which is the trap. It locks onto a whole lit
window - thousands of pixels of it - and tracks the wandering centroid of a shape that is
drifting out of frame.

Measured on a real main-camera frame (lit windows, nearly focused), error over one calibration
step in sensor pixels: best single corner 0.45, best 5 0.27, best 20 0.26, all 300 0.37. More is
not better, because weak corners drag the answer down, so it keeps the strongest 15. A
deliberately weak corner gave 3.76 px and lost lock 2 times in 12, which is the case for using
several rather than one - and for letting `goodFeaturesToTrack` choose them, since the strongest
corners frame-wide beat any patch picked by hand.

There is no "scene mode" to be in, and nothing to select before pressing the button: the mode is
an argument to one press. An earlier version took the clicked target's gate as a region to
restrict the search to, which was wrong twice - a click belongs to no mode, and that gate is a
DETECTION gate that re-centres on whatever blob is found inside it, so on scenery the circle
wanders off whatever was picked. `FeatureTracker` still accepts a `region` for the day a scene
with real depth needs one; nothing sets it.

Three things to know:

* **It cannot measure the boresight.** Each camera follows its own scenery, so positions mean
  nothing across cameras. Scene mode keeps the stored boresight and says so. The guide->main
  handoff needs one identifiable point in both cameras, at 1 km or more - closer than that the
  parallax across the ~0.2 m camera separation exceeds the main camera's 7.3' field.
* **Optical flow fails silently.** It reports confidently tracked points it has completely lost:
  in one test it returned 400+ "tracked" points while being 40 px wrong. `_flow` tracks back to
  the reference and discards whatever fails to return to where it started. Do not remove that.
* **Texture is the whole game, and defocus destroys it.** A frame holding only the smooth
  interior of an over-sized window has no recoverable shift at all, by any method - a gradient
  constrains motion only across itself, a single straight edge likewise. At 200 m the main camera
  sees a 75 x 42 cm patch, so a window IS bigger than the frame; what saves it is focus, which
  brings back brick courses and frame edges.

**The simulator cannot exercise this.** `CalibWorld` renders point sources, not scenery -
`goodFeaturesToTrack` finds no corners in it whatsoever and `FeatureTracker.reset()` correctly
returns False. The scene tracker is tested against synthetic textured frames instead. Phase
correlation was tried here too and removed: less accurate, and blind to rotation in a way that
reports a confident translation that never happened.

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
| "guide camera unavailable: General error" | a control value outside what that body accepts - the SDK reports out-of-range exactly like a dead camera. The ASI120MM Mini's gain range is 0-100, not the 0-600 of the USB3 bodies, and it has no HighSpeedMode control at all. `AsiCamera._control` now clamps to `get_controls()` and skips what is missing, and the error names the step that failed |
| "lost the target in main after 1 steps" with a hand-picked target | `select()` gives the pick a gate of only 3% of the frame width (58 px on the main camera), and it can only follow while it keeps detecting. A calibration step is 0.014 deg = 128 px, and at 9 fps a slew crosses 500 px between frames, so the target is outside its own gate before the gate can follow. Clear the pick ("Auto") or calibrate on scene |
| calibration warns "too close to the pole" from a north-facing balcony | the celestial pole sits at alt=latitude due north, so scenery straight north is AT the pole: alt 48.7 az 349 is dec 82.5, where axis1 moves the image 7.7x less than axis2 and the rescaling amplifies the error as much. Point LOW instead - same azimuth at alt 20 is dec 58, at alt 10 is dec 48 |
| cannot calibrate: nothing in view but lit windows or daylight scenery | blob mode needs a point source, and worse, it does not fail loudly - it locks onto a whole lit window and tracks its wandering centroid. Use scene mode (`C` in the console, "on scene" in the browser). Gives J, NOT the boresight |
| the guide boresight silently became the frame centre | `calibrate_cameras` used to write the frame centre into every result and rely on the cross-camera block to fix the guide's. Any run that could not measure one - scene mode always, blob mode whenever a camera did not see the target - destroyed a good stored boresight while logging that it had left it alone. It now carries the stored value forward |
| centring walks AWAY from the boresight, each pass worse than the last | the axis1 column is being STRETCHED: `jacobian` scales it by cos(dec_now)/cos(dec_cal), and near the pole that is a large number, so every correction overshoots by the same factor and the error grows geometrically (seen for real: 70 -> 194 -> 481 px). `axis1_stretch` is that factor and centring refuses above 3x. Note it is exactly 1 until you slew, because both ends come from the same step counter - an unsynced mount cancels itself out |
| centring refuses, saying axis1 disagrees with the mount | it should not - that check was wrong and is gone. On a hand-pushed mount the MOUNT is what is wrong, and the matrix is still exactly right where it was measured. Only two things now block centring: `axis1_plausible` (the axis1 column longer than the axis2 one, which no declination allows - cos(dec) <= 1) and `axis1_stretch` above 3x. `measured_cos_dec` is the image's own estimate of cos(dec), owing nothing to the counters, and calibration warns when it disagrees with them instead of refusing to work |
| no point source anywhere, so the boresight cannot be measured | set it by hand. Simplest: put the object in the middle of the MAIN image, then press "set boresight" on the guide and click that same object - the green cross lands where you clicked and no matrix, mount position or detection is involved. When the object cannot be centred in main, "boresight from my picks" instead: click it in both images and the offset is carried across with the two matrices. A person can match two pictures without a blob detector. Neither needs the mount to know where it is: each camera's J is P . diag(cos dec, 1), so the shared axis-to-sky factor cancels in J_guide . J_main^-1. Parallax is the real limit, not the clicking: 0.2 m between the cameras is 3.4' at 200 m against a 7.3' main field, so use something at a kilometre or more |
| ASI120MM Mini shows up on a USB 2.0 bus in a blue port | expected: the Mini IS a USB 2.0 camera (the -S is the USB3 one). A blue socket wires both a 2.0 and a 3.0 controller; the plug decides which. Not a fault, and not a clue - it shares that bus with the CH340, which matters only for frame rate |

Useful when diagnosing: calibration prints the implied focal length beside the measured scale, and
`axis-scale` compares a commanded move against a measured one and computes the corrected
`gear_ratio`.

## Layout

`issctl/`: `predict` (TLE, passes, trajectories), `geometry` (alt/az <-> HA/Dec <-> axes, pose
choice), `mount` (serial + simulated), `camera` (ZWO, V4L2, simulated), `detect`, `calib`,
`control` (the tracker), `mask`, `ser`, `preview` (browser UI), `sim`, `model` (unwired), `cli`.
Firmware in `firmware/issmount/`. Tests in `tests/`.
