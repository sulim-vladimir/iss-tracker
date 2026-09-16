# ISS closed-loop tracker

EQ3-2 (steppers, Arduino Uno + 2x DRV8825) · 150/750 Newtonian + ASI290MC · 9x50 finder + ASI120MM · Raspberry Pi 5.

## How it works

```
TLE (Celestrak) --> Skyfield --> refracted alt/az --> HA/Dec --> mount axis angles
                                                  (pier side chosen per pass)
                                                           |
                                    cubic-spline trajectory, feed-forward rates
                                                           v
 guide cam (1.5 deg FOV) --+                    +--------------------+   R rate1 rate2   +---------+
                           +--> blob centroid ->| offset estimator   |------------------>| Uno ISR |--> DRV8825
 main cam (0.4 deg FOV) ---+   px -> axis deg   | time offset (along)|<------------------| stepper |
                               via calibration  | alpha-beta (cross) |   P pos1 pos2     +---------+
                                                +--------------------+
```

* **Inner loop** (50 Hz): rate = predicted axis rate + Kp × (target − step-count position), with a
  stopping-distance limit to avoid overshoot.
* **Outer loop** (every frame): the detected ISS pixel is converted to axis angles with the
  calibrated camera Jacobian (axis1 column scaled by cos Dec). The error vs prediction is split into
  an along-track **time offset** (TLE timing error, the dominant one) and a cross-track
  **axis offset** (polar/sync/cone error). Guide camera acquires; the main camera takes over after
  3 consecutive detections and hands back if it loses the target.
* **Firmware**: Timer1 at 20 kHz, phase-accumulator stepping for both axes, per-axis accel ramps,
  0.5 s watchdog. 1/16 microstepping → RA 2600 steps/deg, Dec 1300 steps/deg.

## Layout

| Path | What |
|---|---|
| `firmware/issmount/issmount.ino` | Uno firmware (pinout = your `serialSpeed.ino` wiring) |
| `issctl/predict.py` | TLE fetch, passes, pier-side planning, trajectories |
| `issctl/geometry.py` | alt/az ↔ HA/Dec ↔ mount axes |
| `issctl/mount.py` | serial driver + simulated mount |
| `issctl/camera.py`, `detect.py` | ZWO capture threads, blob detection |
| `issctl/control.py` | tracking controller |
| `issctl/calib.py` | camera ↔ axis calibration, guide boresight |
| `issctl/mask.py` | sky obstructions (balcony, window frame, buildings) |
| `issctl/sim.py` | simulated sky for end-to-end testing |
| `issctl/preview.py` | browser preview + control panel at `http://<pi>:8080/` (see below) |
| `issctl/ser.py` | SER recorder for the main camera |
| `issctl/model.py` | pointing model for an unaligned mount - **written but not yet wired in** |
| `issctl.sh` | wrapper that runs `issctl` with the project virtualenv, from any directory |

## Setup on the Pi 5

```bash
sudo apt install python3-venv
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
# ZWO SDK: copy the arm64 libASICamera2.so to /usr/local/lib, install asi.rules into /etc/udev/rules.d
sudo usermod -aG dialout $USER
cp config.example.toml config.toml   # set site lat/lon/elevation
```

Then run everything through `./issctl.sh <command>`, which uses the virtualenv's Python and works
from any directory. Without it you need `source .venv/bin/activate` first, or the dependencies
(skyfield, etc.) will not be found.

**Clock accuracy matters**: 1 s of clock error = up to 1° of along-track error at zenith. Use NTP
(chrony) in the field, or a GPS dongle. The tracker absorbs a few seconds, but acquisition gets harder.

If the CH340 Arduino shows in `lsusb` but no `/dev/ttyUSB0` appears (Ubuntu desktop), `brltty` is
stealing it: `sudo apt remove brltty`.

## Flash firmware

```bash
arduino-cli core install arduino:avr
arduino-cli compile --fqbn arduino:avr:uno firmware/issmount
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno firmware/issmount
```

## Field procedure

1. **Axis directions**: `python -m issctl mount-test` runs each axis ±0.5 deg/s. Check measured
   motion matches (verifies `gear_ratio`/`microsteps`). Set `reverse` flags so that
   axis1 + = sidereal direction and, on the east-looking side, axis2 + = toward the pole.
2. **Polar align** as well as you can - a few arcmin is plenty, the loop absorbs the rest. Some
   alignment is currently **required**: the unaligned-mount pointing model is not wired in yet
   (see [Known limits](#known-limits--next-steps)).
3. `./issctl.sh console --port 8090`
   * park at counterweight-down, tube at pole → `H` (home)
   * `g` goto a bright star (`vega`, `arcturus`, `jupiter`, `moon`, or `18.6 38.8`), centre it in
     the **main** camera with arrows, `s` sync. Watch the preview in a browser.
   * `c` calibrates both cameras — see [Camera calibration](#camera-calibration) below.
4. `./issctl.sh passes` — shows side, trackable seconds, peak rates, and what limits each pass.
5. `./issctl.sh track` (next visible pass) or `--pass N`. Recording goes to `captures/*.ser`,
   control log to `logs/track-*.csv`.

## Camera calibration

This is what lets the tracker turn "the ISS is 200 px up-left in the guide frame" into "move axis1
by -0.02 deg and axis2 by +0.05 deg". Without it the cameras are ignored entirely and the mount
follows prediction alone.

### What you do

1. **Focus both cameras.** Calibration measures geometry, not sharpness, but detection needs a
   compact blob.
2. Start the console and open the preview in a browser:
   ```bash
   ./issctl.sh console --port 8090
   ```
3. **Put a bright point source in the centre of the main camera.** Jog with the arrows while
   watching the preview; adjust exposure/gain (`x` selects a camera, `-`/`=` exposure, `[`/`]` gain)
   until both panels say *detected*.
4. **Press `t`** if you are on a star, so sidereal tracking holds it still. Skip for a fixed
   terrestrial target.
5. **Press `c`.** It takes about a minute and saves to `data/state.json` by itself.

### Use a distant light, not a star

A **fixed distant light** - an antenna beacon, chimney light, a lit window a kilometre or more away -
is far easier than a star, especially from a balcony:

* it does not move, so no tracking and no polar alignment are needed;
* it works at dusk or any night, whatever the sky is doing;
* it is easy to find and bright.

The code only needs a compact bright blob that stays put. It records the declination it calibrated at
(`dec_cal`) and rescales the axis1 column by cos(dec) later, so a terrestrial target stays consistent.
At 750 mm, focus for 1 km differs from infinity by ~0.5 mm of focuser travel: refocus on a star before
imaging, but calibration itself is unaffected.

### What it measures

For each axis ([calib.py](issctl/calib.py)): move -0.08 deg then back (taking up backlash from a
consistent side), average the blob position over ~10 frames, move +0.08 deg, measure again. Pixel
shift over axis move gives one column of the 2x2 matrix; two axes give the whole thing - scale,
rotation and mirror flip at once. Then, with the target centred in the main camera, it computes
**where the main camera's centre falls in the guide image** (the guide boresight), which is what makes
the handoff land the ISS in the small main field. It prints what it found:

```
guide: axis1 +0.08 deg -> [ 66.9 -14.2] px
guide: scale [836.4 835.9] px/deg, rotation 12.0 deg
guide boresight (main camera centre) at [648.2 471.5]
```

### Camera rotation does not matter

Mount the cameras at any angle. The matrix absorbs rotation and flip. Measured tolerance to a *wrong*
calibration, from `--cal-rot-error` in simulation:

| Rotation error | Main camera in control | Median error | Outcome |
|---|---|---|---|
| 0 deg | 39% | 30" | fine |
| 30 deg | 38% | 19" | fine |
| 60 deg | 1% | 1038" | fails to settle |
| 85 deg | 0% | - | never locks |

So precision is not important; doing it at all is. Past 90 deg the correction points the wrong way and
the loop diverges.

### Redo it after

* rotating either camera in its holder, or anything that twists them;
* changing focal length, e.g. adding the 2x Barlow (also set `focal_length_mm = 1500`);
* moving the guide scope relative to the main tube (the boresight changes).

Not needed after moving the tripod, re-homing, or simply starting a new session.

## Browser control panel

`track` and `console` serve a page on `[preview] port` (override with `--port`, e.g. if something
already uses 8080). Both cameras appear side by side, each with:

* the live MJPEG view with the aim cross (green), search gate (blue) and detection (red) - image
  only, no text burned in, so it matches the raw sensor data;
* status text under the frame: local time, seconds relative to track start, alt/az with compass
  point, which camera is steering, TLE time offset, sunlit fraction;
* exposure and gain, as `-`/`+` steps (x1.5, +/-25) or an exact value typed in;
* **Start/Stop recording** under the main frame, since that is the camera it records, with live
  frame and dropped counts. Each start writes a new `captures/iss-*.ser`; recording pauses by itself
  whenever the ISS is in shadow or behind a mapped obstruction.

The same controls exist in the terminal console: `x` selects a camera, `-`/`=` exposure, `[`/`]` gain.

## Simulation (no hardware)

```bash
./issctl.sh track --sim --speed 2 --no-preview
./issctl.sh track --sim --port 8090            # watch it in the browser, real time
./issctl.sh track --sim --clouds 3             # unpredicted dropouts
./issctl.sh track --sim --cal-rot-error 45     # deliberately bad camera calibration
./issctl.sh track --sim --record               # exercise the SER writer
./issctl.sh console --sim
.venv/bin/python -m pytest tests
```

The simulator injects a 1.5 s TLE timing error, a cross-track offset, 0.35/−0.25° pointing error and
an imperfect camera calibration (3% scale, 2° rotation). Current result for a 64° pass:
main camera in control 99% of the time, true error median 12″ / 95th percentile 41″
(main camera half-height is 7.3′).

## Earth's shadow

The ISS is only visible while sunlit, and passes routinely fade out partway through (evening) or
appear partway in (morning). `illumination()` computes a conical umbra/penumbra with an 80 km
absorbing shell, so the fade takes a few seconds, as it does in reality.

* `issctl passes` reports `lit` (sunlit seconds above the horizon), `usable` (trackable **and** lit)
  and when the ISS enters or leaves shadow. Passes are chosen by `usable`, not by altitude.
* While the ISS is in shadow the tracker ignores the cameras entirely and coasts on prediction,
  keeping the learned time offset and axis offsets but freezing their rates. This matters: with the
  ISS invisible, the brightest blob in the frame is a **star**, and following it would drag the mount
  away. On shadow exit the gates reopen and it re-acquires.
* Detections that imply a jump larger than `max_offset_jump_arcmin` are rejected once locked, which
  covers stars, hot pixels and satellites crossing the frame.
* SER recording pauses in shadow.

Coasting accuracy in simulation (165 s of shadow after a 104 s lit segment): median 199", 95th
percentile 693". That stays inside the main camera 69% of the time, and always inside a wide guide
field, so a morning re-acquisition should succeed.

## Obstructions and clouds

Two different problems, handled differently.

**Mapped obstructions** (balcony walls, window frame, neighbouring buildings) go in `config.toml` as
azimuth/altitude rectangles:

```toml
[site.sky]
openings = [[100, 260, 18, 80]]   # a south-facing balcony
blockers = [[168, 176, 0, 90]]    # a window frame post
```

`issctl passes` then reports `blocked` seconds and splits each pass into **usable windows**, and
picks passes by usable time rather than altitude. The tracker coasts through a mapped obstruction
exactly as it does through shadow, and looks again on the far side. To build the mask, point at an
obstruction edge in the console and press `m` - it prints the azimuth/altitude to put in the config.

**Clouds** cannot be predicted, so the tracker keeps following its model and keeps looking. Two
mechanisms stop it locking onto a star in the gap:

* detections that imply a jump beyond `max_offset_jump_arcmin` are rejected, and that limit grows by
  `reacquire_growth_arcmin_per_s` while coasting (the prediction drifts), capped at
  `max_reacquire_arcmin`;
* the search is restricted to a circle around the boresight that grows the same way, instead of
  accepting the brightest thing anywhere in the frame.

Simulated results for the same pass (104 s lit, then shadow):

| Case | Median error while coasting | Inside main FOV |
|---|---|---|
| 3 cloud gaps (4-12 s) | 54" | 100% |
| Mapped building (30 s) | 184" | 100% |
| Earth's shadow (165 s) | 209" | 68% |

```bash
./issctl.sh track --sim --clouds 3                   # unpredicted gaps
./issctl.sh --config mask.toml track --sim           # mapped obstruction
```

## Known limits / next steps

* **`axis1_hour_limit` is now the main constraint.** With `max_rate_deg_s = 3` (7800 steps/s at 1/16,
  ~150 rpm at the motor - check torque at your supply voltage) speed rarely limits a pass, but a pass
  crossing the meridian needs the counterweight high: real passes lose 150-170 s of 370 s at a limit
  of 120 deg. Raising it toward 170 recovers most of that **if the tube clears the tripod and railing**.
* **No polar alignment support yet.** `model.py` has a 5-parameter pointing model (orientation, Dec
  index, cone error) that would let the mount work without polar alignment, but it is **not wired into**
  the planner, mount or tracker. Plate solving was ruled out (twilight passes, restricted balcony view);
  the intended path is an accelerometer for tilt plus azimuth learned from the first satellite crossing.
* **Sensor work (accelerometer/magnetometer) is postponed** - see the conversation notes: an
  accelerometer gives every model parameter except azimuth; a magnetometer is only good for
  repeatability between sessions, not absolute heading, near rebar and stepper magnets.
* A wider guide lens (12-16 mm, 17-23 deg field) would make acquisition far more forgiving than the
  current 9x50 finder (1.5 deg), at 48-64"/px - still ample for handing off to the main camera.
* No backlash compensation yet. The camera loop covers it while tracking in one direction, but
  direction reversals on Dec will show up as a short error transient.
* Camera latency (`latency_s`) and `command_latency_s` should be tuned from real logs.
* Nothing has run against real cameras or a real mount yet: firmware protocol and the Python mount
  driver are verified on the bench, everything else is verified in simulation.
