# ISS closed-loop tracker

EQ3-2 (steppers, Arduino Uno + 2x DRV8825) · 150/750 Newtonian + ASI290MC · ASI120MM with a 16 mm lens · Raspberry Pi 5.

## How it works

```
TLE (Celestrak) --> Skyfield --> refracted alt/az --> HA/Dec --> mount axis angles
                                                 (pier side chosen per pass)
                                                           |
                                    cubic-spline trajectory, feed-forward rates
                                                           v
 guide cam (17 deg FOV) ---+                    +--------------------+   R rate1 rate2   +---------+
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
| `issctl/camera.py`, `detect.py` | ZWO + V4L2 capture threads, blob detection |
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
sudo usermod -aG dialout $USER       # serial access to the Arduino (log out and back in)
cp config.example.toml config.toml   # set site lat/lon/elevation - this file is gitignored
```

### ZWO camera SDK (`libASICamera2.so`)

The cameras need ZWO's closed-source library, which is **not** on PyPI: the `zwoasi` package is only a
wrapper around it. Easiest on Debian/Ubuntu/Raspberry Pi OS, from the INDI PPA:

```bash
sudo add-apt-repository ppa:mutlaqja/ppa   # Raspberry Pi OS: see indilib.org for the repo line
sudo apt install libasi                    # installs libASICamera2.so + udev rules
```

Alternatives, in order of convenience:

* **You may already have it.** FireCapture, INDI and the ZWO desktop apps all ship it - e.g.
  `/opt/FireCapture_v2.7/libASICamera2.so` on this laptop. `find / -name 'libASICamera2*'` will say.
* **Download from ZWO** (astronomy-imaging-camera.com, "Developers" / ASI Camera SDK), pick the right
  architecture - **arm64 for a 64-bit Pi**, x86-64 for a PC - and copy it to `/usr/local/lib`, then
  `sudo ldconfig`.

The code searches `/usr/local/lib`, `/usr/lib`, the multiarch paths (x86-64 and aarch64) and any
FireCapture install, so usually no config is needed; `[cameras] sdk_lib` overrides the search. If it
cannot find it the error lists everywhere it looked.

### Non-ZWO guide cameras (V4L2)

Set `driver = "v4l2"` on a camera to use any V4L2 device instead - a **Philips SPC900NC** (`pwc`
driver), a UVC webcam, a capture stick. With the same 16 mm lens an SPC900NC (640x480, 5.6 um) gives
**12.8 x 9.6 deg at 72"/px**, against 17.2 x 12.9 deg at 48"/px for the ASI120MM: a slightly smaller
field, still far more forgiving than the 9x50 finder, and ample for the handoff (a bright ISS
centroids to ~0.2 px, about 14").

Install `v4l-utils` (`sudo apt install v4l-utils`) so exposure and gain go through `v4l2-ctl`;
without it the code falls back to OpenCV's properties, which many drivers ignore. Control names
differ per driver, so list them and put the right ones in the config:

```bash
v4l2-ctl -d /dev/video0 --list-ctrls        # names, ranges and units
```

Two caveats: the SPC900NC is a **colour** CCD, so it is 2-3x less sensitive than the mono ASI120MM
(fine for the ISS, harder for faint calibration stars), and its exposure is capped near 1/25 s
unless long-exposure modified. The lens also needs an adapter - the SPC900's thread is not CS.

**udev rules matter too**: without them the cameras need root, and USB transfers can be capped. The
`libasi` package installs them; with a manual SDK copy `asi.rules` into `/etc/udev/rules.d/` (it is in
the SDK archive) and replug the camera.

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

1. **Axis directions**: `./issctl.sh mount-test` runs each axis ±0.5 deg/s. Check measured
   motion matches (verifies `gear_ratio`/`microsteps`). Set `reverse` flags so that
   axis1 + = sidereal direction and, on the east-looking side, axis2 + = toward the pole.
   Do this **before** homing or any goto: it moves only ~1 deg each way, so it is the safe way to
   find out which way the motors actually turn.
2. **Polar align** as well as you can - a few arcmin is plenty, the loop absorbs the rest. Some
   alignment is currently **required**: the unaligned-mount pointing model is not wired in yet
   (see [Known limits](#known-limits--next-steps)).
3. `./issctl.sh console --port 8090`
   * put the scope in the [home position](#starting-position-home) → `H` (or **set home**)
   * `g` goto a bright star (`vega`, `arcturus`, `jupiter`, `moon`, or `18.6 38.8`), centre it in
     the **main** camera with arrows, `s` sync. Watch the preview in a browser.
   * `c` calibrates both cameras — see [Camera calibration](#camera-calibration) below.
4. `./issctl.sh passes` — shows side, trackable seconds, peak rates, and what limits each pass.
5. `./issctl.sh track` (next visible pass) or `--pass N`. Recording goes to `captures/*.ser`,
   control log to `logs/track-*.csv`. Keep the browser page open: it carries the
   [emergency stop](#emergency-stop).

## Starting position (home)

**Counterweight straight down, tube parallel to the polar axis** (pointing at the celestial pole).
That pose is what the software calls axis1 = 0, axis2 = 90, and everything else is measured from it.

Each session:

1. Put the mount in that pose by hand - a degree or two out is fine.
2. Start `console` and press **set home** (`H`).
3. Later, **sync** on a star, planet, the Moon or a distant light to remove what is left.

**Why every session:** the Uno resets when the serial port opens, so its step counters always start at
zero. The driver then adds the offset saved in `data/state.json`, which effectively assumes *the scope
is still where you last homed it*. If the tube was moved by hand in between, the program starts out
believing a stale position. So either park back at home before quitting, or press **set home** at the
start with the scope physically at home.

**How exact?** Not very. Three layers absorb the error: `sync` on a known object, the camera loop once
the ISS is acquired, and a wide search gate at first acquisition. What home *does* need to be is
roughly right, so that the first slew goes the right way and `axis1_hour_limit` means what it says -
a home that is 90 deg out can swing the tube into the tripod or the railing.

**Before any large move**, check cable slack and clearance, and keep the browser page open: it carries
the [emergency stop](#emergency-stop).

## Camera calibration

This is what lets the tracker turn "the ISS is 200 px up-left in the guide frame" into "move axis1
by -0.02 deg and axis2 by +0.05 deg". Without it the cameras are ignored entirely and the mount
follows prediction alone.

### What you do

Everything below can be done **entirely from the browser** - jog, home, sync, goto, calibrate - so
you never need the terminal outside. `--web` skips the curses UI altogether:

```bash
./issctl.sh console --web --port 8090      # then open http://<pi>:8090/ on a laptop or phone
```

1. **Focus both cameras.** Calibration measures geometry, not sharpness, but detection needs a
   compact blob.
2. **Put a bright point source in the centre of the main camera**, using the jog arrows in the mount
   panel while watching the image. Adjust exposure/gain per camera until both panels say *detected*.
3. **Turn sidereal tracking on** if you are on a star, so it holds still. Skip for a fixed
   terrestrial target.
4. **Press "calibrate cameras".** It takes about a minute; the panel shows progress, then the
   resulting scale, rotation and boresight. Results save to `data/state.json` automatically
   (simulation writes `data/state-sim.json` instead, so it never overwrites real calibration).

The terminal console does the same with keys: arrows jog, `t` sidereal, `c` calibrate, `s` sync,
`g` goto, `H` home, `m` mask point.

**Avoid calibrating near the pole.** Axis1's effect on the image shrinks with cos(dec), so near
axis2 = 90 deg the measurement degenerates and the matrix becomes ill-conditioned. Pick a target well
away from the celestial pole - a terrestrial light is usually fine by definition.

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

For each axis ([calib.py](issctl/calib.py)): move one step negative then back (taking up backlash
from a consistent side), average the blob position over ~10 frames, move one step positive, measure
again. Pixel shift over axis move gives one column of the 2x2 matrix; two axes give the whole thing -
scale, rotation and mirror flip at once.

**The step size is chosen per camera**, aiming to shift the target ~20% of that camera's frame
height. One size cannot serve both: 0.08 deg is 375 px in the main camera but only 6 px in a 16 mm
guide. In practice it uses ~2.6 deg for the guide and ~0.05 deg for the main camera.

Then, with the target centred in the main camera, it computes
**where the main camera's centre falls in the guide image** (the guide boresight), which is what makes
the handoff land the ISS in the small main field. It prints what it found:

```
guide: axis1 +2.578 deg -> [144.3  30.7] px
guide: axis2 +2.578 deg -> [-39.9 187.9] px
main:  axis1 +0.049 deg -> [167.2 -20.5] px
guide: 74.5 px/deg, rot 12.0 deg, boresight [639.5, 479.5]
main: 4510.3 px/deg, rot -7.0 deg, boresight [967.5, 547.5]
```

Verified in simulation, where the true values are known: it recovered 74.5 px/deg / 12.0 deg for the
guide and 4510 vs 4514 px/deg / -7.0 deg for the main camera. You can repeat that check yourself:

```bash
./issctl.sh console --sim --web --port 8090   # simulated mount + a fixed "distant light"
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

**One session can do the whole evening.** `console` has a **Track next pass** button: it plans the next
usable pass (or an index from `passes`), hands the mount to the tracker, and switches the page to
tracking mode - the jog/goto/calibrate controls grey out, the sky chart shows the pass and a countdown,
and recording starts if configured. **Stop tracking** gives the mount back. The separate
`./issctl.sh track` command still exists for a headless one-shot run, but the button avoids juggling
two processes that would fight over the serial port and cameras.

`console` adds a **mount panel**: a cross-shaped jog pad with a speed selector, sidereal tracking on/off,
goto/sync by target name, set home, calibrate cameras, and a mask-point readout - plus live axis
angles, alt/az and the current calibration. With `--web` there is no terminal UI at all, which suits
a phone at the mount. `track` shows the camera panels only; jogging mid-pass is not offered on purpose.

### Emergency stop

A red **EMERGENCY STOP** sits at the top of the page in both `console` and `track`, and `X` does the
same in the terminal console. Unlike the pad's `stop` button (which only drops the jog), it:

* sends the firmware's `X` command - rates to zero immediately, with no deceleration ramp;
* cancels sidereal tracking and any jog;
* aborts a running goto, sync or calibration, and abandons a pass in `track`;
* stays latched until your next deliberate command.

Verified mid-calibration in simulation: the axis froze instantly and the calibration unwound with
`calibration aborted` a few seconds later; jogging worked again afterwards.

Because it skips the ramp it **can lose steps**, so re-sync (or re-home) before trusting the position
afterwards. Two other safety nets exist: the firmware halts both axes if no command arrives for 0.5 s
(so a crashed Pi or unplugged USB stops the mount), and `Ctrl-C` stops motion on the way out.

**The arrows follow the image, not the axes.** Pick the frame in the *arrows* selector (`f` cycles it
in the terminal): with `guide` or `main` selected, pressing right moves the target right in that
camera's picture and up moves it up, whatever the camera's rotation, because the calibration matrix
converts the screen direction into axis rates. `axes` drives axis1/axis2 raw, which is the only option
before the cameras are calibrated. Verified in simulation with a camera rotated 12 deg: right/left move
the target purely in x, up/down purely in y.

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
* The guide camera runs a 16 mm lens (17.2 x 12.9 deg at 48"/px), chosen so acquisition tolerates
  several degrees of pointing and TLE error. Centroiding a bright ISS to ~0.2 px is ~10", far finer
  than the main camera's 7.3' half-height, so the handoff still lands it.
* No backlash compensation yet. The camera loop covers it while tracking in one direction, but
  direction reversals on Dec will show up as a short error transient.
* Camera latency (`latency_s`) and `command_latency_s` should be tuned from real logs.
* Nothing has run against real cameras or a real mount yet: firmware protocol and the Python mount
  driver are verified on the bench, everything else is verified in simulation.
