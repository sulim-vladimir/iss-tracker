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
| `issctl/sim.py` | simulated sky for end-to-end testing |
| `issctl/preview.py` | MJPEG preview at `http://<pi>:8080/` |
| `issctl/ser.py` | SER recorder for the main camera |

## Setup on the Pi 5

```bash
sudo apt install python3-venv
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
# ZWO SDK: copy the arm64 libASICamera2.so to /usr/local/lib, install asi.rules into /etc/udev/rules.d
sudo usermod -aG dialout $USER
cp config.example.toml config.toml   # set site lat/lon/elevation
```

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
2. **Polar align** as well as you can (a few arcmin is plenty; the loop absorbs the rest).
3. `python -m issctl console`
   * park at counterweight-down, tube at pole → `H` (home)
   * `g` goto a bright star (`vega`, `arcturus`, `jupiter`, `moon`, or `18.6 38.8`), centre it in
     the **main** camera with arrows, `s` sync. Watch the preview in a browser.
   * with the star centred in the main camera and tracking on: `c` calibrates both cameras and
     records where the main camera's centre falls on the guide image. Prefer a star at Dec < 60°.
4. `python -m issctl passes` — shows side, trackable seconds, peak rates, and what limits each pass.
5. `python -m issctl track` (next visible pass) or `--pass N`. Recording goes to `captures/*.ser`,
   control log to `logs/track-*.csv`.

Re-run step 3c after changing the Barlow (and update `focal_length_mm` to 1500).

## Simulation (no hardware)

```bash
python -m issctl track --sim --speed 2 --no-preview
python -m issctl console --sim
python -m pytest tests
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
python -m issctl track --sim --clouds 3          # unpredicted gaps
python -m issctl track --sim --config mask.toml  # mapped obstruction
```

## Known limits / next steps

* **Equatorial geometry**: high passes need fast RA rates and can cross axis1 limits. In the sim pass
  only 163 of 322 s were trackable at `max_rate_deg_s = 2`, limited by RA rate (2.04 deg/s peak) and
  `axis1_hour_limit`. Raising the rate to 3 deg/s (7800 steps/s at 1/16) is within the firmware cap;
  check the motors keep torque at ~150 rpm at your supply voltage.
* No backlash compensation yet. The camera loop covers it while tracking in one direction, but
  direction reversals on Dec will show up as a short error transient.
* Camera latency (`latency_s`) and `command_latency_s` should be tuned from real logs.
