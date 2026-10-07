# How it works, and working on it

The control loop, the code layout, the simulator and the tests. Back to the [README](../README.md).
[AGENTS.md](../AGENTS.md) has the handover notes: conventions, traps and the debugging playbook.

## How it works

```
TLE (Celestrak) --> Skyfield --> refracted alt/az --> HA/Dec --+
                                                               |  pointing model (plate-solved
                                                               +-> star alignment) -> axis angles
                                                                          |
                                           cubic-spline trajectory, feed-forward rates  (pass mode)
                                           or rates estimated from the camera           (servo mode)
                                                                          v
 guide cam (17.7 deg) --+                     +--------------------+   R rate1 rate2   +---------+
                        +--> blob centroid -->| offset estimator   |------------------>| Uno ISR |--> DRV8825
 main cam (12.9' x 7.3')+   px -> axis deg    | time offset (along)|<------------------| stepper |
                            via camera matrix | alpha-beta (cross) |   P pos1 pos2     +---------+
                                              +--------------------+
```

* **Inner loop** (50 Hz): rate = feed-forward rate + Kp x (target - step-count position), with a
  stopping-distance limit to avoid overshoot.
* **Outer loop** (every frame): the detected pixel is converted to axis angles with the calibrated
  camera matrix (axis1 column scaled by cos Dec). In pass mode the error against the prediction is
  split into an along-track **time offset** (TLE timing error, the dominant one) and a cross-track
  **axis offset**. In servo mode there is no prediction: the same filter estimates the target's
  position and rate from the camera alone. The guide acquires; the main camera takes over after
  `main_handoff_frames` consecutive detections (when **Main steers** is on) and hands back if it
  loses the target.
* **Pointing model** ([model.py](../issctl/model.py), [align.py](../issctl/align.py)): the mount as a
  two-axis gimbal of unknown orientation - polar axis direction, Dec index and cone error - fitted
  to plate-solved pointings. Goto, sync, sidereal tracking rates and pass planning all go through it.
* **Pass planning** ([predict.py](../issctl/predict.py) `plan_pass`): for each pier side, and each
  turn of axis1, the longest run inside the axis limits and rate limits; scored by the sunlit,
  unobstructed seconds of it. `side=` keeps one side, `after=` ignores what has gone by.
* **Firmware**: Timer1 at 20 kHz, phase-accumulator stepping for both axes, per-axis accel ramps,
  0.5 s watchdog. 1/16 microstepping -> RA 2600 steps/deg, Dec 1300 steps/deg.

## Layout

| Path | What |
|---|---|
| `firmware/issmount/issmount.ino` | Uno firmware (pinout = the original `serialSpeed.ino` wiring) |
| `issctl/cli.py` | the commands; `console` holds the browser actions and the session threads |
| `issctl/predict.py` | TLE fetch, passes, pier-side planning, trajectories, Earth's shadow |
| `issctl/geometry.py` | alt/az <-> HA/Dec <-> mount axes |
| `issctl/model.py`, `align.py` | pointing model; star alignment, sync, star calibration of the guide |
| `issctl/solve.py` | star detection, plate solving (astrometry.net's `solve-field`), brightness |
| `issctl/mount.py` | serial driver + simulated mount |
| `issctl/camera.py`, `detect.py` | ZWO + V4L2 capture threads, blob detection |
| `issctl/calib.py` | camera <-> axis calibration on a point source, boresight |
| `issctl/search.py` | spiral search for a star in the main camera; main calibration on that star |
| `issctl/control.py` | tracking controller (pass and servo mode) |
| `issctl/identify.py` | Identify: name the satellite a session followed, live and afterwards; History |
| `issctl/forecast.py` | Coming up: bright satellites due through the guide field or the sky |
| `issctl/favorites.py` | Favourites: the observed satellites worth coming back to, and their passes |
| `issctl/deepsky.py` | goto by name: Messier, NGC, IC and common names (OpenNGC), CDS online |
| `issctl/mask.py` | sky obstructions; window-frame lines for the chart |
| `issctl/sim.py` | simulated sky for end-to-end testing |
| `issctl/preview.py`, `web/` | browser control panel (`panels.html` holds the page's building blocks) |
| `issctl/ser.py` | SER recorder for the main camera |
| `issctl.sh` | wrapper that runs `issctl` with the project virtualenv, from any directory |

## Simulation (no hardware)

The simulator is the test bench: hardware is often not connected, and anything in the control path
should be exercised here before it goes near the mount.

```bash
./issctl.sh console --sim --web --port 8090   # simulated mount, cameras, a "distant light", a tilted tripod
./issctl.sh track --sim --speed 2 --no-preview
./issctl.sh track --sim --port 8090            # watch it in the browser, real time
./issctl.sh track --sim --sat 42065 --speed 10 # pass mode on another satellite
./issctl.sh track --sim --servo --azimuth-error 90   # servo mode on a tripod turned 90 deg
./issctl.sh track --sim --clouds 3             # unpredicted dropouts
./issctl.sh track --sim --cal-rot-error 45     # deliberately bad camera calibration
./issctl.sh track --sim --record               # exercise the SER writer
```

The simulator injects a 1.5 s TLE timing error, a cross-track offset, 0.35/-0.25 deg pointing error
and an imperfect camera calibration (3% scale, 2 deg rotation). It writes `data/state-sim.json`, so it
never overwrites the real calibration - but sessions in `console --sim` do write logs to `logs/`.
Plate solving is simulated from the known sky; three real guide frames in `tests/data/` exercise the
real solver. At `--speed` much above 10 the loop cannot keep up and runs on prediction - a simulator
artefact.

## Tests

```bash
.venv/bin/python -m pytest -q tests        # ~170 tests, about 2.5 minutes
```

Several exist because a real bug slipped through - the shadow and gate tests, the pole tests, the
27 s leap-second offset, the pier-side turn search. `tests/data/` holds sample orbits (NOSS 3-8,
COSMOS 2226, a Starlink), real guide frames and an OpenNGC sample.
