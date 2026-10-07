# ISS closed-loop tracker

EQ3-2 (steppers, Arduino Uno + 2x DRV8825) · 150/750 Newtonian with a 2x Barlow (1500 mm) + ASI290MC ·
ASI120MM Mini guide camera with a 16 mm lens (15.5 mm measured) · Raspberry Pi 5.

![The console in the browser](docs/console.png)

*The console on 2026-09-30: the guide (left, 17.7° field) locked on a picked object, the main camera
(12.9' x 7.3' field) with a star in view, the mount and calibration controls on the right, and along
the bottom the sky chart, the **Coming up** list of bright satellites due through the guide field,
messages, the calibration summary and the log.*

## What it does

It follows the ISS and other satellites with a small German equatorial mount, on a balcony with a
restricted view and without polar alignment, and records them with the main camera:

* **Star alignment by plate solving.** The guide camera's frames are solved against a star catalogue,
  which calibrates the guide camera and fits a pointing model of the mount - however the tripod
  stands. It reports the polar error as a correction you can make.
* **Two tracking modes.** *Pass mode* follows an orbit prediction and corrects it from the cameras.
  *Servo mode* (**Follow**) follows whatever you click in the guide image, with no orbit at all.
* **Satellite names and plans.** It names what it is following, live and after the session; lists
  the bright satellites due in the next hour; and keeps **Favourites** - the satellites worth coming
  back to - with their visible passes over the next two days.

Everything is driven from a browser page served by the Pi.

## Documentation

| | |
|---|---|
| [Setup](docs/setup.md) | installing on the Pi, camera drivers, firmware, updating the Pi from a laptop |
| [Observing](docs/observing.md) | a night at the telescope step by step; home, star alignment, main camera |
| [Tracking](docs/tracking.md) | pass mode, servo mode (Follow), the pier side, shadow, obstructions and clouds |
| [Satellites](docs/satellites.md) | Identify, naming a pick, History, Coming up, Favourites, Brightness |
| [The console](docs/console.md) | the browser page panel by panel, the sky chart, terminal keys, emergency stop |
| [How it works](docs/development.md) | the control loop, code layout, simulator and tests |
| [AGENTS.md](AGENTS.md) | handover notes: state of play, conventions, debugging playbook |

## Quick start

```bash
# once, on the Pi (details in docs/setup.md)
sudo apt install python3-venv astrometry.net
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
cp config.example.toml config.toml        # your site's coordinates
./issctl.sh solve-setup

# every night
./issctl.sh console --web --port 8090     # open http://<pi>:8090/
```

Then follow [a night at the telescope](docs/observing.md#a-night-at-the-telescope). No hardware at
hand? `./issctl.sh console --sim --web --port 8090` runs the same page on a simulated mount and sky.

## Status

**Verified on the real rig** (2026-09-28 to 10-07): firmware and serial protocol, both cameras,
plate solving (0.2-0.6 s on the Pi), star alignment (10-40" rms), goto and sync through the pointing
model, the spiral search finding a star in the main camera, servo tracks of satellites - NOSS 3-8 (B),
a classified satellite, held on the guide boresight to under a pixel for 80 s - and naming them from
the log afterwards (NOSS 3-8 (B), SENTINEL-6A). Pass mode has run on real passes; the 27 s timing
error they showed (leap seconds) is fixed. **By 2026-10-07, with the guide steering, satellites
appear in the guide frame where the prediction puts them, and stay inside the main camera's
12.9' x 7.3' field for a long time** - Main steers off.

**Verified in simulation only so far**: shadow and obstruction coasting, SER recording.

## Known limits / next steps

* **The main camera does not steer yet.** The guide alone keeps satellites in the main field, and
  with **Main steers** on the mount was not controlled well from main, so it stays off. Not
  investigated yet - a session log with it on would show why.
* **Exposure is set by hand** for both cameras. Planned: automatic guide exposure from the predicted
  brightness, and automatic main exposure for bright satellites.
* **Backlash**: moves that matter (star calibration, spiral search, main calibration, centring)
  approach from one side, but the tracking loop does not compensate. A Dec reversal mid-track shows
  as a short error transient, and servo on a *star* (which needs Dec reversals) can rock in the
  ~10' of Dec slack.
* **`axis1_hour_limit` limits meridian passes and passes near the pole** - see
  [the pier side](docs/tracking.md#the-pier-side).
* **The balcony's view is not mapped**, so "Anywhere" also lists passes behind the wall or buildings
  ([why](docs/tracking.md#obstructions-and-clouds)).
* **USB**: the guide camera sometimes drops off the bus; it is reopened automatically
  ([setup](docs/setup.md#running-it)).
* Camera latency (`latency_s`) and `command_latency_s` should be tuned from real logs.
* Sensors (accelerometer for tilt, magnetometer for repeatability) were considered and postponed:
  the plate-solved alignment does their job.
