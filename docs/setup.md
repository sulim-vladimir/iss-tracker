# Setup

Installing on the Raspberry Pi 5, the camera drivers, flashing the Arduino, and updating the Pi
from a laptop. Back to the [README](../README.md).

## On the Pi 5

```bash
sudo apt install python3-venv astrometry.net
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
sudo usermod -aG dialout $USER       # serial access to the Arduino (log out and back in)
cp config.example.toml config.toml   # set site lat/lon/elevation - this file is gitignored
./issctl.sh solve-setup              # fetch the star index files plate solving needs (once)
```

`config.toml` is layered over `config.example.toml`, so it only needs the settings you change - and
a new setting added to the example reaches old configs by itself. `solve-setup` downloads the
Tycho-2 index files that match the guide field (4111-4118 for 17.7 deg) into `data/astrometry/`;
after that plate solving works offline.

The satellite catalogues are downloaded into `data/catalog/` on first use and refreshed once a day
(see [Satellites](satellites.md#the-catalogues)). Both folders are gitignored, and so are the
other files that belong to this rig and this site:

| File | What |
|---|---|
| `config.toml` | the site's real coordinates and this rig's settings |
| `data/state.json` | camera matrices, star alignment, mount position, page settings |
| `data/favorites.json` | [Favourites](satellites.md#favourites) |
| `logs/` | one CSV (+ JSON) per tracking session, `console.log` |
| `captures/` | SER recordings |

Simulation writes `data/state-sim.json` instead, so it can never overwrite the real calibration.

## ZWO camera SDK (`libASICamera2.so`)

The cameras need ZWO's closed-source library, which is **not** on PyPI: the `zwoasi` package is only a
wrapper around it. Easiest on Debian/Ubuntu/Raspberry Pi OS, from the INDI PPA:

```bash
sudo add-apt-repository ppa:mutlaqja/ppa   # Raspberry Pi OS: see indilib.org for the repo line
sudo apt install libasi                    # installs libASICamera2.so + udev rules
```

Alternatives, in order of convenience:

* **You may already have it.** FireCapture, INDI and the ZWO desktop apps all ship it - e.g.
  `/opt/FireCapture_v2.7/libASICamera2.so`. `find / -name 'libASICamera2*'` will say.
* **Download from ZWO** (astronomy-imaging-camera.com, "Developers" / ASI Camera SDK), pick the right
  architecture - **arm64 for a 64-bit Pi**, x86-64 for a PC - and copy it to `/usr/local/lib`, then
  `sudo ldconfig`.

The code searches `/usr/local/lib`, `/usr/lib`, the multiarch paths (x86-64 and aarch64) and any
FireCapture install, so usually no config is needed; `[cameras] sdk_lib` overrides the search. If it
cannot find it the error lists everywhere it looked.

**udev rules matter too**: without them the cameras need root, and USB transfers can be capped. The
`libasi` package installs them; with a manual SDK copy `asi.rules` into `/etc/udev/rules.d/` (it is in
the SDK archive) and replug the camera.

The ASI120MM **Mini** is a USB 2.0 camera: its gain range is 0-100 (not the 0-600 of the USB 3
bodies) and it has no HighSpeedMode control. The code clamps every control to what the body
reports, so the start-up messages about both are expected.

## Non-ZWO guide cameras (V4L2)

Set `driver = "v4l2"` on a camera to use any V4L2 device instead - a **Philips SPC900NC** (`pwc`
driver), a UVC webcam, a capture stick. With the same 16 mm lens an SPC900NC (640x480, 5.6 um) gives
**12.8 x 9.6 deg at 72"/px**: a smaller field than the ASI120MM's, still ample for the handoff.

Install `v4l-utils` (`sudo apt install v4l-utils`) so exposure and gain go through `v4l2-ctl`;
without it the code falls back to OpenCV's properties, which many drivers ignore. Control names
differ per driver, so list them and put the right ones in the config:

```bash
v4l2-ctl -d /dev/video0 --list-ctrls        # names, ranges and units
```

The SPC900NC is a **colour** CCD, 2-3x less sensitive than the mono ASI120MM, and its exposure is
capped near 1/25 s unless long-exposure modified - too short for plate solving. The lens also needs
an adapter: the SPC900's thread is not CS.

## Running it

Run everything through `./issctl.sh <command>`, which uses the virtualenv's Python and works from any
directory. The console usually runs headless on the Pi and is used from a browser:

```bash
./issctl.sh console --web --port 8090      # then open http://<pi>:8090/ on a laptop or phone
```

To keep it running after you log out, start it detached, with its output in a log:

```bash
cd ~/iss-tracker
setsid nohup .venv/bin/python -m issctl console --web --port 8090 > logs/console.log 2>&1 < /dev/null &
```

Other commands: `passes`, `favorites`, `identify`, `track` (a headless one-shot session - it would
fight a running console over the serial port and cameras, so run one or the other), `mount-test`,
`axis-scale`, `solve`, `solve-setup`. `./issctl.sh <command> --help` lists the options.

**Clock accuracy matters**: 1 s of clock error = up to 1 deg of along-track error at zenith. Use NTP
(chrony) in the field, or a GPS dongle.

If the CH340 Arduino shows in `lsusb` but no `/dev/ttyUSB0` appears (Ubuntu desktop), `brltty` is
stealing it: `sudo apt remove brltty`.

**USB.** The guide camera sometimes stalls or drops off the bus; it is reopened automatically, and
start-up survives it re-enumerating. Keep the Arduino and the guide camera on separate USB
controllers, keep the guide cable away from the Dec motor, and avoid hubs - a powered hub dropping
out took both cameras with it once.

## Flash firmware

```bash
arduino-cli core install arduino:avr
arduino-cli compile --fqbn arduino:avr:uno firmware/issmount
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno firmware/issmount
```

The firmware answers `ISSMOUNT 1`; the serial protocol is documented in the `.ino` header. It halts
both axes if the host sends nothing for 0.5 s.

## Updating the Pi from a laptop

Development happens on a laptop; the Pi keeps its own checkout in `~/iss-tracker`. Copy the code
over (your `config.toml`, `data/` and `logs/` on the Pi are left alone):

```bash
rsync -a --exclude __pycache__ issctl/ pi@<pi>:iss-tracker/issctl/
```

What it takes for a change to show:

* **Only the page's look** (`issctl/web/style.css`, `app.js`, `panels.html`, `index.html`): these
  are read on every request, so **reload the page**.
* **Anything in Python** - or a page change that calls something new in the console: **restart the
  console**.

To restart, stop it gracefully and wait for it to go:

```bash
ps -eo pid,args | grep "[p]ython -m issctl console"     # its pid
kill -TERM <pid>                                         # saves the position, closes the cameras
```

Make sure it has gone before starting another: **two consoles on the serial port steal each
other's replies.** Shutting down can take up to half a minute, and the ZWO library may print
`terminate called without an active exception` on the way out - the position was already saved, so
wait for it rather than `kill -9`. Then start it again (detached, as above) and check that both
cameras deliver frames: the guide camera tends to drop off USB when the console releases it and
comes back within ~20 s.
