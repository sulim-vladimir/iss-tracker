# The console

The browser page, panel by panel, the terminal keys, and the emergency stop. Back to the
[README](../README.md).

`console` (and `track`) serve the page on `[preview] port`, or `--port`. With `--web` there is no
terminal UI at all, which suits a Pi with no screen and a phone at the mount:

```bash
./issctl.sh console --web --port 8090      # then open http://<pi>:8090/
```

## Camera panels

Each camera panel has the live view with the aim cross, the search gate (blue circle) and the
detection (red circle) - image only, no text burned in. On the guide the **green cross** marks the
boresight - where the main camera looks - and a **grey cross** the frame centre; on main the grey
cross is its frame centre, which is its aim point.

Under the image:

* **Exposure** and **Gain**, as `-`/`+` steps or an exact value; remembered across restarts.
* **Target**: click the image to pick an object; **Auto** goes back to the brightest in frame. The
  line says *locked on your pick*, *your pick - nothing there* or *brightest in frame*. Double-click
  centres the object.
* **Follow** (guide) - servo mode, see [Tracking](tracking.md#servo-mode-follow).
* **Main steers** (main) - may main take over steering, see [Tracking](tracking.md#pass-mode).
* **Set boresight** and **Marks on/off** (guide) - move the green cross; hide or show both crosses
  (works during tracking too).
* **Stretch: frame / sky** - how the picture is brightened. *Frame* makes the brightest 0.1% of the
  frame white, so lit windows coming into view turn the sky and stars dark, as if the exposure had
  dropped (it has not - exposure is always manual). *Sky* sets white from the sky background and its
  noise, so the view holds still and windows just saturate. The picture only: detection, solving
  and recording use the raw frames. Works during tracking and is remembered.
* **Brightness** (guide) - measure a star's or a satellite's magnitude, see
  [Satellites](satellites.md#brightness).
* **Frame line** (guide) - press it, then click two points along a window-frame edge in the guide
  image: the edge is carried onto the sky through the guide calibration and the star alignment, and
  saved as an orange line on the sky chart. For a long edge add several lines from different
  pointings. A picture only - it does not stop tracking.
* **Send to main** (guide) moves the object onto the green cross, where main looks; **Centre it**
  (main) onto main's centre; **Centre in frame** (guide) onto the guide's own frame centre.
* **Start/Stop recording** (main) - each start writes a new `captures/iss-*.ser`. Recording pauses by
  itself while the target is in shadow or behind a mapped obstruction; it is never started or
  stopped for you.

## Slew

The arrow pad with a speed selector and **stop**, which drops the jog or cancels a goto, sync or
calibration with a normal deceleration. The *Arrows* selector starts on **axes**, which drives
axis1/axis2 directly; with `guide` or `main` selected, right moves the target right in that camera's
picture and up moves it up, whatever the camera's rotation, through its matrix. The red
**EMERGENCY STOP** is here too - see [below](#emergency-stop).

## Target & tracking

* **goto** / **sync** by name: stars and planets (`vega`, `jupiter`, `moon`), any Messier, NGC or IC
  object or its common name (`M31`, `NGC 7000`, `IC 434`, `Orion Nebula`, `whirlpool` - OpenNGC,
  downloaded once into `data/catalog/`), anything else CDS knows while online (`HD 209458`),
  coordinates (`18.6 38.8`, `18:36:56 +38:47:01`) or `altaz ALT AZ`.
* **go home** - slew to home (counterweight down, tube at the pole); tracking stops.
* **centre by solve** - put the named object on the green cross using the plate solve.
* **sidereal on/off**, **Keep pier side** ([why](tracking.md#the-pier-side)), **Identify** and
  **naming on/off** ([Satellites](satellites.md#identify)), **motors off**.

## Calibration

**Set home** and the star tools - **Solve**, **Sync on stars**, **Calibrate on stars**, **Add
star**, **Spiral search in main**, **Calibrate main on star**, **Boresight on star**, **Clear
alignment** - all described in [Observing](observing.md). Under them a summary: when it was
calibrated, each camera's scale and rotation, the star alignment and the polar error.

## Sky chart

Zenith in the middle, the horizon at the rim, north up and east to the right.

| Mark | Meaning |
|---|---|
| yellow dot and ring | where the mount points, and the guide field round it |
| red dot | where the tracker's target is now |
| cyan / grey / red line | the pass being tracked: sunlit / in Earth's shadow / behind a mapped obstruction |
| violet line | the pass picked in Coming up |
| gold line | the pass picked in Favourites |
| green line | the object just named live, past and future |
| orange lines | window-frame lines (Frame line) |
| dashed circle | `min_altitude` |
| green / red sectors | mapped sky openings / blockers |
| yellow cross | a point clicked on the chart |

On a path, a filled dot marks where the satellite is now, a hollow circle where it will come in.

Under the chart:

* **Go to point** - click a point on the chart first; the mount slews there and holds that altitude
  and azimuth still (sidereal off) - a place to wait for a satellite.
* **Track it** - follow the object just named on its orbit (pass mode).
* **Frame on/off** and **Undo line** - show or hide the frame lines; remove the last one. Works
  during tracking.
* the tracking state (countdown to the pass, or time left), the mount's alt/az and the axis rates.

## Lists and messages

**Coming up**, **Favourites** and **History** are described in [Satellites](satellites.md).
**Messages** shows the latest message, the **Log** everything with times (**copy** puts it on the
clipboard), **Warnings** what the last calibration found doubtful.

**One session can do the whole evening.** **Track selected** hands the mount to the tracker and
switches the page to tracking mode: the jog/goto/calibrate controls refuse ("tracking a pass - stop
it first"), the sky chart shows the pass and a countdown, and the button becomes **Stop tracking**,
which gives the mount back (so does the slew pad's **stop**). Display switches - Marks, Stretch,
Frame, Keep pier side, Main steers - work throughout. `./issctl.sh track` still exists for a headless
one-shot run, but it would fight the console over the serial port and cameras, so run one or the
other.

## Terminal console

Without `--web` the console also runs in the terminal, with the same controls on keys:

| Key | | Key | |
|---|---|---|---|
| arrows | jog | `p` | track the next ISS pass |
| `t` | sidereal on/off | `v` | servo (Follow) |
| `g` | goto | `f` | arrow frame (axes / guide / main) |
| `s` | sync | `x` | switch camera |
| `H` | set home | `-` `=` | exposure |
| `c` | calibrate on target | `[` `]` | gain |
| `S` | solve | `m` | print the current alt/az |
| `Y` | sync on stars | `X` | emergency stop |
| `K` | calibrate on stars | `q` | quit |
| `A` | add star | | |
| `B` | boresight on star | | |

## Emergency stop

A red **EMERGENCY STOP** sits in the slew panel in both `console` and `track`, and `X` does the same
in the terminal console. Unlike the pad's **stop**, it:

* sends the firmware's `X` command - rates to zero immediately, with no deceleration ramp;
* cancels sidereal tracking and any jog;
* aborts a running goto, sync, calibration or search, and abandons a pass;
* stays latched until your next deliberate command.

Because it skips the ramp it **can lose steps**, so **Sync on stars** before trusting the position
afterwards. Two other safety nets exist: the firmware halts both axes if no command arrives for 0.5 s
(so a crashed Pi or unplugged USB stops the mount), and `Ctrl-C` or `kill -TERM` stops motion and
saves the position on the way out. The console's keepalive survives a garbled serial reply rather
than dying and leaving the mount stopped.
