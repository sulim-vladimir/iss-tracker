# Observing

A night at the telescope, step by step, and the calibrations behind each step. Back to the
[README](../README.md).

## A night at the telescope

No polar alignment is needed - the star alignment measures however the tripod stands. Each step is
a button in the browser; the sections below explain them.

1. **Once, for a new build: axis directions.** `./issctl.sh mount-test` runs each axis +/-0.5 deg/s.
   Check the measured motion matches (verifies `gear_ratio`/`microsteps`), and set the `reverse`
   flags so that axis1 + is the sidereal direction and, on the east-looking side, axis2 + is toward
   the pole. It moves only ~1 deg each way, so it is the safe way to find out which way the motors
   turn. (On this rig both axes need `reverse = true`.)
2. **Home.** Put the tube in the [home position](#starting-position-home) and press **Set home** -
   or, if the console was only restarted and nobody touched the tube, skip this.
3. **Point at stars.** **goto** a star away from the pole (e.g. `mizar`, `vega`), turn **sidereal**
   on, set the guide exposure to 0.5-2 s, then **Sync on stars**.
4. **Calibrate the guide and align the mount: Calibrate on stars.** Then do it again somewhere at
   least 20 deg away and look for **AGREE** in the messages - see
   [Star alignment](#star-alignment-and-the-guide-camera).
5. **Main camera.** goto a bright star, **Centre by solve**, then **Spiral search in main** if the
   main camera does not see it, and **Calibrate main on star** - see [Main camera](#main-camera).
6. **Guide boresight.** With the star in the middle of the main image, press **Set boresight** on
   the guide and click the star (the click snaps to the nearest bright spot) - or **Boresight on
   star**.
7. **Track.** Pick a pass in **Coming up** or **Favourites** and press **Track selected**, or click
   something in the guide image and press **Follow** - see [Tracking](tracking.md). Start and stop
   recording yourself. Keep the page open: it carries the
   [emergency stop](console.md#emergency-stop). Recording goes to `captures/*.ser`, the control log
   to `logs/track-*.csv` or `logs/servo-*.csv`.

The calibration and the star alignment are kept in `data/state.json` and survive restarts. Steps
4-6 are needed again only when something changes - see [when to redo](#when-to-redo-what).

## Starting position (home)

**Counterweight straight down, tube parallel to the polar axis** (pointing at the celestial pole).
That pose is what the software calls axis1 = 0, axis2 = 90, and everything else is measured from it.

1. Put the mount in that pose by hand - a degree or two out is fine.
2. Press **Set home** (`H` in the terminal console). It does not move the mount.
3. **Sync on stars** removes what is left.

**Why it matters:** the Uno resets when the serial port opens, so its step counters always start at
zero. The console saves the position every few seconds and restores it on start-up
(`position restored: ... - re-home if the mount was moved by hand`), so a restart is fine as long as
nobody moved the tube. If someone did, press **Set home** with the tube at home, or simply
**Sync on stars** from wherever it points: sync re-indexes the counters through the pointing model
and keeps the alignment. **Go home** is the other way round: it slews the mount back to home.

**How exact?** Not very. What home needs to be is roughly right, so that the first slew goes the
right way and `axis1_hour_limit` means what it says - a home that is 90 deg out can swing the tube
into the tripod or the railing. **Before any large move**, check cable slack and clearance.

## Star alignment and the guide camera

The guide camera sees 17.7 x 13.3 deg at 49.9"/px, which always holds enough stars to plate-solve.
That one fact does three jobs.

**Solve** plate-solves the current guide frame and says where the tube really points and what is in
view. **Sync on stars** sets the mount's counters from that. The detector
([solve.py](../issctl/solve.py) `find_stars`) smooths the frame by about a star's width first - on a
1 s, 8-bit guide frame the single-pixel noise otherwise outnumbers the faint stars - and rejects lit
windows and walls by their shape, so a frame with buildings in it still solves.

**Calibrate on stars** measures the guide camera's matrix and aligns the mount in one go: it solves
a frame, moves one axis by `star_cal_step_deg` (1 deg), solves again, and does the same for the other
axis. Each frame is solved on its own, so nothing has to stay in view between moves. From those
solves it gets:

* the guide matrix (scale, rotation, flip), and the focal length the stars imply - it warns if that
  differs from `focal_length_mm` (the 16 mm lens measured 15.5 mm);
* the real angle each axis turned against what the counters said (the RA drive gives ~3.5% short
  measure; the report flags anything over 3%);
* four alignment points and the direction of both turning axes, which fit the pointing model:
  polar axis direction, Dec index and cone error. Cone is only fitted once points span 30 deg of sky.

It prints the polar error of the mount as a correction:

```
guide: this run alone: RA axis 1.00 deg below the pole and 2.00 deg east of it: raise it 1.00, turn it 2.00 west (on the sky)
guide: vs the previous run 28 deg away (independent check): polar axis 0.02 deg apart, camera rotation +0.08 deg, scale +0.00% - AGREE
```

**Repeat it at least 20 deg away.** Each run is compared with the previous one: the polar axis must
agree to 0.5 deg, the camera rotation to 0.5 deg and the scale to 1%, otherwise the messages say
DISAGREE and why. One run's polar axis rests on a ~1 deg turn, so it is only as good as the solves:
about 0.1 deg with 2" solves, 0.6 deg with 10" (simulation). A bigger `star_cal_step_deg` tightens
it, where the window allows. Runs closer than 20 deg only show repeatability.

**Stay away from the pole** (Dec above ~60 deg): axis1 there rotates the field instead of shifting it.
From a north-facing balcony that means pointing low - the same azimuth at alt 20 is Dec ~58.

Other star tools: **Add star** adds the current pointing as another alignment point; **Clear
alignment** forgets the model - only after the tripod itself moved. **Centre by solve** puts a named
object on the guide boresight using the plate solve rather than the counters, so it works however
little the mount knows where it is.

## Main camera

With the Barlow the main camera sees only 12.9' x 7.3' - about 9 x 16 guide pixels - so getting a
star into it and measuring it needs its own tools.

**Spiral search in main** walks a square spiral round the current pointing, one main field per step
(4.6'), out to `search_radius_deg` (30'), pausing `search_dwell_s` (1.5 s) at each stop. **You decide
when it has found the star**: the button turns into **Stop here** while it runs - press it when the
bright star is in the main image and the mount stays at that stop. Every stop is approached from the
same side, so the ~10' of Dec backlash cannot leave holes; left alone, it covers the whole square and
returns to the start.

**Calibrate main on star** measures the main camera's matrix in its own pixels: each axis goes to
-1.8', 0 and +1.8' about the start, always arriving from the same side, and a line through the three
star positions is that axis's column. It takes up the Dec slack first, and if that moves the star
out of view it steps back until it reappears. It checks the axes are ~90 deg apart, the scale matches
the optics (0.399"/px at 1500 mm), axis1/axis2 matches cos(Dec) and the star comes back where it
started, and refuses a result whose axes are more than 10 deg from square. Then it centres the star.

**Main's aim point is its frame centre**, shown as a grey cross. The **guide boresight** - the green
cross in the guide image - marks where that centre looks: set it with **Set boresight** on the guide
(clicks snap to the nearest bright spot), or **Boresight on star**, which identifies the main
camera's star in the guide's plate solve. **Marks on/off** beside it hides both crosses on the guide
image, for a clean look at what sits under them; it works during tracking and is remembered.

### Without stars

The page only offers the star-based calibrations. The older way - following one bright point source
(a distant lamp, a planet) through small moves - is still there from the terminal console, `c`
(**calibrate on target**), for a cloudy night. It calibrates main against the guide, which came out
15-20% wrong on the rig (moves that keep a star inside 7' shift the guide image by only 1-4 px), so
prefer the star tools whenever there are stars. Calibration on scenery and "boresight from my picks"
were removed on 2026-09-30.

## How wrong may a calibration be?

Mount the cameras at any angle: the matrix absorbs rotation and flip. Tolerance to a *wrong*
calibration, from `--cal-rot-error` in simulation:

| Rotation error | Main camera in control | Median error | Outcome |
|---|---|---|---|
| 0 deg | 39% | 30" | fine |
| 30 deg | 38% | 19" | fine |
| 60 deg | 1% | 1038" | fails to settle |
| 85 deg | 0% | - | never locks |

Scale is less forgiving where it matters: 15% is fine, 30% loses the guide -> main handoff.

## When to redo what

| After | Redo |
|---|---|
| a console restart, nobody touched the tube | nothing |
| the tube was pushed by hand | **Sync on stars** (or Set home at home) |
| the tripod moved | **Calibrate on stars** twice, 20 deg apart (a new alignment, not a new camera calibration) |
| rotating a camera, changing the focal length (Barlow: also set `focal_length_mm`), refocusing the guide lens, moving the guide scope on the tube | the camera calibrations: **Calibrate on stars**, **Calibrate main on star**, the boresight |
