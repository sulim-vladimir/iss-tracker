# Tracking

Pass mode, servo mode, the pier side, and what happens in shadow, behind obstructions and in
clouds. Back to the [README](../README.md).

Both modes run the same control loop ([control.py](../issctl/control.py)); only the reference
differs. **Pass mode** follows a planned trajectory from the orbit, so the mount already moves at
nearly the right rate before the cameras see anything - it needs the star alignment. **Servo mode**
follows what the camera sees, with no orbit and no alignment - only the camera calibration.

## Pass mode

Start it from **Track selected** in [Coming up or Favourites](satellites.md), **Track it** under the
sky chart (the object just named), `p` in the terminal console (the next ISS pass), or
`./issctl.sh track`. It works for **any satellite** in the catalogues, classified ones included:

```bash
./issctl.sh passes --sat 42065      # a name or NORAD number; the ISS without --sat
./issctl.sh track --sat 42065
```

`passes` lists the coming passes with the pier side, trackable, sunlit and blocked seconds, peak axis
rates and what limits each pass.

The pass is planned through the pointing model and the sky mask. The mount slews to the start
`lead_s` (90 s) early and follows the prediction, correcting the timing (the dominant TLE error) and
the cross-track error from the cameras. **A pass already up** is planned from now - the part gone by
no longer counts - and the mount waits a little *ahead* of the satellite on its path instead of
chasing it.

**The cameras only take over once the target is really there.** Before the first lock:

* nothing counts until the pass has started and the mount has arrived;
* the guide searches only within `acquire_radius_arcmin` (3 deg) of the prediction;
* a blob must hold still in the frame while the stars drift past - the mount follows the
  prediction, so the satellite is nearly still and the stars stream at its rate.

The main camera then only confirms what the guide has: it must agree with the estimate to
`main_agree_arcmin` (1.5', opening to 3x that while nothing is seen), so a star in its small field
cannot take over. A click in either image overrides all of this: it means "that one". In simulation,
pass mode on NOSS 3-8 (B) kept the main camera in control 96% of the run at a median 10".

**Main steers** (main camera panel, off by default, `[tracking] main_steers`): may the main camera
take over steering from the guide once it holds the target steady? Off, the guide steers and main
only shows and records.

## The pier side

A German equatorial mount reaches every point of the sky from two sides of the pier. The software
calls them **east_looking** (axis2 at or below 90) and **west_looking** (above 90). Going from one to
the other is a meridian flip: the RA axis turns by up to 180 deg and the tube swings right round.

**Keep pier side** (Target & tracking, on by default, `[tracking] keep_pier_side`) plans a pass on
the side the tube is on now, so the mount never flips to reach one. A pass this side cannot reach at
all is refused, with how much a flip would have given. Off, the planner takes whichever side tracks
the pass longest. At the pole (home, axis2 near 90) either side is fine - it is only a Dec turn away.

Why it matters: a pass close to the **celestial pole** needs up to ~180 deg of RA while it crosses
the north, more than `axis1_hour_limit` (120 deg either side of counterweight-down) allows on one
side. So each side can follow only its own part of it - typically one the first half, the other the
second. With the switch off, the planner may pick the other side and swing the tube round to wait
there with the counterweight up (seen for real on 2026-10-07 with COSMOS 2226).

`axis1_hour_limit` also costs meridian passes: a pass crossing the meridian needs the counterweight
high, and real passes lose 150-170 s of 370 s at 120 deg. Raising it recovers most of that **if the
tube clears the tripod and railing**.

## Servo mode (Follow)

**Follow** (at the right of the guide's exposure row; `v` in the terminal, `track --servo`) follows
whatever you click in the guide image. The position and rate come from the camera alone
(`servo_alpha`/`servo_beta`). This is what works for anything you can see, and on a mount facing the
wrong way: in simulation, with the tripod turned 90 deg in azimuth, pass mode is 78 deg off while
servo mode holds the target to a median 5-7" on the main camera.

Two ways to start it:

* **Click the object first** - the red circle follows it while the mount stands still - then press
  **Follow**. The motion measured meanwhile becomes the starting rate.
* **Press Follow first** (it says **Click the object**), then click the object.

A later click in the image moves the lock to what you clicked, so if it has grabbed a star, click
the satellite. It stops by itself after `servo_give_up_s` (20 s) with nothing detected.

Once it has followed a target steadily for `lock_frames`, it searches only along the target's own
predicted track, so a faint gap no longer lets a star nearby take over. Corrections are capped so
the target smears at most `servo_smear_px` across the guide image in one exposure.

**Shorten the guide exposure first**: 8 ms for the ISS, 50-200 ms for fainter satellites. At 1 s a
satellite smears into a streak and the stars are the sharpest things in the frame - and with
sidereal tracking off a star drifts slowly through the image, so a click can lock onto a star.

While tracking, each camera caption says whether the loop is steering with it (`TRACKING uses this
camera`), standing by (on main with the handoff count, e.g. `standby (handoff 2/3 frames)`), or
coasting on prediction.

## Earth's shadow

A satellite is only visible while sunlit, and passes routinely fade out partway through (evening) or
appear partway in (morning). `illumination()` computes a conical umbra/penumbra with an 80 km
absorbing shell, so the fade takes a few seconds, as it does in reality.

* `passes` reports `lit` (sunlit seconds above the horizon), `usable` (trackable **and** lit) and
  when the satellite enters or leaves shadow. Passes are chosen by `usable`, not by altitude.
* While the satellite is in shadow the tracker ignores the cameras and coasts on prediction, keeping
  the learned offsets - with the target invisible the brightest blob in the frame is a **star**, and
  following it would drag the mount away. On shadow exit the gates reopen and it re-acquires.
* SER recording pauses in shadow.

Coasting accuracy in simulation (165 s of shadow after a 104 s lit segment): median 199", 95th
percentile 693" - inside the main camera 69% of the time and always inside the guide field, so a
morning re-acquisition should succeed.

## Obstructions and clouds

**Mapped obstructions** go in `config.toml` as azimuth/altitude rectangles:

```toml
[site.sky]
openings = [[100, 260, 18, 80]]   # a south-facing balcony
blockers = [[168, 176, 0, 90]]    # a window frame post
```

Empty `openings` means the whole sky above `min_altitude`. `passes` then reports `blocked` seconds
and splits each pass into **usable windows**, the tracker coasts through a mapped obstruction as it
does through shadow, and **Coming up -> Anywhere** lists only what the openings show. `m` in the
terminal console prints the azimuth/altitude of the current pointing, for the config.

This balcony's view is **not** mapped (`openings = []`): the window frame is close, so any small move
of the tripod shifts its edges by degrees. Pass mode does not need it - it follows the prediction
behind the wall and the guide picks the satellite up when it comes into view. For seeing where the
frame is, draw it on the sky chart with **Frame line** ([console](console.md#sky-chart)); that is a
picture only, not a tracking mask.

**Clouds** cannot be predicted, so the tracker keeps following its model and keeps looking. It does
not lock onto a star in the gap: once locked, detections that imply a jump beyond
`max_offset_jump_arcmin` are rejected, and that limit and the search circle grow only slowly while
coasting (`reacquire_growth_arcmin_per_s`, capped at `max_reacquire_arcmin`).

Simulated results for the same pass (104 s lit, then shadow):

| Case | Median error while coasting | Inside main FOV |
|---|---|---|
| 3 cloud gaps (4-12 s) | 54" | 100% |
| Mapped building (30 s) | 184" | 100% |
| Earth's shadow (165 s) | 209" | 68% |
