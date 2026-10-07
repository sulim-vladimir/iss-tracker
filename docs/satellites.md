# Satellites

Naming what you followed, what is coming up, the session history and the favourites. Back to the
[README](../README.md).

## The catalogues

Everything here uses the same catalogues, kept in `data/catalog/` and refreshed at most once a day:

* CelesTrak's **active** satellites and its **visual** group (the brightest objects, some rocket
  bodies included);
* **Mike McCants' classified orbits** (`classfd.tle`), observed by amateurs - the NOSS pairs and other
  military satellites are only there;
* McCants' **standard magnitudes** (`qs.mag`), for brightness;
* **bright.tle**: every other object `qs.mag` rates bright (standard mag 8 or brighter) and still in
  orbit - dead satellites and rocket bodies, often the brightest things up there, which "active"
  leaves out. CelesTrak has no group for them, so it is built from its by-launch-year queries,
  about 60 requests and a few minutes, in the Coming-up thread. About 2,500 objects, against ~330
  before it; no Space-Track account needed.

CelesTrak answers the same query from the same public IP at most once every 2 hours ("GP data has
not updated since your last successful download"); that is harmless, the copy from before is used.

## Identify

**Identify** names the satellite a session followed. From the log it rebuilds the object's own sky
track - the counters through the pointing model, plus where the object sat in the guide image
through the guide matrix - and ranks every catalogued satellite by how closely it flew that path at
the same moments. It answers with a name, "probably" a name, both members of a formation pair, or
"nothing in the catalogues flew this path" (an aircraft, a star, or an object no catalogue carries).

It runs by itself when a session ends, and live during it: the guide caption shows the name within
~15 s of locking. **naming on/off** (Target & tracking) switches both automatic runs; the
**Identify** button works either way. From the command line: `./issctl.sh identify [logs/servo-....csv]`.

Each session saves the pointing model it ran with next to its log, so a later re-alignment cannot
skew the answer. **Identify needs the star alignment** - without it the counters say nothing about
the sky.

## Naming a pick, without moving the mount

Click a moving object in the guide image and leave the mount alone: the red circle follows the
object across the frame, and the guide caption names it. What it takes:

1. **naming on**, and a **star alignment** - without one nothing is named at all;
2. the pick must be **detected**: the Target line under the guide image says *locked on your pick*.
   *Your pick - nothing there* means the camera does not see it, and nothing can be named;
3. **at least 4 detections spanning 3 s or more**. A ranking runs every 5 s, so at a 1 s exposure
   the name comes 5-10 s after the click. The first ranking after a start loads the whole catalogue
   and takes a few seconds more on the Pi;
4. it stays named while the circle keeps the object, until it leaves the guide frame. A fast
   satellite at a long exposure can move further between frames than the circle reaches - shorten
   the exposure.

Once named, its path is drawn on the sky chart in **green** - where it has been and where it goes
until it sets - and **Track it** under the chart follows it on its orbit in pass mode (stopping a
running Follow first: the orbit carries the target through faint spells the camera alone loses).

## History

The **History** panel lists the latest real sessions (simulations are left out), newest first: when,
pass or follow, how long, and what Identify made of it. Pick a row, then:

* **Identify** - run it (again) on that session;
* **Add to favourites** - keep the satellite it was identified as, see [Favourites](#favourites).

**Refresh** lists them again. Favourites are starred in the list.

## Coming up

**Coming up** lists the bright satellites due in the next `[forecast] minutes` (60):

* **Through the guide field** - held fixed on the stars when sidereal tracking is on, otherwise fixed
  where the tube points;
* **Anywhere** - above `min_altitude` and inside the sky mask.

Only satellites that are sunlit while the sky here is dark (sun 6 deg below the horizon or more) are
listed, with a countdown, the estimated magnitude, where it will be and its range. The brightness
comes from the standard magnitude, the range and the phase angle - good to about a magnitude, and a
tumbling rocket body does what it likes. The best hours are the first two after dusk and before dawn;
around midnight most low satellites are in Earth's shadow.

Click a row to pick it: its path is drawn on the sky chart in **violet**, with a dot where it is now
(a hollow circle where it will come in). **Track selected** plans and tracks that pass in
[pass mode](tracking.md#pass-mode). Satellites seen before (a History session identified as them) are
shown in green with the day; favourites are starred.

## Favourites

**Favourites** is a small database of the satellites worth coming back to, kept in
`data/favorites.json` on the Pi.

* **Add**: from **History** - **Add to favourites** on a session Identify has named - or type a name
  or NORAD number in the box and press **Add** (a name matching several satellites lists them and
  asks for the number).
* **Remove**: pick one in the list first.
* Each entry shows its NORAD number, when it was last seen (from History) and its own brightness
  rating if it has one; the tooltip says how many sessions it was added from.
* **Predict passes** lists the visible passes of all favourites - or of the picked one - over the
  next `[forecast] favorite_hours` (48): sunlit while the sky here is dark, above `min_altitude` and
  inside the sky mask. Each row gives the day, the visible span, how long until it starts, the
  estimated magnitude, the highest altitude and where it is brightest.
* Click a pass to draw it on the sky chart in **gold**; **Track selected** tracks it in pass mode,
  even a day or two ahead (the mount waits until then).

From the terminal:

```bash
./issctl.sh favorites                       # list them and their visible passes
./issctl.sh favorites --add "NOSS 3-8 (B)"  # or a NORAD number
./issctl.sh favorites --remove 42065 --hours 24
```

**Brightness ratings.** Coming up lists only what has a standard magnitude, and `qs.mag` has almost
nothing launched since ~2018. A favourite no catalogue rates gets a rating of its own, so Coming up
lists it too: from a **Brightness** measurement taken on it during that session (converted to a
standard magnitude with its distance and sun angle from the orbit), otherwise
`[forecast] default_std_mag` (5.0). `[forecast] std_mags` in the config holds hand-set ones. Ratings
the old "Add to Coming up" button kept in `data/state.json` became favourites on the first start of
this version.

## Brightness

**Brightness** (guide panel): press it, then click a star or a satellite in the guide image. A plate
solve of a fresh frame gives the catalogue star there with its magnitude, and a *measured* magnitude
from the frame's own zero point, fitted to every Tycho-2 star the solve matched - so it also works
for a satellite, which no star catalogue has. On the real guide frames the measured magnitudes agree
with Tycho-2 to 0.3-0.6 mag; the spread is given with each answer. It needs a frame that solves:
0.5-2 s exposure, stars held still.
