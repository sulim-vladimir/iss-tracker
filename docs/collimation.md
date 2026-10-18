# Collimation

The guide camera doubles as an electronic Cheshire. The ASI120MM Mini has a 1.25" body, so with
its M12 lens on it slides straight into the Newtonian's eyepiece holder and looks down the focuser
at the secondary. The page draws the circles it finds and reads out how far each pair is off.

Open it from the **collimate** link on the guide panel's title (`/collimate` on the console), or
run it on its own, with no mount:

```
issctl collimate              # stop the console first: systemctl --user stop iss-console
issctl collimate --sim        # a synthetic, miscollimated tube, with screws to turn
```

Only one program can hold a camera, so use the console's page while the console runs, and the
command when it does not (a tube on another mount, or on the table). Both keep the picked circles,
the axis and the lens in `data/collimation.json`.

## What it measures

Four circles in one picture, and three pairs that should be concentric:

| | circle | should sit on | adjust |
|---|---|---|---|
| 1 | **S** the secondary's outline | the focuser axis | secondary's centre bolt and turn (rarely needed) |
| 2 | **P** the primary's edge (its clips), seen in the secondary | S | the secondary's 3 tilt screws |
| 3 | **D** the primary's centre donut | **C** the camera's reflection | the primary's collimation screws |

Step 3 needs no reference: the primary sends the camera's own reflection straight back, so D on C
means the primary's axis passes through the camera. The camera sees D and C `2e/f` apart, `e`
being how far the primary's axis passes from the focuser axis at the focal plane and `f` the
telescope's focal length. The page turns the offset into `e` in mm, using the lens focal length
set on the page, and grades it against the coma tolerance `0.0072 F³` mm - coma at the field
centre as large as the Airy disk, 0.9 mm at f/5. *Good* is under half of that.

Steps 1 and 2 are graded as a share of the radius (good: under 3% and 1.5%). Both need the
focuser axis from the 180 deg turn - never the frame centre, which the M12 holder alone can put
140 px off. Until S is picked, step 2 compares P with the axis instead: the same question (does
the focuser look at the primary's centre?) without the secondary's outline, which only shows with
white paper behind the secondary.

Each reading is drawn as a target: the dot is the offset, magnified, as in the picture; bring it
into the green. A grey tail shows where it has just been, so a small turn of a screw shows which
way that screw moves it. A tube sketch beside it shows which mirror's screws to turn.

## Setting up

* **Lens.** A wide lens keeps everything in focus at once: the secondary's edge is ~15 cm away,
  the donut ~80 cm, the camera's reflection further still. With a 12 mm lens put a 1-1.5 mm hole
  in black card over it; a 3-4 mm lens needs no stop. Blur does not spoil the reading - a blurred
  circle's centre is still exact - but it does weaken edges. Focus between the secondary and the
  donut: from infinity that is about a turn and a half outwards for a 12 mm M12 lens.
* **Field.** The whole secondary must fit. The 12 mm lens gives 23 x 17 deg on the ASI120MM:
  rack the focuser out if the secondary's edge is cut off. The 16 mm is too narrow.
* **Light.** A white card in the tube opposite the focuser (the secondary's outline then stands out
  against it), the tube aimed at a bright wall or the daytime sky - never the Sun. A white paper
  collar round the lens makes its reflection easier to see.
* **Seat the camera** square and snug in the holder - a compression ring centres it best.

## Doing it

1. **Pick each circle**: its button, then 3-6 clicks spread round its edge, then **Done**. The circle
   snaps onto the nearest edge and follows it from then on (*locked*); one that found no clear edge
   is drawn where you clicked and marked `?`. For D use the donut's outer edge, for C the dark spot
   of the lens' reflection. The dark disk in the middle is the secondary's *reflection*, not S.
   Once a circle has locked, a click far from its edge is refused rather than bending it - press
   Done before the next circle.
2. **Focuser axis**: **1st frame**, turn the camera 180 deg in the holder (a mark on its body helps),
   **2nd frame**. Whatever stays put when the camera turns is on the axis; the circles are moved to
   where the turn put them. Until then the frame centre stands in, shown by a grey cross instead of
   green. Redo it whenever the camera is reseated.
3. **Adjust in order** 1, 2, 3, then check 2 and 3 again. The arrows in the picture are drawn 5x
   longer than the offset. Directions are in the picture, not on the tube: turn a screw a little and
   watch which way its arrow moves.

**Following edges / Frozen** stops the circles moving - handy while a hand is in front of the tube.

## Limits

* Tuned on the synthetic view; the edge thresholds may want adjusting on a real tube.
* The camera's lens must sit on the focuser axis: 1 mm off moves the primary's reading by about as
  much. The 180 deg turn shows how far off the camera sits (the axis' distance from the frame
  centre is mostly that).
* Where the primary's edge runs into the secondary's, the two fits lean ~0.5 px towards each other.
