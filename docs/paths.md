# Generating trajectories

`flick path` turns a cursor position and a click target into (t, x, y) points. The output
is data, a CSV or a set of arrays. Nothing here moves a cursor.

```bash
flick path my.model.json --from 100,200 --to 640,360 -n 5 --seed 1 > paths.csv
```

```python
import numpy as np
from flick import ClickGenerator, CopulaModel, DEFAULT_COLUMNS, make_demo_features, measure_path

model = CopulaModel.fit(make_demo_features(300), DEFAULT_COLUMNS, rng=np.random.default_rng(0))
generator = ClickGenerator(model, np.random.default_rng(1))

path = generator.generate_path(start=(100, 200), target=(640, 360))
print(path.t[-1], path.x[-1], path.y[-1])       # ends on the target
print(path.click.get("n_submovements"))         # what the model drew

measured = measure_path(path)                   # what the path really has
print(round(measured.duration_s, 3), measured.n_submovements)
```

A path starts exactly on `start` and ends exactly on `target`, in integer pixels, with `t`
in seconds from zero. Paths are fully reproducible from a seed.

## Seeing them

`flick view` writes a page you can open in any browser:

```bash
flick view my.model.json --from 100,200 --to 640,360 -n 20 --open
```

The page is one self-contained HTML file with no dependencies, so it also works offline
and can be sent to someone. It shows:

- **Paths on screen.** Every path from start to target, with equal scales on both axes and
  y increasing downward. Hover to highlight the nearest path, and click to keep it
  highlighted.
- **Speed over time.** Each path's speed curve, smoothed the same way features are measured.
  Hover to read the median speed at that moment.
- **Replay.** Moves a dot along every path at real speed, or at 1/2 or 1/4 speed.
- **Drawn and measured.** A table of what the model asked for against what each path has.
  This is the quickest way to see where a model and the path generator disagree.

### Comparing with your recordings

Pass your raw sessions to overlay real clicks in a second colour:

```bash
flick view my.model.json --from 100,200 --to 640,360 --real data/ --activity "photo editing"
```

The page picks the recorded clicks nearest your distance, up to `-n` of them, and only those
within 30% of it. It rotates and scales them onto the same start and target, so the shapes
can be compared directly. Their speed curves and table values come from the original,
unscaled movement.

If nothing is close enough, the command says so, and you get the page without an overlay.
Choose a `--from` and `--to` about as far apart as your recorded moves: `flick summary` prints
the distribution of `distance_px`.

**A page with an overlay contains recorded movements**, and is as sensitive as the recording.
Pages use the suffix `.view.html` by default, which the `.gitignore` here excludes.

From Python, `write_view` does the same:

```python
import numpy as np
from flick import ClickGenerator, CopulaModel, DEFAULT_COLUMNS, make_demo_features, write_view

model = CopulaModel.fit(make_demo_features(300), DEFAULT_COLUMNS, rng=np.random.default_rng(0))
generator = ClickGenerator(model, np.random.default_rng(1))
data = write_view("/tmp/flick-example.view.html", generator, (100, 200), (640, 360), n=5)
print(len(data["movements"]))
```

## How a path is built

1. **Draw features.** The model draws a feature vector for the distance.
2. **Place waypoints.** `overshoot_px` and `n_submovements` become a chain of targets.
   - With no overshoot, the movement stops short of the target and resumes in the same
     direction, once per correction.
   - With an overshoot, it reaches past the target, then walks back.
   - Waypoints stay on the line from start to target, with up to 2 px of sideways wobble.
3. **Walk each leg.** `wind_walk` moves toward each waypoint under gravity and random wind,
   starting from rest each time. That restart gives each leg its own velocity peak, which
   is what a submovement is.
4. **Assign times.** `time_schedule` decides how long each step takes without moving any
   point. The main leg's speed follows a bell peaking at `t_peak_frac`. Each correction
   reserves its own time and starts with a 45 ms pause, or smoothing would erase it.

## What the model controls

| Column | In a path |
|---|---|
| `distance_px` | Exact, from the two points |
| `duration_s` | Exact, after clamping to 0.02 to 8 s |
| `overshoot_px` | Steered, through waypoints |
| `n_submovements` | Steered, through waypoints |
| `t_peak_frac` | Steered, through timing |
| Everything else | **Not steered.** These emerge from the walk, whatever the model drew. |

So `straightness`, `max_dev_px`, `max_dev_frac`, `v_peak`, `accel_peak`, `decel_peak`,
`jerk_rms`, `settle_s` and `n_direction_changes` in `path.click.features` describe what the
model drew, not what the path has. Use `measure_path` to get the path's real values.

## How close paths get

Measured on one real recording of 196 clicks, comparing the recording to 300 generated
paths, each measured back through the same feature extractor. A scaled gap is the
Wasserstein distance divided by the column's spread, so smaller is better.

| Column | Real median | Path median | Scaled gap |
|---|---|---|---|
| `duration_s` | 0.490 | 0.503 | 0.08 |
| `distance_px` | 59.8 | 53.3 | 0.06 |
| `n_submovements` | 2 | 2 | 0.29 |
| `straightness` | 0.933 | 0.969 | 0.31 |
| `accel_peak` | 14,220 | 9,359 | 0.42 |
| `decel_peak` | -13,030 | -10,970 | 0.54 |
| `v_peak` | 543 | 358 | 0.56 |
| `overshoot_px` | 0 | 0 | 0.60 |
| `jerk_rms` | 362,600 | 265,800 | 0.65 |
| `max_dev_px` | 1.15 | 0.93 | 0.80 |
| `settle_s` | 0.250 | 0.141 | 0.97 |
| `max_dev_frac` | 0.295 | 0.426 | 1.11 |
| `t_peak_frac` | 0.199 | 0.350 | 1.19 |
| `n_direction_changes` | 0 | 0 | 1.76 |

Overall, the energy distance from these paths to the real rows was **1.15**. Drawn features
scored 0.10, and real data split against itself had a 95th percentile of 0.26. So paths are
clearly distinguishable from the recording on these features, even though the features they
were generated from are not.

Things to take from this:

- **Duration, distance and submovement count are reliable.**
- **Paths peak too late and too slowly.** `t_peak_frac` is steered, and still misses. The
  `flick view` table shows the pattern on short moves: with corrections and no overshoot, a
  drawn peak of 0.28 can measure 0.93. Timing only shapes the main leg, and a correction leg
  has a 25 px minimum length and a fixed time, so on a short move it is likely the fastest
  part of the path.
- **Direction changes are too common in paths**, though the medians agree.
- **One recording isn't a general result.** Measure on your own data with `measure_path`.

## Replacing the walk

`wind_walk` follows the wind-and-gravity model published by Benjamin Land as WindMouse,
implemented from its description. To use something else, pass a `leg_fn`:

```python
import numpy as np
from flick import ClickGenerator, CopulaModel, DEFAULT_COLUMNS, make_demo_features


def straight_line(start_x, start_y, dest_x, dest_y, rng):
    for i in range(1, 21):
        yield (start_x + (dest_x - start_x) * i / 20, start_y + (dest_y - start_y) * i / 20)


model = CopulaModel.fit(make_demo_features(300), DEFAULT_COLUMNS, rng=np.random.default_rng(0))
generator = ClickGenerator(model, np.random.default_rng(1), leg_fn=straight_line)
print(len(generator.generate_path((0, 0), (300, 400)).t))
```

A leg function takes `(start_x, start_y, dest_x, dest_y, rng)` and yields points, leaving out
the start. It should begin from rest on each call and take its randomness from `rng`, or legs
won't read as submovements and paths won't be reproducible.

The `windmouse` package on PyPI would also fit here, with a small adapter. It's licensed
GPL-3.0-only, so check what that means for your project before depending on it. FLICK's own
walk exists so it doesn't have to.
