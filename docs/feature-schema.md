# Feature schema

A feature table has one row per pointer movement. `flick extract` produces one from a
[recording](recording.md), computing each column as defined here. If you have your own
pointer data, you can supply your own table instead, as long as the columns mean the
same thing.

Models need `distance_px`, and `ClickGenerator` needs `duration_s`. Beyond that you can
fit whichever columns you like.

## Default columns

These 14 columns are fitted by default, in this order. "Bounds" are what the fit enforces.

Below, a movement runs from its start point to its end point in a straight-line
direction $u$ with distance $d$. *Along* means the projection onto $u$.

| Column | Unit | Bounds | Definition |
|---|---|---|---|
| `distance_px` | px | ≥ 0 | Straight-line distance from start to end. Conditioning on this is the main use. |
| `duration_s` | s | ≥ 0 | Elapsed time of the movement. |
| `straightness` | ratio | 0 to 1 | `distance / path_length`. Exactly 1 for a perfectly straight path. |
| `max_dev_px` | px | none | Signed perpendicular deviation at its largest absolute value. The sign records which side the path bows toward. |
| `max_dev_frac` | fraction | 0 to 1 | Normalized time at which that largest deviation occurs. |
| `v_peak` | px/s | ≥ 0 | Maximum speed. |
| `t_peak_frac` | fraction | 0 to 1 | Normalized time of peak speed. |
| `accel_peak` | px/s² | ≥ 0 | Maximum of the rate of change of speed. |
| `decel_peak` | px/s² | ≤ 0 | Minimum of the rate of change of speed. |
| `jerk_rms` | px/s³ | ≥ 0 | Root mean square of the rate of change of acceleration. |
| `n_submovements` | count | ≥ 0 | Velocity peaks minus one. A clean single motion scores 0. |
| `overshoot_px` | px | ≥ 0 | `max(along) - distance`: how far past the end point the cursor traveled. |
| `settle_s` | s | ≥ 0 | Time from first entering a small radius around the end point until the end. |
| `n_direction_changes` | count | ≥ 0 | Sign changes in along-axis velocity, ignoring a small deadband around zero. |

## Other recognized columns

These have declared bounds, but aren't fitted by default. Pass them with `columns=` or
`--columns`.

| Column | Unit | Bounds | Definition |
|---|---|---|---|
| `pause_before_s` | s | ≥ 0 | Idle time before the movement. |
| `dwell_s` | s | ≥ 0 | Time the button is held. |
| `path_length_px` | px | ≥ 0 | Total length of the path. |
| `v_mean` | px/s | ≥ 0 | `path_length / duration`. |

**Don't fit `path_length_px` or `v_mean` alongside the default columns.** They're exact
functions of columns already there, which makes the correlation matrix singular.
Calculate them from a drawn row instead:

```python
import numpy as np
from flick import ClickGenerator, CopulaModel, DEFAULT_COLUMNS, make_demo_features

model = CopulaModel.fit(make_demo_features(300), DEFAULT_COLUMNS, rng=np.random.default_rng(0))
click = ClickGenerator(model, np.random.default_rng(1)).generate((0, 0), (150, 200))

path_length_px = click.distance_px / click.get("straightness")
v_mean = path_length_px / click.duration_s
print(round(path_length_px, 1), round(v_mean, 1))
```

## Extraction parameters

Several columns depend on how you process the pointer stream as much as on the movement.
Raw samples are integer pixels on a jittery clock, so differentiating them directly
amplifies noise with each derivative. A workable approach:

1. Resample positions onto a uniform time grid.
2. Smooth **position** with a Gaussian.
3. Only then differentiate to get speed, acceleration and jerk.

`flick extract` does this, with these defaults:

| Parameter | Default | Affects |
|---|---|---|
| Resampling rate | 200 Hz | All kinematic columns |
| Position smoothing, sigma | 12 ms | `jerk_rms`, `accel_peak`, `decel_peak`, `n_submovements` |
| Submovement peak prominence | 10% of `v_peak` | `n_submovements` |
| Submovement peak minimum height | 5% of `v_peak` | `n_submovements` |
| Submovement peak minimum separation | 40 ms | `n_submovements` |
| Settle radius | 15 px | `settle_s` |
| Direction deadband | 5% of `v_peak` | `n_direction_changes` |

The smoothing width matters a lot. In one recording, moving it from 6 ms to 45 ms changed
`n_submovements` from 3.94 to 0.59 and `jerk_rms` by a factor of 17.

**Never pool rows extracted with different parameters.** A `jerk_rms` at 6 ms and one at
12 ms aren't the same quantity, and pooling them produces a bimodal marginal that's pure
artifact. `flick extract` stores the parameters in the file's header, and a copula fitted
through the CLI carries them along.

## What a recording can't tell you

A recording shows where a click landed, not where the person was aiming, so there's no
"landing error" column. Target width isn't visible either, so neither is a true Fitts index
of difficulty. The columns `overshoot_px`, `settle_s`, `n_direction_changes` and
`n_submovements` are the observable traces of aiming precision.

## File formats

### CSV

The first line holds column names. Blank cells count as missing. A file must end in `.csv`
to be read this way.

```csv
activity,kind,distance_px,duration_s,straightness
photo editing,click,312.5,0.71,0.96
photo editing,click,88.1,0.42,0.91
```

### JSONL

One JSON object per line. Any extension other than `.csv` is read this way. An optional
header record holds the extraction parameters:

```json
{"kind_record": "header", "version": 1, "extract_params": {"hz": 200.0, "smooth_sigma_s": 0.012}}
{"kind_record": "row", "activity": "photo editing", "kind": "click", "distance_px": 312.5, "duration_s": 0.71}
```

The header's `version` must be `1`. Its `extract_params` are copied into the fitted model
when you use the CLI. Other fields, such as `session_id` or `start_x`, are ignored.

### Stratification columns

Two optional columns select which rows to fit:

- **`activity`** is a free-form label for the task. Different tasks have different target
  sizes, distances and urgency, so fit them separately.
- **`kind`** is the type of movement. `flick extract` writes `click`, `drift`, `drag` or
  `excursion`. Fit on `click`.

### Missing values

Any row with a missing or non-finite value in a selected column is dropped, so choose
columns deliberately. Asking for `dwell_s` on a table where only clicks have it discards
every other movement.
