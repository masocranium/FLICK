# Python API

Everything below can be imported directly from `flick`, unless a module is named.

## Quick reference

| Task | Use |
|---|---|
| Fit a model | `CopulaModel.fit`, `KNNModel.fit`, or `get_model(name).fit` |
| Features for a position and target | `ClickGenerator.generate` |
| A trajectory for a position and target | `ClickGenerator.generate_path` |
| Measure a trajectory's real features | `measure_path` |
| Display trajectories on an HTML page | `write_view` |
| Save and load | `model.save`, `load_model` |
| Add a model | Subclass `MovementModel`, and use `@register_model` |
| Record | `Recorder`, `XQueryPointerSource` |
| Turn a recording into features | `read_session`, `segment`, `extract_all` |
| Load a feature table | `read_rows`, `select_rows`, `to_matrix` |
| Score a model | `cross_validate`, `energy_distance`, `feature_gaps`, `check_bounds` |

## From recording to generated data

Every step of the command line is available from Python. This runs the pipeline on a canned pointer
stream rather than a real one:

```python
import numpy as np
from flick import (
    ClickGenerator, CopulaModel, DEFAULT_COLUMNS, Recorder, ReplaySource, extract_all,
    make_demo_features, read_session, segment,
)

# A straight move, a pause, a left click, and a rest, in (x, y, button bitmask) samples.
LEFT = 1 << 8
samples = [(100 + i * 4, 200, 0) for i in range(100)] + [(496, 200, 0)] * 20
samples += [(496, 200, LEFT)] * 12 + [(496, 200, 0)] * 80

now = [0.0]
def clock(): return now[0]
def sleep(dt): now[0] += dt

recorder = Recorder(ReplaySource(samples), "example", hz=200.0, clock=clock, sleep=sleep)
raw = recorder.record("/tmp/flick-example.raw.jsonl", seconds=len(samples) / 200.0)

rows = extract_all(segment(read_session(raw)))
print([r.kind for r in rows], round(rows[0].distance_px))

# One click isn't enough to fit, so the fit uses fabricated rows.
model = CopulaModel.fit(make_demo_features(300), DEFAULT_COLUMNS, rng=np.random.default_rng(0))
generator = ClickGenerator(model, np.random.default_rng(1))

click = generator.generate(start=(100, 200), target=(640, 360))
print(round(click.distance_px, 1), round(click.duration_s, 3), round(click.get("v_peak")))

path = generator.path_for(click)        # a trajectory realizing that same click
print(path.to_rows()[0], path.to_rows()[-1]["x"])
```

## Generating

### `ClickGenerator(model, rng=None, leg_fn=wind_walk)`

`rng` is a NumPy `Generator`, and `None` seeds from entropy. `leg_fn` walks each leg of a trajectory.
See [paths.md](paths.md#replacing-the-walk).

| Method | Returns |
|---|---|
| `ClickGenerator.from_file(path, rng=None)` | A generator for a saved model. |
| `generate(start, target)` | `SyntheticClick`, or `None` if the points coincide or aren't finite, or the draw has no usable `duration_s`. |
| `generate_many(start, target, n)` | A list of `SyntheticClick`, possibly empty. |
| `generate_path(start, target)` | `SyntheticPath`, or `None` under the same conditions. |
| `path_for(click)` | A `SyntheticPath` realizing an existing click. |

### `SyntheticClick`

| Member | Meaning |
|---|---|
| `start`, `target` | The two points. |
| `distance_px` | Distance between them. |
| `angle_rad` | Heading, `atan2(dy, dx)`, in the coordinates given. |
| `duration_s` | Drawn duration, clamped to [0.02, 8.0] seconds. |
| `features` | Dict of every drawn column, **unclamped**. |
| `get(column, default=nan)` | Read one column from `features`. |
| `to_row()` | A flat dict for writing out. |

### `SyntheticPath`

| Member | Meaning |
|---|---|
| `click` | The `SyntheticClick` it realizes. |
| `t` | Seconds from the start, as an array. |
| `x`, `y` | Integer pixels, as arrays. |
| `to_rows()` | A list of `{"t", "x", "y"}` dicts. |

### `measure_path(path, params=None) -> FeatureRow`

The features a path really has, computed by the extractor used on recordings.

### In `flick.path`

These are exported from `flick` as well.

| Function | Purpose |
|---|---|
| `generate_path(start, dest, duration_s, overshoot_px, n_submovements, t_peak_frac, rng, leg_fn=wind_walk)` | Returns `(points, delays)`, with one delay per gap. |
| `build_waypoints(start, dest, overshoot_px, n_submovements, rng)` | The chain of targets. |
| `time_schedule(step_lengths, duration_s, t_peak_frac=None, sharpness=5.0)` | Per-step delays summing to `duration_s`. |
| `wind_walk(start_x, start_y, dest_x, dest_y, rng, gravity=9, wind=3, max_step=15, damped_distance=12)` | Yields integer points. |

## Displaying

[paths.md](paths.md#seeing-them) describes the page.

| Function | Purpose |
|---|---|
| `write_view(path, generator, start, target, n=20, real_paths=(), activity=None, seed=None)` | Generate paths, write the page, and return the data it displays. |
| `build_view_data(generator, start, target, n=20, real=(), seed=None, model_name="")` | That data, as a JSON-safe dict. `real` is a list of `Movement`. |
| `render_page(data)` | The page, as a string. |

## Models

[models.md](models.md) has the full contract, and an example.

| Name | Purpose |
|---|---|
| `MovementModel` | The abstract base class. |
| `CopulaModel` | The default. `.copula` is the fitted `GaussianCopula`. |
| `KNNModel` | The resampling baseline. |
| `register_model` | Class decorator making a model available by name. |
| `available_models()` | Registered and installed names. |
| `get_model(name)` | The class for a name. Raises `KeyError`, listing what's available. |
| `load_model(path)` | Load a saved model of any registered type. |

All models share:

- **`Model.fit(data, columns, bounds=FEATURE_BOUNDS, rng=None, **options)`** fits to a rows-by-columns
  array. It raises `ValueError` if `columns` lacks `distance_px`.
- **`sample(distance_px, n=1, rng=None)`** returns a dict of column name to an array of length `n`.
- **`sample_rows(distances, rng=None)`** returns one row per distance, as an array.
- **`save(path)`**, **`to_json()`**, **`Model.from_json(data)`** and **`describe()`**.

## `GaussianCopula`

`CopulaModel` wraps this, and you can use it directly on any table. Nothing in it is specific to mouse
data:

```python
import math
import numpy as np
from flick import GaussianCopula

rng = np.random.default_rng(0)
latent = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=500)
table = np.column_stack([np.exp(latent[:, 0]), np.floor(np.exp(latent[:, 1]))])

copula = GaussianCopula.fit(
    table, ["income", "visits"],
    bounds={"income": (0.0, math.inf), "visits": (0.0, math.inf)},
    rng=np.random.default_rng(1),
)
print(copula.describe())
print(copula.sample(3, np.random.default_rng(2)))
print(copula.sample_conditional({"income": 2.0}, n=2, rng=np.random.default_rng(3)))
```

**`GaussianCopula.fit` is unbounded by default**, unlike `CopulaModel.fit`, which applies
`FEATURE_BOUNDS`. A positive column fitted without its bound gets no log transform, and can sample
negative values.

| Member | Meaning |
|---|---|
| `fit(data, columns, rng=None, shrinkage=None, atom_frac=0.05, calibrate=True, calibration_samples=20000, min_attenuation=0.15, bounds=None, **meta)` | Fit. `meta` is stored: `activity`, `kind`, `fitted_at`, `extract_params`. |
| `sample(n, rng=None)` | `n` rows from the joint distribution. |
| `sample_conditional(known, n=1, rng=None)` | A dict of arrays, given known column values. |
| `columns`, `marginals`, `correlation` | The fitted model. |
| `n_rows`, `shrinkage`, `calibrated_pairs` | Details of the fit. |
| `describe()`, `strongest_pairs(top=12)` | Reports. |
| `save(path)`, `GaussianCopula.load(path)`, `to_json()`, `GaussianCopula.from_json(data)` | Persistence. |

### In `flick.copula`

| Name | Purpose |
|---|---|
| `ConstantMarginal`, `DiscreteMarginal`, `KDEMarginal`, `HurdleMarginal` | The marginal families. Each has `ppf`, `cdf`, `cdf_interval`, `has_atoms`, `to_json` and `describe`. |
| `choose_marginal(v, name="", bounds=None, atom_frac=0.05, max_discrete=12)` | Pick and fit a marginal. |
| `marginal_from_json(data)` | Reverses `to_json`. |
| `spearman(a, b)` | Tie-aware rank correlation. |
| `ledoit_wolf_intensity(Z)` | Shrinkage intensity from a latent matrix. |
| `shrink_correlation(R, n, lam=None, latent=None)` | Returns `(shrunk, intensity)`. |
| `calibrate_pair(ma, mb, target, rng, ...)` | Latent rho reproducing a target Spearman correlation. |
| `calibrate_correlation(R, data, marginals, rng, ...)` | Returns `(R, changed_pairs)`. |
| `nearest_correlation(A, ...)`, `nearest_psd(R, ...)` | Matrix repair. |

## Recording

| Name | Purpose |
|---|---|
| `Recorder(source, activity, hz=200.0, origin=(0, 0), note="", chunk=1000, clock=..., sleep=...)` | `.record(path, seconds=None, stop=None, on_chunk=None)` writes a raw file. `clock` and `sleep` can be replaced in tests. |
| `XQueryPointerSource(display=None, window=None)` | The X11 source. Needs python-xlib. |
| `ReplaySource(samples)` | A canned source, for tests. |
| `PointerSource` | The protocol: `read() -> (x, y, state)`, and `close()`. |
| `read_session(path)` | Load a raw file as a `RawSession` (in `flick.capture`). |

## Features

| Name | Purpose |
|---|---|
| `segment(session, params=None)` | A list of `Movement`. |
| `SegmentParams` | Segmentation thresholds. |
| `extract(movement, params=None)`, `extract_all(movements, params=None)` | `FeatureRow`s. |
| `ExtractParams` | Extraction parameters. |
| `write_rows(path, rows, params, segment_params=None)` | Write a JSONL features file. |
| `read_rows(path)` | `(rows, header)`, with rows as dicts, from CSV or JSONL. |
| `select_rows(rows, activity=None, kind=None)` | Filter to a stratum. |
| `to_matrix(rows, columns=DEFAULT_COLUMNS)` | A float array, dropping rows with missing or non-finite values. |

## Evaluation

[evaluation.md](evaluation.md) shows these in use.

| Function | Returns |
|---|---|
| `cross_validate(data, columns, model="copula", folds=5, seed=0, bounds=FEATURE_BOUNDS, **options)` | One dict per scored fold: `fold`, `n_train`, `n_test`, `score`, `floor`. |
| `sample_like(model, train, columns, n, rng)` | Generated rows, with distances drawn from `train`. |
| `energy_distance(x, y)` | Distance between two samples. Standardize first. |
| `noise_floor(x, repeats=25, rng=None)` | `(median, p95)` distance of `x` split against itself. |
| `wasserstein1(a, b, quantiles=256)` | 1D distance in the feature's own units. |
| `Standardizer.fit(x, columns=())` | A scaler with `.transform(x)`, `.center` and `.scale`. |
| `feature_gaps(real, sim, columns, standardizer=None)` | A gap per column, worst first. |
| `check_bounds(samples, columns, bounds=FEATURE_BOUNDS, tol=1e-9)` | `BoundsReport`, with `violations`, `n_impossible`, `nonfinite` and `rate`. |
| `gaussian_baseline(data, n, rng=None)` | Samples from a multivariate Gaussian fitted to `data`. |

## Schema and demo data

| Name | Meaning |
|---|---|
| `DEFAULT_COLUMNS` | The 14 columns fitted by default, in order. |
| `FEATURE_BOUNDS` | Column name to physical `(lo, hi)`. |
| `DISTANCE_COLUMN` | `"distance_px"`. |
| `make_demo_features(n=400, seed=0)` | Fabricated rows, in `DEFAULT_COLUMNS` order. **Not human data.** |
| `write_demo_csv(path, n=400, seed=0)` | Writes them to a CSV. |

## Reproducibility

Every random operation takes an explicit NumPy `Generator`, and a seeded `ClickGenerator` reproduces
exact paths. Passing `rng=None` to a sampling method draws from fresh entropy, while
`GaussianCopula.fit` falls back to a fixed seed of 0.
