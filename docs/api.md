# Python API

Everything below can be imported directly from `norta_copula`, unless a module is named.

## Quick reference

| Task | Use |
|---|---|
| Fit to mouse features | `fit_mouse_copula` |
| Fit to any table | `GaussianCopula.fit` |
| Draw whole rows | `GaussianCopula.sample` |
| Draw given known values | `GaussianCopula.sample_conditional` |
| Draw one movement for a distance | `MovementSampler.draw` |
| Save and load | `GaussianCopula.save`, `GaussianCopula.load` |
| Load a feature table | `read_rows`, `select_rows`, `to_matrix` |
| Score a fit | `cross_validate`, `energy_distance`, `feature_gaps`, `check_bounds` |

## Fitting and sampling mouse features

```python
import numpy as np
from norta_copula import (
    GaussianCopula, MovementSampler, fit_mouse_copula, make_demo_features,
)

data = make_demo_features(n=300)
copula = fit_mouse_copula(data, rng=np.random.default_rng(0))

rows = copula.sample(1000, np.random.default_rng(1))       # shape (1000, 14)

given = copula.sample_conditional(
    {"distance_px": 250.0}, n=100, rng=np.random.default_rng(2))
print(given["duration_s"].shape)                           # (100,)

sampler = MovementSampler(copula, rng=np.random.default_rng(3))
movement = sampler.draw(distance_px=250.0)
print(movement.duration_s, movement.get("v_peak"))
```

### `fit_mouse_copula(data, columns=DEFAULT_COLUMNS, **kwargs) -> GaussianCopula`

`GaussianCopula.fit` with `FEATURE_BOUNDS` applied. Use this for mouse features, because
without bounds, sampled values can be impossible.

### `MovementSampler(copula, rng=None)`

Draws movements conditioned on distance. Raises `ValueError` if the copula has no
`distance_px` column. `rng` is a NumPy `Generator`, and `None` seeds from entropy.

- **`MovementSampler.from_file(path, rng=None)`** loads a saved model.
- **`draw(distance_px) -> MovementSpec | None`** returns one movement. It returns `None`
  for a non-finite or non-positive distance, or if the model has no usable `duration_s`.

### `MovementSpec`

| Member | Meaning |
|---|---|
| `distance_px` | The distance conditioned on. |
| `duration_s` | Drawn duration, clamped to [0.02, 8.0] seconds. |
| `features` | Dict of every drawn column, **unclamped**. |
| `get(column, default=nan)` | Read one column from `features`. |

### Constants

| Name | Meaning |
|---|---|
| `DEFAULT_COLUMNS` | The 14 columns fitted by default, in order. |
| `FEATURE_BOUNDS` | Column name to physical `(lo, hi)`. |
| `DISTANCE_COLUMN` | `"distance_px"`. |

## `GaussianCopula`

### `GaussianCopula.fit(data, columns, ...)`

A class method that fits marginals and the latent correlation.

| Parameter | Default | Meaning |
|---|---|---|
| `data` | required | 2D array of rows by columns. Rows with a non-finite value are dropped. |
| `columns` | required | Column names. |
| `rng` | seed 0 | NumPy `Generator` for tie-spreading and calibration. |
| `shrinkage` | `None` | Force an intensity from 0 to 1. `None` estimates it by Ledoit-Wolf. |
| `atom_frac` | 0.05 | Share of the mass a value needs to count as an atom. |
| `calibrate` | `True` | Run NORTA correlation matching. |
| `calibration_samples` | 20000 | Monte Carlo samples per correlation-matching evaluation. |
| `min_attenuation` | 0.15 | Calibrate a pair only if it misses its target by more than this. |
| `bounds` | `None` | Column name to `(lo, hi)`. Columns not named are unbounded. |
| `**meta` | | Stored metadata: `activity`, `kind`, `fitted_at`, `extract_params`. |

Raises `ValueError` if the shape doesn't match `columns`, or if fewer than 3 complete rows
remain.

Note that **`fit` is unbounded by default**. A positive column fitted without its bound
gets no log transform, and it can sample negative values.

### Sampling

- **`sample(n, rng=None) -> ndarray`** draws `n` rows from the joint distribution.
- **`sample_conditional(known, n=1, rng=None) -> dict[str, ndarray]`** draws the other
  columns given `known`, a dict of column name to value. The result holds every column,
  with known ones echoed back. Raises `KeyError` for a column that wasn't fitted.

### Inspecting a fit

| Member | Meaning |
|---|---|
| `columns`, `marginals`, `correlation` | The fitted model. |
| `n_rows`, `shrinkage` | Rows used, and the intensity applied. |
| `n_features` | Number of columns. |
| `calibrated_pairs`, `n_calibrated` | Which entries correlation matching replaced. |
| `was_calibrated(i, j)` | Whether entry `(i, j)` was one of them. |
| `describe()` | List of `(column, description of its marginal)`. |
| `strongest_pairs(top=12)` | List of `(a, b, latent_rho, was_calibrated)`, largest first. |

### Persistence

- **`save(path) -> Path`** writes JSON, creating parent directories.
- **`GaussianCopula.load(path)`** reads it back.
- **`to_json() -> dict`** and **`GaussianCopula.from_json(dict)`** work without files.

[A saved model contains its training rows.](model-format.md)

## Using it on non-mouse data

Nothing in `GaussianCopula` is specific to mouse data:

```python
import math
import numpy as np
from norta_copula import GaussianCopula

rng = np.random.default_rng(0)
latent = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=500)
table = np.column_stack([np.exp(latent[:, 0]), np.floor(np.exp(latent[:, 1]))])

model = GaussianCopula.fit(
    table, ["income", "visits"],
    bounds={"income": (0.0, math.inf), "visits": (0.0, math.inf)},
    rng=np.random.default_rng(1),
)
print(model.describe())
print(model.sample(3, np.random.default_rng(2)))
```

## Marginals

All marginals share one interface:

| Member | Meaning |
|---|---|
| `ppf(u)` | Inverse CDF: uniform in, value out. |
| `cdf(x)` | The ordinary CDF. |
| `cdf_interval(x)` | `(F(x-), F(x))`, the jump at `x`. |
| `has_atoms` | Whether any single point has positive probability. |
| `to_json()`, `describe()` | Serialize, and a one-line summary. |

| Class | Use |
|---|---|
| `ConstantMarginal(value, name="")` | No variation. |
| `DiscreteMarginal(values, probs, name="")` | Empirical PMF. |
| `KDEMarginal(samples, bandwidth, transform, lo, hi, name="", grid_points=2048)` | KDE in a transformed space. `samples` are already transformed. |
| `HurdleMarginal(atom, atom_prob, body, tol=0.0, name="")` | Point mass plus a body. |

- **`choose_marginal(v, name="", bounds=None, atom_frac=0.05, max_discrete=12)`** picks
  and fits the right class, as described in [theory.md](theory.md#marginals).
- **`marginal_from_json(dict)`** reverses `to_json`.

To use a marginal family `choose_marginal` wouldn't pick, build the `GaussianCopula`
directly from `columns`, `marginals` and `correlation`.

## Correlation utilities

| Function | Purpose |
|---|---|
| `spearman(a, b)` | Tie-aware rank correlation. |
| `ledoit_wolf_intensity(Z)` | Shrinkage intensity from a latent matrix. |
| `shrink_correlation(R, n, lam=None, latent=None)` | Returns `(shrunk, intensity)`. |
| `calibrate_pair(ma, mb, target, rng, samples=20000, tol=5e-3, max_iter=24)` | Latent rho that reproduces a target Spearman correlation. |
| `calibrate_correlation(R, data, marginals, rng, samples=20000, min_attenuation=0.15)` | Returns `(R, changed_pairs)`. |
| `nearest_correlation(A, max_iter=100, tol=1e-8)` | Higham's nearest correlation matrix. |
| `nearest_psd(R, min_eig=1e-8)` | Eigenvalue floor, for Cholesky safety. |

## Feature tables

- **`read_rows(path) -> (rows, header)`** loads a CSV or JSONL table as a list of dicts,
  plus the header record (empty for CSV). Raises `ValueError` on an unknown format version.
- **`select_rows(rows, activity=None, kind=None) -> list`** filters to a stratum.
- **`to_matrix(rows, columns=DEFAULT_COLUMNS) -> ndarray`** builds a float array and drops
  any row with a missing or non-finite value in those columns.

## Evaluation

[evaluation.md](evaluation.md) shows these in use.

| Function | Returns |
|---|---|
| `energy_distance(x, y)` | Distance between two samples. Standardize first. |
| `noise_floor(x, repeats=25, rng=None)` | `(median, p95)` distance of `x` split against itself. |
| `wasserstein1(a, b, quantiles=256)` | 1D distance in the feature's own units. |
| `Standardizer.fit(x, columns=())` | A scaler with `.transform(x)`, `.center` and `.scale`. |
| `feature_gaps(real, sim, columns, standardizer=None)` | `FeatureGap` per column, worst first. |
| `cross_validate(data, columns, folds=5, seed=0, shrinkage=None, bounds=None)` | One dict per scored fold: `fold`, `n_train`, `n_test`, `copula`, `floor`. |
| `check_bounds(samples, columns, bounds)` | `BoundsReport` with `violations`, `n_impossible`, `nonfinite`, `rate`. |
| `gaussian_baseline(data, n, rng=None)` | Samples from a multivariate Gaussian fitted to `data`. |

## Demo data

- **`make_demo_features(n=400, seed=0) -> ndarray`** returns fabricated rows in
  `DEFAULT_COLUMNS` order.
- **`write_demo_csv(path, n=400, seed=0) -> Path`** writes them to a CSV.

These aren't human data. Use them to try the code, not to measure model quality.

## Reproducibility

Every random operation takes an explicit NumPy `Generator`. The same data, column order
and seeds give identical results. Passing `rng=None` to a sampling method draws from fresh
entropy, while `fit` falls back to a fixed seed of 0.
