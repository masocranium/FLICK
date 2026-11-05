# Evaluating a model

`flick fit` prints a report, for any model. This page explains how to read it, and how to
compute the same numbers from Python. To evaluate trajectories, see
[paths.md](paths.md#how-close-paths-get).

## The metric: energy distance

$$D(X, Y) = 2\,\mathbb{E}\lVert X - Y \rVert - \mathbb{E}\lVert X - X' \rVert - \mathbb{E}\lVert Y - Y' \rVert$$

It's zero exactly when two distributions match, it's always finite, and it needs no density
estimate. KL divergence would be the obvious alternative, but it's infinite wherever a model
gives zero probability to something the data contains, so it can't rank two bad models.

Columns are standardized first, or the metric would only measure whichever feature has the
largest units. `Standardizer` centers on the median and scales by the mean absolute deviation
from it, fitted on the real data alone. It deliberately doesn't use the IQR, which can be tiny
but nonzero on a zero-inflated column and would then let that column dominate everything.

## Reading the report

### Cross-validation: the number to trust

```
median                0.1247              0.1009   <- TRUST THIS ROW
```

The left column fits on k-1 folds, generates rows, and measures the distance to the held-out
fold. The right column is the floor: the distance from the same held-out fold to the real
training rows.

Models predict features *given* a distance, so distances for the generated rows are drawn from
the training rows, and the model supplies the rest. This treats every model the same way.

| Result | Meaning |
|---|---|
| At or below the floor | As close to held-out data as more real data would be. Nothing is left to learn at this sample size. |
| Slightly above the floor | Some structure is unexplained, or there are too few rows to tell. More data is the first thing to try. |
| Well above the floor | The model is missing something. Look at the per-column table. |

The floor is a guide rather than a strict threshold, for two reasons. An energy distance computed
this way is biased upward at small sample sizes, and the two columns aren't perfectly matched in
size: the model is scored on `max(n_test, 256)` draws, and the floor on the training rows.

Below 16 complete rows there's no honest split, and the report says so.

### In-sample energy distance

This always looks better than the cross-validated number, because the model has seen these rows.
`knn` looks especially good, since it returns those very rows. Don't use it to judge a fit.

### Feature recovery

Median and 95th percentile per column, real against generated. Differences in the 95th percentile
are normal at a few hundred rows, because the tail is the least-determined part of any
distribution. Large differences in the median are not.

### Strongest fitted correlations (copula only)

```
    latent  sampled  observed   pair
    +0.646*  +0.499    +0.496   n_submovements ~ n_direction_changes
```

Compare **sampled** to **observed**, which are both Spearman correlations. `latent` is the model's
internal parameter, and a `*` marks a calibrated entry that was inflated on purpose. Here, a latent
0.646 reproduces an observed 0.496, so reading 0.646 as the strength of the relationship would
overstate it.

### Physical-bounds check

```
  copula               : 0 impossible values, 0 non-finite
  multivariate Gaussian: 9477 impossible values (13.5%)
```

The count should be zero for the built-in models. The Gaussian line is measured on your data each
time, as a baseline. Differences under 1e-9 are ignored, because derived columns hold
floating-point dust such as `-1e-14`.

## How much data?

Aim for about 10x as many rows as columns, or about 140 for the defaults. The copula's correlation
matrix has 91 free parameters at 14 columns. Shrinkage keeps a fit valid below that, and the
cross-validated score tells you whether it's still useful.

To find out whether more data would help, plot the cross-validated score against row count. If it
has stopped falling, recording more won't help.

## The same checks from Python

```python
import numpy as np
from flick import (
    DEFAULT_COLUMNS, CopulaModel, Standardizer, check_bounds, cross_validate,
    feature_gaps, gaussian_baseline, make_demo_features, noise_floor, sample_like,
)

data = make_demo_features(n=300)

# Compare models on the same folds.
for name in ("copula", "knn"):
    results = cross_validate(data, DEFAULT_COLUMNS, name, folds=5, seed=0)
    print(name, round(float(np.median([r["score"] for r in results])), 3),
          "floor", round(float(np.median([r["floor"] for r in results])), 3))

model = CopulaModel.fit(data, DEFAULT_COLUMNS, rng=np.random.default_rng(0))
drawn = sample_like(model, data, DEFAULT_COLUMNS, 2000, np.random.default_rng(1))

# Which features is the model getting most wrong?
for gap in feature_gaps(data, drawn, DEFAULT_COLUMNS)[:3]:
    print(gap.feature, round(gap.scaled_distance, 3))

# Real against itself: the distance expected from sampling noise alone.
std = Standardizer.fit(data, DEFAULT_COLUMNS)
median, p95 = noise_floor(std.transform(data), rng=np.random.default_rng(2))
print("self-split p95", round(p95, 3))

# Impossible values, against the baseline.
print(check_bounds(drawn, DEFAULT_COLUMNS).n_impossible)
baseline = gaussian_baseline(data, 2000, np.random.default_rng(3))
print(round(check_bounds(baseline, DEFAULT_COLUMNS).rate, 3))
```

`noise_floor` and the floor from `cross_validate` answer slightly different questions. `noise_floor`
splits all the data in half repeatedly, and `cross_validate` compares each held-out fold to the rest.
Quote whichever matches the comparison you're making.
