# Models

A model predicts a movement's features given its distance. FLICK's other parts only use
the interface on this page, so any model that implements it works with `flick generate`,
`flick path`, `flick fit`, cross-validation and saving.

## Built-in models

```bash
flick models
```

### `copula` (default)

A NORTA Gaussian copula. Each column gets its own marginal distribution, and one
correlation matrix couples them. Conditioning on distance is exact and closed-form. See
[theory.md](theory.md).

Options are those of `GaussianCopula.fit`:

| Option | Default | Meaning |
|---|---|---|
| `shrinkage` | Ledoit-Wolf | Force a shrinkage intensity from 0 to 1 |
| `calibrate` | `True` | Run NORTA correlation matching |
| `calibration_samples` | 20000 | Monte Carlo samples for correlation matching |
| `min_attenuation` | 0.15 | Calibrate a pair only if it misses its target by more than this |
| `atom_frac` | 0.05 | Share of the mass a value needs to count as an atom |

### `knn`

Finds the `k` recorded rows nearest in log distance, and returns one at random, with
`distance_px` set to the value asked for.

| Option | Default | Meaning |
|---|---|---|
| `k` | square root of the row count | Neighbours to choose among |

It makes a useful baseline because it assumes almost nothing, and keeps dependence because
whole rows stay together. It has two real weaknesses:

- **It only repeats recorded rows**, so it generates nothing new, and its in-sample score
  looks excellent for that reason. Rely on the cross-validated score.
- **It doesn't rescale to the distance.** Asked for 1000 px when your longest recording is
  500 px, it returns 500 px movements labelled as 1000 px.

## Choosing a model

Fit each on the same data, and compare the cross-validated medians:

```bash
flick fit data/features.jsonl --kind click --model copula
flick fit data/features.jsonl --kind click --model knn --option k=9
```

With the same `--seed` and data, the folds and the floor are identical, so the scores are
directly comparable. See [evaluation.md](evaluation.md).

## Writing your own

Subclass `MovementModel`, set `name`, and implement four methods:

| Member | Requirement |
|---|---|
| `name` | Unique. Stored in saved files and passed to `--model`. |
| `columns` | Instance attribute listing every feature produced. Must include `distance_px`. |
| `fit(cls, data, columns, bounds=..., rng=None, **options)` | Class method taking a rows-by-columns array, returning a fitted model. |
| `sample(self, distance_px, n=1, rng=None)` | Returns a dict of column name to an array of length `n`, for every column, with `distance_px` equal to the value given. |
| `to_json(self)` | Returns a JSON-serializable dict. |
| `from_json(cls, data)` | Class method rebuilding the model. `data` also has a `model` key. |
| `describe(self)` | Optional. `(label, description)` pairs for the `fit` report. |

For your model to work with everything else:

- **`rng` is a NumPy `Generator`, and all randomness must come from it** when it's given,
  or seeded runs won't be reproducible.
- **`bounds` maps column names to physical `(lo, hi)` limits.** Generated values should
  stay inside them, or `flick fit`'s bounds check will report them.
- **Raise `ValueError` from `fit` if the data can't be fitted.** Cross-validation skips that
  fold rather than crashing.
- **`duration_s` has to be finite** for `ClickGenerator` to use a draw.
- **Path generation reads `overshoot_px`, `n_submovements` and `t_peak_frac`,** if present.
  A model without them gets a plain single-leg path with the default peak timing.

A worked example, predicting duration with Fitts-style regression and bootstrapping
residuals, with other features from nearby rows:

```python
import math

import numpy as np

from flick import (
    ClickGenerator, DEFAULT_COLUMNS, KNNModel, MovementModel, cross_validate,
    make_demo_features, register_model,
)


@register_model
class RegressionModel(MovementModel):
    """Duration from a log-distance regression, other features from nearby rows."""

    name = "regression"

    def __init__(self, nearby, slope, intercept, residuals):
        self.nearby = nearby
        self.columns = nearby.columns
        self.slope, self.intercept = slope, intercept
        self.residuals = np.asarray(residuals, dtype=float)

    @classmethod
    def fit(cls, data, columns, bounds=None, rng=None, k=None):
        columns = list(columns)
        distance = data[:, columns.index("distance_px")]
        log_duration = np.log(data[:, columns.index("duration_s")])
        x = np.log2(distance + 1.0)
        slope, intercept = np.polyfit(x, log_duration, 1)
        residuals = log_duration - (slope * x + intercept)
        return cls(KNNModel.fit(data, columns, k=k), slope, intercept, residuals)

    def sample(self, distance_px, n=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        out = self.nearby.sample(distance_px, n, rng)
        mean = self.slope * math.log2(float(distance_px) + 1.0) + self.intercept
        out["duration_s"] = np.exp(mean + rng.choice(self.residuals, size=n))
        return out

    def to_json(self):
        return {"nearby": self.nearby.to_json(), "slope": self.slope,
                "intercept": self.intercept, "residuals": self.residuals.tolist()}

    @classmethod
    def from_json(cls, data):
        return cls(KNNModel.from_json(data["nearby"]), data["slope"],
                   data["intercept"], data["residuals"])


data = make_demo_features(200)
model = RegressionModel.fit(data, DEFAULT_COLUMNS)
click = ClickGenerator(model, np.random.default_rng(0)).generate((0, 0), (300, 400))
print(round(click.duration_s, 3))

scores = cross_validate(data, DEFAULT_COLUMNS, "regression", folds=3)
print(round(float(np.median([r["score"] for r in scores])), 3))
```

## Making a model available

From Python, `@register_model` is all you need, or you can pass the instance straight to
`ClickGenerator`, which doesn't look at the registry.

**On the command line,** put the class in a module and name it with `--plugin`, which looks
in the current directory and on your Python path:

```bash
flick fit data/features.jsonl --kind click --plugin my_models --model regression --out r.model.json
flick path r.model.json --plugin my_models --from 0,0 --to 300,400
flick models --plugin my_models
```

`--option NAME=VALUE` passes numeric options to `fit`, such as `--option k=9`.

**As an installed package,** publish an entry point, and FLICK will find it with no flag:

```toml
[project.entry-points."flick.models"]
regression = "my_package.models:RegressionModel"
```

## Saved files can't run code

A saved model names its type with a registered name, not an import path, and loading only
looks that name up. So a model file can't make FLICK import anything, and a custom model
only loads if you've chosen to make it available. `--plugin` does run the module you name,
so only use it with code you trust.
