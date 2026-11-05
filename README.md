# FLICK

**Feature-Level Imitation of Cursor Kinematics.** Record mouse movements, fit a model
of them, and generate synthetic movement data for a given cursor position and click
target.

```bash
flick record --activity "photo editing"     # 1. record your pointer (X11)
flick extract                               # 2. turn the recording into a feature table
flick fit --kind click --out my.model.json  # 3. fit a model, and see how good it is

# 4. generate data for a cursor at (100, 200) and a target at (640, 360)
flick generate my.model.json --from 100,200 --to 640,360    # features, as CSV
flick path     my.model.json --from 100,200 --to 640,360    # (t, x, y) points, as CSV
flick view     my.model.json --from 100,200 --to 640,360 --open    # see them in a browser
```

FLICK produces **data**. It writes numbers to a file or a page, or returns arrays, and has
no way to move a cursor or send input to anything.

## How it works

```
pointer -> raw session -> movements -> features -> model -> features -> trajectory
          flick record   flick extract           flick fit  flick generate  flick path
```

The core idea is that a movement is summarized by a row of features: how long it took,
how straight it was, when it peaked, how many corrections it made. A model learns how
those features vary together given the **distance** to travel. Your position and target
fix that distance, the model draws the rest, and the path generator builds a trajectory
to match.

## Swap in your own model

Anything that predicts features from distance can be a model. Two are built in:

| Model | Approach | Trade-off |
|---|---|---|
| `copula` (default) | NORTA Gaussian copula | Generalizes past recorded values and works at small sample sizes, but dependence is limited to a Gaussian structure |
| `knn` | Resamples recorded rows of a similar distance | Assumes almost nothing, but can only repeat what it has seen, and doesn't extrapolate |

Choose one with `flick fit --model knn`. Everything downstream works the same, because a
saved model records its own type. Adding your own takes four methods:

```python
import numpy as np
from flick import ClickGenerator, DEFAULT_COLUMNS, MovementModel, make_demo_features, register_model


@register_model
class MeanModel(MovementModel):
    """Every feature at its mean, ignoring distance."""

    name = "mean"

    def __init__(self, columns, means):
        self.columns, self.means = list(columns), list(means)

    @classmethod
    def fit(cls, data, columns, bounds=None, rng=None, **options):
        return cls(columns, np.mean(data, axis=0).tolist())

    def sample(self, distance_px, n=1, rng=None):
        out = {c: np.full(n, m) for c, m in zip(self.columns, self.means)}
        out["distance_px"] = np.full(n, float(distance_px))
        return out

    def to_json(self):
        return {"columns": self.columns, "means": self.means}

    @classmethod
    def from_json(cls, data):
        return cls(data["columns"], data["means"])


model = MeanModel.fit(make_demo_features(100), DEFAULT_COLUMNS)
path = ClickGenerator(model, np.random.default_rng(0)).generate_path((100, 200), (640, 360))
print(path.t[-1], path.x[-1], path.y[-1])
```

[docs/models.md](docs/models.md) covers the contract, using a custom model from the
command line, and comparing models.

## Install

FLICK isn't on PyPI. Install from a clone, using Python 3.10 or later:

```bash
pip install -e .              # fitting and generating; needs only NumPy
pip install -e ".[record]"    # also recording, which needs python-xlib
pip install -e ".[dev]"       # also the test suite
```

## Try it without recording

`demo-data` writes a table of **fabricated** features:

```bash
flick demo-data demo.csv
flick fit demo.csv --out demo.model.json
flick path demo.model.json --from 100,200 --to 640,360 --seed 1
```

```
movement,t,x,y
0,0.000000,100,200
0,0.069163,108,202
0,0.096370,118,203
...
0,0.916473,640,360
```

To see those paths, run `flick view demo.model.json --from 100,200 --to 640,360 --open`. It
writes a self-contained page with the paths on screen, a real-speed replay, speed curves, and
a table of what the model asked for against what each path has. Add `--real data/` to overlay
your own recorded clicks. See [docs/paths.md](docs/paths.md#seeing-them).

Demo numbers show that the code works. They say nothing about how well a model fits real
people.

## How good is it?

`flick fit` reports a cross-validated score against a noise floor, for any model.
[docs/evaluation.md](docs/evaluation.md) explains how to read it.

The two kinds of output have very different accuracy, and it's important to know which
you're getting. On one real recording of 196 clicks, using energy distance to the real
rows, where lower is better and real data split against itself had a 95th percentile of
0.26:

| Output | Distance to real features |
|---|---|
| `flick generate`: drawn features, in-sample | 0.10 |
| `flick path`: paths, measured back through the feature extractor | 1.15 |

**Generated features fit well. Trajectories fit noticeably worse.** The path generator
honours only some drawn features, and the rest emerge from its random walk. See
[docs/paths.md](docs/paths.md#how-close-paths-get) for the column-by-column detail.

## Limitations

- **Recording is X11 only.** That includes WSLg and XWayland. Other platforms need a
  `PointerSource` implementation, or a feature table from elsewhere.
- **Movements are independent.** Each draw knows nothing about the one before it, so
  there is no rhythm, fatigue or sequence structure.
- **A model describes one person, one mouse, one DPI, and one set of extraction
  parameters.** It doesn't transfer between them.
- **Trajectories are approximate**, as above.
- **Synthetic data isn't indistinguishable from real data.** It reproduces summary
  distributions of individual movements, and that's all it has been measured on.

## Privacy

Mouse dynamics are an identifying biometric, so treat this data with care:

- **The recorder captures the pointer and mouse buttons across the whole display.** It
  never reads the keyboard. Only record yourself, or people who have agreed to it.
- **Saved models contain their training rows.** A [model file](docs/model-format.md) is
  as sensitive as the recording it came from.
- **So do pages with an overlay.** `flick view --real` embeds the recorded movements.
- The `.gitignore` here excludes `data/`, `models/`, `*.model.json` and `*.view.html`.

## Documentation

| Document | Contents |
|---|---|
| [docs/recording.md](docs/recording.md) | Recording, segmentation and feature extraction |
| [docs/feature-schema.md](docs/feature-schema.md) | Every column's definition and the file formats |
| [docs/models.md](docs/models.md) | The model contract, and writing your own |
| [docs/paths.md](docs/paths.md) | How trajectories are built, and how close they get |
| [docs/evaluation.md](docs/evaluation.md) | How to tell whether a model is any good |
| [docs/theory.md](docs/theory.md) | The mathematics of the default copula model |
| [docs/api.md](docs/api.md) | Python API reference |
| [docs/cli.md](docs/cli.md) | Command line reference |
| [docs/model-format.md](docs/model-format.md) | The saved model format |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The pipeline tests run on synthetic pointer streams, so they need no display.
`tests/test_docs.py` runs the Python examples in these documents.
