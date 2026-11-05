# Command line

Installing the package adds a `flick` command. Without installing, run `python -m flick.cli`
with `src` on `PYTHONPATH`.

| Command | Purpose |
|---|---|
| [`record`](#flick-record) | Record the pointer and mouse buttons |
| [`extract`](#flick-extract) | Turn recordings into a feature table |
| [`summary`](#flick-summary) | Show how much data you have, and its statistics |
| [`fit`](#flick-fit) | Fit a model and report on it |
| [`generate`](#flick-generate) | Generate features for a position and target |
| [`path`](#flick-path) | Generate (t, x, y) trajectories for a position and target |
| [`view`](#flick-view) | Write an HTML page displaying generated paths |
| [`models`](#flick-models) | List available models |
| [`demo-data`](#flick-demo-data) | Write fabricated features |

## `flick record`

Records until Ctrl-C or `--seconds`. Needs X11 and `pip install -e ".[record]"`. See
[recording.md](recording.md).

| Option | Default | Meaning |
|---|---|---|
| `--activity NAME` | required | Label for what you're doing. |
| `--hz N` | 200 | Poll rate. |
| `--seconds N` | until Ctrl-C | Stop after this long. |
| `--out DIR` | `data` | Output directory. |
| `--window TEXT` | none | Part of a window title. Marks time outside that window. |
| `--origin X,Y` | `0,0` | Subtract this, for window-relative coordinates. |
| `--note TEXT` | none | Free text stored in the header. |
| `--display NAME` | `$DISPLAY` | X display. |

## `flick extract`

| Option | Default | Meaning |
|---|---|---|
| `--raw PATH [...]` | `data` | Raw session files or directories. |
| `--out PATH` | `data/features.jsonl` | Features file to write. |
| `--csv` | off | Also write a CSV beside it. |
| `--hz N` | 200 | Resampling rate. |
| `--smooth-ms N` | 12 | Position smoothing. [This changes what features mean.](feature-schema.md#extraction-parameters) |
| `--move-enter N` | 45 | Speed in px/s that starts a movement. |
| `--move-exit N` | 12 | Speed in px/s that ends a movement. |
| `--min-distance N` | 12 | Discard movements shorter than this, in px. |
| `--settle-radius N` | 15 | Radius in px that defines `settle_s`. |

## `flick summary`

| Option | Default | Meaning |
|---|---|---|
| `--features PATH` | `data/features.jsonl` | Features file. |
| `--activity NAME` | any | Restrict the statistics to one activity. |
| `--kind NAME` | `click` | Movement kind for the statistics. |

## `flick fit`

```bash
flick fit [FEATURES] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `FEATURES` | `data/features.jsonl` | Feature table: `.csv`, otherwise read as JSONL. |
| `--model NAME` | `copula` | Registered model name. |
| `--plugin MODULE` | none | Import this module first, to register a [custom model](models.md). Repeatable. |
| `--option NAME=VALUE` | none | Numeric option for the model's `fit`, such as `k=9`. Repeatable. |
| `--activity NAME` | any | Only use rows with this `activity`. |
| `--kind NAME` | any | Only use rows with this `kind`. |
| `--columns C [C ...]` | the 14 defaults | Columns to fit. Must include `distance_px`. |
| `--folds K` | 5 | Cross-validation folds, reduced automatically on small tables. |
| `--seed N` | 0 | Seed for the fit and the report. |
| `--samples N` | 2000 | Samples behind the report tables. |
| `--check N` | 5000 | Samples for the bounds check. `0` skips it. |
| `--out PATH` | none | Write the fitted model here. [It contains your training rows.](model-format.md) |

**Pass `--kind click` when fitting a table from `flick extract`.** With no filter, every complete
row is pooled, and other kinds aren't aimed movements. Most lack click-only columns and get dropped
as incomplete anyway, but it's better to be explicit. If your table has several activities, pass
`--activity` too, since each is a different distribution.

Exits with an error if the file is missing, no rows match the filter (listing the combinations
available, which is the quickest way to catch a typo), the model name is unknown, or fewer than 3
complete rows remain.

[evaluation.md](evaluation.md) explains the report.

## `flick generate`

Writes generated features to standard output as CSV, one row per movement.

```bash
flick generate MODEL --from X,Y --to X,Y [-n N] [--seed N] [--plugin MODULE]
```

| Option | Default | Meaning |
|---|---|---|
| `MODEL` | required | A file written by `fit --out`. |
| `--from X,Y` | required | Current mouse position. |
| `--to X,Y` | required | Click target. |
| `-n N` | 1 | Movements to generate. |
| `--seed N` | random | Seed for reproducible output. |
| `--plugin MODULE` | none | Import this first, if the model is a custom one. |

Columns are `start_x`, `start_y`, `target_x`, `target_y`, `angle_rad`, then the model's features.
Values have 6 significant figures, and `duration_s` is clamped to 0.02 to 8 s.

```console
$ flick generate demo.model.json --from 100,200 --to 640,360 -n 2 --seed 1
start_x,start_y,target_x,target_y,angle_rad,duration_s,straightness,max_dev_px,max_dev_frac,v_peak,t_peak_frac,accel_peak,decel_peak,jerk_rms,n_submovements,overshoot_px,settle_s,n_direction_changes,distance_px
100,200,640,360,0.288055,0.916473,0.969993,1.96171,0.211441,1371.53,0.398666,3835.61,-2642.25,22904.9,2,0,0.188707,0,563.205
100,200,640,360,0.288055,0.799151,0.918048,4.35236,0.499793,1186.73,0.2042,4550.92,-3501.76,24792.6,2,8.59322,0.0195605,0,563.205
```

## `flick path`

Writes generated trajectories to standard output as CSV, with columns `movement`, `t`, `x`, `y`. It
takes the same options as `generate`. See [paths.md](paths.md).

```console
$ flick path demo.model.json --from 100,200 --to 640,360 --seed 1
movement,t,x,y
0,0.000000,100,200
0,0.069163,108,202
...
0,0.916473,640,360
```

Times are in seconds from the start of each movement, and positions are in integer pixels.

## `flick view`

Writes one self-contained HTML page showing generated paths, a replay, speed curves and a table
of drawn against measured values. See [paths.md](paths.md#seeing-them).

```bash
flick view MODEL --from X,Y --to X,Y [options]
```

| Option | Default | Meaning |
|---|---|---|
| `MODEL` | required | A file written by `fit --out`. |
| `--from X,Y` | required | Current mouse position. |
| `--to X,Y` | required | Click target. |
| `-n N` | 20 | Paths to generate, and the most recorded clicks to overlay. |
| `--seed N` | random | Seed for reproducible paths. |
| `--real PATH [...]` | none | Raw sessions, to overlay recorded clicks within 30% of the distance. The page then contains recorded movements. |
| `--activity NAME` | any | Only overlay recorded clicks from this activity. |
| `--out PATH` | `paths.view.html` | Page to write. |
| `--open` | off | Open the page in a browser. |
| `--plugin MODULE` | none | Import this first, if the model is a custom one. |

## `flick models`

Lists registered and installed models. `--plugin MODULE` includes a custom one.

## `flick demo-data`

Writes a table of fabricated features for trying FLICK out. **This isn't human data**: it comes from
the formulas in `src/flick/demo.py`, and results measured on it only show that the code works.

```bash
flick demo-data OUT.csv [--rows N] [--seed N]
```

| Option | Default | Meaning |
|---|---|---|
| `OUT.csv` | required | Where to write the table. |
| `--rows N` | 400 | Rows to generate. |
| `--seed N` | 0 | Seed. |
