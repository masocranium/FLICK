# Command line

Installing the package adds a `norta-copula` command. Without installing, run
`python -m norta_copula.cli` with `src` on `PYTHONPATH`.

## `norta-copula fit`

Fits a model to a feature table and prints a report.

```bash
norta-copula fit FEATURES [options]
```

| Option | Default | Meaning |
|---|---|---|
| `FEATURES` | required | Feature table: `.csv`, otherwise read as JSONL. See [feature-schema.md](feature-schema.md). |
| `--activity NAME` | any | Only use rows with this `activity`. |
| `--kind NAME` | any | Only use rows with this `kind`. |
| `--columns C [C ...]` | the 14 default columns | Columns to fit. Bounds apply to names in the schema; any other column is unbounded. |
| `--folds K` | 5 | Cross-validation folds, reduced automatically on small tables. |
| `--seed N` | 0 | Seed for the fit and the report. |
| `--shrinkage L` | Ledoit-Wolf | Force a shrinkage intensity from 0 to 1. |
| `--samples N` | 4000 | Samples behind the reporting tables. |
| `--check N` | 20000 | Samples for the bounds check. `0` skips it. |
| `--condition COL=VALUE [...]` | none | Also print a conditional draw's medians. |
| `--out PATH` | none | Write the fitted model here. [It contains your training rows.](model-format.md) |

**Filter to one stratum if your table holds more than one.** With no filter, every row
with complete values is pooled, and different activities are different distributions.
In a mixed table, also pass `--kind click`: other kinds usually lack click-only columns,
so they'd be dropped as incomplete anyway, and being explicit makes that visible.

[evaluation.md](evaluation.md) explains the report.

Exits with an error message if the file is missing, no rows match the filter (listing
the combinations available, which is the quickest way to catch a typo), or fewer than 3
complete rows remain.

## `norta-copula sample`

Draws rows from a saved model and writes them to standard output as CSV.

```bash
norta-copula sample MODEL [-n N] [--given COL=VALUE ...] [--seed N]
```

| Option | Default | Meaning |
|---|---|---|
| `MODEL` | required | A JSON file written by `fit --out`. |
| `-n N` | 1 | Rows to draw. |
| `--given COL=VALUE [...]` | none | Condition on known values. Without this, rows come from the full joint distribution. |
| `--seed N` | random | Seed for reproducible draws. |

Values are written to 6 significant figures, and this command doesn't apply the
`duration_s` clamp that `MovementSampler` does.

```console
$ norta-copula sample demo.model.json -n 2 --seed 1 --given distance_px=250
distance_px,duration_s,straightness,max_dev_px,max_dev_frac,v_peak,t_peak_frac,accel_peak,decel_peak,jerk_rms,n_submovements,overshoot_px,settle_s,n_direction_changes
250,0.832711,0.970278,1.45697,0.209484,677.498,0.421105,2232.12,-1511.52,15736.8,2,0,0.170313,0
250,0.711327,0.918853,3.4286,0.496964,587.688,0.220106,2625.35,-2010.85,17074,3,6.37159,0.017537,0
```

## `norta-copula demo-data`

Writes a table of fabricated features for trying the package out. **This isn't human
data**: it comes from the formulas in `src/norta_copula/demo.py`, and results measured
on it only show that the code works.

```bash
norta-copula demo-data OUT.csv [--rows N] [--seed N]
```

| Option | Default | Meaning |
|---|---|---|
| `OUT.csv` | required | Where to write the table. |
| `--rows N` | 400 | Rows to generate. |
| `--seed N` | 0 | Seed. |
