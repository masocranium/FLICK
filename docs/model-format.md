# Saved model format

`GaussianCopula.save` writes one JSON document. `GaussianCopula.load` reads it back,
and a reloaded model produces the same samples for the same RNG state.

## A saved model contains its training data

**This isn't a small summary.** A KDE marginal has to keep its sample points, so the
file stores them, and it stores them in the original row order. For every KDE column
you can reverse the transform to get the original values, and because the order is
shared across columns, you can line them back up into whole rows. Hurdle columns keep
only their non-atom values, so they're harder to align but still readable.

Mouse dynamics are an identifying biometric, so:

- Treat a model file with the same care as the recording it came from.
- Don't commit one to a public repository or attach one to an issue. Removing a
  column from the file doesn't help, because the remaining columns still line up.
- If you need to share something, share the code and the evaluation numbers.

Discrete marginals store only value frequencies, and hurdle marginals also store an
atom probability. The correlation matrix is a summary too. It's the KDE samples that
make a model file sensitive.

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `version` | int | Format version, currently `1`. Loading any other version raises `ValueError`. |
| `columns` | list of str | Column names, in matrix order. |
| `marginals` | list of objects | One per column, in the same order. |
| `correlation` | $p \times p$ array | Latent correlation matrix after calibration, shrinkage and repair. |
| `n_rows` | int | Complete rows used in the fit. |
| `shrinkage` | float | The shrinkage intensity applied, in $[0, 1]$. |
| `calibrated_pairs` | list of `[i, j]` | Entries replaced by correlation matching. These aren't readable as correlations. |
| `activity` | str or null | Free-form metadata. |
| `kind` | str | Free-form metadata. Defaults to `"click"`. |
| `fitted_at` | str | Free-form metadata, an ISO-8601 time when written by the CLI. |
| `extract_params` | object | Feature-extraction parameters, carried from the input header. |

## Marginal objects

Every marginal has `kind` and `name`.

**`constant`**

| Field | Meaning |
|---|---|
| `value` | The single value. |

**`discrete`**

| Field | Meaning |
|---|---|
| `values` | Sorted observed values. |
| `probs` | Probability of each. |

**`kde`**

| Field | Meaning |
|---|---|
| `samples` | The sample points in **transformed** space. |
| `bandwidth` | Bandwidth in transformed space. |
| `transform` | `identity`, `log`, `neglog` or `logit`. |
| `lo`, `hi` | Physical bounds. `null` means infinite, because JSON has no infinity. |
| `grid_points` | CDF grid size, usually 2048. |

**`hurdle`**

| Field | Meaning |
|---|---|
| `atom` | Location of the point mass. |
| `atom_prob` | Probability of the point mass. |
| `tol` | Values this close to `atom` count as the atom. |
| `body` | A nested marginal object for everything else. |

## Example

Heavily abbreviated:

```json
{
  "version": 1,
  "columns": ["distance_px", "n_submovements"],
  "marginals": [
    {"kind": "kde", "name": "distance_px", "samples": [4.61, 5.32, 4.98],
     "bandwidth": 0.2, "transform": "log", "lo": 0.0, "hi": null, "grid_points": 2048},
    {"kind": "discrete", "name": "n_submovements",
     "values": [0.0, 1.0, 2.0], "probs": [0.5, 0.3, 0.2]}
  ],
  "correlation": [[1.0, 0.3], [0.3, 1.0]],
  "n_rows": 182,
  "shrinkage": 0.047,
  "calibrated_pairs": [],
  "activity": "photo editing",
  "kind": "click",
  "fitted_at": "2026-01-01T00:00:00+00:00",
  "extract_params": {}
}
```
