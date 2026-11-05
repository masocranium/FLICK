"""
Command line: `norta-copula {fit,sample,demo-data}`. See docs/cli.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .copula import GaussianCopula, spearman
from .demo import write_demo_csv
from .metrics import Standardizer, energy_distance
from .mouse import DEFAULT_COLUMNS, FEATURE_BOUNDS, read_rows, select_rows, to_matrix
from .validation import (
    MIN_FOLD_TEST_ROWS,
    check_bounds,
    cross_validate,
    gaussian_baseline,
)


def _parse_known(items) -> dict[str, float]:
    known = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"expected COL=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        known[key.strip()] = float(value)
    return known


def _report_cv(data, columns, args) -> None:
    n = data.shape[0]
    folds = min(args.folds, max(1, n // MIN_FOLD_TEST_ROWS))
    cv = cross_validate(data, columns, folds, args.seed, args.shrinkage,
                        bounds=FEATURE_BOUNDS) if folds > 1 else []
    if not cv:
        print(f"\nOnly {n} rows: not enough for an honest train/test split "
              f"(need >= {2 * MIN_FOLD_TEST_ROWS}).")
        print("Everything below is IN-SAMPLE and will look better than it is.")
        return

    scores = np.array([r["copula"] for r in cv], dtype=float)
    floors = np.array([r["floor"] for r in cv], dtype=float)
    print(f"\nCross-validation ({len(cv)} folds, "
          f"~{cv[0]['n_train']} train / {cv[0]['n_test']} test):")
    print(f"{'fold':>6}{'copula vs held-out':>22}{'real vs held-out':>20}")
    for r in cv:
        print(f"{r['fold']:>6}{r['copula']:>22.4f}{r['floor']:>20.4f}")
    print(f"{'median':>6}{np.median(scores):>22.4f}"
          f"{np.median(floors):>20.4f}   <- TRUST THIS ROW")
    if np.median(scores) <= np.median(floors):
        print("\n  The copula is as close to held-out rows as the training rows "
              "themselves\n  are. Nothing left to learn at this sample size.")
    else:
        ratio = np.median(scores) / max(np.median(floors), 1e-9)
        print(f"\n  {ratio:.2f}x the real-vs-real floor: structure remains "
              "unexplained, or\n  there are too few rows to tell. More data is "
              "the first thing to try.")


def cmd_fit(args) -> int:
    path = Path(args.features)
    if not path.exists():
        raise SystemExit(f"No features file at {path}.")
    all_rows, header = read_rows(path)
    rows = select_rows(all_rows, args.activity, args.kind)
    if not rows:
        combos = sorted({(str(r.get("activity")), str(r.get("kind")))
                         for r in all_rows})
        raise SystemExit(
            f"No rows for activity={args.activity!r} kind={args.kind!r}.\n"
            "Available (activity, kind): "
            + ", ".join(f"{a}/{k}" for a, k in combos))

    columns = list(args.columns) if args.columns else list(DEFAULT_COLUMNS)
    data = to_matrix(rows, columns)
    n, p = data.shape
    if n < 3:
        raise SystemExit(f"Only {n} complete rows over these columns: nothing to "
                         "fit. Collect more, or drop sparse columns.")

    print(f"{path}")
    print(f"  activity   : {args.activity or 'any'}")
    print(f"  kind       : {args.kind or 'any'}")
    print(f"  rows       : {len(rows)} ({n} complete over {p} columns)")
    print(f"  ratio      : {n / p:.1f}x rows/feature ({p * (p - 1) // 2} "
          "correlation parameters; ~10x is the rule of thumb)")

    _report_cv(data, columns, args)

    cop = GaussianCopula.fit(
        data, columns, shrinkage=args.shrinkage, bounds=FEATURE_BOUNDS,
        rng=np.random.default_rng(args.seed),
        activity=args.activity, kind=args.kind or "",
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extract_params=header.get("extract_params") or {})
    print(f"\nFitted on all {n} rows. Correlation shrinkage lambda = "
          f"{cop.shrinkage:.3f}"
          + (" (Ledoit-Wolf)" if args.shrinkage is None else " (forced)"))

    print("\nMarginal chosen per column:")
    for name, desc in cop.describe():
        print(f"  {name:<22} {desc}")

    drawn = cop.sample(args.samples, np.random.default_rng(args.seed + 99))
    std = Standardizer.fit(data, columns)
    print(f"\nMarginal recovery ({args.samples} sampled vs {n} real):")
    print(f"{'feature':<22}{'real p50':>12}{'sim p50':>12}"
          f"{'real p95':>12}{'sim p95':>12}")
    print("-" * 70)
    for j, c in enumerate(columns):
        r, s = data[:, j], drawn[:, j]
        print(f"{c:<22}{np.median(r):>12.4g}{np.median(s):>12.4g}"
              f"{np.percentile(r, 95):>12.4g}{np.percentile(s, 95):>12.4g}")
    print("\nIn-sample energy distance: "
          f"{energy_distance(std.transform(data), std.transform(drawn)):.4f}")

    # Three columns, not one. A calibrated latent rho is deliberately inflated
    # so that the SAMPLED rank correlation lands on the real one, so showing
    # what the model produces next to what was observed keeps the fit checkable.
    index = {c: j for j, c in enumerate(columns)}
    print("\nStrongest fitted correlations:")
    print(f"  {'latent':>8} {'sampled':>8} {'observed':>9}   pair")
    any_calibrated = False
    for a, b, r, calibrated in cop.strongest_pairs(10):
        ia, ib = index[a], index[b]
        any_calibrated |= calibrated
        print(f"  {r:+8.3f}{'*' if calibrated else ' '}"
              f"{spearman(drawn[:, ia], drawn[:, ib]):+8.3f} "
              f"{spearman(data[:, ia], data[:, ib]):+9.3f}   {a} ~ {b}")
    print("  `sampled` vs `observed` is the fit; compare those two columns.")
    if any_calibrated:
        print("  * NORTA-calibrated: the latent value is inflated ON PURPOSE so")
        print("    the sampled column matches. Do not read it as a correlation.")

    if args.check:
        rng = np.random.default_rng(args.seed + 7)
        ours = check_bounds(cop.sample(args.check, rng), columns, FEATURE_BOUNDS)
        mvn = check_bounds(gaussian_baseline(data, args.check, rng), columns,
                           FEATURE_BOUNDS)
        print(f"\nPhysical-bounds check on {args.check:,} samples:")
        print(f"  copula               : {ours.n_impossible} impossible values, "
              f"{ours.nonfinite} non-finite"
              + ("" if ours.n_impossible == 0 else "  <- BUG, investigate"))
        print(f"  multivariate Gaussian: {mvn.n_impossible} impossible values "
              f"({mvn.rate:.1%})")

    if args.condition:
        known = _parse_known(args.condition)
        out = cop.sample_conditional(known, n=4000,
                                     rng=np.random.default_rng(args.seed + 5))
        shown = ", ".join(f"{k}={v:g}" for k, v in known.items())
        print(f"\nConditional draw given {shown} (median of 4000):")
        for c in columns:
            if c not in known:
                print(f"  {c:<22} {np.median(out[c]):>12.4g}")

    if args.out:
        cop.save(args.out)
        print(f"\nWrote {args.out}")
        print("This file contains your training rows. See docs/model-format.md "
              "before sharing it.")
    return 0


def cmd_sample(args) -> int:
    cop = GaussianCopula.load(args.model)
    known = _parse_known(args.given)
    drawn = cop.sample_conditional(known, n=args.n,
                                   rng=np.random.default_rng(args.seed))
    writer = csv.writer(sys.stdout)
    writer.writerow(cop.columns)
    for i in range(args.n):
        writer.writerow([f"{float(drawn[c][i]):.6g}" for c in cop.columns])
    return 0


def cmd_demo_data(args) -> int:
    path = write_demo_csv(args.out, n=args.rows, seed=args.seed)
    print(f"Wrote {args.rows} rows of FABRICATED features to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="norta-copula",
        description="Fit and sample a NORTA Gaussian copula over movement features.")
    sub = ap.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a copula to a feature table and report on it")
    fit.add_argument("features", help="feature table, .csv or JSONL")
    fit.add_argument("--activity", default=None, help="only rows with this activity")
    fit.add_argument("--kind", default=None, help="only rows with this kind")
    fit.add_argument("--columns", nargs="*", default=None,
                     help=f"columns to fit (default: the {len(DEFAULT_COLUMNS)} "
                          "in docs/feature-schema.md)")
    fit.add_argument("--folds", type=int, default=5, help="CV folds (default 5)")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--shrinkage", type=float, default=None,
                     help="force shrinkage toward the identity, 0-1 "
                          "(default: Ledoit-Wolf, estimated from the data)")
    fit.add_argument("--samples", type=int, default=4000,
                     help="samples for the reporting tables (default 4000)")
    fit.add_argument("--check", type=int, default=20000,
                     help="samples for the bounds check (0 to skip)")
    fit.add_argument("--condition", nargs="*", default=None, metavar="COL=VALUE",
                     help="also show a conditional draw")
    fit.add_argument("--out", default=None, help="write the fitted model here")
    fit.set_defaults(func=cmd_fit)

    sample = sub.add_parser("sample", help="draw rows from a saved model, as CSV")
    sample.add_argument("model", help="model JSON written by `fit --out`")
    sample.add_argument("-n", type=int, default=1, help="rows to draw (default 1)")
    sample.add_argument("--given", nargs="*", default=None, metavar="COL=VALUE",
                        help="condition on known values")
    sample.add_argument("--seed", type=int, default=None)
    sample.set_defaults(func=cmd_sample)

    demo = sub.add_parser("demo-data", help="write fabricated features to a CSV")
    demo.add_argument("out", help="CSV path to write")
    demo.add_argument("--rows", type=int, default=400)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo_data)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
