"""
Command line: `flick {record,extract,summary,fit,generate,path,view,models,demo-data}`.
See docs/cli.md.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .capture import CHANNEL_WINDOW_BOUNDS, Recorder, XQueryPointerSource, iter_sessions
from .copula import spearman
from .demo import write_demo_csv
from .features import (
    CLICK_ONLY_COLUMNS,
    FEATURE_COLUMNS,
    ExtractParams,
    extract_all,
    read_rows,
    select_rows,
    to_matrix,
    write_csv,
    write_rows,
)
from .generate import ClickGenerator
from .metrics import Standardizer, energy_distance
from .models import CopulaModel, available_models, get_model
from .schema import DEFAULT_COLUMNS, FEATURE_BOUNDS
from .segment import SegmentParams, excursion_events, segment
from .view import REAL_DISTANCE_BAND, write_view
from .validation import (
    MIN_FOLD_TEST_ROWS,
    check_bounds,
    cross_validate,
    gaussian_baseline,
    sample_like,
)

DEFAULT_DIR = Path("data")
DEFAULT_FEATURES = DEFAULT_DIR / "features.jsonl"


def _point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y, got {text!r}") from None
    return x, y


def _load_plugins(names) -> None:
    """Import modules that register models. Named by the person running the command,
    never by a model file, and looked up from the current directory too."""
    if names:
        sys.path.insert(0, os.getcwd())
    for name in names or ():
        importlib.import_module(name)


# --------------------------------------------------------------------------- #
# record / extract / summary
# --------------------------------------------------------------------------- #

def cmd_record(args) -> int:
    origin = _point(args.origin) if args.origin else (0, 0)
    origin = (int(origin[0]), int(origin[1]))
    try:
        source = XQueryPointerSource(args.display, window=args.window)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Could not open the X display: {exc}\n"
              f"Is DISPLAY set? (currently {os.environ.get('DISPLAY', '')!r})",
              file=sys.stderr)
        return 1

    if args.window and CHANNEL_WINDOW_BOUNDS not in source.channels:
        print(f"note: no window with {args.window!r} in its title, so time outside "
              "it won't be recorded this session.")

    recorder = Recorder(source=source, activity=args.activity, hz=args.hz,
                        origin=origin, note=args.note or "")
    path = Path(args.out) / f"{recorder.header.session_id}.raw.jsonl"

    print(f"Recording  activity={args.activity!r}  hz={args.hz:g}  -> {path}")
    print("This records the pointer and mouse buttons across the whole display, "
          "and never the keyboard.")
    print(f"Stopping after {args.seconds:g}s. Ctrl-C to stop sooner." if args.seconds
          else "Ctrl-C to stop. The file stays valid, so this is safe.")

    interactive = sys.stdout.isatty()
    last = [0.0]

    def on_chunk(n: int, elapsed: float) -> None:
        text = f"  {elapsed:7.1f}s   {n:>9,} samples   "\
               f"{(n / elapsed if elapsed > 0 else 0.0):6.1f} Hz effective"
        if interactive:
            sys.stdout.write("\r" + text + "   ")
            sys.stdout.flush()
        elif elapsed - last[0] >= 10.0:
            last[0] = elapsed
            print(text, flush=True)

    started = time.perf_counter()
    try:
        recorder.record(path, seconds=args.seconds, on_chunk=on_chunk)
    finally:
        source.close()
    elapsed = time.perf_counter() - started
    print(f"\n  {elapsed:.1f}s, {recorder.samples_written:,} samples, "
          f"{recorder.samples_written / max(elapsed, 1e-9):.1f} Hz effective")
    print(f"Wrote {path}\nNext:  flick extract --raw {args.out}")
    return 0


def cmd_extract(args) -> int:
    raw = [Path(p) for p in (args.raw or [DEFAULT_DIR])]
    missing = [p for p in raw if not p.exists()]
    if missing:
        print(f"No such path: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    seg = SegmentParams(hz=args.hz, smooth_sigma_s=args.smooth_ms / 1000.0,
                        move_enter_px_s=args.move_enter, move_exit_px_s=args.move_exit,
                        min_distance_px=args.min_distance)
    ext = ExtractParams(hz=args.hz, smooth_sigma_s=args.smooth_ms / 1000.0,
                        settle_radius_px=args.settle_radius)

    all_rows, n_sessions = [], 0
    for session in iter_sessions(raw):
        n_sessions += 1
        rows = extract_all(segment(session, seg), ext)
        all_rows.extend(rows)
        counts = Counter(r.kind for r in rows)
        away = sum(e.duration_s for e in excursion_events(session))
        print(f"  {session.header.session_id:<44} {session.duration_s:7.1f}s  "
              f"{len(session):>8,} samples  ->  {len(rows):>5} movements "
              f"({counts['click']} click, {counts['drift']} drift, {counts['drag']} "
              f"drag, {counts['excursion']} excursion"
              + (f", {away:.0f}s outside window" if away else "") + ")")

    if not n_sessions:
        print('No raw sessions found. Record one first:  flick record --activity "..."',
              file=sys.stderr)
        return 1

    out = Path(args.out)
    n = write_rows(out, all_rows, ext, segment_params=seg.to_json())
    print(f"\nWrote {n} feature rows from {n_sessions} session(s) -> {out}")
    if args.csv:
        write_csv(out.with_suffix(".csv"), all_rows)
        print(f"Wrote {out.with_suffix('.csv')}")
    return 0


def cmd_summary(args) -> int:
    path = Path(args.features)
    if not path.exists():
        print(f"No features file at {path}. Run `flick extract` first.", file=sys.stderr)
        return 1
    rows, header = read_rows(path)
    if not rows:
        print("Features file is empty.", file=sys.stderr)
        return 1

    params = header.get("extract_params") or {}
    print(f"{path}   {len(rows)} rows   extraction: {params or 'not recorded'}\n")

    print(f"{'activity':<30} {'kind':<10} {'rows':>6} {'complete':>9}  status")
    print("-" * 78)
    strata: dict[tuple[str, str], list] = {}
    for r in rows:
        strata.setdefault((str(r.get("activity")), str(r.get("kind"))), []).append(r)
    need = 10 * len(DEFAULT_COLUMNS)
    for (activity, kind), group in sorted(strata.items()):
        complete = to_matrix(group, DEFAULT_COLUMNS).shape[0]
        status = "n/a: not a click" if kind != "click" else (
            "ready" if complete >= 2 * need else
            "workable" if complete >= need else f"need ~{need - complete} more")
        print(f"{activity:<30} {kind:<10} {len(group):>6} {complete:>9}  {status}")

    selected = select_rows(rows, args.activity, args.kind)
    print(f"\nFeatures: activity={args.activity or 'any'}, kind={args.kind}, "
          f"n={len(selected)}")
    print(f"{'feature':<22} {'n':>6} {'mean':>11} {'sd':>11} {'p5':>11} {'p50':>11} {'p95':>11}")
    print("-" * 87)
    for col in FEATURE_COLUMNS:
        if args.kind != "click" and col in CLICK_ONLY_COLUMNS:
            continue
        v = to_matrix(selected, [col])[:, 0]
        if v.size:
            print(f"{col:<22} {v.size:>6} {v.mean():>11.4g} {v.std():>11.4g} "
                  f"{np.percentile(v, 5):>11.4g} {np.percentile(v, 50):>11.4g} "
                  f"{np.percentile(v, 95):>11.4g}")
    return 0


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #

def _report_cv(data, columns, args, options) -> None:
    n = data.shape[0]
    folds = min(args.folds, max(1, n // MIN_FOLD_TEST_ROWS))
    cv = cross_validate(data, columns, args.model, folds=folds, seed=args.seed,
                        **options) if folds > 1 else []
    if not cv:
        print(f"\nOnly {n} rows or too few folds: no honest train/test split "
              f"(need >= {2 * MIN_FOLD_TEST_ROWS} rows and 2 folds).")
        print("Everything below is IN-SAMPLE and will look better than it is.")
        return

    scores = np.array([r["score"] for r in cv], dtype=float)
    floors = np.array([r["floor"] for r in cv], dtype=float)
    print(f"\nCross-validation ({len(cv)} folds, "
          f"~{cv[0]['n_train']} train / {cv[0]['n_test']} test):")
    print(f"{'fold':>6}{'model vs held-out':>22}{'real vs held-out':>20}")
    for r in cv:
        print(f"{r['fold']:>6}{r['score']:>22.4f}{r['floor']:>20.4f}")
    print(f"{'median':>6}{np.median(scores):>22.4f}{np.median(floors):>20.4f}"
          "   <- TRUST THIS ROW")
    if np.median(scores) <= np.median(floors):
        print("\n  As close to held-out rows as the training rows themselves are. "
              "Nothing left\n  to learn at this sample size.")
    else:
        ratio = np.median(scores) / max(np.median(floors), 1e-9)
        print(f"\n  {ratio:.2f}x the real-vs-real floor: structure remains unexplained, "
              "or there\n  are too few rows to tell. More data is the first thing to try.")


def cmd_fit(args) -> int:
    _load_plugins(args.plugin)
    path = Path(args.features)
    if not path.exists():
        raise SystemExit(f"No features file at {path}.")
    all_rows, header = read_rows(path)
    rows = select_rows(all_rows, args.activity, args.kind)
    if not rows:
        combos = sorted({(str(r.get("activity")), str(r.get("kind"))) for r in all_rows})
        raise SystemExit(f"No rows for activity={args.activity!r} kind={args.kind!r}.\n"
                         "Available (activity, kind): "
                         + ", ".join(f"{a}/{k}" for a, k in combos))

    try:
        cls = get_model(args.model)
    except KeyError as exc:
        raise SystemExit(exc.args[0]) from None

    columns = list(args.columns) if args.columns else list(DEFAULT_COLUMNS)
    data = to_matrix(rows, columns)
    n, p = data.shape
    if n < 3:
        raise SystemExit(f"Only {n} complete rows over these columns: nothing to fit. "
                         "Collect more, or drop sparse columns.")

    options = {}
    for item in args.option or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--option wants NAME=VALUE, got {item!r}")
        options[key] = float(value) if "." in value else int(value)

    print(f"{path}")
    print(f"  model      : {args.model}")
    print(f"  activity   : {args.activity or 'any'}")
    print(f"  kind       : {args.kind or 'any'}")
    print(f"  rows       : {len(rows)} ({n} complete over {p} columns)")
    print(f"  ratio      : {n / p:.1f}x rows/feature (~10x is the rule of thumb)")

    _report_cv(data, columns, args, options)

    extra = {}
    if cls is CopulaModel:
        extra = dict(activity=args.activity, kind=args.kind or "",
                     fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     extract_params=header.get("extract_params") or {})
    model = cls.fit(data, columns, bounds=FEATURE_BOUNDS,
                    rng=np.random.default_rng(args.seed), **options, **extra)

    print(f"\nFitted {args.model} on all {n} rows.")
    for label, desc in model.describe():
        print(f"  {label:<22} {desc}")

    gen_rng = np.random.default_rng(args.seed + 99)
    drawn = sample_like(model, data, columns, args.samples, gen_rng)
    std = Standardizer.fit(data, columns)
    print(f"\nFeature recovery ({args.samples} generated vs {n} real):")
    print(f"{'feature':<22}{'real p50':>12}{'gen p50':>12}{'real p95':>12}{'gen p95':>12}")
    print("-" * 70)
    for j, c in enumerate(columns):
        r, s = data[:, j], drawn[:, j]
        print(f"{c:<22}{np.median(r):>12.4g}{np.median(s):>12.4g}"
              f"{np.percentile(r, 95):>12.4g}{np.percentile(s, 95):>12.4g}")
    print("\nIn-sample energy distance: "
          f"{energy_distance(std.transform(data), std.transform(drawn)):.4f}")

    if isinstance(model, CopulaModel):
        # A calibrated latent rho is deliberately inflated so the SAMPLED rank
        # correlation lands on the real one, so show what the model produces next
        # to what was observed.
        index = {c: j for j, c in enumerate(columns)}
        print("\nStrongest fitted correlations:")
        print(f"  {'latent':>8} {'sampled':>8} {'observed':>9}   pair")
        any_calibrated = False
        for a, b, r, calibrated in model.copula.strongest_pairs(10):
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
        ours = check_bounds(sample_like(model, data, columns, args.check, rng), columns)
        mvn = check_bounds(gaussian_baseline(data, args.check, rng), columns)
        print(f"\nPhysical-bounds check on {args.check:,} samples:")
        print(f"  {args.model:<21}: {ours.n_impossible} impossible values, "
              f"{ours.nonfinite} non-finite")
        print(f"  multivariate Gaussian: {mvn.n_impossible} impossible values "
              f"({mvn.rate:.1%})")

    if args.out:
        model.save(args.out)
        print(f"\nWrote {args.out}")
        print("This file contains your training rows. See docs/model-format.md "
              "before sharing it.")
    return 0


# --------------------------------------------------------------------------- #
# generate / models / demo-data
# --------------------------------------------------------------------------- #

def cmd_generate(args) -> int:
    _load_plugins(args.plugin)
    generator = ClickGenerator.from_file(args.model,
                                         rng=np.random.default_rng(args.seed))
    clicks = generator.generate_many(args.start, args.target, args.n)
    if not clicks:
        raise SystemExit("Nothing to generate: --from and --to are the same point.")
    rows = [c.to_row() for c in clicks]
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: f"{v:.6g}" for k, v in row.items()})
    return 0


def cmd_path(args) -> int:
    _load_plugins(args.plugin)
    generator = ClickGenerator.from_file(args.model,
                                         rng=np.random.default_rng(args.seed))
    clicks = generator.generate_many(args.start, args.target, args.n)
    if not clicks:
        raise SystemExit("Nothing to generate: --from and --to are the same point.")
    writer = csv.writer(sys.stdout)
    writer.writerow(["movement", "t", "x", "y"])
    for i, click in enumerate(clicks):
        for row in generator.path_for(click).to_rows():
            writer.writerow([i, f"{row['t']:.6f}", row["x"], row["y"]])
    return 0


def cmd_view(args) -> int:
    _load_plugins(args.plugin)
    generator = ClickGenerator.from_file(args.model,
                                         rng=np.random.default_rng(args.seed))
    try:
        data = write_view(args.out, generator, args.start, args.target, n=args.n,
                          real_paths=args.real or (), activity=args.activity,
                          seed=args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    n_real = sum(1 for m in data["movements"] if m["kind"] == "real")
    print(f"Wrote {args.out}: {len(data['movements']) - n_real} generated paths"
          + (f" and {n_real} recorded movements" if args.real else ""))
    if args.real and not n_real:
        print(f"note: no recorded clicks within {REAL_DISTANCE_BAND:.0%} of "
              f"{data['distance_px']:g} px"
              + (f" for activity {args.activity!r}" if args.activity else "")
              + ". Try a --from and --to about as far apart as your recorded moves.")
    if n_real:
        print("This page contains recorded movements. See docs/model-format.md "
              "before sharing it.")
    if args.open:
        if not webbrowser.open(Path(args.out).resolve().as_uri()):
            print("Could not open a browser. Open the file above yourself.")
    return 0


def cmd_models(args) -> int:
    _load_plugins(args.plugin)
    for name in available_models():
        cls = get_model(name)
        summary = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"{name:<12} {summary}")
    return 0


def cmd_demo_data(args) -> int:
    path = write_demo_csv(args.out, n=args.rows, seed=args.seed)
    print(f"Wrote {args.rows} rows of FABRICATED features to {path}")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="flick",
        description="Record mouse movements, fit a model of them, and generate "
                    "synthetic movement features for a start point and click target.")
    sub = ap.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record the pointer and mouse buttons (X11)")
    rec.add_argument("--activity", required=True, help="label for what you're doing")
    rec.add_argument("--hz", type=float, default=200.0, help="poll rate (default 200)")
    rec.add_argument("--seconds", type=float, default=None, help="stop after this long")
    rec.add_argument("--out", default=str(DEFAULT_DIR), help="output dir (default data)")
    rec.add_argument("--window", default=None,
                     help="substring of a window title; marks time outside that window")
    rec.add_argument("--origin", default=None,
                     help="X,Y to subtract, for window-relative coordinates")
    rec.add_argument("--note", default=None, help="free text stored in the header")
    rec.add_argument("--display", default=None, help="X display (default $DISPLAY)")
    rec.set_defaults(func=cmd_record)

    ext = sub.add_parser("extract", help="segment raw sessions into feature rows")
    ext.add_argument("--raw", nargs="*", default=None,
                     help="raw files or directories (default data)")
    ext.add_argument("--out", default=str(DEFAULT_FEATURES),
                     help=f"features JSONL (default {DEFAULT_FEATURES})")
    ext.add_argument("--csv", action="store_true", help="also write a CSV beside it")
    ext.add_argument("--hz", type=float, default=200.0, help="resample grid (default 200)")
    ext.add_argument("--smooth-ms", type=float, default=12.0,
                     help="position smoothing in ms (default 12)")
    ext.add_argument("--move-enter", type=float, default=45.0,
                     help="px/s to start a movement (default 45)")
    ext.add_argument("--move-exit", type=float, default=12.0,
                     help="px/s to end a movement (default 12)")
    ext.add_argument("--min-distance", type=float, default=12.0,
                     help="discard movements shorter than this in px (default 12)")
    ext.add_argument("--settle-radius", type=float, default=15.0,
                     help="px radius defining settle_s (default 15)")
    ext.set_defaults(func=cmd_extract)

    summ = sub.add_parser("summary", help="how much data you have, and its statistics")
    summ.add_argument("--features", default=str(DEFAULT_FEATURES))
    summ.add_argument("--activity", default=None, help="restrict statistics to one")
    summ.add_argument("--kind", default="click", help="movement kind (default click)")
    summ.set_defaults(func=cmd_summary)

    fit = sub.add_parser("fit", help="fit a model to a feature table and report on it")
    fit.add_argument("features", nargs="?", default=str(DEFAULT_FEATURES),
                     help=f"feature table, .csv or JSONL (default {DEFAULT_FEATURES})")
    fit.add_argument("--model", default="copula",
                     help="registered model name (default copula). See `flick models`")
    fit.add_argument("--plugin", action="append", metavar="MODULE",
                     help="import this module first, to register a custom model")
    fit.add_argument("--option", action="append", metavar="NAME=VALUE",
                     help="numeric option for the model's fit, such as k=9")
    fit.add_argument("--activity", default=None, help="only rows with this activity")
    fit.add_argument("--kind", default=None, help="only rows with this kind")
    fit.add_argument("--columns", nargs="*", default=None,
                     help=f"columns to fit (default: the {len(DEFAULT_COLUMNS)} defaults)")
    fit.add_argument("--folds", type=int, default=5, help="CV folds (default 5)")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--samples", type=int, default=2000,
                     help="samples for the report tables (default 2000)")
    fit.add_argument("--check", type=int, default=5000,
                     help="samples for the bounds check (0 to skip)")
    fit.add_argument("--out", default=None, help="write the fitted model here")
    fit.set_defaults(func=cmd_fit)

    gen = sub.add_parser("generate",
                         help="generate features for a position and click target, as CSV")
    gen.add_argument("model", help="model JSON written by `flick fit --out`")
    gen.add_argument("--from", dest="start", type=_point, required=True, metavar="X,Y",
                     help="current mouse position")
    gen.add_argument("--to", dest="target", type=_point, required=True, metavar="X,Y",
                     help="click target")
    gen.add_argument("-n", type=int, default=1, help="movements to generate (default 1)")
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--plugin", action="append", metavar="MODULE",
                     help="import this module first, if the model is a custom one")
    gen.set_defaults(func=cmd_generate)

    pth = sub.add_parser("path",
                         help="generate (t, x, y) trajectories for a position and "
                              "click target, as CSV")
    pth.add_argument("model", help="model JSON written by `flick fit --out`")
    pth.add_argument("--from", dest="start", type=_point, required=True, metavar="X,Y",
                     help="current mouse position")
    pth.add_argument("--to", dest="target", type=_point, required=True, metavar="X,Y",
                     help="click target")
    pth.add_argument("-n", type=int, default=1, help="trajectories to generate (default 1)")
    pth.add_argument("--seed", type=int, default=None)
    pth.add_argument("--plugin", action="append", metavar="MODULE",
                     help="import this module first, if the model is a custom one")
    pth.set_defaults(func=cmd_path)

    vw = sub.add_parser("view", help="write an HTML page displaying generated paths")
    vw.add_argument("model", help="model JSON written by `flick fit --out`")
    vw.add_argument("--from", dest="start", type=_point, required=True, metavar="X,Y",
                    help="current mouse position")
    vw.add_argument("--to", dest="target", type=_point, required=True, metavar="X,Y",
                    help="click target")
    vw.add_argument("-n", type=int, default=20, help="paths to generate (default 20)")
    vw.add_argument("--seed", type=int, default=None)
    vw.add_argument("--real", nargs="*", default=None, metavar="PATH",
                    help="raw sessions to overlay recorded clicks of a similar distance from")
    vw.add_argument("--activity", default=None,
                    help="only overlay recorded clicks from this activity")
    vw.add_argument("--out", default="paths.view.html",
                    help="page to write (default paths.view.html)")
    vw.add_argument("--open", action="store_true", help="open the page in a browser")
    vw.add_argument("--plugin", action="append", metavar="MODULE",
                    help="import this module first, if the model is a custom one")
    vw.set_defaults(func=cmd_view)

    mods = sub.add_parser("models", help="list the available models")
    mods.add_argument("--plugin", action="append", metavar="MODULE")
    mods.set_defaults(func=cmd_models)

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
