# Recording and extracting

How to go from your own mouse to a feature table you can fit.

## 1. Record

```bash
pip install -e ".[record]"
flick record --activity "photo editing"
flick record --activity "spreadsheets" --seconds 1800
```

Use the application normally, and press Ctrl-C when you're done. The file is written as
it goes, so stopping this way is safe.

**The recorder captures the pointer and mouse buttons across the whole display, in any
application. It never reads the keyboard.** Only record yourself, or people who have
agreed to it.

`--activity` labels what you're doing, and each activity is fitted separately, because
different tasks have different target sizes, distances and urgency. Use exactly the same
label each time: `flick summary` lists the labels it finds, which is the quickest way to
catch a typo.

### Tracking one window

```bash
flick record --activity "photo editing" --window "GIMP"
```

`--window` takes part of a window title. Time the pointer spends outside that window is
marked, and movements overlapping it are filed as `excursion` rather than as `click` or
`drift`. Without this, those movements are mixed in with the rest. If no window matches,
the recording carries on untracked and tells you so.

To store window-relative coordinates, pass the window's top-left corner as
`--origin X,Y`. Features don't depend on position, so this only affects absolute
coordinates.

### Timing

The recorder polls at `--hz`, by default 200. A button press is located to within one poll
interval, so at the default rate `dwell_s` carries up to 5 ms of error. Use `--hz 500` if
that matters to you. The effective rate is printed as you record, so check it stays close
to what you asked for.

### The raw file is the asset

Features are derived, and you can recompute them with different parameters at any time.
You can't regenerate a recording, so keep your `*.raw.jsonl` files. Each is one header line
followed by chunks of samples:

```json
{"kind": "header", "version": 2, "activity": "photo editing", "hz": 200.0, "started_at": "2026-01-01T00:00:00+00:00", "origin": [0, 0], "note": "", "session_id": "photo-editing-20260101-000000", "channels": ["pointer", "buttons"]}
{"kind": "samples", "t": [0.0, 0.005], "x": [100, 101], "y": [200, 200], "b": [0, 0]}
```

`b` is a bitmask: bit 8 is left, bit 9 is middle, bit 10 is right, and bit 20 means the
pointer was outside the tracked window. `channels` says what a file can answer. Without
`client_bounds`, a file can't tell you whether the pointer stayed in a window, which isn't
the same as it staying there.

## 2. Extract

```bash
flick extract
flick extract --raw data/ other/session.raw.jsonl --out data/features.jsonl --csv
```

This cuts each session into movements and computes a [feature row](feature-schema.md) for
each. It's cheap and repeatable, so rerun it whenever you change a parameter.

### How movements are found

1. Compute smoothed speed over the whole session.
2. Mark "moving" using two thresholds: it starts above `--move-enter` (45 px/s) and ends
   below `--move-exit` (12 px/s). A single threshold chatters in the slow tails and breaks
   one movement into several.
3. Move each start back to where speed last left the lower threshold, to recover the start
   of the acceleration.
4. Split at button presses. A press up to 0.4 s after a movement stops belongs to that
   movement, since people often stop, then click.
5. Drop fragments shorter than 50 ms or `--min-distance` (12 px).

### Movement kinds

| Kind | Meaning | Use for clicks? |
|---|---|---|
| `click` | Ends in a left or right press | Yes |
| `drift` | Ends in stillness, with no destination | No |
| `drag` | Made with the middle button held | No |
| `excursion` | Overlaps time outside the tracked window | No |

A movement overlapping time outside the window is an `excursion` even if it also ends in a
press, because whatever it pressed isn't in the window you're studying.

Left-button drags are filed as `click`, because the movement before the press is what's
measured. A middle press isn't aimed at anything, so it doesn't end a click movement.

### Parameters change what features mean

`--smooth-ms` matters most. See
[feature-schema.md](feature-schema.md#extraction-parameters). The parameters are stored in
the file's header, and rows extracted with different ones must never be pooled.

## 3. Check what you have

```bash
flick summary
flick summary --activity "photo editing"
```

This lists rows per activity and kind, how many are complete, and whether that's enough:

| Status | Complete rows, for the default 14 columns |
|---|---|
| `need ~N more` | Under 140 |
| `workable` | 140 or more |
| `ready` | 280 or more |

Then it prints per-feature statistics, which are a good way to catch something broken
before you fit.

## Other platforms

`XQueryPointerSource` is the only built-in source. To support another platform, implement
the `PointerSource` protocol and pass it to `Recorder`:

```python
from flick import Recorder


class MySource:
    channels = ("pointer", "buttons")

    def read(self):
        return 100, 200, 0          # x, y, button bitmask

    def close(self):
        pass


recorder = Recorder(MySource(), activity="example", hz=200.0)
print(recorder.header.session_id)
```

`read` has to be fast, since it's called up to `hz` times per second on the recording
thread.

Alternatively, skip recording altogether and supply a
[feature table](feature-schema.md#file-formats) from your own tooling.
