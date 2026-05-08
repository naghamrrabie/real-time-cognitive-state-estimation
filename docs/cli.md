# CLI Reference

All commands are invoked via the `cognitive-state` entry point.

```
cognitive-state [--version] <command> [options]
```

---

## `extract-features`

Extract per-frame landmark features from a recorded video file.

```
cognitive-state extract-features --video PATH --output PATH [--max-frames N]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--video PATH` | Yes | Path to an RGB video file. |
| `--output PATH` | Yes | CSV output path for per-frame features. |
| `--max-frames N` | No | Limit the number of frames processed. |

**Output schema** — one row per frame, columns defined by `FEATURE_CSV_COLUMNS`:
`frame_index`, `timestamp_seconds`, `face_detected`, `pose_detected`,
followed by all eye, face, head-pose, and posture feature columns.

---

## `infer`

Estimate cognitive state scores from a video file, webcam, or feature CSV.

```
cognitive-state infer [--video PATH | --webcam INDEX | --features-csv PATH]
                      [--checkpoint PATH]
                      [--scores-csv PATH]
                      [--plot [--plot-out PATH]]
                      [--window-seconds N] [--stride-seconds N]
                      [--max-windows N]
```

| Option | Description |
|--------|-------------|
| `--video PATH` | Source video file (mutually exclusive with `--webcam`). |
| `--webcam INDEX` | Webcam device index (mutually exclusive with `--video`). |
| `--features-csv PATH` | Feature CSV input (when no `--video`/`--webcam`). When used with `--video`/`--webcam`, the extracted features are also saved here. |
| `--checkpoint PATH` | Model checkpoint `.pt` file. Random weights are used when omitted. |
| `--scores-csv PATH` | Optional path to save per-window score predictions. |
| `--plot` | Generate a score-over-time plot (requires `--plot-out`). |
| `--plot-out PATH` | PNG output path for the plot. |
| `--window-seconds N` | Window duration in seconds (default: 30 frames at inferred FPS). |
| `--stride-seconds N` | Window stride in seconds (default: 15 frames at inferred FPS). |
| `--max-windows N` | Cap the total number of windows processed. |

**Console output** — one line per window:

```
window=0  t=0.00s  fatigue=0.52  attention=0.47  stress=0.61  engagement=0.38
```

**Score CSV schema** (`SCORE_CSV_COLUMNS`):
`timestamp_seconds`, `source_progress`, `fatigue`, `attention`, `stress`, `engagement`

All score values are bounded to `[0.0, 1.0]`.

---

## `smoke-train`

Run a minimal smoke-training loop to validate data and model contracts.
Uses synthetic labels — **not scientifically valid**.

```
cognitive-state smoke-train --features-csv PATH
                             (--synthetic-labels | --labels-csv PATH)
                             [--epochs N]
                             [--checkpoint-out PATH]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--features-csv PATH` | Yes | Feature CSV produced by `extract-features`. |
| `--synthetic-labels` | Yes* | Generate random labels for smoke testing. |
| `--labels-csv PATH` | Yes* | (*reserved*) Real label CSV. |
| `--epochs N` | No | Training epochs (default: 1). |
| `--checkpoint-out PATH` | No | Save trained model to this `.pt` file. |

*Exactly one label source is required.

**Console output** (example):

```
smoke-train complete
  samples:      2
  input shape:  (2, 30, 28)
  output shape: (2, 4)
  final loss:   0.083142
  MAE   (fatigue/attention/stress/engagement): 0.2104  0.3012  ...
  RMSE  (fatigue/attention/stress/engagement): 0.2411  0.3412  ...
note: These metrics use synthetic or placeholder labels ...
checkpoint saved to artifacts/smoke.pt
```

---

## `plot-scores`

Generate a score-over-time PNG plot from a previously saved scores CSV.

```
cognitive-state plot-scores --scores-csv PATH --output PATH [--title TEXT]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--scores-csv PATH` | Yes | Scores CSV from `infer --scores-csv`. |
| `--output PATH` | Yes | PNG output path. |
| `--title TEXT` | No | Custom plot title. |

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `2` | Input error (bad path, missing required arg, insufficient data). |
