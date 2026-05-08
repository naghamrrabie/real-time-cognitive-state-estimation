# CLI Contracts: Real-time Cognitive State Estimation from Video

## Overview

The MVP exposes a command-line interface under a single command namespace:
`cognitive-state`.

All commands MUST return exit code `0` on success and non-zero on invalid input,
source open failure, invalid dataset schema, or runtime failure. Error messages
MUST identify the failed stage: video, landmarks, features, data, model,
training, inference, or output.

## Common Score Schema

All score outputs use these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| timestamp_seconds | float | yes | Timestamp for the completed scoring window. |
| source_progress | string | yes | Human-readable progress, such as frame/time. |
| fatigue | float | yes | Fatigue score, 0.00 through 1.00. |
| attention | float | yes | Attention score, 0.00 through 1.00. |
| stress | float | yes | Stress score, 0.00 through 1.00. |
| engagement | float | yes | Engagement score, 0.00 through 1.00. |

## Command: `cognitive-state extract-features`

Extract landmark-derived feature windows from a webcam or video file.

### Required Inputs

One source selector is required:

| Option | Value | Description |
|--------|-------|-------------|
| `--video` | path | Process a recorded RGB video file offline. |
| `--webcam` | index | Process a webcam source in live-paced mode. |

### Optional Inputs

| Option | Value | Description |
|--------|-------|-------------|
| `--features-csv` | path | Save per-window or per-frame features to CSV. |
| `--window-seconds` | float | Time window size for samples. |
| `--stride-seconds` | float | Window stride. |
| `--max-windows` | int | Stop after a bounded number of windows for smoke runs. |

### Outputs

- Console summary with source metadata, processed windows, and warnings.
- Optional feature CSV containing timestamps, modality groups, feature values,
  and missing-value indicators.

### Failure Cases

- Source cannot be opened.
- Source has no readable frames.
- No valid feature windows are produced.
- Output CSV path is not writable.

## Command: `cognitive-state smoke-train`

Run the minimal Temporal Transformer regression smoke-training flow.

### Required Inputs

| Option | Value | Description |
|--------|-------|-------------|
| `--features-csv` | path | Feature windows or feature records to load. |

### Optional Inputs

| Option | Value | Description |
|--------|-------|-------------|
| `--synthetic-labels` | flag | Generate synthetic/demo labels for smoke-training. |
| `--labels-csv` | path | Load real or prepared window-level labels. |
| `--epochs` | int | Small epoch count for smoke-training. |
| `--checkpoint-out` | path | Save a smoke checkpoint artifact. |

### Outputs

- Console training summary with sample count, input shape, output shape, and
  smoke metrics.
- Optional checkpoint artifact.

### Validation Rules

- A labeled dataset sample MUST contain one feature window and one four-score
  label vector.
- Labels MUST be finite values from 0.00 through 1.00.
- Synthetic-label metrics MUST be labeled as smoke metrics only.

## Command: `cognitive-state infer`

Run MVP inference on a webcam or video file.

### Required Inputs

One source selector is required:

| Option | Value | Description |
|--------|-------|-------------|
| `--video` | path | Process a recorded RGB video file offline. |
| `--webcam` | index | Process a webcam source in live-paced mode. |

### Optional Inputs

| Option | Value | Description |
|--------|-------|-------------|
| `--checkpoint` | path | Load a model checkpoint when available. |
| `--scores-csv` | path | Save score history to CSV. |
| `--features-csv` | path | Save feature windows to CSV. |
| `--plot` | path | Save score-over-time plot after the run. |
| `--window-seconds` | float | Time window size for samples. |
| `--stride-seconds` | float | Window stride. |
| `--max-windows` | int | Stop after a bounded number of windows. |

### Required Console Output

Console output MUST include a table or stream with:

- Timestamp.
- Source progress.
- Fatigue.
- Attention.
- Stress.
- Engagement.

### Optional CSV Output

Score CSV columns:

```text
timestamp_seconds,source_progress,fatigue,attention,stress,engagement
```

### Failure Cases

- No input source is provided.
- Both `--video` and `--webcam` are provided.
- Source cannot be opened.
- Feature windows cannot be built.
- Model output shape is not four scores.
- Score values are not finite or cannot be bounded.
- Output file path is not writable.

## Command: `cognitive-state plot-scores`

Generate a score-over-time plot from an existing score CSV.

### Required Inputs

| Option | Value | Description |
|--------|-------|-------------|
| `--scores-csv` | path | Score CSV with the common score schema. |
| `--out` | path | Plot image output path. |

### Outputs

- Plot artifact showing Fatigue, Attention, Stress, and Engagement over time.

### Failure Cases

- CSV is missing required score columns.
- CSV contains non-finite score values.
- Output path is not writable.
