# Real-time Cognitive State Estimation from Video

Python research MVP that extracts multimodal landmark-derived features from
webcam or video input and outputs four normalised cognitive state scores:

| Score | Range |
|-------|-------|
| Fatigue | 0.0 – 1.0 |
| Attention | 0.0 – 1.0 |
| Stress | 0.0 – 1.0 |
| Engagement | 0.0 – 1.0 |

> **Important**: The current model is initialised with random weights.
> No real labelled cognitive-state data has been collected yet.
> All scores are for pipeline-validation purposes only.
> See [`docs/research_notes.md`](docs/research_notes.md) for details.

---

## Setup

```powershell
C:\Users\TheExpert\.local\bin\uv.exe pip install -r requirements.txt
C:\Users\TheExpert\.local\bin\uv.exe pip install -r requirements-dev.txt
C:\Users\TheExpert\.local\bin\uv.exe pip install -e .
```

---

## Validate

```powershell
C:\Users\TheExpert\.local\bin\uv.exe run pytest tests/ --tb=short
```

All 278 tests should pass.

---

## CLI commands

### Extract features from video

```powershell
cognitive-state extract-features `
    --video path\to\sample.mp4 `
    --output artifacts\features.csv `
    --max-frames 300
```

### Run inference from feature CSV

```powershell
cognitive-state infer `
    --features-csv artifacts\features.csv `
    --scores-csv artifacts\scores.csv
```

### Run inference from video directly

```powershell
cognitive-state infer `
    --video path\to\sample.mp4 `
    --scores-csv artifacts\scores.csv `
    --plot --plot-out artifacts\scores.png
```

### Run inference from webcam (10 windows by default)

```powershell
cognitive-state infer --webcam 0
```

### Smoke-train with synthetic labels

```powershell
cognitive-state smoke-train `
    --features-csv artifacts\features.csv `
    --synthetic-labels `
    --epochs 3 `
    --checkpoint-out artifacts\smoke.pt
```

> Warning: synthetic labels are randomly generated and produce no
> meaningful accuracy signal.  See `docs/research_notes.md`.

### Plot score trends from saved CSV

```powershell
cognitive-state plot-scores `
    --scores-csv artifacts\scores.csv `
    --output artifacts\scores.png
```

---

## Full CLI reference

See [`docs/cli.md`](docs/cli.md).

---

## Project layout

```
src/cognitive_state/
  cli/          Command-line interface
  data/         Schemas, CSV helpers, windowing, dataset
  features/     MediaPipe landmark extraction and feature computation
  inference/    Inference pipeline, scoring, output formatting, plotting
  models/       Temporal Transformer and modality fusion
  training/     Batching, metrics, smoke-train loop, checkpoint helpers
  video/        Video and webcam frame readers

tests/
  unit/         Fast isolated unit tests (no real video or webcam)
  integration/  CLI and pipeline integration tests
  fixtures/     Shared synthetic data helpers

docs/
  cli.md              Full CLI reference
  research_notes.md   Synthetic label limitations
specs/
  001-cognitive-state-estimation/   Spec Kit documents
```
