# Research Notes — Synthetic / Demo Label Limitations

## Overview

The current MVP uses **synthetic (randomly generated) labels** for all
smoke-training and pipeline-validation workflows.  These labels have no
relationship to actual human cognitive states and exist solely to verify
that the data pipeline, model architecture, and evaluation code function
correctly end-to-end.

---

## What synthetic labels are

`generate_synthetic_labels(n, seed=0)` produces `n` vectors of four
uniformly-distributed random floats in `[0.0, 1.0]`, one for each of:

- **Fatigue**
- **Attention**
- **Stress**
- **Engagement**

The values are seeded for reproducibility but carry no cognitive-state
meaning whatsoever.

---

## Why real labels are absent

Real continuous cognitive-state labels require:

1. **Physiological ground truth** — e.g. EEG, GSR, eye-tracking, validated
   self-report questionnaires, or expert annotations.
2. **A collection protocol** — controlled sessions, participant consent,
   synchronised video capture.
3. **Label alignment** — mapping ground-truth signals to per-window label
   vectors at the correct temporal resolution.

None of these exist in this repository.  Collecting them is outside the
scope of the current engineering MVP.

---

## What smoke-train results mean

Any MAE / RMSE / R² values printed by `cognitive-state smoke-train` are
evaluated against synthetic targets.  They measure **model output shape
and loss convergence**, not predictive accuracy.

The CLI prints the following disclaimer after every smoke run:

> *These metrics use synthetic or placeholder labels and are not evidence
> of real cognitive-state predictive accuracy.*

---

## Roadmap to real labels

When real annotated data becomes available:

1. Format labels as a CSV with columns matching `SCORE_CSV_COLUMNS` and
   one row per temporal window.
2. Use `--labels-csv PATH` with `cognitive-state smoke-train`
   (currently reserved; returns an error).
3. Remove or quarantine the synthetic-label code path before publication.

---

## Placeholder label source

The `"placeholder"` label source tag is semantically equivalent to
`"synthetic"` and is treated identically in the smoke-training pipeline.
Both trigger the `SYNTHETIC_LABEL_WARNING` / `SMOKE_METRIC_WARNING`
disclaimers.
