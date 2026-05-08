# Implementation Plan: Real-time Cognitive State Estimation from Video

**Branch**: `001-cognitive-state-estimation` | **Date**: 2026-05-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-cognitive-state-estimation/spec.md`

## Summary

Build a Python 3.11+ research MVP that reads webcam streams or RGB video files,
extracts MediaPipe face and pose landmarks, computes grouped eye, face, head,
and posture features, builds time-window samples, runs a PyTorch Temporal
Transformer Encoder regression path with attention-based multimodal fusion, and
emits Fatigue, Attention, Stress, and Engagement scores in the range 0.00-1.00.
The MVP includes synthetic/demo labels only for smoke-training and validation of
data flow, not scientific model claims. The first user-facing workflow is a
simple CLI with required console score output and optional CSV/plot artifacts.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: opencv-python, mediapipe, numpy, pandas, torch, scikit-learn, matplotlib
**Storage**: Local CSV files for optional feature export and score history; local model/checkpoint artifacts for smoke-training outputs; no database in MVP
**Testing**: pytest for unit and integration tests; shape/range assertions for feature windows and model I/O; smoke evaluation with scikit-learn metrics when labels exist
**Target Platform**: Desktop/laptop environment with normal RGB webcam or local video files; CPU-first MVP with optional PyTorch device selection later
**Project Type**: Python research package with CLI, inference, feature extraction, dataset, model, and training modules
**Performance Goals**: Webcam mode emits scores as live-paced windows complete; recorded video files process offline as fast as practical; sample video workflow displays scores within 5 minutes after setup
**Constraints**: Continuous Fatigue/Attention/Stress/Engagement regression; MediaPipe face and pose landmarks; PyTorch Temporal Transformer Encoder primary model; attention-based or modular fusion; no heavy GUI; no deep research extensions in MVP
**Scale/Scope**: MVP end-to-end pipeline first, then research extensions for self-supervised pretraining, domain adaptation, explainability, and CNN/Grad-CAM only after the baseline is stable

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research-first modularity**: PASS. The plan separates video input,
  MediaPipe landmark extraction, multimodal feature extraction, temporal
  windowing, data/dataset handling, PyTorch modeling, smoke-training,
  inference, CLI, and tests.
- **Python stack**: PASS. Python 3.11+ is the only implementation language;
  MediaPipe is the MVP face/pose landmark extractor; PyTorch is the temporal
  modeling stack.
- **Multimodal features**: PASS. MVP includes eyes, face, head pose, and
  posture groups, each represented as structured numeric features with
  missing-value metadata.
- **Transformer regression**: PASS. The primary model is a PyTorch Temporal
  Transformer Encoder with a four-score regression head. Recurrent models are
  not part of the MVP.
- **MVP-first delivery**: PASS. The plan prioritizes the video-to-feature-to-
  window-to-score path before optional exports, plots, or research extensions.
- **Testability**: PASS. Planned tests cover feature utility behavior,
  missing-landmark handling, temporal window shapes, dataset label alignment,
  model input/output shapes, score range checks, and CLI smoke flows.

## Project Structure

### Documentation (this feature)

```text
specs/001-cognitive-state-estimation/
  plan.md
  research.md
  data-model.md
  quickstart.md
  contracts/
    cli.md
```

### Source Code (repository root)

```text
src/
  cognitive_state/
    video/          # webcam/video-file readers and timing/progress metadata
    features/       # MediaPipe landmark adapters and eye/face/head/posture features
    data/           # feature windows, CSV I/O, dataset samples, synthetic labels
    models/         # Temporal Transformer Encoder regression model and fusion
    training/       # smoke-training loop, validation, metrics, checkpoints
    inference/      # end-to-end scoring pipeline and output formatting
    cli/            # command-line commands and argument parsing

tests/
  unit/
  integration/
  fixtures/
```

**Structure Decision**: Use the user-requested `src/cognitive_state/*` package
layout. This still satisfies the constitution because each major research and
runtime responsibility remains separated by module boundary.

## Phase 0: Research

Research decisions are recorded in [research.md](./research.md). All technical
context decisions are resolved; no unresolved clarification items remain.

## Phase 1: Design and Contracts

Design artifacts:

- [data-model.md](./data-model.md): entities, fields, relationships, validation
  rules, and state transitions.
- [contracts/cli.md](./contracts/cli.md): CLI command contracts for feature
  extraction, smoke-training, inference, and plotting.
- [quickstart.md](./quickstart.md): setup and verification workflow for the MVP.

Post-design Constitution Check:

- **Research-first modularity**: PASS. Design artifacts keep each pipeline stage
  independently testable and replaceable.
- **Python stack**: PASS. Contracts and quickstart use only the approved Python
  3.11+ dependency set.
- **Multimodal features**: PASS. Data model requires grouped eye, face, head,
  and posture feature records.
- **Transformer regression**: PASS. Model contract remains a Temporal
  Transformer Encoder regression path with four bounded outputs.
- **MVP-first delivery**: PASS. Quickstart validates the smallest complete path
  before optional CSV/plot artifacts.
- **Testability**: PASS. Design artifacts identify concrete validation points
  for feature extraction, windowing, datasets, model I/O, output schema, and CLI
  behavior.

## Complexity Tracking

No constitution violations or justified complexity exceptions.
