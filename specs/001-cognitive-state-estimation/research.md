# Research: Real-time Cognitive State Estimation from Video

## Decision: Python 3.11+ MVP Package

Use Python 3.11+ with a package layout rooted at `src/cognitive_state/`.

**Rationale**: Python 3.11+ supports the requested scientific stack, modern type
hinting, and fast enough local experimentation for an MVP.

**Alternatives considered**:
- Python 3.10: viable, but the user explicitly requested 3.11+.
- Notebook-first project: useful for exploration, but less testable and less
suited to CLI workflows.

## Decision: OpenCV Input Abstraction for Webcam and Video Files

Use a shared video source abstraction that supports both live-paced webcam input
and offline/as-fast-as-practical recorded video processing.

**Rationale**: The spec requires both input paths, but they have different
timing semantics. A shared abstraction keeps downstream landmark and feature
code independent from source type.

**Alternatives considered**:
- Webcam-only MVP: rejected by clarification.
- Video-file-only MVP: rejected by clarification.
- Real-time pacing for recorded files: rejected because offline processing is
better for repeatable research runs.

## Decision: MediaPipe Face Mesh and Pose for Landmark Extraction

Use MediaPipe Face Mesh and Pose to produce face and pose landmarks with
confidence/availability metadata.

**Rationale**: The constitution requires MediaPipe for MVP face/pose landmarking,
and MediaPipe provides enough geometry for eye, face, head pose, shoulder, and
spine/posture proxy features from normal RGB input.

**Alternatives considered**:
- dlib or OpenFace: possible research extensions, but they violate the MVP
landmarking constraint.
- CNN-only feature extraction: deferred because MVP features must remain
interpretable landmark-derived signals.

## Decision: Structured Multimodal Feature Groups

Represent features as grouped modalities: eyes, face, head pose, and posture.
Each feature record includes timestamp, frame index, values, units where useful,
and missing-value indicators.

**Rationale**: Grouping preserves modality structure for attention-based fusion,
testing, and future interpretability. It also avoids untraceable flat
concatenation.

**Alternatives considered**:
- Single flat vector from the start: rejected by spec and constitution.
- Raw landmarks only: insufficient because the MVP requires derived numerical
feature groups.

## Decision: Window-Level Samples and Labels

Build fixed time-window samples from per-frame feature records. Each sample has
one window-level four-score label vector for Fatigue, Attention, Stress, and
Engagement.

**Rationale**: The model predicts one score vector per completed window, so
window-level labels align the dataset, model, and evaluation contracts.

**Alternatives considered**:
- Frame-level labels: requires aggregation policy before training and increases
scope.
- Session-level labels: too coarse for continuous temporal scoring.

## Decision: Synthetic Labels for Smoke-Training Only

Generate synthetic/demo labels only to validate dataset loading, training loop,
model I/O shape, score bounds, and metric reporting.

**Rationale**: Real cognitive-state labels may not exist during MVP work, and
synthetic labels must not be used as evidence of model accuracy.

**Alternatives considered**:
- No training loop: lower implementation cost, but leaves the modeling path
untested.
- Full real-label training: not feasible or honest without a real labeled
dataset.

## Decision: PyTorch Temporal Transformer Encoder With Modular Fusion

Use a PyTorch Temporal Transformer Encoder over feature windows. Preserve
modality grouping before fusion using a modular fusion layer or attention-based
fusion path, then output four bounded regression scores.

**Rationale**: The constitution requires transformer-first temporal modeling and
the spec requires structured fusion rather than raw concatenation.

**Alternatives considered**:
- LSTM/GRU primary model: rejected by constitution.
- Pure per-frame regression: rejected because temporal modeling is a core
project goal.
- Unstructured concatenation: rejected because modality importance should remain
learnable and inspectable.

## Decision: CLI-First MVP Outputs

Provide CLI commands with required console score output. CSV logging of features
and scores plus plots are optional user-enabled outputs.

**Rationale**: The MVP explicitly avoids a heavy GUI while still giving a clear
verification path and research artifacts.

**Alternatives considered**:
- GUI dashboard: deferred as over-scoped for MVP.
- Required CSV for every run: unnecessary for live use and rejected by
clarification.
- JSON Lines as primary output: useful later, but console output and optional CSV
match the current MVP.

## Decision: Evaluation Metrics

For smoke-training with synthetic labels, report shape/range validation plus
MAE and RMSE only as pipeline sanity metrics. When real window-level labels are
available, report per-score MAE, RMSE, R2, and Pearson correlation.

**Rationale**: MAE/RMSE are easy to understand for normalized 0.00-1.00 scores,
R2 offers regression baseline context, and Pearson correlation captures whether
predicted score trends track label trends. Synthetic-label metrics are not
scientific evidence.

**Alternatives considered**:
- Classification accuracy/F1: rejected because outputs are continuous
regression scores.
- Single aggregate metric only: rejected because each cognitive score may behave
differently.

## Decision: Pytest Coverage Focus

Use pytest for focused tests around feature calculations, missing landmarks,
window construction, dataset label alignment, transformer model I/O shape, score
range enforcement, CSV schemas, and CLI smoke behavior.

**Rationale**: These tests directly protect the MVP pipeline and constitution
quality gates.

**Alternatives considered**:
- End-to-end tests only: too brittle and slow.
- No model tests until real data exists: rejected because model shape/range
contracts are independent of real labels.
