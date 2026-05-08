# Tasks: Real-time Cognitive State Estimation from Video

**Input**: Design documents from `/specs/001-cognitive-state-estimation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Tests are required by the specification and plan. Test tasks appear
before related implementation tasks in each phase.

**Organization**: Tasks are grouped by user story so each story can be
implemented and tested independently after foundational prerequisites are done.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other tasks in the same phase because it
  touches different files and does not depend on incomplete tasks.
- **[Story]**: Required only for user story phases: `[US1]`, `[US2]`, `[US3]`,
  `[US4]`.
- Every task includes an exact file or directory path.

## Phase 1: Setup (Project Scaffold and Dependencies)

**Purpose**: Create the Python package skeleton, dependency metadata, and test
configuration.

- [ ] T001 Create package metadata and Python 3.11+ project configuration in `pyproject.toml`
- [ ] T002 Add runtime and development dependencies in `requirements.txt` and `requirements-dev.txt`
- [ ] T003 Create package entry files in `src/cognitive_state/__init__.py`, `src/cognitive_state/video/__init__.py`, `src/cognitive_state/features/__init__.py`, `src/cognitive_state/data/__init__.py`, `src/cognitive_state/models/__init__.py`, `src/cognitive_state/training/__init__.py`, `src/cognitive_state/inference/__init__.py`, and `src/cognitive_state/cli/__init__.py`
- [ ] T004 Configure pytest discovery and import paths in `pyproject.toml`
- [ ] T005 Create artifact directory policy in `artifacts/.gitignore`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define shared schemas, constants, exceptions, and fixtures used by
all stories. This phase MUST be complete before any user story work begins.

- [ ] T006 [P] Create score schema tests for four bounded scores in `tests/unit/test_score_schema.py`
- [ ] T007 [P] Create shared fixture helpers for synthetic frames, landmarks, feature windows, and labels in `tests/fixtures/synthetic_data.py`
- [ ] T008 [P] Create package-wide exception tests in `tests/unit/test_exceptions.py`
- [ ] T009 Implement cognitive score and prediction dataclasses in `src/cognitive_state/data/schemas.py`
- [ ] T010 Implement shared modality, score, and CSV column constants in `src/cognitive_state/data/constants.py`
- [ ] T011 Implement domain-specific exceptions in `src/cognitive_state/data/exceptions.py`
- [ ] T012 Implement validation helpers for finite values, score ranges, and modality names in `src/cognitive_state/data/validation.py`

**Checkpoint**: Shared schemas and validation utilities are ready for video,
features, model, inference, and CLI modules.

---

## Phase 3: User Story 1 - Estimate Scores From Video (Priority: P1)

**Goal**: Run MVP inference on a webcam stream or RGB video file and display
Fatigue, Attention, Stress, and Engagement scores in the console.

**Independent Test**: Run `cognitive-state infer` on a short sample video and a
bounded webcam session and confirm that each completed window emits four scores
between 0.00 and 1.00.

### Tests for User Story 1

- [ ] T013 [P] [US1] Create webcam/video source tests in `tests/unit/test_video_sources.py`
- [ ] T014 [P] [US1] Create frame timing tests for live-paced and offline modes in `tests/unit/test_frame_timing.py`
- [ ] T015 [P] [US1] Create MediaPipe landmark extractor tests with mocked face-only, pose-only, and missing-landmark results in `tests/unit/test_landmark_extractor.py`
- [ ] T016 [P] [US1] Create eye, face, head, and posture feature utility tests in `tests/unit/test_feature_extractors.py`
- [ ] T017 [P] [US1] Create Temporal Transformer model shape tests in `tests/unit/test_temporal_transformer.py`
- [ ] T018 [P] [US1] Create score bounding and console output tests in `tests/unit/test_inference_output.py`
- [ ] T019 [US1] Create end-to-end inference integration test for sample video input in `tests/integration/test_infer_pipeline.py`
- [ ] T020 [US1] Create CLI contract test for `cognitive-state infer` in `tests/integration/test_infer_cli.py`

### Implementation for User Story 1

- [ ] T021 [US1] Implement `VideoSource` and source metadata models in `src/cognitive_state/video/sources.py`
- [ ] T022 [US1] Implement webcam live-paced frame reader and video-file offline frame reader in `src/cognitive_state/video/readers.py`
- [ ] T023 [US1] Implement frame timing and source progress helpers in `src/cognitive_state/video/timing.py`
- [ ] T024 [US1] Implement MediaPipe Face Mesh and Pose extractor wrapper in `src/cognitive_state/features/landmarks.py`
- [ ] T025 [US1] Implement eye feature calculations including EAR and gaze proxy values in `src/cognitive_state/features/eyes.py`
- [ ] T026 [US1] Implement face feature calculations including mouth openness and landmark-distance proxies in `src/cognitive_state/features/face.py`
- [ ] T027 [US1] Implement head pose and posture proxy feature calculations in `src/cognitive_state/features/posture.py`
- [ ] T028 [US1] Implement multimodal feature extraction orchestrator with missing-value markers in `src/cognitive_state/features/pipeline.py`
- [ ] T029 [US1] Implement temporal feature window builder for inference windows in `src/cognitive_state/data/windowing.py`
- [ ] T030 [US1] Implement attention-based or modular modality fusion layer in `src/cognitive_state/models/fusion.py`
- [ ] T031 [US1] Implement Temporal Transformer Encoder regression model with four-score output in `src/cognitive_state/models/transformer.py`
- [ ] T032 [US1] Implement score bounding and prediction formatting in `src/cognitive_state/inference/scoring.py`
- [ ] T033 [US1] Implement end-to-end inference pipeline from source to console-ready predictions in `src/cognitive_state/inference/pipeline.py`
- [ ] T034 [US1] Implement console score table/stream formatter in `src/cognitive_state/inference/output.py`
- [ ] T035 [US1] Implement `cognitive-state infer` CLI command and argument validation in `src/cognitive_state/cli/main.py`

**Checkpoint**: US1 is complete when video-file and webcam inference paths
produce bounded console scores without requiring CSV, plots, or real labels.

---

## Phase 4: User Story 2 - Build Time-Series Feature Windows (Priority: P2)

**Goal**: Transform video observations into timestamped multimodal feature
windows that can be inspected, exported, and reused by training or inference.

**Independent Test**: Run `cognitive-state extract-features` on a sample video
and confirm feature windows include eye, face, head pose, and posture groups
with timestamps and missing-value indicators.

### Tests for User Story 2

- [ ] T036 [P] [US2] Create feature CSV schema tests in `tests/unit/test_feature_csv.py`
- [ ] T037 [P] [US2] Create feature window validation tests in `tests/unit/test_windowing.py`
- [ ] T038 [P] [US2] Create missing-landmark summary tests in `tests/unit/test_missing_summary.py`
- [ ] T039 [US2] Create CLI contract test for `cognitive-state extract-features` in `tests/integration/test_extract_features_cli.py`

### Implementation for User Story 2

- [ ] T040 [US2] Implement feature CSV writer and reader in `src/cognitive_state/data/feature_csv.py`
- [ ] T041 [US2] Implement per-window missing-value summaries in `src/cognitive_state/features/missing.py`
- [ ] T042 [US2] Extend feature window metadata and validation in `src/cognitive_state/data/windowing.py`
- [ ] T043 [US2] Implement feature extraction service for reusable window generation in `src/cognitive_state/features/export.py`
- [ ] T044 [US2] Implement `cognitive-state extract-features` CLI command in `src/cognitive_state/cli/main.py`

**Checkpoint**: US2 is complete when feature windows can be generated and
optionally exported to CSV without running model inference.

---

## Phase 5: User Story 3 - Smoke-Test Baseline Regression Flow (Priority: P3)

**Goal**: Run a minimal smoke-training and evaluation flow using feature windows
and synthetic/demo labels to validate model and dataset contracts.

**Independent Test**: Run `cognitive-state smoke-train` with exported features
and synthetic labels, then confirm one four-score prediction per sample and
smoke metrics marked as non-scientific.

### Tests for User Story 3

- [ ] T045 [P] [US3] Create dataset sample and label alignment tests in `tests/unit/test_dataset.py`
- [ ] T046 [P] [US3] Create synthetic label generation tests in `tests/unit/test_synthetic_labels.py`
- [ ] T047 [P] [US3] Create training batch shape tests in `tests/unit/test_training_batches.py`
- [ ] T048 [P] [US3] Create evaluation metric tests for MAE, RMSE, R2, and Pearson correlation in `tests/unit/test_metrics.py`
- [ ] T049 [US3] Create CLI smoke-training integration test in `tests/integration/test_smoke_train_cli.py`

### Implementation for User Story 3

- [ ] T050 [US3] Implement window-level dataset loader and validation in `src/cognitive_state/data/dataset.py`
- [ ] T051 [US3] Implement synthetic/demo label generator with smoke-label metadata in `src/cognitive_state/data/synthetic_labels.py`
- [ ] T052 [US3] Implement training batch collation for grouped modalities in `src/cognitive_state/training/batching.py`
- [ ] T053 [US3] Implement smoke-training loop for Temporal Transformer regression in `src/cognitive_state/training/smoke_train.py`
- [ ] T054 [US3] Implement evaluation metrics with synthetic-label warnings in `src/cognitive_state/training/metrics.py`
- [ ] T055 [US3] Implement checkpoint save/load helpers for smoke artifacts in `src/cognitive_state/training/checkpoints.py`
- [ ] T056 [US3] Implement `cognitive-state smoke-train` CLI command in `src/cognitive_state/cli/main.py`

**Checkpoint**: US3 is complete when the smoke-training flow validates shapes,
score ranges, and metrics without claiming real cognitive-state accuracy.

---

## Phase 6: User Story 4 - Export and Review Score Trends (Priority: P4)

**Goal**: Optionally save score histories and generate simple score-over-time
plots from completed inference runs.

**Independent Test**: Enable `--scores-csv` and `--plot` during inference, then
confirm CSV columns and plot output contain all four score series.

### Tests for User Story 4

- [ ] T057 [P] [US4] Create score CSV writer tests in `tests/unit/test_score_csv.py`
- [ ] T058 [P] [US4] Create score plot generation tests in `tests/unit/test_plot_scores.py`
- [ ] T059 [US4] Create inference export integration test in `tests/integration/test_score_exports.py`
- [ ] T060 [US4] Create CLI contract test for `cognitive-state plot-scores` in `tests/integration/test_plot_scores_cli.py`

### Implementation for User Story 4

- [ ] T061 [US4] Implement score CSV writer and schema validation in `src/cognitive_state/data/score_csv.py`
- [ ] T062 [US4] Integrate optional score CSV output into `src/cognitive_state/inference/pipeline.py`
- [ ] T063 [US4] Implement score-over-time plotting in `src/cognitive_state/inference/plotting.py`
- [ ] T064 [US4] Implement `--plot` option for `cognitive-state infer` in `src/cognitive_state/cli/main.py`
- [ ] T065 [US4] Implement `cognitive-state plot-scores` CLI command in `src/cognitive_state/cli/main.py`

**Checkpoint**: US4 is complete when score CSV and plot artifacts are optional,
schema-valid, and never required for the core console inference workflow.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, final validation, and cleanup across the MVP.

- [ ] T066 [P] Write project overview, setup, and CLI examples in `README.md`
- [ ] T067 [P] Document CLI command details and output schemas in `docs/cli.md`
- [ ] T068 [P] Document synthetic/demo label limitations in `docs/research_notes.md`
- [ ] T069 [P] Document test fixture usage in `tests/fixtures/README.md`
- [ ] T070 Run the full pytest suite for `tests/` and record the validation command in `README.md`
- [ ] T071 Review CLI help text and synthetic-label warnings in `src/cognitive_state/cli/main.py`
- [ ] T072 Review dependency list and console script entry point in `pyproject.toml`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **US1 Estimate Scores From Video (Phase 3)**: Depends on Foundational.
- **US2 Build Time-Series Feature Windows (Phase 4)**: Depends on Foundational and can reuse US1 feature primitives; can be developed after US1 feature extractor tests are stable.
- **US3 Smoke-Test Baseline Regression Flow (Phase 5)**: Depends on Foundational and feature window contracts; can start after US2 window validation exists.
- **US4 Export and Review Score Trends (Phase 6)**: Depends on US1 inference output schema; can proceed after US1 console predictions are stable.
- **Polish (Phase 7)**: Depends on selected user stories for the release slice.

### User Story Dependencies

- **US1 (P1)**: First MVP slice; no user-story dependency after foundational work.
- **US2 (P2)**: Depends on shared feature and window entities; independently testable through `extract-features`.
- **US3 (P3)**: Depends on feature windows and model contracts; independently testable through `smoke-train`.
- **US4 (P4)**: Depends on score output schema; independently testable through CSV and plot commands.

### Within Each User Story

- Write tests before implementation tasks in the same phase.
- Implement lower-level data and service modules before CLI integration.
- Complete CLI contract tests before considering the story done.
- Keep console output required and CSV/plots optional.

---

## Parallel Execution Examples

### User Story 1

```text
Parallel tests:
- T013 tests/unit/test_video_sources.py
- T015 tests/unit/test_landmark_extractor.py
- T016 tests/unit/test_feature_extractors.py
- T017 tests/unit/test_temporal_transformer.py

Parallel implementation after tests exist:
- T025 src/cognitive_state/features/eyes.py
- T026 src/cognitive_state/features/face.py
- T027 src/cognitive_state/features/posture.py
- T030 src/cognitive_state/models/fusion.py
```

### User Story 2

```text
Parallel tests:
- T036 tests/unit/test_feature_csv.py
- T037 tests/unit/test_windowing.py
- T038 tests/unit/test_missing_summary.py
```

### User Story 3

```text
Parallel tests:
- T045 tests/unit/test_dataset.py
- T046 tests/unit/test_synthetic_labels.py
- T048 tests/unit/test_metrics.py

Parallel implementation after dataset contracts exist:
- T051 src/cognitive_state/data/synthetic_labels.py
- T054 src/cognitive_state/training/metrics.py
- T055 src/cognitive_state/training/checkpoints.py
```

### User Story 4

```text
Parallel tests:
- T057 tests/unit/test_score_csv.py
- T058 tests/unit/test_plot_scores.py

Parallel implementation:
- T061 src/cognitive_state/data/score_csv.py
- T063 src/cognitive_state/inference/plotting.py
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 / US1 to prove the video-to-score console workflow.
3. Stop and validate both video-file and webcam inference with bounded scores.

### Incremental Delivery

1. Add Phase 4 / US2 for inspectable feature windows and optional feature CSV.
2. Add Phase 5 / US3 for smoke-training and metrics.
3. Add Phase 6 / US4 for optional score CSV and plots.
4. Add Phase 7 documentation and final validation.

### Requested Work Area Coverage

- Project scaffold: T001, T003, T005
- Dependencies: T002, T004, T072
- Video input: T013, T014, T021, T022, T023
- MediaPipe extraction: T015, T024
- Feature extraction: T016, T025, T026, T027, T028
- CSV export: T036, T040, T057, T061, T062
- Dataset windowing: T029, T037, T042, T050
- Transformer model: T017, T030, T031
- Training loop: T047, T052, T053, T056
- Evaluation metrics: T048, T054
- Inference: T018, T019, T032, T033, T034, T035
- CLI: T020, T039, T049, T056, T060, T064, T065
- Tests: T006, T008, T013-T020, T036-T039, T045-T049, T057-T060, T070
- README: T066, T070
