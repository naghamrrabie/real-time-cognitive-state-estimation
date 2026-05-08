# Feature Specification: Real-time Cognitive State Estimation from Video

**Feature Branch**: `001-cognitive-state-estimation`
**Created**: 2026-05-08
**Status**: Draft
**Input**: User description: "Build a Python research MVP that estimates continuous fatigue, attention, stress, and engagement scores from webcam or video input using multimodal video features, temporal modeling, attention-based fusion, and a simple CLI-first workflow."

## Clarifications

### Session 2026-05-08

- Q: What should the MVP include for model training? -> A: Pipeline plus smoke-training using synthetic or placeholder labels only to validate data flow.
- Q: What label granularity should datasets use? -> A: Window-level labels, with one four-score label vector per feature window.
- Q: Which input sources are required for the MVP? -> A: Both webcam streams and video files are required in the MVP.
- Q: How should recorded video processing be paced? -> A: Webcam processing is live-paced; video files process offline as fast as practical.
- Q: What output format is required for the MVP? -> A: Console table/stream is required; CSV logging and plots are optional.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Estimate Scores From Video (Priority: P1)

As a researcher or developer, I can run the MVP on a webcam stream or normal RGB
video file and see continuous cognitive state scores update over time.

**Why this priority**: This is the smallest end-to-end value path and proves the
project is not just a collection of isolated feature extractors.

**Independent Test**: Run the system on a short sample video with one visible
participant and confirm that every completed time window emits Fatigue,
Attention, Stress, and Engagement scores between 0.00 and 1.00.

**Acceptance Scenarios**:

1. **Given** a readable video file with a visible face and upper body, **When**
   the user starts MVP inference, **Then** the system displays the four
   continuous scores for each completed time window.
2. **Given** an available webcam and one visible participant, **When** the user
   starts live inference, **Then** the system displays updated scores without
   requiring a heavy graphical interface.
3. **Given** a window where some landmarks are missing, **When** inference
   reaches that window, **Then** the system applies the documented missing-data
   policy and keeps scores bounded between 0.00 and 1.00.

---

### User Story 2 - Build Time-Series Feature Windows (Priority: P2)

As a researcher, I can transform raw video observations into timestamped
multimodal feature windows so model experiments can be repeated and inspected.

**Why this priority**: Continuous cognitive state estimation depends on temporal
patterns, not isolated frame-level decisions.

**Independent Test**: Run feature extraction on a sample video and confirm that
the exported or in-memory windows contain eye, face, head, and posture feature
groups with timestamps and missing-value indicators.

**Acceptance Scenarios**:

1. **Given** a sample video with visible facial and pose landmarks, **When**
   feature extraction completes, **Then** each time window contains grouped
   feature values for eyes, face, head pose, and posture where observable.
2. **Given** low-confidence or unavailable landmarks for a frame, **When** the
   frame is processed, **Then** the feature record marks unavailable values
   without fabricating unsupported measurements.
3. **Given** videos with different frame rates, **When** windows are generated,
   **Then** each window records timing information so later modeling can align
   features consistently.

---

### User Story 3 - Smoke-Test Baseline Regression Flow (Priority: P3)

As a researcher, I can run a minimal baseline temporal regression smoke test
using the feature windows and placeholder or real labels so the modeling path is
ready for future datasets.

**Why this priority**: The project goal is continuous regression across four
cognitive scores, so the MVP must validate the model input and output contract
even before real labels are available.

**Independent Test**: Use a small placeholder dataset and confirm that the
smoke-training and evaluation flow accepts feature windows, produces one
prediction vector per sample, and returns four continuous scores per prediction.

**Acceptance Scenarios**:

1. **Given** feature windows and synthetic labels, **When** the user runs the
   baseline smoke-training flow, **Then** the system completes the
   data-to-prediction flow and reports prediction outputs for all four scores.
2. **Given** a future dataset with real labels in the documented format, **When**
   it is loaded through the dataset abstraction, **Then** the same experiment
   flow can use real labels without changing the user-facing workflow.
3. **Given** grouped modality features, **When** the model prepares a prediction,
   **Then** the fusion step preserves modality structure instead of treating all
   inputs as an undifferentiated flat feature list.

---

### User Story 4 - Export and Review Score Trends (Priority: P4)

As a researcher, I can optionally record score histories and generate simple
trend artifacts to inspect how cognitive scores change over time.

**Why this priority**: Exported traces help validate behavior, but they are not
required for the first end-to-end scoring path.

**Independent Test**: Enable logging for a sample run and confirm that the
resulting artifact contains timestamps and the four cognitive scores for each
completed window.

**Acceptance Scenarios**:

1. **Given** CSV logging is enabled, **When** a video or webcam session ends,
   **Then** the system writes a timestamped score history with all four score
   columns.
2. **Given** plotting is enabled and score history exists, **When** the run
   completes, **Then** the user can inspect a simple visualization of score
   changes over time.

---

### Edge Cases

- The webcam is unavailable, permission is denied, or the video file cannot be
  read.
- The participant leaves the frame, turns away, is partially occluded, or
  appears under poor lighting.
- Only face landmarks or only pose landmarks are available for part of a window.
- Multiple people appear in the frame; the MVP tracks only the primary visible
  participant and reports that limitation.
- The video is shorter than the minimum time window required for temporal
  scoring.
- Frame rates vary or frames arrive irregularly during webcam capture.
- Pupil-specific measurements cannot be inferred reliably from available RGB
  landmarks.
- Placeholder labels are used for pipeline validation and must not be presented
  as scientifically valid ground truth.
- Optional output files already exist or the destination directory is not
  writable.

## Requirements *(mandatory)*

### Research Scope and Targets *(mandatory for this project)*

- **Affected Pipeline Area**: video input, MediaPipe landmarks, features,
  sequences, temporal model, inference, evaluation, and optional export.
- **Affected Modalities**: eyes, face, head pose, and posture.
- **Required Scores**: Fatigue Score, Attention Score, Stress Score,
  Engagement Score.
- **Score Type**: Continuous regression outputs normalized to 0.00 through 1.00,
  where higher values mean more of the named state.
- **Real-Time Scope**: MVP supports live-paced webcam scoring and offline
  recorded-video scoring that processes as fast as practical through a simple
  command-line workflow.
- **Primary Model Constraint**: PyTorch Temporal Transformer Encoder remains
  primary unless amended by the project constitution.

### Functional Requirements

- **FR-001**: System MUST accept both a webcam stream and a normal RGB video
  file as input sources for MVP inference.
- **FR-001a**: Webcam input MUST be processed in live-paced mode, while video
  file input MUST be processed offline as fast as practical without simulating
  real-time playback unless explicitly requested.
- **FR-002**: System MUST process a single primary visible participant per run
  and report when that assumption is violated or uncertain.
- **FR-003**: System MUST extract per-frame face and pose landmark observations
  for the supported feature groups.
- **FR-004**: System MUST compute eye features including Eye Aspect Ratio, blink
  rate variability, gaze direction entropy, and available landmark-based eye
  movement proxies.
- **FR-005**: System MUST compute face features including facial landmark
  geometry, micro-expression proxy features, muscle tension proxy features, and
  mouth openness duration.
- **FR-006**: System MUST compute head and posture features including head pose
  variance, head micro-jitter, shoulder drop, and spine curvature approximation
  when pose landmarks are available.
- **FR-007**: System MUST store or pass feature observations as timestamped
  time-series windows with modality grouping, timing metadata, and
  missing-value indicators.
- **FR-008**: System MUST provide a dataset abstraction that supports synthetic
  labels for MVP pipeline validation and real labels for later research.
- **FR-008a**: System MUST include a minimal smoke-training and evaluation flow
  using synthetic or placeholder labels only to validate data flow, model input
  shape, model output shape, and score range handling.
- **FR-008b**: Dataset labels MUST be window-level: each feature window has one
  four-score label vector covering Fatigue, Attention, Stress, and Engagement.
- **FR-009**: System MUST predict Fatigue, Attention, Stress, and Engagement as
  four continuous scores for each completed time window.
- **FR-010**: System MUST keep every emitted score within the inclusive range
  0.00 to 1.00.
- **FR-011**: System MUST preserve modality structure during fusion so eye,
  face, head pose, and posture groups can be weighted or inspected separately.
- **FR-012**: System MUST provide a simple command-line workflow for MVP
  inference and baseline experiment execution.
- **FR-013**: System MUST support optional CSV logging of score history when the
  user enables it.
- **FR-013a**: System MUST display MVP score output as a console table or
  streaming console updates with timestamp, source progress, and all four score
  values.
- **FR-014**: System MUST treat CSV logging and plots as optional; when
  visualization is enabled, it MUST generate score-over-time plots from
  existing score history, and when visualization is disabled, the MVP workflow
  MUST remain complete.
- **FR-015**: System MUST include automated checks for feature extraction
  utilities and model input/output shapes.
- **FR-016**: System MUST document that synthetic or placeholder labels and
  smoke-training results are for pipeline validation only and are not evidence
  of model accuracy.
- **FR-017**: System MUST exclude self-supervised pretraining, domain
  adaptation, attention visualization, and Grad-CAM from MVP implementation
  scope while keeping them eligible as future research extensions.

### Key Entities *(include if feature involves data)*

- **Video Source**: A required MVP input source, either webcam stream or video
  file, including source type,
  capture timing, resolution metadata, and availability status.
- **Frame Observation**: A processed frame with timestamp, frame index, landmark
  confidence, and any warnings about visibility or quality.
- **Landmark Set**: Face and pose landmark coordinates plus confidence or
  availability metadata for downstream feature extraction.
- **Modality Feature Group**: A named group of numerical features for eyes,
  face, head pose, or posture, including units and missing-value markers.
- **Feature Window**: An ordered time-series segment containing grouped features,
  timing metadata, and the policy used for missing observations.
- **Dataset Sample**: A feature window paired with one synthetic, placeholder,
  or real four-score label vector for that same window.
- **Cognitive Score Vector**: The four normalized outputs: Fatigue, Attention,
  Stress, and Engagement.
- **Run Artifact**: Optional logs, score history, plots, configuration
  references, and run metadata used for reproducibility.
- **Console Score Output**: The required MVP output stream or table containing
  timestamp, source progress, Fatigue, Attention, Stress, and Engagement for
  each completed scoring window.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new project user can run a sample video through the offline MVP
  workflow and see all four score names with numeric values in under 5 minutes
  after setup is complete.
- **SC-002**: For a 2-minute sample video processed offline with one visible
  participant, at least 95% of completed windows emit four bounded numeric
  scores without a crash.
- **SC-002a**: During a webcam session, score updates are emitted as live-paced
  windows complete rather than after the session ends.
- **SC-003**: With simulated missing landmarks in 20% of frames, the system
  records missing-data status and still emits bounded scores for at least 90% of
  completed windows.
- **SC-004**: Feature-window output includes timestamped eye, face, head pose,
  and posture groups for 100% of windows where the corresponding landmarks are
  available.
- **SC-005**: A placeholder dataset smoke-training run produces one four-value
  prediction for every provided sample and rejects samples with invalid score
  ranges.
- **SC-005a**: Dataset validation rejects samples where the number of
  window-level label vectors does not match the number of feature windows.
- **SC-006**: When CSV logging is enabled, the output contains timestamps plus
  Fatigue, Attention, Stress, and Engagement columns for every completed scoring
  window.
- **SC-006a**: Every MVP inference run displays console output with timestamp,
  source progress, and all four score names for each completed scoring window.
- **SC-007**: MVP review confirms no heavy GUI or deep research extension is
  required to complete the core video-to-score workflow.

## Assumptions

- The MVP focuses on one primary visible participant at a time.
- Inputs are normal RGB camera frames; depth, infrared, EEG, audio, and wearable
  sensors are out of scope.
- Webcam streams and video files are both required MVP input paths.
- Recorded video files are processed offline as fast as practical; webcam
  streams are processed in live-paced mode.
- Score ranges are normalized to 0.00 through 1.00, and higher values indicate a
  stronger presence of the named cognitive state.
- Real labeled cognitive-state datasets may not be available during MVP work, so
  placeholder labels and smoke-training are acceptable only for validating data
  flow and shapes.
- Each labeled dataset sample represents one feature window and one corresponding
  four-score label vector.
- Landmark-derived pupil and gaze proxies are best-effort approximations because
  a normal RGB camera may not provide reliable pupil detail.
- The first version prioritizes a clear command-line workflow over a graphical
  application.
- Optional CSV logging and plotting are secondary to the required end-to-end
  scoring pipeline.
- Console score output is required for MVP runs; CSV logging and plots are
  optional user-enabled outputs.
- Future research extensions include self-supervised pretraining, domain
  adaptation, attention visualization, and Grad-CAM only if compatible visual
  feature models are introduced later.
