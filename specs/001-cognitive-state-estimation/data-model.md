# Data Model: Real-time Cognitive State Estimation from Video

## Overview

The MVP data model follows a single pipeline:

`VideoSource -> FrameObservation -> LandmarkSet -> ModalityFeatureGroup -> FeatureWindow -> DatasetSample -> CognitiveScoreVector -> RunArtifact`

All labeled training/evaluation samples are window-level. Synthetic labels are
allowed only for smoke-training and data-flow validation.

## Entities

### VideoSource

Represents an input source for frame capture.

**Fields**:
- `source_id`: Stable identifier for this run input.
- `source_type`: `webcam` or `video_file`.
- `uri`: Camera index or video file path.
- `mode`: `live_paced` for webcam, `offline_batch` for video files.
- `fps`: Reported or measured frame rate.
- `width`: Frame width in pixels, when known.
- `height`: Frame height in pixels, when known.
- `duration_seconds`: Known for video files, optional for webcam.
- `opened`: Whether the source opened successfully.
- `warnings`: List of source-level warnings.

**Validation rules**:
- `source_type` MUST be one of `webcam` or `video_file`.
- Webcam sources MUST use `live_paced` mode.
- Video-file sources MUST use `offline_batch` mode by default.
- Source open failures MUST produce a user-facing error.

### FrameObservation

Represents one decoded RGB frame and its timing metadata.

**Fields**:
- `source_id`: Reference to `VideoSource`.
- `frame_index`: Zero-based frame index.
- `timestamp_seconds`: Timestamp relative to source start.
- `image_shape`: `(height, width, channels)`.
- `read_ok`: Whether the frame was decoded.
- `quality_flags`: Lighting, occlusion, visibility, or decode warnings.

**Validation rules**:
- `timestamp_seconds` MUST be monotonic per source.
- `image_shape` MUST describe normal RGB-compatible frames.
- Failed reads MUST stop or skip according to source mode and error policy.

### LandmarkSet

Represents available MediaPipe face and pose landmark observations for a frame.

**Fields**:
- `frame_index`: Reference to `FrameObservation`.
- `timestamp_seconds`: Copied from `FrameObservation`.
- `face_landmarks`: Face landmark coordinates when available.
- `pose_landmarks`: Pose landmark coordinates when available.
- `face_available`: Boolean.
- `pose_available`: Boolean.
- `face_confidence`: Confidence/quality proxy when available.
- `pose_confidence`: Confidence/quality proxy when available.
- `missing_reason`: Optional reason for absent landmarks.

**Validation rules**:
- Missing landmarks MUST be represented explicitly rather than as silent zeros.
- Feature extraction MUST tolerate face-only and pose-only frames.

### ModalityFeatureGroup

Represents computed numerical features for one modality group.

**Fields**:
- `frame_index`: Reference to `FrameObservation`.
- `timestamp_seconds`: Frame timestamp.
- `modality`: `eyes`, `face`, `head_pose`, or `posture`.
- `values`: Named numeric feature values.
- `missing_mask`: Per-feature missing/available markers.
- `units`: Optional units or coordinate-frame notes.
- `quality_flags`: Feature-level warnings.

**Feature groups**:
- `eyes`: EAR, blink variability proxy, gaze direction entropy, available eye
  movement proxies.
- `face`: landmark distances, micro-expression proxies, muscle tension proxies,
  mouth openness duration.
- `head_pose`: pose variance and micro-jitter proxies.
- `posture`: shoulder drop and spine curvature approximation when available.

**Validation rules**:
- All emitted feature groups MUST preserve modality name.
- Missing values MUST carry missing markers.
- Pupil-specific proxies MUST be flagged as unavailable when landmarks cannot
  support them reliably.

### FeatureWindow

Represents an ordered temporal sample used by inference and smoke-training.

**Fields**:
- `window_id`: Unique identifier.
- `source_id`: Reference to `VideoSource`.
- `start_time_seconds`: Window start time.
- `end_time_seconds`: Window end time.
- `frame_indices`: Ordered frame indices included in the window.
- `modalities`: Grouped feature arrays for eyes, face, head pose, and posture.
- `sampling_metadata`: Window length, stride, observed frame count, and timing
  gaps.
- `missing_summary`: Per-modality missing-value summary.

**Validation rules**:
- Windows MUST preserve frame order.
- Windows MUST preserve modality grouping.
- Windows shorter than the minimum sample length MUST be rejected or flagged.
- The number of feature timesteps MUST match model input expectations.

### DatasetSample

Represents a model-ready sample for smoke-training or real-label training.

**Fields**:
- `sample_id`: Unique sample identifier.
- `window_id`: Reference to `FeatureWindow`.
- `features`: Model-ready grouped feature tensors or arrays.
- `label`: Optional `CognitiveScoreVector`.
- `label_source`: `synthetic`, `placeholder`, `real`, or `missing`.
- `split`: Optional `train`, `validation`, or `test`.

**Validation rules**:
- Each labeled sample MUST pair exactly one feature window with one label vector.
- Labels MUST contain all four scores.
- Synthetic and placeholder labels MUST be marked and excluded from scientific
  accuracy claims.

### CognitiveScoreVector

Represents the four normalized cognitive state values.

**Fields**:
- `fatigue`: Float from 0.00 to 1.00.
- `attention`: Float from 0.00 to 1.00.
- `stress`: Float from 0.00 to 1.00.
- `engagement`: Float from 0.00 to 1.00.

**Validation rules**:
- All four fields are required.
- Values MUST be finite and within the inclusive range 0.00 to 1.00.
- Higher values mean more of the named state.

### ModelPrediction

Represents one model output for one feature window.

**Fields**:
- `window_id`: Reference to `FeatureWindow`.
- `timestamp_seconds`: Window end or center timestamp.
- `scores`: `CognitiveScoreVector`.
- `source_progress`: Frame/time progress string or numeric progress fields.
- `model_id`: Model or checkpoint identifier.

**Validation rules**:
- Each prediction MUST contain all four scores.
- Prediction values MUST be bounded to 0.00 through 1.00 before presentation.

### RunArtifact

Represents optional and required outputs from a run.

**Fields**:
- `run_id`: Unique run identifier.
- `source_id`: Reference to `VideoSource`.
- `console_output`: Required score stream/table summary.
- `feature_csv_path`: Optional path for exported features.
- `score_csv_path`: Optional path for score history.
- `plot_path`: Optional path for score-over-time plot.
- `config_snapshot`: Runtime parameters and artifact references.
- `metrics`: Optional smoke or real-label evaluation metrics.
- `warnings`: Run-level warnings.

**Validation rules**:
- Console score output is required for MVP inference runs.
- CSV and plot artifacts are optional and created only when enabled.
- Metrics from synthetic labels MUST be labeled as smoke metrics.

## Relationships

- One `VideoSource` has many `FrameObservation` records.
- One `FrameObservation` has zero or one `LandmarkSet`.
- One `LandmarkSet` produces zero or more `ModalityFeatureGroup` records.
- Many `ModalityFeatureGroup` records form one `FeatureWindow`.
- One `FeatureWindow` can become one `DatasetSample`.
- One `DatasetSample` may have one `CognitiveScoreVector` label.
- One `FeatureWindow` produces one `ModelPrediction` during inference.
- One run produces one `RunArtifact` and many `ModelPrediction` rows.

## State Transitions

### VideoSource

`configured -> opened -> reading -> exhausted`

Failure states:
- `open_failed`
- `read_failed`
- `permission_denied`

### FeatureWindow

`collecting_frames -> ready -> model_ready -> scored`

Failure states:
- `too_short`
- `insufficient_landmarks`
- `invalid_shape`

### DatasetSample

`window_created -> label_attached -> validated -> used_for_smoke_training`

Failure states:
- `missing_label`
- `invalid_label_range`
- `label_window_mismatch`

## Metric Records

Smoke-training metrics:
- Shape validation pass/fail.
- Score range validation pass/fail.
- MAE and RMSE reported as smoke metrics only.

Real-label metrics, when real labels are available:
- Per-score MAE.
- Per-score RMSE.
- Per-score R2.
- Per-score Pearson correlation.
