# Quickstart: Real-time Cognitive State Estimation from Video

This quickstart validates the MVP workflow after implementation. It does not
require real cognitive-state labels.

## 1. Create Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy pandas torch scikit-learn matplotlib pytest
```

## 2. Run Tests

```powershell
pytest
```

Expected result:

- Feature utility tests pass.
- Missing-landmark behavior tests pass.
- Window construction tests pass.
- Dataset label alignment tests pass.
- Model input/output shape tests pass.
- CLI smoke tests pass.

## 3. Extract Features From a Video File

```powershell
cognitive-state extract-features --video path\to\sample.mp4 --features-csv artifacts\features.csv --max-windows 20
```

Expected result:

- The command reads the video file offline as fast as practical.
- The console summary reports processed windows and any warnings.
- `artifacts\features.csv` contains timestamped eye, face, head pose, and
  posture feature groups where landmarks are available.

## 4. Run Smoke-Training With Synthetic Labels

```powershell
cognitive-state smoke-train --features-csv artifacts\features.csv --synthetic-labels --epochs 1 --checkpoint-out artifacts\smoke_model.pt
```

Expected result:

- The command validates feature window shape.
- Synthetic labels are marked as smoke/demo labels.
- One four-score output is produced per sample.
- MAE/RMSE are reported only as smoke metrics.
- The result is not presented as scientific model accuracy.

## 5. Run Offline Inference on a Video File

```powershell
cognitive-state infer --video path\to\sample.mp4 --checkpoint artifacts\smoke_model.pt --scores-csv artifacts\scores.csv --plot artifacts\scores.png --max-windows 20
```

Expected result:

- The command processes the recorded video offline as fast as practical.
- Console output displays timestamp, source progress, Fatigue, Attention,
  Stress, and Engagement for each completed scoring window.
- Score values are finite and bounded from 0.00 through 1.00.
- Optional CSV and plot artifacts are created when requested.

## 6. Run Live-Paced Webcam Inference

```powershell
cognitive-state infer --webcam 0 --checkpoint artifacts\smoke_model.pt --max-windows 20
```

Expected result:

- The command opens the webcam in live-paced mode.
- Score updates appear as windows complete.
- The run remains console-first; no heavy GUI is required.

## 7. Validate Output CSV Schema

The score CSV must include:

```text
timestamp_seconds,source_progress,fatigue,attention,stress,engagement
```

Each score must be a finite numeric value from 0.00 through 1.00.

## 8. MVP Completion Signal

The MVP is ready for task generation when:

- Both video-file and webcam input paths are represented in the plan.
- Feature extraction covers eyes, face, head pose, and posture.
- Feature windows preserve modality grouping and timing metadata.
- Synthetic/demo labels support smoke-training only.
- Temporal Transformer Encoder model input/output shape is defined.
- Console score output is required.
- CSV feature/score export and plots are optional.
- Tests cover feature utilities, windowing, dataset labels, model shapes, score
  bounds, and CLI smoke behavior.
