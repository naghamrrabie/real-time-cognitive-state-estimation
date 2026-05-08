# Test Fixtures

`tests/fixtures/` provides shared synthetic helpers used across unit and
integration tests.  All helpers are deterministic and do not depend on
real video files, webcams, or external models.

---

## `synthetic_data.py`

| Helper | Returns | Used by |
|--------|---------|---------|
| `make_rgb_frame(height, width, channels, value)` | Solid-colour `uint8` NumPy array | video source, landmark extractor tests |
| `make_landmark_set(frame_index, timestamp_seconds, face_detected, pose_detected)` | `LandmarkSet` with `None` coordinates | feature extractor tests |
| `make_feature_window(n_frames, feature_dim, seed)` | `(n_frames, feature_dim)` float32 array | windowing, dataset, training tests |
| `make_score_vector(fatigue, attention, stress, engagement)` | `CognitiveScoreVector` | schema, dataset, batching tests |
| `make_label_list(n, seed)` | `list[CognitiveScoreVector]` | dataset, batching, smoke-train tests |

---

## Usage

Import directly from the fixture module:

```python
from tests.fixtures.synthetic_data import make_rgb_frame, make_label_list
```

Or use the `tests` directory on `pythonpath` (configured in `pyproject.toml`)
and import without the `tests.` prefix:

```python
from fixtures.synthetic_data import make_rgb_frame
```

---

## Design rules

- **No file I/O** — helpers return in-memory objects only.
- **Deterministic** — all random helpers accept a `seed` parameter.
- **Minimal size** — default dimensions are kept small (e.g. 4×4 frames,
  30-frame windows) to keep tests fast.
- **No MediaPipe** — landmark helpers return `None` coordinates so tests
  do not require a working MediaPipe installation.
