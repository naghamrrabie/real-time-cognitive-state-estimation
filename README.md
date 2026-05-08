# Real-time Cognitive State Estimation from Video

Python research MVP scaffold for estimating continuous cognitive state scores
from webcam or video input.

The planned system will eventually extract multimodal landmark-derived features
and output four normalized scores:

- Fatigue
- Attention
- Stress
- Engagement

Current implementation status: project scaffold only. MediaPipe extraction,
feature extraction, model training, and inference are intentionally not
implemented yet.

## Setup

```powershell
C:\Users\TheExpert\.local\bin\uv.exe pip install -r requirements.txt
C:\Users\TheExpert\.local\bin\uv.exe pip install -r requirements-dev.txt
C:\Users\TheExpert\.local\bin\uv.exe pip install -e .
```

## Validate Scaffold

```powershell
C:\Users\TheExpert\.local\bin\uv.exe run pytest
```

## CLI Placeholder

```powershell
C:\Users\TheExpert\.local\bin\uv.exe run python -m cognitive_state.cli.main --help
C:\Users\TheExpert\.local\bin\uv.exe run cognitive-state --help
```

The CLI currently exposes only scaffold metadata. Future Spec Kit tasks will add
feature extraction, smoke-training, inference, CSV export, and plotting
commands.
