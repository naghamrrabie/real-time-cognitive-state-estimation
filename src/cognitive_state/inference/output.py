"""Console score table and stream formatter."""

from __future__ import annotations

from cognitive_state.data.constants import SCORE_CSV_COLUMNS, SCORE_NAMES
from cognitive_state.data.schemas import ModelPrediction


def format_score_row(prediction: ModelPrediction) -> dict[str, object]:
    """Return a dict whose keys match SCORE_CSV_COLUMNS for the given prediction."""
    return {
        "timestamp_seconds": prediction.timestamp_seconds,
        "source_progress": prediction.source_progress,
        **prediction.scores.as_dict(),
    }


def format_score_line(prediction: ModelPrediction) -> str:
    """Return a single console line summarising one prediction."""
    s = prediction.scores
    return (
        f"t={prediction.timestamp_seconds:.2f}s"
        f"  [{prediction.source_progress}]"
        f"  fatigue={s.fatigue:.3f}"
        f"  attention={s.attention:.3f}"
        f"  stress={s.stress:.3f}"
        f"  engagement={s.engagement:.3f}"
    )
