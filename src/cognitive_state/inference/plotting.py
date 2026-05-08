"""Score-over-time plotting for inference results (T063)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless / test environments
import matplotlib.pyplot as plt

from cognitive_state.data.schemas import ModelPrediction

_SCORE_COLORS = {
    "fatigue": "#e74c3c",
    "attention": "#2ecc71",
    "stress": "#e67e22",
    "engagement": "#3498db",
}


def plot_scores(
    predictions: list[ModelPrediction],
    output_path: str | Path,
    *,
    title: str = "Cognitive State Scores Over Time",
) -> Path:
    """Plot all four score series over time and save to *output_path*.

    Args:
        predictions: Ordered list of model predictions.  Each prediction
            contributes one point on the time axis.
        output_path: Destination image path (e.g. ``scores.png``).
            Parent directories are created automatically.
        title: Plot title.

    Returns:
        Resolved :class:`~pathlib.Path` where the image was saved.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    timestamps = [p.timestamp_seconds for p in predictions]

    fig, ax = plt.subplots(figsize=(10, 5))

    if predictions:
        for score_name, color in _SCORE_COLORS.items():
            values = [getattr(p.scores, score_name) for p in predictions]
            ax.plot(timestamps, values, label=score_name.capitalize(), color=color)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Score  [0 – 1]")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    if predictions:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return out
