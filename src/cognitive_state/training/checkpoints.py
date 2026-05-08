"""Checkpoint save/load helpers for smoke artifacts (T055)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cognitive_state.data.constants import SCORE_NAMES

#: Type alias for the checkpoint dict returned by :func:`load_checkpoint`.
CheckpointData = dict[str, Any]


def save_checkpoint(
    model: "torch.nn.Module",
    path: str | Path,
    *,
    feature_dim: int,
    label_source: str = "synthetic",
    window_size: int | None = None,
    model_config: dict[str, Any] | None = None,
) -> Path:
    """Save model state dict with metadata to *path*.

    Args:
        model: Trained PyTorch module.
        path: Output ``.pt`` file path.  Parent directories are created.
        feature_dim: Feature dimension the model expects per timestep.
        label_source: Label source tag recorded in metadata.
        window_size: Temporal window size used during training (optional).
        model_config: Optional constructor kwargs (e.g. ``d_model``, ``nhead``).

    Returns:
        Resolved :class:`~pathlib.Path` where the checkpoint was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: CheckpointData = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "feature_dim": feature_dim,
            "label_source": label_source,
            "score_names": list(SCORE_NAMES),
            "window_size": window_size,
            "model_config": model_config or {},
        },
    }
    torch.save(checkpoint, out)
    return out


def load_checkpoint(path: str | Path) -> CheckpointData:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Also accepts legacy checkpoints that contain only a bare ``state_dict``.

    Args:
        path: Path to a ``.pt`` checkpoint file.

    Returns:
        A dict with at minimum a ``"model_state_dict"`` key and optionally a
        ``"metadata"`` key.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be loaded or has an unexpected format.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    try:
        data = torch.load(p, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint at {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid checkpoint at {p}: expected dict, got {type(data).__name__}."
        )
    # Normalise legacy bare state_dict format
    if "model_state_dict" not in data:
        data = {"model_state_dict": data, "metadata": {}}
    return data
