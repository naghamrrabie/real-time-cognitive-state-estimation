"""Per-window missing-landmark summary utilities."""

from __future__ import annotations

from cognitive_state.data.windowing import WindowMissingSummary


def summarize_window_missing(
    rows: list[dict[str, float | int]],
    window_index: int,
) -> WindowMissingSummary:
    """Compute a missing-landmark summary for one window's feature rows.

    Args:
        rows: Per-frame feature dicts for the window.  Each dict may contain
            ``face_detected`` and ``pose_detected`` keys (0.0 = absent,
            1.0 = present).  Missing keys default to present (1.0).
        window_index: Zero-based position of this window in the batch.

    Returns:
        ``WindowMissingSummary`` with frame counts for missing face and pose
        landmarks.
    """
    total = len(rows)
    face_missing = sum(
        1 for r in rows if float(r.get("face_detected", 1.0)) < 0.5
    )
    pose_missing = sum(
        1 for r in rows if float(r.get("pose_detected", 1.0)) < 0.5
    )
    return WindowMissingSummary(
        window_index=window_index,
        total_frames=total,
        face_missing_frames=face_missing,
        pose_missing_frames=pose_missing,
    )


def summarize_batch_missing(
    rows: list[dict[str, float | int]],
    *,
    window_size: int,
    stride: int,
) -> list[WindowMissingSummary]:
    """Compute per-window missing-landmark summaries for a full batch of rows.

    Applies the same sliding-window logic as ``build_windows`` so the resulting
    list aligns with the ``metadata`` list of a ``FeatureWindowBatch`` built
    from the same *rows*, *window_size*, and *stride*.

    Args:
        rows: Ordered per-frame feature dicts.
        window_size: Number of frames per window.  Must be >= 1.
        stride: Frame advance between windows.  Must be >= 1.

    Returns:
        One ``WindowMissingSummary`` per window, in window order.  Returns an
        empty list when ``len(rows) < window_size``.
    """
    summaries: list[WindowMissingSummary] = []
    n_rows = len(rows)
    start = 0
    window_idx = 0
    while start + window_size <= n_rows:
        end = start + window_size
        summaries.append(summarize_window_missing(rows[start:end], window_idx))
        start += stride
        window_idx += 1
    return summaries
