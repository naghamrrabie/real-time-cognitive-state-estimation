"""Evaluation metric tests for MAE, RMSE, R2, and Pearson correlation (T048)."""

from __future__ import annotations

import pytest
import torch

from cognitive_state.training.metrics import (
    SMOKE_METRIC_WARNING,
    compute_mae,
    compute_pearson,
    compute_r2,
    compute_rmse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identical(n: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(n, 4)
    return t.clone(), t.clone()


def _zero_pred_unit_target(n: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(n, 4), torch.ones(n, 4)


# ---------------------------------------------------------------------------
# compute_mae
# ---------------------------------------------------------------------------

def test_mae_is_zero_for_identical_pred_and_target() -> None:
    pred, target = _identical()
    mae = compute_mae(pred, target)
    assert mae.max().item() == pytest.approx(0.0, abs=1e-6)


def test_mae_is_one_for_zero_pred_unit_target() -> None:
    pred, target = _zero_pred_unit_target()
    mae = compute_mae(pred, target)
    assert mae.min().item() == pytest.approx(1.0, abs=1e-6)


def test_mae_shape_is_four() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_mae(pred, target).shape == (4,)


def test_mae_is_non_negative() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_mae(pred, target).min().item() >= 0.0


def test_mae_is_symmetric() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_mae(pred, target).tolist() == pytest.approx(
        compute_mae(target, pred).tolist(), abs=1e-6
    )


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

def test_rmse_is_zero_for_identical_pred_and_target() -> None:
    pred, target = _identical()
    assert compute_rmse(pred, target).max().item() == pytest.approx(0.0, abs=1e-6)


def test_rmse_is_one_for_zero_pred_unit_target() -> None:
    pred, target = _zero_pred_unit_target()
    assert compute_rmse(pred, target).min().item() == pytest.approx(1.0, abs=1e-6)


def test_rmse_shape_is_four() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_rmse(pred, target).shape == (4,)


def test_rmse_is_non_negative() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_rmse(pred, target).min().item() >= 0.0


def test_rmse_is_at_least_mae_for_same_inputs() -> None:
    pred = torch.rand(16, 4)
    target = torch.rand(16, 4)
    mae = compute_mae(pred, target)
    rmse = compute_rmse(pred, target)
    # By Cauchy-Schwarz, RMSE >= MAE
    for i in range(4):
        assert rmse[i].item() >= mae[i].item() - 1e-6


# ---------------------------------------------------------------------------
# compute_r2
# ---------------------------------------------------------------------------

def test_r2_is_one_for_identical_pred_and_target() -> None:
    # Use non-constant random targets to avoid degenerate case
    torch.manual_seed(0)
    t = torch.rand(12, 4)
    pred, target = t.clone(), t.clone()
    r2 = compute_r2(pred, target)
    assert r2.shape == (4,)
    assert r2.min().item() == pytest.approx(1.0, abs=1e-5)


def test_r2_shape_is_four() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_r2(pred, target).shape == (4,)


def test_r2_is_at_most_one() -> None:
    pred, target = torch.rand(16, 4), torch.rand(16, 4)
    assert compute_r2(pred, target).max().item() <= 1.0 + 1e-6


def test_r2_returns_tensor() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert isinstance(compute_r2(pred, target), torch.Tensor)


# ---------------------------------------------------------------------------
# compute_pearson
# ---------------------------------------------------------------------------

def test_pearson_is_one_for_identical_pred_and_target() -> None:
    torch.manual_seed(1)
    t = torch.rand(12, 4)
    pred, target = t.clone(), t.clone()
    r = compute_pearson(pred, target)
    assert r.shape == (4,)
    assert r.min().item() == pytest.approx(1.0, abs=1e-5)


def test_pearson_shape_is_four() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert compute_pearson(pred, target).shape == (4,)


def test_pearson_is_between_minus_one_and_one() -> None:
    pred, target = torch.rand(16, 4), torch.rand(16, 4)
    r = compute_pearson(pred, target)
    assert r.min().item() >= -1.0 - 1e-6
    assert r.max().item() <= 1.0 + 1e-6


def test_pearson_is_minus_one_for_perfectly_anti_correlated() -> None:
    base = torch.linspace(0, 1, 10).unsqueeze(1).expand(-1, 4)
    pred = base
    target = 1.0 - base
    r = compute_pearson(pred, target)
    assert r.max().item() == pytest.approx(-1.0, abs=1e-5)


def test_pearson_returns_tensor() -> None:
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    assert isinstance(compute_pearson(pred, target), torch.Tensor)


# ---------------------------------------------------------------------------
# SMOKE_METRIC_WARNING constant
# ---------------------------------------------------------------------------

def test_smoke_metric_warning_is_non_empty_string() -> None:
    assert isinstance(SMOKE_METRIC_WARNING, str)
    assert len(SMOKE_METRIC_WARNING) > 0


def test_smoke_metric_warning_mentions_synthetic_or_placeholder() -> None:
    lower = SMOKE_METRIC_WARNING.lower()
    assert "synthetic" in lower or "placeholder" in lower
