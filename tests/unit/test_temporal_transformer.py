"""Temporal Transformer model input/output shape contract tests (T017)."""

import torch
import torch.nn as nn

from cognitive_state.data.constants import SCORE_NAMES
from cognitive_state.models.transformer import NUM_SCORES, TemporalTransformer


def test_num_scores_matches_score_names_constant() -> None:
    assert NUM_SCORES == len(SCORE_NAMES)
    assert NUM_SCORES == 4


def test_model_is_a_torch_nn_module() -> None:
    model = TemporalTransformer(input_features=16)
    assert isinstance(model, nn.Module)


def test_forward_produces_correct_output_shape_for_batch() -> None:
    model = TemporalTransformer(input_features=16)
    x = torch.randn(4, 30, 16)
    output = model(x)
    assert output.shape == (4, NUM_SCORES)


def test_forward_single_sample_output_shape() -> None:
    model = TemporalTransformer(input_features=16)
    x = torch.randn(1, 30, 16)
    output = model(x)
    assert output.shape == (1, NUM_SCORES)


def test_forward_output_is_float_tensor() -> None:
    model = TemporalTransformer(input_features=16)
    x = torch.randn(2, 30, 16)
    output = model(x)
    assert output.dtype in (torch.float32, torch.float64)


def test_forward_output_has_no_nan_for_normal_input() -> None:
    model = TemporalTransformer(input_features=16)
    x = torch.randn(2, 30, 16)
    output = model(x)
    assert not torch.isnan(output).any()


def test_forward_output_has_no_inf_for_normal_input() -> None:
    model = TemporalTransformer(input_features=16)
    x = torch.randn(2, 30, 16)
    output = model(x)
    assert not torch.isinf(output).any()


def test_model_accepts_variable_timesteps() -> None:
    model = TemporalTransformer(input_features=8)
    for timesteps in (10, 30, 60):
        x = torch.randn(2, timesteps, 8)
        out = model(x)
        assert out.shape == (2, NUM_SCORES), f"Failed for timesteps={timesteps}"


def test_model_accepts_variable_batch_sizes() -> None:
    model = TemporalTransformer(input_features=8)
    for batch in (1, 2, 8):
        x = torch.randn(batch, 30, 8)
        out = model(x)
        assert out.shape[0] == batch, f"Batch dim mismatch for batch={batch}"


def test_model_output_second_dim_is_num_scores_for_all_batches() -> None:
    model = TemporalTransformer(input_features=8)
    for batch in (1, 4):
        x = torch.randn(batch, 20, 8)
        out = model(x)
        assert out.shape[1] == NUM_SCORES
