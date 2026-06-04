"""Tests for importance-sampling weights."""

from __future__ import annotations

import numpy as np
import pytest

from mlcast_dataset_sampler.sampling import importance_weights


def test_floor_at_zero_mean():
    # 1 - exp(0) == 0, so a zero-mean window gets exactly q_min.
    w = importance_weights(np.array([0.0]), q_min=1e-4, scale=1.0, mean_weight=0.1)
    assert w[0] == pytest.approx(1e-4)


def test_monotonic_in_mean():
    w = importance_weights(np.array([0.0, 0.5, 1.0, 5.0]))
    assert np.all(np.diff(w) > 0)


def test_saturates_below_floor_plus_weight():
    # As mean -> inf the transform -> 1, so weight -> q_min + mean_weight.
    w = importance_weights(np.array([1e9]), q_min=1e-4, scale=1.0, mean_weight=0.1)
    assert w[0] == pytest.approx(1e-4 + 0.1)


def test_nan_propagates():
    w = importance_weights(np.array([np.nan, 1.0]))
    assert np.isnan(w[0]) and not np.isnan(w[1])


def test_shape_and_dtype_preserved():
    w = importance_weights(np.zeros((3, 2), dtype=np.float32))
    assert w.shape == (3, 2)
    assert w.dtype == np.float64


@pytest.mark.parametrize("kwargs", [
    {"scale": 0.0},
    {"scale": -1.0},
    {"q_min": -1e-9},
    {"mean_weight": -0.1},
])
def test_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        importance_weights(np.array([1.0]), **kwargs)
