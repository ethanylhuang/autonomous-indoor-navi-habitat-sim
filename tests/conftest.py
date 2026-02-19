"""Shared pytest fixtures for habitat-sim-2 tests."""

import numpy as np
import pytest


@pytest.fixture
def identity_quat():
    """Identity quaternion [w, x, y, z] = [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture
def zero_position():
    """Origin position [0, 0, 0]."""
    return np.zeros(3, dtype=np.float64)
