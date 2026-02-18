"""Tests for sensor observation shapes, dtypes, and action validation.

Covers AC1 (simulator init), AC2 (sensor shapes/types), AC3 (actions).
"""

import numpy as np
import pytest

from src.vehicle import Vehicle


@pytest.fixture(scope="module")
def vehicle():
    """Create a single Vehicle instance for all sensor tests."""
    v = Vehicle()
    yield v
    v.close()


# -- AC1: Simulator initializes successfully --------------------------------

class TestSimulatorInit:
    def test_vehicle_creates_successfully(self, vehicle):
        """Vehicle() creates without exception and pathfinder is loaded."""
        assert vehicle.pathfinder.is_loaded

    def test_initial_observations_returned(self, vehicle):
        """get_initial_observations() returns a valid Observations object."""
        obs = vehicle.get_initial_observations()
        assert obs is not None
        assert obs.forward_rgb is not None
        assert obs.rear_rgb is not None
        assert obs.depth is not None


# -- AC2: Sensor observations have correct shapes and types -----------------

class TestSensorShapes:
    def test_forward_rgb_shape_dtype(self, vehicle):
        obs = vehicle.get_initial_observations()
        assert obs.forward_rgb.shape == (480, 640, 4)
        assert obs.forward_rgb.dtype == np.uint8

    def test_rear_rgb_shape_dtype(self, vehicle):
        obs = vehicle.get_initial_observations()
        assert obs.rear_rgb.shape == (480, 640, 4)
        assert obs.rear_rgb.dtype == np.uint8

    def test_depth_shape_dtype(self, vehicle):
        obs = vehicle.get_initial_observations()
        assert obs.depth.shape == (480, 640)
        assert obs.depth.dtype == np.float32

    def test_forward_and_rear_are_different(self, vehicle):
        """Forward and rear cameras should show different views."""
        obs = vehicle.get_initial_observations()
        assert not np.array_equal(obs.forward_rgb, obs.rear_rgb)

    def test_depth_center_patch_valid(self, vehicle):
        """Center 100x100 patch of depth should be finite and >= 0."""
        obs = vehicle.get_initial_observations()
        center = obs.depth[190:290, 270:370]
        assert np.all(np.isfinite(center))
        assert np.all(center >= 0)


# -- AC3: Discrete actions produce expected state changes -------------------

class TestActions:
    def test_move_forward_changes_position(self, vehicle):
        obs_before = vehicle.get_initial_observations()
        obs_after = vehicle.step("move_forward")
        delta = np.linalg.norm(
            obs_after.state.position - obs_before.state.position
        )
        assert delta > 0.01

    def test_turn_left_changes_rotation(self, vehicle):
        obs_before = vehicle.get_initial_observations()
        obs_after = vehicle.step("turn_left")
        assert not np.array_equal(
            obs_after.state.rotation, obs_before.state.rotation
        )

    def test_turn_right_changes_rotation(self, vehicle):
        obs_before = vehicle.get_initial_observations()
        obs_after = vehicle.step("turn_right")
        assert not np.array_equal(
            obs_after.state.rotation, obs_before.state.rotation
        )

    def test_invalid_action_raises_valueerror(self, vehicle):
        with pytest.raises(ValueError, match="Invalid action"):
            vehicle.step("jump")

    def test_step_count_increments(self, vehicle):
        obs = vehicle.reset()
        assert obs.state.step_count == 0
        obs = vehicle.step("move_forward")
        assert obs.state.step_count == 1
