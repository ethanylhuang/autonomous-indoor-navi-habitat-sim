"""Tests for the simulated IMU.

Covers AC4 (IMU readings). Tests operate against both the raw SimulatedIMU
class (unit tests) and through the Vehicle facade (integration tests).
"""

import numpy as np
import pytest

from src.sensors.imu import SimulatedIMU
from src.vehicle import Vehicle


# ---------------------------------------------------------------------------
# Unit tests for SimulatedIMU
# ---------------------------------------------------------------------------

class TestSimulatedIMUUnit:
    def test_first_update_returns_zeros(self):
        imu = SimulatedIMU(dt=1.0)
        pos = np.array([0.0, 0.0, 0.0])
        rot = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        reading = imu.update(pos, rot)
        np.testing.assert_array_equal(reading.linear_acceleration, np.zeros(3))
        np.testing.assert_array_equal(reading.angular_velocity, np.zeros(3))
        np.testing.assert_array_equal(reading.linear_velocity, np.zeros(3))

    def test_constant_velocity_zero_acceleration(self):
        """Moving at constant velocity should yield zero acceleration after warm-up."""
        imu = SimulatedIMU(dt=1.0)
        rot = np.array([1.0, 0.0, 0.0, 0.0])
        # Step 1: origin
        imu.update(np.array([0.0, 0.0, 0.0]), rot)
        # Step 2: moved 1m in X -> velocity = [1, 0, 0], accel = [1, 0, 0]
        imu.update(np.array([1.0, 0.0, 0.0]), rot)
        # Step 3: moved another 1m in X -> velocity = [1, 0, 0], accel = [0, 0, 0]
        reading = imu.update(np.array([2.0, 0.0, 0.0]), rot)
        np.testing.assert_allclose(reading.linear_acceleration, np.zeros(3), atol=1e-10)

    def test_linear_velocity_tracks_motion(self):
        """linear_velocity should reflect position differencing."""
        imu = SimulatedIMU(dt=1.0)
        rot = np.array([1.0, 0.0, 0.0, 0.0])
        imu.update(np.array([0.0, 0.0, 0.0]), rot)
        reading = imu.update(np.array([0.25, 0.0, -0.5]), rot)
        np.testing.assert_allclose(reading.linear_velocity, [0.25, 0.0, -0.5], atol=1e-10)

    def test_acceleration_on_velocity_change(self):
        """Changing velocity produces non-zero acceleration."""
        imu = SimulatedIMU(dt=1.0)
        rot = np.array([1.0, 0.0, 0.0, 0.0])
        imu.update(np.array([0.0, 0.0, 0.0]), rot)
        imu.update(np.array([1.0, 0.0, 0.0]), rot)
        # Step 3: stop -> velocity = [0, 0, 0], accel = [-1, 0, 0]
        reading = imu.update(np.array([1.0, 0.0, 0.0]), rot)
        assert np.linalg.norm(reading.linear_acceleration) > 0.5

    def test_rotation_produces_angular_velocity(self):
        """Rotating the quaternion should produce non-zero angular velocity."""
        imu = SimulatedIMU(dt=1.0)
        pos = np.array([0.0, 0.0, 0.0])
        rot_identity = np.array([1.0, 0.0, 0.0, 0.0])
        # Small Y-axis rotation: 10 degrees
        angle = np.radians(10.0)
        rot_turned = np.array([
            np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0
        ])
        imu.update(pos, rot_identity)
        reading = imu.update(pos, rot_turned)
        # Y component should be non-zero
        assert abs(reading.angular_velocity[1]) > 0.01

    def test_reset_clears_state(self):
        imu = SimulatedIMU(dt=1.0)
        pos = np.array([1.0, 2.0, 3.0])
        rot = np.array([1.0, 0.0, 0.0, 0.0])
        imu.update(pos, rot)
        imu.update(pos + 1.0, rot)
        imu.reset()
        reading = imu.update(np.array([0.0, 0.0, 0.0]), rot)
        np.testing.assert_array_equal(reading.linear_acceleration, np.zeros(3))
        np.testing.assert_array_equal(reading.angular_velocity, np.zeros(3))
        np.testing.assert_array_equal(reading.linear_velocity, np.zeros(3))
        assert reading.timestamp_step == 1

    def test_dtype_and_shape(self):
        imu = SimulatedIMU(dt=1.0)
        reading = imu.update(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        assert reading.linear_acceleration.dtype == np.float64
        assert reading.angular_velocity.dtype == np.float64
        assert reading.linear_velocity.dtype == np.float64
        assert reading.linear_acceleration.shape == (3,)
        assert reading.angular_velocity.shape == (3,)
        assert reading.linear_velocity.shape == (3,)


# ---------------------------------------------------------------------------
# Integration tests via Vehicle
# ---------------------------------------------------------------------------

class TestIMUViaVehicle:
    @pytest.fixture(scope="class")
    def vehicle(self):
        v = Vehicle()
        yield v
        v.close()

    def test_initial_imu_is_zeros(self, vehicle):
        """First observations should have zero IMU readings."""
        obs = vehicle.reset()
        np.testing.assert_array_equal(obs.imu.linear_acceleration, np.zeros(3))
        np.testing.assert_array_equal(obs.imu.angular_velocity, np.zeros(3))

    def test_forward_steps_produce_nonzero_acceleration(self, vehicle):
        """After 3 forward steps, acceleration should be non-zero."""
        vehicle.reset()
        for _ in range(3):
            obs = vehicle.step("move_forward")
        assert np.linalg.norm(obs.imu.linear_acceleration) > 0 or obs.state.collided

    def test_turn_produces_angular_velocity_y(self, vehicle):
        """Turning left should produce angular velocity in Y."""
        vehicle.reset()
        obs = vehicle.step("turn_left")
        assert abs(obs.imu.angular_velocity[1]) > 0.001

    def test_reset_clears_imu(self, vehicle):
        """After reset, IMU should return zeros."""
        vehicle.step("move_forward")
        obs = vehicle.reset()
        np.testing.assert_array_equal(obs.imu.linear_acceleration, np.zeros(3))
        np.testing.assert_array_equal(obs.imu.angular_velocity, np.zeros(3))
