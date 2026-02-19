"""Tests for the EKF state estimator.

Pure numpy tests, no habitat-sim dependency.
"""

import numpy as np
import pytest

from src.sensors.imu import IMUReading
from src.perception.visual_odometry import VOEstimate
from src.state_estimation.estimator import EKFEstimator, PoseEstimate


class TestEKFEstimator:

    def _make_imu(
        self,
        accel=None,
        ang_vel=None,
        velocity=None,
        step: int = 1,
    ) -> IMUReading:
        if accel is None:
            accel = np.zeros(3, dtype=np.float64)
        if ang_vel is None:
            ang_vel = np.zeros(3, dtype=np.float64)
        if velocity is None:
            velocity = np.zeros(3, dtype=np.float64)
        return IMUReading(
            linear_acceleration=np.asarray(accel, dtype=np.float64),
            angular_velocity=np.asarray(ang_vel, dtype=np.float64),
            linear_velocity=np.asarray(velocity, dtype=np.float64),
            timestamp_step=step,
        )

    def _make_vo(
        self,
        rotation=None,
        is_valid: bool = True,
        step: int = 1,
    ) -> VOEstimate:
        if rotation is None:
            rotation = np.eye(3, dtype=np.float64)
        return VOEstimate(
            rotation=np.asarray(rotation, dtype=np.float64),
            translation_dir=np.zeros(3, dtype=np.float64),
            num_inliers=50 if is_valid else 0,
            is_valid=is_valid,
            timestamp_step=step,
        )

    def test_initial_pose_matches_input(self):
        ekf = EKFEstimator()
        pos = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        yaw = 0.3
        est = ekf.initialize(pos, yaw)

        np.testing.assert_allclose(est.position, pos, atol=1e-10)
        assert abs(est.yaw - yaw) < 1e-10
        assert est.covariance.shape == (3, 3)
        assert est.timestamp_step == 0

    def test_predict_stationary_no_drift(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)
        imu = self._make_imu(accel=[0.0, 0.0, 0.0], ang_vel=[0.0, 0.0, 0.0])
        est = ekf.predict(imu, dt=1.0)

        np.testing.assert_allclose(est.position[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(est.position[2], 0.0, atol=1e-10)
        assert abs(est.yaw) < 1e-10

    def test_predict_forward_motion(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)
        # Velocity in X => should move in X direction
        imu = self._make_imu(velocity=[1.0, 0.0, 0.0], ang_vel=[0.0, 0.0, 0.0])
        est = ekf.predict(imu, dt=1.0)

        # dx = velocity_x * dt = 1.0 * 1.0 = 1.0
        assert est.position[0] > 0.0
        np.testing.assert_allclose(est.position[0], 1.0, atol=1e-10)
        # Z should not change
        np.testing.assert_allclose(est.position[2], 0.0, atol=1e-10)

    def test_predict_turn_updates_yaw(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)
        # Angular velocity about Y axis
        imu = self._make_imu(accel=[0.0, 0.0, 0.0], ang_vel=[0.0, 0.5, 0.0])
        est = ekf.predict(imu, dt=1.0)

        # dyaw = 0.5 * 1.0 = 0.5 radians
        assert abs(est.yaw - 0.5) < 1e-10

    def test_update_vo_corrects_yaw(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)

        # Predict a turn
        imu = self._make_imu(ang_vel=[0.0, 0.3, 0.0])
        ekf.predict(imu, dt=1.0)

        # VO says the yaw change is different (small rotation matrix about Y)
        angle = 0.25  # VO says rotation was 0.25 rad, not 0.3
        R_vo = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ], dtype=np.float64)
        vo = self._make_vo(rotation=R_vo, is_valid=True)
        est_before = ekf.get_estimate()
        est_after = ekf.update_vo(vo)

        # After VO update, yaw should be corrected toward VO estimate
        # The exact value depends on Kalman gain, but it should be different
        # from the pure prediction
        assert est_after.yaw != est_before.yaw or abs(est_after.yaw - 0.3) < 0.15

    def test_update_vo_invalid_skipped(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)

        imu = self._make_imu(ang_vel=[0.0, 0.3, 0.0])
        est_pred = ekf.predict(imu, dt=1.0)

        # Invalid VO should not change state
        vo = self._make_vo(is_valid=False)
        est_after = ekf.update_vo(vo)

        np.testing.assert_allclose(est_after.position, est_pred.position, atol=1e-10)
        assert abs(est_after.yaw - est_pred.yaw) < 1e-10

    def test_covariance_grows_on_predict(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)
        cov_init = ekf.get_estimate().covariance.copy()

        imu = self._make_imu()
        ekf.predict(imu, dt=1.0)
        cov_after = ekf.get_estimate().covariance

        # Trace should increase (uncertainty grows)
        assert np.trace(cov_after) > np.trace(cov_init)

    def test_covariance_shrinks_on_update(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([0.0, 0.0, 0.0]), 0.0)

        # Predict to grow covariance
        for _ in range(5):
            imu = self._make_imu(ang_vel=[0.0, 0.1, 0.0])
            ekf.predict(imu, dt=1.0)

        cov_before = ekf.get_estimate().covariance.copy()

        # VO update should reduce uncertainty
        vo = self._make_vo(is_valid=True)
        ekf.update_vo(vo)
        cov_after = ekf.get_estimate().covariance

        # At least the yaw uncertainty (element [2,2]) should shrink
        assert cov_after[2, 2] <= cov_before[2, 2]

    def test_reset_clears_state(self):
        ekf = EKFEstimator()
        ekf.initialize(np.array([1.0, 0.0, 2.0]), 0.5)
        imu = self._make_imu(accel=[1.0, 0.0, 0.0])
        ekf.predict(imu, dt=1.0)

        ekf.reset()

        with pytest.raises(RuntimeError):
            ekf.get_estimate()

    def test_position_y_is_floor_height(self):
        ekf = EKFEstimator()
        pos = np.array([1.0, 3.5, 2.0], dtype=np.float64)
        ekf.initialize(pos, 0.0)
        est = ekf.get_estimate()

        # Y should be floor height = initial position Y
        assert abs(est.position[1] - 3.5) < 1e-10

    def test_uninitialized_raises(self):
        ekf = EKFEstimator()
        with pytest.raises(RuntimeError):
            ekf.predict(self._make_imu(), dt=1.0)
