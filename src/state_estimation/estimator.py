"""Extended Kalman Filter for 2D+heading pose estimation.

Fuses IMU-derived displacement with VO-derived rotation to produce a robust
2D pose estimate (x, z, yaw). The vehicle operates on a floor plane, so the
EKF state is 2D+heading. The Y coordinate is fixed by the NavMesh height.

State vector: [x, z, yaw] -- position in world XZ plane, yaw about Y axis.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.sensors.imu import IMUReading
from src.perception.visual_odometry import VOEstimate
from src.utils.transforms import normalize_angle


@dataclass
class PoseEstimate:
    position: NDArray[np.float64]  # [3] world frame [x, y, z] (y from navmesh)
    yaw: float  # radians, rotation about Y axis
    covariance: NDArray[np.float64]  # [3, 3] uncertainty in [x, z, yaw]
    timestamp_step: int


class EKFEstimator:
    """Extended Kalman Filter for 2D+heading pose estimation.

    State: [x, z, yaw]
    Predict: IMU-derived displacement (dx, dz) and rotation (dyaw)
    Update: VO-derived rotation (dyaw) when valid
    """

    def __init__(
        self,
        process_noise_pos: float = 0.01,
        process_noise_yaw: float = 0.005,
        measurement_noise_yaw: float = 0.02,
        floor_height: float = 0.0,
    ) -> None:
        self._Q = np.diag([process_noise_pos, process_noise_pos, process_noise_yaw])
        self._R_yaw = measurement_noise_yaw
        self._floor_height = floor_height

        # State: [x, z, yaw]
        self._state: Optional[NDArray[np.float64]] = None
        self._P: Optional[NDArray[np.float64]] = None
        self._step: int = 0
        self._prev_yaw: Optional[float] = None

    def initialize(
        self,
        position: NDArray[np.float64],
        yaw: float,
    ) -> PoseEstimate:
        """Initialize EKF state from known position and yaw.

        Args:
            position: [3] initial world position.
            yaw: Initial yaw in radians.

        Returns:
            PoseEstimate at the initial state.
        """
        self._state = np.array([position[0], position[2], yaw], dtype=np.float64)
        self._P = np.diag([1e-6, 1e-6, 1e-6]).astype(np.float64)
        self._floor_height = position[1]
        self._step = 0
        self._prev_yaw = yaw
        return self.get_estimate()

    def predict(
        self,
        imu_reading: IMUReading,
        dt: float,
    ) -> PoseEstimate:
        """Predict step using IMU-derived displacement.

        Uses IMU linear velocity directly for displacement (velocity * dt).
        Updates state mean and covariance.

        Args:
            imu_reading: IMU reading with linear_velocity and angular_velocity.
            dt: Time step in seconds.

        Returns:
            Updated PoseEstimate.
        """
        if self._state is None:
            raise RuntimeError("EKF not initialized. Call initialize() first.")

        self._step += 1

        # Extract displacement from IMU velocity.
        # Velocity is derived from position differencing in the simulated IMU,
        # so displacement = velocity * dt gives correct position updates.
        # Using acceleration * dt^2 only captures velocity *changes* and misses
        # constant-velocity motion, causing severe drift.
        vx = imu_reading.linear_velocity[0]
        vz = imu_reading.linear_velocity[2]
        dyaw = imu_reading.angular_velocity[1] * dt

        dx = vx * dt
        dz = vz * dt

        # State transition
        self._state[0] += dx
        self._state[1] += dz
        self._state[2] += dyaw

        # Normalize yaw to [-pi, pi]
        self._state[2] = normalize_angle(self._state[2])

        # Jacobian F = identity for linear transition
        F = np.eye(3, dtype=np.float64)

        # Covariance prediction
        self._P = F @ self._P @ F.T + self._Q

        # Track predicted yaw for VO innovation computation
        self._prev_yaw = self._state[2]

        return self.get_estimate()

    def update_vo(
        self,
        vo_estimate: VOEstimate,
    ) -> PoseEstimate:
        """Update step using VO rotation measurement.

        Extracts yaw change from VO rotation matrix.
        Only applied when vo_estimate.is_valid is True.

        Args:
            vo_estimate: VO estimate with rotation matrix and validity flag.

        Returns:
            Updated PoseEstimate.
        """
        if self._state is None:
            raise RuntimeError("EKF not initialized. Call initialize() first.")

        if not vo_estimate.is_valid:
            return self.get_estimate()

        # Extract yaw from VO rotation matrix
        # VO rotation is frame-to-frame, so extract the Y-axis rotation
        R = vo_estimate.rotation
        # atan2(R[0,2], R[0,0]) gives yaw for a Y-axis rotation matrix
        vo_dyaw = np.arctan2(R[0, 2], R[0, 0])

        # Measurement model: H measures yaw only
        H = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        R_noise = np.array([[self._R_yaw]], dtype=np.float64)

        # Innovation: difference between VO yaw change and EKF predicted yaw change
        # We compare against the current state yaw
        # The measurement is the VO-estimated yaw increment
        if self._prev_yaw is not None:
            predicted_dyaw = normalize_angle(self._state[2] - self._prev_yaw)
        else:
            predicted_dyaw = 0.0

        innovation = np.array([normalize_angle(vo_dyaw - predicted_dyaw)], dtype=np.float64)

        # Kalman gain
        S = H @ self._P @ H.T + R_noise
        K = self._P @ H.T @ np.linalg.inv(S)

        # State update
        self._state = self._state + (K @ innovation).flatten()
        self._state[2] = normalize_angle(self._state[2])

        # Covariance update
        I = np.eye(3, dtype=np.float64)
        self._P = (I - K @ H) @ self._P

        self._prev_yaw = self._state[2]

        return self.get_estimate()

    def get_estimate(self) -> PoseEstimate:
        """Return the current pose estimate."""
        if self._state is None:
            raise RuntimeError("EKF not initialized. Call initialize() first.")

        position = np.array([
            self._state[0],
            self._floor_height,
            self._state[1],
        ], dtype=np.float64)

        return PoseEstimate(
            position=position,
            yaw=float(self._state[2]),
            covariance=self._P.copy(),
            timestamp_step=self._step,
        )

    def reset(self) -> None:
        """Clear all state. Must call initialize() before next use."""
        self._state = None
        self._P = None
        self._step = 0
        self._prev_yaw = None
