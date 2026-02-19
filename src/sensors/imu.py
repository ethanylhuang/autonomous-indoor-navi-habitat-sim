"""Simulated IMU via AgentState differencing.

Computes linear acceleration and angular velocity by differencing consecutive
agent state snapshots. This is NOT a habitat-sim sensor -- it is a pure Python
class that operates on position/rotation arrays.

Quaternion convention: [w, x, y, z] matching habitat-sim's Magnum quaternion.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.utils.transforms import quat_multiply, quat_to_rotation_matrix


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IMUReading:
    linear_acceleration: NDArray[np.float64]  # [3] m/s^2, world frame
    angular_velocity: NDArray[np.float64]  # [3] rad/s, world frame
    linear_velocity: NDArray[np.float64]  # [3] m/s, world frame
    timestamp_step: int  # sim step count

    def to_body_frame(self, rotation_quat: NDArray[np.float64]) -> "IMUReading":
        """Convert world-frame readings to body-frame using agent rotation.

        Body frame convention: X=right, Y=up, Z=backward (agent-local).

        Args:
            rotation_quat: [4] agent rotation quaternion [w, x, y, z] in world frame.

        Returns:
            New IMUReading with acceleration and angular velocity in body frame.
        """
        # R transforms body -> world, so R^T transforms world -> body
        R = quat_to_rotation_matrix(rotation_quat)
        R_inv = R.T
        return IMUReading(
            linear_acceleration=R_inv @ self.linear_acceleration,
            angular_velocity=R_inv @ self.angular_velocity,
            linear_velocity=R_inv @ self.linear_velocity,
            timestamp_step=self.timestamp_step,
        )


# ---------------------------------------------------------------------------
# SimulatedIMU
# ---------------------------------------------------------------------------

class SimulatedIMU:
    """State-differencing IMU.

    Call update() once per sim step with the agent's current position and
    rotation quaternion. Returns an IMUReading with linear acceleration and
    angular velocity.

    First call returns zeros (no prior state to difference against).
    Second call has velocity but zero acceleration (only one velocity sample).
    """

    def __init__(self, dt: float = 1.0) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._dt: float = dt
        self._step: int = 0
        self._prev_position: Optional[NDArray[np.float64]] = None
        self._prev_velocity: Optional[NDArray[np.float64]] = None
        self._prev_rotation: Optional[NDArray[np.float64]] = None

    def update(
        self,
        position: NDArray,
        rotation_quat: NDArray,
    ) -> IMUReading:
        """Compute IMU reading from current agent state.

        Args:
            position: [3] world-frame position.
            rotation_quat: [4] quaternion [w, x, y, z].

        Returns:
            IMUReading with acceleration, angular velocity, and step count.
        """
        pos = np.asarray(position, dtype=np.float64)
        rot = np.asarray(rotation_quat, dtype=np.float64)

        self._step += 1

        # -- First call: no history, return zeros --------------------------
        if self._prev_position is None:
            self._prev_position = pos
            self._prev_rotation = rot
            self._prev_velocity = np.zeros(3, dtype=np.float64)
            return IMUReading(
                linear_acceleration=np.zeros(3, dtype=np.float64),
                angular_velocity=np.zeros(3, dtype=np.float64),
                linear_velocity=np.zeros(3, dtype=np.float64),
                timestamp_step=self._step,
            )

        # -- Linear velocity and acceleration ------------------------------
        velocity = (pos - self._prev_position) / self._dt
        acceleration = (velocity - self._prev_velocity) / self._dt

        # -- Angular velocity via quaternion delta -------------------------
        # q_delta = q_current * q_prev_inv
        # Inverse of a unit quaternion = conjugate
        q_prev_inv = self._prev_rotation.copy()
        q_prev_inv[1:] *= -1.0
        q_delta = quat_multiply(rot, q_prev_inv)

        # Ensure positive scalar part for consistent small-angle approx
        if q_delta[0] < 0:
            q_delta = -q_delta

        # Small-angle approximation: omega ~= 2 * vec(q_delta) / dt
        angular_velocity = 2.0 * q_delta[1:] / self._dt

        # -- Update history ------------------------------------------------
        self._prev_position = pos
        self._prev_velocity = velocity
        self._prev_rotation = rot

        return IMUReading(
            linear_acceleration=acceleration,
            angular_velocity=angular_velocity,
            linear_velocity=velocity.copy(),
            timestamp_step=self._step,
        )

    def reset(self) -> None:
        """Clear all history. Next update() returns zeros."""
        self._step = 0
        self._prev_position = None
        self._prev_velocity = None
        self._prev_rotation = None
