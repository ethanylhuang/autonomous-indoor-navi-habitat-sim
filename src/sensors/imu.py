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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IMUReading:
    linear_acceleration: NDArray[np.float64]  # [3] m/s^2, world frame
    angular_velocity: NDArray[np.float64]  # [3] rad/s, world frame
    timestamp_step: int  # sim step count


# ---------------------------------------------------------------------------
# Quaternion helpers (private)
# ---------------------------------------------------------------------------

def _quat_inverse(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a unit quaternion [w, x, y, z]."""
    inv = q.copy()
    inv[1:] *= -1.0  # conjugate == inverse for unit quats
    return inv


def _quat_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product q1 * q2 for [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


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
                timestamp_step=self._step,
            )

        # -- Linear velocity and acceleration ------------------------------
        velocity = (pos - self._prev_position) / self._dt
        acceleration = (velocity - self._prev_velocity) / self._dt

        # -- Angular velocity via quaternion delta -------------------------
        # q_delta = q_current * q_prev_inv
        q_prev_inv = _quat_inverse(self._prev_rotation)
        q_delta = _quat_multiply(rot, q_prev_inv)

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
            timestamp_step=self._step,
        )

    def reset(self) -> None:
        """Clear all history. Next update() returns zeros."""
        self._step = 0
        self._prev_position = None
        self._prev_velocity = None
        self._prev_rotation = None
