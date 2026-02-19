"""Shared math utilities for coordinate transforms and camera geometry.

Quaternion convention: [w, x, y, z] matching habitat-sim's Magnum quaternion.
Camera convention: X=right, Y=up, Z=backward (camera looks along -Z).
"""

import math

import numpy as np
from numpy.typing import NDArray


def focal_length_from_hfov(hfov_deg: float, image_width: int) -> float:
    """Compute focal length in pixels from horizontal field of view.

    Args:
        hfov_deg: Horizontal field of view in degrees.
        image_width: Image width in pixels.

    Returns:
        Focal length in pixels (assumes square pixels, so fx == fy).
    """
    hfov_rad = math.radians(hfov_deg)
    return image_width / (2.0 * math.tan(hfov_rad / 2.0))


def yaw_from_quaternion(q: NDArray[np.float64]) -> float:
    """Extract yaw (rotation about Y axis) from a [w, x, y, z] quaternion.

    Returns yaw in radians. Positive yaw = counter-clockwise when viewed
    from above (right-hand rule about Y-up).
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quaternion_from_yaw(yaw: float) -> NDArray[np.float64]:
    """Create a Y-axis rotation quaternion [w, x, y, z] from yaw angle.

    Args:
        yaw: Rotation about Y axis in radians.

    Returns:
        [4] quaternion [w, x, y, z].
    """
    return np.array([
        math.cos(yaw / 2.0),
        0.0,
        math.sin(yaw / 2.0),
        0.0,
    ], dtype=np.float64)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]. NaN-safe and O(1)."""
    if math.isnan(angle):
        return 0.0
    return math.atan2(math.sin(angle), math.cos(angle))


def quat_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product q1 * q2 for [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
