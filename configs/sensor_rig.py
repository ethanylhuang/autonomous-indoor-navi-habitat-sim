"""Sensor specification factories and resolution/FOV/position constants.

All sensor magic numbers live here. Each factory returns a habitat_sim.CameraSensorSpec
ready for mounting on an agent. IMU is not a habitat-sim sensor and is not defined here.
"""

import math
from typing import List, Tuple

import habitat_sim
from habitat_sim.sensor import SensorType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_HEIGHT: float = 1.5  # meters above agent origin (Y axis)
RGB_RESOLUTION: Tuple[int, int] = (480, 640)  # (H, W)
DEPTH_RESOLUTION: Tuple[int, int] = (480, 640)  # (H, W)
SEMANTIC_RESOLUTION: Tuple[int, int] = (480, 640)  # match RGB for pixel alignment
HFOV: int = 90  # degrees

_FORWARD: List[float] = [0.0, 0.0, 0.0]
_REAR: List[float] = [0.0, math.pi, 0.0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_spec(
    uuid: str,
    sensor_type: SensorType,
    orientation: List[float],
    resolution: Tuple[int, int],
) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = list(resolution)
    spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    spec.orientation = orientation
    spec.hfov = HFOV
    return spec


def forward_rgb_spec() -> habitat_sim.CameraSensorSpec:
    """Forward-facing RGB camera."""
    return _make_spec("forward_rgb", SensorType.COLOR, _FORWARD, RGB_RESOLUTION)


def rear_rgb_spec() -> habitat_sim.CameraSensorSpec:
    """Rear-facing RGB camera (rotated 180 degrees about Y)."""
    return _make_spec("rear_rgb", SensorType.COLOR, _REAR, RGB_RESOLUTION)


def depth_spec() -> habitat_sim.CameraSensorSpec:
    """Forward-facing depth sensor, co-located with forward_rgb."""
    return _make_spec("depth", SensorType.DEPTH, _FORWARD, DEPTH_RESOLUTION)


def rear_depth_spec() -> habitat_sim.CameraSensorSpec:
    """Rear-facing depth sensor, co-located with rear_rgb."""
    return _make_spec("rear_depth", SensorType.DEPTH, _REAR, DEPTH_RESOLUTION)


def forward_semantic_spec() -> habitat_sim.CameraSensorSpec:
    """Forward-facing semantic sensor, co-located with forward_rgb."""
    return _make_spec("forward_semantic", SensorType.SEMANTIC, _FORWARD, SEMANTIC_RESOLUTION)


def rear_semantic_spec() -> habitat_sim.CameraSensorSpec:
    """Rear-facing semantic sensor, co-located with rear_rgb."""
    return _make_spec("rear_semantic", SensorType.SEMANTIC, _REAR, SEMANTIC_RESOLUTION)


def all_sensor_specs() -> List[habitat_sim.CameraSensorSpec]:
    """All sensor specs for M1+M2."""
    return [
        forward_rgb_spec(),
        rear_rgb_spec(),
        depth_spec(),
        rear_depth_spec(),
        forward_semantic_spec(),
        rear_semantic_spec(),
    ]
