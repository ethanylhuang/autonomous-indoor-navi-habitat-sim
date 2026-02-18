"""Sensor specification factories and resolution/FOV/position constants.

All sensor magic numbers live here. Each factory returns a habitat_sim.CameraSensorSpec
ready for mounting on an agent. IMU is not a habitat-sim sensor and is not defined here.
"""

import math
from typing import List

import habitat_sim
from habitat_sim.sensor import SensorType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_HEIGHT: float = 1.5  # meters above agent origin (Y axis)
RGB_RESOLUTION: tuple[int, int] = (480, 640)  # (H, W)
DEPTH_RESOLUTION: tuple[int, int] = (480, 640)  # (H, W)
HFOV: int = 90  # degrees


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def forward_rgb_spec() -> habitat_sim.CameraSensorSpec:
    """Forward-facing RGB camera."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "forward_rgb"
    spec.sensor_type = SensorType.COLOR
    spec.resolution = list(RGB_RESOLUTION)
    spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    spec.orientation = [0.0, 0.0, 0.0]
    spec.hfov = HFOV
    return spec


def rear_rgb_spec() -> habitat_sim.CameraSensorSpec:
    """Rear-facing RGB camera (rotated 180 degrees about Y)."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "rear_rgb"
    spec.sensor_type = SensorType.COLOR
    spec.resolution = list(RGB_RESOLUTION)
    spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    spec.orientation = [0.0, math.pi, 0.0]
    spec.hfov = HFOV
    return spec


def depth_spec() -> habitat_sim.CameraSensorSpec:
    """Forward-facing depth sensor, co-located with forward_rgb."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "depth"
    spec.sensor_type = SensorType.DEPTH
    spec.resolution = list(DEPTH_RESOLUTION)
    spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    spec.orientation = [0.0, 0.0, 0.0]
    spec.hfov = HFOV
    return spec


def all_sensor_specs() -> List[habitat_sim.CameraSensorSpec]:
    """All sensor specs for M1. Extend this list in future milestones."""
    return [forward_rgb_spec(), rear_rgb_spec(), depth_spec()]
