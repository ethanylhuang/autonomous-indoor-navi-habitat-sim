"""Central vehicle facade.

Owns the habitat_sim.Simulator instance. This is the ONLY module that touches
the simulator directly. Provides typed dataclasses for observations and state,
and wraps pathfinding for NavMesh access.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat_sim.nav import ShortestPath
from numpy.typing import NDArray

from configs.sim_config import AgentParams, SimParams, make_sim_config
from src.sensors.imu import IMUReading, SimulatedIMU


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    position: NDArray[np.float64]  # [3] world frame
    rotation: NDArray[np.float64]  # [4] quaternion [w, x, y, z]
    step_count: int
    collided: bool
    imu: IMUReading


@dataclass
class Observations:
    # M1 fields (unchanged)
    forward_rgb: NDArray[np.uint8]  # [480, 640, 4] RGBA
    rear_rgb: NDArray[np.uint8]  # [480, 640, 4] RGBA
    depth: NDArray[np.float32]  # [480, 640] meters (forward)
    imu: IMUReading
    state: VehicleState
    # M2 additions
    rear_depth: NDArray[np.float32]  # [480, 640] meters (rear)
    forward_semantic: NDArray[np.uint32]  # [480, 640] semantic IDs (forward)
    rear_semantic: NDArray[np.uint32]  # [480, 640] semantic IDs (rear)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _magnum_quat_to_array(q) -> NDArray[np.float64]:
    """Convert a habitat-sim quaternion to a numpy [w, x, y, z] array.

    habitat-sim 0.3.x returns numpy-quaternion objects from AgentState.rotation,
    which use .w/.x/.y/.z accessors (not .scalar/.vector like Magnum).
    """
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

class Vehicle:
    """Autonomous vehicle facade over habitat-sim.

    Creates the simulator, mounts sensors, and exposes step/observe/pathfinding.
    """

    VALID_ACTIONS = {"move_forward", "turn_left", "turn_right"}

    def __init__(
        self,
        sim_params: Optional[SimParams] = None,
        agent_params: Optional[AgentParams] = None,
        imu_dt: float = 1.0,
    ) -> None:
        cfg = make_sim_config(sim_params, agent_params)
        self._sim = habitat_sim.Simulator(cfg)
        self._agent = self._sim.initialize_agent(0)
        self._imu = SimulatedIMU(dt=imu_dt)
        self._step_count: int = 0

        # Fail fast if NavMesh is not loaded
        if not self._sim.pathfinder.is_loaded:
            self.close()
            raise RuntimeError(
                "NavMesh not loaded. Ensure the scene file has an associated "
                ".navmesh file or a baked-in navigation mesh."
            )

    # -- Public API --------------------------------------------------------

    def step(self, action: str) -> Observations:
        """Execute one discrete action and return observations."""
        if action not in self.VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {self.VALID_ACTIONS}"
            )

        raw_obs = self._sim.step(action)
        self._step_count += 1
        return self._build_observations(raw_obs)

    def get_initial_observations(self) -> Observations:
        """Get observations for the current state without stepping."""
        raw_obs = self._sim.get_sensor_observations()
        return self._build_observations(raw_obs)

    def find_path(
        self,
        start: NDArray,
        goal: NDArray,
    ) -> Tuple[List[NDArray], float]:
        """Find a path between two points on the NavMesh.

        Returns:
            (waypoints, geodesic_distance). Waypoints is a list of [3] arrays.
        """
        path = ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        self._sim.pathfinder.find_path(path)
        waypoints = [np.array(p, dtype=np.float64) for p in path.points]
        return waypoints, path.geodesic_distance

    def get_navmesh_bounds(self) -> Tuple[NDArray, NDArray]:
        """Get the axis-aligned bounding box of the NavMesh.

        Returns:
            (lower_bound, upper_bound), each [3] arrays.
        """
        bounds = self._sim.pathfinder.get_bounds()
        return (
            np.array(bounds[0], dtype=np.float64),
            np.array(bounds[1], dtype=np.float64),
        )

    def get_topdown_navmesh(
        self,
        meters_per_pixel: float = 0.05,
    ) -> NDArray[np.bool_]:
        """Rasterize the NavMesh into a 2D boolean grid.

        True = navigable, False = obstacle/void. Uses the built-in
        pathfinder.get_topdown_view() for correct and fast rasterization.
        Height is taken from the current agent Y position.
        """
        height = self._agent.get_state().position[1]
        return self._sim.pathfinder.get_topdown_view(
            meters_per_pixel, height
        )

    def reset(self) -> Observations:
        """Reset the agent to a random navigable position."""
        # Get a random navigable point and place the agent there
        start = self._sim.pathfinder.get_random_navigable_point()
        agent_state = self._agent.get_state()
        agent_state.position = start
        self._agent.set_state(agent_state)
        self._imu.reset()
        self._step_count = 0
        return self.get_initial_observations()

    def close(self) -> None:
        """Release simulator resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    @property
    def pathfinder(self):
        """Direct access to the simulator's PathFinder."""
        return self._sim.pathfinder

    # -- Private -----------------------------------------------------------

    def _build_observations(self, raw_obs: dict) -> Observations:
        """Convert raw habitat-sim observations dict to typed Observations."""
        agent_state = self._agent.get_state()
        position = np.array(agent_state.position, dtype=np.float64)
        rotation = _magnum_quat_to_array(agent_state.rotation)

        imu_reading = self._imu.update(position, rotation)
        collided = getattr(self._sim, "previous_step_collided", False)

        state = VehicleState(
            position=position,
            rotation=rotation,
            step_count=self._step_count,
            collided=collided,
            imu=imu_reading,
        )

        return Observations(
            forward_rgb=raw_obs["forward_rgb"],
            rear_rgb=raw_obs["rear_rgb"],
            depth=raw_obs["depth"],
            imu=imu_reading,
            state=state,
            rear_depth=raw_obs["rear_depth"],
            forward_semantic=raw_obs["forward_semantic"],
            rear_semantic=raw_obs["rear_semantic"],
        )
