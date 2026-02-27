"""VLM-guided pixel-based navigation orchestrator.

Combines VLM pixel target selection with classical global/local planning.
VLM outputs pixel coordinates -> projected to world -> snapped to NavMesh ->
global planner computes path -> local planner selects obstacle-aware actions.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.planning.global_planner import GlobalPlanner
from src.planning.local_planner import LocalPlanner
from src.utils.transforms import yaw_from_quaternion
from src.vehicle import Observations
from src.vlm.client import PixelTarget, VLMClient, VLMPixelResponse, WorldTarget
from src.vlm.projection import pixel_to_world, snap_to_navmesh

logger = logging.getLogger(__name__)


@dataclass
class VLMNavStatus:
    """Status of VLM-guided navigation episode."""

    mode: str  # "vlm_nav" or "idle"
    instruction: str
    goal_reached: bool
    steps_taken: int
    vlm_calls: int
    last_vlm_reasoning: str
    confidence: float
    target_visible: bool
    termination_reason: Optional[str]
    # Pixel-based navigation fields
    current_pixel: Optional[Tuple[int, int]]  # (u, v)
    current_world_target: Optional[Tuple[float, float, float]]  # [x, y, z]
    target_valid: bool
    target_failure_reason: Optional[str]
    depth_at_target: float


class VLMNavigator:
    """Hierarchical navigator: VLM pixel selection + classical planning.

    VLM is queried every ~30 steps for (u, v) pixel target.
    Pixel -> world projection -> NavMesh snap -> GlobalPlanner path.
    LocalPlanner runs every step for obstacle-aware action selection.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        global_planner: GlobalPlanner,
        local_planner: LocalPlanner,
        pathfinder,
        query_interval: int = 30,
        max_steps: int = 500,
    ) -> None:
        """Initialize the VLM navigator.

        Args:
            vlm_client: VLM API client for pixel target queries.
            global_planner: NavMesh-based global path planner.
            local_planner: DWA-based local obstacle-aware planner.
            pathfinder: habitat_sim PathFinder instance for NavMesh ops.
            query_interval: Steps between VLM queries.
            max_steps: Maximum steps before terminating episode.
        """
        self._vlm = vlm_client
        self._global_planner = global_planner
        self._local_planner = local_planner
        self._pathfinder = pathfinder
        self._query_interval = query_interval
        self._max_steps = max_steps

        # Episode state
        self._instruction: Optional[str] = None
        self._current_pixel: Optional[PixelTarget] = None
        self._current_world_target: Optional[WorldTarget] = None
        self._steps_since_query: int = 0
        self._total_steps: int = 0
        self._vlm_calls: int = 0
        self._last_reasoning: str = ""
        self._last_confidence: float = 0.0
        self._last_target_visible: bool = False
        self._goal_reached: bool = False
        self._termination_reason: Optional[str] = None

    def start_episode(self, instruction: str) -> None:
        """Begin navigation to a semantic goal.

        Args:
            instruction: Natural language goal (e.g., "go to the bedroom").
        """
        self._instruction = instruction
        self._current_pixel = None
        self._current_world_target = None
        self._steps_since_query = self._query_interval  # Force immediate query
        self._total_steps = 0
        self._vlm_calls = 0
        self._last_reasoning = ""
        self._last_confidence = 0.0
        self._last_target_visible = False
        self._goal_reached = False
        self._termination_reason = None
        self._global_planner.reset()
        self._local_planner.reset()
        logger.info("VLM pixel nav episode started: '%s'", instruction)

    def step(
        self,
        obs: Observations,
        occupancy_grid,
        rear_obstacle_detected: bool,
        vehicle_find_path,
    ) -> Tuple[str, VLMNavStatus]:
        """Execute one navigation step.

        Steps:
        1. Check termination conditions
        2. If time to query VLM: get pixel -> project -> snap -> plan global path
        3. Use local planner to select action (obstacle-aware)
        4. Return action and status

        Args:
            obs: Current observations from vehicle.
            occupancy_grid: OccupancyGridData for local planning.
            rear_obstacle_detected: From rear camera obstacle detector.
            vehicle_find_path: Reference to Vehicle.find_path(start, goal).

        Returns:
            Tuple of (action_to_execute, status).
        """
        self._total_steps += 1
        self._steps_since_query += 1

        # Check max steps
        if self._total_steps >= self._max_steps:
            self._termination_reason = "max_steps"
            return "move_forward", self._build_status()

        # Check if goal already reached
        if self._goal_reached:
            return "move_forward", self._build_status()

        # Query VLM for new pixel target if interval elapsed
        if self._steps_since_query >= self._query_interval:
            self._steps_since_query = 0
            vlm_response = self._query_vlm_pixel(obs)

            if vlm_response.goal_reached:
                self._goal_reached = True
                self._termination_reason = "goal_reached"
                logger.info("VLM indicated goal reached after %d steps", self._total_steps)
                return "move_forward", self._build_status()

            # Project pixel to world and snap to NavMesh
            self._current_pixel = vlm_response.pixel
            if self._current_pixel.is_valid:
                self._project_and_plan(obs, vehicle_find_path)
            else:
                logger.warning("VLM pixel invalid, skipping projection")
                self._current_world_target = None

        # Select action using local planner if we have a valid target
        if (
            self._current_world_target is not None
            and self._current_world_target.is_valid
        ):
            current_waypoint = self._global_planner.get_current_waypoint()
            if current_waypoint is not None:
                # Advance waypoint if reached
                goal_reached_via_waypoints = self._global_planner.advance_waypoint(
                    obs.state.position
                )
                if goal_reached_via_waypoints:
                    # Reached NavMesh target, query VLM again soon
                    self._steps_since_query = self._query_interval - 5
                    logger.info("Reached NavMesh target from VLM pixel")

                # Get current waypoint (may have advanced)
                current_waypoint = self._global_planner.get_current_waypoint()
                if current_waypoint is not None:
                    # Use local planner for obstacle-aware action
                    agent_yaw = yaw_from_quaternion(obs.state.rotation)
                    plan_result = self._local_planner.plan(
                        obs.state.position,
                        agent_yaw,
                        current_waypoint,
                        occupancy_grid,
                        rear_obstacle_detected,
                    )
                    action = plan_result.best_action
                else:
                    # No more waypoints, move forward (shouldn't happen)
                    action = "move_forward"
            else:
                # No waypoints, fallback
                action = "move_forward"
        else:
            # No valid world target, explore forward
            action = "move_forward"

        return action, self._build_status()

    def _query_vlm_pixel(self, obs: Observations) -> VLMPixelResponse:
        """Query VLM for pixel-coordinate navigation target.

        Args:
            obs: Current observations.

        Returns:
            VLMPixelResponse with pixel target or goal_reached flag.
        """
        # Build context from depth/state
        context = self._build_context(obs)

        response = self._vlm.get_pixel_target(
            image=obs.forward_rgb,
            instruction=self._instruction,
            context=context,
        )

        self._vlm_calls += 1
        if response.pixel.is_valid:
            self._last_reasoning = response.pixel.reasoning
            self._last_confidence = response.pixel.confidence
            self._last_target_visible = response.pixel.target_visible

            logger.info(
                "VLM query %d: pixel=(%d, %d), confidence=%.2f, reasoning='%s'",
                self._vlm_calls,
                response.pixel.u,
                response.pixel.v,
                response.pixel.confidence,
                response.pixel.reasoning,
            )
        else:
            logger.warning("VLM query %d: invalid pixel response", self._vlm_calls)

        return response

    def _project_and_plan(
        self,
        obs: Observations,
        vehicle_find_path,
    ) -> None:
        """Project current pixel to world, snap to NavMesh, and plan path.

        Updates self._current_world_target and calls GlobalPlanner.

        Args:
            obs: Current observations.
            vehicle_find_path: Reference to Vehicle.find_path(start, goal).
        """
        if self._current_pixel is None or not self._current_pixel.is_valid:
            self._current_world_target = None
            return

        # Project pixel to world
        proj_result = pixel_to_world(
            self._current_pixel.u,
            self._current_pixel.v,
            obs.depth,
            obs.state.position,
            obs.state.rotation,
        )

        if not proj_result.is_valid:
            logger.warning(
                "Pixel (%d, %d) projection failed: %s",
                self._current_pixel.u,
                self._current_pixel.v,
                proj_result.failure_reason,
            )
            self._current_world_target = WorldTarget(
                position=np.zeros(3, dtype=np.float64),
                navmesh_position=np.zeros(3, dtype=np.float64),
                depth_value=proj_result.depth_value,
                is_valid=False,
                failure_reason=proj_result.failure_reason,
            )
            return

        # Snap to NavMesh
        snapped, snap_success, snap_reason = snap_to_navmesh(
            proj_result.world_point, self._pathfinder
        )

        if not snap_success:
            logger.warning(
                "NavMesh snap failed for pixel (%d, %d): %s",
                self._current_pixel.u,
                self._current_pixel.v,
                snap_reason,
            )
            self._current_world_target = WorldTarget(
                position=proj_result.world_point,
                navmesh_position=np.zeros(3, dtype=np.float64),
                depth_value=proj_result.depth_value,
                is_valid=False,
                failure_reason=snap_reason,
            )
            return

        # Success: store world target and plan global path
        self._current_world_target = WorldTarget(
            position=proj_result.world_point,
            navmesh_position=snapped,
            depth_value=proj_result.depth_value,
            is_valid=True,
            failure_reason=None,
        )

        logger.info(
            "Pixel (%d, %d) -> world [%.2f, %.2f, %.2f] -> NavMesh [%.2f, %.2f, %.2f]",
            self._current_pixel.u,
            self._current_pixel.v,
            proj_result.world_point[0],
            proj_result.world_point[1],
            proj_result.world_point[2],
            snapped[0],
            snapped[1],
            snapped[2],
        )

        # Plan global path to NavMesh target
        global_path = self._global_planner.plan(
            vehicle_find_path, obs.state.position, snapped
        )
        if not global_path.is_valid:
            logger.warning("Global path planning failed to NavMesh target")
            self._current_world_target.is_valid = False
            self._current_world_target.failure_reason = "global_path_failed"
        else:
            logger.info(
                "Global path planned: %d waypoints, geodesic=%.2fm",
                len(global_path.waypoints),
                global_path.geodesic_distance,
            )

    def _build_context(self, obs: Observations) -> str:
        """Build context string from observations for VLM prompt.

        Args:
            obs: Current observations.

        Returns:
            Context string with step count and obstacle info.
        """
        context_parts = [f"Step {self._total_steps} of {self._max_steps}."]

        # Analyze depth for obstacle context
        depth = obs.depth
        h, w = depth.shape
        y_start, y_end = int(h * 0.2), int(h * 0.8)
        center_depth = depth[y_start:y_end, :]

        # Split into left, center, right zones
        third = w // 3
        left_zone = center_depth[:, :third]
        center_zone = center_depth[:, third : 2 * third]
        right_zone = center_depth[:, 2 * third :]

        def safe_percentile(arr, pct=10):
            valid = arr[(arr > 0.1) & (arr < 50) & np.isfinite(arr)]
            return float(np.percentile(valid, pct)) if len(valid) > 0 else 50.0

        left_dist = safe_percentile(left_zone, 10)
        center_dist = safe_percentile(center_zone, 10)
        right_dist = safe_percentile(right_zone, 10)

        BLOCKED_THRESHOLD = 1.0
        CLOSE_THRESHOLD = 2.0

        obstacles = []
        if center_dist < BLOCKED_THRESHOLD:
            obstacles.append(f"BLOCKED ahead ({center_dist:.1f}m)")
        elif center_dist < CLOSE_THRESHOLD:
            obstacles.append(f"Obstacle ahead ({center_dist:.1f}m)")

        if left_dist < BLOCKED_THRESHOLD:
            obstacles.append(f"Wall/obstacle on left ({left_dist:.1f}m)")
        if right_dist < BLOCKED_THRESHOLD:
            obstacles.append(f"Wall/obstacle on right ({right_dist:.1f}m)")

        if obstacles:
            context_parts.append("DEPTH: " + "; ".join(obstacles) + ".")
        else:
            context_parts.append(
                f"DEPTH: Clear ahead ({center_dist:.1f}m), "
                f"left ({left_dist:.1f}m), right ({right_dist:.1f}m)."
            )

        return " ".join(context_parts)

    def _build_status(self) -> VLMNavStatus:
        """Build current navigation status.

        Returns:
            VLMNavStatus dataclass.
        """
        pixel_tuple = None
        if self._current_pixel is not None and self._current_pixel.is_valid:
            pixel_tuple = (self._current_pixel.u, self._current_pixel.v)

        world_tuple = None
        if (
            self._current_world_target is not None
            and self._current_world_target.is_valid
        ):
            world_tuple = (
                float(self._current_world_target.navmesh_position[0]),
                float(self._current_world_target.navmesh_position[1]),
                float(self._current_world_target.navmesh_position[2]),
            )

        target_valid = (
            self._current_world_target is not None
            and self._current_world_target.is_valid
        )
        target_failure = (
            self._current_world_target.failure_reason
            if self._current_world_target is not None
            else None
        )
        depth_val = (
            self._current_world_target.depth_value
            if self._current_world_target is not None
            else 0.0
        )

        return VLMNavStatus(
            mode="vlm_nav" if self._instruction else "idle",
            instruction=self._instruction or "",
            goal_reached=self._goal_reached,
            steps_taken=self._total_steps,
            vlm_calls=self._vlm_calls,
            last_vlm_reasoning=self._last_reasoning,
            confidence=self._last_confidence,
            target_visible=self._last_target_visible,
            termination_reason=self._termination_reason,
            current_pixel=pixel_tuple,
            current_world_target=world_tuple,
            target_valid=target_valid,
            target_failure_reason=target_failure,
            depth_at_target=depth_val,
        )

    @property
    def is_done(self) -> bool:
        """Check if episode has terminated."""
        return self._termination_reason is not None

    def reset(self) -> None:
        """Reset all episode state."""
        self._instruction = None
        self._current_pixel = None
        self._current_world_target = None
        self._steps_since_query = 0
        self._total_steps = 0
        self._vlm_calls = 0
        self._last_reasoning = ""
        self._last_confidence = 0.0
        self._last_target_visible = False
        self._goal_reached = False
        self._termination_reason = None
        self._global_planner.reset()
        self._local_planner.reset()
