"""Semantic constrained navigator for VLM-guided object navigation.

Uses VLM to select target objects from a constrained list, then navigates
to the selected object's navmesh position using classical planning.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.control.controller import NavigationController
from src.perception.occupancy_grid import OccupancyGridData
from src.perception.semantic_scene import SemanticSceneIndex
from src.planning.global_planner import GlobalPlanner
from src.state_estimation.estimator import PoseEstimate
from src.vehicle import Observations
from src.vlm.client import VLMClient
from src.vlm.constrained import ConstrainedVLMResponse, ObjectCandidateBuilder

logger = logging.getLogger(__name__)


@dataclass
class SemanticNavStatus:
    """Status of semantic navigation episode."""

    mode: str  # "semantic_nav" or "idle"
    instruction: str
    phase: str  # "selecting" | "navigating" | "completed"
    selected_object_label: Optional[str]
    selected_object_id: Optional[int]
    goal_position: Optional[Tuple[float, float, float]]
    goal_reached: bool
    steps_taken: int
    vlm_calls: int
    vlm_reasoning: str
    vlm_confidence: float
    termination_reason: Optional[str]
    distance_to_goal: Optional[float]
    spl: Optional[float]


@dataclass
class EpisodeMetrics:
    """Metrics for a single semantic navigation episode."""

    instruction: str
    ground_truth_object_id: Optional[int]
    selected_object_id: Optional[int]
    selection_correct: Optional[bool]
    goal_reached: bool
    spl: float
    path_length: float
    geodesic_distance: float
    steps: int
    vlm_calls: int
    termination_reason: str


class SemanticConstrainedNavigator:
    """Navigator for VLM-guided semantic object navigation.

    Uses constrained VLM selection from a pre-computed object list,
    then delegates navigation to classical stack.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        semantic_index: SemanticSceneIndex,
        global_planner: GlobalPlanner,
        controller: NavigationController,
        candidate_builder: ObjectCandidateBuilder,
        pathfinder,
        max_steps: int = 500,
    ) -> None:
        """Initialize the semantic navigator.

        Args:
            vlm_client: VLM API client for object selection.
            semantic_index: Scene semantic index.
            global_planner: NavMesh-based global path planner.
            controller: Navigation controller.
            candidate_builder: Builder for object candidate lists.
            pathfinder: habitat_sim PathFinder instance.
            max_steps: Maximum steps before terminating episode.
        """
        self._vlm = vlm_client
        self._semantic_index = semantic_index
        self._global_planner = global_planner
        self._controller = controller
        self._candidate_builder = candidate_builder
        self._pathfinder = pathfinder
        self._max_steps = max_steps

        # Episode state
        self._instruction: Optional[str] = None
        self._phase: str = "idle"  # "selecting" | "navigating" | "completed"
        self._selected_object_id: Optional[int] = None
        self._selected_label: Optional[str] = None
        self._goal_position: Optional[NDArray[np.float64]] = None
        self._vlm_response: Optional[ConstrainedVLMResponse] = None
        self._vlm_calls: int = 0
        self._start_position: Optional[NDArray[np.float64]] = None
        self._geodesic_distance: float = 0.0

    def start_episode(
        self,
        instruction: str,
        start_position: NDArray[np.float64],
        start_rotation: NDArray[np.float64],
    ) -> SemanticNavStatus:
        """Begin semantic navigation episode.

        Args:
            instruction: Natural language instruction (e.g., "find something to sit on").
            start_position: [3] initial agent position.
            start_rotation: [4] initial agent rotation quaternion.

        Returns:
            Initial SemanticNavStatus.
        """
        self._instruction = instruction
        self._phase = "selecting"
        self._selected_object_id = None
        self._selected_label = None
        self._goal_position = None
        self._vlm_response = None
        self._vlm_calls = 0
        self._start_position = start_position.copy()
        self._geodesic_distance = 0.0
        self._global_planner.reset()
        self._controller.reset()

        logger.info("Semantic nav episode started: '%s'", instruction)

        # Immediately select object
        return self._select_object(start_position, start_rotation)

    def _select_object(
        self,
        agent_position: NDArray[np.float64],
        agent_rotation: NDArray[np.float64],
    ) -> SemanticNavStatus:
        """Query VLM to select target object from candidates.

        Updates phase to "navigating" if successful, "completed" if no match.

        Args:
            agent_position: [3] current agent position.
            agent_rotation: [4] current agent rotation.

        Returns:
            SemanticNavStatus after selection.
        """
        # Build candidate list (only reachable objects)
        candidates = self._candidate_builder.build_candidates(
            self._semantic_index, agent_position, pathfinder=self._pathfinder
        )

        if not candidates:
            logger.warning("No navigable objects in scene for selection")
            self._phase = "completed"
            self._vlm_response = ConstrainedVLMResponse(
                selected_object_id=None,
                selected_label="",
                reasoning="No navigable objects in scene",
                confidence=0.0,
                is_valid=False,
                no_match_reason="no_candidates",
            )
            return self._build_status()

        # Log candidates being sent to VLM
        unique_labels = sorted(set(c.label for c in candidates))
        logger.info(
            "Sending %d candidates to VLM (%d unique labels): %s",
            len(candidates),
            len(unique_labels),
            ", ".join(unique_labels[:20]) + ("..." if len(unique_labels) > 20 else ""),
        )

        # Query VLM
        self._vlm_calls += 1
        response = self._vlm.select_object_constrained(
            self._instruction, candidates
        )
        self._vlm_response = response

        logger.info(
            "VLM selection: label='%s', confidence=%.2f, reasoning='%s'",
            response.selected_label,
            response.confidence,
            response.reasoning,
        )

        if not response.is_valid or response.selected_object_id is None:
            logger.warning("VLM selection failed: %s", response.no_match_reason)
            self._phase = "completed"
            return self._build_status()

        # Lookup selected object in semantic index
        obj = self._semantic_index.objects.get(response.selected_object_id)
        if obj is None or obj.navmesh_position is None:
            logger.error(
                "Selected object_id %d not found or not navigable",
                response.selected_object_id,
            )
            self._phase = "completed"
            self._vlm_response.is_valid = False
            self._vlm_response.no_match_reason = "object_not_found"
            return self._build_status()

        # Store selection and goal
        self._selected_object_id = obj.object_id
        self._selected_label = obj.label
        self._goal_position = obj.navmesh_position.copy()

        logger.info(
            "Selected '%s' at position [%.2f, %.2f, %.2f]",
            obj.instance_name,
            self._goal_position[0],
            self._goal_position[1],
            self._goal_position[2],
        )

        # Plan global path
        global_path = self._global_planner.plan(
            lambda start, goal: self._pathfinder_wrapper(start, goal),
            agent_position,
            self._goal_position,
        )

        if not global_path.is_valid:
            logger.error("Global path planning failed to selected object")
            self._phase = "completed"
            self._vlm_response.is_valid = False
            self._vlm_response.no_match_reason = "path_planning_failed"
            return self._build_status()

        self._geodesic_distance = global_path.geodesic_distance
        logger.info(
            "Global path planned: %d waypoints, geodesic=%.2fm",
            len(global_path.waypoints),
            global_path.geodesic_distance,
        )

        # Start navigation controller
        self._controller.start_episode(
            agent_position,
            agent_rotation,
            self._goal_position,
            global_path,
            vehicle_find_path=lambda start, goal: self._pathfinder_wrapper(
                start, goal
            ),
        )

        self._phase = "navigating"
        return self._build_status()

    def step(
        self,
        pose_estimate: PoseEstimate,
        occupancy_grid: OccupancyGridData,
        rear_obstacle_detected: bool,
        collided: bool,
    ) -> Tuple[str, SemanticNavStatus]:
        """Execute one navigation step.

        Args:
            pose_estimate: Current pose estimate from state estimator.
            occupancy_grid: OccupancyGridData for local planning.
            rear_obstacle_detected: From rear camera obstacle detector.
            collided: True if collision occurred last step.

        Returns:
            Tuple of (action_to_execute, status).
        """
        if self._phase != "navigating":
            # Not in navigation phase, return idle
            return "move_forward", self._build_status()

        # Delegate to navigation controller
        nav_status = self._controller.step(
            pose_estimate, occupancy_grid, rear_obstacle_detected, collided
        )

        # Check termination
        if self._controller.is_episode_done:
            self._phase = "completed"
            logger.info(
                "Semantic nav episode completed: %s",
                nav_status.steps_taken,
            )

        return nav_status.action, self._build_status()

    def _pathfinder_wrapper(
        self,
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
    ) -> Tuple[list, float]:
        """Wrapper for pathfinder to match GlobalPlanner vehicle_find_path signature.

        Args:
            start: [3] start position.
            goal: [3] goal position.

        Returns:
            Tuple of (waypoints, geodesic_distance).
        """
        from habitat_sim import ShortestPath

        path = ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        found = self._pathfinder.find_path(path)

        if not found or len(path.points) == 0:
            return [], float("inf")

        return path.points, path.geodesic_distance

    def _build_status(self) -> SemanticNavStatus:
        """Build current navigation status.

        Returns:
            SemanticNavStatus dataclass.
        """
        goal_tuple = None
        if self._goal_position is not None:
            goal_tuple = (
                float(self._goal_position[0]),
                float(self._goal_position[1]),
                float(self._goal_position[2]),
            )

        goal_reached = self._controller._goal_reached if self._phase in ("navigating", "completed") else False
        steps_taken = self._controller._steps_taken if self._phase in ("navigating", "completed") else 0
        termination_reason = self._controller._termination_reason if self._phase == "completed" else None

        distance_to_goal = None
        if self._goal_position is not None and self._controller._prev_position is not None:
            distance_to_goal = float(
                np.linalg.norm(
                    self._goal_position - self._controller._prev_position
                )
            )

        # Compute SPL if episode completed
        spl = None
        if self._phase == "completed" and self._controller._termination_reason is not None:
            result = self._controller.finish_episode()
            spl = result.spl

        vlm_reasoning = ""
        vlm_confidence = 0.0
        if self._vlm_response is not None:
            vlm_reasoning = self._vlm_response.reasoning
            vlm_confidence = self._vlm_response.confidence

        mode = "semantic_nav" if self._instruction else "idle"

        return SemanticNavStatus(
            mode=mode,
            instruction=self._instruction or "",
            phase=self._phase,
            selected_object_label=self._selected_label,
            selected_object_id=self._selected_object_id,
            goal_position=goal_tuple,
            goal_reached=goal_reached,
            steps_taken=steps_taken,
            vlm_calls=self._vlm_calls,
            vlm_reasoning=vlm_reasoning,
            vlm_confidence=vlm_confidence,
            termination_reason=termination_reason,
            distance_to_goal=distance_to_goal,
            spl=spl,
        )

    def finish_episode(self) -> EpisodeMetrics:
        """Compute final episode metrics.

        Returns:
            EpisodeMetrics dataclass.
        """
        result = self._controller.finish_episode()

        return EpisodeMetrics(
            instruction=self._instruction or "",
            ground_truth_object_id=None,  # Not available without benchmark dataset
            selected_object_id=self._selected_object_id,
            selection_correct=None,  # Not available without ground truth
            goal_reached=result.success,
            spl=result.spl,
            path_length=result.path_length,
            geodesic_distance=result.geodesic_distance,
            steps=result.steps,
            vlm_calls=self._vlm_calls,
            termination_reason=result.termination_reason,
        )

    @property
    def is_done(self) -> bool:
        """Check if episode has terminated."""
        return self._phase == "completed"

    def reset(self) -> None:
        """Reset all episode state."""
        self._instruction = None
        self._phase = "idle"
        self._selected_object_id = None
        self._selected_label = None
        self._goal_position = None
        self._vlm_response = None
        self._vlm_calls = 0
        self._start_position = None
        self._geodesic_distance = 0.0
        self._global_planner.reset()
        self._controller.reset()
