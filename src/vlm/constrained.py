"""Constrained VLM object selection for semantic navigation.

Builds candidate lists from SemanticSceneIndex and queries VLM with a
constrained selection prompt to prevent hallucination.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from src.perception.semantic_scene import SemanticSceneIndex, get_navigable_objects


@dataclass
class ObjectCandidate:
    """Single object candidate for VLM selection."""

    object_id: int
    label: str
    instance_name: str  # e.g., "sofa #1 (region 3)"
    region_id: int
    navmesh_position: NDArray[np.float64]  # [3]
    distance_from_agent: float  # meters


@dataclass
class ConstrainedVLMResponse:
    """Response from constrained VLM object selection query."""

    selected_object_id: Optional[int]  # None if no match
    selected_label: str
    reasoning: str
    confidence: float  # 0-1
    is_valid: bool
    no_match_reason: Optional[str]  # "none_match", "parse_error", etc.


class ObjectCandidateBuilder:
    """Builds and filters object candidate lists for VLM selection."""

    def __init__(self, max_candidates: int = 100) -> None:
        """Initialize the candidate builder.

        Args:
            max_candidates: Maximum number of candidates to return.
        """
        self._max_candidates = max_candidates

    def build_candidates(
        self,
        semantic_index: SemanticSceneIndex,
        agent_position: NDArray[np.float64],
        pathfinder=None,
    ) -> List[ObjectCandidate]:
        """Build candidate list from semantic index.

        Args:
            semantic_index: Scene semantic index.
            agent_position: [3] current agent world position.
            pathfinder: Optional pathfinder to filter reachable objects only.

        Returns:
            List of ObjectCandidate, sorted by distance, capped at max_candidates.
        """
        navigable_objects = get_navigable_objects(semantic_index)

        candidates: List[ObjectCandidate] = []
        for obj in navigable_objects:
            if obj.navmesh_position is None:
                continue

            # If pathfinder provided, check reachability
            if pathfinder is not None:
                from habitat_sim import ShortestPath
                path = ShortestPath()
                path.requested_start = agent_position
                path.requested_end = obj.navmesh_position
                found = pathfinder.find_path(path)
                if not found or len(path.points) == 0 or path.geodesic_distance == float("inf"):
                    continue

            distance = float(
                np.linalg.norm(obj.navmesh_position - agent_position)
            )

            candidate = ObjectCandidate(
                object_id=obj.object_id,
                label=obj.label,
                instance_name=obj.instance_name,
                region_id=obj.region_id,
                navmesh_position=obj.navmesh_position,
                distance_from_agent=distance,
            )
            candidates.append(candidate)

        # Sort by distance ascending
        candidates.sort(key=lambda c: c.distance_from_agent)

        # Cap at max_candidates
        return candidates[: self._max_candidates]
