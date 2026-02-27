"""Spatial clustering for VLM object selection.

Groups objects by HM3D region_id (actual room assignments from scene annotations)
and infers room types from object composition.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray




@dataclass
class SpatialCluster:
    """A spatial cluster of objects with optional room type inference."""

    cluster_id: int
    cluster_label: str  # "Region A", "Region B"
    centroid: NDArray[np.float64]  # [3]
    distance_from_agent: float
    object_ids: List[int]
    inferred_room_type: Optional[str] = None  # "kitchen", "bedroom" (NO "likely" prefix)


@dataclass
class ClusteredCandidates:
    """Object candidates organized by spatial proximity."""

    clusters: List[SpatialCluster]  # sorted by distance
    candidates_by_cluster: Dict[int, List]  # cluster_id -> List[ObjectCandidate]
    unclustered_candidates: List  # List[ObjectCandidate]


class SpatialClusterer:
    """Region-based clustering using HM3D region_id annotations."""

    def __init__(self, eps: float = 2.5, min_samples: int = 2) -> None:
        """Initialize the clusterer.

        Args:
            eps: Unused (kept for API compatibility).
            min_samples: Minimum objects required to form a region cluster.
        """
        self._min_samples = min_samples

    def cluster_candidates(
        self,
        candidates: list,
        agent_position: NDArray[np.float64],
    ) -> ClusteredCandidates:
        """Cluster object candidates by HM3D region_id.

        Uses the existing region_id from HM3D semantic scene annotations,
        which represent actual room assignments from the dataset.

        Args:
            candidates: List of ObjectCandidate instances.
            agent_position: [3] current agent position.

        Returns:
            ClusteredCandidates with clusters sorted by distance.
        """
        if not candidates:
            return ClusteredCandidates(
                clusters=[],
                candidates_by_cluster={},
                unclustered_candidates=[],
            )

        # Filter out candidates with None navmesh_position
        valid_candidates = [
            c for c in candidates if c.navmesh_position is not None
        ]

        if not valid_candidates:
            return ClusteredCandidates(
                clusters=[],
                candidates_by_cluster={},
                unclustered_candidates=[],
            )

        # Group candidates by region_id
        by_region: Dict[int, List] = {}
        for candidate in valid_candidates:
            region_id = candidate.region_id
            if region_id not in by_region:
                by_region[region_id] = []
            by_region[region_id].append(candidate)

        # Build SpatialCluster objects for regions with enough objects
        spatial_clusters = []
        unclustered = []

        for region_id, region_candidates in by_region.items():
            if len(region_candidates) < self._min_samples:
                # Too few objects - treat as unclustered
                unclustered.extend(region_candidates)
                continue

            # Compute cluster centroid (mean of object positions)
            cluster_positions = np.array(
                [c.navmesh_position for c in region_candidates]
            )
            centroid = np.mean(cluster_positions, axis=0)

            # Distance from agent to centroid
            distance = float(np.linalg.norm(centroid - agent_position))

            spatial_cluster = SpatialCluster(
                cluster_id=region_id,
                cluster_label="",  # Assigned after sorting
                centroid=centroid,
                distance_from_agent=distance,
                object_ids=[c.object_id for c in region_candidates],
                inferred_room_type=None,
            )
            spatial_clusters.append(spatial_cluster)

        # Sort clusters by distance
        spatial_clusters.sort(key=lambda c: c.distance_from_agent)

        # Re-label clusters as A, B, C, ... after sorting by distance
        # For >26 clusters, use numeric labels
        for i, cluster in enumerate(spatial_clusters):
            if i < 26:
                cluster.cluster_label = f"Region {chr(65 + i)}"  # 65 = 'A'
            else:
                cluster.cluster_label = f"Region {i + 1}"

        # Build candidates_by_cluster dict keyed by cluster_id (region_id)
        candidates_by_cluster = {}
        for cluster in spatial_clusters:
            candidates_by_cluster[cluster.cluster_id] = by_region[cluster.cluster_id]

        return ClusteredCandidates(
            clusters=spatial_clusters,
            candidates_by_cluster=candidates_by_cluster,
            unclustered_candidates=unclustered,
        )

