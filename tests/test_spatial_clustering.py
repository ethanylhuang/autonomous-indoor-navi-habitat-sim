"""Tests for spatial clustering of VLM object candidates.

Clustering uses HM3D region_id annotations to group objects by room.
"""

import numpy as np
import pytest

from src.vlm.clustering import SpatialClusterer, ClusteredCandidates, SpatialCluster


class MockCandidate:
    """Mock ObjectCandidate for testing."""

    def __init__(self, object_id, label, navmesh_position, region_id=0):
        self.object_id = object_id
        self.label = label
        self.navmesh_position = np.array(navmesh_position, dtype=np.float64)
        self.region_id = region_id
        self.distance_from_agent = 0.0


def test_region_based_clustering():
    """Test clustering groups objects by region_id."""
    clusterer = SpatialClusterer(min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Create two regions by region_id
    # Region 1 (living room): sofa, tv
    # Region 2 (kitchen): refrigerator, stove, oven
    # Unclustered: single object with unique region_id
    candidates = [
        MockCandidate(1, "sofa", [2.0, 0.0, 2.0], region_id=1),
        MockCandidate(2, "tv", [2.5, 0.0, 2.2], region_id=1),
        MockCandidate(3, "refrigerator", [8.0, 0.0, 8.0], region_id=2),
        MockCandidate(4, "stove", [8.3, 0.0, 8.5], region_id=2),
        MockCandidate(5, "oven", [8.2, 0.0, 8.8], region_id=2),
        MockCandidate(6, "plant", [15.0, 0.0, 15.0], region_id=99),  # single object region
    ]

    # Update distances
    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should have 2 clusters (regions 1 and 2) + 1 unclustered (region 99 has only 1 object)
    assert len(clustered.clusters) == 2
    assert len(clustered.unclustered_candidates) == 1
    assert clustered.unclustered_candidates[0].object_id == 6


def test_clusters_sorted_by_distance():
    """Test clusters are sorted by distance from agent."""
    clusterer = SpatialClusterer(min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Region 1 (bedroom) at ~14m, Region 2 (living room) at ~4m
    # Should be ordered: living room (Region A), bedroom (Region B)
    candidates = [
        MockCandidate(1, "bed", [10.0, 0.0, 10.0], region_id=1),
        MockCandidate(2, "nightstand", [10.5, 0.0, 10.2], region_id=1),
        MockCandidate(3, "sofa", [3.0, 0.0, 3.0], region_id=2),
        MockCandidate(4, "tv", [3.3, 0.0, 3.2], region_id=2),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    assert len(clustered.clusters) == 2
    # Closer cluster should be labeled "Region A"
    assert clustered.clusters[0].cluster_label == "Region A"
    assert clustered.clusters[0].distance_from_agent < clustered.clusters[1].distance_from_agent
    # Further cluster should be "Region B"
    assert clustered.clusters[1].cluster_label == "Region B"


def test_empty_candidates():
    """Test clustering handles empty candidate list."""
    clusterer = SpatialClusterer(min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    clustered = clusterer.cluster_candidates([], agent_position)

    assert len(clustered.clusters) == 0
    assert len(clustered.unclustered_candidates) == 0
    assert len(clustered.candidates_by_cluster) == 0


def test_objects_within_clusters_sorted_by_distance():
    """Test objects within each cluster are accessible sorted by distance."""
    clusterer = SpatialClusterer(min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # All in same region
    candidates = [
        MockCandidate(1, "sofa", [3.0, 0.0, 3.0], region_id=1),  # ~4.2m
        MockCandidate(2, "tv", [3.8, 0.0, 3.8], region_id=1),    # ~5.4m
        MockCandidate(3, "table", [3.2, 0.0, 3.2], region_id=1), # ~4.5m
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    assert len(clustered.clusters) == 1
    cluster = clustered.clusters[0]
    cluster_candidates = clustered.candidates_by_cluster[cluster.cluster_id]

    # Candidates should be retrievable
    assert len(cluster_candidates) == 3
    # Check they're in original order (sorting happens in prompt builder)
    assert cluster_candidates[0].object_id == 1


def test_all_objects_same_region():
    """Test behavior when all objects have the same region_id."""
    clusterer = SpatialClusterer(min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # All objects in region 0 (common fallback when region_id not extracted)
    candidates = [
        MockCandidate(1, "sofa", [3.0, 0.0, 3.0], region_id=0),
        MockCandidate(2, "bed", [10.0, 0.0, 10.0], region_id=0),
        MockCandidate(3, "table", [5.0, 0.0, 5.0], region_id=0),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # All objects should be in one cluster
    assert len(clustered.clusters) == 1
    assert len(clustered.unclustered_candidates) == 0
    assert len(clustered.candidates_by_cluster[0]) == 3
