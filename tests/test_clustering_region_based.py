"""Shield verification tests for region_id-based clustering.

Tests edge cases and failure modes for the updated clustering.py implementation
that uses HM3D region_id instead of DBSCAN.
"""

import numpy as np
import pytest

from src.vlm.clustering import SpatialClusterer, ClusteredCandidates, SpatialCluster


class MockCandidate:
    """Mock ObjectCandidate for testing."""

    def __init__(self, object_id, label, navmesh_position, region_id):
        self.object_id = object_id
        self.label = label
        self.navmesh_position = np.array(navmesh_position, dtype=np.float64)
        self.distance_from_agent = 0.0
        self.region_id = region_id


def test_region_id_grouping():
    """Test that objects are grouped by region_id, not spatial proximity."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Two objects from region 1, very far apart (would not cluster with DBSCAN)
    # Two objects from region 2, also far apart
    candidates = [
        MockCandidate(1, "sofa", [1.0, 0.0, 1.0], region_id=1),
        MockCandidate(2, "tv", [100.0, 0.0, 100.0], region_id=1),  # Far from sofa
        MockCandidate(3, "bed", [2.0, 0.0, 2.0], region_id=2),
        MockCandidate(4, "nightstand", [101.0, 0.0, 101.0], region_id=2),  # Far from bed
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should have 2 clusters (one per region_id), not 4 separate objects
    assert len(clustered.clusters) == 2
    assert len(clustered.unclustered_candidates) == 0

    # Check region_ids are preserved
    region_ids = {cluster.cluster_id for cluster in clustered.clusters}
    assert region_ids == {1, 2}


def test_all_objects_region_zero():
    """Test edge case where all objects have region_id=0."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # All objects have region_id=0 (common fallback/default value)
    candidates = [
        MockCandidate(1, "chair", [1.0, 0.0, 1.0], region_id=0),
        MockCandidate(2, "table", [2.0, 0.0, 2.0], region_id=0),
        MockCandidate(3, "lamp", [3.0, 0.0, 3.0], region_id=0),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should form one cluster with all objects
    assert len(clustered.clusters) == 1
    assert clustered.clusters[0].cluster_id == 0
    assert len(clustered.clusters[0].object_ids) == 3
    assert len(clustered.unclustered_candidates) == 0


def test_region_with_single_object():
    """Test that regions with only 1 object become unclustered."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    candidates = [
        MockCandidate(1, "sofa", [1.0, 0.0, 1.0], region_id=1),
        MockCandidate(2, "tv", [1.5, 0.0, 1.5], region_id=1),  # region 1 has 2 objects
        MockCandidate(3, "bed", [10.0, 0.0, 10.0], region_id=2),  # region 2 has only 1
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should have 1 cluster (region 1) and 1 unclustered (region 2)
    assert len(clustered.clusters) == 1
    assert clustered.clusters[0].cluster_id == 1
    assert len(clustered.unclustered_candidates) == 1
    assert clustered.unclustered_candidates[0].object_id == 3


def test_min_samples_threshold():
    """Test that min_samples parameter is respected."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=3)  # Require 3 objects
    agent_position = np.array([0.0, 0.0, 0.0])

    candidates = [
        # Region 1: 3 objects (should cluster)
        MockCandidate(1, "sofa", [1.0, 0.0, 1.0], region_id=1),
        MockCandidate(2, "tv", [1.5, 0.0, 1.5], region_id=1),
        MockCandidate(3, "table", [1.2, 0.0, 1.2], region_id=1),
        # Region 2: 2 objects (should not cluster)
        MockCandidate(4, "bed", [10.0, 0.0, 10.0], region_id=2),
        MockCandidate(5, "nightstand", [10.5, 0.0, 10.5], region_id=2),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should have 1 cluster (region 1 with 3 objects) and 2 unclustered (region 2 objects)
    assert len(clustered.clusters) == 1
    assert clustered.clusters[0].cluster_id == 1
    assert len(clustered.unclustered_candidates) == 2


def test_cluster_label_overflow():
    """Test cluster labeling when there are > 26 clusters (beyond 'Z')."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Create 30 regions with 2 objects each
    candidates = []
    for region_id in range(30):
        candidates.append(MockCandidate(
            region_id * 2,
            "object",
            [float(region_id * 5), 0.0, float(region_id * 5)],
            region_id=region_id
        ))
        candidates.append(MockCandidate(
            region_id * 2 + 1,
            "object",
            [float(region_id * 5 + 1), 0.0, float(region_id * 5 + 1)],
            region_id=region_id
        ))

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should have 30 clusters
    assert len(clustered.clusters) == 30

    # Check first few labels
    assert clustered.clusters[0].cluster_label == "Region A"
    assert clustered.clusters[25].cluster_label == "Region Z"

    # Labels beyond Z will have invalid characters (chr(91) = '[', chr(92) = '\\', etc.)
    # This is a potential bug - labels should handle > 26 clusters gracefully
    # For now, just verify it doesn't crash
    assert clustered.clusters[26].cluster_label.startswith("Region")


def test_centroid_calculation():
    """Test that cluster centroid is computed correctly as mean of object positions."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    candidates = [
        MockCandidate(1, "sofa", [2.0, 0.0, 2.0], region_id=1),
        MockCandidate(2, "tv", [4.0, 0.0, 4.0], region_id=1),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Centroid should be midpoint: (3.0, 0.0, 3.0)
    expected_centroid = np.array([3.0, 0.0, 3.0])
    np.testing.assert_array_almost_equal(clustered.clusters[0].centroid, expected_centroid)

    # Distance should be ||[3.0, 0.0, 3.0]|| = sqrt(18) ≈ 4.24
    expected_distance = np.linalg.norm(expected_centroid)
    assert abs(clustered.clusters[0].distance_from_agent - expected_distance) < 0.01


def test_candidates_by_cluster_keying():
    """Test that candidates_by_cluster dict is keyed by region_id (cluster_id)."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    candidates = [
        MockCandidate(1, "sofa", [1.0, 0.0, 1.0], region_id=5),
        MockCandidate(2, "tv", [1.5, 0.0, 1.5], region_id=5),
        MockCandidate(3, "bed", [10.0, 0.0, 10.0], region_id=12),
        MockCandidate(4, "nightstand", [10.5, 0.0, 10.5], region_id=12),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)

    # Should be keyed by actual region_id, not sequential index
    assert 5 in clustered.candidates_by_cluster
    assert 12 in clustered.candidates_by_cluster
    assert len(clustered.candidates_by_cluster[5]) == 2
    assert len(clustered.candidates_by_cluster[12]) == 2


def test_missing_region_id_attribute():
    """Test that missing region_id attribute causes AttributeError."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Create candidate without region_id
    class BadCandidate:
        def __init__(self):
            self.object_id = 1
            self.label = "sofa"
            self.navmesh_position = np.array([1.0, 0.0, 1.0])
            # Missing region_id!

    candidates = [BadCandidate()]

    with pytest.raises(AttributeError):
        clustered = clusterer.cluster_candidates(candidates, agent_position)


def test_empty_navmesh_positions():
    """Test handling of None or invalid navmesh_position values."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Create candidates with None navmesh_position
    class CandidateWithNonePosition:
        def __init__(self, object_id, region_id):
            self.object_id = object_id
            self.label = "object"
            self.navmesh_position = None
            self.region_id = region_id

    candidates = [
        CandidateWithNonePosition(1, region_id=1),
        CandidateWithNonePosition(2, region_id=1),
    ]

    # This should raise an error when trying to compute centroid
    with pytest.raises((TypeError, AttributeError)):
        clustered = clusterer.cluster_candidates(candidates, agent_position)
