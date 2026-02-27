"""Tests for clustered VLM prompts."""

import numpy as np

from src.vlm.clustering import SpatialClusterer, ClusteredCandidates
from src.vlm.prompts import build_clustered_selection_prompt


class MockCandidate:
    """Mock ObjectCandidate for testing."""

    def __init__(self, object_id, label, navmesh_position):
        self.object_id = object_id
        self.label = label
        self.navmesh_position = np.array(navmesh_position, dtype=np.float64)
        self.distance_from_agent = 0.0


def test_clustered_prompt_format():
    """Test clustered prompt shows regions with distances and room labels."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    # Create kitchen cluster and living room cluster
    candidates = [
        MockCandidate(1, "refrigerator", [2.0, 0.0, 2.0]),
        MockCandidate(2, "stove", [2.5, 0.0, 2.2]),
        MockCandidate(3, "sofa", [8.0, 0.0, 8.0]),
        MockCandidate(4, "tv", [8.3, 0.0, 8.5]),
        MockCandidate(5, "plant", [15.0, 0.0, 15.0]),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)
    prompt = build_clustered_selection_prompt("find the kitchen", clustered)

    # Check key format elements
    assert "Instruction: find the kitchen" in prompt
    assert "Objects grouped by spatial proximity:" in prompt
    assert "Region A" in prompt
    assert "Region B" in prompt
    assert "[kitchen]" in prompt or "[living room]" in prompt
    assert "m away" in prompt
    assert "Unclustered:" in prompt
    assert "plant" in prompt

    # Verify NO "likely" prefix
    assert "likely" not in prompt.lower()

    # Check distances are shown
    assert "2." in prompt or "3." in prompt  # distances in meters


def test_clustered_prompt_objects_sorted_within_region():
    """Test objects within each region are sorted by distance."""
    clusterer = SpatialClusterer(eps=3.0, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    candidates = [
        MockCandidate(1, "table", [5.0, 0.0, 5.0]),   # ~7.1m
        MockCandidate(2, "chair", [3.0, 0.0, 3.0]),   # ~4.2m
        MockCandidate(3, "lamp", [4.0, 0.0, 4.0]),    # ~5.7m
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)
    prompt = build_clustered_selection_prompt("find seating", clustered)

    # Find the region section in prompt
    lines = prompt.split("\n")
    region_start = None
    for i, line in enumerate(lines):
        if "Region" in line:
            region_start = i
            break

    assert region_start is not None

    # Extract object lines (indented with "  - ")
    object_lines = []
    for i in range(region_start + 1, len(lines)):
        if lines[i].startswith("  - "):
            object_lines.append(lines[i])
        elif not lines[i].strip() or lines[i].startswith("Region"):
            break

    # Objects should be sorted by distance
    # chair (4.2m) < lamp (5.7m) < table (7.1m)
    assert len(object_lines) == 3
    assert "chair" in object_lines[0]
    assert "lamp" in object_lines[1]
    assert "table" in object_lines[2]


def test_clustered_prompt_no_likely_prefix():
    """Test room labels do NOT have 'likely' prefix."""
    clusterer = SpatialClusterer(eps=2.5, min_samples=2)
    agent_position = np.array([0.0, 0.0, 0.0])

    candidates = [
        MockCandidate(1, "bed", [3.0, 0.0, 3.0]),
        MockCandidate(2, "nightstand", [3.3, 0.0, 3.2]),
    ]

    for c in candidates:
        c.distance_from_agent = float(np.linalg.norm(c.navmesh_position - agent_position))

    clustered = clusterer.cluster_candidates(candidates, agent_position)
    prompt = build_clustered_selection_prompt("go to bedroom", clustered)

    # Should show [bedroom], NOT [likely bedroom]
    assert "[bedroom]" in prompt
    assert "[likely" not in prompt
