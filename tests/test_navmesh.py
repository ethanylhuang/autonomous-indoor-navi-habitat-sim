"""Tests for NavMesh pathfinding and top-down rasterization.

Covers AC5 (NavMesh pathfinding).
"""

import numpy as np
import pytest

from src.vehicle import Vehicle


@pytest.fixture(scope="module")
def vehicle():
    """Create a single Vehicle instance for all navmesh tests."""
    v = Vehicle()
    yield v
    v.close()


class TestPathfinder:
    def test_pathfinder_is_loaded(self, vehicle):
        assert vehicle.pathfinder.is_loaded

    def test_random_navigable_point_shape(self, vehicle):
        point = vehicle.pathfinder.get_random_navigable_point()
        arr = np.array(point)
        assert arr.shape == (3,)

    def test_find_path_returns_waypoints(self, vehicle):
        start = vehicle.pathfinder.get_random_navigable_point()
        goal = vehicle.pathfinder.get_random_navigable_point()
        waypoints, distance = vehicle.find_path(start, goal)
        assert len(waypoints) > 0
        assert np.isfinite(distance)

    def test_find_path_positive_distance(self, vehicle):
        """Path between two distinct navigable points has positive distance."""
        start = vehicle.pathfinder.get_random_navigable_point()
        goal = vehicle.pathfinder.get_random_navigable_point()
        # Keep trying until we get distinct points
        attempts = 0
        while np.allclose(start, goal) and attempts < 10:
            goal = vehicle.pathfinder.get_random_navigable_point()
            attempts += 1
        _, distance = vehicle.find_path(start, goal)
        assert distance > 0


class TestTopdownNavmesh:
    def test_topdown_is_2d_bool(self, vehicle):
        grid = vehicle.get_topdown_navmesh()
        assert grid.ndim == 2
        assert grid.dtype == np.bool_

    def test_topdown_has_both_values(self, vehicle):
        """NavMesh grid should have both navigable (True) and non-navigable (False)."""
        grid = vehicle.get_topdown_navmesh()
        assert np.any(grid)
        assert np.any(~grid)

    def test_bounds_valid(self, vehicle):
        """Lower bounds should be strictly less than upper bounds on X and Z."""
        lower, upper = vehicle.get_navmesh_bounds()
        assert lower[0] < upper[0], "X lower >= upper"
        assert lower[2] < upper[2], "Z lower >= upper"

    def test_bounds_shape(self, vehicle):
        lower, upper = vehicle.get_navmesh_bounds()
        assert np.array(lower).shape == (3,)
        assert np.array(upper).shape == (3,)
