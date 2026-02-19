"""Tests for the occupancy grid.

Covers AC5: grid shape, values, projection, bounds, height filtering,
agent-centering, free space, unknown cells.
All tests are pure numpy -- no habitat-sim dependency.
"""

import numpy as np
import pytest

from src.perception.occupancy_grid import OccupancyGrid, OccupancyGridData
from src.sensors.lidar import PointCloud


class TestOccupancyGridShape:
    def test_default_shape_200x200(self):
        """Default parameters produce a 200x200 grid."""
        grid = OccupancyGrid()
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = grid.update(agent_pos, agent_rot, [])
        assert result.grid.shape == (200, 200)
        assert result.shape == (200, 200)
        assert result.resolution == 0.05

    def test_custom_shape(self):
        """Custom grid_size and resolution produce correct shape."""
        grid = OccupancyGrid(grid_size=5.0, resolution=0.1)
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = grid.update(agent_pos, agent_rot, [])
        assert result.grid.shape == (50, 50)


class TestOccupancyGridValues:
    def test_values_in_valid_range(self):
        """All grid values are in [0.0, 1.0]."""
        grid = OccupancyGrid()
        agent_pos = np.array([5.0, 1.0, 5.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Single point
        pc = PointCloud(
            points=np.array([[6.0, 1.0, 6.0]], dtype=np.float32),
            num_valid=1,
        )
        result = grid.update(agent_pos, agent_rot, [pc])
        assert np.all(result.grid >= 0.0)
        assert np.all(result.grid <= 1.0)

    def test_unknown_cells_at_half(self):
        """Unobserved cells have value 0.5."""
        grid = OccupancyGrid()
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = grid.update(agent_pos, agent_rot, [])
        # With no points, all cells should be 0.5
        assert np.all(result.grid == 0.5)


class TestOccupancyGridProjection:
    def test_single_point_marks_correct_cell(self):
        """A point at known XZ marks the correct grid cell as occupied."""
        grid = OccupancyGrid(grid_size=10.0, resolution=0.05)
        agent_pos = np.array([5.0, 1.0, 5.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Point at world (6.0, 1.0, 6.0) = 1m right and 1m forward of agent
        # Grid origin = (5.0 - 5.0, 5.0 - 5.0) = (0.0, 0.0)
        # col = (6.0 - 0.0) / 0.05 = 120
        # row = (6.0 - 0.0) / 0.05 = 120
        pc = PointCloud(
            points=np.array([[6.0, 1.0, 6.0]], dtype=np.float32),
            num_valid=1,
        )
        result = grid.update(agent_pos, agent_rot, [pc])
        assert result.grid[120, 120] > 0.5  # occupied

    def test_points_outside_grid_no_error(self):
        """Points outside the grid area do not cause IndexError."""
        grid = OccupancyGrid(grid_size=2.0, resolution=0.05)
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Point far outside the grid
        pc = PointCloud(
            points=np.array([[100.0, 1.0, 100.0]], dtype=np.float32),
            num_valid=1,
        )
        # Should not raise
        result = grid.update(agent_pos, agent_rot, [pc])
        assert result.grid.shape == (40, 40)


class TestOccupancyGridHeightFilter:
    def test_points_outside_height_range_filtered(self):
        """Points below height_min or above height_max are ignored."""
        grid = OccupancyGrid(
            grid_size=10.0, resolution=0.05,
            height_min=0.1, height_max=2.0,
        )
        agent_pos = np.array([5.0, 1.0, 5.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Point below floor (y=0.0)
        pc_low = PointCloud(
            points=np.array([[6.0, 0.0, 6.0]], dtype=np.float32),
            num_valid=1,
        )
        result = grid.update(agent_pos, agent_rot, [pc_low])
        # Cell at (120, 120) should remain unknown
        assert result.grid[120, 120] == 0.5

        # Point above ceiling (y=3.0)
        pc_high = PointCloud(
            points=np.array([[6.0, 3.0, 6.0]], dtype=np.float32),
            num_valid=1,
        )
        result = grid.update(agent_pos, agent_rot, [pc_high])
        assert result.grid[120, 120] == 0.5


class TestOccupancyGridAgentCentered:
    def test_origin_shifts_with_agent(self):
        """Grid origin is centered on agent position."""
        grid = OccupancyGrid(grid_size=10.0, resolution=0.05)

        pos_a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result_a = grid.update(pos_a, rot, [])
        np.testing.assert_allclose(result_a.origin, [-5.0, -5.0])

        pos_b = np.array([10.0, 0.0, 20.0], dtype=np.float64)
        result_b = grid.update(pos_b, rot, [])
        np.testing.assert_allclose(result_b.origin, [5.0, 15.0])


class TestOccupancyGridFreeSpace:
    def test_free_space_between_agent_and_obstacle(self):
        """Cells between agent and occupied cell should be marked free."""
        grid = OccupancyGrid(
            grid_size=10.0, resolution=0.05,
            height_min=0.0, height_max=3.0,
        )
        agent_pos = np.array([5.0, 1.0, 5.0], dtype=np.float64)
        agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Place obstacle 2m ahead in X: world (7.0, 1.0, 5.0)
        pc = PointCloud(
            points=np.array([[7.0, 1.0, 5.0]], dtype=np.float32),
            num_valid=1,
        )
        result = grid.update(agent_pos, agent_rot, [pc])

        # Agent is at grid center (100, 100).
        # Obstacle at col = (7.0 - 0.0) / 0.05 = 140, row = (5.0 - 0.0) / 0.05 = 100
        # Cells between (100, 100) and (100, 140) along row 100 should be free
        # Check a cell midway: (100, 120)
        assert result.grid[100, 120] < 0.5  # free
        # The obstacle cell itself
        assert result.grid[100, 140] > 0.5  # occupied
