"""Observation encoding for browser delivery.

Stateless functions that convert raw numpy observations to browser-friendly
byte buffers (JPEG for RGB, PNG for depth colormap and top-down view).
Uses OpenCV (cv2) for image encoding.
"""

import math
from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray

from src.perception.occupancy_grid import OccupancyGridData
from src.sensors.lidar import PointCloud
from src.utils.transforms import yaw_from_quaternion


def encode_rgb_jpeg(rgba: NDArray[np.uint8], quality: int = 80) -> bytes:
    """Encode an RGBA image as JPEG bytes (drops alpha channel).

    Args:
        rgba: [H, W, 4] uint8 RGBA image.
        quality: JPEG quality 0-100.

    Returns:
        JPEG-encoded bytes.
    """
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def colorize_depth(
    depth: NDArray[np.float32],
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> bytes:
    """Colorize a depth map using TURBO colormap and encode as JPEG.

    Args:
        depth: [H, W] float32 depth in meters.
        min_depth: Minimum depth for normalization.
        max_depth: Maximum depth for normalization.

    Returns:
        JPEG-encoded bytes of the colorized depth.
    """
    # Clamp and normalize to 0-255
    clamped = np.clip(depth, min_depth, max_depth)
    if max_depth > min_depth:
        normalized = ((clamped - min_depth) / (max_depth - min_depth) * 255).astype(
            np.uint8
        )
    else:
        normalized = np.zeros_like(depth, dtype=np.uint8)

    # Apply TURBO colormap
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def render_topdown_view(
    navmesh_grid: NDArray[np.bool_],
    agent_position: NDArray[np.float64],
    agent_rotation: NDArray[np.float64],
    navmesh_bounds: tuple,
    meters_per_pixel: float = 0.05,
    canvas_size: int = 400,
) -> bytes:
    """Render a top-down NavMesh view with agent position and heading.

    Args:
        navmesh_grid: 2D bool array (True = navigable).
        agent_position: [3] world-frame position.
        agent_rotation: [4] quaternion [w, x, y, z].
        navmesh_bounds: (lower_bound, upper_bound), each [3] arrays.
        meters_per_pixel: Resolution of the navmesh grid.
        canvas_size: Output image side length in pixels.

    Returns:
        PNG-encoded bytes.
    """
    lower, upper = navmesh_bounds

    # Create grayscale image from navmesh grid
    grid_img = np.where(navmesh_grid, 220, 40).astype(np.uint8)

    # Resize to canvas
    canvas = cv2.resize(
        grid_img, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST
    )
    # Convert to BGR for colored overlays
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Guard against zero-range bounds
    x_range = max(upper[0] - lower[0], 1e-6)
    z_range = max(upper[2] - lower[2], 1e-6)

    # Project agent world XZ to pixel coords
    agent_px_x = int((agent_position[0] - lower[0]) / x_range * canvas_size)
    agent_px_y = int((agent_position[2] - lower[2]) / z_range * canvas_size)

    # Clamp to canvas bounds
    agent_px_x = max(0, min(canvas_size - 1, agent_px_x))
    agent_px_y = max(0, min(canvas_size - 1, agent_px_y))

    # Draw agent as red circle
    cv2.circle(canvas, (agent_px_x, agent_px_y), 6, (0, 0, 255), -1)

    yaw = yaw_from_quaternion(agent_rotation)

    # Heading arrow: agent forward is -Z, so in pixel coords:
    # pixel_x maps to world X, pixel_y maps to world Z
    # heading direction in world XZ: (-sin(yaw), -cos(yaw))
    arrow_len = 20
    dx = -math.sin(yaw) * arrow_len * (canvas_size / x_range) * meters_per_pixel
    dy = -math.cos(yaw) * arrow_len * (canvas_size / z_range) * meters_per_pixel
    end_x = int(agent_px_x + dx)
    end_y = int(agent_px_y + dy)

    # Draw heading arrow in green
    cv2.arrowedLine(
        canvas,
        (agent_px_x, agent_px_y),
        (end_x, end_y),
        (0, 255, 0),
        2,
        tipLength=0.3,
    )

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# M2 renderers
# ---------------------------------------------------------------------------

def render_point_cloud_bev(
    point_cloud: PointCloud,
    agent_position: NDArray[np.float64],
    view_range: float = 10.0,
    canvas_size: int = 400,
) -> bytes:
    """Render point cloud as bird's-eye-view image (PNG).

    Projects 3D points onto XZ plane. Agent at center.
    Color encodes height (Y value) using a colormap.

    Args:
        point_cloud: World-frame point cloud.
        agent_position: [3] world-frame agent position.
        view_range: Total side length of the view in meters.
        canvas_size: Output image side length in pixels.

    Returns:
        PNG-encoded bytes.
    """
    canvas = np.full((canvas_size, canvas_size, 3), 40, dtype=np.uint8)

    # Draw crosshair at center (agent position)
    center = canvas_size // 2
    cv2.drawMarker(canvas, (center, center), (0, 0, 255),
                   cv2.MARKER_CROSS, 10, 1)

    if point_cloud.num_valid == 0:
        _, buf = cv2.imencode(".png", canvas)
        return buf.tobytes()

    pts = point_cloud.points
    half_range = view_range / 2.0

    # Project: pixel_x from world X, pixel_y from world Z
    px_x = ((pts[:, 0] - agent_position[0]) / half_range * (canvas_size / 2.0) + center).astype(np.int32)
    px_y = ((pts[:, 2] - agent_position[2]) / half_range * (canvas_size / 2.0) + center).astype(np.int32)

    # Filter to canvas bounds
    valid = (px_x >= 0) & (px_x < canvas_size) & (px_y >= 0) & (px_y < canvas_size)
    px_x = px_x[valid]
    px_y = px_y[valid]
    heights = pts[valid, 1]

    if len(heights) == 0:
        _, buf = cv2.imencode(".png", canvas)
        return buf.tobytes()

    # Normalize height to 0-255 for colormap
    h_min, h_max = float(heights.min()), float(heights.max())
    if h_max - h_min < 1e-6:
        h_normalized = np.full(len(heights), 128, dtype=np.uint8)
    else:
        h_normalized = ((heights - h_min) / (h_max - h_min) * 255).astype(np.uint8)

    # Apply colormap to get colors
    color_strip = cv2.applyColorMap(h_normalized.reshape(-1, 1), cv2.COLORMAP_TURBO)
    colors = color_strip.reshape(-1, 3)

    # Draw points
    for i in range(len(px_x)):
        canvas[px_y[i], px_x[i]] = colors[i]

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def render_occupancy_grid(
    grid_data: OccupancyGridData,
    agent_position: NDArray[np.float64],
    agent_rotation: NDArray[np.float64],
    canvas_size: int = 400,
) -> bytes:
    """Render occupancy grid as colored image (PNG).

    Color scheme: green=free, red=occupied, gray=unknown.
    Agent position + heading arrow overlaid.

    Args:
        grid_data: OccupancyGridData from the occupancy grid module.
        agent_position: [3] world-frame position.
        agent_rotation: [4] quaternion [w, x, y, z].
        canvas_size: Output image side length in pixels.

    Returns:
        PNG-encoded bytes.
    """
    grid = grid_data.grid
    gh, gw = grid.shape

    # Build RGB canvas: green=free, red=occupied, gray=unknown
    canvas = np.zeros((gh, gw, 3), dtype=np.uint8)

    # Unknown (0.5): gray
    unknown_mask = np.abs(grid - 0.5) < 0.01
    canvas[unknown_mask] = [128, 128, 128]

    # Free (< 0.5): green
    free_mask = grid < 0.5
    # Intensity varies with confidence (lower value = more confident free)
    free_intensity = ((0.5 - grid[free_mask]) / 0.5 * 200 + 55).astype(np.uint8)
    canvas[free_mask, 0] = 0  # B
    canvas[free_mask, 1] = free_intensity  # G
    canvas[free_mask, 2] = 0  # R

    # Occupied (> 0.5): red
    occ_mask = grid > 0.5
    occ_intensity = ((grid[occ_mask] - 0.5) / 0.5 * 200 + 55).astype(np.uint8)
    canvas[occ_mask, 0] = 0  # B
    canvas[occ_mask, 1] = 0  # G
    canvas[occ_mask, 2] = occ_intensity  # R

    # Resize to canvas_size
    canvas = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)

    # Draw agent position
    res = grid_data.resolution
    agent_col = (agent_position[0] - grid_data.origin[0]) / res
    agent_row = (agent_position[2] - grid_data.origin[1]) / res
    scale_x = canvas_size / gw
    scale_y = canvas_size / gh
    agent_px_x = int(agent_col * scale_x)
    agent_px_y = int(agent_row * scale_y)
    agent_px_x = max(0, min(canvas_size - 1, agent_px_x))
    agent_px_y = max(0, min(canvas_size - 1, agent_px_y))

    cv2.circle(canvas, (agent_px_x, agent_px_y), 5, (255, 255, 0), -1)

    yaw = yaw_from_quaternion(agent_rotation)
    arrow_len = 15
    dx = -math.sin(yaw) * arrow_len
    dy = -math.cos(yaw) * arrow_len
    cv2.arrowedLine(
        canvas,
        (agent_px_x, agent_px_y),
        (int(agent_px_x + dx), int(agent_px_y + dy)),
        (255, 255, 0), 2, tipLength=0.3,
    )

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def render_semantic_overlay(
    rgb: NDArray[np.uint8],
    semantic: NDArray[np.uint32],
    obstacle_mask: NDArray[np.bool_],
    alpha: float = 0.4,
) -> bytes:
    """Overlay obstacle detections on RGB image as semi-transparent red (JPEG).

    Args:
        rgb: [H, W, 4] RGBA image.
        semantic: [H, W] semantic IDs (for reference, not directly drawn).
        obstacle_mask: [H, W] True where obstacle detected.
        alpha: Overlay transparency.

    Returns:
        JPEG-encoded bytes.
    """
    # Convert RGBA to BGR
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    # Draw red overlay on obstacle pixels
    overlay[obstacle_mask] = [0, 0, 255]

    # Blend
    result = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)

    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def render_vo_trajectory(
    positions: List[NDArray[np.float64]],
    current_position: NDArray[np.float64],
    navmesh_bounds: tuple,
    canvas_size: int = 400,
) -> bytes:
    """Render VO-estimated trajectory on top-down canvas (PNG).

    Shows breadcrumb trail of estimated positions.

    Args:
        positions: List of [3] world positions (trajectory history).
        current_position: [3] current agent world position.
        navmesh_bounds: (lower_bound, upper_bound), each [3] arrays.
        canvas_size: Output image side length in pixels.

    Returns:
        PNG-encoded bytes.
    """
    canvas = np.full((canvas_size, canvas_size, 3), 40, dtype=np.uint8)
    lower, upper = navmesh_bounds

    x_range = max(upper[0] - lower[0], 1e-6)
    z_range = max(upper[2] - lower[2], 1e-6)

    def world_to_px(pos: NDArray) -> tuple:
        px_x = int((pos[0] - lower[0]) / x_range * canvas_size)
        px_y = int((pos[2] - lower[2]) / z_range * canvas_size)
        px_x = max(0, min(canvas_size - 1, px_x))
        px_y = max(0, min(canvas_size - 1, px_y))
        return px_x, px_y

    # Draw trail
    if len(positions) >= 2:
        for i in range(1, len(positions)):
            pt1 = world_to_px(positions[i - 1])
            pt2 = world_to_px(positions[i])
            cv2.line(canvas, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

    # Draw breadcrumb dots
    for pos in positions:
        px = world_to_px(pos)
        cv2.circle(canvas, px, 2, (0, 200, 200), -1)

    # Draw current position as bright circle
    curr_px = world_to_px(current_position)
    cv2.circle(canvas, curr_px, 5, (0, 255, 0), -1)

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()
