"""Semantic scene parsing and object indexing for HM3D scenes.

Parses .semantic.txt annotation files and computes object centroids for
per-instance navigation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from configs.sensor_rig import HFOV


@dataclass
class SemanticObject:
    """Single semantic object instance in the scene."""

    object_id: int  # Unique ID from semantic sensor
    label: str  # Category label (e.g., "bed", "chair")
    hex_color: str  # Color code from .semantic.txt
    region_id: int  # Spatial region grouping
    centroid: Optional[NDArray[np.float64]]  # [3] world coords, or None if not computed
    navmesh_position: Optional[NDArray[np.float64]]  # [3] snapped to NavMesh
    instance_number: int  # 1-indexed per label (e.g., 1 for "bed #1")

    def to_dict(self) -> dict:
        """Serialize for JSON API response."""
        return {
            "object_id": self.object_id,
            "label": self.label,
            "region_id": self.region_id,
            "instance_name": self.instance_name,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "navmesh_position": (
                self.navmesh_position.tolist()
                if self.navmesh_position is not None
                else None
            ),
            "is_navigable": self.navmesh_position is not None,
        }

    @property
    def instance_name(self) -> str:
        """Human-readable instance name with location hint."""
        return f"{self.label} #{self.instance_number} (region {self.region_id})"


@dataclass
class SemanticSceneIndex:
    """Index of all semantic objects in a scene."""

    objects: Dict[int, SemanticObject]  # object_id -> SemanticObject
    objects_by_label: Dict[str, List[int]]  # label -> list of object_ids
    region_to_objects: Dict[int, List[int]]  # region_id -> list of object_ids


class SemanticSceneParser:
    """Parser for HM3D .semantic.txt annotation files."""

    # Labels to skip - structural elements and unknown objects
    SKIP_LABELS = {
        "floor",
        "ceiling",
        "wall",
        "door frame",
        "unknown",
        "beam",
        "door",
        "window",
        "stair",
        "stairs",
    }

    @classmethod
    def parse_semantic_txt(cls, filepath: Path) -> Optional[SemanticSceneIndex]:
        """Parse .semantic.txt file into SemanticSceneIndex.

        Args:
            filepath: Path to .semantic.txt file.

        Returns:
            SemanticSceneIndex or None if parsing fails.
        """
        if not filepath.exists():
            return None

        objects: Dict[int, SemanticObject] = {}
        label_counts: Dict[str, int] = {}

        try:
            with open(filepath, "r") as f:
                # Skip header line
                next(f)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse: object_id,hex_color,"label",region_id
                    parts = line.split(",")
                    if len(parts) != 4:
                        continue

                    object_id = int(parts[0])
                    hex_color = parts[1]
                    label = parts[2].strip('"').lower()
                    region_id = int(parts[3])

                    # Skip structural elements
                    if label in cls.SKIP_LABELS:
                        continue

                    # Assign instance number per label
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += 1
                    instance_number = label_counts[label]

                    obj = SemanticObject(
                        object_id=object_id,
                        label=label,
                        hex_color=hex_color,
                        region_id=region_id,
                        centroid=None,
                        navmesh_position=None,
                        instance_number=instance_number,
                    )
                    objects[object_id] = obj

        except Exception as e:
            print(f"Failed to parse semantic file {filepath}: {e}")
            return None

        # Build reverse indices
        objects_by_label: Dict[str, List[int]] = {}
        region_to_objects: Dict[int, List[int]] = {}

        for obj_id, obj in objects.items():
            # By label
            if obj.label not in objects_by_label:
                objects_by_label[obj.label] = []
            objects_by_label[obj.label].append(obj_id)

            # By region
            if obj.region_id not in region_to_objects:
                region_to_objects[obj.region_id] = []
            region_to_objects[obj.region_id].append(obj_id)

        return SemanticSceneIndex(
            objects=objects,
            objects_by_label=objects_by_label,
            region_to_objects=region_to_objects,
        )


def compute_object_centroids(
    semantic_index: SemanticSceneIndex,
    vehicle,
    num_views: int = 8,
    snap_max_distance: float = 2.0,
) -> None:
    """Compute world-space centroids for all objects by multi-view projection.

    Updates objects in semantic_index IN-PLACE.

    Strategy:
    1. Save current agent state
    2. For each of num_views directions (360-degree sweep):
       - Rotate agent to face that direction
       - Capture semantic + depth images
       - For each visible object_id, back-project all pixels to world coords
    3. For each object, compute median centroid from all observations
    4. Snap centroid to navmesh
    5. Restore agent to original state

    Args:
        semantic_index: Index to update with centroids.
        vehicle: Vehicle instance with simulator access.
        num_views: Number of viewing directions (uniformly spaced).
        snap_max_distance: Max distance to snap centroid to navmesh.
    """
    import math
    from src.utils.transforms import quaternion_from_yaw

    # Save original agent state
    orig_obs = vehicle.get_initial_observations()
    orig_pos = orig_obs.state.position.copy()
    orig_rot = orig_obs.state.rotation.copy()

    # Collect all 3D points per object across all views
    object_points: Dict[int, List[NDArray[np.float64]]] = {
        obj_id: [] for obj_id in semantic_index.objects.keys()
    }

    # Camera intrinsics
    hfov_rad = math.radians(float(HFOV))
    H, W = 480, 640
    focal_length = (W / 2.0) / math.tan(hfov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0

    # Sweep 360 degrees
    for i in range(num_views):
        yaw = (2.0 * math.pi * i) / num_views
        vehicle.turn_to_heading(yaw)
        obs = vehicle.get_initial_observations()

        semantic = obs.forward_semantic
        depth = obs.depth
        agent_pos = obs.state.position
        agent_rot = obs.state.rotation

        # Get rotation matrix for camera-to-world transform
        from src.utils.transforms import quat_to_rotation_matrix

        R = quat_to_rotation_matrix(agent_rot)

        # Sensor height offset (camera is mounted above agent origin)
        from configs.sensor_rig import SENSOR_HEIGHT

        sensor_pos = agent_pos.copy()
        sensor_pos[1] += SENSOR_HEIGHT

        # For each object_id in this view, collect 3D points
        unique_ids = np.unique(semantic)
        for obj_id in unique_ids:
            obj_id = int(obj_id)
            if obj_id not in semantic_index.objects:
                continue

            # Get all pixels with this object_id
            mask = semantic == obj_id
            v_coords, u_coords = np.where(mask)

            # Sample at most 500 pixels per object per view (performance)
            if len(v_coords) > 500:
                indices = np.random.choice(len(v_coords), 500, replace=False)
                v_coords = v_coords[indices]
                u_coords = u_coords[indices]

            # Back-project each pixel to 3D
            for u, v in zip(u_coords, v_coords):
                d = depth[v, u]
                if not np.isfinite(d) or d <= 0.3 or d > 10.0:
                    continue

                # Camera frame: X=right, Y=up, Z=backward
                x_cam = (u - cx) * d / focal_length
                y_cam = -(v - cy) * d / focal_length
                z_cam = -d

                cam_point = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
                world_point = R @ cam_point + sensor_pos

                object_points[obj_id].append(world_point)

    # Restore agent to original state
    vehicle.set_agent_state(orig_pos, orig_rot)

    # Compute centroid for each object and snap to navmesh
    from src.vlm.projection import snap_to_navmesh

    for obj_id, points in object_points.items():
        if len(points) == 0:
            continue

        # Use median centroid (more robust to outliers than mean)
        points_array = np.array(points)
        centroid = np.median(points_array, axis=0)

        # Snap to navmesh
        snapped, success, reason = snap_to_navmesh(
            centroid, vehicle.pathfinder, max_distance=snap_max_distance
        )

        obj = semantic_index.objects[obj_id]
        obj.centroid = centroid

        if success and snapped is not None:
            obj.navmesh_position = snapped
        else:
            obj.navmesh_position = None


def get_navigable_objects(semantic_index: SemanticSceneIndex) -> List[SemanticObject]:
    """Get all navigable objects (with valid navmesh positions), sorted by label.

    Args:
        semantic_index: Semantic scene index.

    Returns:
        List of SemanticObject sorted by (label, instance_number).
    """
    navigable = [
        obj
        for obj in semantic_index.objects.values()
        if obj.navmesh_position is not None
    ]
    navigable.sort(key=lambda obj: (obj.label, obj.instance_number))
    return navigable


def build_semantic_index_from_sim(
    vehicle,
    snap_max_distance: float = 2.0,
    num_projection_views: int = 12,
) -> Optional[SemanticSceneIndex]:
    """Build SemanticSceneIndex from habitat-sim semantic scene with multi-view projection.

    Uses the SemanticScene API to get object IDs and labels, then uses multi-view
    projection (rendering semantic + depth from multiple angles) to compute accurate
    world-space centroids. This is necessary because the AABB coordinates from
    SemanticScene are in asset coordinates, not world coordinates.

    Requires the scene to be loaded via scene_dataset_config.json that
    references semantic assets (.semantic.glb + .semantic.txt).

    Args:
        vehicle: Vehicle instance with simulator access.
        snap_max_distance: Max distance to snap centroid to navmesh.
        num_projection_views: Number of views for multi-view projection (default 12).

    Returns:
        SemanticSceneIndex with accurate world-space centroids, or None if
        semantic scene is not available.
    """
    semantic_scene = vehicle.semantic_scene
    if semantic_scene is None:
        return None

    sim_objects = semantic_scene.objects
    if not sim_objects or len(sim_objects) == 0:
        return None

    objects: Dict[int, SemanticObject] = {}
    label_counts: Dict[str, int] = {}

    # First pass: build index with object metadata (no centroids yet)
    for sim_obj in sim_objects:
        try:
            # Get category label
            category = sim_obj.category
            if category is None:
                continue
            label = category.name().lower()

            # Skip structural elements
            if label in SemanticSceneParser.SKIP_LABELS:
                continue

            # Get object ID (semantic sensor pixel value)
            # sim_obj.semantic_id is the unique per-instance ID that maps to
            # semantic sensor pixel values.
            object_id = sim_obj.semantic_id

            # Skip objects without AABB (used for size filtering only)
            aabb = sim_obj.aabb
            if aabb is None:
                continue

            # Get size for filtering large scene-level annotations
            size = aabb.size()
            size_array = np.array([size[0], size[1], size[2]], dtype=np.float64)
            if np.any(size_array > 20.0):
                continue

            # Assign instance number per label
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            instance_number = label_counts[label]

            # Get region_id from sim_obj if available
            region_id = 0
            if hasattr(sim_obj, "region") and sim_obj.region:
                region_id = sim_obj.region.index if hasattr(sim_obj.region, "index") else 0

            obj = SemanticObject(
                object_id=object_id,
                label=label,
                hex_color="000000",
                region_id=region_id,
                centroid=None,  # Will be computed by multi-view projection
                navmesh_position=None,
                instance_number=instance_number,
            )
            objects[object_id] = obj

        except Exception:
            continue

    if not objects:
        return None

    # Build reverse indices
    objects_by_label: Dict[str, List[int]] = {}
    region_to_objects: Dict[int, List[int]] = {}

    for obj_id, obj in objects.items():
        if obj.label not in objects_by_label:
            objects_by_label[obj.label] = []
        objects_by_label[obj.label].append(obj_id)

        if obj.region_id not in region_to_objects:
            region_to_objects[obj.region_id] = []
        region_to_objects[obj.region_id].append(obj_id)

    index = SemanticSceneIndex(
        objects=objects,
        objects_by_label=objects_by_label,
        region_to_objects=region_to_objects,
    )

    # Second pass: compute accurate centroids via multi-view projection
    compute_object_centroids(
        index,
        vehicle,
        num_views=num_projection_views,
        snap_max_distance=snap_max_distance,
    )

    return index
