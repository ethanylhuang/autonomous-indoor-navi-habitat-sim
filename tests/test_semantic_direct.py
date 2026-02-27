#!/usr/bin/env python3
"""Test direct semantic scene API extraction.

Uses habitat_sim directly to test SemanticScene object access.
"""

import numpy as np

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg

from src.perception.semantic_scene import SemanticSceneParser


def main():
    # Use habitat_sim directly with scene_dataset_config
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene_dataset_config_file"] = (
        "data/hm3d-minival-habitat-v0.2/00802-wcojb4TFT35/"
        "wcojb4TFT35.scene_dataset_config.json"
    )
    cfg_settings["scene"] = "wcojb4TFT35"

    print("Initializing simulator with HM3D scene...")
    hab_cfg = make_cfg(cfg_settings)
    sim = habitat_sim.Simulator(hab_cfg)

    # Load navmesh manually for pathfinder access
    navmesh_path = "data/hm3d-minival-habitat-v0.2/00802-wcojb4TFT35/wcojb4TFT35.basis.navmesh"
    sim.pathfinder.load_nav_mesh(navmesh_path)
    print(f"NavMesh loaded: {sim.pathfinder.is_loaded}")

    print("\nExtracting objects from SemanticScene API...")
    semantic_scene = sim.semantic_scene

    if semantic_scene is None:
        print("ERROR: No semantic scene available")
        sim.close()
        return

    sim_objects = semantic_scene.objects
    print(f"Total objects in scene: {len(sim_objects) if sim_objects else 0}")

    if not sim_objects:
        print("ERROR: No objects found")
        sim.close()
        return

    # Process objects similar to build_semantic_index_from_sim
    objects = {}
    label_counts = {}

    for sim_obj in sim_objects:
        try:
            category = sim_obj.category
            if category is None:
                continue

            label = category.name().lower()

            # Skip structural elements
            if label in SemanticSceneParser.SKIP_LABELS:
                continue

            object_id = category.index()

            aabb = sim_obj.aabb
            if aabb is None:
                continue

            center = aabb.center()
            centroid = np.array([center[0], center[1], center[2]], dtype=np.float64)

            size = aabb.size()
            size_array = np.array([size[0], size[1], size[2]], dtype=np.float64)

            # Skip huge objects
            if np.any(size_array > 20.0):
                continue

            # Count instances per label
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

            # Snap to navmesh
            snapped = sim.pathfinder.snap_point(centroid)
            is_navigable = snapped is not None and sim.pathfinder.is_navigable(snapped)

            objects[object_id] = {
                "label": label,
                "instance": label_counts[label],
                "centroid": centroid,
                "size": size_array,
                "navmesh_pos": np.array(snapped) if is_navigable else None,
                "navigable": is_navigable,
            }

        except Exception as e:
            continue

    print(f"\nFiltered objects (excluding structural): {len(objects)}")
    print(f"Unique labels: {len(label_counts)}")

    # Show label counts
    print("\nObjects by label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Show navigable objects
    navigable = [o for o in objects.values() if o["navigable"]]
    print(f"\nNavigable objects: {len(navigable)}")

    print("\nFirst 15 navigable objects:")
    for obj in list(navigable)[:15]:
        c = obj["centroid"]
        n = obj["navmesh_pos"]
        print(f"  {obj['label']} #{obj['instance']}")
        print(f"    centroid: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]")
        if n is not None:
            print(f"    navmesh:  [{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]")

    sim.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
