#!/usr/bin/env python3
"""Test habitat_sim.SemanticScene API for direct object access."""

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg

# Test scenes
SCENES = [
    # Test scenes (basic)
    "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    # HM3D with semantic annotations
    "data/hm3d-minival-habitat-v0.2/00802-wcojb4TFT35/wcojb4TFT35.basis.glb",
]


def test_semantic_scene_api(scene_path: str) -> None:
    """Test SemanticScene object access on a given scene."""
    print(f"\n{'='*60}")
    print(f"Testing: {scene_path}")
    print("=" * 60)

    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = scene_path

    try:
        hab_cfg = make_cfg(cfg_settings)
        sim = habitat_sim.Simulator(hab_cfg)

        semantic_scene = sim.semantic_scene
        print(f"semantic_scene exists: {semantic_scene is not None}")

        if semantic_scene is None:
            print("  -> No semantic scene available")
            sim.close()
            return

        # Check objects
        objects = semantic_scene.objects
        print(f"Number of objects: {len(objects) if objects else 0}")

        if objects:
            print("\nFirst 10 objects:")
            for i, obj in enumerate(objects[:10]):
                try:
                    obj_id = obj.id if hasattr(obj, "id") else "N/A"
                    category = obj.category.name() if obj.category else "N/A"
                    center = obj.aabb.center if obj.aabb else "N/A"
                    size = obj.aabb.size if obj.aabb else "N/A"
                    print(f"  [{i}] id={obj_id}, category={category}")
                    print(f"       center={center}, size={size}")
                except Exception as e:
                    print(f"  [{i}] Error accessing object: {e}")

        # Check regions
        regions = semantic_scene.regions
        print(f"\nNumber of regions: {len(regions) if regions else 0}")

        if regions:
            print("\nRegions:")
            for i, region in enumerate(regions[:5]):
                try:
                    region_id = region.id if hasattr(region, "id") else "N/A"
                    category = region.category.name() if region.category else "N/A"
                    print(f"  [{i}] id={region_id}, category={category}")
                except Exception as e:
                    print(f"  [{i}] Error accessing region: {e}")

        # Check levels
        levels = semantic_scene.levels
        print(f"\nNumber of levels: {len(levels) if levels else 0}")

        sim.close()

    except Exception as e:
        print(f"Error: {e}")


def main():
    for scene in SCENES:
        test_semantic_scene_api(scene)

    # Also try with scene dataset config for HM3D
    print(f"\n{'='*60}")
    print("Testing with scene_dataset_config.json")
    print("=" * 60)

    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene_dataset_config_file"] = (
        "data/hm3d-minival-habitat-v0.2/00802-wcojb4TFT35/"
        "wcojb4TFT35.scene_dataset_config.json"
    )
    cfg_settings["scene"] = "wcojb4TFT35"

    try:
        hab_cfg = make_cfg(cfg_settings)
        sim = habitat_sim.Simulator(hab_cfg)

        semantic_scene = sim.semantic_scene
        print(f"semantic_scene exists: {semantic_scene is not None}")

        if semantic_scene:
            objects = semantic_scene.objects
            print(f"Number of objects: {len(objects) if objects else 0}")

            if objects:
                # Inspect first object's aabb to find correct attributes
                first_obj = objects[0]
                print(f"\nAABB type: {type(first_obj.aabb)}")
                print(f"AABB dir: {[a for a in dir(first_obj.aabb) if not a.startswith('_')]}")

                print("\nFirst 20 objects with AABB centers:")
                for i, obj in enumerate(objects[:20]):
                    try:
                        category = obj.category.name() if obj.category else "N/A"
                        aabb = obj.aabb
                        if aabb:
                            # Range3D: use .center() method and .size() for dimensions
                            center = aabb.center()
                            size = aabb.size()
                            center_list = [center[0], center[1], center[2]]
                            size_list = [size[0], size[1], size[2]]
                            print(f"  {category}: center={center_list}, size={size_list}")
                        else:
                            print(f"  {category}: no AABB")
                    except Exception as e:
                        print(f"  Error: {e}")

        sim.close()

    except Exception as e:
        print(f"Error with scene dataset config: {e}")


if __name__ == "__main__":
    main()
