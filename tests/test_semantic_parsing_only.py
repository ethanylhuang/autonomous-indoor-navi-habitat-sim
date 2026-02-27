"""Isolated test for semantic parsing (no habitat-sim dependency).

Tests only the parsing logic without importing compute_object_centroids.
"""

import sys
from pathlib import Path

# Mock the configs.sensor_rig import before loading semantic_scene
sys.modules['configs.sensor_rig'] = type(sys)('sensor_rig')
sys.modules['configs.sensor_rig'].HFOV = 90

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_parsing_logic():
    """Test SemanticSceneParser.parse_semantic_txt directly."""
    # Import after mocking
    from src.perception.semantic_scene import (
        SemanticObject,
        SemanticSceneParser,
        SemanticSceneIndex,
    )

    # Create test file
    content = """object_id,hex_color,"label",region_id
100,#ff0000,"chair",1
101,#ff0001,"chair",1
102,#00ff00,"bed",2
103,#0000ff,"table",1
104,#ffffff,"wall",1
105,#000000,"floor",1
"""

    tmp_path = Path("/tmp/test_semantic_parser.txt")
    tmp_path.write_text(content)

    try:
        index = SemanticSceneParser.parse_semantic_txt(tmp_path)

        assert index is not None, "Parser returned None"
        assert len(index.objects) == 4, f"Expected 4 objects, got {len(index.objects)}"

        # Check chairs
        assert "chair" in index.objects_by_label
        assert len(index.objects_by_label["chair"]) == 2

        # Check instance numbering
        chair_ids = index.objects_by_label["chair"]
        chair_instances = [index.objects[cid].instance_number for cid in chair_ids]
        assert sorted(chair_instances) == [1, 2], f"Wrong instance numbers: {chair_instances}"

        # Check that structural labels were skipped
        for obj in index.objects.values():
            assert obj.label not in ["wall", "floor", "ceiling"]

        print("✓ Parsing logic test passed")

        # Test to_dict serialization
        for obj_id, obj in index.objects.items():
            obj_dict = obj.to_dict()
            assert "object_id" in obj_dict
            assert "label" in obj_dict
            assert "instance_name" in obj_dict
            assert "is_navigable" in obj_dict
            # Initially no centroids computed
            assert obj_dict["navmesh_position"] is None

        print("✓ to_dict serialization test passed")

        return True

    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n=== Testing Semantic Scene Parser (isolated) ===\n")
    test_parsing_logic()
    print("\n=== All parsing tests passed! ===\n")
