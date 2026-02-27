"""Shield verification tests for semantic scene parsing implementation.

Tests the semantic scene parsing module without requiring habitat-sim.
"""

import sys
from pathlib import Path
from io import StringIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.perception.semantic_scene import (
    SemanticObject,
    SemanticSceneIndex,
    SemanticSceneParser,
    get_navigable_objects,
)


def test_semantic_object_dataclass():
    """Test SemanticObject dataclass structure."""
    obj = SemanticObject(
        object_id=42,
        label="chair",
        hex_color="#ff0000",
        region_id=1,
        centroid=np.array([1.0, 2.0, 3.0]),
        navmesh_position=np.array([1.1, 2.0, 3.1]),
        instance_number=1,
    )

    assert obj.object_id == 42
    assert obj.label == "chair"
    assert obj.instance_number == 1
    assert obj.instance_name == "chair #1 (region 1)"

    # Test to_dict serialization
    obj_dict = obj.to_dict()
    assert obj_dict["object_id"] == 42
    assert obj_dict["label"] == "chair"
    assert obj_dict["instance_name"] == "chair #1 (region 1)"
    assert obj_dict["is_navigable"] is True
    assert obj_dict["centroid"] == [1.0, 2.0, 3.0]
    assert obj_dict["navmesh_position"] == [1.1, 2.0, 3.1]

    # Test object without navmesh position (not navigable)
    obj_not_nav = SemanticObject(
        object_id=43,
        label="bed",
        hex_color="#00ff00",
        region_id=2,
        centroid=np.array([4.0, 5.0, 6.0]),
        navmesh_position=None,
        instance_number=1,
    )
    obj_not_nav_dict = obj_not_nav.to_dict()
    assert obj_not_nav_dict["is_navigable"] is False
    assert obj_not_nav_dict["navmesh_position"] is None

    print("✓ SemanticObject dataclass test passed")


def test_semantic_scene_index_structure():
    """Test SemanticSceneIndex dataclass structure."""
    obj1 = SemanticObject(
        object_id=1, label="chair", hex_color="#ff0000", region_id=1,
        centroid=None, navmesh_position=None, instance_number=1
    )
    obj2 = SemanticObject(
        object_id=2, label="chair", hex_color="#ff0001", region_id=1,
        centroid=None, navmesh_position=None, instance_number=2
    )
    obj3 = SemanticObject(
        object_id=3, label="bed", hex_color="#00ff00", region_id=2,
        centroid=None, navmesh_position=None, instance_number=1
    )

    index = SemanticSceneIndex(
        objects={1: obj1, 2: obj2, 3: obj3},
        objects_by_label={"chair": [1, 2], "bed": [3]},
        region_to_objects={1: [1, 2], 2: [3]},
    )

    assert len(index.objects) == 3
    assert len(index.objects_by_label["chair"]) == 2
    assert len(index.objects_by_label["bed"]) == 1
    assert 1 in index.region_to_objects
    assert 2 in index.region_to_objects

    print("✓ SemanticSceneIndex structure test passed")


def test_parse_semantic_txt_basic():
    """Test basic parsing of .semantic.txt format."""
    # Create temporary semantic.txt content
    content = """object_id,hex_color,"label",region_id
100,#ff0000,"chair",1
101,#ff0001,"chair",1
102,#00ff00,"bed",2
103,#0000ff,"table",1
104,#ffffff,"wall",1
105,#000000,"floor",1
"""

    # Write to temp file
    tmp_path = Path("/tmp/test_semantic_shield.txt")
    tmp_path.write_text(content)

    try:
        index = SemanticSceneParser.parse_semantic_txt(tmp_path)

        assert index is not None
        # Should have 3 objects (chair x2, bed x1, table x1)
        # wall and floor should be skipped
        assert len(index.objects) == 4

        # Check that walls and floors were filtered
        for obj in index.objects.values():
            assert obj.label not in SemanticSceneParser.SKIP_LABELS

        # Check instance numbering for chairs
        chairs = [index.objects[oid] for oid in index.objects_by_label["chair"]]
        assert len(chairs) == 2
        assert chairs[0].instance_number == 1
        assert chairs[1].instance_number == 2

        # Check reverse indices
        assert "chair" in index.objects_by_label
        assert "bed" in index.objects_by_label
        assert 1 in index.region_to_objects
        assert 2 in index.region_to_objects

        print("✓ parse_semantic_txt basic test passed")

    finally:
        tmp_path.unlink(missing_ok=True)


def test_parse_semantic_txt_edge_cases():
    """Test edge cases: empty file, malformed lines, missing file."""
    # Test 1: Missing file
    result = SemanticSceneParser.parse_semantic_txt(Path("/nonexistent/file.txt"))
    assert result is None
    print("✓ Missing file returns None")

    # Test 2: Empty file (header only)
    tmp_path = Path("/tmp/test_semantic_empty.txt")
    tmp_path.write_text("object_id,hex_color,\"label\",region_id\n")
    try:
        index = SemanticSceneParser.parse_semantic_txt(tmp_path)
        assert index is not None
        assert len(index.objects) == 0
        print("✓ Empty file (header only) returns empty index")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Test 3: Malformed lines (should be skipped)
    tmp_path = Path("/tmp/test_semantic_malformed.txt")
    tmp_path.write_text("""object_id,hex_color,"label",region_id
100,#ff0000,"chair",1
INVALID_LINE_NO_COMMAS
200,#00ff00,bed  # missing quotes
300,#0000ff,"table",2
400,incomplete,row
""")
    try:
        index = SemanticSceneParser.parse_semantic_txt(tmp_path)
        assert index is not None
        # Only valid lines should be parsed (chair and table)
        assert len(index.objects) == 2
        print("✓ Malformed lines are skipped gracefully")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_get_navigable_objects():
    """Test get_navigable_objects filtering and sorting."""
    obj1 = SemanticObject(
        object_id=1, label="chair", hex_color="#ff0000", region_id=1,
        centroid=np.array([1.0, 2.0, 3.0]),
        navmesh_position=np.array([1.0, 2.0, 3.0]),  # navigable
        instance_number=1
    )
    obj2 = SemanticObject(
        object_id=2, label="bed", hex_color="#00ff00", region_id=2,
        centroid=np.array([4.0, 5.0, 6.0]),
        navmesh_position=None,  # NOT navigable
        instance_number=1
    )
    obj3 = SemanticObject(
        object_id=3, label="chair", hex_color="#ff0001", region_id=1,
        centroid=np.array([7.0, 8.0, 9.0]),
        navmesh_position=np.array([7.0, 8.0, 9.0]),  # navigable
        instance_number=2
    )
    obj4 = SemanticObject(
        object_id=4, label="table", hex_color="#0000ff", region_id=1,
        centroid=np.array([10.0, 11.0, 12.0]),
        navmesh_position=np.array([10.0, 11.0, 12.0]),  # navigable
        instance_number=1
    )

    index = SemanticSceneIndex(
        objects={1: obj1, 2: obj2, 3: obj3, 4: obj4},
        objects_by_label={
            "chair": [1, 3],
            "bed": [2],
            "table": [4],
        },
        region_to_objects={1: [1, 3, 4], 2: [2]},
    )

    navigable = get_navigable_objects(index)

    # Should only return objects 1, 3, 4 (bed is not navigable)
    assert len(navigable) == 3

    # Check sorting: should be sorted by (label, instance_number)
    # Expected order: chair #1, chair #2, table #1
    assert navigable[0].label == "chair"
    assert navigable[0].instance_number == 1
    assert navigable[1].label == "chair"
    assert navigable[1].instance_number == 2
    assert navigable[2].label == "table"
    assert navigable[2].instance_number == 1

    print("✓ get_navigable_objects filtering and sorting test passed")


def test_skip_labels_coverage():
    """Test that all structural labels are properly skipped."""
    content = """object_id,hex_color,"label",region_id
100,#ffffff,"floor",1
101,#eeeeee,"ceiling",1
102,#dddddd,"wall",1
103,#cccccc,"door frame",1
104,#bbbbbb,"unknown",1
105,#aaaaaa,"beam",1
106,#999999,"door",1
107,#888888,"window",1
108,#777777,"stair",1
109,#666666,"stairs",1
110,#ff0000,"chair",1
"""

    tmp_path = Path("/tmp/test_skip_labels.txt")
    tmp_path.write_text(content)

    try:
        index = SemanticSceneParser.parse_semantic_txt(tmp_path)
        assert index is not None
        # Only chair should be parsed (all others are in SKIP_LABELS)
        assert len(index.objects) == 1
        assert list(index.objects.values())[0].label == "chair"

        print("✓ SKIP_LABELS filter test passed")

    finally:
        tmp_path.unlink(missing_ok=True)


def test_instance_numbering_multiple_regions():
    """Test that instance numbering is global, not per-region."""
    content = """object_id,hex_color,"label",region_id
100,#ff0000,"chair",1
101,#ff0001,"chair",2
102,#ff0002,"chair",3
103,#00ff00,"bed",1
104,#00ff01,"bed",2
"""

    tmp_path = Path("/tmp/test_instance_numbering.txt")
    tmp_path.write_text(content)

    try:
        index = SemanticSceneParser.parse_semantic_txt(tmp_path)
        assert index is not None

        # Check chair instances (across regions)
        chairs = [index.objects[oid] for oid in index.objects_by_label["chair"]]
        assert len(chairs) == 3
        # Instance numbers should be sequential 1, 2, 3 (not reset per region)
        instance_nums = sorted([c.instance_number for c in chairs])
        assert instance_nums == [1, 2, 3]

        # Check bed instances
        beds = [index.objects[oid] for oid in index.objects_by_label["bed"]]
        assert len(beds) == 2
        instance_nums = sorted([b.instance_number for b in beds])
        assert instance_nums == [1, 2]

        print("✓ Instance numbering across regions test passed")

    finally:
        tmp_path.unlink(missing_ok=True)


def run_all_tests():
    """Run all shield verification tests."""
    print("\n=== Shield Verification Tests for Semantic Scene Parsing ===\n")

    test_semantic_object_dataclass()
    test_semantic_scene_index_structure()
    test_parse_semantic_txt_basic()
    test_parse_semantic_txt_edge_cases()
    test_get_navigable_objects()
    test_skip_labels_coverage()
    test_instance_numbering_multiple_regions()

    print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
