"""Unit tests for semantic scene parsing.

Tests annotation parsing without requiring habitat_sim.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.semantic_scene import SemanticSceneParser


def test_semantic_txt_parsing():
    """Test parsing of HM3D .semantic.txt file."""
    # Use the example file from the project
    semantic_file = (
        Path(__file__).parent.parent
        / "data"
        / "hm3d-minival-semantic-annots-v0.2"
        / "00802-wcojb4TFT35"
        / "wcojb4TFT35.semantic.txt"
    )

    if not semantic_file.exists():
        print(f"Skipping test: semantic file not found at {semantic_file}")
        return

    index = SemanticSceneParser.parse_semantic_txt(semantic_file)

    assert index is not None, "Failed to parse semantic file"
    assert len(index.objects) > 0, "No objects parsed"

    # Check that structural elements are filtered out
    for obj_id, obj in index.objects.items():
        assert obj.label not in SemanticSceneParser.SKIP_LABELS, (
            f"Skipped label '{obj.label}' should not be in index"
        )

    # Verify instance numbering for duplicate labels
    # E.g., multiple "bed" objects should have instance_number 1, 2, 3...
    for label, obj_ids in index.objects_by_label.items():
        instance_numbers = [index.objects[oid].instance_number for oid in obj_ids]
        # Should be consecutive starting from 1
        assert instance_numbers == list(range(1, len(obj_ids) + 1)), (
            f"Instance numbers for '{label}' not consecutive: {instance_numbers}"
        )

    # Check that region grouping works
    assert len(index.region_to_objects) > 0, "No region groupings found"

    print(f"Parsed {len(index.objects)} semantic objects")
    print(f"Unique labels: {len(index.objects_by_label)}")
    print(f"Regions: {len(index.region_to_objects)}")

    # Print some examples
    print("\nExample objects by label:")
    for label in list(index.objects_by_label.keys())[:5]:
        obj_ids = index.objects_by_label[label]
        print(f"  {label}: {len(obj_ids)} instances")
        for oid in obj_ids[:3]:
            obj = index.objects[oid]
            print(f"    - {obj.instance_name}")


def test_skip_labels():
    """Verify that SKIP_LABELS are comprehensive for structural elements."""
    # These labels should all be in SKIP_LABELS
    structural = [
        "floor",
        "ceiling",
        "wall",
        "door frame",
        "door",
        "window",
        "stairs",
        "stair",
        "beam",
        "unknown",
    ]

    for label in structural:
        assert label in SemanticSceneParser.SKIP_LABELS, (
            f"Structural label '{label}' should be in SKIP_LABELS"
        )


def test_semantic_object_instance_name():
    """Test instance name formatting."""
    from src.perception.semantic_scene import SemanticObject

    obj = SemanticObject(
        object_id=17,
        label="bed",
        hex_color="3631F3",
        region_id=16,
        centroid=None,
        navmesh_position=None,
        instance_number=1,
    )

    assert obj.instance_name == "bed #1 (region 16)"

    obj2 = SemanticObject(
        object_id=32,
        label="chair",
        hex_color="A9B884",
        region_id=16,
        centroid=None,
        navmesh_position=None,
        instance_number=3,
    )

    assert obj2.instance_name == "chair #3 (region 16)"


def test_semantic_object_to_dict():
    """Test serialization to dict for JSON API."""
    from src.perception.semantic_scene import SemanticObject

    obj = SemanticObject(
        object_id=17,
        label="bed",
        hex_color="3631F3",
        region_id=16,
        centroid=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        navmesh_position=np.array([1.1, 2.0, 3.1], dtype=np.float64),
        instance_number=1,
    )

    d = obj.to_dict()

    assert d["object_id"] == 17
    assert d["label"] == "bed"
    assert d["region_id"] == 16
    assert d["instance_name"] == "bed #1 (region 16)"
    assert d["is_navigable"] is True
    assert d["centroid"] == [1.0, 2.0, 3.0]
    assert d["navmesh_position"] == [1.1, 2.0, 3.1]


def test_get_navigable_objects():
    """Test filtering for navigable objects."""
    from src.perception.semantic_scene import (
        SemanticObject,
        SemanticSceneIndex,
        get_navigable_objects,
    )

    # Create test index with mix of navigable and non-navigable objects
    objects = {
        1: SemanticObject(
            object_id=1,
            label="bed",
            hex_color="FF0000",
            region_id=1,
            centroid=np.array([0.0, 0.0, 0.0]),
            navmesh_position=np.array([0.0, 0.0, 0.0]),
            instance_number=1,
        ),
        2: SemanticObject(
            object_id=2,
            label="chair",
            hex_color="00FF00",
            region_id=1,
            centroid=np.array([1.0, 1.0, 1.0]),
            navmesh_position=None,  # NOT navigable
            instance_number=1,
        ),
        3: SemanticObject(
            object_id=3,
            label="table",
            hex_color="0000FF",
            region_id=2,
            centroid=np.array([2.0, 2.0, 2.0]),
            navmesh_position=np.array([2.0, 2.0, 2.0]),
            instance_number=1,
        ),
    }

    index = SemanticSceneIndex(
        objects=objects, objects_by_label={}, region_to_objects={}
    )

    navigable = get_navigable_objects(index)

    # Should only return objects 1 and 3
    assert len(navigable) == 2
    assert all(obj.navmesh_position is not None for obj in navigable)
    assert navigable[0].object_id == 1  # bed comes before table alphabetically
    assert navigable[1].object_id == 3


if __name__ == "__main__":
    test_skip_labels()
    test_semantic_object_instance_name()
    test_semantic_object_to_dict()
    test_get_navigable_objects()
    test_semantic_txt_parsing()
    print("\nAll semantic parsing tests passed!")
