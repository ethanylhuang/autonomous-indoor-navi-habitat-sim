"""Standalone test for semantic scene parsing - no habitat_sim dependency.

Tests annotation parsing directly without any simulator imports.
"""

from pathlib import Path
from typing import Dict, List


def parse_semantic_txt_test(filepath: Path) -> dict:
    """Simplified parser for testing."""
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

    objects = {}
    label_counts = {}

    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue

            object_id = int(parts[0])
            hex_color = parts[1]
            label = parts[2].strip('"').lower()
            region_id = int(parts[3])

            if label in SKIP_LABELS:
                continue

            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

            objects[object_id] = {
                "label": label,
                "hex_color": hex_color,
                "region_id": region_id,
                "instance_number": label_counts[label],
            }

    return {"objects": objects, "label_counts": label_counts}


def main():
    """Test semantic parsing without habitat_sim."""
    semantic_file = (
        Path(__file__).parent.parent
        / "data"
        / "hm3d-minival-semantic-annots-v0.2"
        / "00802-wcojb4TFT35"
        / "wcojb4TFT35.semantic.txt"
    )

    if not semantic_file.exists():
        print(f"SKIP: Semantic file not found at {semantic_file}")
        return

    print(f"Testing semantic parsing: {semantic_file.name}")
    print("-" * 60)

    result = parse_semantic_txt_test(semantic_file)
    objects = result["objects"]
    label_counts = result["label_counts"]

    print(f"Total objects parsed: {len(objects)}")
    print(f"Unique labels: {len(label_counts)}")
    print()

    # Show label distribution
    print("Label distribution (top 20):")
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:20]
    for label, count in sorted_labels:
        print(f"  {label:25s}: {count:3d} instances")

    print()

    # Verify instance numbering
    print("Verifying instance numbering...")
    for label, count in label_counts.items():
        label_objs = [obj for obj in objects.values() if obj["label"] == label]
        instance_numbers = [obj["instance_number"] for obj in label_objs]
        expected = list(range(1, count + 1))
        assert instance_numbers == expected, (
            f"Instance numbers for '{label}' incorrect: {instance_numbers}"
        )

    print("  Instance numbering: OK")

    # Show some examples
    print()
    print("Example objects:")
    for obj_id in list(objects.keys())[:10]:
        obj = objects[obj_id]
        instance_name = f"{obj['label']} #{obj['instance_number']} (region {obj['region_id']})"
        print(f"  ID {obj_id:4d}: {instance_name}")

    # Check for navigable objects (beds, chairs, tables, etc.)
    navigable_labels = ["bed", "chair", "table", "sofa", "toilet", "sink", "tv"]
    print()
    print("Navigable object candidates:")
    for label in navigable_labels:
        count = label_counts.get(label, 0)
        if count > 0:
            print(f"  {label:15s}: {count:3d} instances")

    print()
    print("SUCCESS: Semantic parsing test passed!")


if __name__ == "__main__":
    main()
