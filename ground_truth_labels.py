"""
ground_truth_labels.py

Extract ground truth PerceptionFrame labels from training data for a specific voxel.
"""

import json
from typing import Dict, Any, Tuple, Optional
from perceptual_frame import StructureType, AffordanceType, RiskType, Severity


def _build_voxel_map(gt_grid: Dict[str, Any]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """Build a lookup map for voxels by absolute coordinates."""
    voxel_map = {}
    for v in gt_grid["voxels"]:
        key = (v["x"], v["y"], v["z"])
        voxel_map[key] = v
    return voxel_map


def _get_neighbor_voxel(
    voxel_map: Dict[Tuple[int, int, int], Dict[str, Any]],
    abs_x: int,
    abs_y: int,
    abs_z: int,
    dx: int,
    dy: int,
    dz: int
) -> Optional[Dict[str, Any]]:
    """Get neighbor voxel at offset (dx, dy, dz)."""
    return voxel_map.get((abs_x + dx, abs_y + dy, abs_z + dz))


def _is_air_or_void(voxel: Optional[Dict[str, Any]]) -> bool:
    """Check if voxel is air or void."""
    if not voxel:
        return True
    name = voxel.get("name", "")
    return name == "air" or "air" in name.lower()


def _is_solid(voxel: Optional[Dict[str, Any]]) -> bool:
    """Check if voxel is solid."""
    if not voxel:
        return False
    props = voxel.get("properties", {})
    return props.get("support", False) and not props.get("passable", True)


def get_ground_truth_labels(
    ground_truth_record: Dict[str, Any],
    relative_x: int,
    relative_y: int,
    relative_z: int
) -> Dict[str, Any]:
    """
    Extract complete ground truth PerceptionFrame labels for a voxel at relative position.
    
    Infers ALL StructureTypes, AffordanceTypes, and RiskTypes from:
    - Direct structure_cues (surface_likeness, wall_likeness, enclosure_likeness)
    - Voxel properties (support, passable, head_clear, breakable, hazard)
    - Block name (air, water, lava, etc.)
    - Neighbor voxels (for CLIFF, FALL, SUFFOCATION detection)
    
    Args:
        ground_truth_record: Full training data record with ground_truth_grid
        relative_x, relative_y, relative_z: Relative position to agent center
    
    Returns:
        Dict with:
        - structure_type_scores: Dict[str, float] - scores for all StructureTypes
        - affordances: List[str] - all applicable AffordanceTypes
        - risks: List[Dict] - all applicable RiskTypes with severity
        - voxel_position: Tuple[int, int, int] - absolute coordinates
        - relative_position: Tuple[int, int, int] - relative coordinates
    """
    gt_grid = ground_truth_record.get("ground_truth_grid")
    if not gt_grid or not gt_grid.get("ok"):
        raise ValueError("Invalid ground truth grid")
    
    center = gt_grid["center"]
    abs_x = center["x"] + relative_x
    abs_y = center["y"] + relative_y
    abs_z = center["z"] + relative_z
    
    # Build voxel map for neighbor lookups
    voxel_map = _build_voxel_map(gt_grid)
    voxel = voxel_map.get((abs_x, abs_y, abs_z))
    
    if not voxel:
        # Voxel not found - return void labels
        return {
            "structure_type_scores": {st.value: 1.0 if st == StructureType.VOID else 0.0 for st in StructureType},
            "affordances": [],
            "risks": []
        }
    
    props = voxel.get("properties", {})
    cues = props.get("structure_cues", {})
    name = voxel.get("name", "").lower()
    
    # ===== COMPLETE STRUCTURE TYPE INFERENCE =====
    structure_scores = {st: 0.0 for st in StructureType}
    
    # Direct cues from ground truth
    if cues.get("surface_likeness"):
        structure_scores[StructureType.SURFACE] = 1.0
    if cues.get("wall_likeness"):
        structure_scores[StructureType.WALL] = 1.0
    
    # Infer VOID (air blocks)
    if "air" in name or _is_air_or_void(voxel):
        structure_scores[StructureType.VOID] = 1.0
    
    # Infer SLOPE (support + passable, or gradual elevation)
    if props.get("support") and props.get("passable"):
        # Check if there's elevation change
        below = _get_neighbor_voxel(voxel_map, abs_x, abs_y, abs_z, 0, -1, 0)
        if below and not _is_air_or_void(below):
            structure_scores[StructureType.SLOPE] = 0.7
    
    # Infer CLIFF (vertical drop nearby)
    if props.get("support"):
        below = _get_neighbor_voxel(voxel_map, abs_x, abs_y, abs_z, 0, -1, 0)
        if _is_air_or_void(below):
            # Check if there's a significant drop
            deep_below = _get_neighbor_voxel(voxel_map, abs_x, abs_y, abs_z, 0, -2, 0)
            if _is_air_or_void(deep_below):
                structure_scores[StructureType.CLIFF] = 0.8
    
    # If no structure identified, mark as UNKNOWN
    if not any(structure_scores.values()):
        structure_scores[StructureType.UNKNOWN] = 1.0
    
    # ===== COMPLETE AFFORDANCE INFERENCE =====
    affordances = []
    
    # STEP_FORWARD: support + passable
    if props.get("support") and props.get("passable"):
        affordances.append(AffordanceType.STEP_FORWARD.value)
    
    # JUMP_UP: passable but block above, or need to jump to reach
    above = _get_neighbor_voxel(voxel_map, abs_x, abs_y, abs_z, 0, 1, 0)
    if props.get("passable") and _is_solid(above):
        affordances.append(AffordanceType.JUMP_UP.value)
    
    # DROP_DOWN: support but void below
    below = _get_neighbor_voxel(voxel_map, abs_x, abs_y, abs_z, 0, -1, 0)
    if props.get("support") and _is_air_or_void(below):
        affordances.append(AffordanceType.DROP_DOWN.value)
    
    # DIG: breakable blocks
    if props.get("breakable"):
        affordances.append(AffordanceType.DIG.value)
    
    # SWIM: water blocks
    if "water" in name:
        affordances.append(AffordanceType.SWIM.value)
    
    # BREATHE: for water, need to surface to breathe
    if "water" in name:
        affordances.append(AffordanceType.BREATHE.value)
    
    # CROUCH: passable but not head_clear
    if props.get("passable") and not props.get("head_clear", True):
        affordances.append(AffordanceType.CROUCH.value)
    
    # PLACE: can place a block here (air, replaceable blocks, or fluids)
    # Air blocks
    if _is_air_or_void(voxel):
        affordances.append(AffordanceType.PLACE.value)
    # Fluids (water, lava)
    elif "water" in name or "lava" in name:
        affordances.append(AffordanceType.PLACE.value)
    # Replaceable blocks (grass, ferns, etc.) - check via passable but not support
    elif props.get("passable") and not props.get("support"):
        affordances.append(AffordanceType.PLACE.value)
    
    # ===== COMPLETE RISK INFERENCE =====
    risks = []
    
    # FALL: void below with no support
    if not props.get("support") and _is_air_or_void(below):
        deep_below = _get_neighbor_voxel(voxel_map, abs_x, abs_y, abs_z, 0, -2, 0)
        if _is_air_or_void(deep_below):
            risks.append({"type": RiskType.FALL.value, "severity": Severity.HIGH.value})
    
    # SUFFOCATION: solid block at head level
    if _is_solid(above) and not props.get("passable"):
        risks.append({"type": RiskType.SUFFOCATION.value, "severity": Severity.HIGH.value})
    
    # DROWNING: water blocks
    if "water" in name:
        risks.append({"type": RiskType.DROWNING.value, "severity": Severity.MEDIUM.value})
    
    # LAVA: lava blocks
    if "lava" in name or props.get("hazard"):
        if "lava" in name:
            risks.append({"type": RiskType.LAVA.value, "severity": Severity.HIGH.value})
    
    return {
        "structure_type_scores": {st.value: float(score) for st, score in structure_scores.items()},
        "affordances": affordances,
        "risks": risks,
        "voxel_position": (abs_x, abs_y, abs_z),
        "relative_position": (relative_x, relative_y, relative_z)
    }


def get_ground_truth_labels_batch(
    ground_truth_record: Dict[str, Any],
    relative_positions: list[Tuple[int, int, int]]
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """
    Get labels for multiple positions at once (more efficient).
    
    Returns dict mapping (x, y, z) -> labels
    """
    gt_grid = ground_truth_record.get("ground_truth_grid")
    if not gt_grid or not gt_grid.get("ok"):
        raise ValueError("Invalid ground truth grid")
    
    center = gt_grid["center"]
    voxel_map = _build_voxel_map(gt_grid)
    
    results = {}
    for rel_x, rel_y, rel_z in relative_positions:
        # Use the main function for consistency
        labels = get_ground_truth_labels(ground_truth_record, rel_x, rel_y, rel_z)
        # Remove position info for batch results
        labels.pop("voxel_position", None)
        labels.pop("relative_position", None)
        results[(rel_x, rel_y, rel_z)] = labels
    
    return results


if __name__ == "__main__":
    # Example usage
    with open("training_data/sample_1768677695428.json") as f:
        record = json.load(f)
    
    # Test at origin (agent position)
    labels = get_ground_truth_labels(record, 0, 0, 0)
    print("Labels at (0, 0, 0):")
    print(json.dumps(labels, indent=2, default=str))
    
    # Test at offset position
    labels2 = get_ground_truth_labels(record, 1, 0, -1)
    print("\nLabels at (1, 0, -1):")
    print(json.dumps(labels2, indent=2, default=str))
