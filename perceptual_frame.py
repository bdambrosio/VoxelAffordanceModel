"""
perception_frame.py

Planner-facing perception output contract.
This is the Ontology₀ / Ontology₁ boundary artifact.

Key properties:
- Relational (not grid-based)
- Variable-cardinality
- Deterministic assembly from perception proposals
- Opaque IDs for planner reference
- No geometry, physics, or raw voxels
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict


# -------------------------
# Enumerations
# -------------------------

class StructureType(str, Enum):
    SURFACE = "surface"
    WALL = "wall"
    SLOPE = "slope"
    CLIFF = "cliff"
    VOID = "void"
    UNKNOWN = "unknown"


class AffordanceType(str, Enum):
    STEP_FORWARD = "step_forward"
    JUMP_UP = "jump_up"
    DROP_DOWN = "drop_down"
    DIG = "dig"
    SWIM = "swim"
    BREATHE = "breathe"
    CROUCH = "crouch"
    PLACE = "place"


class RiskType(str, Enum):
    FALL = "fall"
    SUFFOCATION = "suffocation"
    DROWNING = "drowning"
    LAVA = "lava"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# -------------------------
# Ordered type lists for training/inference
# -------------------------

# Ordered affordance types (multi-label sigmoid + threshold)
AFFORDANCE_TYPES: List[AffordanceType] = [
    AffordanceType.STEP_FORWARD,
    AffordanceType.JUMP_UP,
    AffordanceType.DROP_DOWN,
    AffordanceType.DIG,
    AffordanceType.SWIM,
    AffordanceType.BREATHE,
    AffordanceType.CROUCH,
    AffordanceType.PLACE,
]

RISK_TYPES: List[RiskType] = [
    RiskType.FALL,
    RiskType.SUFFOCATION,
    RiskType.DROWNING,
    RiskType.LAVA,
]

SEVERITIES: List[Severity] = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]

# Ordered structure types (multi-label sigmoid)
STRUCTURE_TYPES: List[StructureType] = [
    StructureType.SURFACE,
    StructureType.WALL,
    StructureType.SLOPE,
    StructureType.CLIFF,
    StructureType.VOID,
    StructureType.UNKNOWN,
]


# -------------------------
# Core relational elements
# -------------------------

@dataclass
class Structure:
    """
    A perceptually grouped spatial entity.
    Identity is opaque; semantics live in relations.
    """
    id: str
    type_scores: Dict[StructureType, float]  # [0,1] confidence scores for each type
    anchor: Tuple[int, int, int]          # representative relative cell
    extent: Optional[Tuple[int, int, int]] = None
    salience: float = 0.0                 # [0,1], perceptual prominence

    def dominant_type(self) -> StructureType:
        """Returns the type with highest score."""
        return max(self.type_scores.items(), key=lambda x: x[1])[0]

    def type_confidence(self, stype: StructureType) -> float:
        """Returns score for a specific type."""
        return self.type_scores.get(stype, 0.0)

    def is_type(self, stype: StructureType, threshold: float = 0.5) -> bool:
        """Returns True if score exceeds threshold."""
        return self.type_scores.get(stype, 0.0) >= threshold


@dataclass
class Affordance:
    """
    An immediately available action opportunity.
    Preconditions are already validated.
    """
    id: str
    type: AffordanceType
    target: Optional[Tuple[int, int, int]] = None
    associated_structure: Optional[str] = None
    salience: float = 0.0


@dataclass
class Risk:
    """
    A locally present danger.
    Severity reflects near-term consequence.
    """
    id: str
    type: RiskType
    severity: Severity
    source: Optional[Tuple[int, int, int]] = None
    associated_structure: Optional[str] = None


# -------------------------
# Perception frame
# -------------------------

@dataclass
class PerceptionFrame:
    """
    Complete perception snapshot for one decision cycle.
    """
    timestamp: float

    structures: List[Structure] = field(default_factory=list)
    affordances: List[Affordance] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)

    def summary(self) -> str:
        """
        Human-readable one-line summary for debugging/logging.
        """
        return (
            f"PerceptionFrame(structures={len(self.structures)}, "
            f"affordances={len(self.affordances)}, "
            f"risks={len(self.risks)})"
        )
    
    def pretty_print(self) -> str:
        """
        LLM-friendly formatted string representation.
        Groups by category and value for easy volume inference.
        Omits IDs, timestamps, salience, and associated_structure.
        """
        from collections import defaultdict
        
        lines = []
        
        # Group structures by type (structures can have multiple types)
        struct_by_type: Dict[StructureType, set] = defaultdict(set)
        for s in self.structures:
            pos_str = f"[{s.anchor[0]},{s.anchor[1]},{s.anchor[2]}]"
            for stype in s.type_scores.keys():
                struct_by_type[stype].add(pos_str)
        
        if struct_by_type:
            lines.append("structures:")
            for stype in sorted(struct_by_type.keys(), key=lambda x: x.value):
                positions = sorted(struct_by_type[stype])
                lines.append(f"  {stype.value}:")
                for pos in positions:
                    lines.append(f"    {pos}")
        
        # Group affordances by type
        aff_by_type: Dict[AffordanceType, set] = defaultdict(set)
        for a in self.affordances:
            if a.target:
                pos_str = f"[{a.target[0]},{a.target[1]},{a.target[2]}]"
                aff_by_type[a.type].add(pos_str)
        
        if aff_by_type:
            lines.append("affordances:")
            for atype in sorted(aff_by_type.keys(), key=lambda x: x.value):
                positions = sorted(aff_by_type[atype])
                lines.append(f"  {atype.value}:")
                for pos in positions:
                    lines.append(f"    {pos}")
        
        # Group risks by type:severity
        risk_by_key: Dict[str, set] = defaultdict(set)
        for r in self.risks:
            if r.source:
                key = f"{r.type.value}:{r.severity.value}"
                pos_str = f"[{r.source[0]},{r.source[1]},{r.source[2]}]"
                risk_by_key[key].add(pos_str)
        
        if risk_by_key:
            lines.append("risks:")
            for key in sorted(risk_by_key.keys()):
                positions = sorted(risk_by_key[key])
                lines.append(f"  {key}:")
                for pos in positions:
                    lines.append(f"    {pos}")
        
        return "\n".join(lines) if lines else "(empty perception frame)"
        # -------------------------
    # Serialization / Deserialization
    # -------------------------

    @staticmethod
    def from_json(data: dict) -> "PerceptionFrame":
        """
        Construct a PerceptionFrame from a JSON-compatible dict.
        Assumes schema correctness; raises KeyError / ValueError otherwise.
        """

        frame = PerceptionFrame(
            timestamp=float(data["timestamp"]),
        )

        for s in data.get("structures", []):
            # Handle both old format (type) and new format (type_scores)
            if "type_scores" in s:
                # New format: type_scores dict
                type_scores = {
                    StructureType(k): float(v)
                    for k, v in s["type_scores"].items()
                }
            elif "type" in s:
                # Old format: single type (backward compatibility)
                single_type = StructureType(s["type"])
                type_scores = {st: 1.0 if st == single_type else 0.0 for st in StructureType}
            else:
                # Default: all unknown
                type_scores = {st: 0.0 for st in StructureType}
                type_scores[StructureType.UNKNOWN] = 1.0

            frame.structures.append(
                Structure(
                    id=str(s["id"]),
                    type_scores=type_scores,
                    anchor=tuple(s["anchor"]),
                    extent=tuple(s["extent"]) if s.get("extent") is not None else None,
                    salience=float(s.get("salience", 0.0)),
                )
            )

        for a in data.get("affordances", []):
            frame.affordances.append(
                Affordance(
                    id=str(a["id"]),
                    type=AffordanceType(a["type"]),
                    target=tuple(a["target"]) if a.get("target") is not None else None,
                    associated_structure=a.get("associated_structure"),
                    salience=float(a.get("salience", 0.0)),
                )
            )

        for r in data.get("risks", []):
            frame.risks.append(
                Risk(
                    id=str(r["id"]),
                    type=RiskType(r["type"]),
                    severity=Severity(r["severity"]),
                    source=tuple(r["source"]) if r.get("source") is not None else None,
                    associated_structure=r.get("associated_structure"),
                )
            )

        return frame

    def to_json(self) -> dict:
        """
        Convert this PerceptionFrame into a JSON-compatible dict.
        """

        return {
            "timestamp": float(self.timestamp),

            "structures": [
                {
                    "id": s.id,
                    "type_scores": {st.value: float(score) for st, score in s.type_scores.items()},
                    "anchor": list(s.anchor),
                    "extent": list(s.extent) if s.extent is not None else None,
                    "salience": float(s.salience),
                }
                for s in self.structures
            ],

            "affordances": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "target": list(a.target) if a.target is not None else None,
                    "associated_structure": a.associated_structure,
                    "salience": float(a.salience),
                }
                for a in self.affordances
            ],

            "risks": [
                {
                    "id": r.id,
                    "type": r.type.value,
                    "severity": r.severity.value,
                    "source": list(r.source) if r.source is not None else None,
                    "associated_structure": r.associated_structure,
                }
                for r in self.risks
            ],
        }
"""
perceptual_assembler.py

Deterministic assembly of a PerceptionFrame from
per-voxel perceptual evidence.

This module:
- Groups voxels into Structures
- Emits validated Affordances
- Emits conservative Risks

NO learning.
NO raw grid exposure.
NO physics simulation.
"""

import uuid
import time
from typing import Dict, Tuple, List

from perceptual_frame import (
    PerceptionFrame,
    Structure,
    Affordance,
    Risk,
    StructureType,
    AffordanceType,
    RiskType,
    Severity,
)


VoxelKey = Tuple[int, int, int]


class PerceptualAssembler:
    def __init__(
        self,
        structure_threshold: float = 0.6,
        affordance_threshold: float = 0.5,
        risk_threshold: float = 0.5,
    ):
        self.structure_threshold = structure_threshold
        self.affordance_threshold = affordance_threshold
        self.risk_threshold = risk_threshold

    # -------------------------
    # Public API
    # -------------------------

    def assemble(
        self,
        voxel_labels: Dict[VoxelKey, Dict],
    ) -> PerceptionFrame:
        """
        voxel_labels:
          (dx,dy,dz) -> {
            "structure_type_scores": {str: float},
            "affordances": [str],
            "risks": [{type, severity}]
          }
        """

        structures = self._assemble_structures(voxel_labels)
        affordances = self._assemble_affordances(voxel_labels, structures)
        risks = self._assemble_risks(voxel_labels, structures)

        return PerceptionFrame(
            timestamp=time.time(),
            structures=structures,
            affordances=affordances,
            risks=risks,
        )

    # -------------------------
    # Structure assembly
    # -------------------------

    def _assemble_structures(
        self,
        voxel_labels: Dict[VoxelKey, Dict],
    ) -> List[Structure]:
        """
        Extremely simple baseline:
        - One structure per high-confidence voxel
        - Non-mutex type scores preserved
        """

        structures: List[Structure] = []

        for pos, labels in voxel_labels.items():
            scores = {
                StructureType(k): v
                for k, v in labels["structure_type_scores"].items()
                if v >= self.structure_threshold
            }

            if not scores:
                continue

            structures.append(
                Structure(
                    id=self._new_id("struct"),
                    type_scores=scores,
                    anchor=pos,
                    extent=None,
                    salience=max(scores.values()),
                )
            )

        return structures

    # -------------------------
    # Affordance assembly
    # -------------------------

    def _assemble_affordances(
        self,
        voxel_labels: Dict[VoxelKey, Dict],
        structures: List[Structure],
    ) -> List[Affordance]:

        affordances: List[Affordance] = []

        for pos, labels in voxel_labels.items():
            for a in labels.get("affordances", []):
                affordances.append(
                    Affordance(
                        id=self._new_id("aff"),
                        type=AffordanceType(a),
                        target=pos,
                        associated_structure=None,
                        salience=1.0,
                    )
                )

        return affordances

    # -------------------------
    # Risk assembly
    # -------------------------

    def _assemble_risks(
        self,
        voxel_labels: Dict[VoxelKey, Dict],
        structures: List[Structure],
    ) -> List[Risk]:

        risks: List[Risk] = []

        for pos, labels in voxel_labels.items():
            for r in labels.get("risks", []):
                risks.append(
                    Risk(
                        id=self._new_id("risk"),
                        type=RiskType(r["type"]),
                        severity=Severity(r["severity"]),
                        source=pos,
                        associated_structure=None,
                    )
                )

        return risks

    # -------------------------
    # Utilities
    # -------------------------

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
