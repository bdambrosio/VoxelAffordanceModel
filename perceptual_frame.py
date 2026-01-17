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
    TUNNEL = "tunnel"
    ROOM = "room"
    VOID = "void"
    UNKNOWN = "unknown"


class AffordanceType(str, Enum):
    STEP_FORWARD = "step_forward"
    JUMP_UP = "jump_up"
    DROP_DOWN = "drop_down"
    TURN_SCAN = "turn_scan"
    DIG = "dig"
    SWIM = "swim"
    SURFACE = "surface"
    CROUCH = "crouch"
    BACKTRACK = "backtrack"


class RiskType(str, Enum):
    FALL = "fall"
    SUFFOCATION = "suffocation"
    DROWNING = "drowning"
    LAVA = "lava"
    HOSTILE_DARK = "hostile_dark"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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

    # Global qualitative context (optional but useful)
    felt_light: Optional[str] = None      # e.g. "bright", "dim", "hostile-dark"
    orientation_confidence: float = 1.0   # 1.0 = fully oriented, 0.0 = lost

    def summary(self) -> str:
        """
        Human-readable one-line summary for debugging/logging.
        """
        return (
            f"PerceptionFrame(structures={len(self.structures)}, "
            f"affordances={len(self.affordances)}, "
            f"risks={len(self.risks)}, "
            f"felt_light={self.felt_light})"
        )
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
            felt_light=data.get("felt_light"),
            orientation_confidence=float(data.get("orientation_confidence", 1.0)),
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
            "felt_light": self.felt_light,
            "orientation_confidence": float(self.orientation_confidence),

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
