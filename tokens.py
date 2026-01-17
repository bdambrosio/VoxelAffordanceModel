"""
token_schema.py

Canonical token schema for the perception model.
This defines the exact, model-facing input contract derived
from CanonicalGrid / CanonicalCell.

This module contains:
- Enum definitions (stable IDs)
- A Token dataclass
- A deterministic builder from CanonicalCell + agent pose

No learning. No geometry inference. No task semantics.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Dict, Any


# -------------------------
# Enumerations (stable IDs)
# -------------------------

class Occupancy(IntEnum):
    AIR = 0
    FLUID = 1
    REPLACEABLE = 2
    SOLID = 3


class MaterialClass(IntEnum):
    STONE = 0
    DIRT = 1
    SNOW = 2
    PLANT = 3
    WATER = 4
    LAVA = 5
    GENERIC = 6


class HazardClass(IntEnum):
    NONE = 0
    DROWNING = 1
    BURN = 2


class Breakability(IntEnum):
    UNBREAKABLE = 0
    BREAKABLE = 1


class ToolHint(IntEnum):
    NONE = 0
    PICKAXE = 1
    SHOVEL = 2
    AXE = 3
    SHEARS = 4


# -------------------------
# Token dataclass
# -------------------------

@dataclass
class PerceptionToken:
    """
    One token = one canonicalized voxel (CanonicalCell),
    expressed in agent-relative coordinates.
    """

    # spatial (agent-relative)
    dx: int
    dy: int
    dz: int
    age_s: float

    # geometry
    occupancy: Occupancy
    support: bool
    passable: bool
    head_obstacle: bool
    thin_layer: bool

    # material
    material_class: MaterialClass
    hazard_class: HazardClass
    breakability: Breakability
    tool_hint: ToolHint


# -------------------------
# Deterministic builder
# -------------------------

class TokenBuilder:
    """
    Builds PerceptionTokens from CanonicalCells.

    Assumes CanonicalCell structure:
      {
        "pos": (x,y,z),
        "age_s": float,
        "geom": {
            "occupancy": str,
            "support": bool,
            "passable": bool,
            "head_obstacle": bool,
            "thin_layer": bool
        },
        "material": {
            "class": str,
            "hazard": str,
            "breakability": str,
            "tool_hint": str
        }
      }
    """

    OCCUPANCY_MAP = {
        "air": Occupancy.AIR,
        "fluid": Occupancy.FLUID,
        "replaceable": Occupancy.REPLACEABLE,
        "solid": Occupancy.SOLID,
    }

    MATERIAL_CLASS_MAP = {
        "stone": MaterialClass.STONE,
        "dirt": MaterialClass.DIRT,
        "snow": MaterialClass.SNOW,
        "plant": MaterialClass.PLANT,
        "water": MaterialClass.WATER,
        "lava": MaterialClass.LAVA,
        "generic": MaterialClass.GENERIC,
    }

    HAZARD_CLASS_MAP = {
        "none": HazardClass.NONE,
        "drowning": HazardClass.DROWNING,
        "burn": HazardClass.BURN,
    }

    BREAKABILITY_MAP = {
        "unbreakable": Breakability.UNBREAKABLE,
        "breakable": Breakability.BREAKABLE,
    }

    TOOL_HINT_MAP = {
        "none": ToolHint.NONE,
        "pickaxe": ToolHint.PICKAXE,
        "shovel": ToolHint.SHOVEL,
        "axe": ToolHint.AXE,
        "shears": ToolHint.SHEARS,
    }

    @classmethod
    def from_canonical_cell(
        cls,
        canonical_cell: Dict[str, Any],
        agent_pos: Tuple[int, int, int],
    ) -> PerceptionToken:
        """
        Convert a CanonicalCell into a PerceptionToken.
        """

        x, y, z = canonical_cell["pos"]
        ax, ay, az = agent_pos

        geom = canonical_cell["geom"]
        material = canonical_cell["material"]

        return PerceptionToken(
            # spatial
            dx=x - ax,
            dy=y - ay,
            dz=z - az,
            age_s=float(canonical_cell["age_s"]),

            # geometry
            occupancy=cls.OCCUPANCY_MAP[geom["occupancy"]],
            support=bool(geom["support"]),
            passable=bool(geom["passable"]),
            head_obstacle=bool(geom["head_obstacle"]),
            thin_layer=bool(geom["thin_layer"]),

            # material
            material_class=cls.MATERIAL_CLASS_MAP[material["class"]],
            hazard_class=cls.HAZARD_CLASS_MAP[material["hazard"]],
            breakability=cls.BREAKABILITY_MAP[material["breakability"]],
            tool_hint=cls.TOOL_HINT_MAP[material["tool_hint"]],
        )
