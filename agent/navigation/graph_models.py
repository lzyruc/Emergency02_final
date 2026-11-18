"""Small public data types shared by logical and SLAM graph navigation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


GRAPH_SCHEMA_VERSION = 1
GRAPH_KIND = "xrocs.navigation_graph"


class NavigationGraphError(RuntimeError):
    """Raised when graph loading, planning, state checks, or execution fails."""


class StateStatus(str, Enum):
    """Minimal runtime state of the graph navigator."""

    AT_NODE = "at_node"
    IN_TRANSIT = "in_transit"
    FAILED = "failed"


@dataclass(frozen=True)
class Node:
    """A named point in the logical coordinate system."""

    name: str
    pose: Mapping[str, float]


@dataclass(frozen=True)
class MapPoint:
    """A named point expressed in the active SLAM map frame."""

    name: str
    pose: Mapping[str, float]
