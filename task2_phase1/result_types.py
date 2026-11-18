"""Shared Phase 1 result data types."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ObjectResult:
    """One detected object in camera coordinates."""

    id: int
    cls: str
    confidence: float
    position_cam: list
    principal_axes: list
    principal_extents: list
    height_above_ground: float
    footprint_area: float
    n_points: int
    solid_ratio: float
    grasp_annotations: list
    in_channel: Any = None
    debug: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class FrameResult:
    """Complete recognition result for one RGB-D frame."""

    objects: list
    ground_plane: tuple
    n_raw_clusters: int
    n_kept: int
    debug: dict = field(default_factory=dict)
