"""SLAM-map access and rigid planar alignment for logical graph points."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional

from .data.motion_log import map_pose_from_response, normalize_degrees, normalize_radians
from .graph_models import MapPoint, NavigationGraphError, Node


class MapSLAM:
    """Own the fixed logical-to-map transform and thin chassis SLAM access."""

    def __init__(
        self,
        name: str,
        points: Mapping[str, Node],
        logical_initial_pose: Mapping[str, float],
        initial_map_pose: Mapping[str, float],
        controller_getter: Callable[[], Any],
    ) -> None:
        self.name = name
        self.points = dict(points)
        self._controller_getter = controller_getter
        self._logical_initial_pose = self._validate_logical_pose(logical_initial_pose)
        self._initial_map_pose = self._validate_map_pose(initial_map_pose)
        logical_theta = math.radians(self._logical_initial_pose["theta_deg"])
        self._rotation_rad = normalize_radians(self._initial_map_pose["yaw_rad"] - logical_theta)
        cos_rotation = math.cos(self._rotation_rad)
        sin_rotation = math.sin(self._rotation_rad)
        logical_x = self._logical_initial_pose["x_m"]
        logical_y = self._logical_initial_pose["y_m"]
        self._translation_x_m = (
            self._initial_map_pose["x_m"] - cos_rotation * logical_x + sin_rotation * logical_y
        )
        self._translation_y_m = (
            self._initial_map_pose["y_m"] - sin_rotation * logical_x - cos_rotation * logical_y
        )

    @staticmethod
    def _validate_logical_pose(pose: Mapping[str, float]) -> Dict[str, float]:
        try:
            result = {
                "x_m": float(pose["x_m"]),
                "y_m": float(pose["y_m"]),
                "theta_deg": normalize_degrees(float(pose["theta_deg"])),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise NavigationGraphError(f"invalid logical initial pose: {pose!r}") from exc
        if not all(math.isfinite(value) for value in result.values()):
            raise NavigationGraphError(f"non-finite logical initial pose: {pose!r}")
        return result

    @staticmethod
    def _validate_map_pose(pose: Mapping[str, float]) -> Dict[str, float]:
        try:
            result = {
                "x_m": float(pose["x_m"]),
                "y_m": float(pose["y_m"]),
                "yaw_rad": normalize_radians(float(pose["yaw_rad"])),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise NavigationGraphError(f"invalid initial map pose: {pose!r}") from exc
        if not all(math.isfinite(value) for value in result.values()):
            raise NavigationGraphError(f"non-finite initial map pose: {pose!r}")
        return result

    def _controller(self) -> Any:
        return self._controller_getter()

    def getPoint(self, name: str) -> MapPoint:
        """Return a logical graph point transformed into the SLAM map frame."""
        if name not in self.points:
            raise NavigationGraphError(f"map point does not exist: {name}")
        return MapPoint(name=name, pose=self.logicalPoseToMap(self.points[name].pose))

    def logicalPoseToMap(self, pose: Mapping[str, float]) -> Dict[str, float]:
        """Transform a logical planar pose into the fixed SLAM map frame."""
        logical = self._validate_logical_pose(pose)
        cos_rotation = math.cos(self._rotation_rad)
        sin_rotation = math.sin(self._rotation_rad)
        return {
            "x_m": self._translation_x_m + cos_rotation * logical["x_m"] - sin_rotation * logical["y_m"],
            "y_m": self._translation_y_m + sin_rotation * logical["x_m"] + cos_rotation * logical["y_m"],
            "yaw_rad": normalize_radians(
                self._rotation_rad + math.radians(logical["theta_deg"])
            ),
        }

    def mapPoseToLogical(self, pose: Mapping[str, float]) -> Dict[str, float]:
        """Transform a SLAM map pose back into the logical frame."""
        map_pose = self._validate_map_pose(pose)
        dx = map_pose["x_m"] - self._translation_x_m
        dy = map_pose["y_m"] - self._translation_y_m
        cos_rotation = math.cos(self._rotation_rad)
        sin_rotation = math.sin(self._rotation_rad)
        return {
            "x_m": cos_rotation * dx + sin_rotation * dy,
            "y_m": -sin_rotation * dx + cos_rotation * dy,
            "theta_deg": normalize_degrees(
                math.degrees(normalize_radians(map_pose["yaw_rad"] - self._rotation_rad))
            ),
        }

    def getCurrentPose(self) -> Dict[str, float]:
        """Return the chassis' current pose in the SLAM map frame."""
        response = self._controller().get_pose()
        pose = map_pose_from_response(response)
        if "confidence" in response:
            try:
                confidence = float(response["confidence"])
            except (TypeError, ValueError) as exc:
                raise NavigationGraphError(
                    f"invalid localization confidence: {response['confidence']!r}"
                ) from exc
            if math.isfinite(confidence):
                pose["confidence"] = confidence
        return pose

    def getCurrentMap(self) -> str:
        """Query and return the current chassis SLAM map name."""
        response = self._controller().get_current_map()
        if not isinstance(response, Mapping):
            raise NavigationGraphError(f"invalid current map response: {response!r}")
        for key in ("map_name", "name"):
            if response.get(key):
                return str(response[key])
        raise NavigationGraphError(f"current map response has no map name: {response!r}")

    def requireCurrentMap(self) -> bool:
        """Warn about a different active map without blocking competition motion."""
        current = self.getCurrentMap()
        if current != self.name:
            print(
                "[NavigationGraph] warning: current chassis map is "
                f"{current!r}, configured map is {self.name!r}; "
                "continuing with the configured logical-to-SLAM alignment",
                flush=True,
            )
            return False
        return True

    def loadMap(self, name: Optional[str] = None) -> Any:
        """Ask the chassis to activate an existing internal map for laser localization."""
        target = name or self.name
        if target != self.name:
            raise NavigationGraphError(
                f"map {target!r} has no logical alignment; expected {self.name!r}"
            )
        return self._controller().start_localization(target, slam_type=1)
