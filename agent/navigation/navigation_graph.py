"""Logical navigation graph built from a recorded move route."""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data.motion_log import (
    advance_pose,
    normalize_degrees,
    normalize_radians,
    read_json,
    utc_now,
    validate_route,
    write_json_atomic,
)
from .graph_models import (
    GRAPH_KIND,
    GRAPH_SCHEMA_VERSION,
    NavigationGraphError,
    Node,
    StateStatus,
)
from .knewbot_chassis_controller_http import KnewbotChassisController
from .map_slam import MapSLAM
from .rotation_control import rotate_by_with_feedback
from .scene_adapter import convert_scene


NAVIGATION_MODES = {"move_by", "track"}
DEFAULT_YAW_EPSILON_DEG = 5.0
MOVE_BY_RECOVERY_POSITION_EPSILON_M = 0.15
MOVE_BY_RECOVERY_YAW_EPSILON_DEG = 5.0
NODE_RECOVERY_POSITION_LIMIT_M = 0.30
NODE_RECOVERY_YAW_LIMIT_DEG = 15.0
MOVE_BY_MIN_DISTANCE_MM = 20
MOVE_BY_MAX_DISTANCE_MM = 3000
MOVE_BY_MIN_DISTANCE_M = MOVE_BY_MIN_DISTANCE_MM / 1000.0
MOVE_BY_MAX_DISTANCE_M = MOVE_BY_MAX_DISTANCE_MM / 1000.0
PATH_COLLINEAR_EPSILON_DEG = 1.0
POSE_RECOVERY_ROUNDS = 4
POSE_RECOVERY_POSITION_EPSILON_M = 0.10
POSE_RECOVERY_YAW_EPSILON_DEG = 2.0
_ACTION_ID_PATTERN = re.compile(r"\bAction\s+([0-9A-Za-z-]+)")


class NavigationGraph:
    """Maintain logical topology, runtime state, and graph navigation."""

    def __init__(
        self,
        route_path: Path,
        controller: Optional[Any] = None,
        initial_node: Optional[str] = None,
        navigation_mode: str = "move_by",
        linear_speed: float = 0.0,
        angular_speed: float = 0.0,
        compress_path: bool = True,
        lateral_translation_headings: Optional[Mapping[str, float]] = None,
        rebuild_graph: bool = False,
    ) -> None:
        if navigation_mode not in NAVIGATION_MODES:
            raise ValueError(
                f"navigation_mode must be one of {sorted(NAVIGATION_MODES)}, got {navigation_mode!r}"
        )
        source_path = Path(route_path).expanduser().resolve()
        source_document = read_json(source_path)
        scene_input = (
            "robots" in source_document or "connections" in source_document
        )
        if scene_input:
            self.route_path, self.graph_path = convert_scene(
                source_path,
                initial_node=initial_node,
            )
            self._route = read_json(self.route_path)
        else:
            self.route_path = source_path
            self.graph_path = self.route_path.with_name(
                f"{self.route_path.stem}.graph.json"
            )
            self._route = source_document
        validate_route(self._route)
        scene_derived_route = isinstance(self._route.get("scene_editor"), Mapping)
        self._nodes: Dict[str, Node] = {}
        self._node_order: List[str] = []
        self._edges: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        self._controller = controller
        self._owns_controller = False
        self.navigation_mode = navigation_mode
        self.compress_path = bool(compress_path)
        self.lateral_translation_headings = {
            str(prefix): normalize_degrees(float(heading))
            for prefix, heading in (lateral_translation_headings or {}).items()
        }
        if any(
            not prefix or not math.isfinite(heading)
            for prefix, heading in self.lateral_translation_headings.items()
        ):
            raise ValueError(
                "lateral_translation_headings requires non-empty prefixes "
                "and finite headings"
            )
        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        if not math.isfinite(self.linear_speed) or self.linear_speed < 0:
            raise ValueError("linear_speed must be a non-negative finite value")
        if not math.isfinite(self.angular_speed) or self.angular_speed < 0:
            raise ValueError("angular_speed must be a non-negative finite value")
        self._last_failure: Optional[Dict[str, Any]] = None
        self._last_warnings: List[Dict[str, Any]] = []
        self._recovery_snapshot: Optional[Dict[str, Any]] = None
        self._adjustment_origin_recorded = False

        if self.graph_path.is_file() and (scene_input or not rebuild_graph):
            self._load_graph(self.graph_path)
            graph_status = "generated" if scene_input else "loaded"
        else:
            if scene_derived_route:
                raise NavigationGraphError(
                    "scene-derived route graph must be regenerated from the source "
                    "scene connections; sequential route rebuild is disabled"
                )
            self._build_from_route()
            self.save()
            graph_status = "rebuilt"

        if not self._node_order:
            raise NavigationGraphError("navigation graph contains no nodes")
        start_name = initial_node or self._node_order[0]
        if start_name not in self._nodes:
            raise NavigationGraphError(f"initial node does not exist: {start_name}")
        self._state = StateStatus.AT_NODE
        self._current_node: Optional[Node] = self._nodes[start_name]
        self._logical_pose: Dict[str, float] = dict(self._current_node.pose)
        logical_initial_pose, initial_map_pose = self._alignment_poses_from_route()
        map_name = self._map_name_from_route()
        if self.navigation_mode == "track" and not map_name:
            raise NavigationGraphError(
                f"{self.navigation_mode} mode requires "
                "map_context.current_map_response.map_name in the route JSON"
            )
        self._map_slam = MapSLAM(
            map_name,
            self._nodes,
            logical_initial_pose,
            initial_map_pose,
            self._ensure_controller,
        )
        edge_count = sum(len(neighbors) for neighbors in self._edges.values())
        print(
            "[NavigationGraph] ready "
            f"mode={self.navigation_mode} node={start_name} "
            f"nodes={len(self._nodes)} edges={edge_count} "
            f"map={map_name or '-'} graph={graph_status} "
            f"route={self.route_path.name}",
            flush=True,
        )

    @property
    def nodes(self) -> Mapping[str, Node]:
        """Return immutable logical nodes keyed by name."""
        return self._nodes

    @property
    def node_order(self) -> Sequence[str]:
        """Return nodes in their original route order."""
        return tuple(self._node_order)

    @property
    def edges(self) -> Mapping[str, Mapping[str, List[Dict[str, float]]]]:
        """Return directed adjacency and action lists."""
        return self._edges

    @staticmethod
    def _route_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _map_name_from_route(self) -> str:
        response = self._route.get("map_context", {}).get("current_map_response", {})
        if isinstance(response, Mapping):
            for key in ("map_name", "name"):
                if response.get(key):
                    return str(response[key])
        return ""

    def _alignment_poses_from_route(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        logical = (
            self._route.get("coordinate_systems", {})
            .get("logical", {})
            .get("initial_pose", {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0})
        )
        map_pose = self._route.get("map_context", {}).get("initial_map_pose")
        if not isinstance(map_pose, Mapping):
            if self.navigation_mode == "track":
                raise NavigationGraphError(
                    f"{self.navigation_mode} mode requires "
                    "map_context.initial_map_pose in the route JSON"
                )
            map_pose = {
                "x_m": float(logical["x_m"]),
                "y_m": float(logical["y_m"]),
                "yaw_rad": math.radians(float(logical["theta_deg"])),
            }
        return dict(logical), dict(map_pose)

    def _build_from_route(self) -> None:
        points = self._route.get("points", [])
        for point in points:
            name = str(point["name"])
            if name in self._nodes:
                raise NavigationGraphError(f"duplicate node name: {name}")
            node = Node(name=name, pose=dict(point["logical_pose"]))
            self._nodes[name] = node
            self._node_order.append(name)
            self._edges[name] = {}

        route_actions = self._route.get("actions", [])
        for index in range(len(points) - 1):
            point_a = points[index]
            point_b = points[index + 1]
            start = int(point_a["before_action"])
            stop = int(point_b["before_action"])
            if start > stop or start < 0 or stop > len(route_actions):
                raise NavigationGraphError(
                    f"invalid action range between {point_a['name']} and {point_b['name']}"
                )
            actions = [self._semantic_action(action) for action in route_actions[start:stop]]
            if not actions and self._nodes[point_a["name"]].pose != self._nodes[point_b["name"]].pose:
                actions = [self._direct_action(point_a["name"], point_b["name"])]
            self.createEdge(str(point_a["name"]), str(point_b["name"]), actions, save=False)

    @staticmethod
    def _semantic_action(route_action: Mapping[str, Any]) -> Dict[str, float]:
        if route_action.get("status") == "failed":
            raise NavigationGraphError(f"cannot build graph from failed action {route_action.get('sequence')}")
        requested = route_action["requested"]
        return {
            "direction_deg_relative": normalize_degrees(float(requested["direction_deg_relative"])),
            "distance_m": float(requested["distance_m"]),
            "rotation_deg_ccw": float(requested["rotation_deg_ccw"]),
        }

    @staticmethod
    def _inverse_action(action: Mapping[str, float]) -> Dict[str, float]:
        rotation = float(action["rotation_deg_ccw"])
        return {
            "direction_deg_relative": normalize_degrees(
                float(action["direction_deg_relative"]) + 180.0 - rotation
            ),
            "distance_m": float(action["distance_m"]),
            "rotation_deg_ccw": -rotation,
        }

    @classmethod
    def _inverse_actions(cls, actions: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
        return [cls._inverse_action(action) for action in reversed(actions)]

    def _direct_action(self, from_name: str, to_name: str) -> Dict[str, float]:
        source = self._nodes[from_name].pose
        target = self._nodes[to_name].pose
        dx = float(target["x_m"]) - float(source["x_m"])
        dy = float(target["y_m"]) - float(source["y_m"])
        world_direction = math.degrees(math.atan2(dy, dx)) if dx or dy else float(source["theta_deg"])
        return {
            "direction_deg_relative": normalize_degrees(world_direction - float(source["theta_deg"])),
            "distance_m": math.hypot(dx, dy),
            "rotation_deg_ccw": self._signed_degrees(
                float(target["theta_deg"]) - float(source["theta_deg"])
            ),
        }

    @staticmethod
    def _signed_degrees(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    def createEdge(
        self,
        node_a: str,
        node_b: str,
        actions: Optional[Sequence[Mapping[str, float]]] = None,
        save: bool = True,
    ) -> None:
        """Create or replace a bidirectional edge."""
        if node_a == node_b:
            raise NavigationGraphError("an edge requires two different nodes")
        if node_a not in self._nodes or node_b not in self._nodes:
            raise NavigationGraphError(f"unknown nodes: {node_a}, {node_b}")
        forward = [dict(action) for action in actions] if actions is not None else [self._direct_action(node_a, node_b)]
        self._edges.setdefault(node_a, {})[node_b] = forward
        self._edges.setdefault(node_b, {})[node_a] = self._inverse_actions(forward)
        if save:
            self.save()

    def deleteEdge(self, node_a: str, node_b: str, save: bool = True) -> None:
        """Delete both directed traversals of one edge."""
        removed = False
        if node_b in self._edges.get(node_a, {}):
            del self._edges[node_a][node_b]
            removed = True
        if node_a in self._edges.get(node_b, {}):
            del self._edges[node_b][node_a]
            removed = True
        if not removed:
            raise NavigationGraphError(f"edge does not exist: {node_a} <-> {node_b}")
        if save:
            self.save()

    def _load_graph(self, path: Path) -> None:
        document = read_json(path)
        if document.get("schema_version") != GRAPH_SCHEMA_VERSION or document.get("kind") != GRAPH_KIND:
            raise NavigationGraphError(f"unsupported graph file: {path}")
        for raw_node in document.get("nodes", []):
            node = Node(name=str(raw_node["name"]), pose=dict(raw_node["pose"]))
            self._nodes[node.name] = node
            self._node_order.append(node.name)
            self._edges[node.name] = {}
        for raw_edge in document.get("edges", []):
            source = str(raw_edge["from"])
            target = str(raw_edge["to"])
            if source not in self._nodes or target not in self._nodes:
                raise NavigationGraphError(f"edge references an unknown node: {source} -> {target}")
            self._edges[source][target] = [dict(action) for action in raw_edge.get("actions", [])]

    def _graph_document(self) -> Dict[str, Any]:
        edges = []
        for source in self._node_order:
            for target, actions in self._edges.get(source, {}).items():
                edges.append({"from": source, "to": target, "actions": actions})
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "kind": GRAPH_KIND,
            "created_from": str(self.route_path),
            "source_route_sha256": self._route_hash(self.route_path),
            "updated_at": utc_now(),
            "map": {
                "class": "MapSLAM",
                "name": self._map_name_from_route(),
                "alignment": {
                    "logical_initial_pose": self._alignment_poses_from_route()[0],
                    "initial_map_pose": self._alignment_poses_from_route()[1],
                },
            },
            "nodes": [
                {"name": name, "pose": dict(self._nodes[name].pose)} for name in self._node_order
            ],
            "edges": edges,
        }

    def save(self) -> Path:
        """Atomically save the editable graph JSON."""
        write_json_atomic(self.graph_path, self._graph_document())
        return self.graph_path

    @staticmethod
    def _edge_weight(actions: Sequence[Mapping[str, float]]) -> float:
        distance = sum(abs(float(action["distance_m"])) for action in actions)
        return distance

    def shortestPath(self, target: str) -> List[str]:
        """Return a distance-shortest path, preferring fewer edges on ties."""
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            raise NavigationGraphError("shortestPath requires AT_NODE state")
        if target not in self._nodes:
            raise NavigationGraphError(f"target node does not exist: {target}")
        start = self._current_node.name
        queue: List[Tuple[float, int, str, List[str]]] = [(0.0, 0, start, [start])]
        best: Dict[str, Tuple[float, int]] = {start: (0.0, 0)}
        while queue:
            distance, hops, node, path = heapq.heappop(queue)
            if (distance, hops) != best.get(node):
                continue
            if node == target:
                return path
            for neighbor, actions in self._edges.get(node, {}).items():
                candidate = (distance + self._edge_weight(actions), hops + 1)
                if neighbor not in best or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate[0], candidate[1], neighbor, path + [neighbor]))
        raise NavigationGraphError(f"no path from {start} to {target}")

    def executionPath(self, target: str) -> List[str]:
        """Return chassis waypoints after safe geometric path compression.

        Only consecutive, same-direction collinear segments are merged.  The
        original graph path still defines which corridor is selected.
        """
        path = self.shortestPath(target)
        if not self.compress_path or self.navigation_mode == "track":
            # Graph edges are safety constraints by default. Skipping a
            # collinear node may cut across a real obstacle that is not
            # represented by point geometry alone. Track mode never merges.
            return path
        scene_editor = self._route.get("scene_editor")
        uses_scene_geometry_actions = isinstance(
            scene_editor, Mapping
        ) and not bool(scene_editor.get("preserves_connection_actions"))
        if self.navigation_mode == "move_by" and not uses_scene_geometry_actions:
            return path
        if len(path) < 3:
            return path
        result = [path[0]]
        for index in range(1, len(path) - 1):
            current_name = path[index]
            previous = self._nodes[path[index - 1]].pose
            current = self._nodes[current_name].pose
            following = self._nodes[path[index + 1]].pose
            first_dx = float(current["x_m"]) - float(previous["x_m"])
            first_dy = float(current["y_m"]) - float(previous["y_m"])
            second_dx = float(following["x_m"]) - float(current["x_m"])
            second_dy = float(following["y_m"]) - float(current["y_m"])
            first_length = math.hypot(first_dx, first_dy)
            second_length = math.hypot(second_dx, second_dy)
            if not first_length or not second_length:
                result.append(path[index])
                continue
            dot = (first_dx * second_dx + first_dy * second_dy) / (
                first_length * second_length
            )
            cross = (first_dx * second_dy - first_dy * second_dx) / (
                first_length * second_length
            )
            turn_deg = abs(math.degrees(math.atan2(cross, dot)))
            if dot > 0 and turn_deg <= PATH_COLLINEAR_EPSILON_DEG:
                anchor = self._nodes[result[-1]].pose
                merged_distance = math.hypot(
                    float(following["x_m"]) - float(anchor["x_m"]),
                    float(following["y_m"]) - float(anchor["y_m"]),
                )
                if merged_distance <= MOVE_BY_MAX_DISTANCE_M:
                    continue
            result.append(path[index])
        result.append(path[-1])
        return result

    def getState(self) -> Tuple[StateStatus, Optional[Node]]:
        """Return state status and the current immutable Node only when at a node."""
        return self._state, self._current_node if self._state is StateStatus.AT_NODE else None

    def _adopt_nearest_node_from_measured_pose(self, operation: str) -> Node:
        """Recover a semantic anchor from live SLAM without enforcing node tolerance."""
        previous_state = self._state
        current_map_pose = self._map_slam.getCurrentPose()
        current_logical_pose = self._map_slam.mapPoseToLogical(current_map_pose)
        anchor = min(
            self._nodes.values(),
            key=lambda node: math.hypot(
                float(node.pose["x_m"]) - current_logical_pose["x_m"],
                float(node.pose["y_m"]) - current_logical_pose["y_m"],
            ),
        )
        self._state = StateStatus.AT_NODE
        self._current_node = anchor
        self._logical_pose = current_logical_pose
        self._recovery_snapshot = None
        self._adjustment_origin_recorded = False
        warning = {
            "category": "semantic_node_adopted_without_tolerance",
            "operation": operation,
            "previous_state": previous_state.value,
            "node": anchor.name,
            "map_pose": dict(current_map_pose),
            "logical_pose": dict(current_logical_pose),
        }
        self._last_warnings.append(warning)
        print(
            f"[NavigationGraph] {operation} continued from "
            f"state={previous_state.value}; adopted nearest node={anchor.name} "
            "without tolerance check",
            flush=True,
        )
        return anchor

    def getPose(self) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Return the expected logical pose and its aligned SLAM map pose."""
        if self._state is not StateStatus.AT_NODE:
            return None, None
        logical = dict(self._logical_pose)
        return logical, self._map_slam.logicalPoseToMap(logical)

    def getNodeMapPose(self, node: str) -> Dict[str, float]:
        """Return one named node transformed into the active SLAM map frame."""
        if node not in self._nodes:
            raise NavigationGraphError(f"node does not exist: {node}")
        return dict(self._map_slam.getPoint(node).pose)

    def getLastFailure(self) -> Optional[Dict[str, Any]]:
        """Return diagnostics captured for the most recent failed graph move."""
        return dict(self._last_failure) if self._last_failure is not None else None

    def getLastWarnings(self) -> List[Dict[str, Any]]:
        """Return non-fatal chassis anomalies accepted by geometric verification."""
        return [dict(warning) for warning in self._last_warnings]

    def getStateError(self, epi: float = 0.15) -> Optional[Dict[str, Any]]:
        """Return live pose errors and tolerances for the current graph node."""
        if not math.isfinite(epi) or epi < 0:
            raise ValueError("epi must be a non-negative finite distance in metres")
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            return None
        actual = self._map_slam.getCurrentPose()
        expected = self._map_slam.logicalPoseToMap(self._logical_pose)
        position_error = math.hypot(
            actual["x_m"] - expected["x_m"],
            actual["y_m"] - expected["y_m"],
        )
        yaw_error_deg = abs(
            math.degrees(normalize_radians(actual["yaw_rad"] - expected["yaw_rad"]))
        )
        return {
            "node": self._current_node.name,
            "actual_map_pose": dict(actual),
            "expected_map_pose": dict(expected),
            "position_error_m": position_error,
            "yaw_error_deg": yaw_error_deg,
            "position_tolerance_m": epi,
            "yaw_tolerance_deg": DEFAULT_YAW_EPSILON_DEG,
            "within_tolerance": (
                position_error <= epi
                and yaw_error_deg <= DEFAULT_YAW_EPSILON_DEG
            ),
        }

    def checkState(self, epi: float = 0.15) -> bool:
        """Compare the live SLAM pose with the current expected logical pose."""
        diagnostics = self.getStateError(epi)
        return (
            diagnostics is not None
            and bool(diagnostics["within_tolerance"])
        )

    def adoptMeasuredNodeState(
        self,
        node: str,
        *,
        position_epsilon_m: float = 0.05,
        yaw_epsilon_deg: float = 1.0,
    ) -> Dict[str, float]:
        """Adopt a node after an externally executed measured-pose correction."""
        if node not in self._nodes:
            raise NavigationGraphError(f"node does not exist: {node}")
        if (
            not math.isfinite(position_epsilon_m)
            or position_epsilon_m < 0.0
            or not math.isfinite(yaw_epsilon_deg)
            or yaw_epsilon_deg < 0.0
        ):
            raise ValueError("measured-state tolerances must be non-negative and finite")
        actual = self._map_slam.getCurrentPose()
        expected = self._map_slam.getPoint(node).pose
        position_error = math.hypot(
            actual["x_m"] - expected["x_m"],
            actual["y_m"] - expected["y_m"],
        )
        yaw_error_deg = abs(
            math.degrees(
                normalize_radians(actual["yaw_rad"] - expected["yaw_rad"])
            )
        )
        if (
            position_error > position_epsilon_m
            or yaw_error_deg > yaw_epsilon_deg
        ):
            raise NavigationGraphError(
                f"measured pose is outside node {node}: "
                f"position_error={position_error:.3f} m, "
                f"yaw_error={yaw_error_deg:.2f} deg"
            )
        target = self._nodes[node]
        self._state = StateStatus.AT_NODE
        self._current_node = target
        self._logical_pose = dict(target.pose)
        self._recovery_snapshot = None
        self._adjustment_origin_recorded = False
        return {
            "position_error_m": position_error,
            "yaw_error_deg": yaw_error_deg,
        }

    def _ensure_controller(self) -> Any:
        if self._controller is None:
            self._controller = KnewbotChassisController()
            self._owns_controller = True
        return self._controller

    @staticmethod
    def _cancel_timed_out_action(
        exc: BaseException,
        controller: Any,
    ) -> None:
        if getattr(getattr(exc, "code", None), "value", None) != 1001:
            return
        match = _ACTION_ID_PATTERN.search(str(exc))
        if match is None:
            raise NavigationGraphError(
                "timed out chassis action has no action_id"
            )
        controller.cancel_action([match.group(1)])
        print(
            f"[NavigationGraph] cancelled timed out action {match.group(1)}",
            flush=True,
        )

    def _retry_failed_translation(
        self,
        exc: BaseException,
        target: Mapping[str, float],
        controller: Any,
        linear_speed: float,
        angular_speed: float,
    ) -> None:
        self._cancel_timed_out_action(exc, controller)
        current = self._map_slam.getCurrentPose()
        dx = float(target["x_m"]) - current["x_m"]
        dy = float(target["y_m"]) - current["y_m"]
        remaining_distance = math.hypot(dx, dy)
        retry_distance_mm = min(
            round(remaining_distance * 1000),
            MOVE_BY_MAX_DISTANCE_MM,
        )
        retry_distance = retry_distance_mm / 1000.0
        print(
            "[NavigationGraph] retry translation after action interruption "
            f"remaining={round(remaining_distance * 1000)} mm "
            f"command={retry_distance_mm if retry_distance_mm >= MOVE_BY_MIN_DISTANCE_MM else 0} mm",
            flush=True,
        )
        if retry_distance_mm >= MOVE_BY_MIN_DISTANCE_MM:
            try:
                controller.move_by(
                    direction=normalize_radians(
                        math.atan2(dy, dx) - current["yaw_rad"]
                    ),
                    distance=retry_distance_mm,
                    move_mode=0,
                    v=linear_speed,
                    w=angular_speed,
                    wait=True,
                )
            except Exception as retry_exc:
                self._cancel_timed_out_action(retry_exc, controller)
                raise
        self._last_warnings.append(
            {
                "category": "move_by_translation_action_interruption_retried",
                "error_code": getattr(getattr(exc, "code", None), "value", None),
                "error": repr(exc),
                "target_map_pose": dict(target),
                "map_pose_before_retry": current,
                "remaining_distance_m": remaining_distance,
                "retry_distance_m": retry_distance,
                "retry_skipped": retry_distance_mm < MOVE_BY_MIN_DISTANCE_MM,
            }
        )

    def _execute_action(
        self,
        action: Mapping[str, float],
        *,
        linear_speed: Optional[float] = None,
        angular_speed: Optional[float] = None,
        rotate_before_translate: bool = False,
        pre_translation_yaw_rad: Optional[float] = None,
        world_travel_yaw_rad: Optional[float] = None,
        retry_rotation_action_failed: bool = False,
        suppress_repeated_rotation_action_failed: bool = False,
        retry_translation_action_failed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        controller = self._ensure_controller()
        command_linear_speed = (
            self.linear_speed if linear_speed is None else float(linear_speed)
        )
        command_angular_speed = (
            self.angular_speed if angular_speed is None else float(angular_speed)
        )
        distance_m = float(action["distance_m"])
        if distance_m < 0.0:
            raise NavigationGraphError(
                f"move_by translation distance must be non-negative, got {distance_m}"
            )
        distance_mm = round(distance_m * 1000)
        direction = normalize_radians(
            math.radians(float(action["direction_deg_relative"]))
        )
        rotation_deg = float(action["rotation_deg_ccw"])
        translation_target: Optional[Dict[str, float]] = None
        if pre_translation_yaw_rad is not None:
            before_rotation = self._map_slam.getCurrentPose()
            yaw_error = normalize_radians(
                float(pre_translation_yaw_rad) - before_rotation["yaw_rad"]
            )
            if abs(yaw_error) >= math.radians(1.0):
                rotate_by_with_feedback(
                    controller,
                    yaw_error,
                    angular_speed=command_angular_speed,
                    initial_map_pose=before_rotation,
                    retry_action_failed=True,
                )
            after_rotation = self._map_slam.getCurrentPose()
            if world_travel_yaw_rad is not None:
                direction = normalize_radians(
                    float(world_travel_yaw_rad) - after_rotation["yaw_rad"]
                )
            rotation_deg = 0.0
        elif rotate_before_translate and abs(rotation_deg) >= 1.0:
            before_rotation = self._map_slam.getCurrentPose()
            world_travel_yaw = before_rotation["yaw_rad"] + direction
            rotate_by_with_feedback(
                controller,
                math.radians(rotation_deg),
                angular_speed=command_angular_speed,
                initial_map_pose=before_rotation,
                retry_action_failed=True,
            )
            after_rotation = self._map_slam.getCurrentPose()
            direction = normalize_radians(world_travel_yaw - after_rotation["yaw_rad"])
            rotation_deg = 0.0
        if distance_m > 0.0 and distance_mm < MOVE_BY_MIN_DISTANCE_MM:
            print(
                "[NavigationGraph] skip translation "
                f"{distance_mm} mm (<{MOVE_BY_MIN_DISTANCE_MM} mm)",
                flush=True,
            )
            self._last_warnings.append(
                {
                    "category": "move_by_translation_below_minimum_skipped",
                    "requested_distance_m": distance_m,
                    "rounded_distance_mm": distance_mm,
                    "minimum_distance_mm": MOVE_BY_MIN_DISTANCE_MM,
                }
            )
        if distance_mm >= MOVE_BY_MIN_DISTANCE_MM:
            if retry_translation_action_failed:
                translation_start = self._map_slam.getCurrentPose()
                translation_yaw = translation_start["yaw_rad"] + direction
                translation_target = {
                    "x_m": translation_start["x_m"]
                    + distance_m * math.cos(translation_yaw),
                    "y_m": translation_start["y_m"]
                    + distance_m * math.sin(translation_yaw),
                }
            if distance_mm <= MOVE_BY_MAX_DISTANCE_MM:
                try:
                    controller.move_by(
                        direction=direction,
                        distance=distance_mm,
                        move_mode=0,
                        v=command_linear_speed,
                        w=command_angular_speed,
                        wait=True,
                    )
                except Exception as exc:
                    if (
                        not retry_translation_action_failed
                        or getattr(getattr(exc, "code", None), "value", None)
                        not in {1001, 2001}
                        or translation_target is None
                    ):
                        raise
                    self._retry_failed_translation(
                        exc,
                        translation_target,
                        controller,
                        command_linear_speed,
                        command_angular_speed,
                    )
            else:
                start = self._map_slam.getCurrentPose()
                travel_yaw = start["yaw_rad"] + direction
                target_x = start["x_m"] + distance_m * math.cos(travel_yaw)
                target_y = start["y_m"] + distance_m * math.sin(travel_yaw)
                nominal_chunks = math.ceil(distance_m / MOVE_BY_MAX_DISTANCE_M)
                chunk_reports: List[Dict[str, Any]] = []
                for chunk_index in range(nominal_chunks + 3):
                    current = start if chunk_index == 0 else self._map_slam.getCurrentPose()
                    dx = target_x - current["x_m"]
                    dy = target_y - current["y_m"]
                    remaining_distance = math.hypot(dx, dy)
                    remaining_distance_mm = round(remaining_distance * 1000)
                    if remaining_distance_mm < MOVE_BY_MIN_DISTANCE_MM:
                        break
                    chunks_left = max(1, nominal_chunks - chunk_index)
                    chunk_distance = min(
                        MOVE_BY_MAX_DISTANCE_M,
                        remaining_distance / chunks_left,
                    )
                    chunk_distance_mm = min(
                        MOVE_BY_MAX_DISTANCE_MM,
                        round(chunk_distance * 1000),
                    )
                    if chunk_distance_mm < MOVE_BY_MIN_DISTANCE_MM:
                        break
                    world_direction = math.atan2(dy, dx)
                    corrected_direction = normalize_radians(
                        world_direction - current["yaw_rad"]
                    )
                    try:
                        controller.move_by(
                            direction=corrected_direction,
                            distance=chunk_distance_mm,
                            move_mode=0,
                            v=command_linear_speed,
                            w=command_angular_speed,
                            wait=True,
                        )
                    except Exception as exc:
                        if (
                            not retry_translation_action_failed
                            or getattr(getattr(exc, "code", None), "value", None)
                            not in {1001, 2001}
                        ):
                            raise
                        self._retry_failed_translation(
                            exc,
                            {"x_m": target_x, "y_m": target_y},
                            controller,
                            command_linear_speed,
                            command_angular_speed,
                        )
                    chunk_reports.append(
                        {
                            "index": chunk_index,
                            "map_pose_before": current,
                            "direction_rad_relative": corrected_direction,
                            "distance_m": chunk_distance_mm / 1000.0,
                            "remaining_before_m": remaining_distance,
                        }
                    )
                final_translation_pose = self._map_slam.getCurrentPose()
                final_error = math.hypot(
                    final_translation_pose["x_m"] - target_x,
                    final_translation_pose["y_m"] - target_y,
                )
                self._last_warnings.append(
                    {
                        "category": "move_by_translation_chunked_with_pose_feedback",
                        "requested_distance_m": distance_m,
                        "target_map_pose": {
                            "x_m": target_x,
                            "y_m": target_y,
                            "yaw_rad": start["yaw_rad"],
                        },
                        "chunks": chunk_reports,
                        "map_pose_after": final_translation_pose,
                        "target_position_error_m": final_error,
                    }
                )
        rotation_report = None
        if rotation_deg:
            rotation_report = rotate_by_with_feedback(
                controller,
                math.radians(rotation_deg),
                angular_speed=command_angular_speed,
                retry_action_failed=retry_rotation_action_failed,
                suppress_repeated_action_failed=(
                    suppress_repeated_rotation_action_failed
                ),
            )
            if rotation_report["action_failed_count"]:
                self._last_warnings.append(
                    {
                        "category": "rotation_action_failed_retried",
                        **rotation_report,
                    }
                )
        return rotation_report

    def _lateral_heading_for_segment(
        self, source_name: str, target_name: str
    ) -> Optional[float]:
        """Return a forced logical heading for one same-row shelf traversal."""
        source_pose = self._nodes[source_name].pose
        target_pose = self._nodes[target_name].pose
        dx = float(target_pose["x_m"]) - float(source_pose["x_m"])
        dy = float(target_pose["y_m"]) - float(source_pose["y_m"])
        if abs(dx) < 1e-9 or abs(dy) > 1e-6:
            return None
        for prefix, heading in self.lateral_translation_headings.items():
            pattern = rf"{re.escape(prefix)}\d+"
            if re.fullmatch(pattern, source_name) and re.fullmatch(
                pattern, target_name
            ):
                return heading
        return None

    @staticmethod
    def _track_pose(pose: Mapping[str, float]) -> Dict[str, float]:
        """Convert an internal map pose to the chassis track API shape."""
        return {
            "x": float(pose["x_m"]),
            "y": float(pose["y_m"]),
            "yaw": normalize_radians(float(pose["yaw_rad"])),
        }

    def _dynamic_track_target(
        self,
        source: Node,
        target: Node,
        current_map_pose: Mapping[str, float],
    ) -> Dict[str, float]:
        """Apply one measured logical edge delta to the live SLAM pose."""
        source_map = self._map_slam.logicalPoseToMap(source.pose)
        target_map = self._map_slam.logicalPoseToMap(target.pose)
        logical_yaw_delta = math.radians(
            self._signed_degrees(
                float(target.pose["theta_deg"]) - float(source.pose["theta_deg"])
            )
        )
        return {
            "x_m": float(current_map_pose["x_m"])
            + float(target_map["x_m"])
            - float(source_map["x_m"]),
            "y_m": float(current_map_pose["y_m"])
            + float(target_map["y_m"])
            - float(source_map["y_m"]),
            "yaw_rad": normalize_radians(
                float(current_map_pose["yaw_rad"]) + logical_yaw_delta
            ),
        }

    def _execute_track_segment(
        self,
        source: Node,
        target: Node,
        current_map_pose: Mapping[str, float],
        target_map_pose: Mapping[str, float],
    ) -> None:
        """Create/update and execute one persistent directed straight track."""
        controller = self._ensure_controller()
        track_name = f"{source.name}-{target.name}"
        start = self._track_pose(current_map_pose)
        end = self._track_pose(target_map_pose)
        waypoints = [
            {"index": 0, **start, "move_mode": 0},
            {"index": 1, **end, "move_mode": 0},
        ]
        tracks_response = controller.get_tracks(map_name=self._map_slam.name)
        tracks = self._track_records(tracks_response)
        existing = next(
            (track for track in tracks if str(track.get("track_name")) == track_name),
            None,
        )
        if existing is not None:
            track_id = int(existing["track_id"])
            controller.update_track(
                track_id=track_id,
                track_name=track_name,
                waypoints=waypoints,
                map_name=self._map_slam.name,
            )
        else:
            used_ids = {
                int(track["track_id"])
                for track in tracks
                if track.get("track_id") is not None
            }
            digest = hashlib.sha256(
                f"{self._map_slam.name}\0{track_name}".encode("utf-8")
            ).digest()
            track_id = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            track_id = max(1, track_id)
            while track_id in used_ids:
                track_id = 1 if track_id == 0x7FFFFFFF else track_id + 1
            controller.create_track(
                track_id=track_id,
                track_name=track_name,
                waypoints=waypoints,
                map_name=self._map_slam.name,
            )
        controller.move_to_locations(
            track_id=track_id,
            start_position=start,
            target_position=end,
            uptrack_enabled=False,
            wait=True,
        )

    @classmethod
    def _track_records(cls, value: Any) -> List[Mapping[str, Any]]:
        """Extract track records from either single-track or list responses."""
        records: List[Mapping[str, Any]] = []
        if isinstance(value, Mapping):
            if value.get("track_id") is not None and value.get("track_name") is not None:
                records.append(value)
            else:
                for nested in value.values():
                    records.extend(cls._track_records(nested))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                records.extend(cls._track_records(nested))
        return records

    def _capture_execution_failure(
        self,
        exc: BaseException,
        source_name: str,
        target: Node,
        final_target: str,
        before: Optional[Mapping[str, float]],
        requested_map_pose: Mapping[str, float],
    ) -> None:
        """Capture best-effort context without implementing bottom-layer recovery."""
        diagnostics: Dict[str, Any] = {
            "exception": str(exc),
            "source_node": source_name,
            "target_node": target.name,
            "final_target_node": final_target,
            "requested_map_pose": dict(requested_map_pose),
        }
        error_code = getattr(getattr(exc, "code", None), "value", None)
        category_by_code = {
            2003: "chassis_action_cancelled",
            1001: "chassis_action_timeout",
            3001: "chassis_communication_error",
            3002: "chassis_communication_error",
        }
        diagnostics["failure_category"] = category_by_code.get(
            error_code, "execution_exception"
        )
        if before is not None:
            diagnostics["map_pose_before"] = dict(before)
        match = _ACTION_ID_PATTERN.search(str(exc))
        if match:
            action_id = match.group(1)
            diagnostics["action_id"] = action_id
            try:
                diagnostics["action_status"] = self._ensure_controller().get_action_status(
                    action_id
                )
            except Exception as status_exc:
                diagnostics["action_status_error"] = repr(status_exc)
        controller = self._ensure_controller()
        for key, method_name in (
            ("robot_health", "get_robot_health"),
            ("localization_status", "get_localization_status"),
            ("current_map", "get_current_map"),
        ):
            method = getattr(controller, method_name, None)
            if method is None:
                continue
            try:
                diagnostics[key] = method()
            except Exception as diagnostic_exc:
                diagnostics[f"{key}_error"] = repr(diagnostic_exc)
        try:
            after = self._map_slam.getCurrentPose()
            diagnostics["map_pose_after"] = after
            measured_translation = None
            measured_yaw_change_deg = None
            if before is not None:
                measured_translation = math.hypot(
                    after["x_m"] - float(before["x_m"]),
                    after["y_m"] - float(before["y_m"]),
                )
                measured_yaw_change_deg = abs(
                    math.degrees(
                        normalize_radians(
                            after["yaw_rad"] - float(before["yaw_rad"])
                        )
                    )
                )
                diagnostics["measured_translation_m"] = measured_translation
                diagnostics["measured_yaw_change_deg"] = measured_yaw_change_deg
            diagnostics["target_position_error_m"] = math.hypot(
                after["x_m"] - float(requested_map_pose["x_m"]),
                after["y_m"] - float(requested_map_pose["y_m"]),
            )
            diagnostics["target_yaw_error_deg"] = abs(
                math.degrees(
                    normalize_radians(
                        after["yaw_rad"] - float(requested_map_pose["yaw_rad"])
                    )
                )
            )
            if error_code == 2001:
                diagnostics["failure_category"] = (
                    "chassis_rejected_before_motion"
                    if measured_translation is not None
                    and measured_translation < 0.02
                    and measured_yaw_change_deg is not None
                    and measured_yaw_change_deg < 1.0
                    else "chassis_failed_during_motion"
                )
        except Exception as pose_exc:
            diagnostics["map_pose_after_error"] = repr(pose_exc)
            diagnostics["diagnostics_status"] = "map_pose_after_unavailable"
        self._last_failure = diagnostics

    def _return_to_node(self, node: Node) -> None:
        point = self._map_slam.getPoint(node.name)
        pose = point.pose
        controller = self._ensure_controller()
        controller.move_to(
            float(pose["x_m"]),
            float(pose["y_m"]),
            float(pose["yaw_rad"]),
            flags=0,
            wait=True,
        )

    def _return_to_node_with_move_by_correction(self, node: Node) -> None:
        """Use SLAM for the long return, then bounded move_by pose correction."""
        target = self._map_slam.getPoint(node.name).pose
        self._return_to_node(node)

        current = self._map_slam.getCurrentPose()
        position_error, yaw_error_deg = self._map_pose_error(current, target)
        if (
            position_error > NODE_RECOVERY_POSITION_LIMIT_M
            or yaw_error_deg > NODE_RECOVERY_YAW_LIMIT_DEG
        ):
            self._last_warnings.append(
                {
                    "category": "recovery_arrival_outside_previous_correction_range",
                    "error": None,
                    "node": node.name,
                    "position_error_m": position_error,
                    "yaw_error_deg": yaw_error_deg,
                }
            )
            print(
                "[NavigationGraph] recovery arrival exceeds the previous "
                f"correction range at {node.name}: position_error="
                f"{position_error:.3f} m, yaw_error={yaw_error_deg:.2f} deg; "
                "continuing with pose correction",
                flush=True,
            )
        self._return_to_map_pose_with_move_by(
            target,
            raise_on_nonconvergence=False,
        )

    def _capture_recovery_snapshot(self, reason: str) -> None:
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            raise NavigationGraphError(
                "a recovery snapshot requires AT_NODE state"
            )
        self._recovery_snapshot = {
            "reason": str(reason),
            "node_name": self._current_node.name,
            "logical_pose": dict(self._logical_pose),
            "map_pose": self._map_slam.getCurrentPose(),
        }

    @staticmethod
    def _map_pose_error(
        actual: Mapping[str, float], target: Mapping[str, float]
    ) -> Tuple[float, float]:
        position_error = math.hypot(
            float(actual["x_m"]) - float(target["x_m"]),
            float(actual["y_m"]) - float(target["y_m"]),
        )
        yaw_error_deg = abs(
            math.degrees(
                normalize_radians(
                    float(actual["yaw_rad"]) - float(target["yaw_rad"])
                )
            )
        )
        return position_error, yaw_error_deg

    def _return_to_map_pose_with_move_by(
        self,
        target: Mapping[str, float],
        *,
        raise_on_nonconvergence: bool = False,
        linear_speed: Optional[float] = None,
        angular_speed: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return to one measured map pose using short feedback move_by actions."""
        for _ in range(POSE_RECOVERY_ROUNDS):
            current = self._map_slam.getCurrentPose()
            dx = float(target["x_m"]) - current["x_m"]
            dy = float(target["y_m"]) - current["y_m"]
            distance_m = math.hypot(dx, dy)
            yaw_error = normalize_radians(
                float(target["yaw_rad"]) - current["yaw_rad"]
            )
            action = {
                "direction_deg_relative": (
                    normalize_degrees(
                        math.degrees(
                            normalize_radians(
                                math.atan2(dy, dx) - current["yaw_rad"]
                            )
                        )
                    )
                    if distance_m >= MOVE_BY_MIN_DISTANCE_M
                    else 0.0
                ),
                "distance_m": (
                    distance_m
                    if distance_m >= MOVE_BY_MIN_DISTANCE_M
                    else 0.0
                ),
                "rotation_deg_ccw": (
                    math.degrees(yaw_error)
                    if abs(math.degrees(yaw_error))
                    >= POSE_RECOVERY_YAW_EPSILON_DEG
                    else 0.0
                ),
            }
            if not action["distance_m"] and not action["rotation_deg_ccw"]:
                return current
            try:
                self._execute_action(
                    action,
                    linear_speed=linear_speed,
                    angular_speed=angular_speed,
                    retry_rotation_action_failed=True,
                    retry_translation_action_failed=bool(action["distance_m"]),
                )
            except Exception as exc:
                if getattr(getattr(exc, "code", None), "value", None) not in {
                    1001,
                    2001,
                }:
                    raise
                measured = self._map_slam.getCurrentPose()
                position_error, yaw_error_deg = self._map_pose_error(
                    measured, target
                )
                if (
                    position_error > POSE_RECOVERY_POSITION_EPSILON_M
                    or yaw_error_deg > POSE_RECOVERY_YAW_EPSILON_DEG
                ):
                    if raise_on_nonconvergence:
                        raise
                    continue
                return measured

        final_pose = self._map_slam.getCurrentPose()
        position_error, yaw_error_deg = self._map_pose_error(final_pose, target)
        if (
            position_error > POSE_RECOVERY_POSITION_EPSILON_M
            or yaw_error_deg > POSE_RECOVERY_YAW_EPSILON_DEG
        ):
            if raise_on_nonconvergence:
                raise NavigationGraphError(
                    "saved-pose recovery did not converge: "
                    f"position_error={position_error:.3f} m, "
                    f"yaw_error={yaw_error_deg:.2f} deg"
                )
            self._last_warnings.append(
                {
                    "category": "move_by_pose_correction_not_converged",
                    "target_map_pose": dict(target),
                    "map_pose_after": final_pose,
                    "position_error_m": position_error,
                    "yaw_error_deg": yaw_error_deg,
                    "correction_rounds": POSE_RECOVERY_ROUNDS,
                }
            )
        return final_pose

    def correctState(self) -> None:
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            self._adopt_nearest_node_from_measured_pose("correctState")

        target = self._map_slam.logicalPoseToMap(self._logical_pose)
        current = self._map_slam.getCurrentPose()
        position_error, yaw_error_deg = self._map_pose_error(current, target)

        if position_error <= 0.03 and yaw_error_deg <= 1.0:
            return

        self._return_to_map_pose_with_move_by(
            target,
            raise_on_nonconvergence=False,
        )

        if not self.checkState(epi=0.03):
            self._last_warnings.append(
                {
                    "category": "state_correction_not_converged",
                    "target_map_pose": dict(target),
                }
            )
            print(
                "[NavigationGraph] state correction did not converge; continuing",
                flush=True,
            )

    def recoverState(
        self,
        node: Optional[str] = None,
        *,
        method: str = "move_to",
    ) -> None:
        """Recover to a named node or the latest measured recovery snapshot."""
        if method not in {"move_by", "move_to"}:
            raise ValueError("recovery method must be 'move_by' or 'move_to'")
        recovery_target = (
            node
            if node is not None
            else ("snapshot" if self._recovery_snapshot is not None else "nearest")
        )
        recovery_method = (
            "move_by"
            if node is None and self._recovery_snapshot is not None
            else method
        )
        print(
            "[NavigationGraph] recoverState "
            f"target={recovery_target} method={recovery_method}",
            flush=True,
        )
        if node is None and self._recovery_snapshot is not None:
            snapshot = self._recovery_snapshot
            target_name = str(snapshot["node_name"])
            if target_name not in self._nodes:
                raise NavigationGraphError(
                    f"recovery snapshot references an unknown node: {target_name}"
                )
            try:
                self._return_to_map_pose_with_move_by(snapshot["map_pose"])
            except Exception:
                self._state = StateStatus.FAILED
                self._current_node = None
                raise
            self._state = StateStatus.AT_NODE
            self._current_node = self._nodes[target_name]
            self._logical_pose = dict(snapshot["logical_pose"])
            self._recovery_snapshot = None
            self._adjustment_origin_recorded = False
            print(
                f"[NavigationGraph] recoverState arrived {target_name}",
                flush=True,
            )
            return

        if node is None:
            current = self._map_slam.getCurrentPose()
            node = min(
                self._node_order,
                key=lambda name: math.hypot(
                    self._map_slam.getPoint(name).pose["x_m"] - current["x_m"],
                    self._map_slam.getPoint(name).pose["y_m"] - current["y_m"],
                ),
            )
        if node not in self._nodes:
            raise NavigationGraphError(f"recovery node does not exist: {node}")
        target = self._nodes[node]
        try:
            if method == "move_by":
                self._return_to_map_pose_with_move_by(
                    self._map_slam.getPoint(target.name).pose
                )
            else:
                self._return_to_node_with_move_by_correction(target)
        except Exception:
            self._state = StateStatus.FAILED
            self._current_node = None
            raise
        self._state = StateStatus.AT_NODE
        self._current_node = target
        self._logical_pose = dict(target.pose)
        try:
            arrived = self.checkState()
        except Exception:
            self._state = StateStatus.FAILED
            self._current_node = None
            raise
        if not arrived:
            diagnostics = self.getStateError()
            self._last_warnings.append(
                {
                    "category": "recovery_arrival_state_check_failed_continued",
                    "node": target.name,
                    "diagnostics": diagnostics,
                }
            )
            print(
                f"[NavigationGraph] recovery arrival check failed at "
                f"{target.name}; continuing at the semantic node",
                flush=True,
            )
        self._recovery_snapshot = None
        self._adjustment_origin_recorded = False
        print(
            f"[NavigationGraph] recoverState arrived {target.name}",
            flush=True,
        )

    def _correct_expected_state(
        self,
        *,
        linear_speed: Optional[float] = None,
        angular_speed: Optional[float] = None,
    ) -> None:
        """Use bounded move_by feedback to align the measured and logical poses."""
        try:
            target = self._map_slam.logicalPoseToMap(self._logical_pose)
            current = self._map_slam.getCurrentPose()
            position_error, yaw_error_deg = self._map_pose_error(current, target)
            if (
                position_error <= POSE_RECOVERY_POSITION_EPSILON_M
                and yaw_error_deg <= POSE_RECOVERY_YAW_EPSILON_DEG
            ):
                return
            self._return_to_map_pose_with_move_by(
                target,
                raise_on_nonconvergence=False,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
            )
        except Exception:
            self._state = StateStatus.FAILED
            self._current_node = None
            raise

    def _align_to_nearest_cardinal_heading(
        self, *, angular_speed: Optional[float] = None
    ) -> None:
        """Align the measured logical yaw to the nearest cardinal heading."""
        anchor_node = self._current_node
        try:
            self._map_slam.requireCurrentMap()
            current_map_pose = self._map_slam.getCurrentPose()
            current_logical_pose = self._map_slam.mapPoseToLogical(
                current_map_pose
            )
            current_heading_deg = float(current_logical_pose["theta_deg"])
            target_heading_deg = min(
                (0.0, 90.0, 180.0, 270.0),
                key=lambda heading: abs(
                    self._signed_degrees(heading - current_heading_deg)
                ),
            )
            rotation_deg = self._signed_degrees(
                target_heading_deg - current_heading_deg
            )
            self._state = StateStatus.IN_TRANSIT
            self._current_node = None
            if abs(rotation_deg) >= 1.0:
                rotation_report = rotate_by_with_feedback(
                    self._ensure_controller(),
                    math.radians(rotation_deg),
                    angular_speed=(
                        self.angular_speed
                        if angular_speed is None
                        else float(angular_speed)
                    ),
                    initial_map_pose=current_map_pose,
                    retry_action_failed=True,
                )
                if rotation_report["action_failed_count"]:
                    self._last_warnings.append(
                        {
                            "category": "rotation_action_failed_retried",
                            **rotation_report,
                        }
                    )
        except Exception:
            self._state = StateStatus.FAILED
            self._current_node = None
            raise
        self._logical_pose["theta_deg"] = target_heading_deg
        self._state = StateStatus.AT_NODE
        self._current_node = anchor_node

    def moveTo(
        self,
        node: str,
        *,
        correct: bool,
        rotate: bool = True,
        align: bool = False,
        rotate_before_translate: bool = False,
        linear_speed: Optional[float] = None,
        angular_speed: Optional[float] = None,
    ) -> List[str]:
        """Navigate to a node and optionally correct its measured final pose."""
        if self._state is StateStatus.FAILED:
            if node not in self._nodes:
                raise NavigationGraphError(f"node does not exist: {node}")
            actual = self._map_slam.getCurrentPose()
            nearest_name = min(
                self._node_order,
                key=lambda name: self._map_pose_error(
                    actual, self._map_slam.getPoint(name).pose
                ),
            )
            nearest_pose = self._map_slam.getPoint(nearest_name).pose
            position_error, yaw_error_deg = self._map_pose_error(
                actual, nearest_pose
            )
            if (
                position_error <= MOVE_BY_RECOVERY_POSITION_EPSILON_M
                and yaw_error_deg <= MOVE_BY_RECOVERY_YAW_EPSILON_DEG
            ):
                self.adoptMeasuredNodeState(
                    nearest_name,
                    position_epsilon_m=MOVE_BY_RECOVERY_POSITION_EPSILON_M,
                    yaw_epsilon_deg=MOVE_BY_RECOVERY_YAW_EPSILON_DEG,
                )
                print(
                    "[NavigationGraph] moveTo recovered FAILED state at "
                    f"{nearest_name}: position_error={position_error:.3f} m, "
                    f"yaw_error={yaw_error_deg:.2f} deg",
                    flush=True,
                )
            else:
                print(
                    "[NavigationGraph] moveTo cannot identify a nearby source "
                    f"node after FAILED; recovering directly to {node} with move_to",
                    flush=True,
                )
                self.recoverState(node, method="move_to")
                return [node]
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            raise NavigationGraphError("moveTo requires AT_NODE state")
        command_linear_speed = (
            self.linear_speed if linear_speed is None else float(linear_speed)
        )
        command_angular_speed = (
            self.angular_speed if angular_speed is None else float(angular_speed)
        )
        if not math.isfinite(command_linear_speed) or command_linear_speed < 0:
            raise ValueError("linear_speed must be a non-negative finite value")
        if not math.isfinite(command_angular_speed) or command_angular_speed < 0:
            raise ValueError("angular_speed must be a non-negative finite value")
        if self.navigation_mode == "track" and (
            linear_speed is not None or angular_speed is not None
        ):
            raise ValueError("track mode does not support per-call speed overrides")
        self._last_failure = None
        self._last_warnings = []
        path = self.shortestPath(node)
        print(
            "[NavigationGraph] moveTo "
            f"{self._current_node.name} -> {node} "
            f"path={' -> '.join(path)} correct={correct} "
            f"rotate={rotate} align={align} "
            f"v={command_linear_speed:.3f} m/s "
            f"w={command_angular_speed:.3f} rad/s",
            flush=True,
        )
        if len(path) > 1:
            self._capture_recovery_snapshot("moveTo")
            self._adjustment_origin_recorded = False
        if align:
            self._align_to_nearest_cardinal_heading(
                angular_speed=command_angular_speed
            )
        if self.navigation_mode == "track":
            execution_heading_deg = float(self._logical_pose["theta_deg"])
            self._state = StateStatus.IN_TRANSIT
            self._current_node = None
            for source_name, target_name in zip(path, path[1:]):
                print(
                    "[NavigationGraph] edge "
                    f"{source_name} -> {target_name} mode=track",
                    flush=True,
                )
                source = self._nodes[source_name]
                target = self._nodes[target_name]
                before: Optional[Dict[str, float]] = None
                requested_map_pose = self._map_slam.logicalPoseToMap(target.pose)
                try:
                    # Each graph edge starts from a fresh SLAM pose.  Its
                    # measured logical displacement is applied to that live
                    # pose, producing a new straight map-frame track.
                    before = self._map_slam.getCurrentPose()
                    requested_map_pose = self._dynamic_track_target(
                        source, target, before
                    )
                    if not rotate:
                        requested_map_pose["yaw_rad"] = before["yaw_rad"]
                    self._execute_track_segment(
                        source, target, before, requested_map_pose
                    )
                except Exception as exc:
                    self._capture_execution_failure(
                        exc,
                        source_name,
                        target,
                        node,
                        before,
                        requested_map_pose,
                    )
                    if self._last_failure is not None:
                        self._last_failure["command_type"] = "track"
                    self._state = StateStatus.FAILED
                    self._current_node = None
                    raise
            final_target = self._nodes[node]
            self._state = StateStatus.AT_NODE
            self._current_node = final_target
            final_pose = dict(final_target.pose)
            if align and rotate:
                final_pose["theta_deg"] = normalize_degrees(
                    execution_heading_deg
                    + float(final_target.pose["theta_deg"])
                    - float(self._nodes[path[0]].pose["theta_deg"])
                )
            elif not rotate or len(path) == 1:
                final_pose["theta_deg"] = execution_heading_deg
            self._logical_pose = final_pose
            if correct:
                self._correct_expected_state()
            print(f"[NavigationGraph] moveTo arrived {node}", flush=True)
            return path

        move_by_path = self.executionPath(node)
        scene_editor = self._route.get("scene_editor")
        gui_scene = isinstance(scene_editor, Mapping) and not bool(
            scene_editor.get("preserves_connection_actions")
        )
        execution_heading_deg = float(self._logical_pose["theta_deg"])
        self._state = StateStatus.IN_TRANSIT
        self._current_node = None
        for source_name, target_name in zip(move_by_path, move_by_path[1:]):
            print(
                "[NavigationGraph] edge "
                f"{source_name} -> {target_name} mode=move_by",
                flush=True,
            )
            target = self._nodes[target_name]
            recorded_heading_deg = float(
                self._nodes[source_name].pose["theta_deg"]
            )
            lateral_heading_deg: Optional[float] = None
            pre_translation_yaw_rad: Optional[float] = None
            world_travel_yaw_rad: Optional[float] = None
            if gui_scene:
                source_pose = self._nodes[source_name].pose
                target_pose = target.pose
                dx = float(target_pose["x_m"]) - float(source_pose["x_m"])
                dy = float(target_pose["y_m"]) - float(source_pose["y_m"])
                world_direction_deg = (
                    math.degrees(math.atan2(dy, dx)) if dx or dy else execution_heading_deg
                )
                is_final = target_name == node
                if rotate:
                    lateral_heading_deg = self._lateral_heading_for_segment(
                        source_name, target_name
                    )
                if lateral_heading_deg is not None:
                    # Shelf scanning keeps the chassis facing the shelf while
                    # translating along the shelf in the body lateral axis.
                    pre_translation_yaw_rad = self._map_slam.logicalPoseToMap(
                        {
                            "x_m": float(source_pose["x_m"]),
                            "y_m": float(source_pose["y_m"]),
                            "theta_deg": lateral_heading_deg,
                        }
                    )["yaw_rad"]
                    world_travel_yaw_rad = self._map_slam.logicalPoseToMap(
                        {
                            "x_m": float(source_pose["x_m"]),
                            "y_m": float(source_pose["y_m"]),
                            "theta_deg": world_direction_deg,
                        }
                    )["yaw_rad"]
                if lateral_heading_deg is not None:
                    planned_rotation_deg = self._signed_degrees(
                        lateral_heading_deg - execution_heading_deg
                    )
                elif rotate and is_final:
                    planned_rotation_deg = self._signed_degrees(
                        float(target_pose["theta_deg"]) - execution_heading_deg
                    )
                else:
                    planned_rotation_deg = 0.0
                actions = [
                    {
                        "direction_deg_relative": normalize_degrees(
                            world_direction_deg - execution_heading_deg
                        ),
                        "distance_m": math.hypot(dx, dy),
                        "rotation_deg_ccw": planned_rotation_deg,
                    }
                ]
                if lateral_heading_deg is not None:
                    execution_heading_deg = lateral_heading_deg
                elif rotate and is_final:
                    execution_heading_deg = normalize_degrees(
                        float(target_pose["theta_deg"])
                    )
            else:
                actions = self._edges[source_name][target_name]
            for action_index, recorded_action in enumerate(actions):
                action = recorded_action
                if (align or not rotate) and not gui_scene:
                    world_direction_deg = (
                        recorded_heading_deg
                        + float(recorded_action["direction_deg_relative"])
                    )
                    action = {
                        "direction_deg_relative": normalize_degrees(
                            world_direction_deg - execution_heading_deg
                        ),
                        "distance_m": float(recorded_action["distance_m"]),
                        "rotation_deg_ccw": (
                            float(recorded_action["rotation_deg_ccw"])
                            if rotate
                            else 0.0
                        ),
                    }
                    if rotate:
                        execution_heading_deg = normalize_degrees(
                            execution_heading_deg
                            + float(recorded_action["rotation_deg_ccw"])
                        )
                    recorded_heading_deg = normalize_degrees(
                        recorded_heading_deg
                        + float(recorded_action["rotation_deg_ccw"])
                    )
                before: Optional[Dict[str, float]] = None
                expected_map_pose = self._map_slam.logicalPoseToMap(target.pose)
                warning_start = len(self._last_warnings)
                try:
                    before = self._map_slam.getCurrentPose()
                    if action_index == len(actions) - 1:
                        # Preserve the graph-selected waypoint sequence, but
                        # aim the edge's final move_by at the waypoint's
                        # aligned map pose from the live SLAM pose. This
                        # absorbs small lateral drift without accumulating it
                        # across later edges.
                        target_map_pose = self._map_slam.logicalPoseToMap(
                            target.pose
                        )
                        dx = target_map_pose["x_m"] - before["x_m"]
                        dy = target_map_pose["y_m"] - before["y_m"]
                        distance_m = math.hypot(dx, dy)
                        if distance_m:
                            world_travel_yaw_rad = math.atan2(dy, dx)
                        action = dict(action)
                        action["direction_deg_relative"] = math.degrees(
                            normalize_radians(
                                (
                                    world_travel_yaw_rad
                                    if world_travel_yaw_rad is not None
                                    else before["yaw_rad"]
                                )
                                - before["yaw_rad"]
                            )
                        )
                        action["distance_m"] = distance_m
                        if rotate:
                            if align and not gui_scene:
                                target_yaw_rad = normalize_radians(
                                    before["yaw_rad"]
                                    + math.radians(
                                        float(action["rotation_deg_ccw"])
                                    )
                                )
                            elif pre_translation_yaw_rad is not None:
                                target_yaw_rad = pre_translation_yaw_rad
                                action["rotation_deg_ccw"] = 0.0
                            else:
                                target_yaw_rad = target_map_pose["yaw_rad"]
                                action["rotation_deg_ccw"] = math.degrees(
                                    normalize_radians(
                                        target_yaw_rad - before["yaw_rad"]
                                    )
                                )
                        else:
                            target_yaw_rad = before["yaw_rad"]
                            action["rotation_deg_ccw"] = 0.0
                        expected_map_pose = {
                            "x_m": target_map_pose["x_m"],
                            "y_m": target_map_pose["y_m"],
                            "yaw_rad": target_yaw_rad,
                        }
                    travel_yaw = (
                        world_travel_yaw_rad
                        if world_travel_yaw_rad is not None
                        else before["yaw_rad"]
                        + math.radians(float(action["direction_deg_relative"]))
                    )
                    distance_m = float(action["distance_m"])
                    if action_index != len(actions) - 1:
                        expected_map_pose = {
                            "x_m": before["x_m"]
                            + distance_m * math.cos(travel_yaw),
                            "y_m": before["y_m"]
                            + distance_m * math.sin(travel_yaw),
                            "yaw_rad": (
                                pre_translation_yaw_rad
                                if pre_translation_yaw_rad is not None
                                else normalize_radians(
                                    before["yaw_rad"]
                                    + math.radians(
                                        float(action["rotation_deg_ccw"])
                                    )
                                )
                            ),
                        }
                    self._execute_action(
                        action,
                        linear_speed=command_linear_speed,
                        angular_speed=command_angular_speed,
                        rotate_before_translate=(
                            rotate
                            and rotate_before_translate
                            and target_name == node
                            and action_index == len(actions) - 1
                        ),
                        pre_translation_yaw_rad=pre_translation_yaw_rad,
                        world_travel_yaw_rad=world_travel_yaw_rad,
                        retry_rotation_action_failed=True,
                        retry_translation_action_failed=bool(
                            action["distance_m"]
                        ),
                    )
                    for warning in self._last_warnings[warning_start:]:
                        warning.setdefault("source_node", source_name)
                        warning.setdefault("target_node", target_name)
                except Exception as exc:
                    self._capture_execution_failure(
                        exc,
                        source_name,
                        target,
                        node,
                        before,
                        expected_map_pose,
                    )
                    if self._last_failure is not None:
                        self._last_failure["command_type"] = "move_by"
                        self._last_failure["edge_action_index"] = action_index
                        self._last_failure["requested_action"] = dict(action)
                    self._state = StateStatus.FAILED
                    self._current_node = None
                    raise
        final_target = self._nodes[node]
        self._state = StateStatus.AT_NODE
        self._current_node = final_target
        final_pose = dict(final_target.pose)
        if align or not rotate or len(path) == 1:
            final_pose["theta_deg"] = execution_heading_deg
        self._logical_pose = final_pose
        if correct:
            self._correct_expected_state(
                linear_speed=command_linear_speed,
                angular_speed=command_angular_speed,
            )
        print(f"[NavigationGraph] moveTo arrived {node}", flush=True)
        return path

    def turnTo(self, theta: float) -> Dict[str, float]:
        """Turn to an absolute logical heading while retaining the semantic node."""
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            self._adopt_nearest_node_from_measured_pose("turnTo")
        target_heading_deg = normalize_degrees(float(theta))
        if not math.isfinite(target_heading_deg):
            raise ValueError("theta must be a finite angle in degrees")
        anchor_node = self._current_node
        print(
            "[NavigationGraph] turnTo "
            f"node={anchor_node.name} theta={target_heading_deg:.2f} deg",
            flush=True,
        )
        rotation_report: Optional[Dict[str, Any]] = None
        try:
            self._map_slam.requireCurrentMap()
            if not self._adjustment_origin_recorded:
                self._capture_recovery_snapshot("turn_to")
                self._adjustment_origin_recorded = True
                current_map_pose = dict(self._recovery_snapshot["map_pose"])
            else:
                current_map_pose = self._map_slam.getCurrentPose()
            current_heading_deg = self._map_slam.mapPoseToLogical(
                current_map_pose
            )["theta_deg"]
            rotation_deg = self._signed_degrees(
                target_heading_deg - current_heading_deg
            )
            self._state = StateStatus.IN_TRANSIT
            self._current_node = None
            if abs(rotation_deg) >= 1.0:
                rotation_report = rotate_by_with_feedback(
                    self._ensure_controller(),
                    math.radians(rotation_deg),
                    angular_speed=self.angular_speed,
                    initial_map_pose=current_map_pose,
                    retry_action_failed=True,
                    suppress_repeated_action_failed=True,
                )
        except Exception:
            self._state = StateStatus.FAILED
            self._current_node = None
            raise
        if rotation_report is not None and rotation_report["action_failed_count"]:
            self._last_warnings.append(
                {
                    "category": "rotation_action_failed_retried",
                    **rotation_report,
                }
            )
        self._logical_pose["theta_deg"] = (
            target_heading_deg
            if rotation_report is None or rotation_report["within_tolerance"]
            else self._map_slam.mapPoseToLogical(
                rotation_report["map_pose_after"]
            )["theta_deg"]
        )
        self._state = StateStatus.AT_NODE
        self._current_node = anchor_node
        print(
            "[NavigationGraph] turnTo completed "
            f"theta={self._logical_pose['theta_deg']:.2f} deg",
            flush=True,
        )
        return dict(self._logical_pose)

    def moveBy(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        correct: bool = True,
        linear_speed: Optional[float] = None,
        angular_speed: Optional[float] = None,
    ) -> Dict[str, float]:
        """Apply a body-frame micro-adjustment while retaining the semantic node."""
        if self._state is not StateStatus.AT_NODE or self._current_node is None:
            self._adopt_nearest_node_from_measured_pose("moveBy")
        command_linear_speed = (
            self.linear_speed if linear_speed is None else float(linear_speed)
        )
        command_angular_speed = (
            self.angular_speed if angular_speed is None else float(angular_speed)
        )
        if not math.isfinite(command_linear_speed) or command_linear_speed < 0:
            raise ValueError("linear_speed must be a non-negative finite value")
        if not math.isfinite(command_angular_speed) or command_angular_speed < 0:
            raise ValueError("angular_speed must be a non-negative finite value")
        if not self._adjustment_origin_recorded:
            self._capture_recovery_snapshot("move_by")
            self._adjustment_origin_recorded = True
            correction_start = (
                dict(self._recovery_snapshot["map_pose"])
                if correct and self._recovery_snapshot is not None
                else None
            )
        else:
            correction_start = self._map_slam.getCurrentPose() if correct else None
        anchor_node = self._current_node
        requested_distance = math.hypot(float(x), float(y))
        distance_mm = round(requested_distance * 1000)
        executed_distance = (
            distance_mm / 1000.0
            if distance_mm >= MOVE_BY_MIN_DISTANCE_MM
            else 0.0
        )
        requested_theta = float(theta)
        executed_theta = (
            requested_theta if abs(requested_theta) >= 1.0 else 0.0
        )
        direction = (
            normalize_degrees(math.degrees(math.atan2(float(y), float(x))))
            if requested_distance
            else 0.0
        )
        action = {
            "direction_deg_relative": direction,
            "distance_m": requested_distance,
            "rotation_deg_ccw": requested_theta,
        }
        correction_target = None
        if correction_start is not None:
            travel_yaw = correction_start["yaw_rad"] + math.radians(direction)
            correction_target = {
                "x_m": correction_start["x_m"]
                + executed_distance * math.cos(travel_yaw),
                "y_m": correction_start["y_m"]
                + executed_distance * math.sin(travel_yaw),
                "yaw_rad": normalize_radians(
                    correction_start["yaw_rad"] + math.radians(executed_theta)
                ),
            }
        print(
            "[NavigationGraph] moveBy "
            f"x={float(x):.3f} m y={float(y):.3f} m "
            f"theta={requested_theta:.2f} deg correct={correct} "
            f"v={command_linear_speed:.3f} m/s "
            f"w={command_angular_speed:.3f} rad/s",
            flush=True,
        )
        self._state = StateStatus.IN_TRANSIT
        self._current_node = None
        rotation_report: Optional[Dict[str, Any]] = None
        correction_pose: Optional[Dict[str, float]] = None
        try:
            try:
                rotation_report = self._execute_action(
                    action,
                    linear_speed=command_linear_speed,
                    angular_speed=command_angular_speed,
                    retry_rotation_action_failed=True,
                    suppress_repeated_rotation_action_failed=(
                        executed_distance == 0.0
                    ),
                    retry_translation_action_failed=(executed_distance != 0.0),
                )
            except Exception as exc:
                error_code = getattr(
                    getattr(exc, "code", None), "value", None
                )
                if error_code not in {1001, 2001}:
                    raise
                if correction_target is None:
                    correction_pose = self._map_slam.getCurrentPose()
                    self._last_warnings.append(
                        {
                            "category": "move_by_action_failed_continued_from_measured_pose",
                            "error": repr(exc),
                            "map_pose_after": dict(correction_pose),
                        }
                    )
                else:
                    self._last_warnings.append(
                        {
                            "category": "move_by_action_failed_continued_with_pose_correction",
                            "error": repr(exc),
                            "target_map_pose": dict(correction_target),
                        }
                    )
                    correction_pose = self._return_to_map_pose_with_move_by(
                        correction_target,
                        raise_on_nonconvergence=False,
                        linear_speed=command_linear_speed,
                        angular_speed=command_angular_speed,
                    )
            if correction_target is not None and correction_pose is None:
                correction_pose = self._map_slam.getCurrentPose()
                position_error, yaw_error_deg = self._map_pose_error(
                    correction_pose, correction_target
                )
                if (
                    position_error > POSE_RECOVERY_POSITION_EPSILON_M
                    or yaw_error_deg > POSE_RECOVERY_YAW_EPSILON_DEG
                ):
                    correction_pose = self._return_to_map_pose_with_move_by(
                        correction_target,
                        raise_on_nonconvergence=False,
                        linear_speed=command_linear_speed,
                        angular_speed=command_angular_speed,
                    )
        except Exception:
            self._state = StateStatus.FAILED
            self._current_node = None
            raise
        self._logical_pose = advance_pose(
            self._logical_pose,
            executed_distance,
            executed_theta,
            direction,
        )
        if correction_target is None and correction_pose is not None:
            self._logical_pose = self._map_slam.mapPoseToLogical(
                correction_pose
            )
        if correction_target is not None and correction_pose is not None:
            position_error, yaw_error_deg = self._map_pose_error(
                correction_pose, correction_target
            )
            if (
                position_error > POSE_RECOVERY_POSITION_EPSILON_M
                or yaw_error_deg > POSE_RECOVERY_YAW_EPSILON_DEG
            ):
                self._logical_pose = self._map_slam.mapPoseToLogical(
                    correction_pose
                )
        if (
            rotation_report is not None
            and rotation_report["action_failed_count"]
            and not rotation_report["within_tolerance"]
        ):
            self._logical_pose["theta_deg"] = self._map_slam.mapPoseToLogical(
                rotation_report["map_pose_after"]
            )["theta_deg"]
        self._state = StateStatus.AT_NODE
        self._current_node = anchor_node
        print(
            "[NavigationGraph] moveBy completed "
            f"x={self._logical_pose['x_m']:.3f} m "
            f"y={self._logical_pose['y_m']:.3f} m "
            f"theta={self._logical_pose['theta_deg']:.2f} deg",
            flush=True,
        )
        return dict(self._logical_pose)

    def close(self) -> None:
        """Close a controller created internally by this graph."""
        if self._owns_controller and self._controller is not None:
            self._controller.close()
        self._controller = None
        self._owns_controller = False

    def __enter__(self) -> "NavigationGraph":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
