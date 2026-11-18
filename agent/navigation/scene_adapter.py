"""Convert a measured GUI scene into NavigationGraph route and graph files."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data.motion_log import (
    LOG_KIND,
    SCHEMA_VERSION,
    json_safe,
    normalize_degrees,
    read_json,
    utc_now,
    validate_point_name,
    write_json_atomic,
)
from .graph_models import (
    GRAPH_KIND,
    GRAPH_SCHEMA_VERSION,
    NavigationGraphError,
)


SceneConnection = Tuple[str, str, Optional[List[Dict[str, float]]]]


def default_navigation_path(scene_path: Path) -> Path:
    """Return the generated route path without modifying the GUI source file."""
    source = Path(scene_path).expanduser().resolve()
    return source.with_name(f"{source.stem}.navigation.json")


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NavigationGraphError(f"{field} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise NavigationGraphError(f"{field} must be finite, got {value!r}")
    return number


def _scene_nodes(document: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    raw_nodes = document.get("robots")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise NavigationGraphError("GUI scene must contain a non-empty robots list")

    nodes: List[Dict[str, Any]] = []
    by_id: Dict[Any, Dict[str, Any]] = {}
    names = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise NavigationGraphError(f"robots[{index}] must be an object")
        source_id = raw.get("id")
        if source_id is None or source_id in by_id:
            raise NavigationGraphError(f"robots[{index}].id is missing or duplicated: {source_id!r}")
        try:
            name = validate_point_name(str(raw.get("name", "")))
        except ValueError as exc:
            raise NavigationGraphError(f"invalid robots[{index}].name: {exc}") from exc
        if name in names:
            raise NavigationGraphError(f"duplicate GUI node name: {name}")
        node = {
            "source_id": source_id,
            "name": name,
            "pose": {
                "x_m": _finite_number(raw.get("x"), f"robots[{index}].x"),
                "y_m": _finite_number(raw.get("y"), f"robots[{index}].y"),
                "theta_deg": normalize_degrees(
                    _finite_number(raw.get("angle", 0.0), f"robots[{index}].angle")
                ),
            },
        }
        nodes.append(node)
        by_id[source_id] = node
        names.add(name)
    return nodes, by_id


def _scene_connections(
    document: Mapping[str, Any], by_id: Mapping[Any, Mapping[str, Any]]
) -> List[SceneConnection]:
    raw_connections = document.get("connections")
    if not isinstance(raw_connections, list):
        raise NavigationGraphError("GUI scene connections must be a list")

    connections: List[SceneConnection] = []
    seen = set()
    for index, raw in enumerate(raw_connections):
        if not isinstance(raw, Mapping):
            raise NavigationGraphError(f"connections[{index}] must be an object")
        source_id = raw.get("fromId")
        target_id = raw.get("toId")
        if source_id not in by_id or target_id not in by_id:
            raise NavigationGraphError(
                f"connections[{index}] references unknown IDs: {source_id!r}, {target_id!r}"
            )
        if source_id == target_id:
            raise NavigationGraphError(f"connections[{index}] is a self-loop")
        key = frozenset((source_id, target_id))
        if key in seen:
            raise NavigationGraphError(
                f"connections[{index}] duplicates an existing undirected edge"
            )
        seen.add(key)
        raw_actions = raw.get("actions")
        actions: Optional[List[Dict[str, float]]] = None
        if raw_actions is not None:
            if not isinstance(raw_actions, list) or not raw_actions:
                raise NavigationGraphError(
                    f"connections[{index}].actions must be a non-empty list"
                )
            actions = []
            for action_index, raw_action in enumerate(raw_actions):
                if not isinstance(raw_action, Mapping):
                    raise NavigationGraphError(
                        f"connections[{index}].actions[{action_index}] must be an object"
                    )
                distance = _finite_number(
                    raw_action.get("distance_m"),
                    f"connections[{index}].actions[{action_index}].distance_m",
                )
                if distance < 0:
                    raise NavigationGraphError(
                        f"connections[{index}].actions[{action_index}].distance_m "
                        "must be non-negative"
                    )
                actions.append(
                    {
                        "direction_deg_relative": normalize_degrees(
                            _finite_number(
                                raw_action.get("direction_deg_relative"),
                                f"connections[{index}].actions[{action_index}]"
                                ".direction_deg_relative",
                            )
                        ),
                        "distance_m": distance,
                        "rotation_deg_ccw": _finite_number(
                            raw_action.get("rotation_deg_ccw"),
                            f"connections[{index}].actions[{action_index}]"
                            ".rotation_deg_ccw",
                        ),
                    }
                )
        connections.append(
            (
                str(by_id[source_id]["name"]),
                str(by_id[target_id]["name"]),
                actions,
            )
        )
    return connections


def _require_connected(
    nodes: Sequence[Mapping[str, Any]], connections: Sequence[SceneConnection]
) -> None:
    adjacency = {str(node["name"]): set() for node in nodes}
    for source, target, _ in connections:
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited = set()
    pending = [str(nodes[0]["name"])]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        pending.extend(adjacency[name] - visited)
    if len(visited) != len(nodes):
        missing = sorted(set(adjacency) - visited)
        raise NavigationGraphError(f"GUI scene graph is disconnected; unreachable nodes: {missing}")


def validate_scene_document(
    document: Mapping[str, Any],
    *,
    require_connected: bool = True,
) -> Dict[str, int]:
    """Validate a graphGUI scene with the same structural rules as conversion."""
    nodes, by_id = _scene_nodes(document)
    connections = _scene_connections(document, by_id)
    if require_connected:
        _require_connected(nodes, connections)
    return {
        "node_count": len(nodes),
        "connection_count": len(connections),
    }


def _signed_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _direct_action(source: Mapping[str, float], target: Mapping[str, float]) -> Dict[str, float]:
    dx = float(target["x_m"]) - float(source["x_m"])
    dy = float(target["y_m"]) - float(source["y_m"])
    world_direction = (
        math.degrees(math.atan2(dy, dx)) if dx or dy else float(source["theta_deg"])
    )
    return {
        "direction_deg_relative": normalize_degrees(
            world_direction - float(source["theta_deg"])
        ),
        "distance_m": math.hypot(dx, dy),
        "rotation_deg_ccw": _signed_degrees(
            float(target["theta_deg"]) - float(source["theta_deg"])
        ),
    }


def _inverse_actions(actions: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    result = []
    for action in reversed(actions):
        rotation = float(action["rotation_deg_ccw"])
        result.append(
            {
                "direction_deg_relative": normalize_degrees(
                    float(action["direction_deg_relative"]) + 180.0 - rotation
                ),
                "distance_m": float(action["distance_m"]),
                "rotation_deg_ccw": -rotation,
            }
        )
    return result


def _map_name(response: Mapping[str, Any]) -> str:
    for key in ("map_name", "name"):
        if response.get(key):
            return str(response[key])
    raise NavigationGraphError(f"current map response has no map name: {response!r}")


def convert_scene(
    scene_path: Path,
    output_path: Optional[Path] = None,
    initial_node: Optional[str] = None,
    current_map_response: Optional[Mapping[str, Any]] = None,
    initial_map_pose: Optional[Mapping[str, float]] = None,
) -> Tuple[Path, Path]:
    """Convert one GUI scene and return generated route and graph paths.

    ``current_map_response`` and ``initial_map_pose`` must either both be supplied
    or both be omitted.  When supplied, the map pose is assumed to be the live
    pose of ``initial_node`` and defines the logical-to-SLAM rigid transform.
    """
    source = Path(scene_path).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_navigation_path(source)
    )
    if destination == source:
        raise NavigationGraphError("conversion output must not overwrite the GUI scene JSON")
    document = read_json(source)
    nodes, by_id = _scene_nodes(document)
    connections = _scene_connections(document, by_id)
    _require_connected(nodes, connections)

    by_name = {str(node["name"]): node for node in nodes}
    anchor_name = initial_node or str(nodes[0]["name"])
    if anchor_name not in by_name:
        raise NavigationGraphError(f"initial node does not exist: {anchor_name}")
    if (current_map_response is None) != (initial_map_pose is None):
        raise NavigationGraphError(
            "current_map_response and initial_map_pose must be supplied together"
        )
    anchor_pose = dict(by_name[anchor_name]["pose"])

    editor_metadata = {
        "source_file": str(source),
        "scene_width_m": document.get("sceneW"),
        "scene_height_m": document.get("sceneH"),
        "origin_x_offset": document.get("originXOffset"),
        "obstacles": json_safe(document.get("obstacles", [])),
        "preserves_connection_actions": any(
            actions is not None for _, _, actions in connections
        ),
        "coordinate_note": (
            "robots.x/y are imported directly as logical metres; "
            "originXOffset is editor display metadata and is not added to coordinates"
        ),
    }
    now = utc_now()
    route: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOG_KIND,
        "created_at": now,
        "updated_at": now,
        "coordinate_systems": {
            "logical": {
                "frame": "gui_scene",
                "x_axis": "forward",
                "y_axis": "left",
                "theta": "degrees_counterclockwise_0_to_359",
                "initial_pose": anchor_pose,
            },
            "observed": {
                "frame": "chassis_map",
                "source": "GET /api/core/navigation/v1/pose",
                "yaw": "radians_counterclockwise",
            },
        },
        "motion_order": "translate_then_rotate",
        "points": [
            {
                "sequence": index,
                "name": node["name"],
                "logical_pose": dict(node["pose"]),
                "before_action": 0,
                "source_id": node["source_id"],
            }
            for index, node in enumerate(nodes)
        ],
        "actions": [],
        "scene_editor": editor_metadata,
    }
    map_name = ""
    map_pose_for_graph = {
        "x_m": float(anchor_pose["x_m"]),
        "y_m": float(anchor_pose["y_m"]),
        "yaw_rad": math.radians(float(anchor_pose["theta_deg"])),
    }
    if current_map_response is not None and initial_map_pose is not None:
        map_name = _map_name(current_map_response)
        map_pose_for_graph = {
            "x_m": _finite_number(initial_map_pose.get("x_m"), "initial_map_pose.x_m"),
            "y_m": _finite_number(initial_map_pose.get("y_m"), "initial_map_pose.y_m"),
            "yaw_rad": _finite_number(initial_map_pose.get("yaw_rad"), "initial_map_pose.yaw_rad"),
        }
        route["map_context"] = {
            "current_map_response": json_safe(current_map_response),
            "initial_map_pose": map_pose_for_graph,
            "anchor_node": anchor_name,
            "captured_at": now,
        }

    write_json_atomic(destination, route)
    graph_path = destination.with_name(f"{destination.stem}.graph.json")
    graph_edges = []
    for source_name, target_name, connection_actions in connections:
        source_pose = by_name[source_name]["pose"]
        target_pose = by_name[target_name]["pose"]
        forward_actions = (
            [dict(action) for action in connection_actions]
            if connection_actions is not None
            else [_direct_action(source_pose, target_pose)]
        )
        reverse_actions = (
            _inverse_actions(forward_actions)
            if connection_actions is not None
            else [_direct_action(target_pose, source_pose)]
        )
        graph_edges.append(
            {
                "from": source_name,
                "to": target_name,
                "actions": forward_actions,
            }
        )
        graph_edges.append(
            {
                "from": target_name,
                "to": source_name,
                "actions": reverse_actions,
            }
        )
    graph = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "kind": GRAPH_KIND,
        "created_from": str(destination),
        "source_route_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "updated_at": now,
        "map": {
            "class": "MapSLAM",
            "name": map_name,
            "alignment": {
                "logical_initial_pose": anchor_pose,
                "initial_map_pose": map_pose_for_graph,
                "anchor_node": anchor_name,
            },
        },
        "nodes": [
            {"name": node["name"], "pose": dict(node["pose"]), "source_id": node["source_id"]}
            for node in nodes
        ],
        "edges": graph_edges,
        "scene_editor": editor_metadata,
    }
    write_json_atomic(graph_path, graph)
    return destination, graph_path
