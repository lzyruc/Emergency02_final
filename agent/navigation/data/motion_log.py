"""Shared JSON format and planar-motion helpers for move-by route tools."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping


SCHEMA_VERSION = 4
LOG_KIND = "xrocs.move_by_route"
POINT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_degrees(angle_deg: float) -> float:
    """Normalize an absolute heading to [0, 360)."""
    normalized = float(angle_deg) % 360.0
    return 0.0 if math.isclose(normalized, 360.0) else normalized


def normalize_radians(angle_rad: float) -> float:
    """Normalize a signed angle to [-pi, pi)."""
    return (float(angle_rad) + math.pi) % (2 * math.pi) - math.pi


def advance_pose(
    pose: Mapping[str, float],
    distance_m: float,
    rotation_deg: float,
    direction_deg: float = 0.0,
) -> Dict[str, float]:
    """Apply body-relative translation followed by counterclockwise rotation."""
    theta_deg = normalize_degrees(float(pose["theta_deg"]))
    translation_yaw = math.radians(theta_deg + normalize_degrees(direction_deg))
    x_m = float(pose["x_m"]) + float(distance_m) * math.cos(translation_yaw)
    y_m = float(pose["y_m"]) + float(distance_m) * math.sin(translation_yaw)
    return {
        "x_m": 0.0 if abs(x_m) < 1e-12 else round(x_m, 9),
        "y_m": 0.0 if abs(y_m) < 1e-12 else round(y_m, 9),
        "theta_deg": normalize_degrees(theta_deg + float(rotation_deg)),
    }


def map_pose_from_response(data: Mapping[str, Any]) -> Dict[str, float]:
    """Extract a finite map-frame x/y/yaw pose from a chassis response."""
    try:
        pose = {"x_m": float(data["x"]), "y_m": float(data["y"]), "yaw_rad": float(data["yaw"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid chassis map pose response: {data!r}") from exc
    if not all(math.isfinite(value) for value in pose.values()):
        raise ValueError(f"non-finite chassis map pose response: {data!r}")
    return pose


def validate_point_name(name: str) -> str:
    """Validate a lower-case hierarchical point name."""
    value = name.strip()
    if not POINT_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "point name must use lower-case tree notation, for example "
            "scene.station_a.approach"
        )
    return value


def new_route() -> Dict[str, Any]:
    """Create an empty route document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LOG_KIND,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "coordinate_systems": {
            "logical": {
                "frame": "local_start",
                "x_axis": "forward",
                "y_axis": "left",
                "theta": "degrees_counterclockwise_0_to_359",
                "initial_pose": {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
            },
            "observed": {
                "frame": "chassis_map",
                "source": "GET /api/core/navigation/v1/pose",
                "yaw": "radians_counterclockwise",
            },
        },
        "motion_order": "translate_then_rotate",
        "points": [],
        "actions": [],
    }


def validate_route(document: Mapping[str, Any]) -> None:
    """Validate fields required to replay a route."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {document.get('schema_version')!r}")
    if document.get("kind") != LOG_KIND:
        raise ValueError(f"unsupported log kind: {document.get('kind')!r}")
    if document.get("motion_order") != "translate_then_rotate":
        raise ValueError("only translate_then_rotate logs are supported")
    actions = document.get("actions")
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")
    for expected_sequence, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise ValueError(f"actions[{expected_sequence}] must be an object")
        if action.get("sequence") != expected_sequence:
            raise ValueError(f"actions[{expected_sequence}].sequence must equal {expected_sequence}")
        commands = action.get("commands")
        if not isinstance(commands, list):
            raise ValueError(f"actions[{expected_sequence}].commands must be a list")
        for command in commands:
            if command.get("type") not in {"translate", "rotate"}:
                raise ValueError(f"invalid command type in action {expected_sequence}")
            if int(command.get("move_mode", -1)) not in {0, 1}:
                raise ValueError(f"invalid move_mode in action {expected_sequence}")
            allowed_methods = (
                {"move_by"}
                if command.get("type") == "translate"
                else {"move_to_at_current_map_xy", "move_by_with_pose_feedback"}
            )
            if command.get("method") not in allowed_methods:
                raise ValueError(f"invalid command method in action {expected_sequence}")
            direction = float(command["direction_rad"])
            distance = int(command["distance_mm"])
            if not math.isfinite(direction) or not -math.pi <= direction <= math.pi:
                raise ValueError(f"invalid direction_rad in action {expected_sequence}")
            if distance < 0:
                raise ValueError(f"invalid distance_mm in action {expected_sequence}")


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object from disk."""
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload["updated_at"] = utc_now()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def json_safe(value: Any) -> Any:
    """Convert controller responses into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)
