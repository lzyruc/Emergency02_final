#!/usr/bin/env python3
"""更新比赛导航 route/graph 的 SLAM 地图锚点，不改变节点和边。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_ROUTE_PATH = ROOT / "config/scene.navigation.json"
DEFAULT_MAP_DIRECTORY = Path("/opt/config/custom/bslam/map/scan_map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "机器人位于startpoint并完成新地图定位后，只更新现有route/graph的"
            "地图名和SLAM锚点；不会重建拓扑或修改scene.json。"
        )
    )
    parser.add_argument(
        "--route",
        type=Path,
        default=DEFAULT_ROUTE_PATH,
        help="route文件，默认config/scene.navigation.json",
    )
    parser.add_argument(
        "--map-name",
        default="final-game-emergency",
        help=(
            "地图名或完整路径；仅给名称时自动补成"
            "/opt/config/custom/bslam/map/scan_map/<name>.pcd"
        ),
    )
    parser.add_argument("--x", type=float, required=True, help="startpoint的SLAM x，单位m")
    parser.add_argument("--y", type=float, required=True, help="startpoint的SLAM y，单位m")
    yaw_group = parser.add_mutually_exclusive_group(required=True)
    yaw_group.add_argument("--yaw-deg", type=float, help="startpoint的SLAM朝向，单位度")
    yaw_group.add_argument("--yaw-rad", type=float, help="startpoint的SLAM朝向，单位弧度")
    parser.add_argument(
        "--anchor-node",
        default="startpoint",
        help="地图锚点节点，默认startpoint",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"JSON根节点必须是对象：{path}")
    return document


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def resolve_map_name(value: str) -> str:
    map_name = value.strip()
    if not map_name:
        raise ValueError("--map-name不能为空")
    if "/" not in map_name:
        if not map_name.endswith(".pcd"):
            map_name += ".pcd"
        return str(DEFAULT_MAP_DIRECTORY / map_name)
    if not map_name.endswith(".pcd"):
        map_name += ".pcd"
    return map_name


def route_anchor_pose(
    route: dict[str, Any],
    anchor_node: str,
) -> dict[str, float]:
    for point in route.get("points", []):
        if point.get("name") == anchor_node:
            pose = point.get("logical_pose")
            if not isinstance(pose, dict):
                break
            return {
                "x_m": float(pose["x_m"]),
                "y_m": float(pose["y_m"]),
                "theta_deg": float(pose["theta_deg"]),
            }
    raise ValueError(f"route中不存在有效锚点：{anchor_node}")


def main() -> int:
    args = parse_args()
    route_path = args.route.expanduser().resolve()
    graph_path = route_path.with_name(f"{route_path.stem}.graph.json")
    if not route_path.is_file():
        raise FileNotFoundError(f"route不存在：{route_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"graph不存在：{graph_path}")

    yaw_rad = (
        math.radians(args.yaw_deg)
        if args.yaw_deg is not None
        else float(args.yaw_rad)
    )
    if not all(math.isfinite(value) for value in (args.x, args.y, yaw_rad)):
        raise ValueError("x、y和yaw必须是有限数值")

    route = read_json(route_path)
    graph = read_json(graph_path)
    if route.get("kind") != "xrocs.move_by_route":
        raise ValueError(f"不是move_by route：{route_path}")
    if graph.get("kind") != "xrocs.navigation_graph":
        raise ValueError(f"不是navigation graph：{graph_path}")

    logical_anchor = route_anchor_pose(route, args.anchor_node)
    logical_initial = (
        route.get("coordinate_systems", {})
        .get("logical", {})
        .get("initial_pose")
    )
    if logical_initial != logical_anchor:
        raise ValueError(
            f"锚点{args.anchor_node}不是route逻辑初始位姿；"
            "本工具拒绝改变逻辑坐标系"
        )
    graph_nodes = {
        str(node.get("name")) for node in graph.get("nodes", [])
    }
    if args.anchor_node not in graph_nodes:
        raise ValueError(f"graph中不存在锚点：{args.anchor_node}")

    map_name = resolve_map_name(args.map_name)
    initial_map_pose = {
        "x_m": float(args.x),
        "y_m": float(args.y),
        "yaw_rad": yaw_rad,
    }
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    route["map_context"] = {
        "current_map_response": {"map_name": map_name},
        "initial_map_pose": initial_map_pose,
        "anchor_node": args.anchor_node,
        "captured_at": now,
    }
    route["updated_at"] = now

    graph_map = graph.setdefault("map", {})
    graph_map["class"] = "MapSLAM"
    graph_map["name"] = map_name
    graph_map["alignment"] = {
        "logical_initial_pose": logical_anchor,
        "initial_map_pose": initial_map_pose,
        "anchor_node": args.anchor_node,
    }
    graph["updated_at"] = now

    backup_directory = (
        route_path.parent
        / "navigation_anchor_backups"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    backup_directory.mkdir(parents=True)
    shutil.copy2(route_path, backup_directory / route_path.name)
    shutil.copy2(graph_path, backup_directory / graph_path.name)

    write_json_atomic(route_path, route)
    graph["source_route_sha256"] = hashlib.sha256(
        route_path.read_bytes()
    ).hexdigest()
    write_json_atomic(graph_path, graph)

    saved_route = read_json(route_path)
    saved_graph = read_json(graph_path)
    route_context = saved_route["map_context"]
    graph_alignment = saved_graph["map"]["alignment"]
    if (
        route_context["current_map_response"]["map_name"] != saved_graph["map"]["name"]
        or route_context["initial_map_pose"] != graph_alignment["initial_map_pose"]
        or route_context["anchor_node"] != graph_alignment["anchor_node"]
        or saved_graph["source_route_sha256"]
        != hashlib.sha256(route_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError(
            f"写入后校验失败；原文件备份位于：{backup_directory}"
        )

    print(f"地图锚点更新完成：map={map_name}")
    print(
        f"startpoint SLAM pose：x={args.x:.6f} m，y={args.y:.6f} m，"
        f"yaw={yaw_rad:.9f} rad（{math.degrees(yaw_rad):.3f}°）"
    )
    print(f"route：{route_path}")
    print(f"graph：{graph_path}")
    print(f"原文件备份：{backup_directory}")
    print("节点和边未重建；运行game_emergency.py时会自动复制到runtime。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
