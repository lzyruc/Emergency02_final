#!/usr/bin/env python3
"""List or edit graphGUI scene connections without generating derived files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

from agent.navigation.data.motion_log import read_json, write_json_atomic
from agent.navigation.graph_models import NavigationGraphError
from agent.navigation.scene_adapter import validate_scene_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="graphGUI source scene JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="validate and list nodes and connections")
    for command in ("connect", "disconnect"):
        editor = subparsers.add_parser(
            command,
            help=f"{command} one undirected connection by node name",
        )
        editor.add_argument("node_a")
        editor.add_argument("node_b")
        output = editor.add_mutually_exclusive_group()
        output.add_argument(
            "--output",
            type=Path,
            help="write a new scene; default: <scene>.edited.json",
        )
        output.add_argument(
            "--in-place",
            action="store_true",
            help="explicitly replace the input scene atomically",
        )
        editor.add_argument(
            "--force",
            action="store_true",
            help="allow --output to replace an existing file",
        )
    return parser.parse_args()


def _nodes_by_name(document: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(node["name"]): node for node in document["robots"]}


def _edge_matches(
    connection: Mapping[str, Any],
    source_id: Any,
    target_id: Any,
) -> bool:
    return {
        connection.get("fromId"),
        connection.get("toId"),
    } == {source_id, target_id}


def _output_path(args: argparse.Namespace, source: Path) -> Path:
    if args.in_place:
        return source
    if args.output is not None:
        destination = args.output.expanduser().resolve()
    else:
        destination = source.with_name(f"{source.stem}.edited.json")
    if destination == source:
        raise NavigationGraphError(
            "use --in-place to explicitly replace the input scene"
        )
    if destination.exists() and not args.force:
        raise NavigationGraphError(
            f"output already exists: {destination}; use --force to replace it"
        )
    return destination


def _list_scene(document: Mapping[str, Any]) -> None:
    by_id = {node["id"]: str(node["name"]) for node in document["robots"]}
    print("节点：")
    for node in document["robots"]:
        print(
            f"  {node['name']}: "
            f"x={float(node['x']):.3f} m, "
            f"y={float(node['y']):.3f} m, "
            f"theta={float(node.get('angle', 0.0)):.1f} deg"
        )
    print("双向连接：")
    for connection in document["connections"]:
        suffix = " [custom actions]" if connection.get("actions") is not None else ""
        print(
            f"  {by_id[connection['fromId']]} <-> "
            f"{by_id[connection['toId']]}{suffix}"
        )


def _edit_scene(document: Dict[str, Any], args: argparse.Namespace) -> None:
    nodes = _nodes_by_name(document)
    missing = [name for name in (args.node_a, args.node_b) if name not in nodes]
    if missing:
        raise NavigationGraphError(
            "unknown node name: " + ", ".join(sorted(set(missing)))
        )
    if args.node_a == args.node_b:
        raise NavigationGraphError("a connection cannot be a self-loop")
    source_id = nodes[args.node_a]["id"]
    target_id = nodes[args.node_b]["id"]
    matching = [
        index
        for index, connection in enumerate(document["connections"])
        if _edge_matches(connection, source_id, target_id)
    ]
    if args.command == "connect":
        if matching:
            raise NavigationGraphError(
                f"connection already exists: {args.node_a} <-> {args.node_b}"
            )
        # This is the exact object shape created by graphGUI. Missing actions
        # means the converter derives direct forward/reverse geometry.
        document["connections"].append(
            {"fromId": source_id, "toId": target_id}
        )
    else:
        if not matching:
            raise NavigationGraphError(
                f"connection does not exist: {args.node_a} <-> {args.node_b}"
            )
        del document["connections"][matching[0]]


def main() -> int:
    args = parse_args()
    source = args.scene.expanduser().resolve()
    try:
        document = read_json(source)
        summary = validate_scene_document(
            document,
            require_connected=args.command == "list",
        )
        if args.command == "list":
            print(
                f"校验通过：{summary['node_count']} 个节点，"
                f"{summary['connection_count']} 条双向连接"
            )
            _list_scene(document)
            return 0

        if args.in_place and args.force:
            raise NavigationGraphError("--force has no effect with --in-place")
        _edit_scene(document, args)
        # A saved edit is always immediately convertible. Deleting a bridge
        # is rejected rather than leaving a disconnected source map.
        summary = validate_scene_document(document)
        destination = _output_path(args, source)
        write_json_atomic(destination, document)
        print(
            f"已保存：{destination}\n"
            f"{summary['node_count']} 个节点，"
            f"{summary['connection_count']} 条双向连接"
        )
        return 0
    except (OSError, ValueError, KeyError, NavigationGraphError) as exc:
        print(f"场景编辑失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
