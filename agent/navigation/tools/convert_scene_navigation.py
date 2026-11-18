#!/usr/bin/env python3
"""Convert a measured GUI scene JSON into NavigationGraph artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.navigation.scene_adapter import convert_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="GUI scene JSON")
    parser.add_argument("--output", type=Path, help="generated route JSON path")
    parser.add_argument("--initial-node", help="initial node; defaults to the first GUI node")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_path, graph_path = convert_scene(
        args.scene,
        output_path=args.output,
        initial_node=args.initial_node,
    )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    print(
        f"转换完成：{len(graph['nodes'])} 个节点，"
        f"{len(graph['edges']) // 2} 条双向边"
    )
    print(f"NavigationGraph 入口：{route_path}")
    print(f"拓扑图：{graph_path}")
    print("尚未写入 SLAM 锚点；实机演示请使用 demo_navigation_graph.py --execute。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
