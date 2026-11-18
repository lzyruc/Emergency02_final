#!/usr/bin/env python3
"""Rebuild the derived NavigationGraph JSON from the latest route JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.navigation import NavigationGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "route",
        type=Path,
        help="source route JSON, for example emergency.navigation.json",
    )
    return parser.parse_args()


def main() -> int:
    route_path = parse_args().route.expanduser().resolve()
    with NavigationGraph(route_path, rebuild_graph=True) as navigation:
        directed_edge_count = sum(
            len(targets) for targets in navigation.edges.values()
        )
        print(f"源路线：{route_path}")
        print(f"已重建：{navigation.graph_path}")
        print(
            f"节点：{len(navigation.nodes)}，"
            f"双向边：{directed_edge_count // 2}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
