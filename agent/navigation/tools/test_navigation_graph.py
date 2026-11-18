#!/usr/bin/env python3
"""Create start<->point2 and navigate start -> point2 -> end."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..navigation_graph import NavigationGraph, NavigationGraphError
except ImportError:  # Direct script execution.
    from agent.navigation.navigation_graph import NavigationGraph, NavigationGraphError


DEFAULT_ROUTE = Path(
    "/home/nvidia/X-humanoid/agent/navigation/data/logs/"
    "route_20260718_210645.json"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", nargs="?", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--execute", action="store_true", help="actually move the chassis")
    parser.add_argument(
        "--navigation-mode",
        choices=("track", "move_by"),
        default="move_by",
        help="graph-wide chassis execution mode",
    )
    return parser.parse_args()


def main() -> int:
    """Create the direct edge and optionally execute the two requested moves."""
    args = parse_args()
    try:
        with NavigationGraph(args.route, navigation_mode=args.navigation_mode) as graph:
            graph.createEdge("emergency.start", "emergency.point2")
            graph.save()
            first_path = graph.shortestPath("emergency.point2")
            print(f"start -> point2 shortest path: {' -> '.join(first_path)}")
            if not args.execute:
                with NavigationGraph(
                    args.route,
                    initial_node="emergency.point2",
                    navigation_mode=args.navigation_mode,
                ) as preview:
                    second_path = preview.shortestPath("emergency.end")
                print(f"point2 -> end shortest path: {' -> '.join(second_path)}")
                print("Dry-run only; add --execute to move the chassis.")
                print(f"graph: {graph.graph_path}")
                return 0
            graph.moveTo("emergency.point2", correct=False)
            second_path = graph.shortestPath("emergency.end")
            print(f"point2 -> end shortest path: {' -> '.join(second_path)}")
            graph.moveTo("emergency.end", correct=False)
            state, node = graph.getState()
            print(f"final state: {state.value}, node={node.name if node else None}")
        return 0
    except (OSError, ValueError, NavigationGraphError) as exc:
        print(f"graph navigation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
