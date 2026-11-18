#!/usr/bin/env python3
"""Test three rounds of moveBy adjustments followed by saved-pose recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

try:
    from ..data.motion_log import map_pose_from_response
    from ..navigation_graph import NavigationGraph, NavigationGraphError
    from ..scene_adapter import convert_scene
except ImportError:  # Direct script execution.
    from agent.navigation.data.motion_log import map_pose_from_response
    from agent.navigation.navigation_graph import (
        NavigationGraph,
        NavigationGraphError,
    )
    from agent.navigation.scene_adapter import convert_scene


DEFAULT_SCENE = Path(
    "xrocs/agent/navigation/data/logs/scene_shop.json"
)
INITIAL_NODE = "start_point"
TEST_NODE = "hub_center"
ROUNDS = 3
MICRO_ADJUSTMENTS = (
    (0.10, 0.00, 0.0),  # forward 10 cm
    (0.00, 0.10, 0.0),  # left 10 cm; cumulative radius about 14.1 cm
    (0.00, 0.00, 5.0),  # rotate counterclockwise 5 degrees
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--execute", action="store_true", help="actually move the chassis")
    parser.add_argument("--linear-speed", type=float, default=0.2)
    parser.add_argument("--angular-speed", type=float, default=0.0)
    parser.add_argument("--nav-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.linear_speed < 0 or args.angular_speed < 0:
        parser.error("speed limits cannot be negative")
    if args.nav_timeout <= 0:
        parser.error("--nav-timeout must be greater than 0")
    return args


def print_plan(route_path: Path) -> None:
    with NavigationGraph(
        route_path,
        initial_node=INITIAL_NODE,
        navigation_mode="move_by",
    ) as graph:
        path = graph.shortestPath(TEST_NODE)
    print(f"初始节点：{INITIAL_NODE}")
    print(f"先导航到：{TEST_NODE}")
    print(f"规划路径：{' -> '.join(path)}")
    print("每轮微调：")
    for index, (x, y, theta) in enumerate(MICRO_ADJUSTMENTS, start=1):
        print(f"  {index}. moveBy(x={x:.2f}, y={y:.2f}, theta={theta:.1f})")
    print("  4. recoverState()")
    print(f"重复轮数：{ROUNDS}")


def print_live_pose(controller: Any, label: str) -> None:
    pose = map_pose_from_response(controller.get_pose())
    print(
        f"{label}: x={pose['x_m']:.3f} m, y={pose['y_m']:.3f} m, "
        f"yaw={pose['yaw_rad']:.4f} rad"
    )


def main() -> int:
    args = parse_args()
    route_path, _ = convert_scene(args.scene, initial_node=INITIAL_NODE)
    print_plan(route_path)
    if not args.execute:
        print("Dry-run 完成；增加 --execute 才会真实移动机器人。")
        return 0

    from agent.navigation.knewbot_chassis_controller_http import (
        KnewbotChassisController,
    )

    controller: Optional[Any] = None
    try:
        controller = KnewbotChassisController({"nav_timeout": args.nav_timeout})
        current_map_response = controller.get_current_map()
        initial_map_pose = map_pose_from_response(controller.get_pose())
        route_path, _ = convert_scene(
            args.scene,
            initial_node=INITIAL_NODE,
            current_map_response=current_map_response,
            initial_map_pose=initial_map_pose,
        )
        with NavigationGraph(
            route_path,
            controller=controller,
            initial_node=INITIAL_NODE,
            navigation_mode="move_by",
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
        ) as graph:
            print_live_pose(controller, "start_point 实测位姿")
            graph.moveTo(TEST_NODE, correct=True)
            print_live_pose(controller, "到达 hub_center")

            for round_index in range(1, ROUNDS + 1):
                print(f"\n恢复测试 {round_index}/{ROUNDS}")
                print_live_pose(controller, "微调前")
                for x, y, theta in MICRO_ADJUSTMENTS:
                    graph.moveBy(x, y, theta)
                print_live_pose(controller, "微调后")
                graph.recoverState()
                status, node = graph.getState()
                print_live_pose(controller, "恢复后")
                print(
                    f"恢复状态：state={status.value}, "
                    f"node={node.name if node else None}"
                )
                if node is None or node.name != TEST_NODE:
                    raise NavigationGraphError(
                        f"recovery did not return to {TEST_NODE}"
                    )

        print("\n三轮 moveBy 恢复测试完成。")
        return 0
    except (OSError, ValueError, NavigationGraphError) as exc:
        print(f"恢复测试失败：{exc}")
        return 1
    except KeyboardInterrupt:
        if controller is not None:
            try:
                controller.cancel_action()
            except Exception:
                pass
        print("\n测试已取消。")
        return 130
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
