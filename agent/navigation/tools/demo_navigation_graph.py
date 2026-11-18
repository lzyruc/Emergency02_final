#!/usr/bin/env python3
"""Interactively navigate to named GUI scene points with NavigationGraph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from ..data.motion_log import map_pose_from_response
    from ..navigation_graph import NavigationGraph, NavigationGraphError
    from ..scene_adapter import convert_scene
except ImportError:  # Direct script execution.
    from agent.navigation.data.motion_log import map_pose_from_response
    from agent.navigation.navigation_graph import NavigationGraph, NavigationGraphError
    from agent.navigation.scene_adapter import convert_scene


ROTATE_BEFORE_TRANSLATE_NODES = frozenset(
    {"recognize_left", "recognize_right"}
)
SHELF_LATERAL_HEADINGS = {"left": 270.0, "right": 90.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="GUI scene JSON")
    parser.add_argument("--output", type=Path, help="generated NavigationGraph route JSON")
    parser.add_argument(
        "--navigation-mode",
        choices=("track", "move_by"),
        default="move_by",
        help="graph-wide execution mode (default: move_by)",
    )
    parser.add_argument("--execute", action="store_true", help="actually move the chassis")
    parser.add_argument(
        "--nav-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for each chassis action (default: 30)",
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=0.2,
        help="move_by linear speed limit in m/s (default: 0.2; 0 means unlimited)",
    )
    parser.add_argument(
        "--angular-speed",
        type=float,
        default=0.0,
        help="move_by angular speed limit in rad/s (default: 0, unlimited)",
    )
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument(
        "--compress-path",
        dest="compress_path",
        action="store_true",
        help="merge collinear graph edges up to 3 m (default)",
    )
    compression.add_argument(
        "--no-compress-path",
        dest="compress_path",
        action="store_false",
        help="execute every original graph edge",
    )
    parser.set_defaults(compress_path=True)
    return parser.parse_args()


def _map_name(response: Mapping[str, Any]) -> str:
    for key in ("map_name", "name"):
        if response.get(key):
            return str(response[key])
    return "<unknown>"


def _scene_node_names(scene_path: Path) -> tuple[str, ...]:
    """Read the selectable node names without connecting to the chassis."""
    document = json.loads(
        Path(scene_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    robots = document.get("robots")
    if not isinstance(robots, list) or not robots:
        raise NavigationGraphError("GUI scene must contain a non-empty robots list")
    names = tuple(
        str(robot.get("name", "")).strip()
        for robot in robots
        if isinstance(robot, Mapping)
    )
    if len(names) != len(robots) or any(not name for name in names):
        raise NavigationGraphError("every GUI scene robot must have a non-empty name")
    if len(set(names)) != len(names):
        raise NavigationGraphError("GUI scene contains duplicate node names")
    return names


def _confirm_initial_node(scene_path: Path, *, execute: bool) -> Optional[str]:
    """Require two identical manual declarations of the physical start node."""
    node_names = _scene_node_names(scene_path)
    print("可选初始点位：")
    print("  " + ", ".join(node_names))
    if execute:
        print("警告：初始点位将用于实机坐标锚定；选择错误可能导致危险移动。")
    else:
        print("当前为 dry-run；仍需确认规划所使用的初始点位。")

    initial_node = input(
        "请输入机器人当前实际所在的初始点位名称（q 退出）："
    ).strip()
    if initial_node.lower() == "q":
        return None
    if initial_node not in node_names:
        raise NavigationGraphError(f"初始点位不存在：{initial_node!r}")

    confirmed_node = input(
        f"请再次输入 {initial_node!r} 以确认初始点位（输入不一致将中止）："
    ).strip()
    if confirmed_node != initial_node:
        raise NavigationGraphError(
            "两次输入的初始点位不一致，已中止；未连接或移动底盘"
        )
    print(f"初始点位已二次确认：{initial_node}")
    return initial_node


def _print_nodes(graph: NavigationGraph) -> None:
    for name in graph.node_order:
        pose = graph.nodes[name].pose
        print(
            f"  {name}: ({float(pose['x_m']):.3f}, "
            f"{float(pose['y_m']):.3f}, {float(pose['theta_deg']):.1f}°)"
        )


def _print_failure(diagnostics: Optional[Mapping[str, Any]]) -> None:
    print("\nNavigationGraph 失败诊断：")
    if not diagnostics:
        print("  未能采集额外诊断信息。")
        return
    print(
        f"  segment: {diagnostics.get('source_node')} -> "
        f"{diagnostics.get('target_node')}"
    )
    print(f"  final target: {diagnostics.get('final_target_node')}")
    if diagnostics.get("failure_category"):
        print(f"  category: {diagnostics['failure_category']}")
    if diagnostics.get("action_id"):
        print(f"  action_id: {diagnostics['action_id']}")
    if diagnostics.get("action_status") is not None:
        print(
            "  chassis action status: "
            + json.dumps(diagnostics["action_status"], ensure_ascii=False, default=str)
        )
    for label, key in (
        ("robot health", "robot_health"),
        ("localization", "localization_status"),
        ("current map", "current_map"),
    ):
        if diagnostics.get(key) is not None:
            print(
                f"  {label}: "
                + json.dumps(diagnostics[key], ensure_ascii=False, default=str)
            )
    print(f"  exception: {diagnostics.get('exception')}")
    if diagnostics.get("requested_action") is not None:
        print(
            "  requested action: "
            + json.dumps(diagnostics["requested_action"], ensure_ascii=False, default=str)
        )
    for label, key in (
        ("map before", "map_pose_before"),
        ("requested", "requested_map_pose"),
        ("map after", "map_pose_after"),
    ):
        pose = diagnostics.get(key)
        if isinstance(pose, Mapping):
            print(
                f"  {label}: x={float(pose['x_m']):.3f}, "
                f"y={float(pose['y_m']):.3f}, "
                f"yaw={float(pose['yaw_rad']):.4f} rad"
            )
    if diagnostics.get("target_position_error_m") is not None:
        print(
            f"  target error: {float(diagnostics['target_position_error_m']):.3f} m, "
            f"{float(diagnostics['target_yaw_error_deg']):.2f}°"
        )
    if diagnostics.get("measured_translation_m") is not None:
        print(
            f"  measured motion: {float(diagnostics['measured_translation_m']):.3f} m, "
            f"{float(diagnostics['measured_yaw_change_deg']):.2f}°"
        )
def _print_warnings(warnings: list[Mapping[str, Any]]) -> None:
    for warning in warnings:
        print("\nNavigationGraph 软成功提示：")
        print(f"  category: {warning.get('category')}")
        print(
            f"  segment: {warning.get('source_node')} -> "
            f"{warning.get('target_node')}"
        )
        print(f"  original action_id: {warning.get('original_action_id')}")
        if warning.get("category") == "move_by_translation_chunked_with_pose_feedback":
            chunks = warning.get("chunks", [])
            distances = ", ".join(
                f"{float(chunk['distance_m']):.3f}m" for chunk in chunks
            )
            print(
                f"  超过 3m 的平移已拆为 {len(chunks)} 段：{distances}；"
                "每段之间使用 get_pose 修正剩余方向和距离。"
            )
            print(
                f"  chunked translation final error: "
                f"{float(warning['target_position_error_m']):.3f} m"
            )


def main() -> int:
    args = parse_args()
    if args.nav_timeout <= 0:
        print("--nav-timeout 必须大于 0。")
        return 2
    if args.linear_speed < 0 or args.angular_speed < 0:
        print("--linear-speed 和 --angular-speed 不能为负数。")
        return 2
    controller: Optional[Any] = None
    current_map_response = None
    initial_map_pose = None
    try:
        initial_node = _confirm_initial_node(args.scene, execute=args.execute)
        if initial_node is None:
            print("已退出；未连接或移动底盘。")
            return 0

        if args.execute:
            from agent.navigation.knewbot_chassis_controller_http import (
                KnewbotChassisController,
            )

            controller = KnewbotChassisController(
                {"nav_timeout": args.nav_timeout}
            )
            current_map_response = controller.get_current_map()
            initial_map_pose = map_pose_from_response(controller.get_pose())

        route_path, graph_path = convert_scene(
            args.scene,
            output_path=args.output,
            initial_node=initial_node,
            current_map_response=current_map_response,
            initial_map_pose=initial_map_pose,
        )
        # A generated graph exists before construction, so NavigationGraph loads
        # the GUI connections instead of inferring sequential record edges.
        construction_mode = args.navigation_mode if args.execute else "move_by"
        with NavigationGraph(
            route_path,
            controller=controller,
            initial_node=initial_node,
            navigation_mode=construction_mode,
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            compress_path=args.compress_path,
            lateral_translation_headings=SHELF_LATERAL_HEADINGS,
        ) as graph:
            status, current = graph.getState()
            print(f"NavigationGraph：{route_path}")
            print(f"拓扑图：{graph_path}")
            print(f"节点/双向边：{len(graph.nodes)}/{sum(len(v) for v in graph.edges.values()) // 2}")
            print(f"导航模式：{args.navigation_mode}")
            print(f"单动作超时：{args.nav_timeout:g} s")
            print(f"跨节点共线合并：{'开启' if args.compress_path else '关闭'}")
            if args.navigation_mode == "move_by":
                print(
                    f"move_by 速度：linear={args.linear_speed:g} m/s, "
                    f"angular={args.angular_speed:g} rad/s"
                )
            print(f"当前节点：{current.name if current else None} ({status.value})")
            if args.execute:
                print(
                    f"SLAM 锚定：当前底盘 pose = {current.name if current else None}，"
                    f"map={_map_name(current_map_response)}"
                )
            elif not args.execute:
                print("当前为 dry-run，只计算路径；添加 --execute 后才会移动机器人。")

            while True:
                raw = input(
                    "请输入目标点位名称（list 查看，recover 点位名 恢复，q 退出）："
                ).strip()
                if raw.lower() == "q":
                    return 0
                if raw.lower() == "list":
                    _print_nodes(graph)
                    continue
                if raw.lower().startswith("recover "):
                    recovery_node = raw.split(maxsplit=1)[1].strip()
                    if recovery_node not in graph.nodes:
                        print(f"恢复点位不存在：{recovery_node}")
                        continue
                    if not args.execute:
                        print("dry-run 不执行恢复动作。")
                        continue
                    try:
                        graph.recoverState(recovery_node)
                    except Exception as exc:
                        print(f"恢复到 {recovery_node} 失败，但程序保持运行：{exc}")
                        continue
                    print(f"已恢复到节点：{recovery_node}")
                    continue
                state, _ = graph.getState()
                if state.value == "failed":
                    print("当前状态为 FAILED；请先输入 recover 点位名，或输入 q 退出。")
                    continue
                if raw not in graph.nodes:
                    print(f"点位不存在：{raw}；输入 list 查看全部点位。")
                    continue
                path = graph.shortestPath(raw)
                print(f"规划路径：{' -> '.join(path)}")
                execution_path = graph.executionPath(raw)
                if args.navigation_mode != "track" and execution_path != path:
                    print(f"合并后执行关键点：{' -> '.join(execution_path)}")
                if not args.execute:
                    continue
                try:
                    graph.moveTo(
                        raw,
                        correct=False,
                        rotate_before_translate=(
                            raw in ROTATE_BEFORE_TRANSLATE_NODES
                        ),
                    )
                except Exception as exc:
                    _print_failure(graph.getLastFailure())
                    print(f"本次导航失败，但程序保持运行：{exc}")
                    print("请确认机器人位置后输入 recover 点位名，或输入 q 退出。")
                    continue
                _print_warnings(graph.getLastWarnings())
                state, node = graph.getState()
                print(f"到达：state={state.value}, node={node.name if node else None}")
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        return 130
    except (OSError, ValueError, NavigationGraphError) as exc:
        print(f"NavigationGraph 演示失败：{exc}")
        return 1
    except Exception as exc:
        print(f"底盘动作失败：{exc}")
        return 1
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
