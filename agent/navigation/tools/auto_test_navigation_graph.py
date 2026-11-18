#!/usr/bin/env python3
"""Run the fixed shop navigation acceptance sequence with live stop input."""

from __future__ import annotations

import argparse
import math
import re
import select
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional

try:
    from ..data.motion_log import map_pose_from_response, normalize_radians
    from ..navigation_graph import NavigationGraph, NavigationGraphError
    from ..scene_adapter import convert_scene
    from .demo_navigation_graph import _print_failure, _print_warnings
except ImportError:  # Direct script execution.
    from agent.navigation.data.motion_log import (
        map_pose_from_response,
        normalize_radians,
    )
    from agent.navigation.navigation_graph import NavigationGraph, NavigationGraphError
    from agent.navigation.scene_adapter import convert_scene
    from agent.navigation.tools.demo_navigation_graph import (
        _print_failure,
        _print_warnings,
    )


DEFAULT_SCENE = (
    Path(__file__).resolve().parent / "graphGUI" / "shop" / "scene_shop.json"
)

TEST_SEQUENCE = (
    "start_point",
    "recognize_left",
    "center3",
    "recognize_left",
    "right6",
    "recognize_left",
    "start_point",
    "left6",
    "center6",
    "right6",
    "recognize_right",
    "left8",
    "recognize_right",
    "center2",
    "start_point",
    "left3",
    "center11",
    "left3",
    "start_point",
)

CRITICAL_CORRECTION_NODES = frozenset(
    {"start_point", "hub_left", "hub_right", "hub_center"}
)
ROTATE_BEFORE_TRANSLATE_NODES = frozenset(
    {"recognize_left", "recognize_right"}
)
SHELF_LATERAL_HEADINGS = {"left": 270.0, "right": 90.0}
CORRECTION_SOFT_SUCCESS_POSITION_M = 0.05
CORRECTION_SOFT_SUCCESS_YAW_DEG = 1.0
_ACTION_ID_PATTERN = re.compile(r"\bAction\s+([0-9A-Za-z-]+)")


def parse_args(
    *,
    default_scene: Path = DEFAULT_SCENE,
    description: Optional[str] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("scene", nargs="?", type=Path, default=default_scene)
    parser.add_argument(
        "--navigation-mode",
        choices=("move_by",),
        default="move_by",
        help="execution mode; this acceptance test only supports move_by",
    )
    parser.add_argument("--execute", action="store_true", help="actually move the chassis")
    parser.add_argument("--nav-timeout", type=float, default=30.0)
    parser.add_argument("--linear-speed", type=float, default=0.2)
    parser.add_argument("--angular-speed", type=float, default=0.0)
    args = parser.parse_args()
    if args.nav_timeout <= 0:
        parser.error("--nav-timeout must be greater than 0")
    if args.linear_speed < 0 or args.angular_speed < 0:
        parser.error("speed limits cannot be negative")
    return args


def _validate_sequence(
    graph: NavigationGraph,
    test_sequence: tuple[str, ...] = TEST_SEQUENCE,
) -> None:
    missing = [name for name in test_sequence if name not in graph.nodes]
    if missing:
        raise NavigationGraphError(
            "test sequence contains unknown nodes: " + ", ".join(sorted(set(missing)))
        )


def _print_dry_run(
    route_path: Path,
    test_sequence: tuple[str, ...] = TEST_SEQUENCE,
    *,
    compress_path: bool = True,
) -> None:
    print("固定测试序列：")
    print(" -> ".join(test_sequence))
    for source, target in zip(test_sequence, test_sequence[1:]):
        with NavigationGraph(
            route_path,
            initial_node=source,
            navigation_mode="move_by",
            compress_path=compress_path,
        ) as graph:
            _validate_sequence(graph, test_sequence)
            path = graph.shortestPath(target)
            execution_path = graph.executionPath(target)
        print(f"  {source} -> {target}: {' -> '.join(path)}")
        if execution_path != path:
            print(f"    合并后：{' -> '.join(execution_path)}")


def _cancel_all(controller: Any, reason: str) -> None:
    try:
        response = controller.cancel_action()
        print(f"{reason}：已发送 cancel_action([])：{response}", flush=True)
    except Exception as exc:
        print(f"{reason}：取消底盘 action 失败：{exc}", flush=True)


def _is_chassis_action_failed(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    value = getattr(code, "value", code)
    return value == 2001


def _failed_action_summary(controller: Any, exc: BaseException) -> str:
    match = _ACTION_ID_PATTERN.search(str(exc))
    if not match:
        return "action_id=<unknown>"
    action_id = match.group(1)
    try:
        status = controller.get_action_status(action_id)
    except Exception as status_exc:
        return f"action_id={action_id}, status_query_failed={status_exc}"
    return f"action_id={action_id}, chassis_status={status}"


def refine_start_pose_with_move_by(
    controller: Any,
    target_map_pose: dict[str, float],
    stop_event: threading.Event,
    *,
    rounds: int = 2,
    point_name: str = "start_point",
    linear_speed: float = 0.2,
    angular_speed: float = 0.0,
    snap_yaw_to_nearest_quarter_turn: bool = False,
) -> dict[str, float]:
    """Iteratively correct the live map pose to one fixed map target."""
    target_map_pose = dict(target_map_pose)
    target_yaw_selected = not snap_yaw_to_nearest_quarter_turn
    for round_index in range(1, rounds + 1):
        current = map_pose_from_response(controller.get_pose())
        if not target_yaw_selected:
            quarter_turn = math.pi / 2.0
            offset = normalize_radians(
                current["yaw_rad"] - float(target_map_pose["yaw_rad"])
            )
            target_map_pose["yaw_rad"] = normalize_radians(
                float(target_map_pose["yaw_rad"])
                + round(offset / quarter_turn) * quarter_turn
            )
            target_yaw_selected = True
            print(
                f"  {point_name} 朝向校正目标：最近的逻辑正交方向 "
                f"{math.degrees(target_map_pose['yaw_rad']):.2f}° (map)",
                flush=True,
            )
        dx = float(target_map_pose["x_m"]) - current["x_m"]
        dy = float(target_map_pose["y_m"]) - current["y_m"]
        position_error = math.hypot(dx, dy)
        print(
            f"  {point_name} 校正 {round_index}/{rounds}: "
            f"position_error={position_error:.3f} m",
            flush=True,
        )
        if stop_event.is_set():
            return current
        if position_error >= 0.02:
            world_direction = math.atan2(dy, dx)
            body_direction = normalize_radians(world_direction - current["yaw_rad"])
            try:
                controller.move_by(
                    direction=body_direction,
                    distance=round(min(position_error, 3.0) * 1000),
                    move_mode=0,
                    v=linear_speed,
                    w=angular_speed,
                    wait=True,
                )
            except Exception as exc:
                if not _is_chassis_action_failed(exc):
                    raise
                measured = map_pose_from_response(controller.get_pose())
                measured_error = math.hypot(
                    measured["x_m"] - float(target_map_pose["x_m"]),
                    measured["y_m"] - float(target_map_pose["y_m"]),
                )
                if measured_error > CORRECTION_SOFT_SUCCESS_POSITION_M:
                    raise
                print(
                    f"  {point_name} 平移软成功：底盘报告 2001，但实测位置误差 "
                    f"{measured_error:.3f} m <= "
                    f"{CORRECTION_SOFT_SUCCESS_POSITION_M:.3f} m；继续校正。\n"
                    f"    {_failed_action_summary(controller, exc)}",
                    flush=True,
                )

        # Translation can slightly disturb yaw, so always sample again before
        # calculating this round's shortest signed rotation correction.
        current = map_pose_from_response(controller.get_pose())
        yaw_error = normalize_radians(
            float(target_map_pose["yaw_rad"]) - current["yaw_rad"]
        )
        print(
            f"  {point_name} 校正 {round_index}/{rounds}: "
            f"yaw_error={math.degrees(yaw_error):.2f}°",
            flush=True,
        )
        if stop_event.is_set():
            return current
        if abs(yaw_error) >= math.radians(1.0):
            try:
                controller.move_by(
                    direction=yaw_error,
                    distance=0,
                    move_mode=1,
                    v=linear_speed,
                    w=angular_speed,
                    wait=True,
                )
            except Exception as exc:
                if not _is_chassis_action_failed(exc):
                    raise
                measured = map_pose_from_response(controller.get_pose())
                measured_error_deg = abs(
                    math.degrees(
                        normalize_radians(
                            measured["yaw_rad"]
                            - float(target_map_pose["yaw_rad"])
                        )
                    )
                )
                if measured_error_deg > CORRECTION_SOFT_SUCCESS_YAW_DEG:
                    raise
                print(
                    f"  {point_name} 旋转软成功：底盘报告 2001，但实测角度误差 "
                    f"{measured_error_deg:.2f}° <= "
                    f"{CORRECTION_SOFT_SUCCESS_YAW_DEG:.2f}°；继续校正。\n"
                    f"    {_failed_action_summary(controller, exc)}",
                    flush=True,
                )

    final_pose = map_pose_from_response(controller.get_pose())
    final_position_error = math.hypot(
        final_pose["x_m"] - float(target_map_pose["x_m"]),
        final_pose["y_m"] - float(target_map_pose["y_m"]),
    )
    final_yaw_error = abs(
        math.degrees(
            normalize_radians(
                final_pose["yaw_rad"] - float(target_map_pose["yaw_rad"])
            )
        )
    )
    print(
        f"  {point_name} {rounds}轮校正完成："
        f"position_error={final_position_error:.3f} m, "
        f"yaw_error={final_yaw_error:.2f}°",
        flush=True,
    )
    return final_pose


def _run_sequence(
    graph: NavigationGraph,
    controller: Any,
    correction_targets: dict[str, dict[str, float]],
    linear_speed: float,
    angular_speed: float,
    stop_event: threading.Event,
    errors: List[BaseException],
    test_sequence: tuple[str, ...] = TEST_SEQUENCE,
    critical_correction_nodes: frozenset[str] = CRITICAL_CORRECTION_NODES,
    rotate_before_translate_nodes: frozenset[str] = (
        ROTATE_BEFORE_TRANSLATE_NODES
    ),
    snap_correction_yaw_to_nearest_quarter_turn: bool = True,
) -> None:
    try:
        total = len(test_sequence) - 1
        for index, target in enumerate(test_sequence[1:], start=1):
            if stop_event.is_set():
                print("测试序列已停止，不再发送后续目标。", flush=True)
                return
            path = graph.shortestPath(target)
            execution_path = graph.executionPath(target)
            print(
                f"\n[{index}/{total}] 目标：{target}\n"
                f"规划路径：{' -> '.join(path)}",
                flush=True,
            )
            if execution_path != path:
                print(f"合并后执行关键点：{' -> '.join(execution_path)}", flush=True)
            try:
                checkpoints = [
                    name
                    for name in execution_path[1:]
                    if name in critical_correction_nodes or name == target
                ]
                for checkpoint in checkpoints:
                    graph.moveTo(
                        checkpoint,
                        correct=False,
                        rotate_before_translate=(
                            checkpoint in rotate_before_translate_nodes
                        ),
                    )
                    _print_warnings(graph.getLastWarnings())
                    if checkpoint not in critical_correction_nodes:
                        continue
                    if stop_event.is_set():
                        print(
                            f"测试序列已停止，跳过 {checkpoint} 校正。",
                            flush=True,
                        )
                        return
                    print(
                        f"已按图导航到达 {checkpoint}；"
                        "开始做两轮 get_pose + move_by 校正。",
                        flush=True,
                    )
                    refine_start_pose_with_move_by(
                        controller,
                        correction_targets[checkpoint],
                        stop_event,
                        rounds=2,
                        point_name=checkpoint,
                        linear_speed=linear_speed,
                        angular_speed=angular_speed,
                        snap_yaw_to_nearest_quarter_turn=(
                            snap_correction_yaw_to_nearest_quarter_turn
                            and checkpoint != test_sequence[0]
                        ),
                    )
            except Exception as exc:
                if stop_event.is_set():
                    print(f"当前动作已被停止：{exc}", flush=True)
                    return
                _print_failure(graph.getLastFailure())
                raise
            status, node = graph.getState()
            print(
                f"到达：state={status.value}, node={node.name if node else None}",
                flush=True,
            )
        print("\n固定测试序列已全部完成；输入 q 退出。", flush=True)
    except BaseException as exc:
        errors.append(exc)
        print(f"\n自动测试失败，已停止后续序列：{exc}", flush=True)


def _control_loop(
    worker: threading.Thread,
    stop_event: threading.Event,
    cancel_controller: Any,
) -> bool:
    """Return True for a normal q exit, False for EOF/interrupt."""
    print(
        "控制命令：输入 stop 取消当前导航并停止后续序列；输入 q 取消导航并退出。",
        flush=True,
    )
    while True:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        except (OSError, ValueError):
            readable = [sys.stdin]
        if not readable:
            continue
        raw = sys.stdin.readline()
        if raw == "":
            stop_event.set()
            _cancel_all(cancel_controller, "输入流已关闭")
            return False
        command = raw.strip().lower()
        if command == "stop":
            if stop_event.is_set() or not worker.is_alive():
                print("测试序列已经停止。输入 q 退出。", flush=True)
                continue
            stop_event.set()
            _cancel_all(cancel_controller, "STOP")
            print("已停止后续序列；输入 q 退出。", flush=True)
            continue
        if command == "q":
            stop_event.set()
            if worker.is_alive():
                _cancel_all(cancel_controller, "退出")
            return True
        if command:
            print("未知命令；请输入 stop 或 q。", flush=True)


def main(
    *,
    default_scene: Path = DEFAULT_SCENE,
    test_sequence: tuple[str, ...] = TEST_SEQUENCE,
    critical_correction_nodes: frozenset[str] = CRITICAL_CORRECTION_NODES,
    rotate_before_translate_nodes: frozenset[str] = ROTATE_BEFORE_TRANSLATE_NODES,
    lateral_translation_headings: Optional[dict[str, float]] = None,
    compress_path: bool = True,
    snap_correction_yaw_to_nearest_quarter_turn: bool = True,
    description: Optional[str] = None,
) -> int:
    args = parse_args(default_scene=default_scene, description=description)
    if not args.execute:
        with tempfile.TemporaryDirectory(prefix="navigation-dry-run-") as directory:
            route_path, _ = convert_scene(
                args.scene,
                output_path=Path(directory) / "scene.navigation.json",
                initial_node=test_sequence[0],
            )
            _print_dry_run(
                route_path,
                test_sequence,
                compress_path=compress_path,
            )
        print("Dry-run 完成；增加 --execute 才会真实移动机器人。")
        return 0

    from agent.navigation.knewbot_chassis_controller_http import (
        KnewbotChassisController,
    )

    config = {"nav_timeout": args.nav_timeout}
    motion_controller: Optional[Any] = None
    cancel_controller: Optional[Any] = None
    worker: Optional[threading.Thread] = None
    stop_event = threading.Event()
    errors: List[BaseException] = []
    try:
        motion_controller = KnewbotChassisController(config)
        cancel_controller = KnewbotChassisController(config)
        current_map_response = motion_controller.get_current_map()
        initial_map_pose = map_pose_from_response(motion_controller.get_pose())
        route_path, _ = convert_scene(
            args.scene,
            initial_node=test_sequence[0],
            current_map_response=current_map_response,
            initial_map_pose=initial_map_pose,
        )
        with NavigationGraph(
            route_path,
            controller=motion_controller,
            initial_node=test_sequence[0],
            navigation_mode="move_by",
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            compress_path=compress_path,
            lateral_translation_headings=(
                SHELF_LATERAL_HEADINGS
                if lateral_translation_headings is None
                else lateral_translation_headings
            ),
        ) as graph:
            _validate_sequence(graph, test_sequence)
            correction_targets = {
                name: graph.getNodeMapPose(name)
                for name in critical_correction_nodes
            }
            print("固定测试序列：")
            print(" -> ".join(test_sequence))
            print(
                f"导航模式：move_by；linear={args.linear_speed:g} m/s；"
                f"angular={args.angular_speed:g} rad/s；timeout={args.nav_timeout:g} s"
            )
            worker = threading.Thread(
                target=_run_sequence,
                args=(
                    graph,
                    motion_controller,
                    correction_targets,
                    args.linear_speed,
                    args.angular_speed,
                    stop_event,
                    errors,
                    test_sequence,
                    critical_correction_nodes,
                    rotate_before_translate_nodes,
                    snap_correction_yaw_to_nearest_quarter_turn,
                ),
                name="navigation-auto-test",
                daemon=True,
            )
            worker.start()
            _control_loop(worker, stop_event, cancel_controller)
            worker.join(timeout=args.nav_timeout + 2.0)
            if worker.is_alive():
                print("导航线程尚未结束，请使用硬件急停并检查底盘。")
                return 2
        return 1 if errors else 0
    except (OSError, ValueError, NavigationGraphError) as exc:
        print(f"自动测试启动失败：{exc}")
        return 1
    except KeyboardInterrupt:
        stop_event.set()
        if cancel_controller is not None:
            _cancel_all(cancel_controller, "Ctrl-C")
        return 130
    finally:
        if cancel_controller is not None:
            cancel_controller.close()
        if motion_controller is not None:
            motion_controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
