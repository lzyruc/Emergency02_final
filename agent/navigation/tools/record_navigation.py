#!/usr/bin/env python3
"""Interactively execute and record a reproducible move-by route."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ..data.motion_log import (
        advance_pose,
        json_safe,
        map_pose_from_response,
        new_route,
        normalize_radians,
        utc_now,
        validate_point_name,
        write_json_atomic,
    )
    from ..rotation_control import rotate_by_with_feedback
except ImportError:  # Direct script execution.
    from agent.navigation.data.motion_log import (
        advance_pose,
        json_safe,
        map_pose_from_response,
        new_route,
        normalize_radians,
        utc_now,
        validate_point_name,
        write_json_atomic,
    )
    from agent.navigation.rotation_control import rotate_by_with_feedback


LOGICAL_X_MIN_M = 0.0
LOGICAL_X_MAX_M = 3.7
LOGICAL_Y_MIN_M = -2.7
LOGICAL_Y_MAX_M = 2.7
DEFAULT_LINEAR_SPEED_MPS = 0.2


def validate_logical_pose_bounds(pose: Dict[str, float]) -> None:
    """Reject a logical pose outside the configured rectangular work area."""
    x_m = float(pose["x_m"])
    y_m = float(pose["y_m"])
    if not LOGICAL_X_MIN_M <= x_m <= LOGICAL_X_MAX_M:
        raise ValueError(
            f"目标逻辑 x={x_m:.3f} m 越界；允许范围为 "
            f"[{LOGICAL_X_MIN_M:.1f}, {LOGICAL_X_MAX_M:.1f}] m"
        )
    if not LOGICAL_Y_MIN_M <= y_m <= LOGICAL_Y_MAX_M:
        raise ValueError(
            f"目标逻辑 y={y_m:.3f} m 越界；允许范围为 "
            f"[{LOGICAL_Y_MIN_M:.1f}, {LOGICAL_Y_MAX_M:.1f}] m"
        )


def parse_args() -> argparse.Namespace:
    default_name = datetime.now().strftime("route_%Y%m%d_%H%M%S.json")
    default_output = Path(__file__).parents[1] / "data" / "logs" / default_name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=default_output)
    parser.add_argument("--host", default="192.168.11.10")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=DEFAULT_LINEAR_SPEED_MPS,
        help=f"maximum translation speed in m/s (default: {DEFAULT_LINEAR_SPEED_MPS})",
    )
    parser.add_argument("--angular-speed", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true", help="actually send move_by commands")
    args = parser.parse_args()
    if args.linear_speed < 0 or args.angular_speed < 0:
        parser.error("speed limits cannot be negative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def ask_yes_no(prompt: str) -> bool:
    """Read a Y/N response."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("请输入 Y/y 或 N/n。")


def ask_point_name(existing_names: Set[str]) -> str:
    """Read one unique canonical point name."""
    while True:
        try:
            name = validate_point_name(input("当前点位名称(按照规则)："))
        except ValueError as exc:
            print(exc)
            continue
        if name in existing_names:
            print(f"点位名称已存在：{name}")
            continue
        return name


def ask_action() -> Optional[Tuple[float, float, float]]:
    """Read a relative direction, distance, and counterclockwise rotation."""
    while True:
        raw = input(
            "请用空格分割，输入移动方向（0～360°）、距离（米）、"
            "旋转角度（逆时针为正，q结束）："
        ).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        parts = raw.split()
        if len(parts) != 3:
            print("请输入三个数字，例如：90 1.0 0")
            continue
        try:
            direction_deg, distance_m, rotation_deg = map(float, parts)
        except ValueError:
            print("距离和角度必须是数字。")
            continue
        if not all(math.isfinite(value) for value in (direction_deg, distance_m, rotation_deg)):
            print("方向、距离和角度必须是有限数值。")
            continue
        if not 0.0 <= direction_deg <= 360.0:
            print("移动方向必须在 0～360 度之间。")
            continue
        if distance_m < 0:
            print("移动距离不能为负；请使用 direction 指定移动方向。")
            continue
        if distance_m != 0 and not 0.05 <= distance_m <= 5.0:
            print("单步前进距离必须为 0，或处于 0.05～5.0 米之间。")
            continue
        if not -180.0 <= rotation_deg <= 180.0:
            print("单步旋转角度必须在 -180 到 180 度之间。")
            continue
        if distance_m == 0 and rotation_deg == 0:
            print("距离和旋转角度不能同时为 0。")
            continue
        return direction_deg, distance_m, rotation_deg


def build_commands(
    logical_pose: Dict[str, float],
    direction_deg: float,
    distance_m: float,
    rotation_deg: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Build exact move-by commands and their logical before/after poses."""
    commands: List[Dict[str, Any]] = []
    current = dict(logical_pose)
    distance_mm = round(distance_m * 1000)
    if distance_mm:
        after_translation = advance_pose(current, distance_mm / 1000.0, 0.0, direction_deg)
        direction_rad = normalize_radians(math.radians(direction_deg))
        commands.append(
            {
                "type": "translate",
                "method": "move_by",
                "move_mode": 0,
                "direction_deg_relative": direction_deg % 360.0,
                "direction_rad": direction_rad,
                "distance_mm": distance_mm,
                "logical_pose_before": current,
                "logical_pose_after": after_translation,
            }
        )
        current = after_translation
    if rotation_deg:
        after_rotation = advance_pose(current, 0.0, rotation_deg)
        commands.append(
            {
                "type": "rotate",
                "method": "move_by_with_pose_feedback",
                "move_mode": 1,
                "direction_rad": math.radians(rotation_deg),
                "distance_mm": 0,
                "logical_pose_before": current,
                "logical_pose_after": after_rotation,
            }
        )
        current = after_rotation
    return commands, current


def recover_map_pose(
    controller: Any,
    target: Dict[str, float],
    command: Dict[str, Any],
    route: Dict[str, Any],
    output: Path,
) -> None:
    """Best-effort return to the map pose captured before a failed command."""
    recovery: Dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "target_map_pose": dict(target),
        "method": "move_to",
    }
    command["recovery"] = recovery
    write_json_atomic(output, route)
    try:
        result = controller.move_to(
            target["x_m"],
            target["y_m"],
            target["yaw_rad"],
            flags=0,
            wait=True,
        )
        recovery["controller_result"] = json_safe(result)
        recovery["map_pose_after"] = map_pose_from_response(controller.get_pose())
        recovery["status"] = "completed"
    except Exception as exc:
        recovery["status"] = "failed"
        recovery["error"] = repr(exc)
        raise
    finally:
        recovery["finished_at"] = utc_now()
        write_json_atomic(output, route)


def execute_and_record_commands(
    controller: Any,
    commands: List[Dict[str, Any]],
    route: Dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    """Execute commands, persisting map poses and status around every command."""
    for command in commands:
        execution: Dict[str, Any] = {
            "status": "pending",
            "started_at": utc_now(),
            "map_pose_before": map_pose_from_response(controller.get_pose()),
        }
        command["execution"] = execution
        write_json_atomic(output, route)
        try:
            if command["type"] == "translate":
                result = controller.move_by(
                    direction=command["direction_rad"],
                    distance=command["distance_mm"],
                    move_mode=command["move_mode"],
                    v=args.linear_speed,
                    w=args.angular_speed,
                    wait=True,
                )
            else:
                before = execution["map_pose_before"]

                def persist_rotation(feedback: Dict[str, Any]) -> None:
                    execution["rotation_feedback"] = json_safe(feedback)
                    write_json_atomic(output, route)

                result = rotate_by_with_feedback(
                    controller,
                    float(command["direction_rad"]),
                    angular_speed=args.angular_speed,
                    initial_map_pose=before,
                    on_update=persist_rotation,
                )
            execution["controller_result"] = json_safe(result)
            execution["map_pose_after"] = map_pose_from_response(controller.get_pose())
            execution["status"] = "completed"
            execution["finished_at"] = utc_now()
            write_json_atomic(output, route)
        except Exception as exc:
            execution["status"] = "failed"
            execution["finished_at"] = utc_now()
            execution["error"] = repr(exc)
            write_json_atomic(output, route)
            try:
                recover_map_pose(controller, execution["map_pose_before"], command, route, output)
            except Exception as recovery_exc:
                raise RuntimeError(f"command failed and recovery also failed: {recovery_exc}") from exc
            raise RuntimeError("command failed; chassis returned to its pre-command map pose") from exc


def print_pose(pose: Dict[str, float]) -> None:
    """Print one logical pose."""
    print(
        f"当前逻辑位姿：x={pose['x_m']:.3f} m, y={pose['y_m']:.3f} m, "
        f"theta={pose['theta_deg']:.2f}°"
    )


def main() -> int:
    """Run the interactive recorder."""
    args = parse_args()
    route = new_route()
    route["execution_defaults"] = {
        "host": args.host,
        "port": args.port,
        "linear_speed_mps": args.linear_speed,
        "angular_speed_radps": args.angular_speed,
    }
    route["recording_mode"] = "execute" if args.execute else "dry_run"
    logical_pose: Dict[str, float] = {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0}
    existing_names: Set[str] = set()
    controller = None

    if args.execute:
        try:
            from agent.navigation.knewbot_chassis_controller_http import (
                KnewbotChassisController,
            )
        except ModuleNotFoundError as exc:
            print(f"无法加载底盘控制器：{exc}")
            return 2
        controller = KnewbotChassisController({"host": args.host, "rest_port": args.port})
        if not controller.is_healthy():
            controller.close()
            print("底盘健康检查失败，拒绝执行。")
            return 2
        if not controller.is_localized():
            controller.close()
            print("底盘未正常定位，无法记录 map 位姿和执行失败恢复。")
            return 2
        route["map_context"] = {
            "current_map_response": json_safe(controller.get_current_map()),
            "initial_map_pose": map_pose_from_response(controller.get_pose()),
        }

    print(f"记录文件：{args.output}")
    print("逻辑坐标：前方 +x，左方 +y，逆时针为正，绝对角度范围 0～359°。")
    print("安全边界：0.0 ≤ x ≤ 3.7 m，-2.7 ≤ y ≤ 2.7 m；越界动作不会执行。")
    print(
        "每一步动作顺序：先按 direction 平移，再使用带 map yaw 反馈修正的 move_by 原地旋转。"
        "输入 q 结束并保存。"
    )
    if not args.execute:
        print("当前为 dry-run，不会向底盘发送命令；增加 --execute 才会真实移动。")
    write_json_atomic(args.output, route)

    try:
        while True:
            print_pose(logical_pose)
            if ask_yes_no("当前点位是否需要记录（Y/y/N/n）："):
                name = ask_point_name(existing_names)
                route["points"].append(
                    {
                        "sequence": len(route["points"]),
                        "name": name,
                        "logical_pose": dict(logical_pose),
                        "before_action": len(route["actions"]),
                        "recorded_at": utc_now(),
                    }
                )
                existing_names.add(name)
                write_json_atomic(args.output, route)
                print(f"已记录点位：{name}")

            requested = ask_action()
            if requested is None:
                break
            direction_deg, distance_m, rotation_deg = requested
            commands, next_logical_pose = build_commands(
                logical_pose,
                direction_deg,
                distance_m,
                rotation_deg,
            )
            try:
                validate_logical_pose_bounds(next_logical_pose)
            except ValueError as exc:
                print(f"拒绝执行：{exc}")
                continue
            action: Dict[str, Any] = {
                "sequence": len(route["actions"]),
                "status": "pending" if args.execute else "planned",
                "started_at": utc_now(),
                "logical_pose_before": dict(logical_pose),
                "requested": {
                    "direction_deg_relative": direction_deg % 360.0,
                    "distance_m": distance_m,
                    "rotation_deg_ccw": rotation_deg,
                },
                "commands": commands,
                "logical_expected_pose_after": dict(next_logical_pose),
            }
            route["actions"].append(action)
            write_json_atomic(args.output, route)

            try:
                if controller is not None:
                    execute_and_record_commands(controller, commands, route, args.output, args)
                    action["status"] = "completed"
                action["finished_at"] = utc_now()
                logical_pose = next_logical_pose
                write_json_atomic(args.output, route)
                print("next action")
            except Exception as exc:
                action["status"] = "failed"
                action["finished_at"] = utc_now()
                action["error"] = repr(exc)
                write_json_atomic(args.output, route)
                raise
    except (EOFError, KeyboardInterrupt):
        print("\n记录已中断，已保存当前日志。")
        return 130
    except Exception as exc:
        print(f"动作执行失败，日志已保存：{exc}")
        return 1
    finally:
        if controller is not None:
            controller.close()

    print(f"记录完成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
