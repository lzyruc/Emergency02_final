#!/usr/bin/env python3
"""Validate, preview, and replay a recorded move-by route."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Dict, Optional

try:
    from ..data.motion_log import (
        json_safe,
        map_pose_from_response,
        normalize_radians,
        read_json,
        utc_now,
        validate_route,
        write_json_atomic,
    )
    from ..rotation_control import rotate_by_with_feedback
except ImportError:  # Direct script execution.
    from agent.navigation.data.motion_log import (
        json_safe,
        map_pose_from_response,
        normalize_radians,
        read_json,
        utc_now,
        validate_route,
        write_json_atomic,
    )
    from agent.navigation.rotation_control import rotate_by_with_feedback


_ACTION_ID_PATTERN = re.compile(r"\bAction\s+([0-9A-Za-z-]+)")


def collect_failure_diagnostics(
    controller: Any,
    command_log: Dict[str, Any],
    exc: BaseException,
) -> Dict[str, Any]:
    """Capture best-effort chassis state and geometric error after a failed command."""
    command = command_log["command"]
    before = command_log["map_pose_before"]
    diagnostics: Dict[str, Any] = {
        "captured_at": utc_now(),
        "exception": repr(exc),
    }

    match = _ACTION_ID_PATTERN.search(str(exc))
    if match:
        action_id = match.group(1)
        diagnostics["action_id"] = action_id
        try:
            diagnostics["controller_action_status"] = json_safe(
                controller.get_action_status(action_id)
            )
        except Exception as status_exc:
            diagnostics["controller_action_status_error"] = repr(status_exc)

    try:
        after = map_pose_from_response(controller.get_pose())
        diagnostics["map_pose_after_failure"] = after
    except Exception as pose_exc:
        diagnostics["map_pose_after_failure_error"] = repr(pose_exc)
        return diagnostics

    dx = after["x_m"] - before["x_m"]
    dy = after["y_m"] - before["y_m"]
    diagnostics["measured_motion"] = {
        "dx_m": dx,
        "dy_m": dy,
        "translation_m": math.hypot(dx, dy),
        "yaw_change_deg": math.degrees(
            normalize_radians(after["yaw_rad"] - before["yaw_rad"])
        ),
    }

    if command["type"] == "translate":
        requested_distance = int(command["distance_mm"]) / 1000.0
        travel_yaw = before["yaw_rad"] + float(command["direction_rad"])
        cos_yaw = math.cos(travel_yaw)
        sin_yaw = math.sin(travel_yaw)
        expected_x = before["x_m"] + requested_distance * cos_yaw
        expected_y = before["y_m"] + requested_distance * sin_yaw
        diagnostics["translation_analysis"] = {
            "requested_distance_m": requested_distance,
            "actual_forward_m": dx * cos_yaw + dy * sin_yaw,
            "actual_lateral_m": -dx * sin_yaw + dy * cos_yaw,
            "expected_target_map_pose": {
                "x_m": expected_x,
                "y_m": expected_y,
                "yaw_rad": before["yaw_rad"],
            },
            "target_position_error_m": math.hypot(
                after["x_m"] - expected_x,
                after["y_m"] - expected_y,
            ),
        }
    else:
        target_yaw = normalize_radians(
            before["yaw_rad"] + float(command["direction_rad"])
        )
        diagnostics["rotation_analysis"] = {
            "requested_rotation_deg": math.degrees(float(command["direction_rad"])),
            "target_yaw_rad": target_yaw,
            "final_yaw_error_deg": math.degrees(
                normalize_radians(target_yaw - after["yaw_rad"])
            ),
        }
    return diagnostics


def print_failure_diagnostics(
    source_sequence: int,
    command_log: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    """Print a concise operator-facing failure report."""
    command = command_log["command"]
    print("\n失败动作诊断：")
    print(
        f"  source action={source_sequence}, command={command_log['index']}, "
        f"type={command['type']}, method={command['method']}"
    )
    if diagnostics.get("action_id"):
        print(f"  action_id={diagnostics['action_id']}")
    status = diagnostics.get("controller_action_status")
    if isinstance(status, dict):
        print(
            f"  chassis status={status.get('status')}, type={status.get('type')}, "
            f"distance_remaining={status.get('distance_remaining')}"
        )
    before = command_log["map_pose_before"]
    print(
        f"  map before: x={before['x_m']:.3f}, y={before['y_m']:.3f}, "
        f"yaw={math.degrees(before['yaw_rad']):.2f}°"
    )
    after = diagnostics.get("map_pose_after_failure")
    if isinstance(after, dict):
        print(
            f"  map after:  x={after['x_m']:.3f}, y={after['y_m']:.3f}, "
            f"yaw={math.degrees(after['yaw_rad']):.2f}°"
        )
    translation = diagnostics.get("translation_analysis")
    if isinstance(translation, dict):
        print(
            f"  translation: requested={translation['requested_distance_m']:.3f} m, "
            f"forward={translation['actual_forward_m']:.3f} m, "
            f"lateral={translation['actual_lateral_m']:.3f} m, "
            f"target_error={translation['target_position_error_m']:.3f} m"
        )
    rotation = diagnostics.get("rotation_analysis")
    if isinstance(rotation, dict):
        print(
            f"  rotation: requested={rotation['requested_rotation_deg']:.2f}°, "
            f"final_error={rotation['final_yaw_error_deg']:.2f}°"
        )
    print("  已停止重放；请先确认机器人真实位置，再决定续播动作编号。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--host", default="192.168.11.10")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--start-action", type=int, default=0)
    parser.add_argument("--stop-action", type=int)
    parser.add_argument(
        "--linear-speed",
        type=float,
        help="override the route maximum translation speed in m/s",
    )
    parser.add_argument("--angular-speed", type=float)
    parser.add_argument("--execute", action="store_true", help="actually send the recorded commands")
    args = parser.parse_args()
    if args.start_action < 0:
        parser.error("--start-action cannot be negative")
    if args.stop_action is not None and args.stop_action <= args.start_action:
        parser.error("--stop-action must be greater than --start-action")
    if args.linear_speed is not None and args.linear_speed < 0:
        parser.error("--linear-speed cannot be negative")
    if args.angular_speed is not None and args.angular_speed < 0:
        parser.error("--angular-speed cannot be negative")
    return args


def print_route(route: Dict[str, Any], start: int, stop: int) -> None:
    print("命名点位：")
    for point in route.get("points", []):
        pose = point["logical_pose"]
        print(
            f"  {point['sequence']}: {point['name']} "
            f"({pose['x_m']:.3f}, {pose['y_m']:.3f}, {pose['theta_deg']:.2f}°)"
        )
    print(f"动作范围：[{start}, {stop})")
    for action in route["actions"][start:stop]:
        requested = action["requested"]
        print(
            f"  {action['sequence']}: direction={requested['direction_deg_relative']:.2f}°, "
            f"distance={requested['distance_m']:.3f} m, "
            f"rotate={requested['rotation_deg_ccw']:.2f}°, status={action['status']}"
        )


def main() -> int:
    args = parse_args()
    try:
        route = read_json(args.route)
        validate_route(route)
    except (OSError, ValueError) as exc:
        print(f"路线文件无效：{exc}")
        return 2

    stop = min(args.stop_action if args.stop_action is not None else len(route["actions"]), len(route["actions"]))
    if args.start_action >= stop:
        print("选择的动作范围为空。")
        return 2
    selected = route["actions"][args.start_action:stop]
    incomplete = [action["sequence"] for action in selected if action.get("status") == "failed"]
    if incomplete:
        print(f"拒绝重放包含失败动作的路线：{incomplete}")
        return 2

    print_route(route, args.start_action, stop)
    if not args.execute:
        print("Dry-run 完成；增加 --execute 才会真实移动。")
        return 0

    defaults = route.get("execution_defaults", {})
    linear_speed = args.linear_speed if args.linear_speed is not None else float(defaults.get("linear_speed_mps", 0.0))
    angular_speed = (
        args.angular_speed if args.angular_speed is not None else float(defaults.get("angular_speed_radps", 0.0))
    )
    replay_name = f"{args.route.stem}.replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    replay_path = args.route.with_name(replay_name)
    replay_log: Dict[str, Any] = {
        "schema_version": 4,
        "kind": "xrocs.move_by_replay",
        "created_at": utc_now(),
        "source_route": str(args.route.resolve()),
        "source_created_at": route.get("created_at"),
        "action_range": {"start": args.start_action, "stop": stop},
        "execution": {
            "host": args.host,
            "port": args.port,
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
        },
        "actions": [],
    }
    write_json_atomic(replay_path, replay_log)

    try:
        from agent.navigation.knewbot_chassis_controller_http import (
            KnewbotChassisController,
        )
    except ModuleNotFoundError as exc:
        print(f"无法加载底盘控制器：{exc}")
        return 2

    controller = KnewbotChassisController({"host": args.host, "rest_port": args.port})
    try:
        if not controller.is_healthy():
            print("底盘健康检查失败，拒绝执行。")
            return 2
        if not controller.is_localized():
            print("底盘未正常定位，无法记录重放过程的 map 位姿。")
            return 2
        for action in selected:
            replay_action: Dict[str, Any] = {
                "source_sequence": action["sequence"],
                "status": "running",
                "started_at": utc_now(),
                "commands": [],
            }
            replay_log["actions"].append(replay_action)
            write_json_atomic(replay_path, replay_log)
            active_command_log: Optional[Dict[str, Any]] = None
            try:
                for command_index, command in enumerate(action["commands"]):
                    command_log: Dict[str, Any] = {
                        "index": command_index,
                        "command": command,
                        "status": "running",
                        "started_at": utc_now(),
                        "map_pose_before": map_pose_from_response(controller.get_pose()),
                    }
                    active_command_log = command_log
                    replay_action["commands"].append(command_log)
                    write_json_atomic(replay_path, replay_log)
                    if command["type"] == "translate":
                        result = controller.move_by(
                            direction=float(command["direction_rad"]),
                            distance=int(command["distance_mm"]),
                            move_mode=int(command["move_mode"]),
                            v=linear_speed,
                            w=angular_speed,
                            wait=True,
                        )
                    else:
                        before = command_log["map_pose_before"]

                        def persist_rotation(feedback: Dict[str, Any]) -> None:
                            command_log["rotation_feedback"] = json_safe(feedback)
                            write_json_atomic(replay_path, replay_log)

                        result = rotate_by_with_feedback(
                            controller,
                            float(command["direction_rad"]),
                            angular_speed=angular_speed,
                            initial_map_pose=before,
                            on_update=persist_rotation,
                        )
                    command_log["status"] = "completed"
                    command_log["finished_at"] = utc_now()
                    command_log["controller_result"] = json_safe(result)
                    command_log["map_pose_after"] = map_pose_from_response(controller.get_pose())
                    write_json_atomic(replay_path, replay_log)
                replay_action["status"] = "completed"
                replay_action["finished_at"] = utc_now()
                write_json_atomic(replay_path, replay_log)
            except Exception as exc:
                if active_command_log is not None and active_command_log.get("status") == "running":
                    active_command_log["status"] = "failed"
                    active_command_log["finished_at"] = utc_now()
                    active_command_log["error"] = repr(exc)
                    diagnostics = collect_failure_diagnostics(
                        controller,
                        active_command_log,
                        exc,
                    )
                    active_command_log["failure_diagnostics"] = diagnostics
                    print_failure_diagnostics(
                        int(action["sequence"]),
                        active_command_log,
                        diagnostics,
                    )
                replay_action["status"] = "failed"
                replay_action["finished_at"] = utc_now()
                replay_action["error"] = repr(exc)
                write_json_atomic(replay_path, replay_log)
                raise
    except (EOFError, KeyboardInterrupt):
        print("重放被中断，正在取消动作。")
        try:
            controller.cancel_action()
        except Exception:
            pass
        return 130
    except Exception as exc:
        print(f"重放失败，执行日志已保存：{exc}")
        try:
            controller.cancel_action()
        except Exception:
            pass
        return 1
    finally:
        controller.close()

    print(f"重放完成，执行日志：{replay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
