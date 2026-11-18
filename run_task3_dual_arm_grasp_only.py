#!/usr/bin/env python3
"""任务三双臂抓取与拉放测试：从双臂抓取点开始，依次抓取并运到放置点。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import settings
from agent.navigation import NavigationGraph
from arm_motion import moveit_joints
from run_task3_only import grasp_black_single_for_test
from waist_leg_motion import execute_synced_waist_leg_action
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


DEFAULT_PREPARE_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_emergency/应急_隔离柱抓取.json"
)
BODY_TO_ARM_DELAY_S = 0.5
CAMERA_SETTLE_S = 3.0
TASK3_PICK_NODE = "h225_task3_pick"
TASK3_RETURN_NODE = "h180_task3_dual_return"
TASK3_PLACE_NODE = "h180_task3_dual_place"
GRASP_RETRY_OFFSETS_Y_M = (0.0, 0.10, -0.10)


def apply_dual_grasp_prepare_pose(
    station: Any,
    pose: dict[str, Any],
) -> bool:
    """先下发头腰腿，0.5秒后阻塞执行双臂预备姿态。"""
    body_action = {
        key: {"position": pose[key]}
        for key in ("head", "waist", "leg")
        if key in pose
    }
    if body_action:
        execute_synced_waist_leg_action(station, body_action)
        time.sleep(BODY_TO_ARM_DELAY_S)
    return moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    )


def grasp_arm_with_lateral_retries(
    station: Any,
    navigation: NavigationGraph,
    arm: str,
    hand_positions: dict[str, float],
    output_dir: Path,
) -> dict[str, Any] | None:
    """原位、左移10 cm、右移10 cm依次抓取，每次偏移后返回pick。"""
    labels = ("origin", "left_10cm", "right_10cm")
    for attempt_index, (offset_y_m, label) in enumerate(
        zip(GRASP_RETRY_OFFSETS_Y_M, labels),
        start=1,
    ):
        moved = offset_y_m != 0.0
        if moved:
            input(
                f"{arm}臂第{attempt_index}/3次抓取前，确认底盘"
                f"{'左移' if offset_y_m > 0.0 else '右移'}10 cm安全，"
                "按 Enter 移动："
            )
            navigation.moveBy(0.0, offset_y_m, 0.0, correct=False)
            time.sleep(CAMERA_SETTLE_S)

        input(
            f"确认{arm}腕视野和机械臂路径安全，按 Enter 执行"
            f"第{attempt_index}/3次识别抓取："
        )
        try:
            result = grasp_black_single_for_test(
                station,
                arm,
                hand_positions,
                output_dir / f"attempt_{attempt_index}_{label}",
                split_tcp_z_motion=False,
                component_selection="leftmost" if arm == "left" else None,
            )
        finally:
            if moved:
                input(
                    f"{arm}臂第{attempt_index}/3次偏移抓取已结束；"
                    "确认回到pick的路径安全，按 Enter 恢复："
                )
                navigation.recoverState(TASK3_PICK_NODE, method="move_by")

        if result is not None and result["motion_succeeded"]:
            return result

        print(
            f"{arm}臂第{attempt_index}/3次识别或IK失败，张开当前夹爪",
            flush=True,
        )
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [hand_positions["left"]],
                        "right": [hand_positions["right"]],
                    }
                }
            }
        )
        time.sleep(2.0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从h225_task3_pick开始，依次测试双臂隔离柱抓取、返回和放置。"
        ),
    )
    parser.add_argument(
        "pose_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PREPARE_POSE_PATH,
        help=f"预备姿态JSON；默认：{DEFAULT_PREPARE_POSE_PATH}",
    )
    args = parser.parse_args()
    pose_path = args.pose_path.expanduser().resolve()
    prepare_pose = json.loads(pose_path.read_text(encoding="utf-8"))

    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    navigation_config = settings.TASK["navigation"]
    try:
        time.sleep(1.0)
        with NavigationGraph(
            settings.prepare_route(settings.TEST_NAVIGATION_ROUTE_STEM),
            initial_node=TASK3_PICK_NODE,
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            if not navigation.checkState():
                raise RuntimeError("机器人实测位置不在h225_task3_pick")
            input(
                f"已确认位于{TASK3_PICK_NODE}，将使用姿态文件 {pose_path}；"
                "确认头、腰、腿和双臂路径安全，按 Enter 松开双爪并进入预备姿态："
            )
            station.execute_action(
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
            )
            time.sleep(2.0)
            if not apply_dual_grasp_prepare_pose(station, prepare_pose):
                raise RuntimeError("双臂切换任务三抓取预备姿态失败")
            time.sleep(CAMERA_SETTLE_S)

            session_dir = (
                settings.RUNTIME_DIR
                / "task3_dual_arm_grasp_only"
                / f"{datetime.now():%Y%m%d_%H%M%S_%f}"
            )
            right_result = grasp_arm_with_lateral_retries(
                station,
                navigation,
                "right",
                {"left": 0.0, "right": 0.0},
                session_dir / "right",
            )
            right_closed = (
                float(settings.TASK["hands"]["right_close_position"])
                if right_result is not None and right_result["hand_executed"]
                else 0.0
            )
            left_result = grasp_arm_with_lateral_retries(
                station,
                navigation,
                "left",
                {"left": 0.0, "right": right_closed},
                session_dir / "left",
            )

            right_success = (
                right_result is not None and right_result["motion_succeeded"]
            )
            left_success = (
                left_result is not None and left_result["motion_succeeded"]
            )
            print(
                "双臂隔离柱抓取结果："
                f"right={'成功' if right_success else '失败'}，"
                f"left={'成功' if left_success else '失败'}，"
                f"诊断目录={session_dir}",
                flush=True,
            )
            if not (right_success and left_success):
                input(
                    "至少一侧三次抓取均失败；确认夹爪周围安全，"
                    "按 Enter 在pick松开双爪并结束："
                )
                station.execute_action(
                    {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
                )
                time.sleep(2.0)
                return

            input(
                "双侧均已抓稳；确认后退和顺时针旋转路径安全，"
                "按 Enter 执行moveBy(-0.30, 0, -45°)："
            )
            navigation.moveBy(-0.30, 0.0, -45.0, correct=False)
            input(
                f"确认横向拖行路径安全，按 Enter 前往{TASK3_RETURN_NODE}："
            )
            navigation.moveTo(TASK3_RETURN_NODE, correct=True)
            input(f"确认放置路径安全，按 Enter 前往{TASK3_PLACE_NODE}：")
            navigation.moveTo(TASK3_PLACE_NODE, correct=False)

            input(
                "已到达放置点；确认双臂携带目标时关节路径安全，按 Enter 执行"
                "左臂第2关节固定到+30°、右臂第2关节固定到-30°："
            )
            arm_joints = station.get_robot_state()["arm_joints"]
            left_joints = [float(value) for value in arm_joints["left"]]
            right_joints = [float(value) for value in arm_joints["right"]]
            left_joints[1] = math.radians(30.0)
            right_joints[1] = math.radians(-30.0)
            if not moveit_joints(
                station,
                left=left_joints,
                right=right_joints,
            ):
                raise RuntimeError("双臂第2关节展开运动失败")

            time.sleep(1.0)
            station.execute_action(
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
            )
            time.sleep(2.0)
            input("双爪已松开；确认后方安全，按 Enter 后退0.50 m：")
            navigation.moveBy(-0.50, 0.0, 0.0, correct=False)
            navigation.recoverState(TASK3_RETURN_NODE, method="move_by")
            input(
                f"已回到{TASK3_RETURN_NODE}；确认头、腰、腿和双臂路径安全，"
                "按 Enter 恢复单测开始时的抓取预备姿态："
            )
            if not apply_dual_grasp_prepare_pose(station, prepare_pose):
                raise RuntimeError("恢复任务三双臂抓取预备姿态失败")
            print(
                f"任务三双臂抓取与拉放测试结束，已回到{TASK3_RETURN_NODE}"
                "并恢复抓取预备姿态",
                flush=True,
            )
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
