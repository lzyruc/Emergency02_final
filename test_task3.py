#!/usr/bin/env python3
"""从 h270_task3_check 单独测试任务3的双目标抓取与返回路线。"""
from __future__ import annotations

from pathlib import Path
import time

import settings
from agent.navigation import NavigationGraph
from arm_motion import moveit_joints
from black_grasp_single import grasp_black_single
from waist_leg_motion import execute_synced_waist_leg_action
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


ROOT = Path(__file__).resolve().parent
ROUTE_PATH = ROOT / "config/scene.navigation.json"


def main() -> None:
    """机器人已位于 h270_task3_check 时，执行 emergency.py 的任务3流程。"""
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
            ROUTE_PATH,
            initial_node="h270_task3_check",
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            input(
                "步骤1：确认机器人实际位于 h270_task3_check 且右侧安全，"
                "按 Enter 执行右侧黑色隔离带识别与抓取；Ctrl+C 终止："
            )
            task3_right = settings.TASK3_RIGHT_PREPARE_POSE
            if not moveit_joints(
                station,
                left=task3_right["left_arm"],
                right=task3_right["right_arm"],
            ):
                raise RuntimeError("双臂到右侧黑色隔离带抓取姿态失败")
            execute_synced_waist_leg_action(
                station,
                {
                    "waist": {"position": task3_right["waist"]},
                    "leg": {"position": task3_right["leg"]},
                },
            )
            time.sleep(3.0)
            print("步骤1夹爪预置命令：left=0.000，right=0.000", flush=True)
            station.execute_action(
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
            )
            time.sleep(2.0)
            right_result = grasp_black_single(
                station,
                "right",
                {"left": 0.0, "right": 0.0},
            )
            if right_result is None or not right_result["hand_executed"]:
                raise RuntimeError("右侧黑色隔离带抓取失败")

            input(
                "步骤2：确认右侧目标已抓稳且移动路径安全，按 Enter 移动到"
                "左右任务点并执行左侧识别与抓取；Ctrl+C 终止："
            )
            navigation.moveTo("h270_task3_right", correct=True)
            task3_left = settings.TASK3_LEFT_PREPARE_POSE
            if not moveit_joints(
                station,
                left=task3_left["left_arm"],
            ):
                raise RuntimeError("左臂到左侧黑色隔离带预备姿态失败")
            execute_synced_waist_leg_action(
                station,
                {
                    "waist": {"position": task3_left["waist"]},
                },
            )
            time.sleep(5.0)
            navigation.moveTo("h270_task3_left", correct=True)
            print("步骤2夹爪预置命令：left=0.000，right=1.000", flush=True)
            station.execute_action(
                {"hand": {"position": {"left": [0.0], "right": [1.0]}}}
            )
            time.sleep(2.0)
            left_result = grasp_black_single(
                station,
                "left",
                {"left": 0.0, "right": 1.0},
            )
            if left_result is None or not left_result["hand_executed"]:
                raise RuntimeError("左侧黑色隔离带抓取失败")

            input(
                "步骤3：确认两个目标均已抓稳且返回路线安全，按 Enter 携带目标"
                "返回 h270_task3_place；Ctrl+C 终止："
            )
            navigation.moveTo("h270_task3_back", correct=True)
            navigation.moveTo("h270_task3_return", correct=True)
            navigation.moveTo("h270_task3_place", correct=True)
            print("已到达放置点，张开双夹爪", flush=True)
            station.execute_action(
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
            )
            time.sleep(2.0)
            recover_pose = settings.TASK3_PLACE_RECOVER_POSE
            print("双臂切换到放置后恢复姿态，然后恢复腰腿", flush=True)
            if not moveit_joints(
                station,
                left=recover_pose["left_arm"],
                right=recover_pose["right_arm"],
            ):
                raise RuntimeError("双臂切换到放置后恢复姿态失败")
            execute_synced_waist_leg_action(
                station,
                {
                    "waist": {"position": recover_pose["waist"]},
                    "leg": {"position": recover_pose["leg"]},
                },
            )
            print("任务3单独测试已完成放置和恢复", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
