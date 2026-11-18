#!/usr/bin/env python3
"""逐步实机测试 back 椅背顶端识别和右手抓取。

本脚本只创建唯一的 xrocs station，不创建 NavigationGraph。运行前先启动
机器人驱动和 MoveIt，然后执行::

    python3 test_back.py

每次相机采集、腰腿、机械臂和夹爪控制前均等待 Enter；Ctrl+C 终止。
识别输入为右腕 RGB-D，输出目录为 ``output/chair_backtop_grasp/<时间戳>``。
运动坐标采用 ``WAIST_YAW_LINK <- right_tcp_link`` 4×4 齐次矩阵，长度单位米，
姿态四元数顺序为 XYZW。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

import settings
from agent.perception.deprojector import Deprojector
from arm_motion import moveit_cartesians, moveit_joints
from chair_backtop_grasp import (
    plan_chair_backtop_grasp_steps,
    recognize_chair_backtop_grasp,
)
from task1 import CHAIR_BACK_POSE_PATHS
from waist_leg_motion import execute_synced_waist_leg_action
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TASK_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = (
    TASK_DIR / "agent/perception/config/camera_calibration.json"
)
OUTPUT_ROOT = TASK_DIR / "output/chair_backtop_grasp"
POSE3_PATH = CHAIR_BACK_POSE_PATHS[-1]
BODY_SETTLE_S = 3.0
CAMERA_SETTLE_S = 3.0
GRASP_SETTLE_S = 2.0
CHAIR_BACK_GRASP_APPROACH_M = 0.40
CHAIR_BACK_GRASP_STANDOFF_M = 0.05
CHAIR_BACK_GRASP_Z_COMPENSATION_M = -0.105
CHAIR_BACK_GRASP_SECOND_STAGE_Z_M = 0.08


def main() -> None:
    """按人工确认点完成姿态3、识别、抓放和姿态3恢复。"""
    pose3 = json.loads(POSE3_PATH.read_text(encoding="utf-8"))
    output_dir = (
        OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get(
            "control_rate_hz",
            100,
        ),
    )
    try:
        time.sleep(1.0)
        right_deprojector = Deprojector(
            station,
            CALIBRATION_PATH,
            "right",
        )
        robot = station.get_robot_handle().get("robot")

        input(
            "步骤1/9：确认夹爪区域安全，按 Enter 张开左右夹爪；"
            "Ctrl+C 终止："
        )
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [
                            settings.TASK["hands"]["left_open_position"]
                        ],
                        "right": [
                            settings.TASK["hands"]["right_open_position"]
                        ],
                    }
                }
            }
        )

        input(
            "步骤2/9：确认腰腿路径安全，按 Enter 切换到 back 姿态3"
            "的腰腿位置；Ctrl+C 终止："
        )
        execute_synced_waist_leg_action(
            station,
            {
                "waist": {"position": pose3["waist"]},
                "leg": {"position": pose3["leg"]},
            },
        )
        time.sleep(BODY_SETTLE_S)

        input(
            "步骤3/9：确认双臂路径安全，按 Enter 切换到 back 姿态3"
            "的双臂位置；Ctrl+C 终止："
        )
        if not moveit_joints(
            station,
            left=pose3["left_arm"],
            right=pose3["right_arm"],
        ):
            raise RuntimeError("切换到 back 姿态3双臂位置失败")

        input(
            "步骤4/9：确认机器人和椅子均静止，按 Enter 使用右腕RGB-D"
            "识别椅背顶端；Ctrl+C 终止："
        )
        time.sleep(CAMERA_SETTLE_S)
        grasp = recognize_chair_backtop_grasp(
            station,
            right_deprojector,
            output_dir,
        )
        target_pose_waist = np.asarray(
            grasp["target_pose_waist_xyz_xyzw"],
            dtype=np.float64,
        )
        state = station.get_robot_state()
        current_pose_waist = np.asarray(
            state["arm_pose"]["right"].get_xyz_m_xyzw_ndarray()
            if hasattr(
                state["arm_pose"]["right"],
                "get_xyz_m_xyzw_ndarray",
            )
            else state["arm_pose"]["right"],
            dtype=np.float64,
        ).copy()
        T_waist_right_tcp = robot.pin_kine.get_right_tcp_pose(
            state["arm_joints"]["right"]
        ).get_affine_matrix_m_ndarray()
        target_right_tcp_m = (
            np.linalg.inv(T_waist_right_tcp)
            @ np.append(target_pose_waist[:3], 1.0)
        )[:3]
        first_delta_tcp_m, second_delta_tcp_m = plan_chair_backtop_grasp_steps(
            target_right_tcp_m,
            approach_m=CHAIR_BACK_GRASP_APPROACH_M,
            standoff_m=CHAIR_BACK_GRASP_STANDOFF_M,
            z_compensation_m=CHAIR_BACK_GRASP_Z_COMPENSATION_M,
            second_stage_z_m=CHAIR_BACK_GRASP_SECOND_STAGE_Z_M,
        )
        print(
            f"识别完成：像素={grasp['pixel_uv']}，"
            f"深度={grasp['depth_m']:.3f} m，方向={grasp['axis']}；"
            f"TCP全量={np.round(target_right_tcp_m, 6).tolist()} m，"
            f"第一段XYZ={np.round(first_delta_tcp_m, 6).tolist()} m，"
            f"第二段Z={np.round(second_delta_tcp_m, 6).tolist()} m；"
            f"诊断目录={output_dir}",
            flush=True,
        )

        input(
            "步骤5/9：确认右臂路径安全，按 Enter 执行TCP XY和第一段Z；"
            "Ctrl+C 终止："
        )
        if np.linalg.norm(first_delta_tcp_m) > 1e-3:
            first_target_pose_waist = current_pose_waist.copy()
            first_target_pose_waist[:3] = (
                T_waist_right_tcp
                @ np.asarray(
                    [
                        first_delta_tcp_m[0],
                        first_delta_tcp_m[1],
                        first_delta_tcp_m[2],
                        1.0,
                    ],
                    dtype=np.float64,
                )
            )[:3]
            if not moveit_cartesians(
                station,
                right=[first_target_pose_waist.tolist()],
                use_cartesian_path=True,
                right_motion_id=2,
            ):
                raise RuntimeError("back 椅子右臂TCP XYZ预接近失败")
        else:
            print("第一段位移不超过1 mm，跳过第一段运动", flush=True)

        input(
            "步骤6/9：确认右臂接近路径安全，按 Enter 执行TCP Z接近；"
            "Ctrl+C 终止："
        )
        if np.linalg.norm(second_delta_tcp_m) > 1e-6:
            state = station.get_robot_state()
            current_pose_waist = np.asarray(
                state["arm_pose"]["right"].get_xyz_m_xyzw_ndarray()
                if hasattr(
                    state["arm_pose"]["right"],
                    "get_xyz_m_xyzw_ndarray",
                )
                else state["arm_pose"]["right"],
                dtype=np.float64,
            ).copy()
            T_waist_right_tcp = robot.pin_kine.get_right_tcp_pose(
                state["arm_joints"]["right"]
            ).get_affine_matrix_m_ndarray()
            z_target_pose_waist = current_pose_waist.copy()
            z_target_pose_waist[:3] = (
                T_waist_right_tcp
                @ np.asarray(
                    [0.0, 0.0, second_delta_tcp_m[2], 1.0],
                    dtype=np.float64,
                )
            )[:3]
            if not moveit_cartesians(
                station,
                right=[z_target_pose_waist.tolist()],
                use_cartesian_path=True,
                right_motion_id=3,
            ):
                raise RuntimeError("back 椅子右臂TCP Z接近失败")
        else:
            print("右臂已在椅背顶端余量内，跳过TCP Z运动", flush=True)

        input(
            "步骤7/9：确认右夹爪位于椅背抓取位置，按 Enter 闭合右夹爪；"
            "Ctrl+C 终止："
        )
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [
                            settings.TASK["hands"]["left_open_position"]
                        ],
                        "right": [
                            settings.TASK["hands"]["right_close_position"]
                        ],
                    }
                }
            }
        )
        time.sleep(GRASP_SETTLE_S)

        input(
            "步骤8/9：确认椅子已稳定，按 Enter 松开右夹爪；"
            "Ctrl+C 终止："
        )
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [
                            settings.TASK["hands"]["left_open_position"]
                        ],
                        "right": [
                            settings.TASK["hands"]["right_open_position"]
                        ],
                    }
                }
            }
        )

        input(
            "步骤9/9：确认右臂从椅背撤回姿态3的路径安全，按 Enter "
            "将双臂切回 back 姿态3；Ctrl+C 终止："
        )
        if not moveit_joints(
            station,
            left=pose3["left_arm"],
            right=pose3["right_arm"],
        ):
            raise RuntimeError("双臂切回 back 姿态3失败")
        print("back 椅背抓放测试完成，机器人保持姿态3", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
