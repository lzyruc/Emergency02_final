#!/usr/bin/env python3
"""逐步实机测试办公转椅双腕深度突变抓取策略。

导航和 YOLO 步骤暂为空，不创建对应资源。每次真实运动和相机采集前均需按 Enter。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import settings
from agent.perception.deprojector import Deprojector
from arm_motion import moveit_cartesians, moveit_joints
from chair_depth_grasp import recognize_chair_grasp
from waist_leg_motion import execute_synced_waist_leg_action
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TASK_DIR = Path(__file__).resolve().parent
INITIAL_POSE_PATH = TASK_DIR / "poses_chair/应急_转椅_正面.json"
CALIBRATION_PATH = TASK_DIR / "agent/perception/config/camera_calibration.json"
OUTPUT_ROOT = TASK_DIR / "output/chair_depth_grasp"
CHAIR_GRASP_OFFSET_TCP_M = np.asarray([0.0, 0.0, 0.10], dtype=np.float64)


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """将 XYZW 四元数转换为 TCP 到 waist_yaw_link 的 3×3 旋转矩阵。"""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_startpoint_pose(
    station: Any,
    pose: dict[str, Any],
) -> None:
    """按 emergency_zyt.py 的顺序恢复腰腿和双臂初始姿态。"""
    print("startpoint姿态恢复：移动双臂，再同步移动头、腰和腿", flush=True)
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": [0.0, 0.001]},
            "leg": {"position": [-0.5, 0.5]},
        },
    )
    time.sleep(5.0)
    input("[start]腰腿回正后按enter继续")

    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        raise RuntimeError("双臂恢复应急初始姿态失败")
    
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        },
    )


def main() -> None:
    """按人工确认点依次执行转椅双腕抓取测试。"""
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_ROOT / run_id
    sequence = 1
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"{run_id}_{sequence}"
        sequence += 1
    output_dir.mkdir(parents=True)
    print(f"本次运行识别图片保存到 {output_dir}", flush=True)
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        time.sleep(1.0)
        deprojectors = {
            "left": Deprojector(station, CALIBRATION_PATH, "left"),
            "right": Deprojector(station, CALIBRATION_PATH, "right"),
        }

        input(
            "步骤1：确认双臂和腰腿路径安全，按 Enter 切换到应急初始姿态；"
            "Ctrl+C 终止："
        )
        apply_startpoint_pose(station, initial_pose)

        input(
            "步骤2：按 Enter 执行转椅识别和导航占位步骤；当前不初始化 "
            "Navigation 或 YOLO，不发送运动命令；Ctrl+C 终止："
        )
        print("转椅识别和导航暂未接入，继续使用当前机器人位置", flush=True)

        available_arms = []
        for arm, step, delta_y_m in (
            ("left", "步骤3", 0.05),
            ("right", "步骤4", -0.05),
        ):
            current_pose = station.get_robot_state()["arm_pose"][arm]
            target_pose = np.asarray(
                current_pose.get_xyz_m_xyzw_ndarray()
                if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
                else current_pose,
                dtype=np.float64,
            ).copy()
            target_pose[0] += 0.05
            target_pose[1] += delta_y_m
            input(
                f"{step}：确认{arm}臂路径安全，按 Enter 在 waist_yaw_link 下"
                f"一次移动 x=+5 cm、y={delta_y_m * 100:+.0f} cm，"
                "TCP方向保持不变；Ctrl+C 终止："
            )
            succeeded = (
                moveit_cartesians(
                    station,
                    left=[target_pose.tolist()],
                    use_cartesian_path=True,
                    left_motion_id=1,
                )
                if arm == "left"
                else moveit_cartesians(
                    station,
                    right=[target_pose.tolist()],
                    use_cartesian_path=True,
                    right_motion_id=1,
                )
            )
            if not succeeded:
                print(f"{arm}臂预移动失败，跳过该侧识别和抓取", flush=True)
                continue
            available_arms.append(arm)

        hand_positions = {
            "left": settings.TASK["hands"]["left_open_position"],
            "right": settings.TASK["hands"]["right_open_position"],
        }
        succeeded_arms = []
        for arm, recognize_step, xy_step, z_step, close_step in (
            ("left", "步骤5", "步骤6", "步骤7", "步骤8"),
            ("right", "步骤9", "步骤10", "步骤11", "步骤12"),
        ):
            if arm not in available_arms:
                continue
            input(
                f"{recognize_step}：确认机器人已静止，按 Enter 读取{arm}腕 RGB-D，"
                "屏蔽夹爪并识别 ROI 内两侧均有深度突变的较小抓取点；"
                "Ctrl+C 终止："
            )
            try:
                grasp = recognize_chair_grasp(
                    station,
                    arm,
                    deprojectors[arm],
                    output_dir,
                )
            except RuntimeError as exc:
                print(
                    f"{arm}腕抓取点识别失败，跳过该侧：{exc}；"
                    f"诊断图已保存到 {output_dir}",
                    flush=True,
                )
                continue
            print(
                f"{arm}腕抓取点：像素={grasp['pixel_uv']}，"
                f"深度={grasp['depth_m']:.3f} m，"
                f"方向={grasp['axis']}，跨度={grasp['span_px']} px/"
                f"{grasp['width_m'] * 100:.1f} cm，"
                f"扶手长轴={grasp['vertical_span_px']} px/"
                f"{grasp['vertical_length_m'] * 100:.1f} cm，"
                f"两侧突变={grasp['side_jumps_mm']} mm，"
                f"连通域面积={grasp['component_area_px']} px，"
                f"长宽比={grasp['elongation_ratio']:.2f}，"
                f"局部深度极差={grasp['patch_depth_range_mm']:.1f} mm，"
                f"夹爪mask像素={grasp['gripper_mask_pixels']}，"
                f"Pose={grasp['target_pose_waist_xyz_xyzw'].tolist()}",
                flush=True,
            )
            target_pose = np.asarray(
                grasp["target_pose_waist_xyz_xyzw"],
                dtype=np.float64,
            ).copy()
            offset_waist_m = rotation_xyzw(target_pose[3:]) @ CHAIR_GRASP_OFFSET_TCP_M
            target_pose[:3] += offset_waist_m
            current_pose = station.get_robot_state()["arm_pose"][arm]
            current_pose_xyz_xyzw = np.asarray(
                current_pose.get_xyz_m_xyzw_ndarray()
                if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
                else current_pose,
                dtype=np.float64,
            ).copy()
            delta_waist_m = target_pose[:3] - current_pose_xyz_xyzw[:3]
            tcp_rotation = rotation_xyzw(current_pose_xyz_xyzw[3:])
            delta_tcp_m = tcp_rotation.T @ delta_waist_m
            xy_delta_tcp_m = np.asarray(
                [delta_tcp_m[0], delta_tcp_m[1], 0.0],
                dtype=np.float64,
            )
            xy_delta_waist_m = tcp_rotation @ xy_delta_tcp_m
            xy_target_pose = current_pose_xyz_xyzw.copy()
            xy_target_pose[:3] += xy_delta_waist_m
            print(
                f"{arm}臂两段抓取："
                f"当前XYZ={np.round(current_pose_xyz_xyzw[:3], 6).tolist()} m，"
                f"识别Pose基础上TCP偏移="
                f"{np.round(CHAIR_GRASP_OFFSET_TCP_M, 6).tolist()} m，"
                f"换算waist偏移={np.round(offset_waist_m, 6).tolist()} m，"
                f"目标XYZ={np.round(target_pose[:3], 6).tolist()} m，"
                f"总delta(TCP)={np.round(delta_tcp_m, 6).tolist()} m",
                flush=True,
            )
            print(
                f"{arm}臂第一段TCP XY："
                f"delta(TCP)={np.round(xy_delta_tcp_m, 6).tolist()} m，"
                f"delta(waist)={np.round(xy_delta_waist_m, 6).tolist()} m，"
                f"目标XYZ={np.round(xy_target_pose[:3], 6).tolist()} m",
                flush=True,
            )
            input(
                f"{xy_step}：确认{arm}臂路径安全，按 Enter 只执行TCP局部"
                "X/Y方向对齐，TCP方向保持不变；"
                "Ctrl+C 终止："
            )
            succeeded = (
                moveit_cartesians(
                    station,
                    left=[xy_target_pose.tolist()],
                    use_cartesian_path=True,
                    left_motion_id=2,
                )
                if arm == "left"
                else moveit_cartesians(
                    station,
                    right=[xy_target_pose.tolist()],
                    use_cartesian_path=True,
                    right_motion_id=2,
                )
            )
            if not succeeded:
                print(f"{arm}臂TCP X/Y对齐失败，跳过该侧抓取", flush=True)
                continue
            z_delta_tcp_m = np.asarray(
                [0.0, 0.0, delta_tcp_m[2]],
                dtype=np.float64,
            )
            z_delta_waist_m = tcp_rotation @ z_delta_tcp_m
            print(
                f"{arm}臂第二段TCP Z："
                f"delta(TCP)={np.round(z_delta_tcp_m, 6).tolist()} m，"
                f"delta(waist)={np.round(z_delta_waist_m, 6).tolist()} m，"
                f"最终目标XYZ={np.round(target_pose[:3], 6).tolist()} m",
                flush=True,
            )
            input(
                f"{z_step}：确认{arm}臂路径安全，按 Enter 只执行TCP局部"
                "Z方向移动到最终目标，TCP方向保持不变；Ctrl+C 终止："
            )
            succeeded = (
                moveit_cartesians(
                    station,
                    left=[target_pose.tolist()],
                    use_cartesian_path=True,
                    left_motion_id=3,
                )
                if arm == "left"
                else moveit_cartesians(
                    station,
                    right=[target_pose.tolist()],
                    use_cartesian_path=True,
                    right_motion_id=3,
                )
            )
            if not succeeded:
                print(f"{arm}臂TCP Z移动失败，跳过该侧抓取", flush=True)
                continue
            input(
                f"{close_step}：确认{arm}手已到达目标抓取点，按 Enter 闭合"
                f"{arm}夹爪；Ctrl+C 终止："
            )
            hand_positions[arm] = settings.TASK["hands"][f"{arm}_close_position"]
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
            time.sleep(1.0)
            succeeded_arms.append(arm)
        if not succeeded_arms:
            print(
                "左右臂均未完成抓取（可能是识别或 IK/MoveIt 失败），未闭合夹爪",
                flush=True,
            )

        input("步骤13：按 Enter 松开双夹爪；Ctrl+C 保持夹持：")
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [settings.TASK["hands"]["left_open_position"]],
                        "right": [settings.TASK["hands"]["right_open_position"]],
                    }
                }
            }
        )
        time.sleep(1.0)

        input(
            "步骤14：确认返回路径安全，按 Enter 恢复应急初始姿态；"
            "Ctrl+C 保持当前位置："
        )
        apply_startpoint_pose(station, initial_pose)
        print(f"办公转椅双腕抓取测试完成，识别结果保存在 {output_dir}", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
