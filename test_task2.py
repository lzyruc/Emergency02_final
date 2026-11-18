#!/usr/bin/env python3
"""在窗户2动作点单步测试左腕识别、机械臂操作和分段底盘开窗。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np

import settings
from agent.navigation import NavigationGraph
from agent.perception import deprojector
from arm_motion import moveit_cartesians, moveit_joints
from window_handle_recognition import recognize_window_handle
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


CALIBRATION_PATH = (
    Path(deprojector.__file__).resolve().parent
    / "config/camera_calibration.json"
)
WINDOW2_HANDLE_STANDOFF_TCP_Z_M = 0.19
WINDOW2_UPWARD_WAIST_Z_M = 0.03


def _apply_window2_prepare(station: Any) -> None:
    """按 xrocs 标准接口依次恢复双臂，以及头、腰、腿准备姿态。"""
    pose = settings.WINDOW2_PREPARE_POSE

    station.execute_action(
        {
            "head": {"position": pose["head"]},
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        }
    )
    time.sleep(1.0)  # 等待腰腿到位，避免 MoveIt 规划失败
    input("确认腰腿回到准备姿态后按 Enter 执行双臂到应急_窗户2_准备姿态；Ctrl+C 终止：")

    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        raise RuntimeError("双臂恢复应急_窗户2_准备姿态失败")



def _print_offset(label: str, delta_tcp_m: np.ndarray, delta_waist_m: np.ndarray) -> None:
    """打印 TCP 局部位移及其在 waist_yaw_link 下的实际位移。"""
    print(
        f"{label} TCP局部偏移："
        f"x={delta_tcp_m[0] * 100:+.2f} cm，"
        f"y={delta_tcp_m[1] * 100:+.2f} cm，"
        f"z={delta_tcp_m[2] * 100:+.2f} cm",
        flush=True,
    )
    print(
        f"{label} waist_yaw_link偏移："
        f"x={delta_waist_m[0] * 100:+.2f} cm，"
        f"y={delta_waist_m[1] * 100:+.2f} cm，"
        f"z={delta_waist_m[2] * 100:+.2f} cm",
        flush=True,
    )


def main() -> None:
    """从h270_window2_action测试窗口2识别、操作与分段底盘移动。"""
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    output_dir = (
        settings.ROOT
        / "output"
        / "task2_recognize"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    navigation: NavigationGraph | None = None
    try:
        navigation_config = settings.TASK["navigation"]
        navigation = NavigationGraph(
            settings.prepare_route(),
            initial_node="h270_window2_action",
            navigation_mode="move_by",
            linear_speed=0.2,
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        )
        if not navigation.checkState():
            raise RuntimeError(
                "机器人实测位置不在h270_window2_action，停止窗口2测试"
            )
        print(
            "已确认机器人实测位置位于h270_window2_action。",
            flush=True,
        )
        input(
            "步骤1：确认双臂、头、腰和腿运动路径安全，按 Enter 切换到"
            "应急_窗户2_准备姿态；Ctrl+C 终止："
        )
        _apply_window2_prepare(station)

        input("步骤2：准备姿态已到达，按 Enter 读取左腕 RGB-D 并识别把手：")
        result = recognize_window_handle(station, output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

        target_pose_waist = deprojector.Deprojector(
            station,
            CALIBRATION_PATH,
            "left",
        ).deproject(
            [*result["pixel_uv"], result["depth_m"]],
            face_target=False,
        )
        robot = station.get_robot_handle().get("robot")
        state = station.get_robot_state()
        T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
            state["arm_joints"]["left"]
        ).get_affine_matrix_m_ndarray()
        point_left_tcp_m = (
            np.linalg.inv(T_waist_tcp)
            @ np.append(target_pose_waist[:3], 1.0)
        )[:3]

        first_delta_tcp_m = np.array(
            [point_left_tcp_m[0], point_left_tcp_m[1], 0.0],
            dtype=np.float64,
        )
        first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
        first_target_pose_waist = target_pose_waist.copy()
        first_target_pose_waist[:3] = (
            T_waist_tcp[:3, 3] + first_delta_waist_m
        )
        print(
            "识别点（left_tcp_link）："
            f"x={point_left_tcp_m[0] * 100:+.2f} cm，"
            f"y={point_left_tcp_m[1] * 100:+.2f} cm，"
            f"z={point_left_tcp_m[2] * 100:+.2f} cm",
            flush=True,
        )
        _print_offset("第一次", first_delta_tcp_m, first_delta_waist_m)
        input(
            "步骤3：第一次只修正 TCP 局部 x、y，方向保持不变；"
            "确认路径安全后按 Enter 执行："
        )
        if not moveit_cartesians(
            station,
            left=first_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=1,
        ):
            raise RuntimeError("左臂第一次 x/y 修正失败")

        state = station.get_robot_state()
        T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
            state["arm_joints"]["left"]
        ).get_affine_matrix_m_ndarray()
        current_pose = state["arm_pose"]["left"]
        current_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        )
        second_delta_tcp_m = np.array(
            [
                0.0,
                0.0,
                point_left_tcp_m[2] + WINDOW2_HANDLE_STANDOFF_TCP_Z_M,
            ],
            dtype=np.float64,
        )
        second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
        second_delta_waist_m[2] = 0.0
        second_target_pose_waist = current_pose_waist.copy()
        second_target_pose_waist[:3] = (
            T_waist_tcp[:3, 3] + second_delta_waist_m
        )
        print(
            "第二次 z 计算："
            f"识别TCP-z {point_left_tcp_m[2] * 100:+.2f} cm + "
            f"保留距离{WINDOW2_HANDLE_STANDOFF_TCP_Z_M * 100:.2f} cm = "
            f"TCP-z计算量{second_delta_tcp_m[2] * 100:+.2f} cm；"
            "转换后固定waist_yaw_link z=+0.00 cm",
            flush=True,
        )
        _print_offset("第二次", second_delta_tcp_m, second_delta_waist_m)
        result.update(
            {
                "point_left_tcp_xyz_m": point_left_tcp_m.tolist(),
                "first_delta_left_tcp_xyz_m": first_delta_tcp_m.tolist(),
                "second_delta_left_tcp_xyz_m": second_delta_tcp_m.tolist(),
                "second_delta_waist_xyz_m": second_delta_waist_m.tolist(),
                "second_z_rule": (
                    "recognized_left_tcp_z_m + 0.19 m, then keep "
                    "waist_yaw_link z unchanged"
                ),
            }
        )
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        input(
            "步骤4：按识别TCP-z计算水平接近量，waist_yaw_link Z保持不变；"
            "确认路径安全后按 Enter 执行："
        )
        if not moveit_cartesians(
            station,
            left=second_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=2,
        ):
            raise RuntimeError("左臂第二次 z 修正失败")
        input(
            "步骤5：左臂 z 轴运动已由 MoveIt 确认成功，"
            "本步骤不闭合夹爪；确认状态后按 Enter 继续："
        )

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        right_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        right_target_pose_waist[1] -= 0.11
        input(
            "步骤6：TCP方向保持不变，将沿 waist_yaw_link -Y（视觉右侧）"
            "移动11.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤6位移：waist_yaw_link x=+0.0 cm，y=-11.0 cm，z=+0.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=right_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=3,
        ):
            raise RuntimeError("左臂向视觉右侧移动11 cm失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        upward_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        upward_target_pose_waist[2] += WINDOW2_UPWARD_WAIST_Z_M
        input(
            "步骤7：TCP方向保持不变，将沿 waist_yaw_link +Z（向上）"
            f"移动{WINDOW2_UPWARD_WAIST_Z_M * 100:.1f} cm；"
            "确认路径安全后按 Enter 执行："
        )
        print(
            "步骤7位移：waist_yaw_link x=+0.0 cm，y=+0.0 cm，z=+3.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=upward_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=4,
        ):
            raise RuntimeError("左臂向上移动3.0 cm失败")

        for step, motion_id, label, delta_waist_m in (
            (8, 5, "沿waist_yaw_link -Y向视觉右侧移动3 cm", [0.0, -0.03, 0.0]),
            #(9, 6, "沿waist_yaw_link +X向前移动1 cm", [0.01, 0.0, 0.0]),
        ):
            state = station.get_robot_state()
            current_pose = state["arm_pose"]["left"]
            target_pose_waist = np.asarray(
                current_pose.get_xyz_m_xyzw_ndarray()
                if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
                else current_pose,
                dtype=np.float64,
            ).copy()
            target_pose_waist[:3] += np.asarray(delta_waist_m, dtype=np.float64)
            input(
                f"步骤{step}：确认路径安全，按 Enter {label}；Ctrl+C终止："
            )
            if not moveit_cartesians(
                station,
                left=target_pose_waist,
                use_cartesian_path=True,
                left_motion_id=motion_id,
            ):
                raise RuntimeError(f"窗口2左臂{label}失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        rotation_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        qx, qy, qz, qw = rotation_target_pose_waist[3:7]
        cos_h = math.cos(math.pi / 4.0)
        sin_h = math.sin(math.pi / 4.0)
        qw_new = qw * cos_h - qz * sin_h
        qx_new = qx * cos_h + qy * sin_h
        qy_new = qy * cos_h - qx * sin_h
        qz_new = qz * cos_h + qw * sin_h
        norm = math.sqrt(qw_new**2 + qx_new**2 + qy_new**2 + qz_new**2)
        rotation_target_pose_waist[3:7] = (
            qx_new / norm,
            qy_new / norm,
            qz_new / norm,
            qw_new / norm,
        )
        input(
            "步骤10：确认路径安全，按 Enter 保持TCP位置不变，"
            "绕当前夹爪局部z轴顺时针旋转90°；Ctrl+C终止："
        )
        if not moveit_cartesians(
            station,
            left=rotation_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=7,
        ):
            raise RuntimeError("窗口2左臂TCP绕局部z轴顺时针旋转90°失败")

        for step, motion_id, label, delta_waist_m in (
            (9, 6, "沿waist_yaw_link +Z向视觉上侧移动2 cm", [0.0, 0.0, 0.02]),
            (10, 7, "沿waist_yaw_link +Y向视觉左侧移动5 cm", [0.0, 0.05, 0.0]),
            #(12, 9, "沿waist_yaw_link +Z向上移动5 cm", [0.0, 0.0, 0.05]),
        ):
            state = station.get_robot_state()
            current_pose = state["arm_pose"]["left"]
            target_pose_waist = np.asarray(
                current_pose.get_xyz_m_xyzw_ndarray()
                if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
                else current_pose,
                dtype=np.float64,
            ).copy()
            target_pose_waist[:3] += np.asarray(delta_waist_m, dtype=np.float64)
            input(
                f"步骤{step}：确认路径安全，按 Enter {label}；Ctrl+C终止："
            )
            if not moveit_cartesians(
                station,
                left=target_pose_waist,
                use_cartesian_path=True,
                left_motion_id=motion_id,
            ):
                raise RuntimeError(f"窗口2左臂{label}失败")

        input(
            "步骤13：确认把手位于夹爪内，按 Enter 闭合左夹爪；"
            "Ctrl+C终止："
        )
        print("闭合左夹爪：left=1.000，right=0.000", flush=True)
        station.execute_action(
            {"hand": {"position": {"left": [1.0], "right": [0.0]}}}
        )
        input("继续：")
        navigation.moveBy(-0.32,-0.15,0.0)

        input("步骤15：检查底盘开窗结果后按 Enter 松开左夹爪：")
        print("松开左夹爪：left=0.000，right=0.000", flush=True)
        station.execute_action(
            {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
        )

        input(
            "步骤16：左夹爪已松开，确认恢复路径安全后按 Enter 恢复"
            "应急_窗户2_准备姿态："
        )
        _apply_window2_prepare(station)
        print("任务2窗口2单步测试完成，已恢复窗户2准备姿态。", flush=True)
    finally:
        if navigation is not None:
            navigation.close()
        station.shutdown()


if __name__ == "__main__":
    main()
