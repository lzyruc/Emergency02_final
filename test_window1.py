#!/usr/bin/env python3
"""在 h270_window1_action 单独测试 Window1，不创建或控制底盘导航。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import time

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from arm_motion import moveit_cartesians, moveit_joints
from waist_leg_motion import execute_synced_waist_leg_action
from window_handle_recognition import recognize_window_handle
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M = 0.35
WINDOW_HANDLE_RECOGNITION_ATTEMPTS = 3
WINDOW_PUSH_MAX_ATTEMPTS = 4
WINDOW_PUSH_RETRY_SCALE = 0.9
WINDOW_PUSH_MIN_RETRY_TRAVEL_M = 0.01
WINDOW_PUSH_INITIAL_DELTA_TCP_M = (0.0, 0.0, -0.35)


def recognize_window_handle_with_tcp_z_retry(
    station,
    output_dir: Path,
    max_delta_z_m: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray] | None:
    """最多识别三次，只接受left_tcp_link前方指定距离内的目标。"""
    deprojector = deprojector_module.Deprojector(
        station,
        CALIBRATION_PATH,
        "left",
    )
    robot = station.get_robot_handle().get("robot")
    for attempt in range(1, WINDOW_HANDLE_RECOGNITION_ATTEMPTS + 1):
        attempt_dir = output_dir / f"attempt_{attempt}"
        result = recognize_window_handle(station, attempt_dir)
        target_pose_waist = deprojector.deproject(
            [*result["pixel_uv"], result["depth_m"]],
            face_target=False,
        )
        state = station.get_robot_state()
        T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
            state["arm_joints"]["left"]
        ).get_affine_matrix_m_ndarray()
        point_left_tcp_m = (
            np.linalg.inv(T_waist_tcp)
            @ np.append(target_pose_waist[:3], 1.0)
        )[:3]
        delta_z_m = float(point_left_tcp_m[2])
        result.update(
            {
                "recognition_attempt": attempt,
                "point_left_tcp_xyz_m": point_left_tcp_m.tolist(),
                "tcp_delta_z_accepted_range_m": [-max_delta_z_m, 0.0],
            }
        )
        (attempt_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"窗把手第{attempt}/{WINDOW_HANDLE_RECOGNITION_ATTEMPTS}次识别："
            f"left_tcp_link delta-x={point_left_tcp_m[0] * 100:+.2f} cm，"
            f"delta-y={point_left_tcp_m[1] * 100:+.2f} cm，"
            f"delta-z={delta_z_m * 100:+.2f} cm，"
            f"前向上限={max_delta_z_m * 100:.1f} cm",
            flush=True,
        )
        if -max_delta_z_m <= delta_z_m <= 0.0:
            return result, target_pose_waist, T_waist_tcp, point_left_tcp_m
        if attempt < WINDOW_HANDLE_RECOGNITION_ATTEMPTS:
            time.sleep(0.1)
    print("窗把手识别失败", flush=True)
    return None


def push_window_with_reduced_offset(
    station,
    robot,
    *,
    first_motion_id: int,
    initial_delta_tcp_m: np.ndarray,
) -> bool:
    """以首次TCP位姿为基准，失败后缩短总推窗偏移，避免叠加位移。"""
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
        state["arm_joints"]["left"]
    ).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    start_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    ).copy()
    initial_delta_tcp_m = np.asarray(
        initial_delta_tcp_m,
        dtype=np.float64,
    ).copy()
    initial_delta_waist_m = T_waist_tcp[:3, :3] @ initial_delta_tcp_m
    initial_distance_m = float(np.linalg.norm(initial_delta_waist_m))
    if initial_distance_m == 0.0:
        raise ValueError("推窗TCP偏移不能为零")
    push_direction_waist = initial_delta_waist_m / initial_distance_m

    for attempt in range(1, WINDOW_PUSH_MAX_ATTEMPTS + 1):
        scale = WINDOW_PUSH_RETRY_SCALE ** (attempt - 1)
        delta_tcp_m = initial_delta_tcp_m * scale
        delta_waist_m = initial_delta_waist_m * scale
        target_pose_waist = start_pose_waist.copy()
        target_pose_waist[:3] = start_pose_waist[:3] + delta_waist_m
        attempt_label = "首次" if attempt == 1 else f"第{attempt}次缩距重试"
        input(
            f"推窗：确认机械臂已停止且路径安全，按 Enter {attempt_label}"
            f"向前移动；相对首次起点的TCP总偏移="
            f"({delta_tcp_m[0] * 100:+.1f}, "
            f"{delta_tcp_m[1] * 100:+.1f}, "
            f"{delta_tcp_m[2] * 100:+.1f}) cm；Ctrl+C 终止："
        )
        if attempt > 1:
            latest_state = station.get_robot_state()
            latest_T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                latest_state["arm_joints"]["left"]
            ).get_affine_matrix_m_ndarray()
            remaining_forward_m = float(
                np.dot(
                    target_pose_waist[:3] - latest_T_waist_tcp[:3, 3],
                    push_direction_waist,
                )
            )
            if remaining_forward_m <= WINDOW_PUSH_MIN_RETRY_TRAVEL_M:
                print("当前TCP已达到或越过缩距目标，视为推窗成功", flush=True)
                return True

        motion_id = first_motion_id + attempt - 1
        if moveit_cartesians(
            station,
            left=target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=motion_id,
        ):
            print(
                f"推窗成功：第{attempt}次尝试，"
                f"偏移={np.linalg.norm(delta_tcp_m) * 100:.1f} cm，"
                f"motion_id={motion_id}",
                flush=True,
            )
            return True
        if attempt < WINDOW_PUSH_MAX_ATTEMPTS:
            next_distance_cm = (
                np.linalg.norm(initial_delta_tcp_m)
                * WINDOW_PUSH_RETRY_SCALE**attempt
                * 100.0
            )
            print(
                f"第{attempt}次推窗失败，"
                f"下一次总偏移缩短为{next_distance_cm:.1f} cm",
                flush=True,
            )
    print(f"连续{WINDOW_PUSH_MAX_ATTEMPTS}次推窗失败", flush=True)
    return False


def run_window1(station) -> bool:
    """执行 Window1 姿态、识别、TCP 对齐、推窗和恢复流程。"""
    input(
        "步骤1：确认机器人实际位于 h270_window1_action，底盘不会移动；"
        "确认腰腿路径安全后按 Enter 回正腰腿："
    )
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": [0.0, 0.02]},
            "leg": {"position": [-0.02, 0.02]},
        },
    )

    for index, window_pose in enumerate(settings.WINDOW1_APPROACH_POSES):
        input(
            f"步骤{index + 2}：确认左臂路径安全，按 Enter 执行"
            f"窗口迫近 new_step{index}；Ctrl+C 终止："
        )
        if not moveit_joints(station, left=window_pose["left_arm"]):
            raise RuntimeError(f"窗口迫近 new_step{index} 执行失败")

    output_dir = (
        settings.ROOT
        / "output/window1_test"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    input(
        "步骤5：new_step0～2已完成，按 Enter 读取左腕 RGB-D、保存图像并"
        "识别窗把手；Ctrl+C 终止："
    )
    recognized = recognize_window_handle_with_tcp_z_retry(
        station,
        output_dir,
        WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M,
    )
    if recognized is None:
        print("Window1 把手连续识别超出 z 阈值，跳过运动", flush=True)
        return False
    result, target_pose_waist, T_waist_tcp, point_left_tcp_m = recognized
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    robot = station.get_robot_handle().get("robot")
    first_delta_tcp_m = np.asarray(
        [point_left_tcp_m[0], point_left_tcp_m[1], 0.0],
        dtype=np.float64,
    )
    first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(
        "识别点（left_tcp_link）："
        f"x={point_left_tcp_m[0] * 100:+.2f} cm，"
        f"y={point_left_tcp_m[1] * 100:+.2f} cm，"
        f"z={point_left_tcp_m[2] * 100:+.2f} cm",
        flush=True,
    )
    print(
        "第一次TCP局部偏移："
        f"x={first_delta_tcp_m[0] * 100:+.2f} cm，"
        f"y={first_delta_tcp_m[1] * 100:+.2f} cm，z=+0.00 cm",
        flush=True,
    )
    input("步骤5：确认路径安全，按 Enter 只修正 TCP 局部 x、y：")
    if not moveit_cartesians(
        station,
        left=first_target_pose_waist,
        use_cartesian_path=True,
        left_motion_id=1,
    ):
        raise RuntimeError("Window1 左臂第一次 x/y 修正失败")

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
    second_delta_tcp_m = np.asarray(
        [0.0, 0.0, point_left_tcp_m[2] + 0.20],
        dtype=np.float64,
    )
    second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
    second_target_pose_waist = current_pose_waist.copy()
    second_target_pose_waist[:3] = T_waist_tcp[:3, 3] + second_delta_waist_m
    print(
        "第二次TCP局部z偏移："
        f"识别z={point_left_tcp_m[2] * 100:+.2f} cm + 20.00 cm "
        f"= {second_delta_tcp_m[2] * 100:+.2f} cm",
        flush=True,
    )
    result.update(
        {
            "point_left_tcp_xyz_m": point_left_tcp_m.tolist(),
            "first_delta_left_tcp_xyz_m": first_delta_tcp_m.tolist(),
            "second_delta_left_tcp_xyz_m": second_delta_tcp_m.tolist(),
            "second_z_rule": "recognized_left_tcp_z_m + 0.20 m",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    input("步骤6：确认路径安全，按 Enter 只沿 TCP 局部 z 移动：")
    if not moveit_cartesians(
        station,
        left=second_target_pose_waist,
        use_cartesian_path=True,
        left_motion_id=2,
    ):
        raise RuntimeError("Window1 左臂第二次 z 修正失败")

    input("步骤7：TCP 的 x/y/z 对齐完成，本步骤不闭合夹爪；按 Enter 继续：")
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    right_target_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    ).copy()
    right_target_pose_waist[1] -= 0.10
    input("步骤8：确认路径安全，按 Enter 沿 waist_yaw_link -Y 向右移动10 cm：")
    if not moveit_cartesians(
        station,
        left=right_target_pose_waist,
        use_cartesian_path=True,
        left_motion_id=3,
    ):
        raise RuntimeError("Window1 左臂向右移动10 cm失败")

    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    upward_target_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    ).copy()
    upward_target_pose_waist[2] += 0.04
    input("步骤9：确认路径安全，按 Enter 沿 waist_yaw_link +Z 向上移动4 cm：")
    if not moveit_cartesians(
        station,
        left=upward_target_pose_waist,
        use_cartesian_path=True,
        left_motion_id=4,
    ):
        raise RuntimeError("Window1 左臂向上移动4 cm失败")

    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    rotate_target_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    ).copy()
    qx, qy, qz, qw = rotate_target_pose_waist[3:]
    half_angle_rad = math.radians(-5.0) / 2.0
    sin_half = math.sin(half_angle_rad)
    cos_half = math.cos(half_angle_rad)
    rotate_target_pose_waist[3:] = np.asarray(
        [
            qx * cos_half - qz * sin_half,
            qw * sin_half + qy * cos_half,
            qx * sin_half + qz * cos_half,
            qw * cos_half - qy * sin_half,
        ],
        dtype=np.float64,
    )
    rotate_target_pose_waist[3:] /= np.linalg.norm(
        rotate_target_pose_waist[3:]
    )
    input(
        "步骤10：确认原地旋转路径安全，按 Enter 绕 TCP 局部 -Y 轴"
        "旋转5°，位置保持不变："
    )
    if not moveit_cartesians(
        station,
        left=rotate_target_pose_waist,
        use_cartesian_path=True,
        left_motion_id=5,
    ):
        raise RuntimeError("Window1 左臂 TCP 旋转5°失败")

    push_succeeded = push_window_with_reduced_offset(
        station,
        robot,
        first_motion_id=7,
        initial_delta_tcp_m=np.asarray(
            WINDOW_PUSH_INITIAL_DELTA_TCP_M,
            dtype=np.float64,
        ),
    )

    for index in range(9, 14):
        input(
            f"恢复步骤 step_{index}：确认左臂路径安全后按 Enter 执行；"
            "Ctrl+C 终止："
        )
        window_pose = settings.WINDOW1_POSES[index]
        if not moveit_joints(station, left=window_pose["left_arm"]):
            raise RuntimeError(f"Window1 恢复 step_{index} 执行失败")

    if not push_succeeded:
        print("推窗缩距重试仍失败，但恢复姿态已执行", flush=True)
        return False
    print(f"Window1 单测完成；结果目录：{output_dir}", flush=True)
    return True


def main() -> None:
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
        run_window1(station)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
