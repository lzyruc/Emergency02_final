"""任务二一号窗：左腕视觉定位并横向对齐左夹爪 TCP 与窗把手。

最小调用::

    succeeded = open_window1(robot_station)

调用前机器人应已停在一号窗操作点，且操作员已确认机械臂路径安全。本阶段不导航、不闭合
夹爪，也不执行拧把手和开窗动作。输入是比赛入口创建的唯一 station；返回值表示观察姿态和
TCP 横向对齐两次 MoveIt 调用是否都成功。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import settings
from agent.perception import deprojector
from agent.perception.window_handle_detection import detect_handle_center
from arm_motion import moveit_cartesians, moveit_joints
from camera_io import read_wrist_camera


CALIBRATION_PATH = (
    Path(deprojector.__file__).resolve().parent
    / "config/camera_calibration.json"
)


def open_window1(robot_station: Any) -> bool:
    """从 ``step_1`` 观察姿态出发，将左夹爪横向对齐到窗把手中心。

    深度输入在相机边界由毫米转成米；目标位姿和 MoveIt 路点均为
    ``WAIST_YAW_LINK`` 下的 ``[x, y, z, qx, qy, qz, qw]``，位置单位米、
    四元数顺序 XYZW。TCP 局部 ``+X`` 是经维护者确认的夹爪逼近方向；本阶段只修正
    垂直于该方向的位置误差，保持逼近深度不变。相机、标定或感知异常直接向上传播。
    """
    observation_pose = settings.WINDOW1_POSES[1]
    if not moveit_joints(
        robot_station,
        left=observation_pose["left_arm"],
        right=observation_pose["right_arm"],
    ):
        return False

    _, depth_mm = read_wrist_camera(robot_station, settings.LEFT_CAMERA)
    camera_matrix = np.asarray(
        settings.LEFT_CAMERA["camera_matrix"],
        dtype=np.float64,
    )
    detection = detect_handle_center(
        np.asarray(depth_mm, dtype=np.float64) / 1000.0,
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
    )
    target_pose_waist = deprojector.Deprojector(
        robot_station,
        CALIBRATION_PATH,
        "left",
    ).deproject(detection.uv_depth_m, face_target=False)

    robot = robot_station.get_robot_handle().get("robot")
    state = robot_station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
        state["arm_joints"]["left"]
    ).get_affine_matrix_m_ndarray()
    current_position_waist = T_waist_tcp[:3, 3]
    approach_axis_waist = T_waist_tcp[:3, 0]
    position_error_waist = target_pose_waist[:3] - current_position_waist
    # WAIST_YAW_LINK 下剔除 LEFT_TCP_LINK +X（逼近方向）分量，只保留横向对齐。
    target_pose_waist[:3] = current_position_waist + position_error_waist - (
        np.dot(position_error_waist, approach_axis_waist) * approach_axis_waist
    )

    return moveit_cartesians(
        robot_station,
        left=target_pose_waist,
        use_cartesian_path=True,
        left_motion_id=1,
    )
