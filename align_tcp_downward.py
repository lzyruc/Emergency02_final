#!/usr/bin/env python3
"""保持当前 TCP XYZ 不变，将所选 TCP 方向转到 waist_yaw_link 竖直向下。

最小调用示例：

    python align_tcp_downward.py --arm left
    python align_tcp_downward.py --arm left --execute

不加 ``--execute`` 时只读取并打印目标位姿，不发送运动命令。输入位姿和输出目标位姿均为
``waist_yaw_link`` 下的 ``[x, y, z, qx, qy, qz, qw]``；XYZ 单位为米，四元数为 XYZW。
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

import settings
from xrocs.common.data_type import Coordinate, PoseArray, Rotation
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


DOWNWARD_XYZW = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def downward_target_pose(current_pose: Any) -> np.ndarray:
    """保留 waist_yaw_link 下的 XYZ，并令 TCP +X/+Z 对齐 waist +X/+Z。"""
    current_xyz_xyzw = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    )
    return np.concatenate((current_xyz_xyzw[:3], DOWNWARD_XYZW))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("left", "right", "both"),
        help="要调整方向的左臂、右臂或双臂",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认后发送真实 MoveIt 笛卡尔运动；缺省仅打印目标",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        arm_names = ("left", "right") if args.arm == "both" else (args.arm,)
        state = station.get_robot_state()
        targets = {
            arm: downward_target_pose(state["arm_pose"][arm]) for arm in arm_names
        }
        for arm, target in targets.items():
            print(
                f"{arm}目标 Pose（waist_yaw_link，m，XYZW）="
                f"{np.round(target, 9).tolist()}",
                flush=True,
            )
        print(
            "方向约束：TCP -Z = waist_yaw_link -Z，TCP +X = waist_yaw_link +X",
            flush=True,
        )
        if not args.execute:
            print("dry-run：未发送运动命令；加 --execute 才会执行", flush=True)
            return

        input(
            "确认所选机械臂原地转向路径安全，按 Enter 执行；Ctrl+C 终止："
        )
        left_target = targets.get("left")
        right_target = targets.get("right")
        motion = settings.TASK["motion"]
        result = station.get_robot_handle().get("robot").mc.moveit_ctrler.reach_cartesian_and_wait(
            group=("dual_arm" if args.arm == "both" else f"{args.arm}_arm"),
            mode="plan_and_execute",
            use_cartesian_path=True,
            left_motion_id=1,
            right_motion_id=1,
            left_waypoints=(
                PoseArray(
                    pose_array_data=[left_target.tolist()],
                    coordinate_system_type=Coordinate.CoordinateSystem.WAIST_YAW_LINK,
                    rotation_form_type=Rotation.FormType.XYZW,
                )
                if left_target is not None
                else None
            ),
            right_waypoints=(
                PoseArray(
                    pose_array_data=[right_target.tolist()],
                    coordinate_system_type=Coordinate.CoordinateSystem.WAIST_YAW_LINK,
                    rotation_form_type=Rotation.FormType.XYZW,
                )
                if right_target is not None
                else None
            ),
            vel_scale=float(motion["moveit_velocity_scale"]),
            acc_scale=float(motion["moveit_acceleration_scale"]),
            timeout=float(motion["moveit_timeout_s"]),
        )
        print(f"MoveIt 返回：{result}", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
