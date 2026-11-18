#!/usr/bin/env python3
"""单步测试文件框/垃圾桶双腕抓取点、MoveIt和夹爪控制。

运行：
    python test_bin_filebox_grasp.py filebox
    python test_bin_filebox_grasp.py bin

目标Pose和位移均使用米；目标Pose位于waist_yaw_link，TCP局部点分别位于
left_tcp_link或right_tcp_link，四元数顺序为XYZW。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from arm_motion import moveit_cartesians
from bin_filebox_grasp import (
    candidate_motion_info,
    select_candidate,
    wrist_yolo_grasp_candidates,
)
from camera_io import read_wrist_camera
from task1 import BIN_PREGRASP_PATHS, _execute_object_pose
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TARGET_NAMES = {"filebox": "文件框", "bin": "垃圾桶"}
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("filebox", "bin"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_name = TARGET_NAMES[args.target]
    prepare_path = BIN_PREGRASP_PATHS[0]
    prepare_pose = json.loads(prepare_path.read_text(encoding="utf-8"))
    output_dir = (
        Path(__file__).resolve().parent
        / "output/bin_filebox_grasp"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.target}"
    )
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
            side: deprojector_module.Deprojector(
                station,
                CALIBRATION_PATH,
                side,
            )
            for side in ("left", "right")
        }
        detector = ObstacleYolo(image_size=640)

        input(
            f"步骤1：确认双臂、腰腿路径安全，按 Enter 张开双夹爪并恢复"
            f"{prepare_path.name}；Ctrl+C 终止："
        )
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
        if not _execute_object_pose(station, prepare_pose):
            raise RuntimeError(f"{prepare_path.name}执行失败")

        input(
            f"步骤2：远观察姿态已到达，按 Enter 使用左右腕RGB-D识别"
            f"{target_name}并保存结果："
        )
        candidates = wrist_yolo_grasp_candidates(
            station,
            args.target,
            detector,
            deprojectors,
            output_dir,
            read_wrist_camera,
        )
        serializable = []
        for candidate in candidates:
            motion = candidate_motion_info(station, candidate)
            candidate.update(motion)
            point_tcp = motion["point_tcp_xyz_m"]
            delta_waist = motion["delta_waist_xyz_m"]
            print(
                f"{candidate['arm']}腕：像素={candidate['pixel_uv']}，"
                f"ROI={candidate['roi_source']}，来源={candidate['point_source']}，"
                f"置信度={candidate['confidence']:.3f}，"
                f"深度={candidate['depth_m']:.3f} m，"
                f"TCP局部点=({point_tcp[0]:+.4f}, {point_tcp[1]:+.4f}, "
                f"{point_tcp[2]:+.4f}) m，"
                f"waist位移=({delta_waist[0]:+.4f}, "
                f"{delta_waist[1]:+.4f}, {delta_waist[2]:+.4f}) m，"
                f"距离={motion['distance_m']:.4f} m，"
                f"特征={candidate['point_details']}",
                flush=True,
            )
            serializable.append(
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in candidate.items()
                }
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "candidates.json"
        result_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"RGB-D、标注图和识别结果目录：{output_dir}", flush=True)

        selected = select_candidate(args.target, candidates)
        if selected is None:
            print(f"左右腕均没有可靠的{target_name}抓取点，不执行运动", flush=True)
            return
        motion = candidate_motion_info(station, selected)
        point_tcp = motion["point_tcp_xyz_m"]
        print(
            f"步骤3选择{selected['arm']}臂："
            f"TCP局部x={point_tcp[0] * 100:+.2f} cm，"
            f"y={point_tcp[1] * 100:+.2f} cm，"
            f"z={point_tcp[2] * 100:+.2f} cm，"
            f"三维位移={motion['distance_m'] * 100:.2f} cm",
            flush=True,
        )
        robot = station.get_robot_handle().get("robot")
        state = station.get_robot_state()
        if selected["arm"] == "left":
            T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                state["arm_joints"]["left"]
            ).get_affine_matrix_m_ndarray()
        else:
            T_waist_tcp = robot.pin_kine.get_right_tcp_pose(
                state["arm_joints"]["right"]
            ).get_affine_matrix_m_ndarray()
        current_pose = state["arm_pose"][selected["arm"]]
        current_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        )
        first_delta_tcp_m = np.asarray(
            [point_tcp[0], point_tcp[1], 0.0],
            dtype=np.float64,
        )
        first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
        first_target_pose_waist = current_pose_waist.copy()
        first_target_pose_waist[:3] = (
            T_waist_tcp[:3, 3] + first_delta_waist_m
        )
        print(
            "第一段仅对齐TCP局部x、y："
            f"TCP=({first_delta_tcp_m[0] * 100:+.2f}, "
            f"{first_delta_tcp_m[1] * 100:+.2f}, +0.00) cm，"
            "末端方向保持不变",
            flush=True,
        )
        input(
            f"确认{selected['arm']}臂第一段x/y路径安全，"
            "按 Enter 执行MoveIt；Ctrl+C 终止："
        )
        if selected["arm"] == "left":
            succeeded = moveit_cartesians(
                station,
                left=[first_target_pose_waist.tolist()],
                use_cartesian_path=True,
                left_motion_id=1,
            )
        else:
            succeeded = moveit_cartesians(
                station,
                right=[first_target_pose_waist.tolist()],
                use_cartesian_path=True,
                right_motion_id=1,
            )
        if not succeeded:
            raise RuntimeError(f"{selected['arm']}臂第一次x/y对齐失败")

        state = station.get_robot_state()
        if selected["arm"] == "left":
            T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                state["arm_joints"]["left"]
            ).get_affine_matrix_m_ndarray()
        else:
            T_waist_tcp = robot.pin_kine.get_right_tcp_pose(
                state["arm_joints"]["right"]
            ).get_affine_matrix_m_ndarray()
        current_pose = state["arm_pose"][selected["arm"]]
        current_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        )
        extra_z_tcp_m = float(
            settings.TASK["bin_filebox_grasp"]
            ["second_approach_extra_z_tcp_m"][args.target]
        )
        second_delta_tcp_m = np.asarray(
            [0.0, 0.0, point_tcp[2] + extra_z_tcp_m],
            dtype=np.float64,
        )
        second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
        second_target_pose_waist = current_pose_waist.copy()
        second_target_pose_waist[:3] = (
            T_waist_tcp[:3, 3] + second_delta_waist_m
        )
        (output_dir / "motion_plan.json").write_text(
            json.dumps(
                {
                    "selected_arm": selected["arm"],
                    "point_tcp_xyz_m": point_tcp.tolist(),
                    "first_delta_tcp_xyz_m": first_delta_tcp_m.tolist(),
                    "first_delta_waist_xyz_m": first_delta_waist_m.tolist(),
                    "second_delta_tcp_xyz_m": second_delta_tcp_m.tolist(),
                    "second_delta_waist_xyz_m": second_delta_waist_m.tolist(),
                    "second_approach_extra_z_tcp_m": extra_z_tcp_m,
                    "z_rule": "recognized_tcp_z_m + configured_extra_z_tcp_m",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "第二段仅沿TCP局部z接近："
            f"识别z={point_tcp[2] * 100:+.2f} cm，"
            f"额外偏移={extra_z_tcp_m * 100:+.2f} cm，"
            f"最终z={second_delta_tcp_m[2] * 100:+.2f} cm；"
            "末端方向保持不变",
            flush=True,
        )
        input(
            f"步骤4：确认{selected['arm']}臂第二段z路径安全，"
            "按 Enter 执行MoveIt；Ctrl+C 终止："
        )
        if selected["arm"] == "left":
            succeeded = moveit_cartesians(
                station,
                left=[second_target_pose_waist.tolist()],
                use_cartesian_path=True,
                left_motion_id=2,
            )
        else:
            succeeded = moveit_cartesians(
                station,
                right=[second_target_pose_waist.tolist()],
                use_cartesian_path=True,
                right_motion_id=2,
            )
        if not succeeded:
            raise RuntimeError(f"{selected['arm']}臂第二次z接近失败")

        input(
            f"步骤5：{selected['arm']}臂已完成x/y和z两段移动，"
            "按 Enter 闭合对应夹爪；"
            "Ctrl+C 终止："
        )
        hand_positions = {
            "left": settings.TASK["hands"]["left_open_position"],
            "right": settings.TASK["hands"]["right_open_position"],
        }
        hand_positions[selected["arm"]] = settings.TASK["hands"][
            f"{selected['arm']}_close_position"
        ]
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

        input(
            f"步骤6：确认{target_name}及双臂路径安全，按 Enter 恢复"
            f"{prepare_path.name}；Ctrl+C 终止："
        )
        if not _execute_object_pose(station, prepare_pose):
            raise RuntimeError(f"恢复{prepare_path.name}失败")

        input(
            f"步骤7：远观察姿态已恢复，按 Enter 松开{selected['arm']}夹爪："
        )
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
        print(f"{target_name}抓取姿态单步测试完成", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
