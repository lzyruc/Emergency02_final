"""Detect a blue RGB-D target and optionally grasp it through xrocs."""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np

import settings
from arm_motion import direct_act_tcp, moveit_cartesians
from blue_depth_target import BlueDepthTargetDetector, save_debug_images
from camera_io import read_wrist_camera


def pose_xyz_xyzw(pose) -> np.ndarray:
    values = (
        pose.get_xyz_m_xyzw_ndarray()
        if hasattr(pose, "get_xyz_m_xyzw_ndarray")
        else pose
    )
    return np.asarray(values, dtype=np.float64)


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_target(
    point_camera_m: np.ndarray,
    camera_to_tcp: np.ndarray,
    tcp_pose_waist: np.ndarray,
    grasp_offset_tcp_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply T_waist_tcp @ T_tcp_camera to a camera-frame target point."""
    point_camera_h = np.append(np.asarray(point_camera_m, dtype=np.float64), 1.0)
    point_tcp = (camera_to_tcp @ point_camera_h)[:3]
    target_tcp = point_tcp + np.asarray(grasp_offset_tcp_m, dtype=np.float64)
    waist_from_tcp = np.eye(4, dtype=np.float64)
    waist_from_tcp[:3, :3] = _quaternion_xyzw_to_matrix(tcp_pose_waist[3:])
    waist_from_tcp[:3, 3] = tcp_pose_waist[:3]
    target_waist = (waist_from_tcp @ np.append(target_tcp, 1.0))[:3]
    target_pose_waist = tcp_pose_waist.copy()
    target_pose_waist[:3] = target_waist
    return point_tcp, target_tcp, target_pose_waist


def run_blue_grasp(
    station,
    config: dict,
    task_config: dict,
    *,
    execute: bool = False,
    output_dir: Path | None = None,
) -> dict | bool:
    """Calculate or execute one right-arm grasp; 相机读取失败返回 None。"""
    state = station.get_robot_state()
    current_tcp_pose = pose_xyz_xyzw(state["arm_pose"]["right"])
    try:
        bgr, depth_mm = read_wrist_camera(station, config)
    except RuntimeError as exc:
        print(f"未获取到右腕相机图像，跳过本次抓取：{exc}", flush=True)
        return False
    camera_matrix = np.asarray(config["camera_matrix"], dtype=np.float64)
    detection = BlueDepthTargetDetector(camera_matrix).detect(bgr, depth_mm)
    point_tcp, target_tcp, target_pose = transform_target(
        detection.point_camera_m,
        np.asarray(
            config["extrinsics_cam_to_right_tcp_link_matrix"],
            dtype=np.float64,
        ),
        current_tcp_pose,
        np.asarray(config["grasp_offset_right_tcp_m"], dtype=np.float64),
    )
    travel_m = float(np.linalg.norm(target_pose[:3] - current_tcp_pose[:3]))

    result = {
        "pixel_uv": list(detection.pixel_uv),
        "confidence": detection.confidence,
        "point_camera_m": detection.point_camera_m.tolist(),
        "point_right_tcp_m": point_tcp.tolist(),
        "grasp_offset_right_tcp_m": list(config["grasp_offset_right_tcp_m"]),
        "target_right_tcp_m": target_tcp.tolist(),
        "target_right_tcp_pose_waist_xyz_xyzw": target_pose.tolist(),
        "current_right_tcp_pose_waist_xyz_xyzw": current_tcp_pose.tolist(),
        "tcp_travel_m": travel_m,
        "coordinate_frame": "waist_yaw_link",
        "quaternion_order": "xyzw",
        "executed": False,
        "motion_succeeded": False,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "rgb_raw.jpg"), bgr)
        cv2.imwrite(str(output_dir / "depth_raw_mm.png"), depth_mm)
        save_debug_images(output_dir, bgr, detection)

    if execute:
        print("抓文件筐中")
        motion_succeeded = moveit_cartesians(station, right=[target_pose.tolist()])
        # motion_succeeded = direct_act_tcp(station, "right", [target_pose.tolist()])
        if not motion_succeeded:
            print("右臂未到达目标位姿，仍继续闭合夹爪", flush=True)
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [0.0],
                        "right": [1.0],
                    }
                }
            }
        )
        time.sleep(1.0)
        result["motion_succeeded"] = motion_succeeded
        result["executed"] = True

    if output_dir is not None:
        (output_dir / "target.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def grasp_blue(
    station,
    *,
    output_dir: Path | None = settings.RUNTIME_DIR / "blue_grasp",
) -> dict | None:
    """右腕 RGB-D 计算笛卡尔目标，经 QP 完成右臂抓取并闭合右夹爪；相机读取失败返回 None。"""
    return run_blue_grasp(
        station,
        settings.RIGHT_CAMERA,
        settings.TASK,
        execute=True,
        output_dir=output_dir,
    )
