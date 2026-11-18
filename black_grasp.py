"""Detect and grasp a black cylinder with the selected wrist and arm."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import settings
from arm_motion import moveit_cartesians
from black_cylinder_detector import BlackCylinderDetector, save_debug_images
from blue_grasp import pose_xyz_xyzw
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask


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


def transform_black_targets(
    point_camera_m: np.ndarray,
    camera_to_tcp: np.ndarray,
    current_tcp_pose_waist: np.ndarray,
    final_offset_tcp_m: np.ndarray,
    pregrasp_extra_tcp_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point in TCP and pregrasp/final positions in waist_yaw_link."""
    point_tcp = (
        np.asarray(camera_to_tcp, dtype=np.float64)
        @ np.append(np.asarray(point_camera_m, dtype=np.float64), 1.0)
    )[:3]
    waist_from_tcp_rotation = _quaternion_xyzw_to_matrix(
        current_tcp_pose_waist[3:]
    )
    final = (
        current_tcp_pose_waist[:3]
        + waist_from_tcp_rotation
        @ (point_tcp + np.asarray(final_offset_tcp_m, dtype=np.float64))
    )
    # The visual correction only changes waist_yaw_link x/y. Keep the current
    # TCP height so the grasp never introduces a world-frame vertical motion.
    final[2] = current_tcp_pose_waist[2]
    pregrasp = (
        final
        + waist_from_tcp_rotation
        @ np.asarray(pregrasp_extra_tcp_m, dtype=np.float64)
    )
    pregrasp[2] = current_tcp_pose_waist[2]
    return point_tcp, pregrasp, final


def run_black_grasp(
    station: Any,
    calibration: dict[str, Any],
    task_config: dict[str, Any],
    hand_positions: dict[str, float],
    *,
    execute: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Calculate or execute a black-cylinder grasp; 相机读取失败返回 None。"""
    arm = str(calibration["arm_name"])
    if arm not in ("left", "right"):
        raise ValueError("black grasp arm_name must be 'left' or 'right'")
    camera_config = {
        "station_camera_key": calibration["station_camera_key"],
        "image_width": int(calibration["image_width"]),
        "image_height": int(calibration["image_height"]),
        "color_encoding": calibration["color_encoding"],
        "depth_unit": calibration["depth_unit"],
    }
    roi = tuple(
        float(value)
        for value in task_config["black_grasp"][f"{arm}_roi_normalized_xyxy"]
    )
    print(
        f"黑色隔离带识别开始：控制臂={arm}，"
        f"相机={camera_config['station_camera_key']}，ROI={roi}",
        flush=True,
    )
    try:
        bgr, depth_mm = read_wrist_camera(station, camera_config)
    except RuntimeError as exc:
        print(f"未获取到{arm}腕相机图像，跳过本次抓取：{exc}", flush=True)
        return None
    state = station.get_robot_state()
    current_tcp = pose_xyz_xyzw(state["arm_pose"][arm])
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    camera_to_tcp = np.asarray(
        calibration[f"extrinsics_cam_to_{arm}_tcp_link_matrix"],
        dtype=np.float64,
    )
    detection = BlackCylinderDetector(camera_matrix).detect(
        bgr,
        depth_mm,
        roi=roi,
        camera_to_tcp=camera_to_tcp,
        exclusion_mask=gripper_mask(depth_mm.shape),
    )
    point_tcp, pregrasp, final = transform_black_targets(
        detection.point_camera_m,
        camera_to_tcp,
        current_tcp,
        np.asarray(
            task_config["black_grasp"][f"final_grasp_offset_{arm}_tcp_m"],
            dtype=np.float64,
        ),
        np.asarray(
            task_config["black_grasp"][f"pregrasp_extra_{arm}_tcp_m"],
            dtype=np.float64,
        ),
    )
    travel_m = max(
        float(np.linalg.norm(pregrasp - current_tcp[:3])),
        float(np.linalg.norm(final - current_tcp[:3])),
    )
    print(
        "黑色隔离带识别结果："
        f"控制臂={arm}，像素={detection.pixel_uv}，"
        f"置信度={detection.confidence:.3f}，深度={detection.depth_m:.3f} m，"
        f"主轴角={detection.angle_deg:.1f}°，"
        f"长轴={detection.major_axis_px:.1f} px，"
        f"短轴={detection.minor_axis_px:.1f} px，"
        f"候选抓取点={len(detection.grasp_candidates)}，"
        f"选择策略={detection.selection_strategy}",
        flush=True,
    )
    print(
        "黑色隔离带三维结果："
        f"camera_xyz={np.round(detection.point_camera_m, 4).tolist()} m，"
        f"{arm}_tcp_xyz={np.round(point_tcp, 4).tolist()} m，"
        f"waist预抓取xyz={np.round(pregrasp, 4).tolist()} m，"
        f"waist抓取xyz={np.round(final, 4).tolist()} m，"
        f"最大TCP位移={travel_m:.3f} m",
        flush=True,
    )

    result = {
        "pixel_uv": list(detection.pixel_uv),
        "controlled_arm": arm,
        "camera_key": camera_config["station_camera_key"],
        "confidence": detection.confidence,
        "depth_m": detection.depth_m,
        "angle_deg": detection.angle_deg,
        "major_axis_px": detection.major_axis_px,
        "minor_axis_px": detection.minor_axis_px,
        "grasp_candidate_count": len(detection.grasp_candidates),
        "selection_strategy": detection.selection_strategy,
        "point_camera_m": detection.point_camera_m.tolist(),
        "point_tcp_m": point_tcp.tolist(),
        "pregrasp_tcp_pose_waist_xyz_xyzw": [
            *pregrasp.tolist(),
            *current_tcp[3:].tolist(),
        ],
        "grasp_tcp_pose_waist_xyz_xyzw": [
            *final.tolist(),
            *current_tcp[3:].tolist(),
        ],
        "coordinate_frame": "waist_yaw_link",
        "quaternion_order": "xyzw",
        "tcp_travel_m": travel_m,
        "hand_positions_before": dict(hand_positions),
        "executed": False,
        "motion_succeeded": False,
    }
    if output_dir is not None:
        save_debug_images(output_dir, bgr, depth_mm, detection)

    if execute:
        waypoints = [
            result["pregrasp_tcp_pose_waist_xyz_xyzw"],
            result["grasp_tcp_pose_waist_xyz_xyzw"],
        ]
        print(
            f"黑色隔离带运动下发：控制臂={arm}，MoveIt路点数={len(waypoints)}",
            flush=True,
        )
        if arm == "left":
            motion_succeeded = moveit_cartesians(station, left=waypoints)
        else:
            motion_succeeded = moveit_cartesians(station, right=waypoints)
        if not motion_succeeded:
            print(f"{arm}臂未到达目标位姿，仍继续闭合夹爪", flush=True)
        next_hand_positions = dict(hand_positions)
        next_hand_positions[arm] = float(
            task_config["hands"][f"{arm}_close_position"]
        )
        print(
            "黑色隔离带夹爪命令："
            f"控制臂={arm}，left={next_hand_positions['left']:.3f}，"
            f"right={next_hand_positions['right']:.3f}",
            flush=True,
        )
        station.execute_action(
            {
                "hand": {
                    "position": {
                        "left": [next_hand_positions["left"]],
                        "right": [next_hand_positions["right"]],
                    }
                }
            }
        )
        print(f"黑色隔离带夹爪命令已由 xrocs 返回：控制臂={arm}", flush=True)
        time.sleep(5.0)
        result["hand_positions_after"] = next_hand_positions
        result["motion_succeeded"] = motion_succeeded
        result["executed"] = True
    if output_dir is not None:
        (output_dir / "target.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def grasp_black(
    station: Any,
    arm: str,
    hand_positions: dict[str, float],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    """用指定侧腕部相机识别黑色柱状体，经 QP 抓取并闭合该侧夹爪；相机读取失败返回 None。"""
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    calibration = (
        settings.RIGHT_CAMERA if arm == "right" else settings.LEFT_CAMERA
    )
    print(
        f"黑色隔离带调用：请求臂={arm}，"
        f"标定臂={calibration['arm_name']}，"
        f"相机={calibration['station_camera_key']}，"
        f"当前夹爪位置={hand_positions}",
        flush=True,
    )
    return run_black_grasp(
        station,
        calibration,
        settings.TASK,
        hand_positions,
        output_dir=output_dir or settings.RUNTIME_DIR / f"{arm}_black_grasp",
    )
