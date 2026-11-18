"""Recognize and vertically grasp the task3 small cylindrical pull cap.

The camera point is transformed as ``WAIST <- right TCP <- right camera``.
Motion is split in the current right-TCP frame: first local x/y with zero z,
then local z with zero x/y. The right gripper closes only after both MoveIt
motions succeed.

Minimal integration call::

    result = grasp_task3_pull_cap(
        station,
        {"left": 0.0, "right": 0.0},
    )
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import settings
from arm_motion import moveit_cartesians
from camera_io import read_wrist_camera
from task3_pull_cap_detector import (
    PullCapDetection,
    PullCapDetector,
    save_pull_cap_debug_images,
)
from wrist_self_mask import gripper_mask


class PullCapRecognitionError(RuntimeError):
    """Task3 cap or slot recognition failed before any new motion is issued."""


def _pose_xyz_xyzw(pose: Any) -> np.ndarray:
    values = (
        pose.get_xyz_m_xyzw_ndarray()
        if hasattr(pose, "get_xyz_m_xyzw_ndarray")
        else pose
    )
    return np.asarray(values, dtype=np.float64)


def _rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def calculate_pull_cap_targets(
    point_camera_m: np.ndarray,
    camera_to_right_tcp: np.ndarray,
    current_tcp_pose_waist: np.ndarray,
    final_offset_right_tcp_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return point TCP, two local deltas and two waist-frame target poses."""
    point_tcp_m = (
        np.asarray(camera_to_right_tcp, dtype=np.float64)
        @ np.append(np.asarray(point_camera_m, dtype=np.float64), 1.0)
    )[:3]
    final_point_tcp_m = point_tcp_m + np.asarray(
        final_offset_right_tcp_m,
        dtype=np.float64,
    )
    first_delta_tcp_m = np.asarray(
        [final_point_tcp_m[0], final_point_tcp_m[1], 0.0],
        dtype=np.float64,
    )
    second_delta_tcp_m = np.asarray(
        [0.0, 0.0, final_point_tcp_m[2]],
        dtype=np.float64,
    )
    rotation_waist_tcp = _rotation_xyzw(current_tcp_pose_waist[3:])
    first_pose_waist = current_tcp_pose_waist.copy()
    first_pose_waist[:3] += rotation_waist_tcp @ first_delta_tcp_m
    second_pose_waist = first_pose_waist.copy()
    second_pose_waist[:3] += rotation_waist_tcp @ second_delta_tcp_m
    return (
        point_tcp_m,
        first_delta_tcp_m,
        second_delta_tcp_m,
        first_pose_waist,
        second_pose_waist,
    )


def recognize_task3_pull_cap(
    station: Any,
    output_dir: Path,
    *,
    confirm: bool = False,
    prompt_label: str = "任务3小圆柱帽",
) -> PullCapDetection:
    """Read right-wrist RGB-D, detect the cap, and save visual diagnostics."""
    if confirm:
        input(f"确认机器人保持静止，按 Enter 开始{prompt_label}识别；Ctrl+C 终止：")
    calibration = settings.RIGHT_CAMERA
    bgr, depth_mm = read_wrist_camera(station, calibration)
    try:
        detection = PullCapDetector(calibration["camera_matrix"]).detect(
            bgr,
            depth_mm,
            roi=tuple(settings.TASK["task3_pull_cap"]["right_roi_normalized_xyxy"]),
            exclusion_mask=gripper_mask(depth_mm.shape),
            slot_radius_scale=float(
                settings.TASK["task3_pull_cap"]["slot_radius_scale"]
            ),
        )
    except RuntimeError as error:
        raise PullCapRecognitionError(str(error)) from error
    save_pull_cap_debug_images(output_dir, bgr, depth_mm, detection)
    return detection


def move_right_tcp_local_z(
    station: Any,
    delta_z_tcp_m: float,
) -> tuple[bool, list[float]]:
    """Move only along current right_tcp_link z and return result plus target Pose."""
    current_pose = _pose_xyz_xyzw(
        station.get_robot_state()["arm_pose"]["right"]
    )
    target_pose = current_pose.copy()
    target_pose[:3] += _rotation_xyzw(current_pose[3:]) @ np.asarray(
        [0.0, 0.0, float(delta_z_tcp_m)],
        dtype=np.float64,
    )
    succeeded = moveit_cartesians(
        station,
        right=target_pose.tolist(),
        use_cartesian_path=True,
    )
    return succeeded, target_pose.tolist()


def grasp_task3_pull_cap(
    station: Any,
    hand_positions: dict[str, float],
    *,
    execute: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Recognize the right-side cap and optionally execute x/y, z, close."""
    calibration = settings.RIGHT_CAMERA
    target_dir = output_dir or settings.RUNTIME_DIR / "task3_right_pull_cap"
    detection = recognize_task3_pull_cap(
        station,
        target_dir,
        confirm=execute,
        prompt_label="初次小圆柱帽",
    )
    current_tcp_pose_waist = _pose_xyz_xyzw(
        station.get_robot_state()["arm_pose"]["right"]
    )
    (
        point_tcp_m,
        first_delta_tcp_m,
        second_delta_tcp_m,
        first_pose_waist,
        second_pose_waist,
    ) = calculate_pull_cap_targets(
        detection.point_camera_m,
        np.asarray(
            calibration["extrinsics_cam_to_right_tcp_link_matrix"],
            dtype=np.float64,
        ),
        current_tcp_pose_waist,
        np.asarray(
            settings.TASK["task3_pull_cap"]["final_grasp_offset_right_tcp_m"],
            dtype=np.float64,
        ),
    )
    result = {
        "success": True,
        "camera_key": calibration["station_camera_key"],
        "cap_pixel_uv": list(detection.cap_pixel_uv),
        "large_circle_uvr": (
            None
            if detection.large_circle_uvr is None
            else list(detection.large_circle_uvr)
        ),
        "cap_circle_uvr": list(detection.cap_circle_uvr),
        "selection_strategy": detection.selection_strategy,
        "confidence": detection.confidence,
        "depth_m": detection.depth_m,
        "point_camera_optical_xyz_m": detection.point_camera_m.tolist(),
        "point_right_tcp_xyz_m": point_tcp_m.tolist(),
        "first_delta_right_tcp_xyz_m": first_delta_tcp_m.tolist(),
        "second_delta_right_tcp_xyz_m": second_delta_tcp_m.tolist(),
        "first_target_pose_waist_xyz_xyzw": first_pose_waist.tolist(),
        "second_target_pose_waist_xyz_xyzw": second_pose_waist.tolist(),
        "camera_axes": "+x right, +y down, +z forward",
        "motion_axes": "right_tcp_link; first x/y, then z",
        "length_unit": "m",
        "quaternion_order": "xyzw",
        "executed": False,
        "xy_motion_succeeded": False,
        "z_motion_succeeded": False,
        "hand_executed": False,
    }

    if execute:
        print(
            "任务3小圆柱帽："
            f"像素={detection.cap_pixel_uv}，深度={detection.depth_m:.3f} m，"
            f"TCP点={point_tcp_m.tolist()} m",
            flush=True,
        )
        print(
            "第一段只移动right_tcp_link局部x/y："
            f"{first_delta_tcp_m.tolist()} m",
            flush=True,
        )
        input(
            "确认初次抓取TCP局部x/y路径安全，按 Enter 执行MoveIt；"
            "Ctrl+C 终止："
        )
        result["executed"] = True
        result["xy_motion_succeeded"] = moveit_cartesians(
            station,
            right=first_pose_waist.tolist(),
            use_cartesian_path=True,
        )
        if result["xy_motion_succeeded"]:
            current_after_xy = _pose_xyz_xyzw(
                station.get_robot_state()["arm_pose"]["right"]
            )
            second_pose_waist = current_after_xy.copy()
            second_pose_waist[:3] += (
                _rotation_xyzw(current_after_xy[3:]) @ second_delta_tcp_m
            )
            result["second_target_pose_waist_xyz_xyzw"] = (
                second_pose_waist.tolist()
            )
            print(
                "第二段只移动right_tcp_link局部z："
                f"{second_delta_tcp_m.tolist()} m",
                flush=True,
            )
            input(
                "确认初次抓取TCP局部z路径安全，按 Enter 执行MoveIt；"
                "Ctrl+C 终止："
            )
            result["z_motion_succeeded"] = moveit_cartesians(
                station,
                right=second_pose_waist.tolist(),
                use_cartesian_path=True,
            )
        if result["z_motion_succeeded"]:
            input("确认夹爪已对准小圆柱帽，按 Enter 闭合右夹爪；Ctrl+C 终止：")
            next_hand_positions = dict(hand_positions)
            next_hand_positions["right"] = float(
                settings.TASK["hands"]["right_close_position"]
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
            time.sleep(1.0)
            result["hand_positions_after"] = next_hand_positions
            result["hand_executed"] = True
        result["success"] = bool(result["hand_executed"])

    (target_dir / "target.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def align_task3_reel_slot(
    station: Any,
    *,
    execute: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Re-detect cap/reel, align x/y to the nearest of three slots, then z."""
    target_dir = output_dir or settings.RUNTIME_DIR / "task3_reel_slot"
    detection = recognize_task3_pull_cap(
        station,
        target_dir,
        confirm=execute,
        prompt_label="拉动后小圆及插槽",
    )
    if len(detection.slot_candidates) != 3:
        raise PullCapRecognitionError("未获得大圆其余三个方向的完整插槽候选")

    camera_to_tcp = np.asarray(
        settings.RIGHT_CAMERA["extrinsics_cam_to_right_tcp_link_matrix"],
        dtype=np.float64,
    )
    candidates = []
    for slot in detection.slot_candidates:
        point_tcp = (
            camera_to_tcp @ np.append(slot.point_camera_m, 1.0)
        )[:3]
        candidates.append((float(np.linalg.norm(point_tcp[:2])), slot, point_tcp))
    _, selected, selected_point_tcp = min(candidates, key=lambda item: item[0])
    current_pose = _pose_xyz_xyzw(
        station.get_robot_state()["arm_pose"]["right"]
    )
    xy_delta_tcp = np.asarray(
        [selected_point_tcp[0], selected_point_tcp[1], 0.0],
        dtype=np.float64,
    )
    xy_target_pose = current_pose.copy()
    xy_target_pose[:3] += _rotation_xyzw(current_pose[3:]) @ xy_delta_tcp
    insert_z_tcp_m = float(
        settings.TASK["task3_pull_cap"]["slot_insert_z_tcp_m"]
    )
    result = {
        "cap_pixel_uv": list(detection.cap_pixel_uv),
        "large_circle_uvr": list(detection.large_circle_uvr),
        "slot_candidates": [
            {
                "quarter_turns_from_cap": slot.direction_quarter_turns_from_cap,
                "pixel_uv": list(slot.pixel_uv),
                "point_camera_m": slot.point_camera_m.tolist(),
            }
            for slot in detection.slot_candidates
        ],
        "selected_slot_quarter_turns_from_cap": (
            selected.direction_quarter_turns_from_cap
        ),
        "selected_slot_pixel_uv": list(selected.pixel_uv),
        "selected_slot_point_right_tcp_m": selected_point_tcp.tolist(),
        "xy_delta_right_tcp_m": xy_delta_tcp.tolist(),
        "slot_insert_z_right_tcp_m": insert_z_tcp_m,
        "xy_motion_succeeded": False,
        "z_motion_succeeded": False,
    }
    if execute:
        print(
            "插槽选择："
            f"候选像素={[slot.pixel_uv for slot in detection.slot_candidates]}，"
            f"选择={selected.pixel_uv}，TCP x/y={xy_delta_tcp[:2].tolist()} m",
            flush=True,
        )
        input(
            "确认插槽TCP局部x/y路径安全，按 Enter 执行MoveIt；"
            "Ctrl+C 终止："
        )
        result["xy_motion_succeeded"] = moveit_cartesians(
            station,
            right=xy_target_pose.tolist(),
            use_cartesian_path=True,
        )
        if result["xy_motion_succeeded"]:
            input(
                f"确认插槽TCP局部z={insert_z_tcp_m:+.3f} m路径安全，"
                "按 Enter 执行MoveIt；Ctrl+C 终止："
            )
            (
                result["z_motion_succeeded"],
                result["z_target_pose_waist_xyz_xyzw"],
            ) = move_right_tcp_local_z(station, insert_z_tcp_m)

    (target_dir / "slot_target.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
