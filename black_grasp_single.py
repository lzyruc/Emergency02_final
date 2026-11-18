"""测试任务3使用单次三维视觉修正抓取黑色隔离带。"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import settings
from arm_motion import moveit_cartesians
from black_cylinder_detector import BlackCylinderDetector, save_debug_images
from blue_grasp import pose_xyz_xyzw
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask


def _waist_from_tcp_matrix(pose_xyz_xyzw: list[float]) -> list[list[float]]:
    """根据 waist_yaw_link <- TCP 位姿生成4×4齐次矩阵，四元数为XYZW。"""
    px, py, pz, x, y, z, w = pose_xyz_xyzw
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), px],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), py],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), pz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    """用4×4齐次矩阵变换三维点。"""
    homogeneous = [*point, 1.0]
    return [
        sum(float(matrix[row][column]) * homogeneous[column] for column in range(4))
        for row in range(3)
    ]


def grasp_black_single(
    station: Any,
    arm: str,
    hand_positions: dict[str, float],
    *,
    output_dir: Path | None = None,
    interactive_confirm: bool = False,
    component_selection: str = "nearest_tcp",
) -> dict[str, Any] | None:
    """识别目标，单次 MoveIt 修正 TCP 三维位置，然后闭合夹爪。

    ``interactive_confirm`` 只供独立实机测试使用：识别后、MoveIt后各暂停一次。
    比赛主流程保持缺省 ``False``，不增加人工门控。
    """
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    calibration = (
        settings.RIGHT_CAMERA if arm == "right" else settings.LEFT_CAMERA
    )
    target_dir = output_dir or settings.RUNTIME_DIR / f"{arm}_black_grasp_single"
    print(
        f"黑色隔离带单次抓取：请求臂={arm}，"
        f"标定臂={calibration['arm_name']}，"
        f"相机={calibration['station_camera_key']}，"
        f"当前夹爪位置={hand_positions}",
        flush=True,
    )
    camera_config = {
        "station_camera_key": calibration["station_camera_key"],
        "image_width": int(calibration["image_width"]),
        "image_height": int(calibration["image_height"]),
        "color_encoding": calibration["color_encoding"],
        "depth_unit": calibration["depth_unit"],
    }
    try:
        bgr, depth_mm = read_wrist_camera(station, camera_config)
    except RuntimeError as exc:
        print(f"未获取到{arm}腕相机图像，跳过本次抓取：{exc}", flush=True)
        return None
    roi = tuple(
        float(value)
        for value in settings.TASK["black_grasp"][f"{arm}_roi_normalized_xyxy"]
    )
    detection = BlackCylinderDetector(calibration["camera_matrix"]).detect(
        bgr,
        depth_mm,
        roi=roi,
        camera_to_tcp=calibration[
            f"extrinsics_cam_to_{arm}_tcp_link_matrix"
        ],
        exclusion_mask=gripper_mask(depth_mm.shape),
        component_selection=component_selection,
    )
    current_tcp = list(pose_xyz_xyzw(station.get_robot_state()["arm_pose"][arm]))
    point_tcp = _transform_point(
        calibration[f"extrinsics_cam_to_{arm}_tcp_link_matrix"],
        [float(value) for value in detection.point_camera_m],
    )
    final_offset = settings.TASK["black_grasp"][
        f"final_grasp_offset_{arm}_tcp_m"
    ]
    target_in_tcp = [
        point_tcp[index] + float(final_offset[index]) for index in range(3)
    ]
    # waist_yaw_link <- TCP <- 目标点；保留完整三维结果，不强制覆盖 waist z。
    target_xyz = _transform_point(
        _waist_from_tcp_matrix(current_tcp),
        target_in_tcp,
    )
    target_pose = [*target_xyz, *current_tcp[3:]]
    delta_waist = [target_xyz[index] - current_tcp[index] for index in range(3)]
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
    save_debug_images(target_dir, bgr, depth_mm, detection)

    print(
        "黑色隔离带三维修正："
        f"point_tcp={point_tcp} m，delta_waist={delta_waist} m，"
        f"目标位姿={target_pose}",
        flush=True,
    )
    if interactive_confirm:
        input(
            f"已完成{arm}腕隔离柱识别；确认目标点和机械臂路径安全，"
            "按 Enter 执行MoveIt三维修正："
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
        "point_camera_m": [float(value) for value in detection.point_camera_m],
        "point_tcp_m": point_tcp,
        "delta_waist_m": delta_waist,
        "grasp_tcp_pose_waist_xyz_xyzw": target_pose,
        "motion_mode": "single_moveit_full_xyz_correction",
        "motion_succeeded": False,
        "tcp_correction_executed": False,
        "hand_positions_before": dict(hand_positions),
        "hand_executed": False,
    }
    print(
        f"黑色隔离带 {arm} 臂：执行完整 x/y/z 修正的单次 MoveIt",
        flush=True,
    )
    print(f"黑色隔离带单次 MoveIt 下发：控制臂={arm}", flush=True)
    if arm == "left":
        motion_succeeded = moveit_cartesians(station, left=target_pose)
    else:
        motion_succeeded = moveit_cartesians(station, right=target_pose)
    result["motion_succeeded"] = motion_succeeded
    result["tcp_correction_executed"] = motion_succeeded
    if not motion_succeeded:
        print(
            f"{arm}臂单次 MoveIt 失败，继续闭合夹爪并执行后续任务",
            flush=True,
        )
    if interactive_confirm:
        input(
            f"{arm}臂MoveIt结果={'成功' if motion_succeeded else '失败'}；"
            "确认夹爪与隔离柱位置安全，按 Enter 闭合夹爪："
        )

    next_hand_positions = dict(hand_positions)
    next_hand_positions[arm] = float(
        settings.TASK["hands"][f"{arm}_close_position"]
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
    time.sleep(2.0)
    result["hand_positions_after"] = next_hand_positions
    result["hand_executed"] = True

    (target_dir / "target.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
