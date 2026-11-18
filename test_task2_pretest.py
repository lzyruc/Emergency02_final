#!/usr/bin/env python3
"""左腕 RGB-D 识别窗把手，分两次修正左臂 TCP；开窗测试追加右移3 cm、前移1 cm、TCP绕夹爪局部z轴顺时针90°旋转、左移2 cm与上移3 cm，最后恢复准备姿态。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

import settings
from agent.perception import deprojector
from agent.perception.window_handle_detection import detect_handle_center
from arm_motion import moveit_cartesians, moveit_joints
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


CALIBRATION_PATH = (
    Path(deprojector.__file__).resolve().parent
    / "config/camera_calibration.json"
)
# 改为 False 即恢复只检查最近深度簇的原算法。
SEARCH_FARTHER_DEPTH_CLUSTERS = True


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


def _save_depth_color(output_path: Path, depth_mm: np.ndarray) -> None:
    """按当前帧有效深度的2%到98%分位生成便于观察的伪彩色图。"""
    valid = depth_mm > 0
    color = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth_mm[valid], [2.0, 98.0])
        if high > low:
            normalized = np.clip((depth_mm.astype(np.float32) - low) / (high - low), 0.0, 1.0)
            colored = cv2.applyColorMap(
                np.uint8((1.0 - normalized) * 255.0),
                cv2.COLORMAP_TURBO,
            )
            color[valid] = colored[valid]
    cv2.imwrite(str(output_path), color)


def recognize_window1_handle(
    robot_station: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """保存左腕RGB-D调试数据，返回把手像素、深度和相机光学坐标。"""
    bgr, depth_mm = read_wrist_camera(robot_station, settings.LEFT_CAMERA)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rgb_raw.jpg"), bgr)
    cv2.imwrite(str(output_dir / "depth_raw_mm.png"), depth_mm)
    _save_depth_color(output_dir / "depth_color.jpg", depth_mm)

    static_gripper_mask = gripper_mask(depth_mm.shape)
    detection_depth_mm = depth_mm.copy()
    detection_depth_mm[static_gripper_mask] = 0
    cv2.imwrite(
        str(output_dir / "gripper_static_mask.png"),
        np.uint8(static_gripper_mask) * 255,
    )
    cv2.imwrite(
        str(output_dir / "depth_gripper_masked_mm.png"),
        detection_depth_mm,
    )
    _save_depth_color(
        output_dir / "depth_gripper_masked_color.jpg",
        detection_depth_mm,
    )
    rgb_gripper_mask = bgr.copy()
    rgb_gripper_mask[static_gripper_mask] = (0, 0, 255)
    cv2.imwrite(
        str(output_dir / "rgb_gripper_static_mask.jpg"),
        rgb_gripper_mask,
    )

    height, width = depth_mm.shape
    roi_x0, roi_x1 = 0, width
    rgb_roi = bgr.copy()
    cv2.rectangle(rgb_roi, (roi_x0, 0), (roi_x1 - 1, height - 1), (0, 255, 255), 2)
    cv2.imwrite(str(output_dir / "rgb_detection_area.jpg"), rgb_roi)

    camera_matrix = np.asarray(
        settings.LEFT_CAMERA["camera_matrix"],
        dtype=np.float64,
    )
    depth_m = np.asarray(detection_depth_mm, dtype=np.float64) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    detection_valid = valid[:, roi_x0:roi_x1]
    detection_depths = depth_m[:, roi_x0:roi_x1][detection_valid]
    depth_percentiles = (
        np.percentile(detection_depths, [1, 5, 10, 20, 50, 80, 95, 99]).tolist()
        if detection_depths.size
        else []
    )
    seed_count = min(50, detection_depths.size)
    seed_depth_m = (
        float(np.median(np.partition(detection_depths, seed_count - 1)[:seed_count]))
        if seed_count
        else None
    )
    diagnostics = {
        "output_dir": str(output_dir),
        "camera_key": settings.LEFT_CAMERA["station_camera_key"],
        "rgb_shape": list(bgr.shape),
        "rgb_dtype": str(bgr.dtype),
        "depth_shape": list(depth_mm.shape),
        "depth_dtype": str(depth_mm.dtype),
        "gripper_mask_rule": "config/gripper_mask.png static open-gripper mask",
        "gripper_mask_pixels": int(np.count_nonzero(static_gripper_mask)),
        "gripper_mask_ratio": float(
            np.count_nonzero(static_gripper_mask) / static_gripper_mask.size
        ),
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "valid_depth_ratio": float(np.count_nonzero(valid) / valid.size),
        "detection_area_xyxy": [roi_x0, 0, roi_x1, height],
        "detection_valid_depth_pixels": int(detection_depths.size),
        "detection_valid_depth_ratio": float(
            detection_depths.size / detection_valid.size
        ),
        "detection_depth_percentiles_m": {
            str(percentile): float(value)
            for percentile, value in zip(
                [1, 5, 10, 20, 50, 80, 95, 99],
                depth_percentiles,
            )
        },
        "nearest_50_seed_depth_m": seed_depth_m,
        "algorithm_depth_band_m": 0.05,
        "search_farther_depth_clusters": SEARCH_FARTHER_DEPTH_CLUSTERS,
        "accepted_component_area_m2": [0.002, 0.0055],
        "accepted_main_length_m": [0.08, 0.18],
        "camera_intrinsics": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("感知输入诊断：", flush=True)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)

    detection = detect_handle_center(
        depth_m,
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        search_farther_depth_clusters=SEARCH_FARTHER_DEPTH_CLUSTERS,
    )
    u, v, depth_m = detection.uv_depth_m
    point_camera_m = [
        float((u - camera_matrix[0, 2]) * depth_m / camera_matrix[0, 0]),
        float((v - camera_matrix[1, 2]) * depth_m / camera_matrix[1, 1]),
        float(depth_m),
    ]
    ys, xs = np.nonzero(detection.mask)
    bbox_xyxy = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    ]
    cv2.imwrite(str(output_dir / "handle_mask.png"), detection.mask)
    rgb_target = bgr.copy()
    contours, _ = cv2.findContours(
        detection.mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rgb_target, contours, -1, (0, 255, 0), 2)
    cv2.circle(rgb_target, (int(round(u)), int(round(v))), 9, (0, 0, 255), -1)
    cv2.rectangle(
        rgb_target,
        (bbox_xyxy[0], bbox_xyxy[1]),
        (bbox_xyxy[2] - 1, bbox_xyxy[3] - 1),
        (255, 0, 0),
        2,
    )
    cv2.imwrite(str(output_dir / "rgb_target.jpg"), rgb_target)

    result = {
        **diagnostics,
        "pixel_uv": [float(u), float(v)],
        "depth_m": float(depth_m),
        "point_camera_optical_xyz_m": point_camera_m,
        "camera_axes": "+x right, +y down, +z forward",
        "component_bbox_xyxy": bbox_xyxy,
        "component_mask_pixels": int(np.count_nonzero(detection.mask)),
        "component_area_m2": float(detection.area_m2),
        "component_main_length_m": float(detection.main_length_m),
        "selected_seed_depth_m": float(detection.seed_depth_m),
        "depth_cluster_attempt": int(detection.depth_cluster_attempt),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """准备、识别、分两次移动左 TCP，最后恢复准备姿态。"""
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
    try:
        input(
            "步骤1：确认双臂、头、腰和腿运动路径安全，按 Enter 切换到"
            "应急_窗户2_准备姿态；Ctrl+C 终止："
        )
        _apply_window2_prepare(station)

        input("步骤2：准备姿态已到达，按 Enter 读取左腕 RGB-D 并识别把手：")
        result = recognize_window1_handle(station, output_dir)
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
        # 常量即"悬停距离"：移动后把手位于新 TCP 前方该距离处（z 负向为前）。
        # 0.19 = 原0.20 - 0.01：要 TCP 朝把手方向总共多走 1 cm（保证能伸入到位）。
        second_delta_tcp_m = np.array(
            [0.0, 0.0, point_left_tcp_m[2] + 0.19],
            dtype=np.float64,
        )
        second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
        second_target_pose_waist = current_pose_waist.copy()
        second_target_pose_waist[:3] = (
            T_waist_tcp[:3, 3] + second_delta_waist_m
        )
        print(
            "第二次 z 计算："
            f"识别 z {point_left_tcp_m[2] * 100:+.2f} cm + 19.00 cm "
            f"= {second_delta_tcp_m[2] * 100:+.2f} cm",
            flush=True,
        )
        _print_offset("第二次", second_delta_tcp_m, second_delta_waist_m)
        result.update(
            {
                "point_left_tcp_xyz_m": point_left_tcp_m.tolist(),
                "first_delta_left_tcp_xyz_m": first_delta_tcp_m.tolist(),
                "second_delta_left_tcp_xyz_m": second_delta_tcp_m.tolist(),
                "second_z_rule": "recognized_left_tcp_z_m + 0.19 m",
            }
        )
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        input(
            "步骤4：第二次只沿 TCP 局部 z 移动，方向保持不变；"
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
        right_target_pose_waist[1] -= 0.08
        input(
            "步骤6：TCP方向保持不变，将沿 waist_yaw_link -Y（视觉右侧）"
            "移动8.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤6位移：waist_yaw_link x=+0.0 cm，y=-8.0 cm，z=+0.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=right_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=3,
        ):
            raise RuntimeError("左臂向视觉右侧移动8 cm失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        upward_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        upward_target_pose_waist[2] += 0.04
        input(
            "步骤7：TCP方向保持不变，将沿 waist_yaw_link +Z（向上）"
            "移动4.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤7位移：waist_yaw_link x=+0.0 cm，y=+0.0 cm，z=+4.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=upward_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=4,
        ):
            raise RuntimeError("左臂向上移动4 cm失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        right3_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        right3_target_pose_waist[1] -= 0.03
        input(
            "步骤8：TCP方向保持不变，将沿 waist_yaw_link -Y（视觉右侧）"
            "移动3.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤8位移：waist_yaw_link x=+0.0 cm，y=-3.0 cm，z=+0.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=right3_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=5,
        ):
            raise RuntimeError("左臂向视觉右侧移动3 cm失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        forward_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        forward_target_pose_waist[0] += 0.01
        input(
            "步骤9：TCP方向保持不变，将沿 waist_yaw_link +X（向前）"
            "移动1.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤9位移：waist_yaw_link x=+1.0 cm，y=+0.0 cm，z=+0.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=forward_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=6,
        ):
            raise RuntimeError("左臂向前移动1 cm失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        current_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        # 绕TCP局部z轴顺时针旋转90°（相对当前夹爪位姿，位置不变）。
        # q_delta = (cos45°, 0, 0, sin45°)，q_new = q ⊗ q_delta；
        # 若实际旋转方向相反，把 sin_h 取负即可（一行改动）。
        qx, qy, qz, qw = current_pose_waist[3:7]
        cos_h = math.cos(math.pi / 4.0)
        sin_h = math.sin(math.pi / 4.0)
        qw_new = qw * cos_h - qz * sin_h
        qx_new = qx * cos_h + qy * sin_h
        qy_new = qy * cos_h - qx * sin_h
        qz_new = qz * cos_h + qw * sin_h
        norm = math.sqrt(qw_new ** 2 + qx_new ** 2 + qy_new ** 2 + qz_new ** 2)
        rotation_target_pose_waist = current_pose_waist.copy()
        rotation_target_pose_waist[3:7] = (
            qx_new / norm, qy_new / norm, qz_new / norm, qw_new / norm,
        )
        input(
            f"步骤10：TCP位置不变，绕当前夹爪局部z轴顺时针旋转90°"
            f"（四元数 {current_pose_waist[3]:.4f}/{current_pose_waist[4]:.4f}/"
            f"{current_pose_waist[5]:.4f}/{current_pose_waist[6]:.4f} → "
            f"{qx_new / norm:.4f}/{qy_new / norm:.4f}/"
            f"{qz_new / norm:.4f}/{qw_new / norm:.4f}）；"
            "确认路径安全后按 Enter 执行；方向不对可 Ctrl+C 中止："
        )
        if not moveit_cartesians(
            station,
            left=rotation_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=7,
        ):
            raise RuntimeError("左臂TCP绕局部z轴顺时针旋转90°失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        left_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        left_target_pose_waist[1] += 0.02
        input(
            "步骤11：TCP方向保持不变，将沿 waist_yaw_link +Y（视觉左侧）"
            "移动2.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤11位移：waist_yaw_link x=+0.0 cm，y=+2.0 cm，z=+0.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=left_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=8,
        ):
            raise RuntimeError("左臂向视觉左侧移动2 cm失败")

        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        up3_target_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        up3_target_pose_waist[2] += 0.03
        input(
            "步骤12：TCP方向保持不变，将沿 waist_yaw_link +Z（向上）"
            "移动3.0 cm；确认路径安全后按 Enter 执行："
        )
        print(
            "步骤12位移：waist_yaw_link x=+0.0 cm，y=+0.0 cm，z=+3.0 cm",
            flush=True,
        )
        if not moveit_cartesians(
            station,
            left=up3_target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=9,
        ):
            raise RuntimeError("左臂向上移动3 cm失败")

        input(
            "步骤13：左移、上移与TCP旋转均已完成，确认把手位于"
            "夹爪内后按 Enter 闭合左夹爪："
        )
        print("闭合左夹爪：left=1.000，right=0.000", flush=True)
        station.execute_action(
            {"hand": {"position": {"left": [1.0], "right": [0.0]}}}
        )

        input(
            "步骤14：检查夹爪闭合状态后按 Enter 松开左夹爪："
        )
        print("松开左夹爪：left=0.000，right=0.000", flush=True)
        station.execute_action(
            {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
        )

        input(
            "步骤15：左夹爪已松开，确认恢复路径安全后按 Enter 恢复"
            "应急_窗户2_准备姿态："
        )
        _apply_window2_prepare(station)
        print("任务2识别移动测试完成，已恢复窗户2准备姿态。", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
