#!/usr/bin/env python3
"""纯感知测试任务3黑色隔离带，不发送导航、运动或夹爪命令。

用法：
    python test_task3_black_perception.py left
    python test_task3_black_perception.py right
    python test_task3_black_perception.py both

机器人应由操作员提前放到对应观察姿态。每按一次 Enter 读取指定腕部RGB-D、
运行正式黑色隔离带算法，并保存图像、候选点及坐标计算结果。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from black_cylinder_detector import BlackCylinderDetector, save_debug_images
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


OUTPUT_ROOT = Path(__file__).resolve().parent / "output/task3_black_perception"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("left", "right", "both"))
    return parser.parse_args()


def pose_xyz_xyzw(pose: Any) -> np.ndarray:
    values = (
        pose.get_xyz_m_xyzw_ndarray()
        if hasattr(pose, "get_xyz_m_xyzw_ndarray")
        else pose
    )
    return np.asarray(values, dtype=np.float64)


def waist_from_tcp_matrix(tcp_pose_waist: np.ndarray) -> np.ndarray:
    """构造T_waist_tcp齐次矩阵；输入Pose为米和XYZW四元数。"""
    px, py, pz, x, y, z, w = tcp_pose_waist
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = [px, py, pz]
    return matrix


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def camera_config(calibration: dict[str, Any]) -> dict[str, Any]:
    """在相机读取边界转换为xrocs腕部相机配置。"""
    return {
        "station_camera_key": calibration["station_camera_key"],
        "image_width": int(calibration["image_width"]),
        "image_height": int(calibration["image_height"]),
        "color_encoding": calibration["color_encoding"],
        "depth_unit": calibration["depth_unit"],
    }


def recognize_arm(
    station: Any,
    arm: str,
    output_dir: Path,
) -> dict[str, Any]:
    calibration = settings.LEFT_CAMERA if arm == "left" else settings.RIGHT_CAMERA
    bgr, depth_mm = read_wrist_camera(station, camera_config(calibration))
    output_dir.mkdir(parents=True, exist_ok=True)
    static_mask = gripper_mask(depth_mm.shape)
    cv2.imwrite(str(output_dir / "rgb_raw.jpg"), bgr)
    cv2.imwrite(str(output_dir / "depth_raw_mm.png"), depth_mm)
    cv2.imwrite(
        str(output_dir / "gripper_static_mask.png"),
        np.uint8(static_mask) * 255,
    )

    roi = tuple(
        float(value)
        for value in settings.TASK["black_grasp"][
            f"{arm}_roi_normalized_xyxy"
        ]
    )
    camera_to_tcp = np.asarray(
        calibration[f"extrinsics_cam_to_{arm}_tcp_link_matrix"],
        dtype=np.float64,
    )
    started = time.perf_counter()
    detection = BlackCylinderDetector(calibration["camera_matrix"]).detect(
        bgr,
        depth_mm,
        roi=roi,
        camera_to_tcp=camera_to_tcp,
        exclusion_mask=static_mask,
    )
    elapsed_s = time.perf_counter() - started
    save_debug_images(output_dir, bgr, depth_mm, detection)

    current_tcp_pose = pose_xyz_xyzw(
        station.get_robot_state()["arm_pose"][arm]
    )
    point_camera_h = np.append(detection.point_camera_m, 1.0)
    point_tcp_m = (camera_to_tcp @ point_camera_h)[:3]
    final_offset_tcp_m = np.asarray(
        settings.TASK["black_grasp"][f"final_grasp_offset_{arm}_tcp_m"],
        dtype=np.float64,
    )
    target_point_tcp_m = point_tcp_m + final_offset_tcp_m
    target_point_waist_m = (
        waist_from_tcp_matrix(current_tcp_pose)
        @ np.append(target_point_tcp_m, 1.0)
    )[:3]
    delta_waist_m = target_point_waist_m - current_tcp_pose[:3]
    target_pose_waist = current_tcp_pose.copy()
    target_pose_waist[:3] = target_point_waist_m

    candidates = [
        {
            "pixel_uv": candidate.pixel_uv,
            "point_camera_m": candidate.point_camera_m,
            "point_tcp_m": candidate.point_tcp_m,
            "distance_to_tcp_m": candidate.distance_to_tcp_m,
            "local_width_px": candidate.local_width_px,
            "selected": candidate.pixel_uv == detection.pixel_uv,
        }
        for candidate in detection.grasp_candidates
    ]
    result = {
        "success": True,
        "arm": arm,
        "camera_key": calibration["station_camera_key"],
        "roi_normalized_xyxy": roi,
        "elapsed_s": elapsed_s,
        "pixel_uv": detection.pixel_uv,
        "confidence": detection.confidence,
        "depth_m": detection.depth_m,
        "angle_deg": detection.angle_deg,
        "major_axis_px": detection.major_axis_px,
        "minor_axis_px": detection.minor_axis_px,
        "selection_strategy": detection.selection_strategy,
        "point_camera_m": detection.point_camera_m,
        "point_tcp_m": point_tcp_m,
        "final_grasp_offset_tcp_m": final_offset_tcp_m,
        "target_point_tcp_m": target_point_tcp_m,
        "current_tcp_pose_waist_xyz_xyzw": current_tcp_pose,
        "delta_waist_m": delta_waist_m,
        "target_pose_waist_xyz_xyzw": target_pose_waist,
        "grasp_candidates": candidates,
    }
    (output_dir / "result.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"{arm}腕识别：像素={detection.pixel_uv}，"
        f"置信度={detection.confidence:.3f}，深度={detection.depth_m:.3f} m，"
        f"主轴角={detection.angle_deg:.1f}°，"
        f"长轴={detection.major_axis_px:.1f} px，"
        f"短轴={detection.minor_axis_px:.1f} px，"
        f"候选点={len(candidates)}，耗时={elapsed_s:.3f} s",
        flush=True,
    )
    print(
        f"{arm}腕三维结果：camera={detection.point_camera_m.tolist()} m，"
        f"TCP={point_tcp_m.tolist()} m，"
        f"TCP最终偏置={final_offset_tcp_m.tolist()} m，"
        f"waist位移={delta_waist_m.tolist()} m，"
        f"目标Pose={target_pose_waist.tolist()}",
        flush=True,
    )
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  候选{index}：像素={candidate['pixel_uv']}，"
            f"TCP点={candidate['point_tcp_m'].tolist()} m，"
            f"TCP距离={candidate['distance_to_tcp_m']:.3f} m，"
            f"局部宽度={candidate['local_width_px']:.1f} px，"
            f"selected={candidate['selected']}",
            flush=True,
        )
    print(f"{arm}腕结果已保存：{output_dir}", flush=True)
    return jsonable(result)


def main() -> None:
    args = parse_args()
    arms = ("left", "right") if args.arm == "both" else (args.arm,)
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        round_index = 1
        while True:
            input(
                f"第{round_index}次任务3黑色隔离带感知（{args.arm}）："
                "确认机器人保持静止，按 Enter 采集并计算；Ctrl+C结束："
            )
            round_dir = OUTPUT_ROOT / datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            round_results = []
            for arm in arms:
                arm_dir = round_dir / arm
                try:
                    result = recognize_arm(station, arm, arm_dir)
                except RuntimeError as error:
                    arm_dir.mkdir(parents=True, exist_ok=True)
                    result = {
                        "success": False,
                        "arm": arm,
                        "error": str(error),
                    }
                    (arm_dir / "failure.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"{arm}腕识别失败：{error}；结果已保存：{arm_dir}",
                        flush=True,
                    )
                round_results.append(result)
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "summary.json").write_text(
                json.dumps(round_results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"第{round_index}次汇总已保存：{round_dir}", flush=True)
            round_index += 1
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
