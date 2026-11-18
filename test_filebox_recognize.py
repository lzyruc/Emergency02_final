#!/usr/bin/env python3
"""用双腕相机和原 BlueDepthTargetDetector 测试文件筐二维抓取点。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

import settings
from arm_motion import moveit_joints
from blue_depth_target import BlueDepthTargetDetector, save_debug_images
from camera_io import read_wrist_camera
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TASK_DIR = Path(__file__).resolve().parent
INITIAL_POSE_PATH = TASK_DIR / "poses_emergency/应急_初始姿态.json"
FILEBOX_PREGRASP_PATHS = tuple(
    TASK_DIR / f"poses_filebox/应急_文件框{index}.json"
    for index in range(3)
)
OUTPUT_DIR = TASK_DIR / "output"
CAMERAS = (
    ("left", "左腕", settings.LEFT_CAMERA),
    ("right", "右腕", settings.RIGHT_CAMERA),
)


def execute_pose(station: Any, pose: dict[str, Any]) -> bool:
    """串行执行双臂及头腰腿姿态；忽略 JSON 中的夹爪字段。"""
    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        return False
    station.execute_action(
        {
            "head": {"position": pose["head"]},
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        }
    )
    time.sleep(1.0)
    return True


def recognize_2d_and_save(
    station: Any,
    camera: str,
    camera_config: dict[str, Any],
    pose_name: str,
    capture_dir: Path,
) -> dict[str, Any]:
    """保存腕部 RGB-D，并返回蓝色文件筐二维像素抓取点。"""
    color, depth_mm = read_wrist_camera(station, camera_config)
    capture_dir.mkdir(parents=True)
    cv2.imwrite(str(capture_dir / "rgb.png"), color)
    cv2.imwrite(str(capture_dir / "depth_mm.png"), depth_mm)
    depth_color = cv2.applyColorMap(
        np.clip(depth_mm.astype(np.float32) * 255.0 / 1500.0, 0, 255)
        .astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    depth_color[depth_mm == 0] = 0
    cv2.imwrite(str(capture_dir / "depth_color.png"), depth_color)

    detector = BlueDepthTargetDetector(
        np.asarray(camera_config["camera_matrix"], dtype=np.float64)
    )
    try:
        detection = detector.detect(color, depth_mm)
    except RuntimeError as exc:
        result = {
            "pose": pose_name,
            "camera": camera,
            "detected": False,
            "error": str(exc),
            "pixel_uv": None,
            "confidence": None,
            "depth_m": None,
        }
    else:
        save_debug_images(capture_dir, color, detection)
        result = {
            "pose": pose_name,
            "camera": camera,
            "detected": True,
            "error": None,
            "pixel_uv": [
                int(detection.pixel_uv[0]),
                int(detection.pixel_uv[1]),
            ],
            "confidence": float(detection.confidence),
            "depth_m": float(detection.depth_m),
        }

    (capture_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    """依次测试文件筐姿态 0、1、2 下的左右腕二维识别。"""
    initial_pose = json.loads(
        INITIAL_POSE_PATH.read_text(encoding="utf-8")
    )
    poses = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in FILEBOX_PREGRASP_PATHS
    ]
    run_dir = OUTPUT_DIR / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_filebox_blue_2d"
    )
    run_dir.mkdir(parents=True)

    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    results = []
    step = 1
    try:
        time.sleep(1.0)
        input(
            f"步骤{step}：确认机器人周围安全，按 Enter 切换应急初始姿态；"
            "Ctrl+C 终止："
        )
        if not execute_pose(station, initial_pose):
            raise RuntimeError("应急初始姿态执行失败")
        step += 1

        for pose_index, (pose_path, pose) in enumerate(
            zip(FILEBOX_PREGRASP_PATHS, poses, strict=True)
        ):
            input(
                f"步骤{step}：确认机器人周围安全，按 Enter 切换 "
                f"{pose_path.name}；Ctrl+C 终止："
            )
            if not execute_pose(station, pose):
                raise RuntimeError(f"姿态执行失败：{pose_path.name}")
            step += 1

            for camera, camera_name, camera_config in CAMERAS:
                input(
                    f"步骤{step}：确认文件筐位于{camera_name}视野，"
                    "按 Enter 采集并执行蓝色二维抓取点识别；Ctrl+C 终止："
                )
                capture_dir = (
                    run_dir
                    / f"{pose_index}_{pose_path.stem}_{camera}"
                )
                result = recognize_2d_and_save(
                    station,
                    camera,
                    camera_config,
                    pose_path.name,
                    capture_dir,
                )
                result["output_dir"] = capture_dir.name
                results.append(result)
                if result["detected"]:
                    u, v = result["pixel_uv"]
                    print(
                        f"[{pose_path.stem} / {camera_name}] "
                        f"抓取点=(u={u}, v={v})，"
                        f"置信度={result['confidence']:.3f}，"
                        f"深度={result['depth_m']:.3f} m",
                        flush=True,
                    )
                else:
                    print(
                        f"[{pose_path.stem} / {camera_name}] 未识别到抓取点："
                        f"{result['error']}",
                        flush=True,
                    )
                step += 1

        summary = {
            "mode": "filebox_blue_2d",
            "uses_extrinsics": False,
            "pose_sequence": [path.name for path in FILEBOX_PREGRASP_PATHS],
            "results": results,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        success_count = sum(result["detected"] for result in results)
        print(
            f"测试完成：{success_count}/{len(results)} 个二维抓取点识别成功；"
            f"输出目录：{run_dir}",
            flush=True,
        )
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
