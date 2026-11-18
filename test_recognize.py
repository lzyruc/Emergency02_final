#!/usr/bin/env python3
"""按物体测试 Task1 的头部识别、姿态切换和双腕识别，不执行导航或抓取。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

import settings
from agent.perception.deprojector import Deprojector
from camera_io import read_head_camera, read_wrist_camera
from task1 import (
    BIN_PREGRASP_PATHS,
    BOX_HOLD_HORIZONTAL_PATH,
    BOX_PREGRASP_PATH,
    CALIBRATION_PATH,
    CHAIR_PREGRASP_PATHS,
    FILEBOX_PREGRASP_PATHS,
    TARGET_NAMES,
    _execute_object_pose,
)
from task2_phase1 import Phase1Model
from task2_phase1.visualize import draw_annotations_on_rgb
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TASK_DIR = Path(__file__).resolve().parent
INITIAL_POSE_PATH = TASK_DIR / "poses_emergency/应急_初始姿态.json"
OUTPUT_DIR = TASK_DIR / "output"
TASK_SPECS = {
    "chair": {
        "target_class": "chair",
        "poses": CHAIR_PREGRASP_PATHS,
    },
    "bin": {
        "target_class": "trash_can",
        "poses": BIN_PREGRASP_PATHS,
    },
    "filebox": {
        "target_class": "basket",
        "poses": FILEBOX_PREGRASP_PATHS,
    },
    "box": {
        "target_class": "box",
        "poses": (BOX_PREGRASP_PATH, BOX_HOLD_HORIZONTAL_PATH),
    },
}
WRIST_CAMERAS = (
    ("left", "左腕", settings.LEFT_CAMERA),
    ("right", "右腕", settings.RIGHT_CAMERA),
)


def _camera_intrinsics(camera: str) -> dict[str, float | int]:
    calibration = json.loads(
        CALIBRATION_PATH.read_text(encoding="utf-8")
    )["cameras"][camera]
    matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    width, height = calibration["image_size"]
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def recognize_and_save(
    station: Any,
    camera: str,
    camera_config: dict[str, Any] | None,
    target_class: str,
    capture_dir: Path,
) -> dict[str, Any]:
    """采集一路 RGB-D，识别指定类别并保存图像及 BASE_FOOTPRINT 结果。"""
    if camera == "head":
        color, depth_mm = read_head_camera(station)
        if color is None or depth_mm is None:
            raise RuntimeError("未获取到头部 RGB-D 帧")
    else:
        color, depth_mm = read_wrist_camera(station, camera_config)

    capture_dir.mkdir(parents=True)
    cv2.imwrite(str(capture_dir / "rgb.png"), color)
    cv2.imwrite(str(capture_dir / "depth_mm.png"), depth_mm)
    depth_color = cv2.applyColorMap(
        np.clip(depth_mm.astype(np.float32) * 255.0 / 5000.0, 0, 255)
        .astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    depth_color[depth_mm == 0] = 0
    cv2.imwrite(str(capture_dir / "depth_color.png"), depth_color)

    intrinsics = _camera_intrinsics(camera)
    model = Phase1Model(intrinsics)
    selected = model.detect(
        {"rgb": color, "depth": depth_mm},
        target_class=target_class,
    )
    frame = model.last_frame_result
    annotated = draw_annotations_on_rgb(color, frame, intrinsics)
    cv2.imwrite(str(capture_dir / "annotated.png"), annotated)

    objects = [obj for obj in frame.objects if obj.cls == target_class]
    targets = []
    if objects:
        samples = []
        for obj in objects:
            x_camera_m, y_camera_m, depth_m = (
                float(value) for value in obj.position_cam
            )
            samples.append(
                [
                    x_camera_m * intrinsics["fx"] / depth_m
                    + intrinsics["cx"],
                    y_camera_m * intrinsics["fy"] / depth_m
                    + intrinsics["cy"],
                    depth_m,
                ]
            )
        deprojector = Deprojector(station, CALIBRATION_PATH, camera)
        poses_waist = deprojector.deproject_many(
            samples,
            face_target=False,
        )
        for obj, pose_waist in zip(objects, poses_waist, strict=True):
            point_base = deprojector.transform_to_ground(pose_waist[:3])
            targets.append(
                {
                    "type": obj.cls,
                    "name": TARGET_NAMES[obj.cls],
                    "confidence": float(obj.confidence),
                    "point_camera_m": [
                        float(value) for value in obj.position_cam
                    ],
                    "point_base_m": point_base.tolist(),
                    "distance_3d_m": float(np.linalg.norm(point_base)),
                    "bearing_deg": math.degrees(
                        math.atan2(
                            float(point_base[1]),
                            float(point_base[0]),
                        )
                    ),
                }
            )

    result = {
        "stage": capture_dir.name,
        "camera": camera,
        "target_class": target_class,
        "detected": bool(targets),
        "selected_pixel": selected,
        "n_raw_clusters": int(frame.n_raw_clusters),
        "n_kept": int(frame.n_kept),
        "targets": targets,
        "files": {
            "rgb": "rgb.png",
            "depth_mm": "depth_mm.png",
            "depth_color": "depth_color.png",
            "annotated": "annotated.png",
        },
    }
    (capture_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _print_result(camera_name: str, result: dict[str, Any]) -> None:
    if not result["targets"]:
        print(
            f"[{camera_name}] 未识别到 {TARGET_NAMES[result['target_class']]}",
            flush=True,
        )
        return
    for target in result["targets"]:
        x_m, y_m, z_m = target["point_base_m"]
        print(
            f"[{camera_name}] {target['name']}，"
            f"置信度={target['confidence']:.3f}，"
            f"base_xyz=[{x_m:.3f}, {y_m:.3f}, {z_m:.3f}] m，"
            f"距离={target['distance_3d_m']:.3f} m，"
            f"方位角={target['bearing_deg']:.1f}°",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测试单个应急物品的头部/双腕识别和预操作姿态",
    )
    parser.add_argument(
        "object",
        choices=tuple(TASK_SPECS),
        help="chair、bin、filebox 或 box",
    )
    args = parser.parse_args()
    spec = TASK_SPECS[args.object]
    target_class = spec["target_class"]
    initial_pose = json.loads(
        INITIAL_POSE_PATH.read_text(encoding="utf-8")
    )
    run_dir = OUTPUT_DIR / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{args.object}"
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
    all_results = []
    step = 1
    try:
        time.sleep(1.0)
        input(
            f"步骤{step}：确认机器人周围安全，按 Enter 切换应急初始姿态；"
            "Ctrl+C 终止："
        )
        if not _execute_object_pose(station, initial_pose):
            raise RuntimeError("应急初始姿态执行失败")
        step += 1

        input(
            f"步骤{step}：把{TARGET_NAMES[target_class]}放入头部视野，"
            "按 Enter 采集并识别；Ctrl+C 终止："
        )
        result = recognize_and_save(
            station,
            "head",
            None,
            target_class,
            run_dir / "00_initial_head",
        )
        all_results.append(result)
        _print_result("头部", result)
        step += 1

        for pose_index, pose_path in enumerate(spec["poses"]):
            input(
                f"步骤{step}：确认机器人周围安全，按 Enter 切换姿态 "
                f"{pose_path.name}；Ctrl+C 终止："
            )
            pose = json.loads(pose_path.read_text(encoding="utf-8"))
            if not _execute_object_pose(station, pose):
                raise RuntimeError(f"姿态执行失败：{pose_path.name}")
            step += 1

            for camera, camera_name, camera_config in WRIST_CAMERAS:
                input(
                    f"步骤{step}：确认{TARGET_NAMES[target_class]}位于"
                    f"{camera_name}视野，按 Enter 采集并识别；Ctrl+C 终止："
                )
                result = recognize_and_save(
                    station,
                    camera,
                    camera_config,
                    target_class,
                    run_dir
                    / f"{pose_index + 1:02d}_{pose_path.stem}_{camera}",
                )
                all_results.append(result)
                _print_result(camera_name, result)
                step += 1

        summary = {
            "object_argument": args.object,
            "target_class": target_class,
            "target_name": TARGET_NAMES[target_class],
            "initial_pose": str(INITIAL_POSE_PATH.relative_to(TASK_DIR)),
            "pose_sequence": [
                str(path.relative_to(TASK_DIR)) for path in spec["poses"]
            ],
            "results": all_results,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        detected_count = sum(result["detected"] for result in all_results)
        print(
            f"测试完成：{detected_count}/{len(all_results)} 次识别成功；"
            f"输出目录：{run_dir}",
            flush=True,
        )
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
