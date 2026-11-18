"""将 phase1_wrist_total 的腕部像素结果转换为腰坐标系抓取候选。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from camera_io import read_wrist_camera
from wrist_phase1 import Phase1Model
from wrist_phase1.visualize import draw_result


TARGET_NAMES = {
    "chair": "办公转椅",
    "filebox": "文件筐",
    "bin": "垃圾桶",
    "box": "纸箱（3 kg）",
}
WRIST_CLASS_BY_TARGET = {
    "chair": "chair",
    "filebox": "basket",
    "bin": "trash_can",
    "box": "box",
}
WRIST_DEPTH_SCALE_M = 0.001
WRIST_OUTPUT_DIR = Path(__file__).resolve().parent / "output/task1_wrist"


def _save_wrist_images(
    target_type: str,
    arm: str,
    color: np.ndarray,
    depth_mm: np.ndarray,
    detailed: dict[str, Any],
) -> Path:
    """保存 BGR、16位毫米深度和检测标注图，返回文件名前缀。"""
    WRIST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = WRIST_OUTPUT_DIR / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{target_type}_{arm}"
    )
    outputs = (
        (prefix.with_name(f"{prefix.name}_rgb.jpg"), color),
        (prefix.with_name(f"{prefix.name}_depth.png"), depth_mm),
        (
            prefix.with_name(f"{prefix.name}_annotated.jpg"),
            draw_result(color, detailed),
        ),
    )
    for path, image in outputs:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"腕部识别图像保存失败：{path}")
    print(f"{arm}腕识别图像已保存：{prefix.name}_*", flush=True)
    return prefix


def wrist_grasp_candidates(
    station: Any,
    target_type: str,
    deprojectors: dict[str, Any],
    models: dict[str, Phase1Model],
) -> list[dict[str, Any]]:
    """处理左右腕毫米制 RGB-D，返回 ``waist_yaw_link`` 下的候选Pose。"""
    expected_class = WRIST_CLASS_BY_TARGET[target_type]
    candidates = []
    for arm, camera_config in (
        ("left", settings.LEFT_CAMERA),
        ("right", settings.RIGHT_CAMERA),
    ):
        color, depth_mm = read_wrist_camera(station, camera_config)
        result = models[arm].detect_for_class(
            {"rgb": color, "depth": depth_mm},
            expected_class,
        )
        _save_wrist_images(
            target_type,
            arm,
            color,
            depth_mm,
            models[arm].last_frame_result,
        )
        if result["class"] != expected_class:
            print(
                f"{arm}腕识别类别={result['class']}，"
                f"期望={expected_class}，忽略该结果",
                flush=True,
            )
            continue
        if result["x"] is None or result["y"] is None:
            print(
                f"{arm}腕识别到{TARGET_NAMES[target_type]}，但没有可靠抓取点",
                flush=True,
            )
            continue

        u, v = int(result["x"]), int(result["y"])
        depth_window = depth_mm[
            max(0, v - 2) : min(depth_mm.shape[0], v + 3),
            max(0, u - 2) : min(depth_mm.shape[1], u + 3),
        ]
        valid_depth = depth_window[
            (depth_window > 0)
            & (depth_window != 65535)
            & np.isfinite(depth_window)
        ]
        detailed = models[arm].last_frame_result
        primary = detailed["primary"]
        if valid_depth.size:
            depth_m = float(np.median(valid_depth)) * WRIST_DEPTH_SCALE_M
            depth_source = "抓取点5x5中值"
        elif primary is not None and primary.get("depth_m") is not None:
            depth_m = float(primary["depth_m"])
            depth_source = "目标区域深度"
        else:
            print(f"{arm}腕抓取点附近没有有效深度", flush=True)
            continue

        target_pose = deprojectors[arm].deproject(
            [float(u), float(v), depth_m],
            face_target=False,
        )
        current_pose = station.get_robot_state()["arm_pose"][arm]
        current_pose = (
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else np.asarray(current_pose, dtype=np.float64)
        )
        candidates.append(
            {
                "arm": arm,
                "class": result["class"],
                "confidence": float(primary["confidence"]),
                "selection_score": float(primary["selection_score"]),
                "pixel_uv": [u, v],
                "point_source": detailed["result"].get("point_source"),
                "depth_m": depth_m,
                "depth_source": depth_source,
                "target_pose_waist_xyz_xyzw": target_pose,
                "tcp_distance_m": float(
                    np.linalg.norm(target_pose[:3] - current_pose[:3])
                ),
            }
        )
    return candidates
