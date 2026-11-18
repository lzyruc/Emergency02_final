"""腕部 RGB-D 窗把手识别与调试数据保存。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from agent.perception.window_handle_detection import detect_handle_center
from agent.perception.window_handle_detection_v2 import detect_handle_center_v2
from agent.perception.window_handle_detection_v3 import (
    detect_handle_depth_in_bbox_v3,
)
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask
from yolo.window_handle_yolo import WindowHandleYolo


SEARCH_FARTHER_DEPTH_CLUSTERS = True


def _save_depth_color(output_path: Path, depth_mm: np.ndarray) -> None:
    """按有效深度的2%到98%分位保存伪彩色深度图。"""
    valid = depth_mm > 0
    color = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth_mm[valid], [2.0, 98.0])
        if high > low:
            normalized = np.clip(
                (depth_mm.astype(np.float32) - low) / (high - low),
                0.0,
                1.0,
            )
            colored = cv2.applyColorMap(
                np.uint8((1.0 - normalized) * 255.0),
                cv2.COLORMAP_TURBO,
            )
            color[valid] = colored[valid]
    cv2.imwrite(str(output_path), color)


def recognize_window_handle(
    station: Any,
    output_dir: Path,
    *,
    algorithm: str = "v2",
    window_handle_detector: WindowHandleYolo | None = None,
) -> dict[str, Any]:
    """读取左腕RGB-D，保存诊断数据并返回把手像素和深度。

    v1/v2保持原深度算法；v3由已加载的YOLO提供语义框，再在框内计算
    稳健深度。三种算法使用相同的返回字段供后续反投影和TCP检查。

    返回的 ``pixel_uv`` 为原始左腕图像像素坐标，``depth_m`` 单位为米；
    ``point_camera_optical_xyz_m`` 位于左腕相机光学坐标系，轴向为
    +x向右、+y向下、+z向前。
    """
    bgr, depth_mm = read_wrist_camera(station, settings.LEFT_CAMERA)
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
    rgb_detection_area = bgr.copy()
    cv2.rectangle(
        rgb_detection_area,
        (0, 0),
        (width - 1, height - 1),
        (0, 255, 255),
        2,
    )
    cv2.imwrite(
        str(output_dir / "rgb_detection_area.jpg"),
        rgb_detection_area,
    )

    camera_matrix = np.asarray(
        settings.LEFT_CAMERA["camera_matrix"],
        dtype=np.float64,
    )
    depth_m = np.asarray(detection_depth_mm, dtype=np.float64) / 1000.0
    valid_depths = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    depth_percentiles = (
        np.percentile(
            valid_depths,
            [1, 5, 10, 20, 50, 80, 95, 99],
        ).tolist()
        if valid_depths.size
        else []
    )
    seed_count = min(50, valid_depths.size)
    seed_depth_m = (
        float(
            np.median(
                np.partition(valid_depths, seed_count - 1)[:seed_count]
            )
        )
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
        "valid_depth_pixels": int(valid_depths.size),
        "valid_depth_ratio": float(valid_depths.size / depth_m.size),
        "detection_area_xyxy": [0, 0, width, height],
        "detection_valid_depth_pixels": int(valid_depths.size),
        "detection_valid_depth_ratio": float(valid_depths.size / depth_m.size),
        "detection_depth_percentiles_m": {
            str(percentile): float(value)
            for percentile, value in zip(
                [1, 5, 10, 20, 50, 80, 95, 99],
                depth_percentiles,
            )
        },
        "nearest_50_seed_depth_m": seed_depth_m,
        "camera_intrinsics": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
    }
    yolo_detections: list[dict[str, Any]] = []
    if algorithm == "v1":
        diagnostics.update(
            {
                "algorithm": "nearest_depth_component_v1",
                "algorithm_depth_band_m": 0.05,
                "search_farther_depth_clusters": SEARCH_FARTHER_DEPTH_CLUSTERS,
                "accepted_component_area_m2": [0.002, 0.0055],
                "accepted_main_length_m": [0.08, 0.18],
            }
        )
    elif algorithm == "v2":
        diagnostics.update(
            {
                "algorithm": "bilateral_depth_prominence_v2",
                "target_point_rule": "component centroid u, top-to-bottom 75% v",
                "accepted_depth_m": [0.08, 0.60],
                "accepted_main_length_m": [0.045, 0.20],
                "accepted_minor_length_max_m": 0.055,
                "accepted_aspect_ratio_min": 1.8,
                "accepted_verticality_min": 0.75,
                "strict_prominence_m": [0.008, 0.080],
                "relaxed_prominence_m": [0.005, 0.080],
            }
        )
    elif algorithm == "v3":
        if window_handle_detector is None:
            raise ValueError("algorithm='v3' 必须传入已加载的window_handle_detector")
        yolo_detections = window_handle_detector.detect(bgr)
        diagnostics.update(
            {
                "algorithm": "window_handle_yolo_depth_cluster_v3",
                "target_point_rule": "YOLO bbox center u, top-to-bottom 60% v",
                "accepted_depth_m": [0.08, 0.60],
                "depth_histogram_bin_m": 0.005,
                "depth_cluster_band_m": 0.025,
                "minimum_cluster_pixels": "max(20, 0.2% of YOLO ROI)",
                "yolo_detections": yolo_detections,
            }
        )
    else:
        raise ValueError("algorithm 必须是 'v1'、'v2' 或 'v3'")
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("感知输入诊断：", flush=True)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)

    selected_yolo: dict[str, Any] | None = None
    if algorithm == "v1":
        detection = detect_handle_center(
            depth_m,
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            search_farther_depth_clusters=SEARCH_FARTHER_DEPTH_CLUSTERS,
        )
    elif algorithm == "v2":
        detection = detect_handle_center_v2(
            depth_m,
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
        )
    else:
        if not yolo_detections:
            raise RuntimeError("YOLO未检出窗把手")
        detection = None
        for yolo_detection in yolo_detections:
            try:
                detection = detect_handle_depth_in_bbox_v3(
                    depth_m,
                    yolo_detection["bbox_xyxy"],
                    fy=float(camera_matrix[1, 1]),
                )
            except RuntimeError:
                continue
            selected_yolo = yolo_detection
            break
        if detection is None or selected_yolo is None:
            raise RuntimeError("YOLO窗把手框内均未获得稳定深度")
    u, v, target_depth_m = detection.uv_depth_m
    point_camera_m = [
        float(
            (u - camera_matrix[0, 2])
            * target_depth_m
            / camera_matrix[0, 0]
        ),
        float(
            (v - camera_matrix[1, 2])
            * target_depth_m
            / camera_matrix[1, 1]
        ),
        float(target_depth_m),
    ]
    if selected_yolo is None:
        ys, xs = np.nonzero(detection.mask)
        bbox_xyxy = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ]
    else:
        bbox_xyxy = [
            int(round(value)) for value in selected_yolo["bbox_xyxy"]
        ]
    cv2.imwrite(str(output_dir / "handle_mask.png"), detection.mask)
    rgb_target = bgr.copy()
    contours, _ = cv2.findContours(
        detection.mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rgb_target, contours, -1, (0, 255, 0), 2)
    cv2.circle(
        rgb_target,
        (int(round(u)), int(round(v))),
        9,
        (0, 0, 255),
        -1,
    )
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
        "depth_m": float(target_depth_m),
        "point_camera_optical_xyz_m": point_camera_m,
        "camera_axes": "+x right, +y down, +z forward",
        "component_bbox_xyxy": bbox_xyxy,
        "component_mask_pixels": int(np.count_nonzero(detection.mask)),
        "component_main_length_m": float(detection.main_length_m),
    }
    if algorithm == "v1":
        result.update(
            {
                "component_area_m2": float(detection.area_m2),
                "selected_seed_depth_m": float(detection.seed_depth_m),
                "depth_cluster_attempt": int(detection.depth_cluster_attempt),
            }
        )
    elif algorithm == "v2":
        result.update(
            {
                "component_minor_length_m": float(detection.minor_length_m),
                "component_aspect_ratio": float(detection.aspect_ratio),
                "component_prominence_m": float(detection.prominence_m),
                "component_score": float(detection.score),
                "target_point_rule": "component centroid u, top-to-bottom 75% v",
            }
        )
    else:
        result.update(
            {
                "yolo_confidence": float(selected_yolo["confidence"]),
                "yolo_bbox_xyxy": selected_yolo["bbox_xyxy"],
                "depth_cluster_center_m": float(detection.cluster_center_m),
                "depth_cluster_pixels": int(detection.cluster_pixel_count),
                "roi_valid_depth_pixels": int(detection.roi_valid_depth_pixels),
                "target_point_rule": "YOLO bbox center u, top-to-bottom 60% v",
            }
        )
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
