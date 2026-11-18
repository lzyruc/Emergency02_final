"""右腕椅背顶端感知，为任务1背面转椅抓取提供目标Pose。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask


DEFAULT_CONFIG = {
    "right_roi_normalized_xyxy": (0.02, 0.00, 0.90, 0.80),
    "min_depth_mm": 150,
    "max_depth_mm": 400,
    "max_component_depth_mm": 380.0,
    "close_kernel_px": 7,
    "min_component_area_px": 5000,
    "min_line_elongation": 3.2,
    "max_line_thickness_px": 120,
    "min_line_length_px": 120,
    "central_axis_fraction": 0.60,
    "depth_score_weight": 0.55,
    "center_score_weight": 0.30,
    "dark_score_weight": 0.15,
    "best_score_fraction": 0.01,
    "max_grasp_roi_y_fraction": 0.85,
    "inward_depth_patch_px": 5,
    "min_patch_valid_ratio": 0.25,
    "max_patch_depth_range_mm": 180.0,
}


def _patch_values(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    radius_px: int,
) -> np.ndarray:
    """返回像素周围有效毫米深度。"""
    height, width = depth_mm.shape
    patch = depth_mm[
        max(0, v - radius_px) : min(height, v + radius_px + 1),
        max(0, u - radius_px) : min(width, u + radius_px + 1),
    ]
    return patch[(patch > 0) & (patch != 65535) & np.isfinite(patch)]


def _component_median_depth_mm(
    crop_mm: np.ndarray,
    labels: np.ndarray,
    component_id: int,
) -> float:
    values = crop_mm[labels == component_id]
    values = values[(values > 0) & (values != 65535) & np.isfinite(values)]
    return 1.0e9 if values.size == 0 else float(np.median(values))


def _pick_backtop_component(
    stats: np.ndarray,
    labels: np.ndarray,
    crop_mm: np.ndarray,
    component_count: int,
) -> int:
    """在近景连通域中优先选择细长椅背框线。"""
    cfg = DEFAULT_CONFIG
    candidates: list[tuple[int, float, int, int, int, float, int]] = []
    for component_id in range(1, component_count):
        _x, top_y, width_px, height_px, area = (
            int(value) for value in stats[component_id][:5]
        )
        if area < int(cfg["min_component_area_px"]):
            continue
        median_depth_mm = _component_median_depth_mm(
            crop_mm,
            labels,
            component_id,
        )
        if median_depth_mm > float(cfg["max_component_depth_mm"]):
            continue
        thickness = min(width_px, height_px)
        length = max(width_px, height_px)
        elongation = length / max(thickness, 1)
        candidates.append(
            (
                component_id,
                elongation,
                length,
                thickness,
                top_y,
                median_depth_mm,
                area,
            )
        )
    if not candidates:
        return 0
    lines = [
        item
        for item in candidates
        if item[1] >= float(cfg["min_line_elongation"])
        and item[3] <= int(cfg["max_line_thickness_px"])
        and item[2] >= int(cfg["min_line_length_px"])
    ]
    if lines:
        lines.sort(key=lambda item: -(item[1] * item[2] / max(item[3], 1)))
        return int(lines[0][0])
    candidates.sort(key=lambda item: (item[5], item[4], -item[6]))
    return int(candidates[0][0])


def _central_near_ridge_uv(
    component: np.ndarray,
    gray: np.ndarray,
    depth_mm: np.ndarray,
    central_axis_fraction: float,
    depth_score_weight: float,
    center_score_weight: float,
    dark_score_weight: float,
    best_score_fraction: float,
) -> tuple[int, int]:
    """在椅背主轴中段按近深度、中点距离和暗度选择稳定像素。"""
    rows, cols = np.nonzero(component)
    if rows.size == 0:
        raise RuntimeError("近景椅背上没有有效像素")
    points = np.column_stack((cols, rows)).astype(np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    projection = centered @ eigenvectors[:, -1]
    projection_range = float(np.ptp(projection))
    if projection_range <= 1e-6:
        return int(round(center[0])), int(round(center[1]))
    axis_position = (projection - float(np.min(projection))) / projection_range
    half_fraction = 0.5 * central_axis_fraction
    central = np.abs(axis_position - 0.5) <= half_fraction
    candidate_rows = rows[central]
    candidate_cols = cols[central]
    candidate_axis_position = axis_position[central]
    candidate_depth = depth_mm[candidate_rows, candidate_cols].astype(np.float64)
    candidate_gray = gray[candidate_rows, candidate_cols].astype(np.float64)

    depth_low, depth_high = np.percentile(candidate_depth, (10, 90))
    gray_low, gray_high = np.percentile(candidate_gray, (10, 90))
    depth_score = np.clip(
        (candidate_depth - depth_low) / max(depth_high - depth_low, 1.0),
        0.0,
        1.0,
    )
    dark_score = np.clip(
        (candidate_gray - gray_low) / max(gray_high - gray_low, 1.0),
        0.0,
        1.0,
    )
    center_score = np.abs(candidate_axis_position - 0.5) / max(
        half_fraction,
        1e-6,
    )
    score = (
        depth_score_weight * depth_score
        + center_score_weight * center_score
        + dark_score_weight * dark_score
    )
    best_count = max(5, int(round(best_score_fraction * score.size)))
    best_indices = np.argpartition(score, best_count - 1)[:best_count]
    best_points = np.column_stack(
        (candidate_cols[best_indices], candidate_rows[best_indices])
    )
    best_center = np.median(best_points, axis=0)
    selected_index = int(
        np.argmin(np.sum((best_points - best_center) ** 2, axis=1))
    )
    return int(best_points[selected_index, 0]), int(best_points[selected_index, 1])


def detect_chair_backtop_grasp(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    roi_normalized_xyxy: tuple[float, float, float, float],
    camera_fx_px: float,
    camera_fy_px: float,
) -> dict[str, Any]:
    """返回椅背顶端抓取像素、深度和诊断特征。"""
    cfg = DEFAULT_CONFIG
    depth = np.asarray(depth_mm)
    color = np.asarray(color_bgr)
    if depth.ndim != 2:
        raise ValueError("depth_mm必须是H×W毫米深度图")
    if color.ndim != 3 or color.shape[:2] != depth.shape:
        raise ValueError("color_bgr必须和深度图同分辨率")
    height, width = depth.shape
    x_min = max(0, min(width - 1, int(round(roi_normalized_xyxy[0] * width))))
    y_min = max(0, min(height - 1, int(round(roi_normalized_xyxy[1] * height))))
    x_max = max(x_min + 1, min(width, int(round(roi_normalized_xyxy[2] * width))))
    y_max = max(y_min + 1, min(height, int(round(roi_normalized_xyxy[3] * height))))
    crop = depth[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(color[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
    foreground = (
        (crop >= int(cfg["min_depth_mm"]))
        & (crop <= int(cfg["max_depth_mm"]))
        & (crop != 65535)
        & np.isfinite(crop)
    )
    kernel_size = int(cfg["close_kernel_px"]) | 1
    closed = cv2.morphologyEx(
        np.uint8(foreground) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    component_id = _pick_backtop_component(
        stats,
        labels,
        crop,
        component_count,
    )
    if component_id == 0:
        raise RuntimeError("ROI内没有可用的近景椅背")

    x, y, width_px, height_px, area = (
        int(value) for value in stats[component_id][:5]
    )
    component = (labels == component_id) & foreground
    vertical = height_px >= width_px
    local_u, local_v = _central_near_ridge_uv(
        component,
        gray,
        crop,
        float(cfg["central_axis_fraction"]),
        float(cfg["depth_score_weight"]),
        float(cfg["center_score_weight"]),
        float(cfg["dark_score_weight"]),
        float(cfg["best_score_fraction"]),
    )
    center_u = x_min + local_u
    center_v = y_min + local_v
    if local_v > float(cfg["max_grasp_roi_y_fraction"]) * crop.shape[0]:
        raise RuntimeError("椅背候选点位于ROI过低区域")
    sample_u = center_u
    sample_v = center_v
    patch_radius = max(1, int(cfg["inward_depth_patch_px"]) // 2)
    patch_values = _patch_values(depth, sample_u, sample_v, patch_radius)
    patch_area = (2 * patch_radius + 1) ** 2
    valid_ratio = patch_values.size / max(patch_area, 1)
    if patch_values.size == 0 or valid_ratio < float(cfg["min_patch_valid_ratio"]):
        raise RuntimeError("椅背细棱深度不稳定")
    depth_value_mm = float(np.median(patch_values))
    depth_range_mm = float(
        np.percentile(patch_values, 90) - np.percentile(patch_values, 10)
    )
    if depth_range_mm > float(cfg["max_patch_depth_range_mm"]):
        raise RuntimeError("椅背抓取点局部深度极差过大")
    focal_px = 0.5 * (float(camera_fx_px) + float(camera_fy_px))
    line_length_px = float(max(width_px, height_px))
    return {
        "pixel_uv": [center_u, center_v],
        "depth_sample_uv": [sample_u, sample_v],
        "depth_m": depth_value_mm / 1000.0,
        "axis": "vertical_rail_top" if vertical else "backtop_ridge",
        "line_length_px": int(round(line_length_px)),
        "line_length_m": float(
            line_length_px * depth_value_mm / (1000.0 * focal_px)
        ),
        "component_area_px": area,
        "patch_valid_ratio": float(valid_ratio),
        "patch_depth_range_mm": depth_range_mm,
        "point_rule": "central_axis_depth55_center30_dark15",
        "roi_xyxy": [x_min, y_min, x_max, y_max],
        "_component_mask": labels == component_id,
    }


def recognize_chair_backtop_grasp(
    station: Any,
    deprojector: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """读取右腕RGB-D并返回waist_yaw_link下保持当前方向的目标Pose。"""
    camera_config = settings.RIGHT_CAMERA
    color, depth_mm = read_wrist_camera(station, camera_config)
    static_mask = gripper_mask(depth_mm.shape)
    detection_depth_mm = depth_mm.copy()
    detection_depth_mm[static_mask] = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_right"
    raw_files = {
        "rgb": (output_dir / f"{prefix}_rgb.jpg", color),
        "depth": (output_dir / f"{prefix}_depth.png", depth_mm),
        "gripper_mask": (
            output_dir / f"{prefix}_gripper_mask.png",
            np.uint8(static_mask) * 255,
        ),
        "depth_masked": (
            output_dir / f"{prefix}_depth_masked.png",
            detection_depth_mm,
        ),
    }
    for path, image in raw_files.values():
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"右腕椅背诊断图保存失败：{path}")

    roi = tuple(DEFAULT_CONFIG["right_roi_normalized_xyxy"])
    detection = detect_chair_backtop_grasp(
        detection_depth_mm,
        color,
        roi,
        float(camera_config["camera_matrix"][0][0]),
        float(camera_config["camera_matrix"][1][1]),
    )
    component_mask = detection.pop("_component_mask")
    u, v = detection["pixel_uv"]
    target_pose = np.asarray(
        deprojector.deproject(
            [float(u), float(v), detection["depth_m"]],
            face_target=False,
        ),
        dtype=np.float64,
    ).copy()
    roi_xyxy = detection["roi_xyxy"]
    mask_image = np.zeros(depth_mm.shape, dtype=np.uint8)
    mask_image[
        roi_xyxy[1] : roi_xyxy[3],
        roi_xyxy[0] : roi_xyxy[2],
    ][component_mask] = 255
    annotated = color.copy()
    cv2.rectangle(
        annotated,
        tuple(roi_xyxy[:2]),
        tuple(roi_xyxy[2:]),
        (0, 255, 255),
        2,
    )
    cv2.circle(annotated, (u, v), 9, (0, 255, 0), -1)
    cv2.circle(
        annotated,
        tuple(detection["depth_sample_uv"]),
        6,
        (0, 128, 255),
        -1,
    )
    annotated_path = output_dir / f"{prefix}_annotated.jpg"
    mask_path = output_dir / f"{prefix}_backtop_mask.png"
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(f"右腕椅背标注图保存失败：{annotated_path}")
    if not cv2.imwrite(str(mask_path), mask_image):
        raise RuntimeError(f"右腕椅背mask保存失败：{mask_path}")
    detection["target_pose_waist_xyz_xyzw"] = target_pose
    detection["output_files"] = {
        **{name: str(path) for name, (path, _) in raw_files.items()},
        "annotated": str(annotated_path),
        "backtop_mask": str(mask_path),
    }
    return detection


def plan_chair_backtop_grasp_steps(
    delta_tcp_m: np.ndarray,
    *,
    approach_m: float,
    standoff_m: float,
    z_compensation_m: float,
    second_stage_z_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """将目标TCP位移分成XYZ预接近和剩余Z接近。"""
    remaining_z = abs(float(delta_tcp_m[2]))
    if remaining_z <= standoff_m + 1e-6:
        return np.asarray(
            [float(delta_tcp_m[0]), float(delta_tcp_m[1]), 0.0],
            dtype=np.float64,
        ), np.zeros(3, dtype=np.float64)
    z_magnitude = min(remaining_z - standoff_m, approach_m)
    z_magnitude = max(0.0, z_magnitude + z_compensation_m)
    second_z_magnitude = min(z_magnitude, second_stage_z_m)
    z_direction = float(np.sign(delta_tcp_m[2]))
    first_delta = np.asarray(
        [
            float(delta_tcp_m[0]),
            float(delta_tcp_m[1]),
            z_direction * (z_magnitude - second_z_magnitude),
        ],
        dtype=np.float64,
    )
    second_delta = np.asarray(
        [0.0, 0.0, z_direction * second_z_magnitude],
        dtype=np.float64,
    )
    return first_delta, second_delta
