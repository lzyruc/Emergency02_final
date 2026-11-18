"""文件框/垃圾桶双腕抓取点检测与坐标转换。

输入为左右腕对齐的 BGR uint8 与毫米制 uint16 深度图；候选像素使用原图
(u, v)，输出目标位姿位于 waist_yaw_link，TCP 局部点单位为米。
YOLO 检测框作为 RGB-D 规则的 ROI，规则输出会还原为原图像素坐标。
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

import settings
from wrist_self_mask import gripper_mask

WRIST_DEPTH_SCALE_M = 0.001
MIN_TARGET_DEPTH_MM = 150
MAX_TARGET_DEPTH_MM = 1500
RIDGE_HALF_WIDTHS_PX = (4, 6, 8, 12, 16, 24, 32)
RIDGE_MIN_JUMP_MM = 35.0
RIDGE_VERTICAL_WINDOW_PX = 41
RIDGE_MIN_VERTICAL_SUPPORT = 0.42
RIDGE_MAX_VERTICAL_STD_MM = 55.0
FILEBOX_RIDGE_ANGLES_DEG = tuple(range(10, 180, 10))
FILEBOX_HORIZONTAL_REJECT_DEG = 25.0
BIN_HORIZONTAL_REJECT_DEG = 30.0
BIN_MAX_RIM_DEPTH_MM = 400
BIN_WHITE_MIN_VALUE = 180
BIN_NEUTRAL_WHITE_MAX_SATURATION = 35
# 决赛场地暖光把白色桶沿染成橙粉色；仅放宽暖色Hue，避免把浅绿和蓝墙当桶沿。
BIN_WARM_WHITE_MAX_HUE = 35
BIN_WARM_WHITE_MAX_SATURATION = 150
BIN_DEPTH_CLUSTER_GAP_MM = 25.0
BIN_DEPTH_CLUSTER_MIN_PIXELS = 8
BIN_DEPTH_SAMPLE_RADIUS_PX = 20
FILEBOX_RULE_SCALE = 0.5

# 比赛回退开关：True使用椭圆白桶缘，False恢复原select_bin_white_rim算法。
USE_BIN_ELLIPSE_RIM = True
BIN_ELLIPSE_MIN_MINOR_PX = 105.0
BIN_ELLIPSE_MIN_MAJOR_PX = 150.0
BIN_ELLIPSE_MIN_AXIS_RATIO = 0.35
BIN_ELLIPSE_MIN_ARC_COVERAGE_DEG = 90.0
BIN_ELLIPSE_MAX_MEDIAN_RADIAL_ERROR = 0.18
BIN_ELLIPSE_ANNULUS_INNER = 0.78
BIN_ELLIPSE_ANNULUS_OUTER = 1.22


def candidate_motion_info(
    station: Any,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """返回目标点在对应TCP下的位置和waist_yaw_link平移量，单位米。"""
    arm = candidate["arm"]
    state = station.get_robot_state()
    robot = station.get_robot_handle().get("robot")
    if arm == "left":
        tcp_pose = robot.pin_kine.get_left_tcp_pose(
            state["arm_joints"]["left"]
        )
    else:
        tcp_pose = robot.pin_kine.get_right_tcp_pose(
            state["arm_joints"]["right"]
        )
    T_waist_tcp = tcp_pose.get_affine_matrix_m_ndarray()
    target_pose = np.asarray(
        candidate["target_pose_waist_xyz_xyzw"],
        dtype=np.float64,
    )
    point_tcp_m = (
        np.linalg.inv(T_waist_tcp) @ np.append(target_pose[:3], 1.0)
    )[:3]
    delta_waist_m = target_pose[:3] - T_waist_tcp[:3, 3]
    return {
        "point_tcp_xyz_m": point_tcp_m,
        "delta_waist_xyz_m": delta_waist_m,
        "distance_m": float(np.linalg.norm(delta_waist_m)),
    }


def select_candidate(
    target_type: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """选择到当前对应TCP三维距离最近的抓取候选。"""
    return (
        None
        if not candidates
        else min(candidates, key=lambda item: item["tcp_distance_m"])
    )


def select_vertical_depth_ridge(
    color: np.ndarray,
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
    target_type: str,
) -> tuple[
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """检测竖向连续且横向两侧深度更远的细长实体。"""
    image_height, image_width = depth_mm.shape
    x_min = max(0, min(image_width - 1, int(np.floor(bbox_xyxy[0]))))
    y_min = max(0, min(image_height - 1, int(np.floor(bbox_xyxy[1]))))
    x_max = max(x_min + 1, min(image_width, int(np.ceil(bbox_xyxy[2]))))
    y_max = max(y_min + 1, min(image_height, int(np.ceil(bbox_xyxy[3]))))
    crop = depth_mm[y_min:y_max, x_min:x_max]
    crop_color = color[y_min:y_max, x_min:x_max]
    if crop.shape[0] < RIDGE_VERTICAL_WINDOW_PX or crop.shape[1] < 20:
        empty = np.zeros_like(depth_mm, dtype=np.uint8)
        return None, empty, empty, empty

    hsv = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
    if target_type == "filebox":
        material_mask = cv2.inRange(
            hsv,
            np.asarray([85, 70, 35], dtype=np.uint8),
            np.asarray([135, 255, 255], dtype=np.uint8),
        )
    else:
        material_mask = cv2.inRange(
            hsv,
            np.asarray([0, 0, 150], dtype=np.uint8),
            np.asarray([179, 80, 255], dtype=np.uint8),
        )
    material_mask = cv2.morphologyEx(
        material_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )

    smooth = cv2.medianBlur(crop, 5).astype(np.float32)
    valid = (
        (smooth >= MIN_TARGET_DEPTH_MM)
        & (smooth <= MAX_TARGET_DEPTH_MM)
        & (smooth != 65535)
    )
    best_score = np.zeros(crop.shape, dtype=np.float32)
    best_support = np.zeros(crop.shape, dtype=np.float32)
    best_half_width = np.zeros(crop.shape, dtype=np.uint8)

    for half_width in RIDGE_HALF_WIDTHS_PX:
        if crop.shape[1] <= 2 * half_width:
            continue
        center = smooth[:, half_width:-half_width]
        left = smooth[:, : -2 * half_width]
        right = smooth[:, 2 * half_width :]
        center_valid = valid[:, half_width:-half_width]
        pair_valid = (
            center_valid
            & valid[:, : -2 * half_width]
            & valid[:, 2 * half_width :]
        )
        left_jump = left - center
        right_jump = right - center
        raw_ridge = (
            pair_valid
            & (left_jump >= RIDGE_MIN_JUMP_MM)
            & (right_jump >= RIDGE_MIN_JUMP_MM)
        )
        vertical_support = cv2.boxFilter(
            raw_ridge.astype(np.float32),
            -1,
            (3, RIDGE_VERTICAL_WINDOW_PX),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE,
        )

        vertical_count = cv2.boxFilter(
            center_valid.astype(np.float32),
            -1,
            (1, RIDGE_VERTICAL_WINDOW_PX),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        center_values = np.where(center_valid, center, 0.0)
        vertical_sum = cv2.boxFilter(
            center_values,
            -1,
            (1, RIDGE_VERTICAL_WINDOW_PX),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        vertical_square_sum = cv2.boxFilter(
            center_values * center_values,
            -1,
            (1, RIDGE_VERTICAL_WINDOW_PX),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        safe_count = np.maximum(vertical_count, 1.0)
        vertical_mean = vertical_sum / safe_count
        vertical_std = np.sqrt(
            np.maximum(vertical_square_sum / safe_count - vertical_mean**2, 0.0)
        )
        jump_strength = np.minimum(left_jump, right_jump)
        supported_jump = cv2.boxFilter(
            np.where(raw_ridge, jump_strength, 0.0).astype(np.float32),
            -1,
            (3, RIDGE_VERTICAL_WINDOW_PX),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE,
        )
        score = supported_jump / (1.0 + vertical_std / 30.0)
        accepted = (
            (vertical_support >= RIDGE_MIN_VERTICAL_SUPPORT)
            & (vertical_count >= 0.75 * RIDGE_VERTICAL_WINDOW_PX)
            & (vertical_std <= RIDGE_MAX_VERTICAL_STD_MM)
        )
        destination = best_score[:, half_width:-half_width]
        update = accepted & (score > destination)
        destination[update] = score[update]
        best_support[:, half_width:-half_width][update] = vertical_support[update]
        best_half_width[:, half_width:-half_width][update] = half_width

    ridge_mask = (best_score > 0.0) & (material_mask > 0)
    ridge_mask = cv2.morphologyEx(
        ridge_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 1), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ridge_mask,
        connectivity=8,
    )
    components = [
        label
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_HEIGHT] >= 25
        and stats[label, cv2.CC_STAT_HEIGHT]
        >= 1.5 * stats[label, cv2.CC_STAT_WIDTH]
    ]
    full_mask = np.zeros_like(depth_mm, dtype=np.uint8)
    score_image = np.zeros_like(depth_mm, dtype=np.uint8)
    full_material_mask = np.zeros_like(depth_mm, dtype=np.uint8)
    full_material_mask[y_min:y_max, x_min:x_max] = material_mask
    if np.max(best_score) > 0.0:
        normalized_score = np.clip(
            best_score * (255.0 / np.max(best_score)),
            0.0,
            255.0,
        ).astype(np.uint8)
        score_image[y_min:y_max, x_min:x_max] = normalized_score
    full_mask[y_min:y_max, x_min:x_max] = ridge_mask * 255
    if not components:
        return None, full_mask, score_image, full_material_mask

    selected_label = max(
        components,
        key=lambda label: float(
            np.mean(best_score[labels == label])
            * stats[label, cv2.CC_STAT_HEIGHT]
        ),
    )
    ys, xs = np.nonzero(labels == selected_label)
    middle_y = 0.5 * (float(np.min(ys)) + float(np.max(ys)))
    middle_indices = np.flatnonzero(np.abs(ys - middle_y) <= 3.0)
    if not middle_indices.size:
        middle_indices = np.arange(xs.size)
    selected_index = int(
        middle_indices[
            np.argmax(best_score[ys[middle_indices], xs[middle_indices]])
        ]
    )
    crop_u, crop_v = int(xs[selected_index]), int(ys[selected_index])
    half_width = int(best_half_width[crop_v, crop_u])
    center_depth = smooth[crop_v, crop_u]
    depth_patch = crop[
        max(0, crop_v - 3) : min(crop.shape[0], crop_v + 4),
        max(0, crop_u - max(1, half_width // 4)) : min(
            crop.shape[1], crop_u + max(1, half_width // 4) + 1
        ),
    ]
    depth_values = depth_patch[
        (depth_patch >= MIN_TARGET_DEPTH_MM)
        & (depth_patch <= MAX_TARGET_DEPTH_MM)
        & (np.abs(depth_patch.astype(np.float32) - center_depth) <= 25.0)
    ]
    if not depth_values.size:
        return None, full_mask, score_image, full_material_mask
    left_depth = smooth[crop_v, crop_u - half_width]
    right_depth = smooth[crop_v, crop_u + half_width]
    return (
        {
            "pixel_uv": [x_min + crop_u, y_min + crop_v],
            "depth_m": float(np.median(depth_values)) * WRIST_DEPTH_SCALE_M,
            "point_source": "vertical_depth_ridge",
            "ridge_half_width_px": half_width,
            "horizontal_jump_left_mm": float(left_depth - center_depth),
            "horizontal_jump_right_mm": float(right_depth - center_depth),
            "vertical_support": float(best_support[crop_v, crop_u]),
            "segment_height_px": int(stats[selected_label, cv2.CC_STAT_HEIGHT]),
            "ridge_score": float(best_score[crop_v, crop_u]),
        },
        full_mask,
        score_image,
        full_material_mask,
    )


def select_bin_white_rim(
    color: np.ndarray,
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
    arm: str,
) -> tuple[
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """在垃圾桶YOLO框内选择白色圆弧桶沿，并排除近水平抓取方向。"""
    image_height, image_width = depth_mm.shape
    x_min = max(0, min(image_width - 1, int(np.floor(bbox_xyxy[0]))))
    y_min = max(0, min(image_height - 1, int(np.floor(bbox_xyxy[1]))))
    x_max = max(x_min + 1, min(image_width, int(np.ceil(bbox_xyxy[2]))))
    y_max = max(y_min + 1, min(image_height, int(np.ceil(bbox_xyxy[3]))))
    crop_color = color[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
    neutral_white_mask = cv2.inRange(
        hsv,
        np.asarray([0, 0, BIN_WHITE_MIN_VALUE], dtype=np.uint8),
        np.asarray(
            [179, BIN_NEUTRAL_WHITE_MAX_SATURATION, 255],
            dtype=np.uint8,
        ),
    )
    warm_white_mask = cv2.inRange(
        hsv,
        np.asarray(
            [0, BIN_NEUTRAL_WHITE_MAX_SATURATION + 1, BIN_WHITE_MIN_VALUE],
            dtype=np.uint8,
        ),
        np.asarray(
            [BIN_WARM_WHITE_MAX_HUE, BIN_WARM_WHITE_MAX_SATURATION, 255],
            dtype=np.uint8,
        ),
    )
    white_mask = cv2.bitwise_or(neutral_white_mask, warm_white_mask)
    valid_depth = (
        (crop_depth >= MIN_TARGET_DEPTH_MM)
        & (crop_depth <= BIN_MAX_RIM_DEPTH_MM)
        & (crop_depth != 65535)
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    white_valid = (white_mask > 0) & valid_depth
    if not np.any(white_valid):
        empty = np.zeros_like(depth_mm, dtype=np.uint8)
        material = np.zeros_like(depth_mm, dtype=np.uint8)
        material[y_min:y_max, x_min:x_max] = white_mask
        return None, empty, empty, material

    crop_height, crop_width = crop_depth.shape
    yy, xx = np.indices(crop_depth.shape, dtype=np.float32)
    center_x = 0.5 * (crop_width - 1)
    center_y = 0.5 * (crop_height - 1)
    radius_x = max(1.0, 0.5 * crop_width)
    radius_y = max(1.0, 0.5 * crop_height)
    normalized_x = (xx - center_x) / radius_x
    normalized_y = (yy - center_y) / radius_y
    normalized_radius = np.sqrt(normalized_x**2 + normalized_y**2)
    bbox_clipped = (
        x_min == 0
        or y_min == 0
        or x_max == image_width
        or y_max == image_height
    )
    if bbox_clipped:
        geometry_mask = np.ones(crop_depth.shape, dtype=bool)
        tangent_angle_deg = np.full(crop_depth.shape, np.nan, dtype=np.float32)
        orientation_score = np.zeros(crop_depth.shape, dtype=np.float32)
    else:
        geometry_mask = (
            (normalized_radius >= 0.72)
            & (normalized_radius <= 1.08)
        )
        tangent_x = -radius_x * normalized_y
        tangent_y = radius_y * normalized_x
        tangent_angle_deg = (
            np.degrees(np.arctan2(tangent_y, tangent_x)) % 180.0
        ).astype(np.float32)
        horizontal_distance_deg = np.minimum(
            tangent_angle_deg,
            180.0 - tangent_angle_deg,
        )
        geometry_mask &= horizontal_distance_deg >= BIN_HORIZONTAL_REJECT_DEG
        orientation_score = np.sin(
            np.deg2rad(horizontal_distance_deg)
        ).astype(np.float32)
    candidate_mask = white_valid & geometry_mask

    smooth_depth = cv2.medianBlur(crop_depth, 5).astype(np.float32)
    gradient_x = cv2.Sobel(smooth_depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    local_density = cv2.boxFilter(
        candidate_mask.astype(np.float32),
        -1,
        (9, 9),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    candidate_mask &= local_density >= 0.16
    if not np.any(candidate_mask):
        full_candidate = np.zeros_like(depth_mm, dtype=np.uint8)
        full_score = np.zeros_like(depth_mm, dtype=np.uint8)
        full_material = np.zeros_like(depth_mm, dtype=np.uint8)
        full_material[y_min:y_max, x_min:x_max] = white_mask
        return None, full_candidate, full_score, full_material

    if bbox_clipped:
        geometry_score = (
            80.0
            - 25.0 * np.abs(normalized_x)
            - 25.0 * np.abs(normalized_y)
        )
    else:
        side_score = normalized_x if arm == "right" else -normalized_x
        geometry_score = (
            55.0 * normalized_y
            - 20.0 * np.abs(normalized_x)
            - 80.0 * np.abs(normalized_radius - 0.90)
            + 70.0 * orientation_score
            + 20.0 * side_score
        )
    score = (
        geometry_score
        + 45.0 * local_density
        + 25.0 * np.minimum(depth_gradient, 160.0) / 160.0
    )
    score[~candidate_mask] = -np.inf
    crop_v, crop_u = np.unravel_index(np.argmax(score), score.shape)

    local_depth = crop_depth[
        max(0, crop_v - BIN_DEPTH_SAMPLE_RADIUS_PX) : min(
            crop_height,
            crop_v + BIN_DEPTH_SAMPLE_RADIUS_PX + 1,
        ),
        max(0, crop_u - BIN_DEPTH_SAMPLE_RADIUS_PX) : min(
            crop_width,
            crop_u + BIN_DEPTH_SAMPLE_RADIUS_PX + 1,
        ),
    ]
    local_white = white_valid[
        max(0, crop_v - BIN_DEPTH_SAMPLE_RADIUS_PX) : min(
            crop_height,
            crop_v + BIN_DEPTH_SAMPLE_RADIUS_PX + 1,
        ),
        max(0, crop_u - BIN_DEPTH_SAMPLE_RADIUS_PX) : min(
            crop_width,
            crop_u + BIN_DEPTH_SAMPLE_RADIUS_PX + 1,
        ),
    ]
    sorted_depths = np.sort(local_depth[local_white].astype(np.float32))
    if sorted_depths.size < BIN_DEPTH_CLUSTER_MIN_PIXELS:
        return None, np.zeros_like(depth_mm), np.zeros_like(depth_mm), np.zeros_like(depth_mm)
    split_indices = np.flatnonzero(
        np.diff(sorted_depths) > BIN_DEPTH_CLUSTER_GAP_MM
    ) + 1
    depth_clusters = np.split(sorted_depths, split_indices)
    depth_values = next(
        (
            cluster
            for cluster in depth_clusters
            if cluster.size >= BIN_DEPTH_CLUSTER_MIN_PIXELS
        ),
        None,
    )
    if depth_values is None:
        return None, np.zeros_like(depth_mm), np.zeros_like(depth_mm), np.zeros_like(depth_mm)

    finite_score = score[np.isfinite(score)]
    normalized_score = np.zeros(crop_depth.shape, dtype=np.uint8)
    if finite_score.size and np.max(finite_score) > np.min(finite_score):
        normalized_score[candidate_mask] = np.clip(
            (score[candidate_mask] - np.min(finite_score))
            * (255.0 / (np.max(finite_score) - np.min(finite_score))),
            0.0,
            255.0,
        ).astype(np.uint8)
    full_candidate = np.zeros_like(depth_mm, dtype=np.uint8)
    full_score = np.zeros_like(depth_mm, dtype=np.uint8)
    full_material = np.zeros_like(depth_mm, dtype=np.uint8)
    full_candidate[y_min:y_max, x_min:x_max] = candidate_mask * 255
    full_score[y_min:y_max, x_min:x_max] = normalized_score
    full_material[y_min:y_max, x_min:x_max] = white_mask
    return (
        {
            "pixel_uv": [x_min + int(crop_u), y_min + int(crop_v)],
            "depth_m": float(np.median(depth_values)) * WRIST_DEPTH_SCALE_M,
            "point_source": "bin_white_rim_arc",
            "depth_cluster_pixels": int(depth_values.size),
            "white_density": float(local_density[crop_v, crop_u]),
            "normalized_radius": float(normalized_radius[crop_v, crop_u]),
            "depth_gradient_mm": float(depth_gradient[crop_v, crop_u]),
            "bbox_clipped": bbox_clipped,
            "tangent_angle_deg": (
                None
                if bbox_clipped
                else float(tangent_angle_deg[crop_v, crop_u])
            ),
            "rim_score": float(score[crop_v, crop_u]),
        },
        full_candidate,
        full_score,
        full_material,
    )


def select_bin_white_rim_ellipse(
    color: np.ndarray,
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
    arm: str,
) -> tuple[
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """以白色、深度和椭圆几何选择垃圾桶侧缘抓取点。"""
    image_height, image_width = depth_mm.shape
    x_min = max(0, min(image_width - 1, int(np.floor(bbox_xyxy[0]))))
    y_min = max(0, min(image_height - 1, int(np.floor(bbox_xyxy[1]))))
    x_max = max(x_min + 1, min(image_width, int(np.ceil(bbox_xyxy[2]))))
    y_max = max(y_min + 1, min(image_height, int(np.ceil(bbox_xyxy[3]))))
    crop_color = color[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
    neutral_white_mask = cv2.inRange(
        hsv,
        np.asarray([0, 0, BIN_WHITE_MIN_VALUE], dtype=np.uint8),
        np.asarray(
            [179, BIN_NEUTRAL_WHITE_MAX_SATURATION, 255],
            dtype=np.uint8,
        ),
    )
    warm_white_mask = cv2.inRange(
        hsv,
        np.asarray(
            [0, BIN_NEUTRAL_WHITE_MAX_SATURATION + 1, BIN_WHITE_MIN_VALUE],
            dtype=np.uint8,
        ),
        np.asarray(
            [BIN_WARM_WHITE_MAX_HUE, BIN_WARM_WHITE_MAX_SATURATION, 255],
            dtype=np.uint8,
        ),
    )
    white_mask = cv2.bitwise_or(neutral_white_mask, warm_white_mask)
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    valid_depth = (
        (crop_depth >= MIN_TARGET_DEPTH_MM)
        & (crop_depth <= BIN_MAX_RIM_DEPTH_MM)
        & (crop_depth != 65535)
    )
    white_valid = (white_mask > 0) & valid_depth
    full_candidate = np.zeros_like(depth_mm, dtype=np.uint8)
    full_score = np.zeros_like(depth_mm, dtype=np.uint8)
    full_material = np.zeros_like(depth_mm, dtype=np.uint8)
    full_material[y_min:y_max, x_min:x_max] = white_mask
    if not np.any(white_valid):
        return None, full_candidate, full_score, full_material

    contours, _ = cv2.findContours(
        np.uint8(white_valid) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    yy, xx = np.indices(crop_depth.shape, dtype=np.float32)
    smooth_depth = cv2.medianBlur(crop_depth, 5).astype(np.float32)
    gradient_x = cv2.Sobel(smooth_depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    best_result = None
    best_ellipse_score = -np.inf
    best_candidate_mask = None
    best_point_score = None

    for contour in contours:
        if len(contour) < 20 or cv2.arcLength(contour, False) < 80.0:
            continue
        ellipse = cv2.fitEllipse(contour)
        (center_x, center_y), axes, angle_deg = ellipse
        minor = float(min(axes))
        major = float(max(axes))
        if (
            minor < BIN_ELLIPSE_MIN_MINOR_PX
            or major < BIN_ELLIPSE_MIN_MAJOR_PX
            or minor / major < BIN_ELLIPSE_MIN_AXIS_RATIO
            or major > 2.2 * max(crop_depth.shape)
        ):
            continue

        angle = np.deg2rad(angle_deg)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        delta_x = xx - center_x
        delta_y = yy - center_y
        local_x = cos_angle * delta_x + sin_angle * delta_y
        local_y = -sin_angle * delta_x + cos_angle * delta_y
        radius_x = max(1.0, 0.5 * axes[0])
        radius_y = max(1.0, 0.5 * axes[1])
        normalized_radius = np.sqrt(
            (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2
        )
        parameter_angle = (
            np.degrees(
                np.arctan2(local_y / radius_y, local_x / radius_x)
            )
            % 360.0
        )
        tangent_local_x = -local_y / (radius_y * radius_y)
        tangent_local_y = local_x / (radius_x * radius_x)
        tangent_x = (
            cos_angle * tangent_local_x - sin_angle * tangent_local_y
        )
        tangent_y = (
            sin_angle * tangent_local_x + cos_angle * tangent_local_y
        )
        tangent_angle_deg = (
            np.degrees(np.arctan2(tangent_y, tangent_x)) % 180.0
        )
        horizontal_distance_deg = np.minimum(
            tangent_angle_deg,
            180.0 - tangent_angle_deg,
        )
        annulus = (
            white_valid
            & (normalized_radius >= BIN_ELLIPSE_ANNULUS_INNER)
            & (normalized_radius <= BIN_ELLIPSE_ANNULUS_OUTER)
        )
        annulus_count = int(np.count_nonzero(annulus))
        if annulus_count < 80:
            continue
        angle_bins = np.unique(
            np.floor(parameter_angle[annulus] / 5.0).astype(np.int32)
        )
        coverage_deg = float(angle_bins.size * 5.0)
        radial_error = float(
            np.median(np.abs(normalized_radius[annulus] - 1.0))
        )
        if (
            coverage_deg < BIN_ELLIPSE_MIN_ARC_COVERAGE_DEG
            or radial_error > BIN_ELLIPSE_MAX_MEDIAN_RADIAL_ERROR
        ):
            continue
        # 小短轴仅允许稳定的局部桶缘，避免把完整的地面反光拟合成桶沿。
        if minor < 150.0 and not (
            major >= 240.0
            and coverage_deg <= 240.0
            and radial_error <= 0.09
        ):
            continue
        candidate_mask = (
            annulus
            & (horizontal_distance_deg >= BIN_HORIZONTAL_REJECT_DEG)
        )
        if not np.any(candidate_mask):
            continue
        local_density = cv2.boxFilter(
            candidate_mask.astype(np.float32),
            -1,
            (9, 9),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE,
        )
        side_score = xx / max(1.0, crop_depth.shape[1] - 1.0)
        if arm == "left":
            side_score = 1.0 - side_score
        point_score = (
            55.0 * local_density
            + 30.0 * np.minimum(depth_gradient, 160.0) / 160.0
            + 15.0 * side_score
            - 60.0 * np.abs(normalized_radius - 1.0)
        )
        point_score[~candidate_mask] = -np.inf
        crop_v, crop_u = np.unravel_index(
            np.argmax(point_score),
            point_score.shape,
        )
        ellipse_score = (
            1.8 * coverage_deg
            + 0.04 * annulus_count
            + 0.02 * np.sqrt(float(crop_depth.size)) * minor / major
            - 250.0 * radial_error
        )
        if ellipse_score <= best_ellipse_score:
            continue

        local_depth = crop_depth[
            max(0, crop_v - BIN_DEPTH_SAMPLE_RADIUS_PX):
            min(crop_depth.shape[0], crop_v + BIN_DEPTH_SAMPLE_RADIUS_PX + 1),
            max(0, crop_u - BIN_DEPTH_SAMPLE_RADIUS_PX):
            min(crop_depth.shape[1], crop_u + BIN_DEPTH_SAMPLE_RADIUS_PX + 1),
        ]
        local_white = white_valid[
            max(0, crop_v - BIN_DEPTH_SAMPLE_RADIUS_PX):
            min(crop_depth.shape[0], crop_v + BIN_DEPTH_SAMPLE_RADIUS_PX + 1),
            max(0, crop_u - BIN_DEPTH_SAMPLE_RADIUS_PX):
            min(crop_depth.shape[1], crop_u + BIN_DEPTH_SAMPLE_RADIUS_PX + 1),
        ]
        sorted_depths = np.sort(local_depth[local_white].astype(np.float32))
        if sorted_depths.size < BIN_DEPTH_CLUSTER_MIN_PIXELS:
            continue
        split_indices = np.flatnonzero(
            np.diff(sorted_depths) > BIN_DEPTH_CLUSTER_GAP_MM
        ) + 1
        depth_values = next(
            (
                cluster
                for cluster in np.split(sorted_depths, split_indices)
                if cluster.size >= BIN_DEPTH_CLUSTER_MIN_PIXELS
            ),
            None,
        )
        if depth_values is None:
            continue
        best_ellipse_score = ellipse_score
        best_candidate_mask = candidate_mask
        best_point_score = point_score
        best_result = {
            "pixel_uv": [x_min + int(crop_u), y_min + int(crop_v)],
            "depth_m": float(np.median(depth_values)) * WRIST_DEPTH_SCALE_M,
            "point_source": "bin_white_rim_ellipse",
            "depth_cluster_pixels": int(depth_values.size),
            "ellipse_center_uv": [
                x_min + float(center_x),
                y_min + float(center_y),
            ],
            "ellipse_axes_px": [float(axes[0]), float(axes[1])],
            "ellipse_angle_deg": float(angle_deg),
            "arc_coverage_deg": coverage_deg,
            "median_radial_error": radial_error,
            "tangent_angle_deg": float(tangent_angle_deg[crop_v, crop_u]),
            "horizontal_distance_deg": float(
                horizontal_distance_deg[crop_v, crop_u]
            ),
            "depth_gradient_mm": float(depth_gradient[crop_v, crop_u]),
            "ellipse_score": float(ellipse_score),
        }

    if best_result is None:
        return None, full_candidate, full_score, full_material
    full_candidate[y_min:y_max, x_min:x_max] = (
        np.uint8(best_candidate_mask) * 255
    )
    finite_score = best_point_score[np.isfinite(best_point_score)]
    normalized_score = np.zeros(crop_depth.shape, dtype=np.uint8)
    if finite_score.size and np.max(finite_score) > np.min(finite_score):
        normalized_score[best_candidate_mask] = np.clip(
            (best_point_score[best_candidate_mask] - np.min(finite_score))
            * (255.0 / (np.max(finite_score) - np.min(finite_score))),
            0.0,
            255.0,
        ).astype(np.uint8)
    full_score[y_min:y_max, x_min:x_max] = normalized_score
    return best_result, full_candidate, full_score, full_material


def select_filebox_directional_depth_ridge(
    color: np.ndarray,
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> tuple[
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """检测除水平线外任意方向的蓝色文件框深度楞。"""
    image_height, image_width = depth_mm.shape
    x_min = max(0, min(image_width - 1, int(np.floor(bbox_xyxy[0]))))
    y_min = max(0, min(image_height - 1, int(np.floor(bbox_xyxy[1]))))
    x_max = max(x_min + 1, min(image_width, int(np.ceil(bbox_xyxy[2]))))
    y_max = max(y_min + 1, min(image_height, int(np.ceil(bbox_xyxy[3]))))
    crop_color = color[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    crop_height, crop_width = crop_depth.shape
    canvas_size = int(np.ceil(np.hypot(crop_width, crop_height)))
    offset_x = (canvas_size - crop_width) // 2
    offset_y = (canvas_size - crop_height) // 2
    color_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    depth_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
    color_canvas[
        offset_y : offset_y + crop_height,
        offset_x : offset_x + crop_width,
    ] = crop_color
    depth_canvas[
        offset_y : offset_y + crop_height,
        offset_x : offset_x + crop_width,
    ] = crop_depth

    center = (0.5 * (canvas_size - 1), 0.5 * (canvas_size - 1))
    best_result = None
    best_masks = None
    best_directional_score = -np.inf
    for line_angle_deg in FILEBOX_RIDGE_ANGLES_DEG:
        rotation_deg = 90.0 - float(line_angle_deg)
        rotation = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        rotated_color = cv2.warpAffine(
            color_canvas,
            rotation,
            (canvas_size, canvas_size),
            flags=cv2.INTER_NEAREST,
            borderValue=(0, 0, 0),
        )
        rotated_depth = cv2.warpAffine(
            depth_canvas,
            rotation,
            (canvas_size, canvas_size),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        result, ridge_mask, ridge_score, material_mask = (
            select_vertical_depth_ridge(
                rotated_color,
                rotated_depth,
                [0.0, 0.0, float(canvas_size), float(canvas_size)],
                "filebox",
            )
        )
        if result is None:
            continue
        directional_score = float(
            result["ridge_score"] * np.sqrt(result["segment_height_px"])
        )
        if directional_score <= best_directional_score:
            continue

        inverse_rotation = cv2.invertAffineTransform(rotation)
        rotated_u, rotated_v = result["pixel_uv"]
        canvas_u, canvas_v = inverse_rotation @ np.asarray(
            [rotated_u, rotated_v, 1.0],
            dtype=np.float64,
        )
        crop_u = int(round(canvas_u - offset_x))
        crop_v = int(round(canvas_v - offset_y))
        if not (0 <= crop_u < crop_width and 0 <= crop_v < crop_height):
            continue
        u, v = x_min + crop_u, y_min + crop_v
        reference_depth_mm = result["depth_m"] / WRIST_DEPTH_SCALE_M
        local_depth = depth_mm[
            max(0, v - 3) : min(image_height, v + 4),
            max(0, u - 3) : min(image_width, u + 4),
        ]
        depth_values = local_depth[
            (local_depth >= MIN_TARGET_DEPTH_MM)
            & (local_depth <= MAX_TARGET_DEPTH_MM)
            & (
                np.abs(local_depth.astype(np.float32) - reference_depth_mm)
                <= 30.0
            )
        ]
        if not depth_values.size:
            continue

        unrotated_ridge = cv2.warpAffine(
            ridge_mask,
            inverse_rotation,
            (canvas_size, canvas_size),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        unrotated_score = cv2.warpAffine(
            ridge_score,
            inverse_rotation,
            (canvas_size, canvas_size),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        unrotated_material = cv2.warpAffine(
            material_mask,
            inverse_rotation,
            (canvas_size, canvas_size),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        candidate_ridge = unrotated_ridge[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]
        _, component_labels = cv2.connectedComponents(
            (candidate_ridge > 0).astype(np.uint8),
            connectivity=8,
        )
        component_label = int(component_labels[crop_v, crop_u])
        if component_label == 0:
            candidate_ys, candidate_xs = np.nonzero(candidate_ridge)
            if not candidate_xs.size:
                continue
            nearest_index = int(
                np.argmin(
                    (candidate_xs - crop_u) ** 2
                    + (candidate_ys - crop_v) ** 2
                )
            )
            component_label = int(
                component_labels[
                    candidate_ys[nearest_index],
                    candidate_xs[nearest_index],
                ]
            )
        line_ys, line_xs = np.nonzero(component_labels == component_label)
        if line_xs.size < 10:
            continue
        line = cv2.fitLine(
            np.column_stack((line_xs, line_ys)).astype(np.float32),
            cv2.DIST_L2,
            0.0,
            0.01,
            0.01,
        ).reshape(-1)
        fitted_angle_deg = float(
            np.degrees(np.arctan2(line[1], line[0])) % 180.0
        )
        horizontal_distance_deg = min(
            fitted_angle_deg,
            180.0 - fitted_angle_deg,
        )
        if horizontal_distance_deg < FILEBOX_HORIZONTAL_REJECT_DEG:
            continue
        best_result = {
            **result,
            "pixel_uv": [u, v],
            "depth_m": float(np.median(depth_values)) * WRIST_DEPTH_SCALE_M,
            "point_source": "directional_depth_ridge",
            "line_angle_deg": fitted_angle_deg,
            "search_angle_deg": float(line_angle_deg),
            "directional_score": directional_score,
        }
        best_masks = (
            unrotated_ridge[
                offset_y : offset_y + crop_height,
                offset_x : offset_x + crop_width,
            ],
            unrotated_score[
                offset_y : offset_y + crop_height,
                offset_x : offset_x + crop_width,
            ],
            unrotated_material[
                offset_y : offset_y + crop_height,
                offset_x : offset_x + crop_width,
            ],
        )
        best_directional_score = directional_score

    full_ridge = np.zeros_like(depth_mm, dtype=np.uint8)
    full_score = np.zeros_like(depth_mm, dtype=np.uint8)
    full_material = np.zeros_like(depth_mm, dtype=np.uint8)
    if best_result is None or best_masks is None:
        return None, full_ridge, full_score, full_material
    full_ridge[y_min:y_max, x_min:x_max] = best_masks[0]
    full_score[y_min:y_max, x_min:x_max] = best_masks[1]
    full_material[y_min:y_max, x_min:x_max] = best_masks[2]
    return best_result, full_ridge, full_score, full_material


def select_filebox_resized_depth_ridge(
    color: np.ndarray,
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> tuple[
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """半分辨率搜索文件框抓取楞，结果映射回原始RGB-D像素。"""
    image_height, image_width = depth_mm.shape
    search_width = max(1, int(round(image_width * FILEBOX_RULE_SCALE)))
    search_height = max(1, int(round(image_height * FILEBOX_RULE_SCALE)))
    scale_x = search_width / image_width
    scale_y = search_height / image_height
    search_color = cv2.resize(
        color,
        (search_width, search_height),
        interpolation=cv2.INTER_AREA,
    )
    search_depth = cv2.resize(
        depth_mm,
        (search_width, search_height),
        interpolation=cv2.INTER_NEAREST,
    )
    result, ridge_mask, ridge_score, material_mask = (
        select_filebox_directional_depth_ridge(
            search_color,
            search_depth,
            [
                bbox_xyxy[0] * scale_x,
                bbox_xyxy[1] * scale_y,
                bbox_xyxy[2] * scale_x,
                bbox_xyxy[3] * scale_y,
            ],
        )
    )
    full_masks = tuple(
        cv2.resize(
            image,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        for image in (ridge_mask, ridge_score, material_mask)
    )
    if result is None:
        return None, *full_masks

    search_u, search_v = result["pixel_uv"]
    u = int(
        np.clip(
            round((search_u + 0.5) / scale_x - 0.5),
            0,
            image_width - 1,
        )
    )
    v = int(
        np.clip(
            round((search_v + 0.5) / scale_y - 0.5),
            0,
            image_height - 1,
        )
    )
    reference_depth_mm = result["depth_m"] / WRIST_DEPTH_SCALE_M
    local_depth = depth_mm[
        max(0, v - 3) : min(image_height, v + 4),
        max(0, u - 3) : min(image_width, u + 4),
    ]
    depth_values = local_depth[
        (local_depth >= MIN_TARGET_DEPTH_MM)
        & (local_depth <= MAX_TARGET_DEPTH_MM)
        & (
            np.abs(local_depth.astype(np.float32) - reference_depth_mm)
            <= 30.0
        )
    ]
    if not depth_values.size:
        return None, *full_masks
    result.update(
        {
            "pixel_uv": [u, v],
            "depth_m": float(np.median(depth_values))
            * WRIST_DEPTH_SCALE_M,
            "rule_image_size": [search_width, search_height],
            "rule_scale_xy": [scale_x, scale_y],
        }
    )
    return result, *full_masks


def wrist_yolo_grasp_candidates(
    station: Any,
    target_type: str,
    detector: Any,
    deprojectors: dict[str, Any],
    output_dir: Path,
    camera_reader: Any,
) -> list[dict[str, Any]]:
    """读取双腕帧，以YOLO框为ROI生成waist_yaw_link抓取Pose。

    ``target_type`` 为 ``filebox`` 或 ``bin``；``camera_reader`` 接收
    ``(station, camera_config)`` 并返回对齐的 BGR 与毫米深度图。每个返回候选
    包含 ``arm``、原图 ``pixel_uv``、米制 ``depth_m``、ROI来源、规则来源、
    YOLO框以及 ``waist_yaw_link`` 下的目标Pose。YOLO无当前类别候选时，规则
    回退到整幅RGB-D搜索。最小调用为：

    ``wrist_yolo_grasp_candidates(station, "bin", detector, deprojectors,``
    ``output_dir, read_wrist_camera)``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for arm, camera_config in (
        ("left", settings.LEFT_CAMERA),
        ("right", settings.RIGHT_CAMERA),
    ):
        color, depth_mm = camera_reader(station, camera_config)
        static_gripper_mask = gripper_mask(depth_mm.shape)
        masked_color = color.copy()
        masked_depth_mm = depth_mm.copy()
        masked_color[static_gripper_mask] = 0
        masked_depth_mm[static_gripper_mask] = 0
        prefix = output_dir / f"{arm}"
        if not cv2.imwrite(str(prefix.with_name(f"{arm}_rgb.jpg")), color):
            raise RuntimeError(f"{arm}腕RGB保存失败")
        if not cv2.imwrite(
            str(prefix.with_name(f"{arm}_depth.png")), depth_mm
        ):
            raise RuntimeError(f"{arm}腕深度保存失败")

        start = time.perf_counter()
        all_detections = detector.detect(color)
        inference_s = time.perf_counter() - start
        detections = [
            detection
            for detection in all_detections
            if detection["type"] == target_type
        ]
        prefix.with_name(f"{arm}_detections.json").write_text(
            json.dumps(all_detections, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        annotated = color.copy()
        for detection in all_detections:
            x_min, y_min, x_max, y_max = (
                int(round(value)) for value in detection["bbox_xyxy"]
            )
            color_bgr = (
                (0, 255, 255)
                if detection["type"] == target_type
                else (160, 160, 160)
            )
            cv2.rectangle(
                annotated,
                (x_min, y_min),
                (x_max, y_max),
                color_bgr,
                2,
            )
            cv2.putText(
                annotated,
                f"{detection['type']} {detection['confidence']:.3f}",
                (max(0, x_min), max(22, y_min + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
        detection_summary = ", ".join(
            f"{item['type']}:{item['confidence']:.3f}"
            for item in all_detections
        ) or "无"
        print(
            f"{arm}腕YOLO：目标={target_type}，候选={len(detections)}，"
            f"全部检测=[{detection_summary}]，推理={inference_s:.3f} s",
            flush=True,
        )
        fallback_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
        if detections:
            detection = max(
                detections,
                key=lambda item: item["confidence"],
            )
            rule_color = masked_color
            rule_depth_mm = masked_depth_mm
            bbox_xyxy = detection["bbox_xyxy"]
            confidence = float(detection["confidence"])
            roi_source = "yolo_bbox"
        else:
            rule_color = masked_color
            rule_depth_mm = masked_depth_mm
            bbox_xyxy = [
                0.0,
                0.0,
                float(depth_mm.shape[1]),
                float(depth_mm.shape[0]),
            ]
            confidence = 0.0
            roi_source = "full_frame_fallback"
            print(
                f"{arm}腕YOLO无{target_type}候选框："
                "在整幅图像执行RGB-D规则",
                flush=True,
            )
        if target_type == "bin":
            bin_selector = (
                select_bin_white_rim_ellipse
                if USE_BIN_ELLIPSE_RIM
                else select_bin_white_rim
            )
            depth_point, ridge_mask, ridge_score, material_mask = (
                bin_selector(
                    rule_color,
                    rule_depth_mm,
                    bbox_xyxy,
                    arm,
                )
            )
        else:
            depth_point, ridge_mask, ridge_score, material_mask = (
                select_filebox_resized_depth_ridge(
                    rule_color,
                    rule_depth_mm,
                    bbox_xyxy,
                )
            )
        for suffix, image in (
            ("ridge_mask", ridge_mask),
            ("ridge_score", ridge_score),
            ("material_mask", material_mask),
            ("fallback_mask", fallback_mask),
            ("gripper_static_mask", np.uint8(static_gripper_mask) * 255),
        ):
            if not cv2.imwrite(
                str(prefix.with_name(f"{arm}_{suffix}.png")),
                image,
            ):
                raise RuntimeError(f"{arm}腕{suffix}保存失败")
        if depth_point is not None:
            u, v = depth_point["pixel_uv"]
            if depth_point["point_source"] == "bin_white_rim_ellipse":
                cv2.ellipse(
                    annotated,
                    (
                        tuple(depth_point["ellipse_center_uv"]),
                        tuple(depth_point["ellipse_axes_px"]),
                        depth_point["ellipse_angle_deg"],
                    ),
                    (0, 255, 255),
                    3,
                )
            cv2.drawMarker(
                annotated,
                (u, v),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                28,
                3,
            )
            target_pose = deprojectors[arm].deproject(
                [u, v, depth_point["depth_m"]],
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
                    "class": target_type,
                    "confidence": confidence,
                    "bbox_xyxy": bbox_xyxy,
                    "roi_source": roi_source,
                    "pixel_uv": depth_point["pixel_uv"],
                    "point_source": depth_point["point_source"],
                    "depth_m": depth_point["depth_m"],
                    "point_details": {
                        key: value
                        for key, value in depth_point.items()
                        if key not in {"pixel_uv", "depth_m", "point_source"}
                    },
                    "inference_s": inference_s,
                    "target_pose_waist_xyz_xyzw": target_pose,
                    "tcp_distance_m": float(
                        np.linalg.norm(target_pose[:3] - current_pose[:3])
                    ),
                }
            )
        else:
            message = (
                "没有可靠白色圆弧桶沿"
                if target_type == "bin"
                else "没有非水平、沿法向双侧深度突变的抓取楞"
            )
            print(f"{arm}腕{roi_source}内{message}", flush=True)
        if not cv2.imwrite(
            str(prefix.with_name(f"{arm}_annotated.jpg")), annotated
        ):
            raise RuntimeError(f"{arm}腕标注图保存失败")
    return candidates
