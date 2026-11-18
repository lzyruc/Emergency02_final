"""正放办公转椅左右扶手的腕部RGB-D抓取点识别。

左右腕分别只在自己的固定ROI中查找近景扶手。算法先用深度范围和
静态夹爪mask分离扶手，再用连通域和PCA拟合扶手长轴，最后在长轴中段选择
两侧深度突变、局部深度稳定的横截面中心。本模块只采集和计算，不下发运动。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from camera_io import read_wrist_camera
from wrist_self_mask import gripper_mask


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


def detect_armrest_grasp(
    depth_mm: np.ndarray,
    roi_normalized_xyxy: tuple[float, float, float, float],
    *,
    min_depth_mm: int,
    max_depth_mm: int,
    close_kernel_px: int,
    min_component_area_px: int,
    min_major_span_px: int,
    min_elongation_ratio: float,
    cross_section_band_px: int,
    side_sample_offset_px: int,
    min_side_jump_mm: float,
    min_grasp_width_m: float,
    max_grasp_width_m: float,
    stability_patch_px: int,
    min_patch_valid_ratio: float,
    max_patch_depth_range_mm: float,
    camera_fx_px: float,
    camera_fy_px: float,
) -> dict[str, Any]:
    """在指定侧ROI内返回近景扶手中段抓取点。

    输入深度单位为毫米，ROI顺序为 ``x_min, y_min, x_max, y_max``。
    输出像素使用原始图像坐标，物理宽度和长度单位为米。
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("depth_mm 必须是 H×W 毫米深度图")
    height, width = depth.shape
    x_min = max(0, min(width - 1, int(round(roi_normalized_xyxy[0] * width))))
    y_min = max(0, min(height - 1, int(round(roi_normalized_xyxy[1] * height))))
    x_max = max(x_min + 1, min(width, int(round(roi_normalized_xyxy[2] * width))))
    y_max = max(y_min + 1, min(height, int(round(roi_normalized_xyxy[3] * height))))
    crop = depth[y_min:y_max, x_min:x_max]
    foreground = (
        (crop >= min_depth_mm)
        & (crop <= max_depth_mm)
        & (crop != 65535)
        & np.isfinite(crop)
    )
    kernel_size = int(close_kernel_px) | 1
    closed = cv2.morphologyEx(
        np.uint8(foreground) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    candidates = []
    focal_px = 0.5 * (float(camera_fx_px) + float(camera_fy_px))
    patch_radius = int(stability_patch_px) // 2

    for component_id in range(1, component_count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < min_component_area_px:
            continue
        component = (labels == component_id) & foreground
        rows, columns = np.nonzero(component)
        if rows.size < min_component_area_px:
            continue
        points_xy = np.column_stack((columns, rows)).astype(np.float64)
        centroid_xy = points_xy.mean(axis=0)
        centered = points_xy - centroid_xy
        covariance = centered.T @ centered / points_xy.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor_axis = np.asarray([-major_axis[1], major_axis[0]])
        major_projection = centered @ major_axis
        minor_projection = centered @ minor_axis
        major_span_px = float(np.ptp(major_projection))
        minor_span_px = float(np.ptp(minor_projection))
        elongation = major_span_px / max(minor_span_px, 1.0)
        if major_span_px < min_major_span_px or elongation < min_elongation_ratio:
            continue

        for fraction in (0.40, 0.45, 0.50, 0.55, 0.60):
            target_major = float(np.quantile(major_projection, fraction))
            band = np.abs(major_projection - target_major) <= cross_section_band_px
            if np.count_nonzero(band) < 10:
                continue
            band_minor = minor_projection[band]
            minor_low, minor_high = np.quantile(band_minor, (0.05, 0.95))
            center_minor = 0.5 * (minor_low + minor_high)
            center_xy = (
                centroid_xy
                + major_axis * target_major
                + minor_axis * center_minor
            )
            center_u = int(round(x_min + center_xy[0]))
            center_v = int(round(y_min + center_xy[1]))
            center_values = _patch_values(depth, center_u, center_v, patch_radius)
            patch_area = (2 * patch_radius + 1) ** 2
            patch_valid_ratio = center_values.size / patch_area
            if center_values.size == 0 or patch_valid_ratio < min_patch_valid_ratio:
                continue
            center_depth_mm = float(np.median(center_values))
            patch_range_mm = float(np.ptp(center_values))
            if (
                center_depth_mm < min_depth_mm
                or center_depth_mm > max_depth_mm
                or patch_range_mm > max_patch_depth_range_mm
            ):
                continue

            width_px = float(minor_high - minor_low)
            width_m = width_px * center_depth_mm / (1000.0 * focal_px)
            if width_m < min_grasp_width_m or width_m > max_grasp_width_m:
                continue
            boundary_low_xy = centroid_xy + major_axis * target_major + minor_axis * minor_low
            boundary_high_xy = centroid_xy + major_axis * target_major + minor_axis * minor_high
            outside_low_xy = boundary_low_xy - minor_axis * side_sample_offset_px
            outside_high_xy = boundary_high_xy + minor_axis * side_sample_offset_px
            outside_depths = []
            outside_pixels = []
            for outside_xy in (outside_low_xy, outside_high_xy):
                outside_u = int(round(x_min + outside_xy[0]))
                outside_v = int(round(y_min + outside_xy[1]))
                values = _patch_values(depth, outside_u, outside_v, 2)
                if values.size == 0:
                    outside_depths.append(None)
                else:
                    outside_depths.append(float(np.median(values)))
                outside_pixels.append([outside_u, outside_v])
            side_jumps_mm = [
                (
                    value - center_depth_mm
                    if value is not None
                    else max_depth_mm + min_side_jump_mm - center_depth_mm
                )
                for value in outside_depths
            ]
            if any(jump < min_side_jump_mm for jump in side_jumps_mm):
                continue

            score = (
                min(side_jumps_mm) / 100.0
                + elongation
                - patch_range_mm / 100.0
                - abs(fraction - 0.50)
            )
            candidates.append(
                {
                    "score": float(score),
                    "pixel_uv": [center_u, center_v],
                    "side_pixels_uv": outside_pixels,
                    "depth_m": center_depth_mm / 1000.0,
                    "axis": "pca_major_axis",
                    "axis_uv": major_axis.tolist(),
                    "span_px": int(round(width_px)),
                    "width_m": float(width_m),
                    "vertical_span_px": int(round(major_span_px)),
                    "vertical_length_m": float(
                        major_span_px * center_depth_mm / (1000.0 * focal_px)
                    ),
                    "side_jumps_mm": [float(value) for value in side_jumps_mm],
                    "side_depths_mm": outside_depths,
                    "patch_valid_ratio": float(patch_valid_ratio),
                    "patch_depth_range_mm": patch_range_mm,
                    "component_area_px": int(rows.size),
                    "elongation_ratio": float(elongation),
                    "roi_xyxy": [x_min, y_min, x_max, y_max],
                    "_component_mask": labels == component_id,
                    "_major_endpoints_uv": [
                        [
                            int(round(x_min + centroid_xy[0] + major_axis[0] * np.min(major_projection))),
                            int(round(y_min + centroid_xy[1] + major_axis[1] * np.min(major_projection))),
                        ],
                        [
                            int(round(x_min + centroid_xy[0] + major_axis[0] * np.max(major_projection))),
                            int(round(y_min + centroid_xy[1] + major_axis[1] * np.max(major_projection))),
                        ],
                    ],
                }
            )

    if not candidates:
        raise RuntimeError("指定侧ROI内没有深度、尺寸和两侧突变均合格的扶手")
    return max(candidates, key=lambda item: item["score"])


def recognize_chair_grasp(
    station: Any,
    arm: str,
    deprojector: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """采集指定侧腕RGB-D，返回保持当前TCP方向的扶手抓取Pose。"""
    if arm not in {"left", "right"}:
        raise ValueError("arm 必须是 'left' 或 'right'")
    config = settings.TASK["chair_grasp"]
    camera_config = settings.LEFT_CAMERA if arm == "left" else settings.RIGHT_CAMERA
    color, depth_mm = read_wrist_camera(station, camera_config)
    static_mask = gripper_mask(depth_mm.shape)
    detection_depth_mm = depth_mm.copy()
    detection_depth_mm[static_mask] = 0
    roi = tuple(config[f"{arm}_roi_normalized_xyxy"])
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{capture_id}_{arm}"
    height, width = depth_mm.shape
    roi_xyxy_input = [
        max(0, min(width - 1, int(round(roi[0] * width)))),
        max(0, min(height - 1, int(round(roi[1] * height)))),
        min(width, int(round(roi[2] * width))),
        min(height, int(round(roi[3] * height))),
    ]
    roi_input = color.copy()
    roi_input[static_mask] = (255, 0, 255)
    cv2.rectangle(
        roi_input,
        tuple(roi_xyxy_input[:2]),
        tuple(roi_xyxy_input[2:]),
        (0, 255, 255),
        2,
    )
    raw_output_images = {
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
        "depth_foreground": (
            output_dir / f"{prefix}_depth_foreground.png",
            np.uint8(
                (detection_depth_mm >= int(config["min_depth_mm"]))
                & (detection_depth_mm <= int(config["max_depth_mm"]))
                & (detection_depth_mm != 65535)
            )
            * 255,
        ),
        "roi_input": (output_dir / f"{prefix}_roi_input.jpg", roi_input),
    }
    for path, image in raw_output_images.values():
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"{arm}腕转椅原始诊断图保存失败：{path}")
    detection = detect_armrest_grasp(
        detection_depth_mm,
        roi,
        min_depth_mm=int(config["min_depth_mm"]),
        max_depth_mm=int(config["max_depth_mm"]),
        close_kernel_px=int(config["close_kernel_px"]),
        min_component_area_px=int(config["min_component_area_px"]),
        min_major_span_px=int(config["min_major_span_px"]),
        min_elongation_ratio=float(config["min_elongation_ratio"]),
        cross_section_band_px=int(config["cross_section_band_px"]),
        side_sample_offset_px=int(config["side_sample_offset_px"]),
        min_side_jump_mm=float(config["min_side_jump_mm"]),
        min_grasp_width_m=float(config["min_grasp_width_m"]),
        max_grasp_width_m=float(config["max_grasp_width_m"]),
        stability_patch_px=int(config["stability_patch_px"]),
        min_patch_valid_ratio=float(config["min_patch_valid_ratio"]),
        max_patch_depth_range_mm=float(config["max_patch_depth_range_mm"]),
        camera_fx_px=float(camera_config["camera_matrix"][0][0]),
        camera_fy_px=float(camera_config["camera_matrix"][1][1]),
    )
    component_mask = detection.pop("_component_mask")
    major_endpoints_uv = detection.pop("_major_endpoints_uv")
    u, v = detection["pixel_uv"]
    target_pose = np.asarray(
        deprojector.deproject(
            [float(u), float(v), detection["depth_m"]],
            face_target=False,
        ),
        dtype=np.float64,
    ).copy()
    target_pose[0] += float(config["grasp_target_x_offset_m"])

    roi_xyxy = detection["roi_xyxy"]
    armrest_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
    armrest_mask[
        roi_xyxy[1] : roi_xyxy[3],
        roi_xyxy[0] : roi_xyxy[2],
    ][component_mask] = 255
    annotated = roi_input.copy()
    cv2.line(annotated, tuple(major_endpoints_uv[0]), tuple(major_endpoints_uv[1]), (255, 0, 0), 3)
    cv2.line(annotated, tuple(detection["side_pixels_uv"][0]), tuple(detection["side_pixels_uv"][1]), (0, 0, 255), 3)
    cv2.circle(annotated, (u, v), 9, (0, 255, 0), -1)
    output_images = {
        "rgb": (output_dir / f"{prefix}_rgb.jpg", color),
        "depth": (output_dir / f"{prefix}_depth.png", depth_mm),
        "gripper_mask": (output_dir / f"{prefix}_gripper_mask.png", np.uint8(static_mask) * 255),
        "depth_masked": (output_dir / f"{prefix}_depth_masked.png", detection_depth_mm),
        "armrest_mask": (output_dir / f"{prefix}_armrest_mask.png", armrest_mask),
        "roi_input": (output_dir / f"{prefix}_roi_input.jpg", roi_input),
        "annotated": (output_dir / f"{prefix}_annotated.jpg", annotated),
    }
    for path, image in output_images.values():
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"{arm}腕转椅识别图保存失败：{path}")
    detection["target_pose_waist_xyz_xyzw"] = target_pose
    detection["gripper_mask_pixels"] = int(np.count_nonzero(static_mask))
    detection["output_files"] = {
        name: str(path) for name, (path, _) in output_images.items()
    }
    return detection
