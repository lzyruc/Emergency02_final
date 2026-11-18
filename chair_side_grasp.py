"""倾斜办公转椅侧面竖框的腕部 RGB-D 抓取点识别。

算法只负责采集、筛选侧视竖框并返回 ``waist_yaw_link`` 下的目标 Pose；
机械臂和夹爪动作由任务编排层串行执行。深度单位为毫米，输出长度单位为米。
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
    """返回像素邻域内有效的毫米深度。"""
    height, width = depth_mm.shape
    patch = depth_mm[
        max(0, v - radius_px) : min(height, v + radius_px + 1),
        max(0, u - radius_px) : min(width, u + radius_px + 1),
    ]
    return patch[(patch > 0) & (patch != 65535) & np.isfinite(patch)]


def _axis_from_horizontal_deg(major_axis: np.ndarray) -> float:
    """PCA 长轴相对画面水平线的夹角，0°水平、90°竖直。"""
    return float(
        np.degrees(
            np.arctan2(abs(float(major_axis[1])), abs(float(major_axis[0])))
        )
    )


def detect_chair_side_grasp(
    depth_mm: np.ndarray,
    roi_normalized_xyxy: tuple[float, float, float, float],
    config: dict[str, Any],
    camera_fx_px: float,
    camera_fy_px: float,
) -> dict[str, Any]:
    """检测侧视椅背竖框；至少一侧存在深度跳变即可。"""
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("depth_mm 必须是 H×W 毫米深度图")
    height, width = depth.shape
    x_min = max(0, min(width - 1, int(round(roi_normalized_xyxy[0] * width))))
    y_min = max(0, min(height - 1, int(round(roi_normalized_xyxy[1] * height))))
    x_max = max(x_min + 1, min(width, int(round(roi_normalized_xyxy[2] * width))))
    y_max = max(y_min + 1, min(height, int(round(roi_normalized_xyxy[3] * height))))
    crop = depth[y_min:y_max, x_min:x_max]
    min_depth_mm = int(config["min_depth_mm"])
    max_depth_mm = int(config["max_depth_mm"])
    foreground = (
        (crop >= min_depth_mm)
        & (crop <= max_depth_mm)
        & (crop != 65535)
        & np.isfinite(crop)
    )
    kernel_px = int(config["close_kernel_px"]) | 1
    closed = cv2.morphologyEx(
        np.uint8(foreground) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_px, kernel_px),
        ),
    )
    vertical_kernel_height = (
        max(17, int(config["min_major_span_px"]) // 6) | 1
    )
    binaries = (
        closed,
        cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, vertical_kernel_height),
            ),
        ),
    )
    focal_px = 0.5 * (float(camera_fx_px) + float(camera_fy_px))
    patch_radius = int(config["stability_patch_px"]) // 2
    candidates: list[dict[str, Any]] = []

    for binary in binaries:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary
        )
        for component_id in range(1, component_count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) < int(
                config["min_component_area_px"]
            ):
                continue
            component = (labels == component_id) & (binary > 0)
            rows, columns = np.nonzero(component)
            if rows.size < int(config["min_component_area_px"]):
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
            axis_deg = _axis_from_horizontal_deg(major_axis)
            if (
                major_span_px < float(config["min_major_span_px"])
                or elongation < float(config["min_elongation_ratio"])
                or axis_deg < float(config["min_axis_from_horizontal_deg"])
            ):
                continue

            for fraction in (0.40, 0.45, 0.50, 0.55, 0.60):
                target_major = float(np.quantile(major_projection, fraction))
                band = np.abs(major_projection - target_major) <= int(
                    config["cross_section_band_px"]
                )
                if np.count_nonzero(band) < 8:
                    continue
                minor_low, minor_high = np.quantile(
                    minor_projection[band],
                    (0.05, 0.95),
                )
                center_xy = (
                    centroid_xy
                    + major_axis * target_major
                    + minor_axis * (0.5 * (minor_low + minor_high))
                )
                center_u = int(round(x_min + center_xy[0]))
                center_v = int(round(y_min + center_xy[1]))
                center_values = _patch_values(
                    depth,
                    center_u,
                    center_v,
                    patch_radius,
                )
                patch_area = (2 * patch_radius + 1) ** 2
                if (
                    center_values.size == 0
                    or center_values.size / patch_area
                    < float(config["min_patch_valid_ratio"])
                ):
                    continue
                center_depth_mm = float(np.median(center_values))
                patch_range_mm = float(np.ptp(center_values))
                if (
                    center_depth_mm < min_depth_mm
                    or center_depth_mm > max_depth_mm
                    or patch_range_mm
                    > float(config["max_patch_depth_range_mm"])
                ):
                    continue
                width_px = float(minor_high - minor_low)
                width_m = width_px * center_depth_mm / (1000.0 * focal_px)
                if not (
                    float(config["min_grasp_width_m"])
                    <= width_m
                    <= float(config["max_grasp_width_m"])
                ):
                    continue

                outside_pixels = []
                side_jumps_mm = []
                for side_name, sign in (("left", -1.0), ("right", 1.0)):
                    sample_xy = (
                        centroid_xy
                        + major_axis * target_major
                        + minor_axis
                        * (minor_low if side_name == "left" else minor_high)
                        + sign
                        * minor_axis
                        * int(config["side_sample_offset_px"])
                    )
                    sample_u = int(round(x_min + sample_xy[0]))
                    sample_v = int(round(y_min + sample_xy[1]))
                    values = _patch_values(depth, sample_u, sample_v, 2)
                    outside_pixels.append([sample_u, sample_v])
                    side_jumps_mm.append(
                        max_depth_mm
                        + float(config["min_side_jump_mm"])
                        - center_depth_mm
                        if values.size == 0
                        else float(np.median(values)) - center_depth_mm
                    )
                if all(
                    jump < float(config["min_side_jump_mm"])
                    for jump in side_jumps_mm
                ):
                    continue
                candidates.append(
                    {
                        "score": float(
                            max(side_jumps_mm) / 100.0
                            + elongation
                            + axis_deg / 45.0
                            - patch_range_mm / 100.0
                            - abs(fraction - 0.50)
                        ),
                        "pixel_uv": [center_u, center_v],
                        "side_pixels_uv": outside_pixels,
                        "depth_m": center_depth_mm / 1000.0,
                        "axis": "pca_major_axis",
                        "axis_uv": major_axis.tolist(),
                        "axis_from_horizontal_deg": axis_deg,
                        "span_px": int(round(width_px)),
                        "width_m": width_m,
                        "vertical_span_px": int(round(major_span_px)),
                        "vertical_length_m": float(
                            major_span_px
                            * center_depth_mm
                            / (1000.0 * focal_px)
                        ),
                        "side_jumps_mm": [
                            float(value) for value in side_jumps_mm
                        ],
                        "patch_valid_ratio": float(
                            center_values.size / patch_area
                        ),
                        "patch_depth_range_mm": patch_range_mm,
                        "component_area_px": int(rows.size),
                        "elongation_ratio": elongation,
                        "roi_xyxy": [x_min, y_min, x_max, y_max],
                        "grasp_mode": "side_rail",
                        "_component_mask": labels == component_id,
                        "_major_endpoints_uv": [
                            [
                                int(
                                    round(
                                        x_min
                                        + centroid_xy[0]
                                        + major_axis[0]
                                        * np.min(major_projection)
                                    )
                                ),
                                int(
                                    round(
                                        y_min
                                        + centroid_xy[1]
                                        + major_axis[1]
                                        * np.min(major_projection)
                                    )
                                ),
                            ],
                            [
                                int(
                                    round(
                                        x_min
                                        + centroid_xy[0]
                                        + major_axis[0]
                                        * np.max(major_projection)
                                    )
                                ),
                                int(
                                    round(
                                        y_min
                                        + centroid_xy[1]
                                        + major_axis[1]
                                        * np.max(major_projection)
                                    )
                                ),
                            ],
                        ],
                    }
                )
    if not candidates:
        raise RuntimeError("指定侧ROI内没有可抓取的倾斜椅背竖框")
    return max(candidates, key=lambda item: item["score"])


def recognize_chair_side_grasp(
    station: Any,
    arm: str,
    deprojector: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """采集指定腕 RGB-D，返回保持当前 TCP 方向的侧框目标 Pose。"""
    if arm not in {"left", "right"}:
        raise ValueError("arm 必须是 'left' 或 'right'")
    config = settings.TASK["chair_side_grasp"]
    camera_config = (
        settings.LEFT_CAMERA if arm == "left" else settings.RIGHT_CAMERA
    )
    color, depth_mm = read_wrist_camera(station, camera_config)
    static_mask = gripper_mask(depth_mm.shape)
    detection_depth_mm = depth_mm.copy()
    detection_depth_mm[static_mask] = 0
    roi = tuple(config[f"{arm}_roi_normalized_xyxy"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{arm}"
    height, width = depth_mm.shape
    roi_xyxy = [
        max(0, min(width - 1, int(round(roi[0] * width)))),
        max(0, min(height - 1, int(round(roi[1] * height)))),
        min(width, int(round(roi[2] * width))),
        min(height, int(round(roi[3] * height))),
    ]
    roi_input = color.copy()
    roi_input[static_mask] = (255, 0, 255)
    cv2.rectangle(
        roi_input,
        tuple(roi_xyxy[:2]),
        tuple(roi_xyxy[2:]),
        (0, 255, 255),
        2,
    )
    raw_images = {
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
        "roi_input": (output_dir / f"{prefix}_roi_input.jpg", roi_input),
    }
    for path, image in raw_images.values():
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"{arm}腕侧框原始诊断图保存失败：{path}")

    detection = detect_chair_side_grasp(
        detection_depth_mm,
        roi,
        config,
        float(camera_config["camera_matrix"][0][0]),
        float(camera_config["camera_matrix"][1][1]),
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

    side_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
    side_mask[
        roi_xyxy[1] : roi_xyxy[3],
        roi_xyxy[0] : roi_xyxy[2],
    ][component_mask] = 255
    annotated = roi_input.copy()
    cv2.line(
        annotated,
        tuple(major_endpoints_uv[0]),
        tuple(major_endpoints_uv[1]),
        (255, 0, 0),
        3,
    )
    cv2.line(
        annotated,
        tuple(detection["side_pixels_uv"][0]),
        tuple(detection["side_pixels_uv"][1]),
        (0, 0, 255),
        3,
    )
    cv2.circle(annotated, (u, v), 9, (0, 255, 0), -1)
    cv2.putText(
        annotated,
        f"side_rail {detection['axis_from_horizontal_deg']:.0f}deg",
        (u + 12, max(24, v - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    annotated_path = output_dir / f"{prefix}_annotated.jpg"
    mask_path = output_dir / f"{prefix}_side_mask.png"
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(f"{arm}腕侧框标注图保存失败：{annotated_path}")
    if not cv2.imwrite(str(mask_path), side_mask):
        raise RuntimeError(f"{arm}腕侧框mask保存失败：{mask_path}")

    edge_distance_px = float(min(u, width - 1 - u, v, height - 1 - v))
    if edge_distance_px < int(config["min_edge_dist_px"]):
        raise RuntimeError(
            f"抓取点太靠画面边缘uv={detection['pixel_uv']}，"
            f"边缘距={edge_distance_px:.1f}px"
        )
    detection["target_pose_waist_xyz_xyzw"] = target_pose
    detection["dist_to_image_center_px"] = float(
        np.hypot(u - 0.5 * (width - 1), v - 0.5 * (height - 1))
    )
    detection["dist_to_image_edge_px"] = edge_distance_px
    detection["gripper_mask_pixels"] = int(np.count_nonzero(static_mask))
    detection["arm"] = arm
    detection["output_files"] = {
        **{name: str(path) for name, (path, _) in raw_images.items()},
        "annotated": str(annotated_path),
        "side_mask": str(mask_path),
    }
    return detection
