#!/usr/bin/env python3
"""只读头部RGB-D，测试纸箱轮廓、长轴方向和横竖姿态分类。

运行后每次按 Enter 自动采集三帧；脚本不会创建 NavigationGraph，也不会发送
任何机器人运动 action。结果像素位于原始头部图像，角度为图像坐标系下长轴角：
0/180度接近水平，90度接近竖直。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from camera_io import read_head_camera
import settings
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


FRAME_COUNT = 3
FRAME_INTERVAL_S = 0.3
BOX_DEPTH_BEFORE_MM = 120
BOX_DEPTH_BEHIND_MM = 180
MIN_COMPONENT_AREA_RATIO = 0.02
MIN_LONG_SHORT_RATIO = 1.15
MAX_AXIS_DISTANCE_DEG = 35.0


def _depth_color(depth_mm: np.ndarray) -> np.ndarray:
    valid = (depth_mm > 0) & (depth_mm != 65535)
    color = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth_mm[valid], [2, 98])
        if high > low:
            normalized = np.clip(
                (depth_mm.astype(np.float32) - low) / (high - low),
                0.0,
                1.0,
            )
            mapped = cv2.applyColorMap(
                np.uint8((1.0 - normalized) * 255.0),
                cv2.COLORMAP_TURBO,
            )
            color[valid] = mapped[valid]
    return color


def _orientation_name(angle_deg: float, ratio: float) -> str:
    if ratio < MIN_LONG_SHORT_RATIO:
        return "uncertain"
    horizontal_distance = min(angle_deg, 180.0 - angle_deg)
    vertical_distance = abs(angle_deg - 90.0)
    if min(horizontal_distance, vertical_distance) > MAX_AXIS_DISTANCE_DEG:
        return "diagonal"
    return "horizontal" if horizontal_distance < vertical_distance else "vertical"


def detect_box_outline(
    color: np.ndarray,
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> tuple[dict[str, Any] | None, np.ndarray]:
    """在YOLO框内返回纸箱轮廓中心、长轴角、长短轴和二值mask。"""
    image_height, image_width = depth_mm.shape
    x_min = max(0, min(image_width - 1, int(math.floor(bbox_xyxy[0]))))
    y_min = max(0, min(image_height - 1, int(math.floor(bbox_xyxy[1]))))
    x_max = max(x_min + 1, min(image_width, int(math.ceil(bbox_xyxy[2]))))
    y_max = max(y_min + 1, min(image_height, int(math.ceil(bbox_xyxy[3]))))
    crop_color = color[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    crop_height, crop_width = crop_depth.shape

    center_depth = crop_depth[
        crop_height * 3 // 10 : crop_height * 7 // 10,
        crop_width * 3 // 10 : crop_width * 7 // 10,
    ]
    center_values = center_depth[
        (center_depth > 0) & (center_depth != 65535)
    ]
    if center_values.size < 20:
        return None, np.zeros_like(depth_mm, dtype=np.uint8)
    reference_depth_mm = float(np.median(center_values))
    depth_foreground = (
        (crop_depth >= reference_depth_mm - BOX_DEPTH_BEFORE_MM)
        & (crop_depth <= reference_depth_mm + BOX_DEPTH_BEHIND_MM)
        & (crop_depth > 0)
        & (crop_depth != 65535)
    )

    hsv = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
    warm_cardboard = (
        (hsv[..., 0] >= 2)
        & (hsv[..., 0] <= 40)
        & (hsv[..., 1] >= 20)
        & (hsv[..., 2] >= 35)
    )
    warm_cardboard = cv2.dilate(
        warm_cardboard.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ) > 0
    warm_depth = depth_foreground & warm_cardboard
    mask_source = "warm_rgb_and_depth"
    if np.count_nonzero(warm_depth) < 0.05 * crop_depth.size:
        candidate = depth_foreground.astype(np.uint8)
        mask_source = "depth_fallback"
    else:
        candidate = warm_depth.astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=2,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )
    if component_count <= 1:
        return None, np.zeros_like(depth_mm, dtype=np.uint8)
    roi_center = np.asarray([0.5 * crop_width, 0.5 * crop_height])
    labels_by_score = sorted(
        range(1, component_count),
        key=lambda label: (
            stats[label, cv2.CC_STAT_AREA]
            - 0.25
            * np.linalg.norm(centroids[label] - roi_center)
        ),
        reverse=True,
    )
    selected_label = labels_by_score[0]
    area = int(stats[selected_label, cv2.CC_STAT_AREA])
    if area < MIN_COMPONENT_AREA_RATIO * crop_depth.size:
        return None, np.zeros_like(depth_mm, dtype=np.uint8)

    component = np.uint8(labels == selected_label) * 255
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea)
    contour[:, 0, 0] += x_min
    contour[:, 0, 1] += y_min
    rectangle = cv2.minAreaRect(contour.astype(np.float32))
    corners = cv2.boxPoints(rectangle)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    long_edge_index = int(np.argmax(lengths))
    long_vector = edges[long_edge_index]
    angle_deg = float(
        np.degrees(np.arctan2(long_vector[1], long_vector[0])) % 180.0
    )
    long_px = float(np.max(lengths))
    short_px = float(np.min(lengths))
    ratio = long_px / max(short_px, 1.0)
    center_u, center_v = rectangle[0]
    touches_image_border = (
        x_min == 0
        or y_min == 0
        or x_max == image_width
        or y_max == image_height
    )
    full_mask = np.zeros_like(depth_mm, dtype=np.uint8)
    full_mask[y_min:y_max, x_min:x_max] = component
    result = {
        "center_uv": [float(center_u), float(center_v)],
        "angle_deg_image": angle_deg,
        "orientation": _orientation_name(angle_deg, ratio),
        "long_px": long_px,
        "short_px": short_px,
        "long_short_ratio": ratio,
        "corners_uv": corners.astype(float).tolist(),
        "reference_depth_m": reference_depth_mm / 1000.0,
        "component_area_px": area,
        "mask_source": mask_source,
        "bbox_touches_image_border": touches_image_border,
    }
    return result, full_mask


def fuse_orientations(results: list[dict[str, Any]]) -> dict[str, Any]:
    """用轴向双角平均融合三帧轮廓方向。"""
    angles_rad = np.deg2rad(
        [2.0 * float(result["angle_deg_image"]) for result in results]
    )
    mean_cos = float(np.mean(np.cos(angles_rad)))
    mean_sin = float(np.mean(np.sin(angles_rad)))
    angle_deg = float(
        (0.5 * np.degrees(np.arctan2(mean_sin, mean_cos))) % 180.0
    )
    mean_ratio = float(
        np.mean([result["long_short_ratio"] for result in results])
    )
    frame_orientations = [result["orientation"] for result in results]
    orientation = _orientation_name(angle_deg, mean_ratio)
    if len(set(frame_orientations)) != 1:
        orientation = "uncertain"
    return {
        "frame_count": len(results),
        "center_uv_mean": np.mean(
            [result["center_uv"] for result in results],
            axis=0,
        ).tolist(),
        "angle_deg_image": angle_deg,
        "orientation": orientation,
        "long_short_ratio_mean": mean_ratio,
        "axis_consistency": math.hypot(mean_cos, mean_sin),
        "frame_orientations": frame_orientations,
    }


def main() -> None:
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    detector = ObstacleYolo()
    round_index = 1
    try:
        time.sleep(1.0)
        while True:
            input(
                f"\n第{round_index}轮：将纸箱放到头部相机前，按 Enter 自动采集"
                "3帧并识别轮廓方向；Ctrl+C结束："
            )
            output_dir = (
                settings.ROOT
                / "output/box_orientation"
                / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_results = []
            for frame_index in range(1, FRAME_COUNT + 1):
                color, depth_mm = read_head_camera(station)
                if color is None or depth_mm is None:
                    raise RuntimeError("未获取到头部RGB-D帧")
                prefix = output_dir / f"frame_{frame_index:02d}"
                cv2.imwrite(str(prefix.with_name(f"{prefix.name}_rgb.jpg")), color)
                cv2.imwrite(
                    str(prefix.with_name(f"{prefix.name}_depth_mm.png")),
                    depth_mm,
                )
                cv2.imwrite(
                    str(prefix.with_name(f"{prefix.name}_depth_color.jpg")),
                    _depth_color(depth_mm),
                )
                boxes = detector.detect(color, target_type="box")
                prefix.with_name(f"{prefix.name}_detections.json").write_text(
                    json.dumps(boxes, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if not boxes:
                    print(f"第{frame_index}/3帧：YOLO未识别到纸箱", flush=True)
                else:
                    detection = max(boxes, key=lambda item: item["confidence"])
                    outline, mask = detect_box_outline(
                        color,
                        depth_mm,
                        detection["bbox_xyxy"],
                    )
                    cv2.imwrite(
                        str(prefix.with_name(f"{prefix.name}_mask.png")),
                        mask,
                    )
                    annotated = color.copy()
                    x0, y0, x1, y1 = (
                        int(round(value))
                        for value in detection["bbox_xyxy"]
                    )
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    if outline is None:
                        print(
                            f"第{frame_index}/3帧：检测到纸箱但轮廓提取失败",
                            flush=True,
                        )
                    else:
                        corners = np.rint(outline["corners_uv"]).astype(np.int32)
                        cv2.polylines(annotated, [corners], True, (0, 255, 0), 3)
                        center = tuple(
                            np.rint(outline["center_uv"]).astype(int)
                        )
                        cv2.drawMarker(
                            annotated,
                            center,
                            (0, 0, 255),
                            cv2.MARKER_CROSS,
                            30,
                            3,
                        )
                        outline["confidence"] = float(detection["confidence"])
                        outline["bbox_xyxy"] = detection["bbox_xyxy"]
                        frame_results.append(outline)
                        print(
                            f"第{frame_index}/3帧："
                            f"中心=({outline['center_uv'][0]:.1f}, "
                            f"{outline['center_uv'][1]:.1f})，"
                            f"长轴角={outline['angle_deg_image']:.1f}°，"
                            f"长短比={outline['long_short_ratio']:.2f}，"
                            f"判断={outline['orientation']}，"
                            f"置信度={detection['confidence']:.3f}",
                            flush=True,
                        )
                    cv2.imwrite(
                        str(prefix.with_name(f"{prefix.name}_annotated.jpg")),
                        annotated,
                    )
                if frame_index < FRAME_COUNT:
                    time.sleep(FRAME_INTERVAL_S)

            if len(frame_results) == FRAME_COUNT:
                summary = fuse_orientations(frame_results)
                print(
                    "三帧融合："
                    f"中心=({summary['center_uv_mean'][0]:.1f}, "
                    f"{summary['center_uv_mean'][1]:.1f})，"
                    f"长轴角={summary['angle_deg_image']:.1f}°，"
                    f"方向一致性={summary['axis_consistency']:.3f}，"
                    f"最终判断={summary['orientation']}",
                    flush=True,
                )
            else:
                summary = {
                    "frame_count": len(frame_results),
                    "orientation": "uncertain",
                    "reason": "not_all_three_frames_have_box_outline",
                }
                print(
                    f"三帧中只有{len(frame_results)}帧得到轮廓，最终判断不确定",
                    flush=True,
                )
            (output_dir / "result.json").write_text(
                json.dumps(
                    {"frames": frame_results, "summary": summary},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"结果已保存：{output_dir}", flush=True)
            round_index += 1
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
