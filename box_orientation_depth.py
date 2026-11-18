"""基于头部RGB-D水平物理跨度判断横放或竖放纸箱。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _fill_short_false_gaps(mask: np.ndarray, max_gap_px: int) -> np.ndarray:
    """只填补两侧均为True、长度不超过阈值的一维内部空洞。"""
    filled = np.asarray(mask, dtype=bool).copy()
    false_indices = np.flatnonzero(~filled)
    if false_indices.size == 0:
        return filled
    split = np.flatnonzero(np.diff(false_indices) > 1) + 1
    for group in np.split(false_indices, split):
        if (
            group.size <= max_gap_px
            and group[0] > 0
            and group[-1] + 1 < filled.size
            and filled[group[0] - 1]
            and filled[group[-1] + 1]
        ):
            filled[group] = True
    return filled


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回一维mask中所有左闭右开True区间。"""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _depth_continuous_runs(
    row_depth: np.ndarray,
    max_gap_px: int,
    max_depth_step_mm: int,
) -> list[np.ndarray]:
    """返回每个深度连续段的像素偏移，允许少量无效深度空洞。"""
    depth = np.asarray(row_depth)
    valid = (
        (depth > 0)
        & (depth != 65535)
        & np.isfinite(depth)
    )
    offsets = np.flatnonzero(valid)
    if offsets.size == 0:
        return []
    split = np.flatnonzero(
        (np.diff(offsets) > max_gap_px + 1)
        | (
            np.abs(
                depth[offsets[1:]].astype(np.float64)
                - depth[offsets[:-1]].astype(np.float64)
            )
            > max_depth_step_mm
        )
    ) + 1
    return [group for group in np.split(offsets, split) if group.size >= 2]


def detect_box_orientation_from_horizontal_span(
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
    deprojector: Any,
    *,
    min_height_m: float = 0.15,
    horizontal_span_threshold_m: float = 0.40,
    scan_row_ratios: tuple[float, ...] = (
        0.38,
        0.42,
        0.46,
        0.50,
        0.54,
        0.58,
        0.62,
    ),
    max_depth_step_mm: int = 80,
    max_depth_gap_px: int = 8,
) -> dict[str, Any]:
    """在YOLO框中间7条横线上测量高于地面的连续物理跨度。

    像素和深度先批量反投影到 ``WAIST_YAW_LINK``，再批量转换到
    ``BASE_FOOTPRINT``。每行只剔除无效深度，允许短空洞，按相邻
    深度突变分段，再选取
    ``base z > min_height_m`` 的最长连续段。有效行跨度中位数不小于
    ``horizontal_span_threshold_m`` 判为 ``horizontal``，否则判为
    ``vertical``。没有可用跨度时按 0 m 处理，仍判为 ``vertical``。
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("depth_mm 必须是H×W数组")
    image_height, image_width = depth.shape
    x_min = max(0, min(image_width - 1, int(math.floor(bbox_xyxy[0]))))
    y_min = max(0, min(image_height - 1, int(math.floor(bbox_xyxy[1]))))
    x_max = max(x_min + 1, min(image_width, int(math.ceil(bbox_xyxy[2]))))
    y_max = max(y_min + 1, min(image_height, int(math.ceil(bbox_xyxy[3]))))
    box_width = x_max - x_min
    box_height = y_max - y_min

    row_results = []
    for ratio in scan_row_ratios:
        row_v = min(y_max - 1, y_min + int(round(ratio * box_height)))
        row_depth = depth[row_v, x_min:x_max]
        best = None
        for sample_offsets in _depth_continuous_runs(
            row_depth,
            max_depth_gap_px,
            max_depth_step_mm,
        ):
            samples = np.column_stack(
                (
                    sample_offsets + x_min,
                    np.full(sample_offsets.size, row_v),
                    row_depth[sample_offsets].astype(np.float64) / 1000.0,
                )
            )
            points_waist = deprojector.deproject_many(
                samples,
                face_target=False,
            )[:, :3]
            points_base = deprojector.transform_many_to_ground(points_waist)
            above_ground = points_base[:, 2] > min_height_m
            above_ground = _fill_short_false_gaps(
                above_ground,
                max_depth_gap_px,
            )
            for high_start, high_end in _true_runs(above_ground):
                valid_indices = np.flatnonzero(
                    points_base[high_start:high_end, 2] > min_height_m
                ) + high_start
                if valid_indices.size < 2:
                    continue
                left = points_base[valid_indices[0]]
                right = points_base[valid_indices[-1]]
                span_m = float(np.linalg.norm(right[:2] - left[:2]))
                candidate = {
                    "row_v": row_v,
                    "span_m": span_m,
                    "left_uv": [int(samples[valid_indices[0], 0]), row_v],
                    "right_uv": [int(samples[valid_indices[-1], 0]), row_v],
                    "point_count": int(valid_indices.size),
                }
                if best is None or span_m > best["span_m"]:
                    best = candidate
        if best is not None:
            row_results.append(best)

    spans = [result["span_m"] for result in row_results]
    span_m = float(np.median(spans)) if spans else 0.0
    orientation = (
        "horizontal"
        if span_m >= horizontal_span_threshold_m
        else "vertical"
    )
    reason = (
        "horizontal_physical_span"
        if spans
        else "no_valid_above_ground_span"
    )
    return {
        "orientation": orientation,
        "reason": reason,
        "span_m": span_m,
        "row_spans_m": spans,
        "rows": row_results,
        "valid_row_count": len(row_results),
        "min_height_m": float(min_height_m),
        "horizontal_span_threshold_m": float(horizontal_span_threshold_m),
    }
