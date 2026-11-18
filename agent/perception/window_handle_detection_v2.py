"""用局部双侧深度突出量提取细长窗户把手。

最小调用::

    detection = detect_handle_center_v2(depth_m, fx=658.57, fy=658.10)
    print(detection.uv_depth_m)

输入为已经屏蔽夹爪的 ``H×W`` 米制深度图；输出像素为 ``[u, v, depth_m]``。
该模块只做离线感知计算，不读取相机或控制机器人。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HandleCenterDetectionV2:
    """把手中心、原始候选掩膜及用于离线调参的几何诊断。"""

    uv_depth_m: npt.NDArray[np.float64]
    mask: npt.NDArray[np.uint8]
    main_length_m: float
    minor_length_m: float
    aspect_ratio: float
    prominence_m: float
    score: float


def _horizontal_band_mean(
    depth_integral: npt.NDArray[np.float32],
    count_integral: npt.NDArray[np.int32],
    start_offsets: npt.NDArray[np.int32],
    end_offsets: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """返回每行多个水平区间的有效深度均值和像素数。"""
    sums = depth_integral[:, end_offsets] - depth_integral[:, start_offsets]
    counts = count_integral[:, end_offsets] - count_integral[:, start_offsets]
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float32),
        where=counts > 0,
    )
    return means, counts


def _select_candidate(
    depth: npt.NDArray[np.float32],
    prominence: npt.NDArray[np.float32],
    *,
    fx: float,
    fy: float,
) -> tuple[
    float,
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    float,
    float,
    float,
    float,
] | None:
    """从双侧深度突出掩膜中选择最佳细长连通域。"""
    height, width = depth.shape
    candidate = np.uint8(prominence > 0.0) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((9, 3), dtype=np.uint8),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    best = None
    for label in range(1, label_count):
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if (
            x0 <= 8
            or y0 <= 8
            or x0 + component_width >= width - 8
            or y0 + component_height >= height - 8
        ):
            continue

        component = labels == label
        ys, xs = np.where(component)
        original = component & (prominence > 0.0)
        if xs.size < 20 or np.count_nonzero(original) < 12:
            continue
        component_depth_m = float(np.median(depth[original]))
        points = np.column_stack((xs, ys)).astype(np.float64)
        center_uv = points.mean(axis=0)
        covariance = np.cov((points - center_uv).T)
        _, eigenvectors = np.linalg.eigh(covariance)
        main_axis = eigenvectors[:, 1]
        main_projection = (points - center_uv) @ main_axis
        minor_projection = (points - center_uv) @ eigenvectors[:, 0]
        pixel_to_m = component_depth_m / ((fx + fy) * 0.5)
        main_length_m = float(
            main_projection.max() - main_projection.min()
        ) * pixel_to_m
        minor_length_m = max(
            float(minor_projection.max() - minor_projection.min())
            * pixel_to_m,
            pixel_to_m,
        )
        aspect_ratio = main_length_m / minor_length_m
        verticality = abs(float(main_axis[1]))
        prominence_m = float(np.median(prominence[original]))
        if not (
            0.045 <= main_length_m <= 0.200
            and minor_length_m <= 0.055
            and aspect_ratio >= 1.8
            and verticality >= 0.75
        ):
            continue

        score = (
            prominence_m
            * min(aspect_ratio, 8.0)
            * verticality
            * min(main_length_m / 0.08, 1.5)
        )
        if best is None or score > best[0]:
            best = (
                score,
                np.uint8(component) * 255,
                center_uv,
                component_depth_m,
                main_length_m,
                minor_length_m,
                prominence_m,
            )
    return best


def detect_handle_center_v2(
    depth_m: npt.ArrayLike,
    *,
    fx: float,
    fy: float,
) -> HandleCenterDetectionV2:
    """检测两侧都有更远背景的细长竖向把手。

    算法只考虑 0.08～0.60 m 的有效深度。先要求两侧各6个有效像素且突出
    8～80 mm；完全失败后才回退到两侧各4个有效像素且突出5～80 mm。
    两级使用相同的物理尺寸和图像边界约束。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m 必须是H×W米制深度图")

    height, width = depth.shape
    valid = (
        np.isfinite(depth)
        & (depth >= 0.08)
        & (depth <= 0.60)
    )
    if np.count_nonzero(valid) < 100:
        raise RuntimeError("整张深度图有效近场深度像素不足")

    x = np.arange(width, dtype=np.int32)
    depth_integral = np.pad(
        np.cumsum(np.where(valid, depth, 0.0), axis=1),
        ((0, 0), (1, 0)),
    )
    count_integral = np.pad(
        np.cumsum(valid.astype(np.int32), axis=1),
        ((0, 0), (1, 0)),
    )
    strict_prominence = np.zeros_like(depth, dtype=np.float32)
    relaxed_prominence = np.zeros_like(depth, dtype=np.float32)
    band_width_px = 12
    for gap_px in (8, 16, 24, 36, 52):
        left_start = np.clip(x - gap_px - band_width_px, 0, width)
        left_end = np.clip(x - gap_px, 0, width)
        right_start = np.clip(x + gap_px, 0, width)
        right_end = np.clip(x + gap_px + band_width_px, 0, width)
        left_mean, left_count = _horizontal_band_mean(
            depth_integral,
            count_integral,
            left_start,
            left_end,
        )
        right_mean, right_count = _horizontal_band_mean(
            depth_integral,
            count_integral,
            right_start,
            right_end,
        )
        left_difference = left_mean - depth
        right_difference = right_mean - depth
        prominence = np.minimum(left_difference, right_difference)
        strict = (
            valid
            & (left_count >= 6)
            & (right_count >= 6)
            & (prominence >= 0.008)
            & (prominence <= 0.080)
        )
        strict_prominence = np.maximum(
            strict_prominence,
            np.where(strict, prominence, 0.0),
        )
        relaxed = (
            valid
            & (left_count >= 4)
            & (right_count >= 4)
            & (prominence >= 0.005)
            & (prominence <= 0.080)
        )
        relaxed_prominence = np.maximum(
            relaxed_prominence,
            np.where(relaxed, prominence, 0.0),
        )

    best = _select_candidate(
        depth,
        strict_prominence,
        fx=fx,
        fy=fy,
    )
    if best is None:
        best = _select_candidate(
            depth,
            relaxed_prominence,
            fx=fx,
            fy=fy,
        )

    if best is None:
        raise RuntimeError("没有找到具有局部深度突出的细长竖向把手")

    (
        score,
        mask,
        center_uv,
        component_depth_m,
        main_length_m,
        minor_length_m,
        prominence_m,
    ) = best
    component_y = np.flatnonzero(np.any(mask > 0, axis=1))
    target_v = float(
        component_y[0] + 0.75 * (component_y[-1] - component_y[0])
    )
    return HandleCenterDetectionV2(
        uv_depth_m=np.asarray(
            [center_uv[0], target_v, component_depth_m],
            dtype=np.float64,
        ),
        mask=mask,
        main_length_m=main_length_m,
        minor_length_m=minor_length_m,
        aspect_ratio=main_length_m / minor_length_m,
        prominence_m=prominence_m,
        score=score,
    )
