"""在YOLO窗把手框内用深度簇确定抓取深度。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HandleCenterDetectionV3:
    """YOLO框内的目标像素、稳健深度簇和诊断掩膜。"""

    uv_depth_m: npt.NDArray[np.float64]
    mask: npt.NDArray[np.uint8]
    main_length_m: float
    cluster_center_m: float
    cluster_pixel_count: int
    roi_valid_depth_pixels: int


def detect_handle_depth_in_bbox_v3(
    depth_m: npt.ArrayLike,
    bbox_xyxy: npt.ArrayLike,
    *,
    fy: float,
) -> HandleCenterDetectionV3:
    """返回YOLO框中心偏下像素及框内最近的稳定深度簇。

    ``depth_m`` 为H×W米制深度图，``bbox_xyxy`` 为原图像素坐标。
    抓取像素取检测框水平中心、从上到下60%处；深度取框内0.08～0.60 m
    有效像素中最近且至少占ROI 0.2%的25 mm深度簇中位数。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m 必须是H×W米制深度图")
    x_min, y_min, x_max, y_max = (
        float(value) for value in np.asarray(bbox_xyxy).reshape(4)
    )
    height, width = depth.shape
    x0 = max(0, min(width - 1, int(np.floor(x_min))))
    y0 = max(0, min(height - 1, int(np.floor(y_min))))
    x1 = max(x0 + 1, min(width, int(np.ceil(x_max))))
    y1 = max(y0 + 1, min(height, int(np.ceil(y_max))))

    roi = depth[y0:y1, x0:x1]
    valid = np.isfinite(roi) & (roi >= 0.08) & (roi <= 0.60)
    valid_depths = roi[valid]
    minimum_cluster_pixels = max(20, int(np.ceil(roi.size * 0.002)))
    if valid_depths.size < minimum_cluster_pixels:
        raise RuntimeError("YOLO窗把手框内有效近场深度像素不足")

    bin_edges = np.arange(0.08, 0.605, 0.005, dtype=np.float32)
    histogram, _ = np.histogram(valid_depths, bins=bin_edges)
    band_counts = np.convolve(histogram, np.ones(5, dtype=np.int32), mode="same")
    stable_bins = np.flatnonzero(band_counts >= minimum_cluster_pixels)
    if stable_bins.size == 0:
        raise RuntimeError("YOLO窗把手框内没有稳定深度簇")

    cluster_index = int(stable_bins[0])
    cluster_center_m = float(
        0.5 * (bin_edges[cluster_index] + bin_edges[cluster_index + 1])
    )
    cluster = valid & (np.abs(roi - cluster_center_m) <= 0.0125)
    cluster_depths = roi[cluster]
    if cluster_depths.size < minimum_cluster_pixels:
        raise RuntimeError("YOLO窗把手框内最近深度簇像素不足")

    target_depth_m = float(np.median(cluster_depths))
    u = 0.5 * (x0 + x1 - 1)
    v = y0 + 0.60 * (y1 - y0 - 1)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1][cluster] = 255
    return HandleCenterDetectionV3(
        uv_depth_m=np.asarray([u, v, target_depth_m], dtype=np.float64),
        mask=mask,
        main_length_m=float((y1 - y0) * target_depth_m / fy),
        cluster_center_m=cluster_center_m,
        cluster_pixel_count=int(cluster_depths.size),
        roi_valid_depth_pixels=int(valid_depths.size),
    )
