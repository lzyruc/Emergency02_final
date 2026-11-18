"""固定观察位下，从米制深度图提取最近的窗户把手中心。

最小调用::

    detection = detect_handle_center(depth_m, fx=658.57, fy=658.10)
    print(detection.uv_depth_m)

输入深度图形状为 ``H×W``，单位米；输出为 ``[u, v, depth_m]``。
该模块只做感知计算，不读取相机、机器人状态或发送控制命令。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HandleCenterDetection:
    """把手中心像素、中心邻域深度和被选中的连通区域。"""

    uv_depth_m: npt.NDArray[np.float64]
    mask: npt.NDArray[np.uint8]
    area_m2: float
    main_length_m: float
    seed_depth_m: float
    depth_cluster_attempt: int


def detect_handle_center(
    depth_m: npt.ArrayLike,
    *,
    fx: float,
    fy: float,
    search_farther_depth_clusters: bool = False,
) -> HandleCenterDetection:
    """按物理尺寸筛选把手；可在最近簇失败后继续检查更远深度簇。

    ``search_farther_depth_clusters=False`` 保持原行为，只检查最近深度簇；设为
    ``True`` 时最多向后检查八个深度簇，并限制在相机前方 0.6 m 内。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m 必须是H×W米制深度图")

    valid = np.isfinite(depth) & (depth > 0.0)
    height, width = depth.shape
    valid_depths = depth[valid]
    if valid_depths.size < 100:
        raise RuntimeError("整张深度图有效深度像素不足")

    farther_than_m = -np.inf
    attempt_count = 8 if search_farther_depth_clusters else 1
    for attempt in range(1, attempt_count + 1):
        cluster_depths = valid_depths[valid_depths > farther_than_m]
        if cluster_depths.size < 50:
            break
        seed_depth_m = float(
            np.median(np.partition(cluster_depths, 49)[:50])
        )
        if search_farther_depth_clusters and seed_depth_m > 0.6:
            break
        near_seed_depth = valid & (np.abs(depth - seed_depth_m) < 0.05)
        candidate = np.uint8(near_seed_depth) * 255
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            np.ones((15, 15), dtype=np.uint8),
            iterations=2,
        )
        candidate = cv2.dilate(
            candidate,
            np.ones((3, 3), dtype=np.uint8),
            iterations=4,
        )

        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        best: tuple[
            float,
            int,
            float,
            float,
            npt.NDArray[np.float64],
        ] | None = None
        for label in range(1, label_count):
            component = labels == label
            component_depths = depth[component & near_seed_depth]
            if component_depths.size < 10:
                continue
            mean_depth_m = float(np.mean(component_depths))
            area_m2 = float(stats[label, cv2.CC_STAT_AREA]) * (
                mean_depth_m / fx
            ) * (mean_depth_m / fy)

            ys, xs = np.where(component)
            if xs.size < 20:
                continue
            points = np.column_stack((xs, ys)).astype(np.float64)
            center_uv = points.mean(axis=0)
            covariance = np.cov((points - center_uv).T)
            _, eigenvectors = np.linalg.eigh(covariance)
            projection = (points - center_uv) @ eigenvectors[:, 1]
            main_length_m = float(projection.max() - projection.min()) * (
                mean_depth_m / fx
            )
            if not (
                0.002 <= area_m2 <= 0.0055
                and 0.08 <= main_length_m <= 0.18
            ):
                continue
            if best is None or mean_depth_m < best[0]:
                best = (
                    mean_depth_m,
                    label,
                    area_m2,
                    main_length_m,
                    center_uv,
                )

        if best is not None:
            _, label, area_m2, main_length_m, center_uv = best
            u, v = center_uv
            center_x = int(round(u))
            center_y = int(round(v))
            radius = 10
            y0 = max(0, center_y - radius)
            y1 = min(height, center_y + radius + 1)
            x0 = max(0, center_x - radius)
            x1 = min(width, center_x + radius + 1)
            component_mask = labels == label
            local_valid = (
                component_mask[y0:y1, x0:x1]
                & near_seed_depth[y0:y1, x0:x1]
            )
            local_depths = depth[y0:y1, x0:x1][local_valid]
            if local_depths.size == 0:
                local_depths = depth[component_mask & near_seed_depth]
            center_depth_m = float(np.median(local_depths))
            return HandleCenterDetection(
                uv_depth_m=np.asarray(
                    [u, v, center_depth_m],
                    dtype=np.float64,
                ),
                mask=np.uint8(component_mask) * 255,
                area_m2=area_m2,
                main_length_m=main_length_m,
                seed_depth_m=seed_depth_m,
                depth_cluster_attempt=attempt,
            )

        farther_than_m = seed_depth_m + 0.05

    raise RuntimeError("整张深度图没有符合比赛把手尺寸的连通区域")
