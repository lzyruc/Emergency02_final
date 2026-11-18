"""文件筐专用 RGB-D 入口。

它绕开通用 DBSCAN 障碍物流水线，只运行文件筐所需的地面拟合、HSV 色块检测、
尺寸验证和五板自由边候选提取。这样既不受网孔导致的点云碎裂影响，也便于对
文件筐数据集做确定性的逐帧回归；整帧公开结果最多包含一个文件筐和一条
最高置信安全抓取线。
"""

from __future__ import annotations

import numpy as np

from . import config
from .basket_cv import detect_baskets
from .basket_grasp_cv import ALGORITHM_VERSION
from .geometry import depth_to_pointcloud
from .ground import fit_ground
from .pipeline import FrameResult, _v14_to_objects


def detect_basket(
    depth_mm,
    intr,
    rgb_bgr,
    ground_stride=config.BASKET_GROUND_STRIDE,
    *,
    navigation_mode=False,
):
    """识别一帧中的文件筐并返回标准 :class:`FrameResult`。

    ``ground_stride`` 只控制默认抓取模式的地面 RANSAC 均匀采样。
    ``navigation_mode=True`` 时使用更稀疏的地面采样并跳过抓取线，只保留
    文件筐区域中心供底盘接近；默认模式的抓取线仍使用完整分辨率数据。
    """
    if rgb_bgr is None:
        raise ValueError("文件筐专用识别需要与深度对齐的 RGB 图")
    if depth_mm.shape[:2] != rgb_bgr.shape[:2]:
        raise ValueError(
            f"RGB/深度尺寸不一致：rgb={rgb_bgr.shape[:2]} "
            f"depth={depth_mm.shape[:2]}"
        )
    stride = max(
        1,
        int(
            config.BASKET_NAVIGATION_GROUND_STRIDE
            if navigation_mode
            else ground_stride
        ),
    )
    points, _ = depth_to_pointcloud(depth_mm, intr)
    if len(points) < 100:
        return FrameResult(
            objects=[],
            ground_plane=(0.0, -1.0, 0.0, 0.0),
            n_raw_clusters=0,
            n_kept=0,
            debug={
                "algorithm": ALGORITHM_VERSION,
                "reason": "insufficient_depth",
            },
        )
    sampled = np.asarray(points[::stride], dtype=np.float64)
    plane, _ = fit_ground(
        sampled,
        num_iter=(
            config.BASKET_NAVIGATION_GROUND_RANSAC_ITER
            if navigation_mode
            else None
        ),
    )
    hits = detect_baskets(
        rgb_bgr,
        depth_mm,
        intr,
        plane,
        include_grasps=not navigation_mode,
    )
    objects = _v14_to_objects(hits, plane)
    return FrameResult(
        objects=objects,
        ground_plane=plane,
        n_raw_clusters=len(hits),
        n_kept=len(objects),
        debug={
            "algorithm": ALGORITHM_VERSION,
            "ground_stride": stride,
            "ground_sample_points": int(len(sampled)),
            "navigation_mode": navigation_mode,
            "semantic_contract": (
                "one frame-global top-1 basket region center for navigation"
                if navigation_mode
                else (
                    "one frame-global top-1 directly observed line from the "
                    "five allowed L-panel free contours; "
                    "never crossbars, base edges, or hole edges"
                )
            ),
        },
    )
