"""每个 cluster 的几何特征提取 (供分类和抓取用)

关键: 用 2D minAreaRect (在"地面平面"内) + 高度 作为主尺寸.
比 PCA 更稳: PCA 对方块状 + 稀疏点云会翻转/拉长; minAreaRect 直接找最紧矩形.
"""
import numpy as np
import cv2
from .geometry import pca_axes, solid_ratio


def _make_ground_frame(plane):
    """构造地面平面上的正交基
    Returns:
        origin: (3,) 平面上一个原点 (垂足)
        u_axis, v_axis: (3,) 平面内两个正交单位向量
        n_axis: (3,) 平面法向量 (向上)
    """
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n = n / np.linalg.norm(n)
    # 找一个不与 n 平行的向量
    tmp = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = tmp - np.dot(tmp, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    origin = -d * n   # (0,0,0) 到平面的最近点
    return origin, u, v, n


def project_to_ground_frame(points, origin, u_axis, v_axis, n_axis):
    """3D 点云 -> (u, v, h) 地面坐标 (u,v = 地面 2D, h = 离地高度)"""
    rel = points - origin
    u = rel @ u_axis
    v = rel @ v_axis
    h = rel @ n_axis
    return np.stack([u, v, h], axis=1)


def compute_features(cluster, plane):
    """对一个 cluster 计算分类需要的特征"""
    pts = cluster["points"]

    # 地面坐标 (uv 平面 + 高度)
    origin, u_axis, v_axis, n_axis = _make_ground_frame(plane)
    uvh = project_to_ground_frame(pts, origin, u_axis, v_axis, n_axis)
    uv = uvh[:, :2].astype(np.float32)
    h = uvh[:, 2]

    # 2D minAreaRect 找地面上的最紧矩形
    rect = cv2.minAreaRect(uv)
    (cx, cy), (w, h_rect), angle = rect
    footprint_long = float(max(w, h_rect))
    footprint_short = float(min(w, h_rect))
    footprint_area = footprint_long * footprint_short
    aspect_lonshort = footprint_long / max(footprint_short, 1e-6)

    # 高度
    height_abs = float(h.max())
    height_bottom = float(h.min())

    # 三主尺寸 (地面世界的稳定量): dim_max 是 3 者中最大, 排序
    dims_raw = [footprint_long, footprint_short, height_abs]
    dims_sorted = sorted(dims_raw, reverse=True)
    d1, d2, d3 = dims_sorted

    # 保留 PCA 供抓取标注用 (抓取需要主轴方向)
    axes_pca, extents_pca, center_pca = pca_axes(pts)

    # 更新 cluster
    cluster["footprint_long"] = footprint_long
    cluster["footprint_short"] = footprint_short
    cluster["footprint_area"] = footprint_area
    cluster["footprint_angle_deg"] = float(angle)
    cluster["footprint_center_uv"] = (float(cx), float(cy))
    cluster["footprint_aspect"] = float(aspect_lonshort)

    cluster["height_abs"] = height_abs
    cluster["height_bottom"] = height_bottom

    # 主要分类用的三维: sorted (long, short, height)
    cluster["dims"] = (d1, d2, d3)
    cluster["dim_long"] = d1
    cluster["dim_mid"] = d2
    cluster["dim_short"] = d3

    # 供 debug 用
    cluster["pca_axes"] = axes_pca
    cluster["pca_extents"] = extents_pca
    cluster["pca_center"] = center_pca

    # 保留 solid_ratio 作 debug
    cluster["solid_ratio"] = solid_ratio(pts, voxel_size=0.02)

    # 供 classifier 用
    cluster["aspect_12"] = d1 / max(d2, 1e-6)
    cluster["aspect_23"] = d2 / max(d3, 1e-6)
    # aspect 只依据 footprint 判"圆" (垃圾桶用)
    cluster["footprint_is_round"] = aspect_lonshort < 1.20

    # 地面基
    cluster["_ground_frame"] = (origin, u_axis, v_axis, n_axis)

    return cluster
