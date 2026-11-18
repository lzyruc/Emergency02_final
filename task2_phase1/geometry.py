"""基础几何运算"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import config


def depth_to_pointcloud(depth_mm, intr):
    """深度图 (uint16 mm) -> 点云 (N, 3) 米 + 像素索引 (N, 2)

    Returns:
        pts:    (N, 3) 3D 点 [X, Y, Z] 相机坐标系, 米
        pix_uv: (N, 2) 每个点对应的像素 (u, v)
    """
    h, w = depth_mm.shape
    v_idx, u_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    valid = (depth_mm > 0) & (depth_mm < config.MAX_DEPTH_MM)
    z = depth_mm[valid].astype(np.float32) / 1000.0
    x = (u_idx[valid].astype(np.float32) - intr["cx"]) * z / intr["fx"]
    y = (v_idx[valid].astype(np.float32) - intr["cy"]) * z / intr["fy"]
    pts = np.stack([x, y, z], axis=1)
    pix_uv = np.stack([u_idx[valid], v_idx[valid]], axis=1).astype(np.int32)
    return pts, pix_uv


def project_to_image(pts_3d, intr):
    """3D 点 (N,3) -> 图像坐标 (N,2)"""
    z = np.clip(pts_3d[:, 2], 1e-6, None)
    u = pts_3d[:, 0] * intr["fx"] / z + intr["cx"]
    v = pts_3d[:, 1] * intr["fy"] / z + intr["cy"]
    return np.stack([u, v], axis=1)


def point_to_plane_distance(points, plane):
    """点到平面的有符号距离 (平面法向量方向为正)"""
    a, b, c, d = plane
    return a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d


def pca_axes(points):
    """对点云做 PCA, 返回三个主轴 (3,3) + 每个方向上的跨度 (3,)

    Returns:
        axes:    (3, 3) 每一行是一个主方向 (单位向量), 从主到次
        extents: (3,) 每个方向上点云的最大-最小跨度 (m)
        center:  (3,) 点云中心
    """
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh 返回升序,反转成降序
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]  # 列是主向量

    # 投影计算跨度
    proj = centered @ eigvecs
    extents = proj.max(axis=0) - proj.min(axis=0)
    # 按 extents 排序 (保证从大到小)
    idx = np.argsort(extents)[::-1]
    axes = eigvecs[:, idx].T   # 转为 row-major
    extents = extents[idx]
    return axes, extents, center


def convex_hull_volume(points):
    """凸包体积 (m³). 如果点太少或退化返回 0."""
    if len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except QhullError:
        return 0.0


def solid_ratio(points, voxel_size=0.02):
    """实占体积 / 凸包体积 (体现物体的"实心度")

    实占体积 = 点云占据的体素数 * voxel³
    值域: [0, 1]. 1 = 实心, 接近 0 = 中空 (但深度只看到表面,所以实际远 <1)
    """
    hull_vol = convex_hull_volume(points)
    if hull_vol < 1e-6:
        return 0.0
    # 体素化: 独立体素数 = 占据体积
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels = np.unique(voxel_idx, axis=0)
    occupied_vol = len(unique_voxels) * (voxel_size ** 3)
    return float(occupied_vol / hull_vol)
