"""地面平面拟合"""
import numpy as np

from . import config


def _ransac_plane_numpy(points, dist_thresh, ransac_n, num_iter, seed=42):
    """纯 numpy RANSAC 平面拟合. Jetson 上 open3d 装不上时用."""
    n = len(points)
    if n < ransac_n:
        return (0., 1., 0., 0.), np.array([], dtype=int)
    rng = np.random.RandomState(seed)
    best_inliers = np.array([], dtype=int)
    best_plane = None
    for _ in range(num_iter):
        idx = rng.choice(n, ransac_n, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        # 平面法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        # 所有点到平面的距离
        dists = np.abs(points @ normal + d)
        inliers = np.where(dists < dist_thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (float(normal[0]), float(normal[1]),
                          float(normal[2]), float(d))
    if best_plane is None:
        return (0., 1., 0., 0.), np.array([], dtype=int)
    # 用所有内点再拟合一次 (least squares) 提高精度
    if len(best_inliers) >= 10:
        inl_pts = points[best_inliers]
        centroid = inl_pts.mean(axis=0)
        centered = inl_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
            d = -float(np.dot(normal, centroid))
            best_plane = (float(normal[0]), float(normal[1]),
                          float(normal[2]), d)
    return best_plane, best_inliers


def _ls_plane(P, idx, n_ref):
    """给定内点索引, 用最小二乘算唯一确定的平面 (法向量朝向对齐 n_ref)"""
    inl = P[idx]
    ctr = inl.mean(axis=0)
    X = inl - ctr
    # 3x3 协方差的最小特征向量 = 最佳拟合平面法向量.
    # 比对整个大矩阵做 SVD 快很多, 结果一样, 而且完全确定性.
    C = X.T @ X
    w, V = np.linalg.eigh(C)
    n_new = V[:, 0]
    if n_new @ n_ref < 0:
        n_new = -n_new
    return n_new, float(-n_new @ ctr)


def _refine_plane(points, plane, dist_thresh, max_iter=60):
    """v13.4: RANSAC 之后迭代"取内点 → 最小二乘重拟合", 一直迭到内点集合不再变.

    为什么必须加: open3d 0.19 的 segment_plane 没有 seed 参数, 每次跑随机取点,
    地面法向量会有零点几度的抖动. 高度 = 到地面的距离, 于是同一张图两次跑出来
    footprint / height 能差 0.04m, 物体分类跟着翻来覆去 (实测 124638_550 的
    垃圾桶 fp 在 0.26 和 0.30 之间跳, 一会儿判 trash_can 一会儿判 basket).

    v13.3 只固定跑 8 轮, 平面还剩 ~1e-4 m 的抖动. 看着很小, 但它会被放大:
    离地 5cm 的截断线挪 0.1mm → 多/少几十个点 → DBSCAN (eps=3.5cm) 链接方式变了
    → 整簇合并/分裂不同 (实测 124740_017 的筐点数在 1183 和 1969 之间跳,
    top-1 跟着换人). 所以这里改成迭代到**内点索引集合完全不变**为止:
    到了不动点, 平面就完全由这个点集决定, 与 RANSAC 初值无关 → 逐位可复现.
    """
    P = np.asarray(points, dtype=np.float64)
    n = np.asarray(plane[:3], dtype=np.float64)
    d = float(plane[3])
    nrm = np.linalg.norm(n)
    if nrm < 1e-9:
        return plane, np.array([], dtype=int)
    n, d = n / nrm, d / nrm
    # 先统一朝向 (相机 Y+ 向下, 地面法向量取朝上那侧), 让不同随机初值走同一条收敛路
    if n[1] > 0:
        n, d = -n, -d
    n_ref = n.copy()
    # 先宽后窄: 宽阈值先把整块地面都收进来 (不同随机初值都会落进同一个"盆地"),
    # 再收窄精修.
    idx = np.array([], dtype=int)
    prev1 = prev2 = None
    for k in [3.0, 2.0, 1.5] + [1.0] * max_iter:
        dist = np.abs(P @ n + d)
        idx = np.flatnonzero(dist < dist_thresh * k)
        if len(idx) < 200:
            break
        if k == 1.0:
            # 不动点: 内点集合和上一轮 (或上上轮, 防 2-周期振荡) 一样就停
            if prev1 is not None and len(idx) == len(prev1) and np.array_equal(idx, prev1):
                break
            if prev2 is not None and len(idx) == len(prev2) and np.array_equal(idx, prev2):
                # 2-周期: 取两个里点多的那个, 保证同一张图永远选同一个
                if len(prev1) > len(idx):
                    idx = prev1
                n, d = _ls_plane(P, idx, n_ref)
                break
            prev2, prev1 = prev1, idx
        n, d = _ls_plane(P, idx, n_ref)
    return (float(n[0]), float(n[1]), float(n[2]), float(d)), idx


def fit_ground(points, *, num_iter=None):
    """RANSAC 拟合最大平面 = 地面

    Returns:
        plane:   (a, b, c, d) ax+by+cz+d=0, 法向量 (a,b,c) 指向"上"
        inliers: 内点索引数组
    """
    plane, inliers = _ransac_plane_numpy(
        points,
        config.GROUND_RANSAC_DIST,
        config.GROUND_RANSAC_N,
        config.GROUND_RANSAC_ITER if num_iter is None else int(num_iter),
    )
    # v13.3: 确定性精修, 消掉 RANSAC 的随机抖动
    plane, refined = _refine_plane(points, plane, config.GROUND_RANSAC_DIST)
    if len(refined) >= 200:
        inliers = refined
    # OpenCV 相机坐标系: Y+ 向下. 地面法向量应该指向"上" (Y-)
    a, b, c, d = plane
    if b > 0:
        plane = (-a, -b, -c, -d)
    return plane, inliers
