"""每类物体的抓取标注 (点/线/圆) —— 基于 features 里的 minAreaRect + 高度

标注约定 (相机坐标系, 米):
- line:   endpoints = [[x,y,z], [x,y,z]]
- circle: center=[x,y,z], radius=float, normal=[x,y,z]
每个标注附加 orientation: "up" / "side" / "down"

v11.6: 颜色过滤抓取
    如果传了 rgb_bgr, 按类过滤 cluster 里的点 (box→只用棕点, basket/trash→蓝点,
    chair→深色点), 排除粘在一起的其他物体的干扰. 找不够点就 fallback 到原点云.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree as _cKDTree

from . import config
from .geometry import point_to_plane_distance


# 每类物体的"期望颜色" (用于颜色过滤)
CLASS_TO_COLOR = {
    "box": "brown",
    "basket": "blue",
    "trash_can": "blue",
    "chair": "dark",
}

# 颜色过滤的最少点数 (< 这个数就不启用过滤, 用原点云 fallback)
COLOR_FILTER_MIN_POINTS = 30


def _get_color_mask(rgb_bgr, u_pix, v_pix, expected_color):
    """给定像素坐标 (u,v) 数组, 返回 boolean mask 表示每个像素是否是期望颜色.

    Args:
        rgb_bgr: (H,W,3) BGR 图, 或 None
        u_pix, v_pix: (N,) float 像素坐标
        expected_color: "brown"/"blue"/"dark"/"gray" 或 None

    Returns:
        (N,) bool mask; rgb_bgr 或 expected_color 为 None 时全 True
    """
    if rgb_bgr is None or expected_color is None:
        return np.ones(len(u_pix), dtype=bool)
    h, w = rgb_bgr.shape[:2]
    u_int = np.clip(u_pix.astype(int), 0, w - 1)
    v_int = np.clip(v_pix.astype(int), 0, h - 1)
    bgr = rgb_bgr[v_int, u_int]  # (N, 3)
    hsv = cv2.cvtColor(bgr[None, :, :], cv2.COLOR_BGR2HSV)[0]  # (N, 3)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    if expected_color == "brown":
        # 棕: OpenCV H in [5, 25] (10-50°), S>40, V>30
        return (H >= 5) & (H <= 25) & (S > 40) & (V > 30)
    if expected_color == "blue":
        # 蓝: H in [90, 130], S>40
        return (H >= 90) & (H <= 130) & (S > 40)
    if expected_color == "dark":
        # 深色 (黑办公椅): V<90
        return V < 90
    if expected_color == "gray":
        # 灰: 低饱和度 + 中亮度 (排除黑)
        return (S < 40) & (V > 60)
    return np.ones(len(u_pix), dtype=bool)


def _filter_pts_by_color(pts, rgb_bgr, intr, expected_color):
    """把 3D 点云 投影到像素 → 用颜色 mask 过滤 → 返回过滤后的点云.
    如果过滤后点数 < COLOR_FILTER_MIN_POINTS, 返回原点云 (避免抓取彻底失败).

    Returns:
        (filtered_pts, mask_used_bool)   mask_used_bool 表示是否真的过滤了
    """
    if rgb_bgr is None or intr is None or expected_color is None:
        return pts, False
    z_safe = np.clip(pts[:, 2], 1e-6, None)
    u_pix = pts[:, 0] * intr["fx"] / z_safe + intr["cx"]
    v_pix = pts[:, 1] * intr["fy"] / z_safe + intr["cy"]
    mask = _get_color_mask(rgb_bgr, u_pix, v_pix, expected_color)
    if mask.sum() < COLOR_FILTER_MIN_POINTS:
        return pts, False
    return pts[mask], True


# ==================== 姿态标签 ====================
def _annot_center(annot):
    if annot["type"] == "line":
        return (np.array(annot["endpoints"][0]) + np.array(annot["endpoints"][1])) / 2
    elif annot["type"] == "circle":
        return np.array(annot["center"])
    elif annot["type"] == "point":
        return np.array(annot["position"])
    return None


def tag_orientation(annot, cluster_center_3d, ground_normal):
    mid = _annot_center(annot)
    if mid is None:
        annot["orientation"] = "unknown"
        return annot
    direction = mid - np.asarray(cluster_center_3d)
    n = np.linalg.norm(direction)
    if n < 1e-6:
        annot["orientation"] = "up"
        return annot
    direction = direction / n
    ground_normal = np.asarray(ground_normal) / (np.linalg.norm(ground_normal) + 1e-9)
    cos = float(np.clip(np.dot(direction, ground_normal), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))
    if ang < config.UP_ANGLE_DEG:
        annot["orientation"] = "up"
    elif ang > config.DOWN_ANGLE_DEG:
        annot["orientation"] = "down"
    else:
        annot["orientation"] = "side"
    return annot


# ==================== 辅助: 从 minAreaRect 得到"世界的"长/短方向 ====================
def _footprint_axes(cluster):
    """从 minAreaRect 得到 uv 平面内的长/短方向 (3D 相机坐标系单位向量)
    + footprint 中心 (3D, 位于地面)
    Returns:
        long_axis_3d, short_axis_3d, footprint_center_3d
    """
    origin, u_ax, v_ax, n_ax = cluster["_ground_frame"]
    pts = cluster["points"]
    # 投影到 uv 平面
    rel = pts - origin
    u = rel @ u_ax
    v = rel @ v_ax
    uv = np.stack([u, v], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(uv)
    (cx, cy), (w, h), angle_deg = rect
    # rect 的 angle 是 uv 平面内 rect 长边与 u 轴的夹角
    # 长边方向 (在 uv 平面里)
    if w >= h:
        long_len, short_len = w, h
    else:
        long_len, short_len = h, w
        angle_deg = angle_deg + 90.0
    ang_rad = np.deg2rad(angle_deg)
    long_dir_uv = np.array([np.cos(ang_rad), np.sin(ang_rad)])
    short_dir_uv = np.array([-np.sin(ang_rad), np.cos(ang_rad)])
    # 3D
    long_dir_3d = long_dir_uv[0] * u_ax + long_dir_uv[1] * v_ax
    short_dir_3d = short_dir_uv[0] * u_ax + short_dir_uv[1] * v_ax
    footprint_center_3d = origin + cx * u_ax + cy * v_ax
    return long_dir_3d, short_dir_3d, footprint_center_3d, long_len, short_len, n_ax


# ==================== v11.8: 基于 3D OBB 的可见面侧棱 ====================
def _pick_visible_face_verticals(pts, ground_frame):
    """从 3D 点云 + 地面基, 算 OBB, 挑最对相机的侧面, 返回该面的两条竖侧棱.

    Args:
        pts:          (N, 3) 相机坐标系下 3D 点云
        ground_frame: (origin, u_ax, v_ax, n_ax) 地面正交基

    Returns:
        (edges, face_name, box_center_3d) 或 None
        edges = [(p_bot_3d, p_top_3d, tag), (p_bot_3d, p_top_3d, tag)]
        tag = 'left' | 'right' (相对相机 x)
    """
    if len(pts) < 3:
        return None
    origin, u_ax, v_ax, n_ax = ground_frame

    # 投影到地面平面, minAreaRect 找 OBB (v11.9: 直接用 min/max, 不 percentile 修剪)
    rel = pts - origin
    u = rel @ u_ax
    v = rel @ v_ax
    h = rel @ n_ax   # 高度
    uv = np.stack([u, v], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(uv)
    (cx, cy), (w, hh), angle_deg = rect
    if w >= hh:
        long_len, short_len = w, hh
    else:
        long_len, short_len = hh, w
        angle_deg = angle_deg + 90.0
    ang_rad = np.deg2rad(angle_deg)
    long_dir_uv = np.array([np.cos(ang_rad), np.sin(ang_rad)])
    short_dir_uv = np.array([-np.sin(ang_rad), np.cos(ang_rad)])
    long_dir_3d = long_dir_uv[0] * u_ax + long_dir_uv[1] * v_ax
    short_dir_3d = short_dir_uv[0] * u_ax + short_dir_uv[1] * v_ax
    fp_center = origin + cx * u_ax + cy * v_ax

    height_bot = float(h.min())
    height_abs = float(h.max())

    # 4 个候选竖直侧面, 每个记:
    # (法向, 面中心水平偏移方向, 面中心偏移距, 面侧棱水平偏移方向, 侧棱偏移距, 面名)
    candidates = [
        (+long_dir_3d,  +long_dir_3d,  long_len / 2,  short_dir_3d, short_len / 2, "+long"),
        (-long_dir_3d,  -long_dir_3d,  long_len / 2,  short_dir_3d, short_len / 2, "-long"),
        (+short_dir_3d, +short_dir_3d, short_len / 2, long_dir_3d,  long_len / 2,  "+short"),
        (-short_dir_3d, -short_dir_3d, short_len / 2, long_dir_3d,  long_len / 2,  "-short"),
    ]

    # 相机看向 +z. 正对相机的面, 其外法向指向 -z (normal.z 最负)
    best = min(candidates, key=lambda c: c[0][2])
    normal, offset_dir, offset_dist, edge_dir, edge_dist, name = best

    # 该面中心在地面上的位置
    face_center_ground = fp_center + offset_dir * offset_dist

    # 该面的两条竖侧棱: face_center_ground ± edge_dir * edge_dist
    # tag 用"对相机看是左还是右" — 靠 edge_base 的 x (相机坐标) 排
    edge_base_pos = face_center_ground + edge_dir * (+edge_dist)
    edge_base_neg = face_center_ground + edge_dir * (-edge_dist)
    if edge_base_pos[0] > edge_base_neg[0]:
        edges_ordered = [(edge_base_neg, "left"), (edge_base_pos, "right")]
    else:
        edges_ordered = [(edge_base_pos, "left"), (edge_base_neg, "right")]

    edges = []
    for edge_base, tag in edges_ordered:
        p_bot = edge_base + n_ax * height_bot
        p_top = edge_base + n_ax * height_abs
        edges.append((p_bot, p_top, tag))

    box_center = fp_center + n_ax * (height_bot + (height_abs - height_bot) / 2)
    return edges, name, box_center


def _grasp_visible_face_verticals(cluster, plane, rgb_bgr=None, expected_color=None):
    """v11.9: 输出最对相机那个侧面的两条竖侧棱 (基于 3D OBB, 非像素).
    (v11.10/v11.11 的 percentile + density 过滤已回滚, 使用 min/max)
    """
    pts = cluster["points"]
    intr = cluster.get("_intr")
    pts_filt, color_filtered = _filter_pts_by_color(pts, rgb_bgr, intr, expected_color)

    ground_frame = cluster["_ground_frame"]
    result = _pick_visible_face_verticals(pts_filt, ground_frame)
    if result is None:
        return [], cluster.get("center_3d", np.array([0, 0, 0]))

    edges, face_name, center_3d = result

    suffix = "_color" if color_filtered else ""
    annots = []
    for p_bot, p_top, tag in edges:
        annots.append({
            "type": "line",
            "name": f"visible_face_edge_{tag}_{face_name}{suffix}",
            "endpoints": [p_bot.tolist(), p_top.tolist()],
        })
    return annots, center_3d


def grasp_box(cluster, plane, rgb_bgr=None):
    """纸箱抓取: v11.8 = 最对相机那个侧面的两条竖侧棱 (基于 3D OBB)"""
    return _grasp_visible_face_verticals(cluster, plane, rgb_bgr=rgb_bgr,
                                          expected_color="brown")


# ==================== 椅子: 椅背上沿中心线 ====================
def grasp_chair(cluster, plane, rgb_bgr=None):
    pts = cluster["points"]
    # v11.6: 深色过滤, 排除粘上的桌沿 (亮色) 干扰
    intr = cluster.get("_intr")
    pts, _ = _filter_pts_by_color(pts, rgb_bgr, intr, "dark")

    heights = point_to_plane_distance(pts, plane)
    top = pts[heights >= heights.max() - 0.05]
    if len(top) < 8:
        return [], cluster["center_3d"]
    # 椅背上沿在 uv 平面里的方向 = 顶部点云的 2D 主方向
    origin, u_ax, v_ax, n_ax = cluster["_ground_frame"]
    rel = top - origin
    u = rel @ u_ax
    v = rel @ v_ax
    uv = np.stack([u, v], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(uv)
    (cx, cy), (w, h), angle = rect
    if w >= h:
        long_len = w
    else:
        long_len = h
        angle += 90.0
    ang = np.deg2rad(angle)
    d_uv = np.array([np.cos(ang), np.sin(ang)])
    d_3d = d_uv[0] * u_ax + d_uv[1] * v_ax
    center_uv_3d = origin + cx * u_ax + cy * v_ax
    # 用顶部点云的实际最高高度
    center_3d = center_uv_3d + n_ax * heights.max()
    p1 = center_3d - (long_len * 0.4) * d_3d
    p2 = center_3d + (long_len * 0.4) * d_3d
    return [{
        "type": "line",
        "name": "backrest_top",
        "endpoints": [p1.tolist(), p2.tolist()],
    }], center_3d


# ==================== 垃圾桶: 顶圈 ====================
def grasp_trash(cluster, plane, rgb_bgr=None):
    pts = cluster["points"]
    # v11.6: 蓝色过滤, 排除桌面/纸箱等杂点
    intr = cluster.get("_intr")
    pts, _ = _filter_pts_by_color(pts, rgb_bgr, intr, "blue")

    heights = point_to_plane_distance(pts, plane)
    top = pts[heights >= heights.max() - config.TRASH_RIM_LAYER_THICKNESS]
    if len(top) < 8:
        return [], cluster["center_3d"]
    center = top.mean(axis=0)
    dists = np.linalg.norm(top - center, axis=1)
    radius = float(np.median(dists))
    normal = np.asarray(plane[:3])
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return [{
        "type": "circle",
        "name": "top_rim",
        "center": center.tolist(),
        "radius": radius,
        "normal": normal.tolist(),
    }], center


# ==================== 文件筐 v13.0: 单线抓取 = "最高顶棱" ====================
# 物理依据 (在 basket_training_org 的两帧上逐点量过):
#   文件筐是蓝色塑料**网状**阶梯形收纳筐, 网眼很大 -> 深度会直接"看穿"网面读到地面,
#   所以网面区域的深度是假的(≈0.02m), 只有实心的框边/隔板顶沿才有可信深度.
#   => 唯一稳定可抓的几何 = 最高那面墙(后背板)的顶棱: 最高、最长、最直、外侧悬空,
#      而且永远不会被自己挡住, 正是两指夹爪最自然的目标.
#
# 算法 (深度拟合为主 + HSV 辅助, 就是用户提的思路):
#   1. HSV 蓝掩膜挑出筐的点 (每点顺带记下 S/V)
#   2. 地面平面 -> 每点离地高度 h
#   3. 取"顶部带" h >= p98(h) - TOP_BAND
#   4. 顶部带在地面上的投影里做 2D 直线 RANSAC (找墙的水平走向)
#   5. 脊线精修: 沿线分箱, 每箱取最高点 -> 这些脊点再 SVD 拟合 3D 直线
#      => 线正好落在真实顶棱上 (贴合), rms 也变成真有意义的"直不直"
#   6. 悬空性: 线至少一侧 3~7.5cm 外要"掉下去" (真棱, 不是平面内部的一条弦)
#   7. 打分 = 内点 x 长度 x 直线度 x 高度 x 悬空 x 颜色(深蓝优先, 压制浅蓝文件夹)
#   8. **只输出分最高的一条**

B_VOX       = 0.004    # 体素
B_TOP_BAND  = 0.070    # 顶部带厚度 (m)
B_TOP_PCT   = 98.0     # 顶高取百分位 (抗离群点)
B_LINE_THR  = 0.014    # 2D 直线内点阈值 (m)
B_ITERS     = 4000     # RANSAC 迭代
B_MIN_INL   = 25       # 最少内点
B_MIN_LEN   = 0.10     # 最短线长 (m)
B_GAP_MAX   = 0.06     # 沿线断开阈值 (m)
B_CREST_BIN = 0.020    # 脊线分箱宽度 (m)
B_CREST_RMS = 0.015    # 脊线直线度归一化尺度 (m)
B_RMS_MAX   = 0.022    # 脊线 rms 超过这个 = 根本不是直线, 丢掉
B_FREE_IN   = 0.030    # 悬空检测内边界 (m)
B_FREE_OUT  = 0.075    # 悬空检测外边界 (m)
B_FREE_DROP = 0.08     # 该侧最高点比线低这么多 => 完全悬空 (m)
B_ROUNDS    = 5        # 顺序 RANSAC 轮数
B_SEED      = 12345    # 固定随机种子 -> 结果可复现


def _b_ground_frame(plane):
    """返回 (上方向 g, 地面轴 u, 地面轴 v, 高度函数 h_of)"""
    a, b, c, d = plane
    nn = np.array([a, b, c], dtype=float)
    L = float(np.linalg.norm(nn)) + 1e-12
    g = nn / L
    sgn = 1.0
    if g[1] > 0:          # 相机 y 轴朝下, 法线要朝上
        g = -g
        sgn = -1.0

    def h_of(X):
        return sgn * ((np.asarray(X) @ nn + d) / L)

    u = np.cross(g, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(g, np.array([1.0, 0.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(g, u)
    return g, u, v, h_of


def _b_voxel(P, vox, extra=None):
    key = np.floor(P / vox).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    return P[idx], (None if extra is None else extra[idx])


def _b_blue_points(cluster, rgb_bgr):
    """从 cluster 里挑出蓝色的点, 并带上每点的 (S, V).
    返回 (P, col) —— HSV 不可用时 col 全给中性值 128."""
    pts = cluster.get("points_all")
    if pts is None or len(pts) < 50:
        pts = cluster["points"]
    pts = np.asarray(pts, dtype=np.float64)
    intr = cluster.get("_intr")
    if rgb_bgr is None or intr is None:
        return pts, np.full((len(pts), 2), 128.0)

    z = np.clip(pts[:, 2], 1e-6, None)
    u_pix = pts[:, 0] * intr["fx"] / z + intr["cx"]
    v_pix = pts[:, 1] * intr["fy"] / z + intr["cy"]
    H_img, W_img = rgb_bgr.shape[:2]
    ui = np.clip(u_pix.astype(int), 0, W_img - 1)
    vi = np.clip(v_pix.astype(int), 0, H_img - 1)
    bgr = rgb_bgr[vi, ui]
    hsv = cv2.cvtColor(bgr[None, :, :], cv2.COLOR_BGR2HSV)[0].astype(np.int16)
    Hc, Sc, Vc = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    m = (Hc >= 90) & (Hc <= 135) & (Sc >= 60) & (Vc >= 30)
    if int(m.sum()) < 80:                       # HSV 挂了就不过滤, 保证还能出线
        return pts, np.full((len(pts), 2), 128.0)
    return pts[m], np.stack([Sc[m], Vc[m]], axis=1).astype(np.float64)


def _b_free_side(Pd, h, cen, d3, m3, half, h_line):
    """线两侧 FREE_IN~FREE_OUT 处最高点相对线高的落差 (越大 = 越悬空)"""
    rel = Pd - cen
    t = rel @ d3
    s = rel @ m3
    band = np.abs(t) <= half
    out = []
    for sg in (1.0, -1.0):
        sel = band & (sg * s >= B_FREE_IN) & (sg * s <= B_FREE_OUT)
        if int(sel.sum()) < 4:
            out.append(0.40)                    # 那一侧根本没东西 = 完全悬空
        else:
            out.append(float(h_line - np.percentile(h[sel], 92)))
    return out


def _b_crest_fit(Q, hQ, d_init):
    """沿 d_init 分箱, 每箱取最高点 -> 这些脊点 SVD 拟合 3D 直线.
    返回 (p1, p2, d3, cen, rms, length, n_crest) 或 None"""
    cen0 = Q.mean(axis=0)
    t = (Q - cen0) @ d_init
    lo, hi = float(t.min()), float(t.max())
    nb = max(3, int(np.ceil((hi - lo) / B_CREST_BIN)))
    bid = np.clip(((t - lo) / max(hi - lo, 1e-6) * nb).astype(int), 0, nb - 1)
    idx = []
    for b in range(nb):
        sel = np.nonzero(bid == b)[0]
        if len(sel) == 0:
            continue
        idx.append(sel[int(np.argmax(hQ[sel]))])
    if len(idx) < 4:
        return None
    C = Q[np.asarray(idx)]

    def _fit(Cm):
        cen = Cm.mean(axis=0)
        _, _, Vt = np.linalg.svd(Cm - cen, full_matrices=False)
        d3 = Vt[0]
        if float(d3 @ d_init) < 0:
            d3 = -d3
        tc = (Cm - cen) @ d3
        res = np.linalg.norm((Cm - cen) - np.outer(tc, d3), axis=1)
        return cen, d3, tc, res

    cen, d3, tc, res = _fit(C)
    if len(C) >= 7:                              # 去掉最差 15% 脊点再拟一次
        keep = res <= np.percentile(res, 85)
        if int(keep.sum()) >= 4:
            C = C[keep]
            cen, d3, tc, res = _fit(C)
    t1, t2 = float(tc.min()), float(tc.max())
    return (cen + d3 * t1, cen + d3 * t2, d3, cen,
            float(np.sqrt(float((res ** 2).mean()))), float(t2 - t1), len(C))


def _basket_top_ridges(P, col, plane, seed=B_SEED, keep_all=False):
    """找文件筐最高的那几条顶棱, 按分数从高到低返回."""
    if P is None or len(P) < B_MIN_INL:
        return []
    Pd, cd = _b_voxel(P, B_VOX, extra=col)
    g, u, v, h_of = _b_ground_frame(plane)
    h = h_of(Pd)
    if len(Pd) < B_MIN_INL:
        return []

    h_top = float(np.percentile(h, B_TOP_PCT))
    band = h >= h_top - B_TOP_BAND
    if int(band.sum()) < B_MIN_INL:
        band = h >= h_top - 2 * B_TOP_BAND
    if int(band.sum()) < B_MIN_INL:
        return []
    B = Pd[band]
    hB = h[band]
    cB = cd[band] if cd is not None else None

    uvB = np.stack([B @ u, B @ v], axis=1)
    rng = np.random.default_rng(seed)
    tree = _cKDTree(uvB)
    cands = []
    used = np.zeros(len(B), dtype=bool)

    for _round in range(B_ROUNDS):
        avail = np.nonzero(~used)[0]
        if len(avail) < B_MIN_INL:
            break
        best = None
        for _it in range(B_ITERS):
            i0 = int(avail[rng.integers(len(avail))])
            if tree is not None:
                near = [j for j in tree.query_ball_point(uvB[i0], 0.35) if not used[j]]
                if len(near) < 2:
                    continue
                i1 = int(near[rng.integers(len(near))])
            else:
                i1 = int(avail[rng.integers(len(avail))])
            dv = uvB[i1] - uvB[i0]
            nr = float(np.linalg.norm(dv))
            if nr < 0.06 or nr > 0.35:
                continue
            dv = dv / nr
            nv = np.array([-dv[1], dv[0]])
            rel = uvB - uvB[i0]
            inl = (np.abs(rel @ nv) < B_LINE_THR) & (~used)
            if int(inl.sum()) < B_MIN_INL:
                continue
            t = rel @ dv
            tt = np.sort(t[inl])
            gaps = np.nonzero(np.diff(tt) > B_GAP_MAX)[0]
            segs = np.split(np.arange(len(tt)), gaps + 1)
            sg = max(segs, key=len)
            if len(sg) < B_MIN_INL:
                continue
            lo, hi = float(tt[sg[0]]), float(tt[sg[-1]])
            if hi - lo < B_MIN_LEN:
                continue
            sel = inl & (t >= lo - 1e-9) & (t <= hi + 1e-9)
            sc = int(sel.sum()) * min((hi - lo) / 0.25, 1.0)
            if best is None or sc > best[0]:
                best = (sc, sel.copy(), dv.copy())
        if best is None:
            break
        sel, dv = best[1], best[2]
        used |= sel
        d_init = dv[0] * u + dv[1] * v
        d_init = d_init / (np.linalg.norm(d_init) + 1e-12)
        cf = _b_crest_fit(B[sel], hB[sel], d_init)
        if cf is None:
            continue
        p1, p2, d3, cen, rms, length, n_crest = cf
        if length < B_MIN_LEN or rms > B_RMS_MAX:
            continue
        h_line = float(np.median(h_of(np.stack([p1, p2]))))
        m3 = np.cross(g, d3)
        if np.linalg.norm(m3) < 1e-6:
            continue
        m3 = m3 / np.linalg.norm(m3)
        fr = _b_free_side(Pd, h, cen, d3, m3, length / 2 + 0.01, h_line)
        o = {
            "p1": p1, "p2": p2, "dir": d3, "center": cen,
            "n": int(sel.sum()), "n_crest": int(n_crest),
            "length": float(length), "rms": float(rms),
            "h": h_line, "free_a": fr[0], "free_b": fr[1], "free": float(max(fr)),
            "S": float(np.median(cB[sel, 0])) if cB is not None else 128.0,
            "V": float(np.median(cB[sel, 1])) if cB is not None else 128.0,
        }
        cands.append(o)

    if not cands:
        return []
    for o in cands:
        straight = float(np.clip(1.0 - o["rms"] / B_CREST_RMS, 0.0, 1.0))
        lenf = float(min(o["length"] / 0.25, 1.0))
        hf = float(np.clip((o["h"] - 0.06) / 0.20, 0.0, 1.0))
        edgef = float(np.clip(o["free"] / B_FREE_DROP, 0.0, 1.0))
        # 颜色: 筐是又深又饱和的蓝; 筐里插的浅蓝文件夹是 S 低 V 高 -> 被压下去
        colf = (float(np.clip(o["S"] / 130.0, 0.35, 1.0))
                * float(np.clip((250.0 - o["V"]) / 70.0, 0.35, 1.0)))
        o["straightness"] = straight
        o["edgeness"] = edgef
        o["colorness"] = colf
        o["score"] = (o["n"] * lenf * (0.30 + 0.70 * straight)
                      * (0.15 + 0.85 * hf) * (0.20 + 0.80 * edgef) * colf)
        o["conf"] = float(np.clip(0.30 + 0.28 * lenf * straight + 0.20 * edgef
                                  + 0.12 * colf + 0.10 * hf, 0.30, 0.97))
    cands.sort(key=lambda x: -x["score"])
    return cands if keep_all else cands[:1]


def grasp_basket(cluster, plane, rgb_bgr=None):
    """安全禁用旧单最高棱入口；basket 必须走 RGB-D 五板语义检测器。"""
    del plane, rgb_bgr
    center_3d = cluster.get("center_3d", np.array([0.0, 0.0, 1.0]))
    return [], center_3d


def _legacy_grasp_basket_reference(cluster, plane, rgb_bgr=None):
    """文件筐抓取 v13.0: **只输出 1 条线** = 置信度最高的那条顶棱."""
    center_3d = cluster.get("center_3d", np.array([0.0, 0.0, 1.0]))
    P, col = _b_blue_points(cluster, rgb_bgr)
    cands = _basket_top_ridges(P, col, plane, keep_all=True)
    if not cands:
        return [], center_3d
    o = cands[0]
    runner_up = float(cands[1]["score"]) if len(cands) > 1 else 0.0
    margin = float(o["score"] / max(runner_up, 1e-6)) if runner_up > 0 else 99.0
    return [{
        "type": "line",
        "name": "top_ridge",
        "endpoints": [o["p1"].tolist(), o["p2"].tolist()],
        "score": float(o["score"]),
        "confidence": float(o["conf"]),
        "length_m": float(o["length"]),
        "height_above_ground": float(o["h"]),
        "straightness": float(o["straightness"]),
        "edgeness": float(o["edgeness"]),
        "colorness": float(o["colorness"]),
        "rms_mm": float(o["rms"] * 1000.0),
        "n_support": int(o["n"]),
        "n_candidates": int(len(cands)),
        "margin_vs_2nd": margin,
    }], center_3d


# ==================== 主入口 ====================
CLASS_TO_GRASP = {
    "box": grasp_box,
    # chair / basket / trash_can 都有独立 RGB-D 语义分支。旧的点云近似
    # 抓取不再进入公开总流水线，避免椅背侧沿、任意文件筐外轮廓或缩在
    # 蓝色桶身内部的小圆成为降级标注。
}


def compute_grasp_annotations(cluster, plane, rgb_bgr=None):
    """v11.6: 新增 rgb_bgr 参数. 传了就启用颜色过滤抓取."""
    cls = cluster.get("class")
    fn = CLASS_TO_GRASP.get(cls)
    if fn is None:
        return []
    try:
        annots, obj_center = fn(cluster, plane, rgb_bgr=rgb_bgr)
    except Exception as e:
        print(f"[grasp] compute failed for {cls}: {e}")
        return []
    ground_normal = np.asarray(plane[:3])
    ground_normal = ground_normal / (np.linalg.norm(ground_normal) + 1e-9)
    for a in annots:
        tag_orientation(a, obj_center, ground_normal)
    return annots
