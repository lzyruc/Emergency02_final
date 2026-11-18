"""凸起提取 + DBSCAN + 后处理合并 + 过滤"""
import cv2
import numpy as np
from sklearn.cluster import DBSCAN as SKDBSCAN

from . import config
from .geometry import point_to_plane_distance


def extract_obstacles(points, plane):
    """提取离地面 3~100cm 的凸起点"""
    dist = point_to_plane_distance(points, plane)
    mask = (dist > config.OBSTACLE_MIN_HEIGHT) & (dist < config.OBSTACLE_MAX_HEIGHT)
    return points[mask], mask, dist


def voxel_downsample(points, voxel_size=None):
    """使用 NumPy 做确定性体素降采样。"""
    if voxel_size is None:
        voxel_size = config.VOXEL_SIZE
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    # 找每个 voxel 里的第一个点 (足够近似)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    return points[unique_idx]


def dbscan(points, eps=None, min_points=None):
    """使用 scikit-learn DBSCAN 聚类。"""
    if eps is None:
        eps = config.DBSCAN_EPS
    if min_points is None:
        min_points = config.DBSCAN_MIN_POINTS
    return SKDBSCAN(
        eps=eps,
        min_samples=min_points,
        n_jobs=-1,
    ).fit(points).labels_


def cv_detect_colored_regions(rgb_bgr, depth_mm, intr, plane, kept_clusters):
    """在 RGB 里直接找**棕色** (纸箱) 和 **蓝色** (文件筐) 的连通区域,
    结合 depth 算 3D 位置, 返回额外候选. 用于 DBSCAN 没抓到的情况.

    只加"深度聚类没覆盖到"的候选, 避免重复.

    Returns:
        list of dicts, 每个是一个候选物体, 结构类似 kept cluster.
    """
    if rgb_bgr is None:
        return []
    img_h, img_w = rgb_bgr.shape[:2]

    # CLAHE 阴影增强: 在 LAB 空间只增强 L 通道, 保持颜色, 让阴影里的纸箱棕色更明显
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(rgb_enhanced, cv2.COLOR_BGR2HSV)

    # 已有聚类的 3D 中心 (用于去重: CV 候选距离已有 <30cm 就跳过)
    existing_centers = []
    for c in kept_clusters:
        cxz = c.get("center_xz")
        if cxz is not None:
            existing_centers.append((cxz, (c.get("color") or {}).get("category")))

    color_defs = [
        # (name, HSV_low, HSV_high, min_area_px, class)
        # 棕色 H 范围 [5, 22] (原 [5, 25]) — 排除黄色胶带 (黄 H~25-35), 但含纸箱 (H~22)
        # 饱和度 S 提到 80 — 排除木头浅色 (木头 S<50)
        ("brown", (5, 80, 40), (22, 255, 220), 2500, "box"),
        ("blue",  (95, 80, 30), (130, 255, 220), 1500, "basket"),
    ]
    a, b_, c_, d_ = plane

    out = []
    for name, low, high, min_area, cls in color_defs:
        mask = cv2.inRange(hsv, low, high)
        # 清理: open (去噪) + close (填洞)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        n_lbl, lbl_img, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n_lbl):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            # 深度采样
            region_depth = depth_mm[y:y+h, x:x+w]
            region_mask = lbl_img[y:y+h, x:x+w] == i
            valid = region_depth[region_mask & (region_depth > 0) & (region_depth < config.MAX_DEPTH_MM)]
            if len(valid) < 100:
                continue
            z_med = float(np.median(valid)) / 1000.0
            # 中心 3D
            cx_pix, cy_pix = centroids[i]
            X = (cx_pix - intr["cx"]) * z_med / intr["fx"]
            Y = (cy_pix - intr["cy"]) * z_med / intr["fy"]
            # 去重: 3D 位置和已有 kept cluster 距离 <30cm 就跳过
            # v13.0: 蓝筐是网状的, DBSCAN 只会抓到零碎的一小块;
            #        这时 CV 色块给的是完整轮廓, 反而更好, 所以**不让蓝碎片压制蓝色块**,
            #        由 pipeline 在后面把重叠的蓝碎片替换掉.
            skip = False
            for (cxz_e, cat_e) in existing_centers:
                if name == "blue" and cat_e == "blue":
                    continue
                if np.hypot(X - cxz_e[0], z_med - cxz_e[1]) < 0.30:
                    skip = True
                    break
            if skip:
                continue
            # 用 4 corners 估地面尺寸
            corners_uv = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
            corners_z = z_med
            corners_X = (corners_uv[:, 0] - intr["cx"]) * corners_z / intr["fx"]
            corners_Y = (corners_uv[:, 1] - intr["cy"]) * corners_z / intr["fy"]
            # footprint 大约 = 世界 X 范围 (水平方向)
            footprint_w = float(corners_X.max() - corners_X.min())
            # 高度: 从物体顶到底的世界高度 (通过 corners_Y 估计)
            # Y 越小 = 越高 (相机坐标 Y+ 向下)
            world_h = float(corners_Y.max() - corners_Y.min())

            # 3D 中心到地面的距离 = 高度
            center_pt = np.array([[X, Y, z_med]])
            h_center = abs(float(point_to_plane_distance(center_pt, plane)[0]))

            # 过滤: 太远或太侧
            if z_med > config.FILTER_MAX_DIST_CAM_M + 0.3:  # 稍宽一点
                continue
            if abs(X) > config.FILTER_MAX_LATERAL_X:
                continue

            # v13.2: 把**这个连通域自己的掩膜**带出去.
            #   以前 pipeline 是拿整个 bbox 矩形里的所有深度点当这个物体的点云,
            #   矩形里有大量背景 (地面 / 后面的桌腿 / 障碍带), 会把 footprint 撑大
            #   (实测筐 0.31m 被撑到 0.65~0.69m, 于是被判成 "box_shape_but_blue" 丢掉).
            #   再往内腐蚀几个像素, 去掉边界上的 "飞点" (深度相机在物体边缘会
            #   把物体和背景插值成一串假点, 沿视线方向拉长 z).
            er = config.CV_MASK_ERODE_PX
            m_full = (lbl_img == i).astype(np.uint8)
            if er > 0:
                m_full = cv2.erode(m_full, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
            m_crop = m_full[y:y+h, x:x+w].astype(bool)
            if int(m_crop.sum()) < 100:          # 腐蚀过头就退回未腐蚀的
                m_crop = (lbl_img[y:y+h, x:x+w] == i)
            out.append({
                "id": 9000 + i,   # 特殊 id 段
                "cls_hint": cls,
                "cls_via": "cv_color",
                "color_name": name,
                "n_points": int(area),
                "center_3d": np.array([X, Y, z_med]),
                "footprint_w_estimate": footprint_w,
                "world_h_estimate": world_h,
                "h_center_above_ground": h_center,
                "pixel_bbox": (int(x), int(y), int(w), int(h)),
                "region_mask": m_crop,
            })
    return out


def _sample_cluster_color(cluster_pix, rgb_bgr):
    """在 RGB 图上采样聚类像素颜色, 返回颜色签名 dict.

    Args:
        cluster_pix: (N, 2) 像素 (u, v)
        rgb_bgr:     (H, W, 3) BGR 图 (cv2 格式)
    Returns:
        {
            "bgr_median": (B, G, R),
            "hsv_median": (H, S, V),
            "r_minus_b": float,        # 偏红棕 (纸箱)
            "b_minus_r": float,        # 偏蓝 (筐)
            "category": "brown"/"blue"/"gray"/"dark"/"other"
        }
    """
    img_h, img_w = rgb_bgr.shape[:2]
    u = cluster_pix[:, 0].astype(np.int32)
    v = cluster_pix[:, 1].astype(np.int32)
    # 只取在图内的像素
    m = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if m.sum() < config.COLOR_MIN_SAMPLES:
        return None
    bgr = rgb_bgr[v[m], u[m]]   # (K, 3)
    bgr_med = np.median(bgr, axis=0)
    b, g, r = float(bgr_med[0]), float(bgr_med[1]), float(bgr_med[2])

    # HSV 中位数
    hsv = cv2.cvtColor(
        bgr.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2HSV,
    )
    hsv_med = np.median(hsv.reshape(-1, 3), axis=0)
    h_, s_, v_ = float(hsv_med[0]), float(hsv_med[1]), float(hsv_med[2])

    r_minus_b = r - b
    b_minus_r = b - r

    # 分类颜色
    blue_intensity = None
    if r_minus_b >= config.COLOR_BROWN_MIN_RB and s_ >= config.COLOR_BROWN_MIN_S:
        category = "brown"
    elif b_minus_r >= config.COLOR_BLUE_MIN_BR and s_ >= config.COLOR_BLUE_MIN_S:
        category = "blue"
        # v11.7: 亚类 — vivid(筐) vs muted(桶/阴影筐)
        # v13.3: 两条分支都要求 B-R 有下限 —— 白灰垃圾桶在冷光下 S 能到 108,
        #        但 B-R 最多 45; 真筐 B-R 恒 >=55. 见 config 里的实测表.
        if ((s_ >= config.COLOR_VIVID_BLUE_MIN_S
             and b_minus_r >= config.COLOR_VIVID_BLUE_S_MIN_BR)
                or b_minus_r >= config.COLOR_VIVID_BLUE_MIN_BR):
            blue_intensity = "vivid"
        else:
            blue_intensity = "muted"
    elif s_ <= config.COLOR_GRAY_MAX_S:
        category = "gray"
    elif v_ <= config.COLOR_DARK_MAX_V:
        category = "dark"
    else:
        category = "other"

    return {
        "bgr_median": (b, g, r),
        "hsv_median": (h_, s_, v_),
        "r_minus_b": r_minus_b,
        "b_minus_r": b_minus_r,
        "category": category,
        "blue_intensity": blue_intensity,   # v11.7: "vivid" | "muted" | None
    }


def compute_cluster_stats(points, labels, plane, pixels=None, img_hw=None, rgb_bgr=None):
    """每个 cluster 的初步统计, 用于过滤

    Args:
        points: (N,3) 降采样后的点
        labels: (N,) DBSCAN 标签
        plane:  (a,b,c,d) 地面
        pixels: (N,2) 每个点对应的图像像素 (u,v). None 表示不算图像 bbox
        img_hw: (H, W) 图像尺寸. 配合 pixels 用于图像边缘过滤
        rgb_bgr: (H, W, 3) BGR 图. 配合 pixels 用于颜色分析
    """
    n_clusters = int(labels.max()) + 1
    stats = []
    for i in range(n_clusters):
        mask = labels == i
        pts = points[mask]
        if len(pts) == 0:
            stats.append(None)
            continue
        xz = pts[:, [0, 2]]
        heights = point_to_plane_distance(pts, plane)
        long_edge = max(xz[:, 0].max() - xz[:, 0].min(),
                        xz[:, 1].max() - xz[:, 1].min())
        short_edge = min(xz[:, 0].max() - xz[:, 0].min(),
                         xz[:, 1].max() - xz[:, 1].min())
        s = {
            "id": i,
            "n_points": len(pts),
            "points": pts,
            "center_xz": xz.mean(0),
            "center_3d": pts.mean(0),
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
            "height_span": float(heights.max() - heights.min()),
            "long_edge": float(long_edge),
            "short_edge": float(short_edge),
            "area": float(long_edge * short_edge),
            "elongation": float(long_edge / max(short_edge, 1e-6)),  # 长宽比
            "dist_from_cam": float(np.linalg.norm(xz.mean(0))),  # 水平距离
        }
        if pixels is not None:
            cluster_pix = pixels[mask]
            s["img_bbox_uv"] = (
                float(cluster_pix[:, 0].min()),
                float(cluster_pix[:, 1].min()),   # v_min = 图像顶
                float(cluster_pix[:, 0].max()),
                float(cluster_pix[:, 1].max()),
            )
            # 颜色分析
            if rgb_bgr is not None:
                color = _sample_cluster_color(cluster_pix, rgb_bgr)
                if color is not None:
                    s["color"] = color
        if img_hw is not None:
            s["img_hw"] = tuple(img_hw)
        stats.append(s)
    return stats


def merge_nearby(stats, plane):
    """把中心接近 + 高度重叠的 cluster 合并 (镂空物体被 DBSCAN 切碎的情况)"""
    if not stats:
        return stats
    valid_stats = [s for s in stats if s is not None]
    if len(valid_stats) <= 1:
        return stats

    # 并查集
    parent = list(range(len(valid_stats)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 两两检查
    for i in range(len(valid_stats)):
        for j in range(i + 1, len(valid_stats)):
            si, sj = valid_stats[i], valid_stats[j]
            # 中心距离
            d = np.linalg.norm(si["center_xz"] - sj["center_xz"])
            if d > config.MERGE_CENTER_DIST:
                continue
            # 高度重叠率
            lo = max(si["height_min"], sj["height_min"])
            hi = min(si["height_max"], sj["height_max"])
            overlap = max(0, hi - lo)
            span = max(si["height_span"], sj["height_span"], 1e-6)
            if overlap / span < config.MERGE_HEIGHT_OVERLAP:
                continue
            # 防止把桌腿合进大物体: 合并后 dim1 增长过多则不合并
            merged_pts = np.vstack([si["points"], sj["points"]])
            xz = merged_pts[:, [0, 2]]
            merged_long = max(xz[:, 0].max() - xz[:, 0].min(),
                              xz[:, 1].max() - xz[:, 1].min())
            max_long_before = max(si["long_edge"], sj["long_edge"])
            if merged_long / max(max_long_before, 1e-6) > config.MERGE_MAX_DIM1_GROWTH:
                continue
            union(i, j)

    # 合并
    merged = {}
    for i, s in enumerate(valid_stats):
        r = find(i)
        if r not in merged:
            merged[r] = {
                "id": s["id"],
                "points_list": [s["points"]],
                "member_ids": [s["id"]],
                "img_bboxes": [s["img_bbox_uv"]] if "img_bbox_uv" in s else [],
                "img_hw": s.get("img_hw"),
                "colors": [(len(s["points"]), s["color"])] if "color" in s else [],
            }
        else:
            merged[r]["points_list"].append(s["points"])
            merged[r]["member_ids"].append(s["id"])
            if "img_bbox_uv" in s:
                merged[r]["img_bboxes"].append(s["img_bbox_uv"])
            if "color" in s:
                merged[r]["colors"].append((len(s["points"]), s["color"]))

    # 重算每个合并 cluster 的属性
    out = []
    for r, m in merged.items():
        all_pts = np.vstack(m["points_list"])
        xz = all_pts[:, [0, 2]]
        heights = point_to_plane_distance(all_pts, plane)
        long_edge = max(xz[:, 0].max() - xz[:, 0].min(),
                        xz[:, 1].max() - xz[:, 1].min())
        short_edge = min(xz[:, 0].max() - xz[:, 0].min(),
                         xz[:, 1].max() - xz[:, 1].min())
        entry = {
            "id": min(m["member_ids"]),   # 用最小的 id 作为新 id
            "n_points": len(all_pts),
            "points": all_pts,
            "center_xz": xz.mean(0),
            "center_3d": all_pts.mean(0),
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
            "height_span": float(heights.max() - heights.min()),
            "long_edge": float(long_edge),
            "short_edge": float(short_edge),
            "area": float(long_edge * short_edge),
            "elongation": float(long_edge / max(short_edge, 1e-6)),
            "dist_from_cam": float(np.linalg.norm(xz.mean(0))),
            "merged_from": m["member_ids"],
        }
        # 合并 img_bbox = 并集
        if m["img_bboxes"]:
            bboxes = np.array(m["img_bboxes"])   # (K, 4) [u_min, v_min, u_max, v_max]
            entry["img_bbox_uv"] = (
                float(bboxes[:, 0].min()),
                float(bboxes[:, 1].min()),
                float(bboxes[:, 2].max()),
                float(bboxes[:, 3].max()),
            )
        if m["img_hw"] is not None:
            entry["img_hw"] = m["img_hw"]
        # 颜色: 取 sub-cluster 里点最多的那个的颜色
        if m.get("colors"):
            m["colors"].sort(key=lambda x: -x[0])
            entry["color"] = m["colors"][0][1]
        out.append(entry)
    return out


def _is_object_color(s):
    """v13.1: 这个 cluster 是不是"目标物体的颜色" (鲜蓝的文件筐 / 棕的纸箱).
    机器人本体是白 / 灰 / 黑, 绝不会是这两种. 用来豁免"机器人自身"过滤.

    v13.1 收严: 机器人躯干上有一条**深藏青色**的缝, HSV 里也会落进"蓝"区间
    (实测 S≈86~118, V≈40~81). 真筐即使在阴影里 V 也 >=63. 所以:
      - blue_intensity 必须真的是 "vivid" (不再放过 None)
      - 且亮度 V 不能太暗
    """
    c = s.get("color") or {}
    cat = c.get("category")
    if cat not in ("blue", "brown"):
        return False
    if cat == "blue":
        # 只有"鲜艳"的蓝才算 (muted / 未判定 的蓝可能是阴影或机器人自己)
        if c.get("blue_intensity") != "vivid":
            return False
        hsv = c.get("hsv_median")
        if hsv is not None and float(hsv[2]) < config.OBJ_BLUE_MIN_V:
            return False    # 太暗 = 机器人躯干的深色缝, 不是筐
    return True


def filter_clusters(stats):
    """标注 reject 原因; 返回带 reject 字段的 stats + 保留的 stats 列表"""
    kept = []
    for s in stats:
        reject = None
        if s["n_points"] < config.FILTER_MIN_POINTS:
            reject = f"too_few_pts({s['n_points']})"
        elif s["area"] < config.FILTER_MIN_AREA:
            reject = f"too_small({s['area']:.3f})"
        elif s["area"] > config.FILTER_MAX_AREA:
            reject = f"too_large({s['area']:.2f})"
        elif s["long_edge"] > config.FILTER_MAX_LONG_EDGE:
            reject = f"long_edge({s['long_edge']:.2f})"
        elif s["dist_from_cam"] > config.FILTER_MAX_DIST_CAM_M:
            reject = f"too_far({s['dist_from_cam']:.1f})"
        elif s["height_span"] < config.FILTER_MIN_HEIGHT_SPAN:
            reject = f"too_flat({s['height_span']:.3f})"
        elif s["height_min"] > config.FILTER_MIN_BOTTOM_HEIGHT:
            reject = f"floating(bot={s['height_min']:.2f})"
        # 机器人自己头顶 (低+近+中)
        # v13.0: 鲜明的蓝/棕色永远不是机器人本体 (机器人是白/灰/黑).
        #        之前文件筐摆在正前方 1.1~1.3m 会被这两条误杀成 robot_head.
        elif (not _is_object_color(s)
              and abs(s["center_xz"][0]) < config.FILTER_ROBOT_CX_TOL
              and s["center_xz"][1] < config.FILTER_ROBOT_MAX_Z
              and s["height_max"] < config.FILTER_ROBOT_MAX_HEIGHT):
            reject = "robot_head"
        # 机器人自己身体 (Z<1.3m 且靠中心 = 机器人本体或屏幕, 不管高度)
        elif (not _is_object_color(s)
              and s["center_xz"][1] < 1.3 and abs(s["center_xz"][0]) < 0.7):
            reject = f"robot_self_close(z={s['center_xz'][1]:.2f})"
        elif abs(s["center_xz"][0]) > config.FILTER_MAX_LATERAL_X:
            reject = f"too_lateral({s['center_xz'][0]:+.2f})"
        # 长条过滤: 桌腿/桌沿 elongation > 5
        elif (s.get("elongation", 0) > config.FILTER_ELONGATION_MAX
              and s["long_edge"] > config.FILTER_ELONGATION_MIN_LONG):
            reject = f"elongated({s['elongation']:.1f})"
        # 图像边缘过滤: 家具的聚类会出画 (顶部触边 或 横跨太宽)
        elif "img_bbox_uv" in s and "img_hw" in s:
            u_min, v_min, u_max, v_max = s["img_bbox_uv"]
            img_h, img_w = s["img_hw"]
            u_span = u_max - u_min
            # 大聚类顶部触边 = 桌沿等家具出画
            # 但必须是"矮的" (h_max<50cm), 否则可能是椅子/立起纸箱, 不能过滤
            if (v_min < config.FILTER_IMG_TOP_PX
                    and s["long_edge"] > config.FILTER_IMG_TOP_MIN_DIM1
                    and s["height_max"] < 0.50):
                reject = f"img_top({v_min:.0f}px,long={s['long_edge']:.2f})"
            # 横跨太宽 = 大家具 (同样只对矮的)
            elif (u_span / img_w > config.FILTER_IMG_WIDTH_RATIO
                    and s["height_max"] < 0.50):
                reject = f"img_wide({u_span/img_w:.2f})"

        s["reject"] = reject
        if reject is None:
            kept.append(s)
    return kept
