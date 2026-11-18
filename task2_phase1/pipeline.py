"""主流水线: 一帧深度图 -> 物体列表"""
import cv2
import numpy as np

from . import config
from . import basket_cv
from .chair_cv import (
    ALGORITHM_VERSION as CHAIR_ALGORITHM_VERSION,
    detect_chair,
)
from .trash_cv import (
    ALGORITHM_VERSION as TRASH_ALGORITHM_VERSION,
    detect_trash,
)
from .geometry import (depth_to_pointcloud, point_to_plane_distance,
                        project_to_image, pca_axes, solid_ratio)
from .ground import fit_ground
from .clustering import (extract_obstacles, voxel_downsample, dbscan,
                          compute_cluster_stats, merge_nearby, filter_clusters,
                          cv_detect_colored_regions, _sample_cluster_color)
from .features import compute_features
from .classifier import classify
from .grasp import compute_grasp_annotations
from .result_types import FrameResult, ObjectResult

PIPELINE_VERSION = "phase1_depth_hsv_geometry_unified_v5"


def _enhance_shadows(rgb_bgr):
    """CLAHE 阴影增强 (LAB 空间只提亮 L 通道, 保持颜色)
    让阴影里的纸箱棕色更饱和, 帮助颜色判断; 但白色金属还是白 (灰色) 不会变色.
    """
    if rgb_bgr is None:
        return rgb_bgr
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _basket_object_rank(obj):
    """文件筐的确定性整帧排序键。

    安全抓取线存在性优先；之后综合物体与唯一抓取线置信度。长度和像素位置
    只用于稳定打破平局，避免同一帧多个 HSV 连通域各自发布一条线。
    """
    lines = [
        item
        for item in obj.grasp_annotations
        if isinstance(item, dict)
        and item.get("type") == "line"
        and item.get("name") == "basket_divider_free_edge"
        and item.get("grasp_safe")
        and not item.get("inferred")
    ]
    line = max(
        lines,
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            float(item.get("length_m", 0.0)),
        ),
        default=None,
    )
    line_conf = float(line.get("confidence", 0.0)) if line else 0.0
    line_length = float(line.get("length_m", 0.0)) if line else 0.0
    bbox = obj.debug.get("pixel_bbox") or (1e9, 1e9, 0, 0)
    return (
        int(line is not None),
        0.5 * float(obj.confidence) + 0.5 * line_conf,
        line_conf,
        float(obj.confidence),
        line_length,
        -float(bbox[0]),
        -float(bbox[1]),
    )


def _ground_dist(p1, p2, plane):
    """两个物体在**地面上**隔多远 (把连线里垂直于地面的那一截减掉).

    === 为什么不能直接用相机坐标里的 XZ 距离 ===
    头戴相机是**低头看**的 (实测地面法向 ~(0.05, -0.58, -0.81), 约低头 36 度),
    所以相机的 XZ 平面和地面差了 36 度. 两个明明在地上离了 0.71m 的东西
    (124740_017 的文件筐和灰垃圾桶), 算成相机 XZ 距离只有 0.3996m —— 正好
    卡在 0.40m 的去重门槛下面, 于是筐被当成垃圾桶的重复检测删掉了.
    把连线投影到地面上再量, 才是"它们在地上离多远"这个真问题.
    """
    d = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    n = np.asarray(plane[:3], dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return float(np.hypot(d[0], d[2]))
    n = n / nn
    return float(np.linalg.norm(d - (d @ n) * n))


def _v14_to_objects(hits, plane, start_id=1000):
    """把专用检测结果转成标准对象，并执行整帧文件筐 Top-1。

    ``detect_baskets`` 内部最多保留三个 HSV 连通域用于消歧，但机械臂接口只
    能收到一个文件筐和一条抓取线。因此转换边界必须再次做全帧确定性选择，
    不能让多个连通域各发布自己的局部 Top-1。
    """
    objs = []
    for k, hb in enumerate(hits):
        p = hb["points"]
        try:
            axes, extents, _ = pca_axes(p)
        except Exception:
            axes = np.eye(3)
            extents = np.array([hb["fp_long"], hb["fp_short"],
                                hb["height_above_ground"]])
        try:
            sr = solid_ratio(p)
        except Exception:
            sr = 0.0
        raw_grasps = hb.get("grasps") or []
        # 文件筐标准接口只接受用户指定的五块 L 板自由边。旧 ``grasp``
        # 字段可能来自 best_grasp_line，语义上允许任意外轮廓甚至横梁，因此
        # 不能再作为无条件回退。过滤也放在转换边界上，避免未来调用方绕过
        # basket_cv 的新主入口把旧线带进绘图或机械臂接口。
        target_grasps = [
            item
            for item in raw_grasps
            if isinstance(item, dict)
            and item.get("name") == "basket_divider_free_edge"
        ]
        # 转换边界再次执行 Top-1，防止旧调用方或测试夹具把多条候选直接
        # 送进绘图/机械臂接口。当前 Top-1 入口在这里本来就只有 0/1 条。
        target_grasps.sort(
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                -float(item.get("length_m", 0.0)),
                int(item.get("ridge_id", 0)),
            )
        )
        grasps = target_grasps[:1]
        rejected_grasps = len(raw_grasps) - len(grasps)
        objs.append(ObjectResult(
            id=start_id + k,
            cls="basket",
            confidence=float(hb["confidence"]),
            position_cam=[float(v) for v in hb["center_3d"]],
            principal_axes=np.asarray(axes).tolist(),
            principal_extents=np.asarray(extents).tolist(),
            height_above_ground=float(hb["height_above_ground"]),
            footprint_area=float(hb["footprint_area"]),
            n_points=int(hb["n_points"]),
            solid_ratio=float(sr),
            grasp_annotations=grasps,
            in_channel=None,
            debug={
                   "reason": (
                       f"{basket_cv.GRASP_ALGORITHM_VERSION}"
                       "(color+depth+five-panel-geometry+top1)"
                   ),
                   "src": basket_cv.GRASP_ALGORITHM_VERSION,
                   "dedicated_detector": True,
                   "fp_long": round(float(hb["fp_long"]), 3),
                   "fp_short": round(float(hb["fp_short"]), 3),
                   "pixel_bbox": hb["pixel_bbox"],
                   "rejected_non_target_grasps": rejected_grasps,
                   **hb["debug"]},
        ))
    if not objs:
        return []
    candidate_count = len(objs)
    winner = max(objs, key=_basket_object_rank)
    winner.id = start_id
    winner.debug["frame_basket_candidate_count"] = candidate_count
    winner.debug["selected_frame_top1"] = True
    return [winner]


def _dedicated_chair_objects(depth_mm, intr, rgb_bgr):
    """运行椅子专用 RGB-D 分支并标记其标准对象。

    专用分支已执行椅座、椅背及上沿硬门；这里只保留最高置信的一把椅子。
    """
    result = detect_chair(
        depth_mm,
        intr,
        rgb_bgr=rgb_bgr,
        ground_stride=config.CHAIR_GROUND_STRIDE,
    )
    chairs = [obj for obj in result.objects if obj.cls == "chair"]
    if not chairs:
        return [], result, CHAIR_ALGORITHM_VERSION
    winner = max(chairs, key=lambda obj: float(obj.confidence))
    winner.debug = dict(winner.debug or {})
    winner.debug.update(
        {
            "dedicated_detector": True,
            "src": CHAIR_ALGORITHM_VERSION,
            "selected_frame_top1": True,
            "frame_chair_candidate_count": len(chairs),
        }
    )
    return [winner], result, CHAIR_ALGORITHM_VERSION


def _dedicated_trash_objects(depth_mm, intr, rgb_bgr, plane):
    """运行垃圾桶专用 RGB-D 桶口分支并转成标准对象。"""
    hits, detector_debug = detect_trash(
        rgb_bgr,
        depth_mm,
        intr,
        plane,
    )
    objects = []
    for index, hit in enumerate(hits[:1]):
        objects.append(
            ObjectResult(
                id=2000 + index,
                cls="trash_can",
                confidence=float(hit["confidence"]),
                position_cam=np.asarray(hit["center_3d"]).tolist(),
                principal_axes=np.asarray(
                    hit["principal_axes"]
                ).tolist(),
                principal_extents=np.asarray(
                    hit["principal_extents"]
                ).tolist(),
                height_above_ground=float(hit["height_above_ground"]),
                footprint_area=float(hit["footprint_area"]),
                n_points=int(hit["n_points"]),
                solid_ratio=float(hit["solid_ratio"]),
                grasp_annotations=list(hit["grasp_annotations"]),
                in_channel=None,
                debug={
                    "reason": (
                        f"{TRASH_ALGORITHM_VERSION}"
                        "(white-rim+muted-blue-body+depth+D255+top1)"
                    ),
                    "src": TRASH_ALGORITHM_VERSION,
                    "dedicated_detector": True,
                    "selected_frame_top1": True,
                    **dict(hit["debug"]),
                },
            )
        )
    return objects, detector_debug, TRASH_ALGORITHM_VERSION


def _dedicated_only_objects(
    chairs,
    basket_hits,
    basket_plane,
    trash_objects,
):
    """组合三个专用类别结果，供通用聚类提前返回时使用。"""
    objects = list(chairs)
    objects.extend(
        _v14_to_objects(
            basket_hits,
            basket_plane,
            start_id=1000 + len(objects),
        )
    )
    objects.extend(trash_objects)
    objects.sort(key=lambda obj: obj.cls)
    for index, obj in enumerate(objects):
        obj.id = index
    return objects


def process_frame(depth_mm, intr, rgb_bgr=None, task1_channel_polygon=None):
    """主入口: 处理一帧深度图, 返回 FrameResult

    Args:
        depth_mm: (H,W) uint16 深度图 (毫米)
        intr:     {fx, fy, cx, cy} color 相机内参 (深度已对齐到 color)
        rgb_bgr:  (H,W,3) 可选 BGR 图 (cv2 读的). 传了就启用颜色分类
                  (必须与 depth 大小一致或对齐到 color 视角)
        task1_channel_polygon: 兼容性预留参数；当前 Phase 1 尚未接入通道多边形，
                               即使传入也不会计算 ``in_channel``。

    Returns:
        FrameResult
    """
    # 0. RGB 阴影增强 (CLAHE): 提高阴影里物体的颜色饱和度, 帮助分类
    rgb_enhanced = _enhance_shadows(rgb_bgr) if rgb_bgr is not None else None

    # 1. 反投影
    pts, _ = depth_to_pointcloud(depth_mm, intr)

    # 2. 地面。RANSAC 的每次迭代都要扫描传入的全部点；对 1280x720 深度图
    # 直接喂约 90 万点会让一帧仅地面拟合就耗时十几秒。这里仅对“拟合地面”
    # 做确定性均匀抽样，后续凸起、聚类、分类以及抓取端点仍使用全量点/RGB-D。
    phase1_ground_stride = max(1, int(config.PHASE1_GROUND_STRIDE))
    phase1_ground_points = np.asarray(
        pts[::phase1_ground_stride],
        dtype=np.float64,
    )
    plane, ground_inliers = fit_ground(phase1_ground_points)
    phase1_ground_sample_points = int(len(phase1_ground_points))

    # 2a. 轮式办公椅专用分支。RGB-D 输入下它是 chair 的权威入口：只允许
    # 物理椅背上沿，椅背侧边、座垫和桌沿都不能作为降级抓取线。
    dedicated_chairs = []
    chair_detector_error = None
    chair_detector_debug = {}
    chair_detector_algorithm = None
    chair_detector_attempted = rgb_bgr is not None
    if chair_detector_attempted:
        try:
            (
                dedicated_chairs,
                chair_frame,
                chair_detector_algorithm,
            ) = _dedicated_chair_objects(
                depth_mm,
                intr,
                rgb_bgr,
            )
            chair_detector_debug = dict(chair_frame.debug or {})
        except Exception as exc:      # 单帧异常不能让其它类别流水线挂掉
            dedicated_chairs = []
            chair_detector_error = f"{type(exc).__name__}: {exc}"

    # 2b. v14: 文件筐**颜色优先**检测 (独立于下面整条深度聚类流水线).
    #     筐是镂空网格, 深度点云稀疏而且被网洞切碎, DBSCAN 串不起来 ->
    #     每块碎片都过不了 FILTER_MIN_POINTS/FILTER_MIN_AREA -> 整个漏掉.
    #     所以筐单独走一条路: 亮蓝掩膜检测, 深度只做验证和定位.
    #     只影响 basket, 纸箱/椅子/垃圾桶完全不动.
    v14_baskets = []
    basket_detector_error = None
    basket_detector_attempted = rgb_bgr is not None
    basket_plane = plane
    basket_ground_sample_points = 0
    if basket_detector_attempted:
        try:
            basket_stride = max(1, int(config.BASKET_GROUND_STRIDE))
            basket_points = np.asarray(pts[::basket_stride], dtype=np.float64)
            basket_ground_sample_points = int(len(basket_points))
            if len(basket_points) >= 100:
                basket_plane, _ = fit_ground(basket_points)
            v14_baskets = basket_cv.detect_baskets(
                rgb_bgr,
                depth_mm,
                intr,
                basket_plane,
            )
        except Exception as exc:      # 单帧异常不能让整条流水线挂掉
            v14_baskets = []
            basket_detector_error = f"{type(exc).__name__}: {exc}"

    # 2c. 垃圾桶专用全图 RGB-D 分支。只接管 trash_can：候选必须同时通过
    # 白色桶沿椭圆、灰蓝桶身、深度和 D=255 mm 物理尺寸硬门。它不会进入
    # 后面的跨类别去重，因此不可能删掉或改变椅子、文件筐和纸箱结果。
    dedicated_trash = []
    trash_detector_error = None
    trash_detector_debug = {}
    trash_detector_algorithm = None
    trash_detector_attempted = rgb_bgr is not None
    if trash_detector_attempted:
        try:
            (
                dedicated_trash,
                trash_detector_debug,
                trash_detector_algorithm,
            ) = _dedicated_trash_objects(
                depth_mm,
                intr,
                rgb_bgr,
                plane,
            )
        except Exception as exc:      # 专用分支异常不能拖垮其它类别
            dedicated_trash = []
            trash_detector_error = f"{type(exc).__name__}: {exc}"

    # 3. 凸起
    obs_pts, _, _ = extract_obstacles(pts, plane)
    if len(obs_pts) < 100:
        return FrameResult(
            objects=_dedicated_only_objects(
                dedicated_chairs,
                v14_baskets,
                basket_plane,
                dedicated_trash,
            ),
            ground_plane=plane,
            n_raw_clusters=0,
            n_kept=0,
            debug={
                "pipeline_algorithm": PIPELINE_VERSION,
                "task1_channel_polygon_ignored": (
                    task1_channel_polygon is not None
                ),
                "phase1_ground_stride": phase1_ground_stride,
                "phase1_ground_sample_points": phase1_ground_sample_points,
                "chair_detector_attempted": chair_detector_attempted,
                "chair_detector_algorithm": chair_detector_algorithm,
                "chair_detector_error": chair_detector_error,
                "chair_detector_objects": len(dedicated_chairs),
                "chair_detector_debug": chair_detector_debug,
                "basket_detector_error": basket_detector_error,
                "basket_detector_attempted": basket_detector_attempted,
                "basket_ground_stride": config.BASKET_GROUND_STRIDE,
                "basket_ground_sample_points": basket_ground_sample_points,
                "trash_detector_attempted": trash_detector_attempted,
                "trash_detector_algorithm": trash_detector_algorithm,
                "trash_detector_error": trash_detector_error,
                "trash_detector_objects": len(dedicated_trash),
                "trash_detector_debug": trash_detector_debug,
            },
        )

    # 4. 降采样 + 聚类
    obs_ds = voxel_downsample(obs_pts, voxel_size=config.VOXEL_SIZE)
    labels = dbscan(obs_ds)
    if labels.max() < 0:
        return FrameResult(
            objects=_dedicated_only_objects(
                dedicated_chairs,
                v14_baskets,
                basket_plane,
                dedicated_trash,
            ),
            ground_plane=plane,
            n_raw_clusters=0,
            n_kept=0,
            debug={
                "pipeline_algorithm": PIPELINE_VERSION,
                "task1_channel_polygon_ignored": (
                    task1_channel_polygon is not None
                ),
                "phase1_ground_stride": phase1_ground_stride,
                "phase1_ground_sample_points": phase1_ground_sample_points,
                "chair_detector_attempted": chair_detector_attempted,
                "chair_detector_algorithm": chair_detector_algorithm,
                "chair_detector_error": chair_detector_error,
                "chair_detector_objects": len(dedicated_chairs),
                "chair_detector_debug": chair_detector_debug,
                "basket_detector_error": basket_detector_error,
                "basket_detector_attempted": basket_detector_attempted,
                "basket_ground_stride": config.BASKET_GROUND_STRIDE,
                "basket_ground_sample_points": basket_ground_sample_points,
                "trash_detector_attempted": trash_detector_attempted,
                "trash_detector_algorithm": trash_detector_algorithm,
                "trash_detector_error": trash_detector_error,
                "trash_detector_objects": len(dedicated_trash),
                "trash_detector_debug": trash_detector_debug,
            },
        )

    # 4b. 每个降采样后的点算图像像素 (供图像边缘过滤 + 颜色采样)
    obs_pixels = project_to_image(obs_ds, intr)  # (N, 2) (u, v)

    # 5. 每 cluster 统计 + 合并 + 过滤 (用 CLAHE 增强后的 RGB 采样颜色)
    stats = compute_cluster_stats(obs_ds, labels, plane,
                                   pixels=obs_pixels,
                                   img_hw=depth_mm.shape,
                                   rgb_bgr=rgb_enhanced)
    merged = merge_nearby(stats, plane)
    kept = filter_clusters(merged)

    # 5b. CV 色块检测 (棕纸箱 + 蓝筐): 补 DBSCAN 抓漏的
    #     从 depth 图直接采样 3D 点, 构造合成 cluster 加入 kept
    #     注: 传入原始 rgb (cv_detect_colored_regions 内部会自己 CLAHE 一次)
    if rgb_bgr is not None:
        color_hits = cv_detect_colored_regions(rgb_bgr, depth_mm, intr, plane, kept)
        for hit in color_hits:
            x, y, w, h = hit["pixel_bbox"]
            # 从 depth 里采样这个区域的 3D 点
            region_depth = depth_mm[y:y+h, x:x+w]
            v_idx, u_idx = np.meshgrid(np.arange(y, y+h), np.arange(x, x+w), indexing='ij')
            depth_ok = (region_depth > 0) & (region_depth < config.MAX_DEPTH_MM)
            # v13.2: **物体自己的点** = 只取颜色掩膜内的像素 (已向内腐蚀去飞点).
            #        bbox 矩形里剩下的是背景 (地面/桌腿/障碍带), 不能算进物体尺寸.
            rmask = hit.get("region_mask")
            if rmask is not None and rmask.shape == region_depth.shape:
                valid = depth_ok & rmask
                if valid.sum() < 100:
                    valid = depth_ok          # 掩膜太小就退回整个 bbox
            else:
                valid = depth_ok
            if valid.sum() < 100:
                continue

            def _unproject(sel):
                zz = region_depth[sel].astype(np.float32) / 1000.0
                xx = (u_idx[sel].astype(np.float32) - intr["cx"]) * zz / intr["fx"]
                yy = (v_idx[sel].astype(np.float32) - intr["cy"]) * zz / intr["fy"]
                return np.stack([xx, yy, zz], axis=1)

            pts_3d = _unproject(valid)
            # 抓取要用"全量"点 (整个 bbox, 含掩膜外和贴地的部分):
            # 顶棱两侧的悬空判断需要看线旁边有没有东西.
            pts_rect = _unproject(depth_ok) if valid is not depth_ok else pts_3d
            # 只保留地面之上的点 (排除地面反射)
            d_to_plane = point_to_plane_distance(pts_3d, plane)
            obs_mask = (d_to_plane > config.OBSTACLE_MIN_HEIGHT) & (d_to_plane < config.OBSTACLE_MAX_HEIGHT)

            # **深度强证据检查**: CV 色块必须结合 depth 联判, 不能只有颜色
            n_obs = int(obs_mask.sum())
            n_valid = int(valid.sum())
            # (1) 至少 30% 的有效深度点必须离地 (地贴 / 地面画黄线 不满足)
            if n_obs < max(150, 0.30 * n_valid):
                continue
            # (2) 离地点必须有 3D 高度 span (真物体有厚度; 地贴平铺 span 接近 0)
            heights_obs = d_to_plane[obs_mask]
            if float(heights_obs.max() - heights_obs.min()) < 0.05:
                continue   # 3D 上是薄薄一片 = 地贴/胶带

            pts_ok = pts_3d[obs_mask]
            heights_ok = heights_obs
            xz = pts_ok[:, [0, 2]]
            long_edge = float(max(xz[:, 0].max() - xz[:, 0].min(),
                                  xz[:, 1].max() - xz[:, 1].min()))
            short_edge = float(min(xz[:, 0].max() - xz[:, 0].min(),
                                   xz[:, 1].max() - xz[:, 1].min()))
            # CV 色块合成 cluster 也过一遍尺寸检查, 排除桌面木头/大片墙 等
            if long_edge > 1.0 or long_edge / max(short_edge, 1e-6) > 5.0:
                continue   # 太大 or 太长条 = 桌面/墙, 不是纸箱/筐
            # (3) 高度必须 >5cm 才算物体 (进一步排薄贴)
            if float(heights_ok.max()) < 0.05:
                continue
            # (4) 深度一致性检查: 深度 std 太大 = 采到了背景 / 桌面下有洞
            z_std = float(np.std(pts_ok[:, 2]))
            if z_std > 0.30:
                continue
            # (5) 机器人自身: v13.1 —— **靠像素位置判, 不靠距离判**.
            #     头部相机和机器人身体是刚性连接的, 所以机器人自己的躯干永远出现在
            #     画面最底下同一块地方. 实测 36 个蓝色块里, 机器人躯干那条深色缝
            #     每一帧都是 bbox≈(589~646, 673~682, 74~132, 38~47), 底边正好贴 720;
            #     而真的文件筐 bbox 底边最低也只到 472. 两者完全不重叠.
            #     深度在那条缝里读出来是 z=1.23, 所以**用距离判会误杀 1.2m 处的真筐**
            #     (201127 帧就是这么被毙掉的), 必须用像素位置.
            img_h_, img_w_ = depth_mm.shape[:2]
            if (y + h) >= img_h_ - 3 and y >= config.CV_ROBOT_BODY_TOP_RATIO * img_h_:
                continue    # 贴着画面底边的一条窄带 = 机器人自己的躯干
            center_xz_ = xz.mean(0)
            if float(np.linalg.norm(center_xz_)) < config.CV_COLOR_MIN_DIST:
                continue
            # (6) 距离/侧偏检查 (与 filter_clusters 一致)
            if float(np.linalg.norm(center_xz_)) > config.FILTER_MAX_DIST_CAM_M:
                continue
            if abs(center_xz_[0]) > config.FILTER_MAX_LATERAL_X:
                continue
            # v13.4: CV 色块也要算**完整的颜色签名** (S / V / B-R / vivid 亚类).
            #   以前这里只写死 {"category": 色块名}, 于是 blue_intensity 和 b_minus_r
            #   全是空的 —— classifier 里那条"灰蓝(B-R<48)当垃圾桶"的判据整个失效,
            #   色块一路掉到最后的兜底规则 basket_by_blue, 把灰蓝垃圾桶认成了筐
            #   (124740_017 帧的抓取线就画到了垃圾桶口沿上).
            #   采样用掩膜内的像素, 和 DBSCAN 那条路一样用 CLAHE 增强图, 口径统一.
            _uu = u_idx[valid].astype(np.int32)
            _vv = v_idx[valid].astype(np.int32)
            cv_color = _sample_cluster_color(np.stack([_uu, _vv], axis=1),
                                             rgb_enhanced)
            if cv_color is None:
                cv_color = {"category": hit["color_name"]}
            else:
                cv_color = dict(cv_color)
                # 类别仍以 HSV inRange 的判定为准 (色块本来就是按它切出来的),
                # 只是把数值字段补齐给 classifier 用
                cv_color["category"] = hit["color_name"]
            synthetic = {
                "id": hit["id"],
                "n_points": int(obs_mask.sum()),
                "points": pts_ok,
                # v13.0: 抓取用的"全量"点 (含贴地的部分), 顶棱两侧的悬空判断需要它
                # v13.2: 这里要用整个 bbox 矩形的点 (pts_rect), 不是掩膜内的
                "points_all": pts_rect,
                "center_xz": xz.mean(0),
                "center_3d": pts_ok.mean(0),
                "height_min": float(heights_ok.min()),
                "height_max": float(heights_ok.max()),
                "height_span": float(heights_ok.max() - heights_ok.min()),
                "long_edge": long_edge,
                "short_edge": short_edge,
                "area": float(long_edge * short_edge),
                "elongation": float(long_edge / max(short_edge, 1e-6)),
                "dist_from_cam": float(np.linalg.norm(xz.mean(0))),
                "from_cv_color": hit["color_name"],
                "img_bbox_uv": (float(x), float(y), float(x + w), float(y + h)),
                "img_hw": depth_mm.shape,
                # 颜色: category 强制用色块名, 其余数值字段由掩膜像素实测 (v13.4)
                "color": cv_color,
            }
            # v13.0: 蓝色 CV 色块 = 完整的筐轮廓, 把它附近被 DBSCAN 切碎的蓝碎片顶掉,
            #        避免同一个筐既有碎片又有完整块 (碎片会让抓取线只贴到半截).
            if hit["color_name"] == "blue":
                kept = [c for c in kept
                        if not ((c.get("color") or {}).get("category") == "blue"
                                and float(np.linalg.norm(
                                    np.asarray(c["center_xz"]) - center_xz_)) < 0.40)]
            kept.append(synthetic)

    # 6. 特征 + 分类 + 抓取
    objects = []
    for i, c in enumerate(kept):
        c = compute_features(c, plane)
        cls, conf, reason = classify(c)
        c["class"] = cls
        c["debug_reason"] = reason
        c["_intr"] = intr   # 供 grasp 用 (直接用像素 bbox 画抓取线)
        # v11.6: 传 rgb 给 grasp, 用于颜色过滤 (排除粘上的其他物体)
        # RGB-D 总入口下，chair 必须由上面的专用分支接管；这里不再浪费时间
        # 生成旧的顶部点云近似线。无 RGB 时通用识别仍可保留 chair 对象，但
        # grasp.py 已安全禁用旧椅子抓取，不会发布侧边/座垫降级线。
        # 旧 trash 抓取仅从蓝色桶身薄层估一个小圆，既不贴白色外沿也没有
        # 透视椭圆；即使缺 RGB 或专用分支异常也绝不能回退发布。通用
        # trash_can 类别在无 RGB 降级时可以保留，但抓取列表必须为空。
        specialized_grasp_disabled = (
            (cls == "chair" and chair_detector_attempted)
            or cls == "trash_can"
        )
        grasps = (
            []
            if specialized_grasp_disabled
            else compute_grasp_annotations(c, plane, rgb_bgr=rgb_enhanced)
        )
        obj = ObjectResult(
            id=i,
            cls=cls,
            confidence=conf,
            position_cam=c["center_3d"].tolist() if hasattr(c["center_3d"], "tolist")
                         else list(c["center_3d"]),
            principal_axes=c["pca_axes"].tolist(),
            principal_extents=c["pca_extents"].tolist(),
            height_above_ground=c["height_abs"],
            footprint_area=c["area"],
            n_points=c["n_points"],
            solid_ratio=c["solid_ratio"],
            grasp_annotations=grasps,
            in_channel=None,  # TODO: Task1 集成
            debug={"reason": reason,
                   "aspect_12": c["aspect_12"],
                   "aspect_23": c["aspect_23"],
                   "orig_cluster_id": c["id"]},
        )
        objects.append(obj)

    # 6b. 专用类别接管。
    #     RGB-D 总入口中，专用椅子检测是 chair 的唯一权威来源；即使它判定
    #     “没有椅子”，也不能让旧通用分类器补出一条不受上沿硬门约束的线。
    if chair_detector_attempted:
        objects = [obj for obj in objects if obj.cls != "chair"]
        objects.extend(dedicated_chairs)

    #     RGB-D 总入口中，专用文件筐分支同样是 basket 的唯一权威来源。
    #     通用 DBSCAN 常把网孔筐切成零散蓝块，即使专用分支明确否决，旧分类器
    #     仍可能补出一个假 basket；因此只要专用分支已尝试，就必须先清掉全部
    #     通用 basket，再按 0/1 个专用结果接管。没有 RGB 时仅保留无抓取线的
    #     通用类别估计，不伪造指定红边。
    if basket_detector_attempted:
        objects = [o for o in objects if o.cls != "basket"]
        if v14_baskets:
            objects.extend(
                _v14_to_objects(
                    v14_baskets,
                    basket_plane,
                    start_id=1000 + len(objects),
                )
            )

    # 专用垃圾桶也必须在跨类别去重**之前**清掉通用 trash。否则一个稍后
    # 虽会被删除的通用 trash 碎片仍可能先占据 40 cm 邻域，把真实 box 等
    # 非垃圾桶对象删掉。专用桶仍在全部去重结束后追加，完全不参与跨类去重。
    if trash_detector_attempted:
        objects = [obj for obj in objects if obj.cls != "trash_can"]

    # 7. 最终去重:
    #   (a) 空间去重: 同一位置 (30cm 内) 只留最高置信度 (跨类别; 处理筐碎片被认为 trash 等)
    #   (b) 每类只留最高置信度 (场上每类只有一个)
    #   (c) 置信度 <每类阈值 = 认为没检测到 (可能清场了)
    if config.TOP1_PER_CLASS:
        # Step 1: 按置信度高到低排序, 空间去重 (用每类阈值)
        def _passes_threshold(obj):
            thresh = config.FINAL_CONF_MIN_BY_CLASS.get(obj.cls, config.FINAL_CONF_MIN_DEFAULT)
            return obj.confidence >= thresh
        candidates = [o for o in objects
                      if o.cls != "unknown" and _passes_threshold(o)]
        def _dedicated_object(obj):
            return bool(
                obj.cls in {"basket", "chair"}
                and obj.debug.get("dedicated_detector") is True
            )

        # 专用 chair / basket 先占据各自物理位置，避免同一物体产生的高分
        # DBSCAN 碎片（常被分成 trash/box）反向删掉。
        candidates.sort(
            key=lambda obj: (not _dedicated_object(obj), -obj.confidence)
        )
        spatially_deduped = []
        for obj in candidates:
            close = False
            for existing in spatially_deduped:
                # 两个不同类别都已通过各自完整 RGB-D 硬门时，不因 40 cm 的
                # 通用碎片去重半径互相删除；真实椅子和文件筐可能相邻摆放。
                if (
                    obj.cls != existing.cls
                    and _dedicated_object(obj)
                    and _dedicated_object(existing)
                ):
                    continue
                # 用**地面上**的距离, 不是相机 XZ 距离 —— 相机是低头看的, 见 _ground_dist
                if _ground_dist(obj.position_cam, existing.position_cam,
                                plane) < 0.40:
                    close = True
                    break
            if not close:
                spatially_deduped.append(obj)
        # Step 2: 每类只留最好。文件筐公开契约只有唯一 Top-1 安全线，
        # 同类排序直接使用这条线的置信度，不再平均一组候选线。
        def _rank(obj):
            if obj.cls != "basket":
                return obj.confidence
            return _basket_object_rank(obj)[1]

        best_per_class = {}
        for obj in spatially_deduped:
            prev = best_per_class.get(obj.cls)
            if prev is None or _rank(obj) > _rank(prev):
                best_per_class[obj.cls] = obj
        final_objects = list(best_per_class.values())
        # 按类别名排序 + 重编号
        final_objects.sort(key=lambda o: o.cls)
        for i, obj in enumerate(final_objects):
            obj.id = i
        objects = final_objects

    # 垃圾桶在通用去重完成后才追加。这样专用桶候选既不会参与跨类 40 cm
    # 去重，也不会改变任何非 trash 对象；旧的点云近似圆绝不回退。
    if trash_detector_attempted:
        objects.extend(dedicated_trash)
        objects.sort(key=lambda obj: obj.cls)
        for index, obj in enumerate(objects):
            obj.id = index

    return FrameResult(
        objects=objects,
        ground_plane=plane,
        n_raw_clusters=int(labels.max()) + 1,
        n_kept=len(kept),
        debug={"n_raw_pts": len(pts),
               "pipeline_algorithm": PIPELINE_VERSION,
               "task1_channel_polygon_ignored": (
                   task1_channel_polygon is not None
               ),
               "phase1_ground_stride": phase1_ground_stride,
               "phase1_ground_sample_points": phase1_ground_sample_points,
               "n_obs_pts": len(obs_pts),
               "n_obs_ds": len(obs_ds),
               "n_merged": len(merged),
               "chair_detector_attempted": chair_detector_attempted,
               "chair_detector_algorithm": chair_detector_algorithm,
               "chair_detector_error": chair_detector_error,
               "chair_detector_objects": len(dedicated_chairs),
               "chair_detector_debug": chair_detector_debug,
               "basket_detector_error": basket_detector_error,
               "basket_detector_attempted": basket_detector_attempted,
               "basket_ground_stride": config.BASKET_GROUND_STRIDE,
               "basket_ground_sample_points": basket_ground_sample_points,
               "trash_detector_attempted": trash_detector_attempted,
               "trash_detector_algorithm": trash_detector_algorithm,
               "trash_detector_error": trash_detector_error,
               "trash_detector_objects": len(dedicated_trash),
               "trash_detector_debug": trash_detector_debug},
    )
