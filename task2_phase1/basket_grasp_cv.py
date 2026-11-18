"""文件筐五块隔板自由上轮廓提取。

这里的“抓取边”严格采用用户红线示意的物理语义：

* 左右外侧板和三块内隔板，共五块平行板；
* 每块板只允许其 L 形自由上轮廓；
* 前梁、后梁、底边、孔洞边和任意“最高外轮廓”都不是降级候选。

算法不使用检测网络。文件筐本体由 :mod:`basket_cv` 的 HSV 滞回掩膜检测；
本模块再结合 RGB 方向边缘、深度高度、五板等距先验以及侧倒 L 模板来确定抓取
线。五块板和侧倒 L 轮廓只作为内部候选池；每个候选还会从一端截取局部抓取
窗，避免一条过长自由边的两端处在不同三维高度。公开结果至多返回其中一条
经过 RGB、深度和可见性终审的最高置信短直线。遮挡、反光或网孔造成的无支撑
区不会用直线补齐，实心端盖上的接缝和高光也被明确排除。
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


ALGORITHM_VERSION = "basket_depth_hsv_geometry_top1_terminal_v18"

# 规则文件：A4 文件筐长 303 mm、宽 257 mm、高 308 mm。五块板沿 303 mm
# 方向排列，中心距约为 303 / 4 = 76 mm；塑料厚度、透视和深度空洞需要容差。
BASKET_PANEL_SPAN_M = 0.303
DIVIDER_GAP_M = BASKET_PANEL_SPAN_M / 4.0

TOP_BAND_M = 0.060
TOP_COMPONENT_RATIO_MIN = 0.22
SIDE_PANEL_SOLIDITY_MAX = 0.855
SIDE_PANEL_DEFECT_MIN = 0.155

MIN_LINE_PX = 24.0
MIN_LINE_M = 0.040
MAX_LINE_M = 0.47
MAX_LINE_RMS_M = 0.032

EDGE_SUPPORT_DISTANCE_PX = 3.0
EDGE_SUPPORT_GAP_CLOSE_PX = 4
# 最终唯一抓取线至少 40 px；圆孔边和侧倒轮廓毛刺通常远短于此。
EDGE_SUPPORT_MIN_RUN_PX = 40
CAP_EDGE_SUPPORT_MIN_RUN_PX = 32
FINAL_EDGE_P90_PX = 3.0
FINAL_EDGE_WITHIN_3PX = 0.95
FINAL_EDGE_MAX_AIR_PX = 4.5
FINAL_DEPTH_SUPPORT_MIN = 24
FINAL_LINE_RMS_M = 0.020

# 抓取时不使用整条长边，而是在一端保留局部窗口。长边约取三分之一，中等边
# 约取一半；已很短的边保持原长。裁后仍必须有足够的像素和物理夹持长度。
TERMINAL_LONG_FRACTION = 1.0 / 3.0
TERMINAL_MEDIUM_FRACTION = 0.5
TERMINAL_LONG_LENGTH_M = 0.22
TERMINAL_MEDIUM_LENGTH_M = 0.13
TERMINAL_LONG_LENGTH_PX = 120.0
TERMINAL_MEDIUM_LENGTH_PX = 70.0
TERMINAL_DEPTH_DELTA_TRIGGER_M = 0.050
TERMINAL_HEIGHT_PRIORITY_DELTA_M = 0.025
TERMINAL_MIN_LENGTH_M = 0.060
TERMINAL_MAX_LENGTH_M = 0.160
TERMINAL_MIN_LENGTH_PX = 30.0
TERMINAL_CAP_MIN_LENGTH_PX = 28.0
TERMINAL_END_INSET_FRACTION = 0.025
TERMINAL_DEPTH_CORRIDOR_PX = 16.0


def _ramp(value, lo, hi):
    if hi <= lo:
        return float(value >= hi)
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def _gauss(value, center, sigma):
    return float(math.exp(-0.5 * ((float(value) - center) / max(sigma, 1e-9)) ** 2))


def _angle_distance_deg(a, b):
    delta = abs((float(a) - float(b)) % 180.0)
    return min(delta, 180.0 - delta)


def _unproject(u, v, z, intr):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = (u - float(intr["cx"])) * z / float(intr["fx"])
    y = (v - float(intr["cy"])) * z / float(intr["fy"])
    return np.stack([x, y, z], axis=-1)


def _project_camera_points(points, intr):
    points = np.asarray(points, dtype=np.float64)
    z = points[..., 2]
    u = float(intr["fx"]) * points[..., 0] / z + float(intr["cx"])
    v = float(intr["fy"]) * points[..., 1] / z + float(intr["cy"])
    return np.stack([u, v], axis=-1)


def _ray_plane_point(uv, intr, point, normal):
    ray = np.array(
        [
            (float(uv[0]) - float(intr["cx"])) / float(intr["fx"]),
            (float(uv[1]) - float(intr["cy"])) / float(intr["fy"]),
            1.0,
        ],
        dtype=np.float64,
    )
    denom = float(normal @ ray)
    if abs(denom) < 1e-8:
        return None
    scale = float(normal @ point) / denom
    if scale <= 0.05:
        return None
    return scale * ray


def _ray_line_point(uv, intr, center, direction):
    ray = np.array(
        [
            (float(uv[0]) - float(intr["cx"])) / float(intr["fx"]),
            (float(uv[1]) - float(intr["cy"])) / float(intr["fy"]),
            1.0,
        ],
        dtype=np.float64,
    )
    ray /= np.linalg.norm(ray) + 1e-12
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    coupling = float(direction @ ray)
    denom = 1.0 - coupling * coupling
    if denom < 1e-7:
        return None
    dd = float(direction @ center)
    er = float(ray @ center)
    t = (coupling * er - dd) / denom
    return center + t * direction


def _geometry_maps(depth_mm, intr, plane):
    height, width = depth_mm.shape[:2]
    vv, uu = np.mgrid[0:height, 0:width]
    z = depth_mm.astype(np.float64) / 1000.0
    valid = (depth_mm > 0) & (depth_mm < 6000)
    x = (uu - float(intr["cx"])) * z / float(intr["fx"])
    y = (vv - float(intr["cy"])) * z / float(intr["fy"])
    a, b, c, d = (float(item) for item in plane)
    object_height = a * x + b * y + c * z + d
    object_height[~valid] = np.nan
    return {
        "x": x,
        "y": y,
        "z": z,
        "height": object_height,
        "valid": valid,
    }


def _largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if count <= 1:
        return None, 0
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8), int(
        stats[index, cv2.CC_STAT_AREA]
    )


def _top_surface(blob, maps):
    valid_blob = (blob > 0) & maps["valid"]
    values = maps["height"][valid_blob]
    if len(values) < 100:
        return None
    base = float(np.nanpercentile(values, 2))
    peak = float(np.nanpercentile(values, 98))
    band = (
        valid_blob
        & (maps["height"] >= peak - TOP_BAND_M)
        & (maps["height"] <= peak + 0.025)
    ).astype(np.uint8)
    # 大核只填圆孔和深度小断口。L 形主凹口宽几十到上百像素，不会被填平。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
    component, area = _largest_component(band)
    if component is None:
        return None
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / max(hull_area, 1.0)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = (
        cv2.convexityDefects(contour, hull_indices)
        if hull_indices is not None and len(hull_indices) >= 3
        else None
    )
    max_defect_px = (
        0.0
        if defects is None
        else float(np.max(np.asarray(defects).reshape(-1, 4)[:, 3])) / 256.0
    )
    x, y, width, height = cv2.boundingRect(contour)
    defect_ratio = max_defect_px / max(min(width, height), 1)
    ratio = float(area) / max(float((blob > 0).sum()), 1.0)
    if ratio < TOP_COMPONENT_RATIO_MIN:
        pose = "open"
    elif (
        solidity <= SIDE_PANEL_SOLIDITY_MAX
        and defect_ratio >= SIDE_PANEL_DEFECT_MIN
    ):
        pose = "side_laid"
    else:
        pose = "cap_visible"
    return {
        "base": base,
        "peak": peak,
        "span": peak - base,
        "mask": component,
        "contour": contour,
        "area": area,
        "ratio": ratio,
        "solidity": solidity,
        "defect_ratio": defect_ratio,
        "bbox": (x, y, width, height),
        "pose": pose,
    }


def _fit_plane(points):
    """稳健拟合朝上板面，返回中心、法向、RMS 和保留点。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 40:
        return None
    keep = np.ones(len(points), dtype=bool)
    center = normal = None
    rms = float("inf")
    for _ in range(3):
        sample = points[keep]
        center = np.median(sample, axis=0)
        matrix = sample - center
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        normal = vh[-1]
        residual = np.abs((points - center) @ normal)
        threshold = max(float(np.percentile(residual[keep], 82)), 0.004)
        new_keep = residual <= min(threshold, 0.025)
        if int(new_keep.sum()) < 30:
            break
        keep = new_keep
    residual = np.abs((points - center) @ normal)
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    return center, normal / (np.linalg.norm(normal) + 1e-12), rms, keep


def _polygon_concave_index(points):
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count < 6:
        return None
    oriented_area = cv2.contourArea(
        points.astype(np.float32).reshape(-1, 1, 2), oriented=True
    )
    orientation = 1.0 if oriented_area >= 0 else -1.0
    signed_cross = []
    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        a = current - previous
        b = following - current
        signed_cross.append(orientation * float(a[0] * b[1] - a[1] * b[0]))
    index = int(np.argmin(signed_cross))
    if signed_cross[index] >= -40.0:
        return None
    return index


def _contour_arc(contour, start_uv, end_uv, through_uv):
    points = contour.reshape(-1, 2).astype(np.float64)

    def nearest(uv):
        return int(np.argmin(np.linalg.norm(points - np.asarray(uv), axis=1)))

    start = nearest(start_uv)
    end = nearest(end_uv)
    through = nearest(through_uv)
    count = len(points)

    def forward(a, b):
        if a <= b:
            indices = np.arange(a, b + 1)
        else:
            indices = np.r_[np.arange(a, count), np.arange(0, b + 1)]
        return points[indices]

    arc_forward = forward(start, end)
    arc_reverse = forward(end, start)[::-1]

    def contains(arc):
        return float(
            np.min(np.linalg.norm(arc - points[through], axis=1))
        ) <= 2.5

    if contains(arc_forward) and not contains(arc_reverse):
        return arc_forward
    if contains(arc_reverse) and not contains(arc_forward):
        return arc_reverse
    # 极少数轮廓索引在圆角处有并列最近点；含主凹口的自由链通常也是较短弧。
    return arc_forward if len(arc_forward) <= len(arc_reverse) else arc_reverse


def _simplify_open_polyline(points, epsilon=2.2):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(points, float(epsilon), False).reshape(-1, 2)
    if len(simplified) < 2:
        return np.asarray(points).reshape(-1, 2)
    # 删除相邻小毛刺；保留 L 转折处真正的 5~20 cm 直段。
    output = [simplified[0].astype(np.float64)]
    for point in simplified[1:]:
        point = point.astype(np.float64)
        if np.linalg.norm(point - output[-1]) >= 8.0:
            output.append(point)
        else:
            output[-1] = point
    if len(output) < 2:
        return simplified.astype(np.float64)
    return np.asarray(output, dtype=np.float64)


def _line_edge_metrics(distance, p0, p1):
    """返回一条二维线到最近 RGB/色度边缘的逐像素贴合度。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length_px = float(np.linalg.norm(p1 - p0))
    count = max(12, int(math.ceil(length_px)) + 1)
    points = p0 + np.linspace(0.0, 1.0, count)[:, None] * (p1 - p0)
    uu = np.clip(
        np.rint(points[:, 0]).astype(np.int32), 0, distance.shape[1] - 1
    )
    vv = np.clip(
        np.rint(points[:, 1]).astype(np.int32), 0, distance.shape[0] - 1
    )
    values = np.asarray(distance[vv, uu], dtype=np.float64)
    far = values > 4.5
    longest_far = current = 0
    for value in far:
        if value:
            current += 1
            longest_far = max(longest_far, current)
        else:
            current = 0
    return {
        "values": values,
        "length_px": length_px,
        "p90_px": float(np.percentile(values, 90)),
        "within_3px": float(np.mean(values <= EDGE_SUPPORT_DISTANCE_PX)),
        "max_air_run_px": float(
            longest_far * length_px / max(count - 1, 1)
        ),
    }


def _line_is_tightly_supported(distance, p0, p1):
    metrics = _line_edge_metrics(distance, p0, p1)
    return bool(
        metrics["p90_px"] <= 4.35
        and metrics["within_3px"] >= 0.84
        and metrics["max_air_run_px"] <= 7.5
    ), metrics


def _trim_to_supported_interval(
    distance,
    p0,
    p1,
    *,
    min_run_px=EDGE_SUPPORT_MIN_RUN_PX,
    allowed_mask=None,
):
    """裁到最长连续 RGB 边缘支撑区，绝不跨长断口补线。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    metrics = _line_edge_metrics(distance, p0, p1)
    count = len(metrics["values"])
    points = p0 + np.linspace(0.0, 1.0, count)[:, None] * (p1 - p0)
    if allowed_mask is None:
        allowed = np.ones(count, dtype=bool)
    else:
        uu = np.clip(
            np.rint(points[:, 0]).astype(np.int32),
            0,
            allowed_mask.shape[1] - 1,
        )
        vv = np.clip(
            np.rint(points[:, 1]).astype(np.int32),
            0,
            allowed_mask.shape[0] - 1,
        )
        allowed = np.asarray(allowed_mask[vv, uu] > 0, dtype=bool)
    if (
        metrics["length_px"] >= float(min_run_px)
        and metrics["p90_px"] <= 4.35
        and metrics["within_3px"] >= 0.84
        and metrics["max_air_run_px"] <= 7.5
        and bool(np.all(allowed))
    ):
        return p0, p1, metrics, False

    good = (metrics["values"] <= EDGE_SUPPORT_DISTANCE_PX) & allowed
    # Canny 双边缘和一两个反光像素会留下很短的孔；只闭合至多 4 px 的小断口。
    index = 0
    while index < len(good):
        if good[index]:
            index += 1
            continue
        end = index
        while end < len(good) and not good[end]:
            end += 1
        if (
            index > 0
            and end < len(good)
            and end - index <= EDGE_SUPPORT_GAP_CLOSE_PX
        ):
            good[index:end] = True
        index = end

    runs = []
    index = 0
    while index < len(good):
        if not good[index]:
            index += 1
            continue
        end = index + 1
        while end < len(good) and good[end]:
            end += 1
        if end - index >= int(math.ceil(float(min_run_px))):
            runs.append((index, end - 1))
        index = end
    if not runs:
        return None, None, metrics, False

    start, end = max(runs, key=lambda item: item[1] - item[0])
    denominator = max(len(good) - 1, 1)
    q0 = p0 + (start / denominator) * (p1 - p0)
    q1 = p0 + (end / denominator) * (p1 - p0)
    trimmed_metrics = _line_edge_metrics(distance, q0, q1)
    trimmed_allowed = (
        True
        if allowed_mask is None
        else _line_mask_fraction(allowed_mask, q0, q1) >= 0.995
    )
    if (
        trimmed_metrics["length_px"] < float(min_run_px) - 1
        or trimmed_metrics["p90_px"] > 4.35
        or trimmed_metrics["within_3px"] < 0.84
        or trimmed_metrics["max_air_run_px"] > 7.5
        or not trimmed_allowed
    ):
        return None, None, trimmed_metrics, True
    return q0, q1, trimmed_metrics, True


def _snap_line_to_rgb_edge(distance, p0, p1, *, max_offset_px=8.0):
    """在给定语义走廊内沿法向寻找最贴实体边的位置。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return p0, p1, 0.0
    direction = vector / length
    normal = np.array([-direction[1], direction[0]])
    best = None
    for offset in np.arange(
        -float(max_offset_px), float(max_offset_px) + 0.01, 1.0
    ):
        q0 = p0 + offset * normal
        q1 = p1 + offset * normal
        metrics = _line_edge_metrics(distance, q0, q1)
        score = (
            1.8 * metrics["p90_px"]
            + 7.0 * (1.0 - metrics["within_3px"])
            + 0.16 * metrics["max_air_run_px"]
            + 0.055 * abs(float(offset))
        )
        key = (score, abs(float(offset)))
        if best is None or key < best[0]:
            best = (key, q0, q1, float(offset))
    return best[1], best[2], best[3]


def _boundary_asymmetry(mask, p0, p1, offset_px=5.0):
    """量化一条线是否真的是实心板面与自由空间之间的外轮廓。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length < 2.0:
        return 0.0, 0.0
    direction = vector / length
    normal = np.array([-direction[1], direction[0]])
    count = max(20, int(math.ceil(length * 0.85)))
    points = p0 + np.linspace(0.075, 0.925, count)[:, None] * vector

    def occupancy(offset):
        query = points + float(offset) * normal
        uu = np.clip(
            np.rint(query[:, 0]).astype(np.int32), 0, mask.shape[1] - 1
        )
        vv = np.clip(
            np.rint(query[:, 1]).astype(np.int32), 0, mask.shape[0] - 1
        )
        return np.asarray(mask[vv, uu] > 0, dtype=bool)

    positive = occupancy(offset_px)
    negative = occupancy(-offset_px)
    xor_fraction = float(np.mean(np.logical_xor(positive, negative)))
    polarity = float(
        max(
            np.mean(positive & ~negative),
            np.mean(negative & ~positive),
        )
    )
    return xor_fraction, polarity


def _line_mask_fraction(mask, p0, p1):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length = float(np.linalg.norm(p1 - p0))
    count = max(12, int(math.ceil(length)) + 1)
    points = p0 + np.linspace(0.0, 1.0, count)[:, None] * (p1 - p0)
    uu = np.clip(
        np.rint(points[:, 0]).astype(np.int32), 0, mask.shape[1] - 1
    )
    vv = np.clip(
        np.rint(points[:, 1]).astype(np.int32), 0, mask.shape[0] - 1
    )
    return float(np.mean(mask[vv, uu] > 0))


def _terminal_depth_corridor_stats(blob, maps, p0, p1):
    """统计最终短窗周围真实存在的文件筐深度点。

    网孔会让线本身的深度像素缺失，所以不能只数直线上的像素；这里仅在蓝色
    文件筐掩膜内、短窗两侧 16 px 的窄走廊计数。这个数是裁后重新测得的局部
    支撑，不再用完整边支撑点数乘裁切比例来估算。
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vector = p1 - p0
    length_px = float(np.linalg.norm(vector))
    if length_px < 1.0:
        return 0, 0.0
    direction = vector / length_px
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    margin = int(math.ceil(TERMINAL_DEPTH_CORRIDOR_PX + 2.0))
    height, width = blob.shape[:2]
    x0 = max(0, int(math.floor(min(p0[0], p1[0]))) - margin)
    x1 = min(width, int(math.ceil(max(p0[0], p1[0]))) + margin + 1)
    y0 = max(0, int(math.floor(min(p0[1], p1[1]))) - margin)
    y1 = min(height, int(math.ceil(max(p0[1], p1[1]))) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0.0

    local_valid = (
        (np.asarray(blob[y0:y1, x0:x1]) > 0)
        & np.asarray(maps["valid"][y0:y1, x0:x1], dtype=bool)
    )
    vv, uu = np.nonzero(local_valid)
    if len(uu) == 0:
        return 0, 0.0
    points = np.stack([uu + x0, vv + y0], axis=1).astype(np.float64)
    relative = points - p0
    along = relative @ direction
    lateral = np.abs(relative @ normal)
    keep = (
        (along >= -1.5)
        & (along <= length_px + 1.5)
        & (lateral <= TERMINAL_DEPTH_CORRIDOR_PX)
    )
    count = int(np.sum(keep))
    if count == 0:
        return 0, 0.0

    # 覆盖率只用于端点间的证据排序和审计；RGB 连续边仍负责阻止跨孔补线。
    bin_count = max(6, min(16, int(math.ceil(length_px / 5.0))))
    normalized = np.clip(along[keep] / max(length_px, 1e-9), 0.0, 1.0)
    bin_index = np.minimum(
        (normalized * bin_count).astype(np.int32),
        bin_count - 1,
    )
    coverage = float(len(np.unique(bin_index))) / float(bin_count)
    return count, coverage


def _terminal_base_fraction(length_m, length_px, depth_delta_m):
    if (
        length_m >= TERMINAL_LONG_LENGTH_M
        or length_px >= TERMINAL_LONG_LENGTH_PX
    ):
        return TERMINAL_LONG_FRACTION, "long_edge_one_third"
    if (
        length_m >= TERMINAL_MEDIUM_LENGTH_M
        or length_px >= TERMINAL_MEDIUM_LENGTH_PX
        or depth_delta_m >= TERMINAL_DEPTH_DELTA_TRIGGER_M
    ):
        return TERMINAL_MEDIUM_FRACTION, "medium_edge_one_half"
    return 1.0, "already_short"


def _terminal_variant(
    endpoints_3d,
    end_index,
    fraction,
    intr,
    edge_distance,
    maps,
    blob,
    pose,
):
    inset = min(
        TERMINAL_END_INSET_FRACTION,
        max(0.0, (1.0 - fraction) * 0.20),
    )
    if end_index == 0:
        t0 = inset
        t1 = min(1.0 - inset, inset + fraction)
    else:
        t1 = 1.0 - inset
        t0 = max(inset, t1 - fraction)
    points_3d = (
        endpoints_3d[0]
        + np.asarray([t0, t1])[:, None]
        * (endpoints_3d[1] - endpoints_3d[0])
    )
    if np.any(points_3d[:, 2] <= 0.05):
        return None
    points_uv = _project_camera_points(points_3d, intr)
    metrics = _line_edge_metrics(
        edge_distance,
        points_uv[0],
        points_uv[1],
    )
    length_m = float(np.linalg.norm(points_3d[1] - points_3d[0]))
    minimum_px = (
        TERMINAL_CAP_MIN_LENGTH_PX
        if pose == "cap_visible"
        else TERMINAL_MIN_LENGTH_PX
    )
    direct_valid_fraction = _line_mask_fraction(
        maps["valid"],
        points_uv[0],
        points_uv[1],
    )
    local_depth_points, local_depth_coverage = (
        _terminal_depth_corridor_stats(
            blob,
            maps,
            points_uv[0],
            points_uv[1],
        )
    )
    strict = bool(
        length_m >= TERMINAL_MIN_LENGTH_M - 1e-6
        and length_m <= TERMINAL_MAX_LENGTH_M + 1e-6
        and metrics["length_px"] >= minimum_px - 1.0
        and metrics["p90_px"] <= FINAL_EDGE_P90_PX
        and metrics["within_3px"] >= FINAL_EDGE_WITHIN_3PX
        and metrics["max_air_run_px"] <= FINAL_EDGE_MAX_AIR_PX
        and local_depth_points >= FINAL_DEPTH_SUPPORT_MIN
    )
    midpoint = 0.5 * (points_3d[0] + points_3d[1])
    midpoint_uv = 0.5 * (points_uv[0] + points_uv[1])
    evidence_key = (
        int(strict),
        round(float(metrics["within_3px"]), 6),
        -round(float(metrics["p90_px"]), 6),
        -round(float(metrics["max_air_run_px"]), 6),
        round(float(direct_valid_fraction), 6),
        round(float(local_depth_coverage), 6),
        int(local_depth_points),
        -round(float(midpoint[2]), 6),
        -round(float(midpoint_uv[1]), 6),
        -int(end_index),
    )
    return {
        "end_index": int(end_index),
        "t0": float(t0),
        "t1": float(t1),
        "points_3d": points_3d,
        "points_uv": points_uv,
        "length_m": length_m,
        "metrics": metrics,
        "direct_valid_fraction": direct_valid_fraction,
        "local_depth_points": int(local_depth_points),
        "local_depth_coverage": float(local_depth_coverage),
        "strict": strict,
        "evidence_key": evidence_key,
    }


def _crop_candidate_to_terminal(
    item,
    intr,
    plane,
    edge_distance,
    maps,
    blob,
    top,
):
    """把一个完整语义边候选裁成靠一端的局部抓取窗。"""
    item = dict(item)
    item_debug = dict(item.get("debug") or {})
    endpoints_3d = np.asarray(item.get("endpoints"), dtype=np.float64)
    endpoints_uv = np.asarray(item.get("endpoints_uv"), dtype=np.float64)
    if endpoints_3d.shape != (2, 3) or endpoints_uv.shape != (2, 2):
        item_debug["terminal_crop_reject"] = "invalid_endpoints"
        item["debug"] = item_debug
        return item

    length_m = float(np.linalg.norm(endpoints_3d[1] - endpoints_3d[0]))
    length_px = float(np.linalg.norm(endpoints_uv[1] - endpoints_uv[0]))
    depth_delta_m = abs(
        float(endpoints_3d[1, 2] - endpoints_3d[0, 2])
    )
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal)) + 1e-12
    normal /= normal_norm
    offset = float(plane[3]) / normal_norm
    endpoint_heights = endpoints_3d @ normal + offset
    height_delta_m = abs(
        float(endpoint_heights[1] - endpoint_heights[0])
    )
    base_fraction, policy = _terminal_base_fraction(
        length_m,
        length_px,
        depth_delta_m,
    )
    item_debug.update(
        {
            "terminal_policy": policy,
            "pre_crop_length_m": round(length_m, 5),
            "pre_crop_length_px": round(length_px, 2),
            "pre_crop_endpoint_ground_heights_m": [
                round(float(value), 5) for value in endpoint_heights
            ],
            "pre_crop_endpoint_height_delta_mm": round(
                height_delta_m * 1000.0, 2
            ),
            "pre_crop_endpoint_depth_delta_mm": round(
                depth_delta_m * 1000.0, 2
            ),
        }
    )
    if base_fraction >= 0.999:
        metrics = _line_edge_metrics(
            edge_distance,
            endpoints_uv[0],
            endpoints_uv[1],
        )
        local_depth_points, local_depth_coverage = (
            _terminal_depth_corridor_stats(
                blob,
                maps,
                endpoints_uv[0],
                endpoints_uv[1],
            )
        )
        minimum_px = (
            TERMINAL_CAP_MIN_LENGTH_PX
            if item.get("pose") == "cap_visible"
            else TERMINAL_MIN_LENGTH_PX
        )
        short_window_safe = bool(
            TERMINAL_MIN_LENGTH_M - 1e-6
            <= length_m
            <= TERMINAL_MAX_LENGTH_M + 1e-6
            and length_px >= minimum_px - 1.0
            and metrics["p90_px"] <= FINAL_EDGE_P90_PX
            and metrics["within_3px"] >= FINAL_EDGE_WITHIN_3PX
            and metrics["max_air_run_px"] <= FINAL_EDGE_MAX_AIR_PX
            and local_depth_points >= FINAL_DEPTH_SUPPORT_MIN
        )
        item_debug.update(
            {
                "terminal_crop_applied": False,
                "terminal_fraction": 1.0,
                "terminal_end": "full_short_edge",
                "post_crop_length_m": round(length_m, 5),
                "post_crop_length_px": round(length_px, 2),
                "post_crop_endpoint_height_delta_mm": round(
                    height_delta_m * 1000.0, 2
                ),
                "post_crop_endpoint_depth_delta_mm": round(
                    depth_delta_m * 1000.0, 2
                ),
                "terminal_local_depth_points": int(local_depth_points),
                "terminal_local_depth_coverage": round(
                    local_depth_coverage, 3
                ),
            }
        )
        if not short_window_safe:
            item_debug["terminal_publish_reject"] = (
                "already_short_edge_failed_post_crop_hard_gate"
            )
            item["grasp_safe"] = False
        item["n_support"] = int(local_depth_points)
        if "local_depth_points" in item_debug:
            item_debug["local_depth_points"] = int(local_depth_points)
        item["debug"] = item_debug
        return item

    minimum_px = (
        TERMINAL_CAP_MIN_LENGTH_PX
        if item.get("pose") == "cap_visible"
        else TERMINAL_MIN_LENGTH_PX
    )
    required_fraction = float(
        max(
            base_fraction,
            TERMINAL_MIN_LENGTH_M / max(length_m, 1e-9),
            minimum_px / max(length_px, 1e-9),
        )
    )
    maximum_fraction = float(
        min(0.90, TERMINAL_MAX_LENGTH_M / max(length_m, 1e-9))
    )
    if required_fraction > maximum_fraction + 1e-9:
        item_debug.update(
            {
                "terminal_crop_applied": False,
                "terminal_crop_reject": "no_window_within_length_contract",
                "terminal_publish_reject": (
                    "no_6_to_16cm_terminal_window_available"
                ),
                "terminal_fraction": 1.0,
                "terminal_end": "none",
                "post_crop_length_m": round(length_m, 5),
                "post_crop_length_px": round(length_px, 2),
            }
        )
        item["grasp_safe"] = False
        item["debug"] = item_debug
        return item
    fraction = min(maximum_fraction, required_fraction)
    variants = []
    for _ in range(5):
        variants = [
            _terminal_variant(
                endpoints_3d,
                end_index,
                fraction,
                intr,
                edge_distance,
                maps,
                blob,
                str(item.get("pose")),
            )
            for end_index in (0, 1)
        ]
        variants = [variant for variant in variants if variant is not None]
        if (
            any(variant["strict"] for variant in variants)
            or fraction >= maximum_fraction - 1e-9
        ):
            break
        fraction = min(maximum_fraction, fraction * 1.15)

    safe_variants = [
        variant for variant in variants if variant["strict"]
    ]
    if not safe_variants:
        item_debug.update(
            {
                "terminal_crop_applied": False,
                "terminal_crop_reject": "no_terminal_window_passed_hard_gate",
                "terminal_publish_reject": (
                    "no_safe_local_window_do_not_publish_full_edge"
                ),
                "terminal_fraction": 1.0,
                "terminal_end": "none",
                "post_crop_length_m": round(length_m, 5),
                "post_crop_length_px": round(length_px, 2),
                "post_crop_endpoint_height_delta_mm": round(
                    height_delta_m * 1000.0, 2
                ),
                "post_crop_endpoint_depth_delta_mm": round(
                    depth_delta_m * 1000.0, 2
                ),
            }
        )
        item["grasp_safe"] = False
        item["debug"] = item_debug
        return item

    pose = str(item.get("pose"))
    preferred_end = None
    terminal_reason = "best_local_rgbd_evidence"
    if pose == "cap_visible":
        preferred_end = int(np.argmax(endpoints_uv[:, 1]))
        terminal_reason = "away_from_solid_cap"
    elif pose == "open":
        if height_delta_m >= TERMINAL_HEIGHT_PRIORITY_DELTA_M:
            preferred_end = int(np.argmax(endpoint_heights))
            terminal_reason = "higher_ground_clearance"
        else:
            preferred_end = int(np.argmin(endpoints_uv[:, 1]))
            terminal_reason = "upper_semantic_terminal"

    chosen = None
    if preferred_end is not None:
        chosen = next(
            (
                variant
                for variant in safe_variants
                if variant["end_index"] == preferred_end
            ),
            None,
        )
    if chosen is None:
        chosen = max(
            safe_variants,
            key=lambda variant: variant["evidence_key"],
        )
        if preferred_end is not None:
            terminal_reason += "_fallback_to_supported_end"

    points_3d = chosen["points_3d"]
    points_uv = chosen["points_uv"]
    metrics = chosen["metrics"]
    post_heights = points_3d @ normal + offset
    post_height_delta_m = abs(float(post_heights[1] - post_heights[0]))
    post_depth_delta_m = abs(float(points_3d[1, 2] - points_3d[0, 2]))
    actual_fraction = float(chosen["t1"] - chosen["t0"])
    support_before = int(item.get("n_support", 0))
    support_after = int(chosen["local_depth_points"])

    item["endpoints"] = points_3d.tolist()
    item["endpoints_uv"] = points_uv.tolist()
    item["length_m"] = float(chosen["length_m"])
    item["height_above_ground"] = float(
        0.5 * (post_heights[0] + post_heights[1])
    )
    item["n_support"] = support_after
    vector_uv = points_uv[1] - points_uv[0]
    line_angle = (
        math.degrees(
            math.atan2(float(vector_uv[1]), float(vector_uv[0]))
        )
        % 180.0
    )
    item_debug.update(
        {
            "terminal_crop_applied": True,
            "terminal_fraction": round(actual_fraction, 4),
            "terminal_end": (
                "endpoint_0"
                if chosen["end_index"] == 0
                else "endpoint_1"
            ),
            "terminal_end_reason": terminal_reason,
            "terminal_parameter_range": [
                round(float(chosen["t0"]), 5),
                round(float(chosen["t1"]), 5),
            ],
            "post_crop_length_m": round(float(chosen["length_m"]), 5),
            "post_crop_length_px": round(
                float(metrics["length_px"]), 2
            ),
            "post_crop_endpoint_ground_heights_m": [
                round(float(value), 5) for value in post_heights
            ],
            "post_crop_endpoint_height_delta_mm": round(
                post_height_delta_m * 1000.0, 2
            ),
            "post_crop_endpoint_depth_delta_mm": round(
                post_depth_delta_m * 1000.0, 2
            ),
            "height_delta_reduction_ratio": round(
                post_height_delta_m / max(height_delta_m, 1e-9), 4
            ),
            "depth_delta_reduction_ratio": round(
                post_depth_delta_m / max(depth_delta_m, 1e-9), 4
            ),
            "pre_crop_n_support": support_before,
            "terminal_direct_valid_fraction": round(
                float(chosen["direct_valid_fraction"]), 3
            ),
            "terminal_local_depth_points": support_after,
            "terminal_local_depth_coverage": round(
                float(chosen["local_depth_coverage"]), 3
            ),
            "edge_rms_px": round(
                float(np.sqrt(np.mean(metrics["values"] ** 2))), 2
            ),
            "edge_p90_px": round(float(metrics["p90_px"]), 2),
            "edge_within_3px": round(float(metrics["within_3px"]), 3),
            "edge_max_air_run_px": round(
                float(metrics["max_air_run_px"]), 2
            ),
            "edge_support": int(
                np.sum(metrics["values"] <= EDGE_SUPPORT_DISTANCE_PX)
            ),
            "line_angle_deg": round(line_angle, 2),
            "length_px": round(float(metrics["length_px"]), 2),
        }
    )
    if "local_depth_points" in item_debug:
        item_debug["local_depth_points"] = support_after
    if pose == "side_laid":
        asymmetry, polarity = _boundary_asymmetry(
            top["mask"],
            points_uv[0],
            points_uv[1],
            offset_px=5.0,
        )
        # 完整语义直边已经通过自由侧拓扑硬门。短窗落到网孔密集区时，局部
        # 5 px 采样会被圆孔随机拉低，不能反过来否定同一条已确认的自由边；
        # 局部值仅保留用于审计，不覆盖完整边的拓扑证据。
        item_debug["terminal_boundary_asymmetry"] = round(asymmetry, 3)
        item_debug["terminal_boundary_polarity"] = round(polarity, 3)
    item["debug"] = item_debug
    return item


def _crop_candidates_to_terminal(
    annotations,
    rgb_bgr,
    blob,
    intr,
    plane,
    maps,
    top,
    debug,
):
    if not annotations:
        return []
    edge_map, _, _ = _edge_evidence(rgb_bgr, blob)
    edge_distance = cv2.distanceTransform(
        (edge_map == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    cropped = [
        _crop_candidate_to_terminal(
            item,
            intr,
            plane,
            edge_distance,
            maps,
            blob,
            top,
        )
        for item in annotations
    ]
    rejected = [
        item
        for item in cropped
        if (item.get("debug") or {}).get("terminal_publish_reject")
    ]
    cropped = [
        item
        for item in cropped
        if not (item.get("debug") or {}).get("terminal_publish_reject")
    ]
    debug["terminal_crop_policy"] = {
        "long_fraction": round(TERMINAL_LONG_FRACTION, 4),
        "medium_fraction": TERMINAL_MEDIUM_FRACTION,
        "minimum_length_m": TERMINAL_MIN_LENGTH_M,
        "maximum_length_m": TERMINAL_MAX_LENGTH_M,
        "minimum_length_px": TERMINAL_MIN_LENGTH_PX,
        "cap_minimum_length_px": TERMINAL_CAP_MIN_LENGTH_PX,
        "end_inset_fraction": TERMINAL_END_INSET_FRACTION,
        "depth_corridor_px": TERMINAL_DEPTH_CORRIDOR_PX,
    }
    debug["terminal_rejected_candidate_count"] = len(rejected)
    if rejected:
        debug["terminal_rejections"] = [
            {
                "ridge_id": item.get("ridge_id"),
                "reason": (item.get("debug") or {}).get(
                    "terminal_publish_reject"
                ),
            }
            for item in rejected
        ]
    return cropped


def _finalize_top1(annotations, debug):
    """严格终审候选，并只公开确定性的最高置信一条。"""
    reviewed = []
    accepted = []
    for item in annotations:
        item = dict(item)
        item_debug = dict(item.get("debug") or {})
        uv = np.asarray(item.get("endpoints_uv"), dtype=np.float64)
        length_px = (
            float(np.linalg.norm(uv[1] - uv[0]))
            if uv.shape == (2, 2)
            else 0.0
        )
        edge_p90 = float(item_debug.get("edge_p90_px", float("inf")))
        edge_within = float(item_debug.get("edge_within_3px", 0.0))
        edge_air = float(
            item_debug.get("edge_max_air_run_px", float("inf"))
        )
        rms_m = float(item.get("rms_mm", float("inf"))) / 1000.0
        n_support = int(item.get("n_support", 0))
        reasons = []
        if item.get("name") != "basket_divider_free_edge":
            reasons.append("wrong_semantic")
        if not item.get("grasp_safe", False) or item.get("inferred", True):
            reasons.append("not_direct_safe")
        if item_debug.get("terminal_crop_applied"):
            minimum_run = (
                TERMINAL_CAP_MIN_LENGTH_PX
                if item.get("pose") == "cap_visible"
                else TERMINAL_MIN_LENGTH_PX
            )
        else:
            minimum_run = (
                CAP_EDGE_SUPPORT_MIN_RUN_PX
                if item.get("pose") == "cap_visible"
                else EDGE_SUPPORT_MIN_RUN_PX
            )
        if length_px < minimum_run - 1:
            reasons.append("short_rgb_run")
        if edge_p90 > FINAL_EDGE_P90_PX:
            reasons.append("edge_p90")
        if edge_within < FINAL_EDGE_WITHIN_3PX:
            reasons.append("edge_coverage")
        if edge_air > FINAL_EDGE_MAX_AIR_PX:
            reasons.append("edge_gap")
        if n_support < FINAL_DEPTH_SUPPORT_MIN:
            reasons.append("depth_support")
        if not np.isfinite(rms_m) or rms_m > FINAL_LINE_RMS_M:
            reasons.append("depth_rms")
        if item.get("pose") == "side_laid":
            if float(item_debug.get("boundary_asymmetry", 0.0)) < 0.72:
                reasons.append("not_free_boundary")
            if float(item_debug.get("direct_depth_coverage", 0.0)) < 0.68:
                reasons.append("local_depth_coverage")
        if (
            item.get("pose") == "cap_visible"
            and float(item_debug.get("forbidden_cap_fraction", 1.0)) > 0.05
        ):
            reasons.append("crosses_solid_cap")
        if item.get("pose") in {"open", "cap_visible"}:
            family_angle = float(
                item_debug.get("family_angle_deg", float("nan"))
            )
            line_angle = float(
                item_debug.get("line_angle_deg", float("nan"))
            )
            if (
                not np.isfinite(family_angle)
                or not np.isfinite(line_angle)
                or _angle_distance_deg(family_angle, line_angle) > 7.0
            ):
                reasons.append("line_direction_outside_family")

        edge_score = float(
            np.clip(
                0.42 * (1.0 - edge_p90 / FINAL_EDGE_P90_PX)
                + 0.38
                * _ramp(
                    edge_within,
                    FINAL_EDGE_WITHIN_3PX,
                    1.0,
                )
                + 0.20
                * (1.0 - edge_air / FINAL_EDGE_MAX_AIR_PX),
                0.0,
                1.0,
            )
        )
        continuity_score = float(
            np.clip(
                0.65 * _ramp(length_px, minimum_run, 135.0)
                + 0.35
                * float(item_debug.get("visible_fraction", 1.0)),
                0.0,
                1.0,
            )
        )
        depth_score = float(
            np.clip(
                0.58 * _ramp(n_support, FINAL_DEPTH_SUPPORT_MIN, 260)
                + 0.42 * _gauss(rms_m, 0.0, 0.012),
                0.0,
                1.0,
            )
        )
        ridge_id = int(item.get("ridge_id", 0))
        interior_score = 1.0 if ridge_id in {1, 2, 3} else 0.84
        topology_score = (
            float(item_debug.get("boundary_asymmetry", 1.0))
            if item.get("pose") == "side_laid"
            else 1.0
        )
        selection_score = float(
            np.clip(
                0.34 * edge_score
                + 0.24 * continuity_score
                + 0.22 * depth_score
                + 0.12 * topology_score
                + 0.08 * interior_score,
                0.0,
                1.0,
            )
        )
        confidence = float(np.clip(0.56 + 0.42 * selection_score, 0.56, 0.98))
        item["confidence"] = confidence
        item["segment_index"] = 0
        item_debug.update(
            {
                "selection_score": round(selection_score, 5),
                "length_px": round(length_px, 2),
                "top1_hard_gate": not reasons,
                "top1_reject_reasons": reasons,
            }
        )
        item["debug"] = item_debug
        reviewed.append(
            {
                "ridge_id": ridge_id,
                "source_segment_index": int(
                    item_debug.get("source_segment_index", 0)
                ),
                "confidence": round(confidence, 5),
                "selection_score": round(selection_score, 5),
                "length_px": round(length_px, 2),
                "accepted": not reasons,
                "reject_reasons": reasons,
            }
        )
        if not reasons:
            accepted.append(item)

    accepted.sort(
        key=lambda item: (
            -float(item["confidence"]),
            -float(item["debug"]["selection_score"]),
            -float(item["debug"]["length_px"]),
            -int(item["n_support"]),
            abs(int(item.get("ridge_id", 0)) - 2),
            int(item.get("ridge_id", 0)),
            float(item["endpoints_uv"][0][0]),
            float(item["endpoints_uv"][0][1]),
        )
    )
    debug["top1_candidate_review"] = reviewed
    debug["candidate_count"] = len(annotations)
    debug["safe_candidate_count"] = len(accepted)
    if not accepted:
        debug["selection_reject"] = "no_candidate_passed_top1_hard_gate"
        debug["selected_confidence"] = None
        debug["confidence_margin_to_second"] = None
        return []

    best = accepted[0]
    best["visible_ridge_count"] = 1
    best["debug"]["selected_top1"] = True
    debug["selected_ridge_id"] = int(best.get("ridge_id", 0))
    debug["selected_source_segment_index"] = int(
        best["debug"].get("source_segment_index", 0)
    )
    debug["selected_confidence"] = round(float(best["confidence"]), 5)
    debug["confidence_margin_to_second"] = (
        None
        if len(accepted) < 2
        else round(
            float(best["confidence"]) - float(accepted[1]["confidence"]),
            5,
        )
    )
    return [best]


def _side_panel_grasps(rgb_bgr, blob, top, maps, intr, debug):
    """侧倒时，只输出朝上且真实可见的那块 L 板自由轮廓。"""
    contour = top["contour"]
    perimeter = float(cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(
        contour, 0.018 * perimeter, True
    ).reshape(-1, 2)
    concave_index = _polygon_concave_index(polygon)
    if concave_index is None or len(polygon) < 6:
        debug["side_reject"] = "no_stable_L_concavity"
        return []

    count = len(polygon)
    # 红链 = 主凹角两侧各跨两条轮廓边。另一条弧是与筐体相连的底/附着链。
    semantic_corners = np.asarray(
        [
            polygon[(concave_index + delta) % count]
            for delta in (-2, -1, 0, 1, 2)
        ],
        dtype=np.float64,
    )
    if len(semantic_corners) < 2:
        debug["side_reject"] = "free_chain_too_short"
        return []

    contour_arc = _contour_arc(
        contour,
        semantic_corners[0],
        semantic_corners[-1],
        semantic_corners[2],
    )
    edge_map, _, _ = _edge_evidence(rgb_bgr, blob)
    edge_distance = cv2.distanceTransform(
        (edge_map == 0).astype(np.uint8), cv2.DIST_L2, 3
    )

    # 直的语义边仍只画一段；圆角若用一个弦连接会离开真实塑料轮廓，因此只对
    # 不通过逐段 RGB 贴边门的边做局部 RDP，并保留真正有边缘支撑的小段。
    arc_indices = []
    for corner in semantic_corners:
        arc_indices.append(
            int(np.argmin(np.linalg.norm(contour_arc - corner, axis=1)))
        )
    if any(
        right <= left for left, right in zip(arc_indices, arc_indices[1:])
    ):
        debug["side_reject"] = "unstable_free_chain_order"
        return []
    segment_uv = []
    for semantic_index, (left, right) in enumerate(
        zip(arc_indices, arc_indices[1:])
    ):
        p0 = semantic_corners[semantic_index]
        p1 = semantic_corners[semantic_index + 1]
        supported, _ = _line_is_tightly_supported(edge_distance, p0, p1)
        if supported:
            segment_uv.append((p0, p1))
            continue
        local_arc = contour_arc[left : right + 1]
        local_polyline = cv2.approxPolyDP(
            local_arc.astype(np.float32).reshape(-1, 1, 2), 1.7, False
        ).reshape(-1, 2)
        for q0, q1 in zip(local_polyline, local_polyline[1:]):
            supported, metrics = _line_is_tightly_supported(
                edge_distance, q0, q1
            )
            if (
                supported
                and metrics["length_px"] >= 7.0
            ):
                segment_uv.append(
                    (
                        np.asarray(q0, dtype=np.float64),
                        np.asarray(q1, dtype=np.float64),
                    )
                )
    if not segment_uv:
        debug["side_reject"] = "no_rgb_supported_free_chain"
        return []

    vv, uu = np.nonzero((top["mask"] > 0) & maps["valid"])
    points = np.column_stack(
        [maps["x"][vv, uu], maps["y"][vv, uu], maps["z"][vv, uu]]
    )
    plane_fit = _fit_plane(points)
    if plane_fit is None:
        debug["side_reject"] = "panel_plane_unavailable"
        return []
    center, normal, plane_rms, keep = plane_fit
    debug["panel_plane_rms_mm"] = round(plane_rms * 1000.0, 2)
    debug["panel_plane_points"] = int(keep.sum())
    if plane_rms > 0.018:
        debug["side_reject"] = "panel_plane_rms"
        return []

    panel_uv = np.column_stack([uu, vv]).astype(np.float64)[keep]
    annotations = []
    total_length = 0.0
    for source_segment_index, (raw_uv0, raw_uv1) in enumerate(segment_uv):
        raw_uv0 = np.asarray(raw_uv0, dtype=np.float64)
        raw_uv1 = np.asarray(raw_uv1, dtype=np.float64)
        raw_length_px = float(np.linalg.norm(raw_uv1 - raw_uv0))
        if raw_length_px < EDGE_SUPPORT_MIN_RUN_PX - 1:
            continue
        uv0, uv1, normal_snap_px = _snap_line_to_rgb_edge(
            edge_distance,
            raw_uv0,
            raw_uv1,
            max_offset_px=4.0,
        )
        uv0, uv1, edge_metrics, was_trimmed = (
            _trim_to_supported_interval(
                edge_distance,
                uv0,
                uv1,
                min_run_px=EDGE_SUPPORT_MIN_RUN_PX,
            )
        )
        if uv0 is None or uv1 is None:
            continue
        length_px = float(np.linalg.norm(uv1 - uv0))
        boundary_asymmetry, boundary_polarity = _boundary_asymmetry(
            top["mask"], uv0, uv1, offset_px=5.0
        )

        vector = uv1 - uv0
        direction = vector / max(length_px, 1e-9)
        line_normal = np.array([-direction[1], direction[0]])
        relative = panel_uv - uv0
        along = relative @ direction
        lateral = np.abs(relative @ line_normal)
        local = (
            (along >= -3.0)
            & (along <= length_px + 3.0)
            # 网孔和深度对齐在轮廓处会留下 10~20 px 空带；只向同一已拟合
            # 板面的内部取不超过 32 px 的直接点，仍远小于相邻板间距。
            & (lateral <= 32.0)
        )
        if int(local.sum()) < FINAL_DEPTH_SUPPORT_MIN:
            continue
        local_along = along[local]
        direct_coverage = float(
            np.clip(
                (
                    np.percentile(local_along, 97)
                    - np.percentile(local_along, 3)
                )
                / max(length_px, 1.0),
                0.0,
                1.0,
            )
        )
        if direct_coverage < 0.68:
            continue

        p0 = _ray_plane_point(uv0, intr, center, normal)
        p1 = _ray_plane_point(uv1, intr, center, normal)
        if p0 is None or p1 is None:
            continue
        length_m = float(np.linalg.norm(p1 - p0))
        if not (MIN_LINE_M <= length_m <= 0.40):
            continue
        total_length += length_m
        midpoint = 0.5 * (p0 + p1)
        ground_normal = np.asarray(debug["ground_normal"], dtype=np.float64)
        height = float(
            midpoint @ ground_normal + float(debug["ground_offset"])
        )
        annotations.append(
            {
                "type": "line",
                "name": "basket_divider_free_edge",
                "ridge_id": 0,
                "segment_index": 0,
                "visible_ridge_count": 1,
                "pose": "side_laid",
                "endpoints": [p0.tolist(), p1.tolist()],
                "endpoints_uv": [
                    uv0.tolist(),
                    uv1.tolist(),
                ],
                "length_m": length_m,
                "confidence": 0.0,
                "orientation": "up",
                "height_above_ground": height,
                "rms_mm": float(plane_rms * 1000.0),
                "n_support": int(local.sum()),
                "grasp_safe": True,
                "inferred": False,
                "debug": {
                    "semantic": (
                        "one_direct_RGBD_segment_of_visible_L_panel_free_edge"
                    ),
                    "panel_plane_rms_mm": round(plane_rms * 1000.0, 2),
                    "lift_source": "direct_local_panel_plane",
                    "source_segment_index": int(source_segment_index),
                    "normal_snap_px": round(float(normal_snap_px), 2),
                    "edge_p90_px": round(
                        float(edge_metrics["p90_px"]), 2
                    ),
                    "edge_within_3px": round(
                        float(edge_metrics["within_3px"]), 3
                    ),
                    "edge_max_air_run_px": round(
                        float(edge_metrics["max_air_run_px"]), 2
                    ),
                    "edge_trimmed": bool(was_trimmed),
                    "visible_fraction": round(
                        length_px / max(raw_length_px, 1.0), 3
                    ),
                    "boundary_asymmetry": round(
                        boundary_asymmetry, 3
                    ),
                    "boundary_polarity": round(boundary_polarity, 3),
                    "direct_depth_coverage": round(
                        direct_coverage, 3
                    ),
                    "local_depth_points": int(local.sum()),
                },
            }
        )
    debug["side_polygon_vertices"] = polygon.astype(float).tolist()
    debug["side_concave_index"] = int(concave_index)
    debug["side_semantic_corners"] = semantic_corners.astype(float).tolist()
    debug["side_candidate_segment_count"] = len(segment_uv)
    debug["side_safe_straight_candidate_count"] = len(annotations)
    debug["side_total_arc_m"] = round(total_length, 3)
    return annotations


def _smooth_profile(values, sigma=2.0):
    radius = max(2, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def _profile_peaks(profile, min_distance, min_prominence):
    candidates = []
    for index in range(1, len(profile) - 1):
        if profile[index] < profile[index - 1] or profile[index] <= profile[index + 1]:
            continue
        left = float(np.min(profile[max(0, index - min_distance) : index + 1]))
        right = float(
            np.min(profile[index : min(len(profile), index + min_distance + 1)])
        )
        if profile[index] - max(left, right) >= min_prominence:
            candidates.append(index)
    candidates.sort(key=lambda index: float(profile[index]), reverse=True)
    selected = []
    for index in candidates:
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    return np.asarray(sorted(selected), dtype=np.int32)


def _edge_evidence(rgb_bgr, blob):
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    blue, _, red = cv2.split(rgb_bgr)
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    blue_score = np.clip(
        2 * (blue.astype(np.int16) - red.astype(np.int16))
        + hsv[:, :, 1].astype(np.int16) // 2,
        0,
        255,
    ).astype(np.uint8)
    blue_score = cv2.GaussianBlur(blue_score, (3, 3), 0)
    edges = cv2.Canny(gray, 28, 86) | cv2.Canny(blue_score, 24, 72)
    region = cv2.dilate(
        (blob > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    edges &= region

    # 梯度法向旋转 90° 后就是局部边线方向，后面用它排除交叉孔边。
    gx = cv2.Sobel(blue_score, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blue_score, cv2.CV_32F, 0, 1, ksize=3)
    tangent = (
        np.degrees(np.arctan2(gy, gx)).astype(np.float64) + 90.0
    ) % 180.0
    return edges, tangent, blue_score


def _family_geometry(candidate, blob, maps, center_uv, top):
    angle = float(candidate["angle"])
    theta = math.radians(angle)
    normal2 = np.array([-math.sin(theta), math.cos(theta)])
    vv, uu = np.nonzero((blob > 0) & maps["valid"])
    if len(uu) < 100:
        return 0.0, {}
    rho = (uu - center_uv[0]) * normal2[0] + (vv - center_uv[1]) * normal2[1]
    heights = maps["height"][vv, uu]
    middle_height = top["base"] + 0.50 * max(top["span"], 0.10)

    height_spans = []
    middle_centers = []
    middle_counts = []
    for offset in candidate["offsets"]:
        selected = np.abs(rho - float(offset)) < 4.0
        values = heights[selected]
        if len(values) >= 20:
            height_spans.append(
                float(np.nanpercentile(values, 90) - np.nanpercentile(values, 10))
            )
        else:
            height_spans.append(0.0)
        at_middle = selected & (np.abs(heights - middle_height) < 0.042)
        middle_counts.append(int(at_middle.sum()))
        if int(at_middle.sum()) >= 20:
            points = np.column_stack(
                [
                    maps["x"][vv[at_middle], uu[at_middle]],
                    maps["y"][vv[at_middle], uu[at_middle]],
                    maps["z"][vv[at_middle], uu[at_middle]],
                ]
            )
            middle_centers.append(
                (len(middle_centers), np.median(points, axis=0))
            )
        else:
            middle_centers.append((len(middle_centers), None))

    span_scores = [_ramp(value, 0.055, 0.18) for value in height_spans]
    height_score = float(np.mean(span_scores))
    broad_count = int(sum(value >= 0.075 for value in height_spans))

    valid_centers = [
        (index, point) for index, point in middle_centers if point is not None
    ]
    spacing_score = 0.0
    hard_geometry = False
    spacing_debug = {}
    if len(valid_centers) >= 4:
        indices = np.asarray([item[0] for item in valid_centers], dtype=np.float64)
        points = np.asarray([item[1] for item in valid_centers], dtype=np.float64)
        centered = points - points.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        locations = centered @ direction
        if np.corrcoef(indices, locations)[0, 1] < 0:
            locations = -locations
        order = np.argsort(indices)
        indices = indices[order]
        locations = locations[order]
        gaps = np.diff(locations)
        index_steps = np.diff(indices)
        normalized_gaps = gaps / np.maximum(index_steps, 1.0)
        positive = normalized_gaps[normalized_gaps > 0.010]
        spread = float(np.ptp(locations))
        if len(positive) >= 2:
            cv_gap = float(np.std(positive) / max(np.mean(positive), 1e-6))
            gap_score = float(
                np.mean([_gauss(gap, DIVIDER_GAP_M, 0.035) for gap in positive])
            )
            spacing_score = (
                0.42 * _gauss(spread, BASKET_PANEL_SPAN_M, 0.085)
                + 0.38 * gap_score
                + 0.20 * math.exp(-cv_gap / 0.55)
            )
            gap_max = 0.132 if len(valid_centers) == 4 else 0.112
            hard_geometry = bool(
                broad_count == 5
                and 0.235 <= spread <= 0.355
                and bool(np.all(positive >= 0.038))
                and bool(np.all(positive <= gap_max))
                and cv_gap <= 0.30
            )
            spacing_debug = {
                "center_count": len(valid_centers),
                "center_indices": indices.astype(int).tolist(),
                "spread_m": round(spread, 3),
                "gaps_m": [round(float(value), 3) for value in gaps],
                "normalized_gaps_m": [
                    round(float(value), 3) for value in normalized_gaps
                ],
                "gap_cv": round(cv_gap, 3),
                "gap_max_m": gap_max,
                "hard_geometry": hard_geometry,
            }

    if not hard_geometry:
        geometry_score = 0.0
    else:
        geometry_score = (0.38 + 0.62 * height_score) * (
            0.30 + 0.70 * spacing_score
        )
    return float(np.clip(geometry_score, 0.0, 1.0)), {
        "height_spans_m": [round(value, 3) for value in height_spans],
        "middle_counts": middle_counts,
        "broad_count": broad_count,
        "hard_geometry": hard_geometry,
        **spacing_debug,
    }


def _family_candidates(edges, blob, bbox, maps, top, forced_angle=None):
    ys, xs = np.nonzero(edges > 0)
    by, bx = np.nonzero(blob > 0)
    if len(xs) < 80 or len(bx) < 100:
        return []
    x, y, width, height = bbox
    center_uv = np.array([x + width / 2.0, y + height / 2.0])
    edge_xy = np.column_stack([xs, ys]).astype(np.float64) - center_uv
    blob_xy = np.column_stack([bx, by]).astype(np.float64) - center_uv

    candidates = []
    for angle in range(180):
        if forced_angle is not None and _angle_distance_deg(angle, forced_angle) > 34:
            continue
        theta = math.radians(angle)
        normal = np.array([-math.sin(theta), math.cos(theta)])
        rho = edge_xy @ normal
        blob_rho = blob_xy @ normal
        lo = float(math.floor(float(blob_rho.min())) - 2)
        hi = float(math.ceil(float(blob_rho.max())) + 2)
        count = int(hi - lo + 1)
        bins = np.clip(np.rint(rho - lo).astype(np.int32), 0, count - 1)
        profile = np.bincount(bins, minlength=count).astype(np.float64)
        profile = _smooth_profile(profile, 2.0)
        span = float(blob_rho.max() - blob_rho.min())
        min_distance = max(6, int(0.085 * span))
        peaks = _profile_peaks(
            profile,
            min_distance=min_distance,
            min_prominence=max(2.0, 0.045 * float(profile.max())),
        )
        if len(peaks) < 5:
            continue

        best = None
        for start, end in itertools.combinations(peaks, 2):
            spread = float(end - start)
            if not (0.56 * span <= spread <= 1.08 * span):
                continue
            step = spread / 4.0
            chosen = []
            regularity_error = 0.0
            for ridge_index in range(5):
                target = start + ridge_index * step
                nearest = int(peaks[int(np.argmin(np.abs(peaks - target)))])
                error = abs(nearest - target) / max(step, 1e-6)
                if error > 0.30:
                    chosen = []
                    break
                chosen.append(nearest)
                regularity_error += error
            if len(set(chosen)) != 5:
                continue

            strengths = np.asarray([profile[index] for index in chosen])
            median_strength = float(np.median(strengths))
            strong_peak_count = int(
                sum(profile[index] > 0.45 * median_strength for index in peaks)
            )
            if strong_peak_count < 5:
                continue
            balance = float(
                np.percentile(strengths, 20)
                / max(np.percentile(strengths, 80), 1e-6)
            )
            structural = (
                float(strengths.mean())
                / max(float(max(width, height)), 1.0)
                * (spread / max(span, 1.0))
                * math.exp(-5.0 * regularity_error / 5.0)
                * (0.70 + 0.30 * balance)
                / (1.0 + 0.25 * max(0, strong_peak_count - 5))
            )
            item = {
                "angle": float(angle),
                "offsets": [float(lo + index) for index in chosen],
                "structural": float(structural),
                "regularity_error": float(regularity_error),
                "strong_peak_count": strong_peak_count,
                "balance": balance,
            }
            if best is None or structural > best["structural"]:
                best = item
        if best is not None:
            candidates.append(best)

    candidates.sort(key=lambda item: -item["structural"])
    # 先用图像结构筛到少量方向，再做较贵的逐线 3D 验证。
    candidates = candidates[:45]
    for candidate in candidates:
        geometry, geometry_debug = _family_geometry(
            candidate, blob, maps, center_uv, top
        )
        candidate["geometry"] = geometry
        candidate["geometry_debug"] = geometry_debug
        candidate["score"] = candidate["structural"] * (
            0.18 + 0.82 * geometry
        )
    candidates.sort(key=lambda item: -item["score"])
    return candidates


def _cap_target_angle(top):
    contour = top["contour"].astype(np.float32)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float64)
    vectors = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(vectors, axis=1)
    direction = vectors[int(np.argmax(lengths))]
    cap_angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
    return (cap_angle + 90.0) % 180.0, cap_angle


def _fit_2d_ridge(
    candidate,
    ridge_index,
    edges,
    edge_distance,
    tangent,
    blob,
    bbox,
    top,
    maps,
    allowed_mask=None,
):
    angle = float(candidate["angle"])
    theta = math.radians(angle)
    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    x, y, width, height = bbox
    center = np.array([x + width / 2.0, y + height / 2.0])
    offset = float(candidate["offsets"][ridge_index])
    gaps = np.diff(candidate["offsets"])
    search_width = max(5.0, 0.18 * float(np.median(gaps)))

    vv, uu = np.nonzero(edges > 0)
    xy = np.column_stack([uu, vv]).astype(np.float64)
    relative = xy - center
    rho = relative @ normal
    tangent_values = tangent[vv, uu]
    angle_delta = np.asarray(
        [_angle_distance_deg(value, angle) for value in tangent_values]
    )
    selected = (np.abs(rho - offset) <= search_width) & (angle_delta <= 23.0)
    points = xy[selected]
    if len(points) < 16:
        selected = np.abs(rho - offset) <= max(4.0, 0.12 * float(np.median(gaps)))
        points = xy[selected]
    if len(points) < 14:
        return None

    # 两遍正交最小二乘；方向偏离族方向过多就拒绝，防止吸到孔洞交叉边。
    keep = np.ones(len(points), dtype=bool)
    fitted_direction = direction.copy()
    fitted_center = points.mean(axis=0)
    for _ in range(3):
        sample = points[keep]
        fitted_center = sample.mean(axis=0)
        matrix = sample - fitted_center
        values, vectors = np.linalg.eigh(matrix.T @ matrix)
        fitted_direction = vectors[:, int(np.argmax(values))]
        fitted_angle = (
            math.degrees(
                math.atan2(
                    float(fitted_direction[1]), float(fitted_direction[0])
                )
            )
            % 180.0
        )
        if _angle_distance_deg(fitted_angle, angle) > 26.0:
            return None
        fitted_normal = np.array([-fitted_direction[1], fitted_direction[0]])
        residual = np.abs((points - fitted_center) @ fitted_normal)
        threshold = max(2.2, float(np.percentile(residual[keep], 68)))
        new_keep = residual <= min(threshold, 4.2)
        if int(new_keep.sum()) < 12:
            break
        keep = new_keep

    support = points[keep]
    if len(support) < 12:
        return None
    if float(fitted_direction @ direction) < 0:
        fitted_direction = -fitted_direction
    locations = (support - fitted_center) @ fitted_direction

    # 只允许抓离物体自身底部至少约 3.5 cm 的自由轮廓，防止延伸进底梁。
    sample_u = np.clip(np.rint(support[:, 0]).astype(int), 0, blob.shape[1] - 1)
    sample_v = np.clip(np.rint(support[:, 1]).astype(int), 0, blob.shape[0] - 1)
    support_heights = maps["height"][sample_v, sample_u]
    high_enough = np.isfinite(support_heights) & (
        support_heights >= top["base"] + 0.032
    )
    if int(high_enough.sum()) < 10:
        return None
    support = support[high_enough]
    locations = (support - fitted_center) @ fitted_direction
    if len(locations) < 10:
        return None

    # 删除离主体最远的少量共线孔边；端点仍落在实际 RGB 边缘支撑上。
    lo, hi = np.percentile(locations, [3, 97])
    if hi - lo < MIN_LINE_PX:
        return None
    p0 = fitted_center + lo * fitted_direction
    p1 = fitted_center + hi * fitted_direction
    raw_length_px = float(np.linalg.norm(p1 - p0))

    image_height, image_width = blob.shape
    p0[0] = np.clip(p0[0], 1, image_width - 2)
    p0[1] = np.clip(p0[1], 1, image_height - 2)
    p1[0] = np.clip(p1[0], 1, image_width - 2)
    p1[1] = np.clip(p1[1], 1, image_height - 2)
    p0, p1, normal_snap_px = _snap_line_to_rgb_edge(
        edge_distance, p0, p1
    )
    p0, p1, edge_metrics, was_trimmed = _trim_to_supported_interval(
        edge_distance,
        p0,
        p1,
        min_run_px=(
            CAP_EDGE_SUPPORT_MIN_RUN_PX
            if top["pose"] == "cap_visible"
            else EDGE_SUPPORT_MIN_RUN_PX
        ),
        allowed_mask=allowed_mask,
    )
    if p0 is None or p1 is None:
        return None

    trimmed_length = float(np.linalg.norm(p1 - p0))
    final_direction = (p1 - p0) / max(trimmed_length, 1e-9)
    support_along = (support - p0) @ final_direction
    local_support = (
        (support_along >= -3.0)
        & (support_along <= trimmed_length + 3.0)
    )
    if int(local_support.sum()) >= 10:
        support = support[local_support]
    residual = np.abs(
        (support - p0)
        @ np.array([-final_direction[1], final_direction[0]])
    )
    return {
        "p0": p0,
        "p1": p1,
        "support": support,
        "support_count": int(len(support)),
        "edge_rms_px": float(np.sqrt(np.mean(residual**2))),
        "edge_p90_px": float(edge_metrics["p90_px"]),
        "edge_within_3px": float(edge_metrics["within_3px"]),
        "edge_max_air_run_px": float(edge_metrics["max_air_run_px"]),
        "edge_trimmed": bool(was_trimmed),
        "visible_fraction": float(
            np.clip(trimmed_length / max(raw_length_px, 1.0), 0.0, 1.0)
        ),
        "normal_snap_px": float(normal_snap_px),
        "angle": float(
            math.degrees(
                math.atan2(float(final_direction[1]), float(final_direction[0]))
            )
            % 180.0
        ),
    }


def _local_panel_plane_lift(
    p0,
    p1,
    along,
    lateral,
    points,
    intr,
    seed,
):
    """从该棱附近的直接蓝色点拟合板面，再与两个端点射线求交。"""
    if len(points) < 24 or float(np.ptp(along)) < 0.45:
        return None
    rng = np.random.RandomState(seed ^ 0x51A7C3)
    best_keep = None
    best_score = -float("inf")
    trial_count = min(220, max(80, 5 * len(points)))
    for _ in range(trial_count):
        sample = rng.choice(len(points), 3, replace=False)
        if float(np.ptp(along[sample])) < 0.34:
            continue
        if float(np.ptp(lateral[sample])) < 1.2:
            continue
        a, b, c = points[sample]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-7:
            continue
        normal /= norm
        residual = np.abs((points - a) @ normal)
        inliers = residual <= 0.012
        if int(inliers.sum()) < 16:
            continue
        coverage = float(np.ptp(along[inliers]))
        if coverage < 0.45:
            continue
        lateral_span = float(np.ptp(lateral[inliers]))
        score = (
            float(inliers.sum())
            + 24.0 * min(coverage, 1.0)
            + 2.0 * min(lateral_span, 8.0)
            - 900.0 * float(np.median(residual[inliers]))
        )
        if score > best_score:
            best_keep = inliers
            best_score = score
    if best_keep is None:
        return None

    keep = best_keep
    center = normal = None
    for _ in range(3):
        sample = points[keep]
        center = np.median(sample, axis=0)
        _, _, vh = np.linalg.svd(sample - center, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal) + 1e-12
        residual = np.abs((points - center) @ normal)
        threshold = float(
            np.clip(np.percentile(residual[keep], 82) * 1.7, 0.006, 0.017)
        )
        new_keep = residual <= threshold
        if int(new_keep.sum()) < 16:
            break
        keep = new_keep
    if center is None or int(keep.sum()) < 16:
        return None
    coverage = float(np.ptp(along[keep]))
    if coverage < 0.42:
        return None
    plane_residual = np.abs((points[keep] - center) @ normal)
    plane_rms = float(np.sqrt(np.mean(plane_residual**2)))
    if plane_rms > 0.017:
        return None

    endpoint0 = _ray_plane_point(p0, intr, center, normal)
    endpoint1 = _ray_plane_point(p1, intr, center, normal)
    if endpoint0 is None or endpoint1 is None:
        return None
    depths = points[keep, 2]
    depth_lo = float(np.percentile(depths, 2)) - 0.12
    depth_hi = float(np.percentile(depths, 98)) + 0.12
    if not (
        depth_lo <= endpoint0[2] <= depth_hi
        and depth_lo <= endpoint1[2] <= depth_hi
    ):
        return None
    length_m = float(np.linalg.norm(endpoint1 - endpoint0))
    if not (MIN_LINE_M <= length_m <= MAX_LINE_M):
        return None

    direction3 = endpoint1 - endpoint0
    direction3 /= np.linalg.norm(direction3) + 1e-12
    line_residual = np.linalg.norm(
        (points[keep] - endpoint0)
        - np.outer((points[keep] - endpoint0) @ direction3, direction3),
        axis=1,
    )
    line_rms = float(np.sqrt(np.mean(line_residual**2)))
    if line_rms > MAX_LINE_RMS_M:
        return None
    return {
        "p0": endpoint0,
        "p1": endpoint1,
        "length_m": length_m,
        "rms_m": line_rms,
        "n_support": int(keep.sum()),
        "valid_fraction": float(keep.sum()) / max(len(points), 1),
        "lift_source": "direct_local_panel_plane_ransac",
        "plane_rms_m": plane_rms,
        "coverage": coverage,
    }


def _lift_ridge(line, depth_mm, intr, top, blob3d):
    """用已经剔除地面穿透点的筐体点，把贴边的 2D 线抬到 3D。

    文件筐的网孔会让边线上大量原始深度落到地面；直接在 7x7 patch 里取分位数
    会把正确的二维棱误判成弯曲线。``blob3d`` 已经同时通过蓝色连通域、近端
    深度窗和离地高度过滤，因此这里只在该点集中拟合 ``1/z`` 随图像线参数的
    仿射关系。这个关系是透视投影下三维直线的精确形式。
    """
    lift_debug = line.setdefault("_lift_debug", {})
    p0 = np.asarray(line["p0"], dtype=np.float64)
    p1 = np.asarray(line["p1"], dtype=np.float64)
    vector = p1 - p0
    length_px = float(np.linalg.norm(vector))
    if length_px < MIN_LINE_PX:
        lift_debug["reject"] = "short_2d_line"
        return None
    direction2 = vector / length_px
    normal2 = np.array([-direction2[1], direction2[0]])
    uv_all = np.asarray(blob3d["uv"], dtype=np.float64)
    points_all = np.asarray(blob3d["pts"], dtype=np.float64)
    height_all = np.asarray(blob3d["h"], dtype=np.float64)
    relative = uv_all - p0
    along = relative @ direction2
    signed_lateral = relative @ normal2
    lateral = np.abs(signed_lateral)

    # 第一遍保持窄带，避免相邻板；远景时至少保留 4 px。
    band_px = float(np.clip(0.055 * length_px, 4.0, 8.0))
    selected = (
        (along >= -0.04 * length_px)
        & (along <= 1.04 * length_px)
        & (lateral <= band_px)
        & (height_all >= top["base"] + 0.020)
    )
    if int(selected.sum()) < 16:
        selected = (
            (along >= -0.06 * length_px)
            & (along <= 1.06 * length_px)
            & (lateral <= min(12.0, 1.7 * band_px))
            & (height_all >= top["base"] + 0.015)
        )
    if int(selected.sum()) < 10:
        lift_debug["selected_points"] = int(selected.sum())
        lift_debug["reject"] = "insufficient_height_filtered_points"
        return None

    along = along[selected] / length_px
    selected_lateral = signed_lateral[selected]
    points = points_all[selected]
    inv_z = 1.0 / np.maximum(points[:, 2], 1e-6)
    lift_debug["selected_points"] = int(len(points))
    lift_debug["selected_coverage"] = round(float(np.ptp(along)), 3)

    # 小规模确定性 RANSAC。评分同时奖励覆盖范围，防止只拟合到一个孔角的小簇。
    seed = int(round(p0[0] * 13.0 + p0[1] * 17.0)) & 0xFFFFFFFF
    plane_lift = _local_panel_plane_lift(
        p0,
        p1,
        along,
        selected_lateral,
        points,
        intr,
        seed,
    )
    rng = np.random.RandomState(seed)
    best_keep = None
    best_model = None
    best_score = -1.0
    unconstrained_best_score = -1.0
    unconstrained_best_length = None
    trial_count = min(140, max(45, 4 * len(points)))
    for _ in range(trial_count):
        pair = rng.choice(len(points), 2, replace=False)
        t0, t1 = along[pair]
        if abs(t1 - t0) < 0.18:
            continue
        slope = float((inv_z[pair[1]] - inv_z[pair[0]]) / (t1 - t0))
        intercept = float(inv_z[pair[0]] - slope * t0)
        predicted_inv = slope * along + intercept
        valid_prediction = predicted_inv > 0.15
        predicted_z = np.full(len(points), np.inf, dtype=np.float64)
        predicted_z[valid_prediction] = 1.0 / predicted_inv[valid_prediction]
        residual_z = np.abs(points[:, 2] - predicted_z)
        inliers = residual_z <= 0.020
        if int(inliers.sum()) < 8:
            continue
        coverage = float(np.ptp(along[inliers]))
        trial_endpoint_depths = 1.0 / np.maximum(
            slope * np.array([0.0, 1.0]) + intercept, 0.15
        )
        trial_endpoint0 = _unproject(
            p0[0], p0[1], trial_endpoint_depths[0], intr
        )
        trial_endpoint1 = _unproject(
            p1[0], p1[1], trial_endpoint_depths[1], intr
        )
        trial_length_m = float(
            np.linalg.norm(trial_endpoint1 - trial_endpoint0)
        )
        score = float(inliers.sum()) + 18.0 * min(coverage, 1.0)
        if coverage >= 0.40 and score > unconstrained_best_score:
            unconstrained_best_score = score
            unconstrained_best_length = trial_length_m
        # 多孔筐会同时出现本板和孔后背景两个深度层。只允许物理尺寸可解释
        # 的直接模型参加评分，避免“先选错背景层、最后才因过长拒绝”。
        if not (MIN_LINE_M <= trial_length_m <= MAX_LINE_M):
            continue
        if coverage >= 0.40 and score > best_score:
            best_keep = inliers
            best_model = (slope, intercept)
            best_score = score
    if best_keep is None:
        lift_debug["reject"] = "inverse_depth_ransac"
        if plane_lift is not None:
            lift_debug["fallback"] = "direct_local_panel_plane_ransac"
            return plane_lift
        return None

    keep = best_keep
    for _ in range(3):
        matrix = np.column_stack([along[keep], np.ones(int(keep.sum()))])
        slope, intercept = np.linalg.lstsq(
            matrix, inv_z[keep], rcond=None
        )[0]
        predicted_inv = slope * along + intercept
        valid_prediction = predicted_inv > 0.15
        predicted_z = np.full(len(points), np.inf, dtype=np.float64)
        predicted_z[valid_prediction] = 1.0 / predicted_inv[valid_prediction]
        residual_z = np.abs(points[:, 2] - predicted_z)
        threshold = float(
            np.clip(np.percentile(residual_z[keep], 78) * 1.8, 0.010, 0.026)
        )
        new_keep = residual_z <= threshold
        if int(new_keep.sum()) < 8:
            break
        keep = new_keep
    if int(keep.sum()) < 8 or float(np.ptp(along[keep])) < 0.36:
        coverage = float(np.ptp(along[keep])) if int(keep.sum()) else 0.0
        lift_debug["inliers"] = int(keep.sum())
        lift_debug["inlier_coverage"] = round(coverage, 3)
        lift_debug["reject"] = "insufficient_inlier_coverage"
        # 有些外侧板只有一小段能返回深度。只发布这段真实支持区，不对整条
        # 可见 RGB 边做深度外推；24 px 是机械抓取线的最小像素尺度。
        min_direct_coverage = EDGE_SUPPORT_MIN_RUN_PX / max(length_px, 1.0)
        if (
            int(keep.sum()) >= 8
            and coverage >= max(0.18, min_direct_coverage)
        ):
            t0, t1 = np.percentile(along[keep], [2, 98])
            uv0 = p0 + float(t0) * vector
            uv1 = p0 + float(t1) * vector
            depths = 1.0 / np.maximum(
                slope * np.asarray([t0, t1]) + intercept, 0.15
            )
            endpoint0 = _unproject(uv0[0], uv0[1], depths[0], intr)
            endpoint1 = _unproject(uv1[0], uv1[1], depths[1], intr)
            trimmed_length_m = float(np.linalg.norm(endpoint1 - endpoint0))
            if MIN_LINE_M <= trimmed_length_m <= MAX_LINE_M:
                direction3 = endpoint1 - endpoint0
                direction3 /= np.linalg.norm(direction3) + 1e-12
                residual_3d = np.linalg.norm(
                    (points[keep] - endpoint0)
                    - np.outer(
                        (points[keep] - endpoint0) @ direction3, direction3
                    ),
                    axis=1,
                )
                rms = float(np.sqrt(np.mean(residual_3d**2)))
                if rms <= MAX_LINE_RMS_M:
                    lift_debug["fallback"] = "direct_supported_subsegment"
                    return {
                        "p0": endpoint0,
                        "p1": endpoint1,
                        "uv0": uv0,
                        "uv1": uv1,
                        "length_m": trimmed_length_m,
                        "rms_m": rms,
                        "n_support": int(keep.sum()),
                        "valid_fraction": float(keep.sum())
                        / max(len(points), 1),
                        "lift_source": (
                            "direct_height_filtered_supported_subsegment"
                        ),
                    }
        if plane_lift is not None:
            lift_debug["fallback"] = "direct_local_panel_plane_ransac"
            return plane_lift
        return None

    endpoint_depths = 1.0 / np.maximum(
        slope * np.array([0.0, 1.0]) + intercept, 0.15
    )
    endpoint0 = _unproject(p0[0], p0[1], endpoint_depths[0], intr)
    endpoint1 = _unproject(p1[0], p1[1], endpoint_depths[1], intr)
    length_m = float(np.linalg.norm(endpoint1 - endpoint0))
    if not (MIN_LINE_M <= length_m <= MAX_LINE_M) and best_model is not None:
        slope, intercept = best_model
        keep = best_keep
        endpoint_depths = 1.0 / np.maximum(
            slope * np.array([0.0, 1.0]) + intercept, 0.15
        )
        endpoint0 = _unproject(p0[0], p0[1], endpoint_depths[0], intr)
        endpoint1 = _unproject(p1[0], p1[1], endpoint_depths[1], intr)
        length_m = float(np.linalg.norm(endpoint1 - endpoint0))
    if not (MIN_LINE_M <= length_m <= MAX_LINE_M):
        lift_debug["length_m"] = round(length_m, 3)
        lift_debug["reject"] = "physical_length"
        if plane_lift is not None:
            lift_debug["fallback"] = "direct_local_panel_plane_ransac"
            return plane_lift
        return None

    direction3 = endpoint1 - endpoint0
    direction3 /= np.linalg.norm(direction3) + 1e-12
    residual_3d = np.linalg.norm(
        (points[keep] - endpoint0)
        - np.outer((points[keep] - endpoint0) @ direction3, direction3),
        axis=1,
    )
    rms = float(np.sqrt(np.mean(residual_3d**2)))
    if rms > MAX_LINE_RMS_M:
        lift_debug["rms_mm"] = round(rms * 1000.0, 2)
        lift_debug["reject"] = "line_rms"
        if plane_lift is not None:
            lift_debug["fallback"] = "direct_local_panel_plane_ransac"
            return plane_lift
        return None
    lift_debug.update(
        {
            "inliers": int(keep.sum()),
            "inlier_coverage": round(float(np.ptp(along[keep])), 3),
            "length_m": round(length_m, 3),
            "rms_mm": round(rms * 1000.0, 2),
            "status": "ok",
        }
    )
    physical_model_disambiguated = bool(
        unconstrained_best_length is not None
        and not (MIN_LINE_M <= unconstrained_best_length <= MAX_LINE_M)
        and unconstrained_best_score > best_score
    )
    lift_debug["physical_model_disambiguated"] = (
        physical_model_disambiguated
    )
    if unconstrained_best_length is not None:
        lift_debug["unconstrained_best_length_m"] = round(
            float(unconstrained_best_length), 3
        )
    return {
        "p0": endpoint0,
        "p1": endpoint1,
        "length_m": length_m,
        "rms_m": rms,
        "n_support": int(keep.sum()),
        "valid_fraction": float(keep.sum()) / max(len(points), 1),
        "lift_source": "height_filtered_inverse_depth_ransac",
        "physical_model_disambiguated": physical_model_disambiguated,
    }


def _family_guided_direct_refit(
    line,
    blob3d,
    intr,
    top,
    prior_depth0,
    prior_depth1,
):
    """用五板族只给深度窗先验，再完全由本板真实点重拟合。

    先验不生成端点，也不算支撑；发布门要求窄至 3 px 的本地条带里至少 24 个
    真实深度点、70% 覆盖和 <=20 mm 三维直线 RMS。
    """
    p0 = np.asarray(line["p0"], dtype=np.float64)
    p1 = np.asarray(line["p1"], dtype=np.float64)
    vector = p1 - p0
    length_px = float(np.linalg.norm(vector))
    if length_px < MIN_LINE_PX:
        return None
    direction2 = vector / length_px
    normal2 = np.array([-direction2[1], direction2[0]])
    uv = np.asarray(blob3d["uv"], dtype=np.float64)
    points = np.asarray(blob3d["pts"], dtype=np.float64)
    heights = np.asarray(blob3d["h"], dtype=np.float64)
    relative = uv - p0
    t = (relative @ direction2) / length_px
    lateral = np.abs(relative @ normal2)
    selected = (
        (t >= -0.04)
        & (t <= 1.04)
        & (lateral <= 3.0)
        & (heights >= top["base"] + 0.020)
    )
    if int(selected.sum()) < 24:
        return None
    t = t[selected]
    points = points[selected]
    prior_inv = (
        (1.0 - t) / max(float(prior_depth0), 1e-6)
        + t / max(float(prior_depth1), 1e-6)
    )
    prior_z = 1.0 / np.maximum(prior_inv, 0.15)
    near_prior = np.abs(points[:, 2] - prior_z) <= 0.050
    t = t[near_prior]
    points = points[near_prior]
    if len(points) < 24 or float(np.ptp(t)) < 0.70:
        return None

    inv_z = 1.0 / np.maximum(points[:, 2], 1e-6)
    keep = np.ones(len(points), dtype=bool)
    slope = intercept = 0.0
    for _ in range(3):
        matrix = np.column_stack([t[keep], np.ones(int(keep.sum()))])
        slope, intercept = np.linalg.lstsq(
            matrix, inv_z[keep], rcond=None
        )[0]
        predicted_inv = slope * t + intercept
        predicted_z = 1.0 / np.maximum(predicted_inv, 0.15)
        residual = np.abs(points[:, 2] - predicted_z)
        threshold = float(
            np.clip(np.percentile(residual[keep], 78) * 1.8, 0.008, 0.020)
        )
        new_keep = residual <= threshold
        if int(new_keep.sum()) < 24:
            break
        keep = new_keep
    coverage = float(np.ptp(t[keep]))
    if int(keep.sum()) < 24 or coverage < 0.70:
        return None

    endpoint_depths = 1.0 / np.maximum(
        slope * np.asarray([0.0, 1.0]) + intercept, 0.15
    )
    if (
        abs(float(endpoint_depths[0]) - float(prior_depth0)) > 0.060
        or abs(float(endpoint_depths[1]) - float(prior_depth1)) > 0.060
    ):
        return None
    endpoint0 = _unproject(p0[0], p0[1], endpoint_depths[0], intr)
    endpoint1 = _unproject(p1[0], p1[1], endpoint_depths[1], intr)
    length_m = float(np.linalg.norm(endpoint1 - endpoint0))
    if not (MIN_LINE_M <= length_m <= MAX_LINE_M):
        return None
    direction3 = endpoint1 - endpoint0
    direction3 /= np.linalg.norm(direction3) + 1e-12
    residual_3d = np.linalg.norm(
        (points[keep] - endpoint0)
        - np.outer((points[keep] - endpoint0) @ direction3, direction3),
        axis=1,
    )
    rms = float(np.sqrt(np.mean(residual_3d**2)))
    if rms > 0.020:
        return None
    return {
        "p0": endpoint0,
        "p1": endpoint1,
        "length_m": length_m,
        "rms_m": rms,
        "n_support": int(keep.sum()),
        "valid_fraction": float(keep.sum()) / max(len(points), 1),
        "lift_source": "family_guided_direct_local_depth_refit",
        "prior_depths_m": [float(prior_depth0), float(prior_depth1)],
        "direct_coverage": coverage,
    }


def _divider_family_grasps(
    rgb_bgr, blob, depth_mm, intr, plane, blob3d, top, debug
):
    edges, tangent, _ = _edge_evidence(rgb_bgr, blob)
    fit_edges = edges
    cap_forbidden = None
    if top["pose"] == "cap_visible":
        # 倒扣/端盖可见时，最高的大实心面会产生规则接缝和强高光。它们在
        # Canny 上比深色隔板还强，却不是用户标红的可夹自由边。先把该实心面
        # 连同 3 px 安全边界从拟合证据中硬删除，再寻找露在两侧的真实板棱。
        cap_forbidden = cv2.dilate(
            (top["mask"] > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        fit_edges = edges.copy()
        fit_edges[cap_forbidden > 0] = 0
        debug["cap_forbidden_area_px"] = int(cap_forbidden.sum())
        debug["cap_semantic_edge_pixels"] = int((fit_edges > 0).sum())
    edge_distance = cv2.distanceTransform(
        (fit_edges == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    forced_angle = cap_angle = None
    if top["pose"] == "cap_visible":
        forced_angle, cap_angle = _cap_target_angle(top)
        debug["cap_angle_deg"] = round(cap_angle, 2)
        debug["forced_ridge_angle_deg"] = round(forced_angle, 2)
    candidates = _family_candidates(
        edges,
        blob,
        top["bbox"] if top["pose"] == "side_laid" else cv2.boundingRect(
            max(
                cv2.findContours(
                    (blob > 0).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[0],
                key=cv2.contourArea,
            )
        ),
        debug["maps"],
        top,
        forced_angle=forced_angle,
    )
    if not candidates:
        debug["family_reject"] = "no_five_ridge_family"
        return []
    best = candidates[0]
    debug["family_angle_deg"] = round(best["angle"], 2)
    debug["family_score"] = round(best["score"], 4)
    debug["family_structural"] = round(best["structural"], 4)
    debug["family_geometry"] = round(best["geometry"], 4)
    debug["family_offsets"] = [round(value, 2) for value in best["offsets"]]
    debug["family_geometry_detail"] = best["geometry_debug"]
    debug["family_top3"] = [
        {
            "angle": round(item["angle"], 1),
            "score": round(item["score"], 4),
            "geometry": round(item["geometry"], 3),
        }
        for item in candidates[:3]
    ]
    if (
        best["geometry"] < 0.65
        or not best["geometry_debug"].get("hard_geometry", False)
    ):
        debug["family_reject"] = "five_ridge_3d_geometry"
        return []

    bbox = cv2.boundingRect(
        max(
            cv2.findContours(
                (blob > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0],
            key=cv2.contourArea,
        )
    )
    fitted = []
    lines_2d = []
    fit_status = []
    for ridge_index in range(5):
        line = _fit_2d_ridge(
            best,
            ridge_index,
            fit_edges,
            edge_distance,
            tangent,
            blob,
            bbox,
            top,
            debug["maps"],
            allowed_mask=(
                None if cap_forbidden is None else (cap_forbidden == 0)
            ),
        )
        lines_2d.append(line)
        if line is None:
            fitted.append(None)
            fit_status.append("no_2d_fit")
            continue
        lifted = _lift_ridge(line, depth_mm, intr, top, blob3d)
        if lifted is None:
            fitted.append(None)
            fit_status.append(
                {
                    "status": "no_3d_lift",
                    "support_px": int(line["support_count"]),
                    "length_px": round(
                        float(np.linalg.norm(line["p1"] - line["p0"])), 1
                    ),
                    **line.get("_lift_debug", {}),
                }
            )
        else:
            if "uv0" in lifted and "uv1" in lifted:
                line = dict(line)
                line["p0"] = np.asarray(lifted["uv0"], dtype=np.float64)
                line["p1"] = np.asarray(lifted["uv1"], dtype=np.float64)
                metrics = _line_edge_metrics(
                    edge_distance, line["p0"], line["p1"]
                )
                line["edge_p90_px"] = metrics["p90_px"]
                line["edge_within_3px"] = metrics["within_3px"]
                line["edge_max_air_run_px"] = metrics["max_air_run_px"]
                line["edge_trimmed"] = True
            fitted.append((line, lifted))
            fit_status.append(
                {
                    "status": "ok",
                    "support_px": int(line["support_count"]),
                    "length_px": round(
                        float(np.linalg.norm(line["p1"] - line["p0"])), 1
                    ),
                    "length_m": round(float(lifted["length_m"]), 3),
                    "rms_mm": round(float(lifted["rms_m"]) * 1000.0, 2),
                    "lift_source": lifted.get("lift_source", "unknown"),
                }
            )
    debug["ridge_fit_status"] = fit_status

    # 对“物理长度贴近上限/仅有短深度子段”的可疑解做第二次本地复核。其它板
    # 只提供 ±50 mm 的深度搜索窗；最终端点、RMS、覆盖和 n_support 全由目标
    # 板自己的 3 px 窄条带真实深度点决定。
    anchor_indices = [
        index
        for index, item in enumerate(fitted)
        if item is not None
        and item[1].get("lift_source")
        in {
            "height_filtered_inverse_depth_ransac",
            "direct_local_panel_plane_ransac",
        }
        and not item[1].get("physical_model_disambiguated", False)
        and float(item[1]["length_m"]) < 0.445
    ]
    if len(anchor_indices) >= 2:
        anchor_x = np.asarray(anchor_indices, dtype=np.float64)
        start_model = np.polyfit(
            anchor_x,
            np.asarray(
                [fitted[index][1]["p0"][2] for index in anchor_indices],
                dtype=np.float64,
            ),
            1,
        )
        end_model = np.polyfit(
            anchor_x,
            np.asarray(
                [fitted[index][1]["p1"][2] for index in anchor_indices],
                dtype=np.float64,
            ),
            1,
        )
        for ridge_index, raw_line in enumerate(lines_2d):
            if raw_line is None:
                continue
            item = fitted[ridge_index]
            source = (
                None if item is None else item[1].get("lift_source")
            )
            should_refit = bool(
                item is None
                or source == "direct_height_filtered_supported_subsegment"
                or bool(
                    item[1].get("physical_model_disambiguated", False)
                )
                or float(item[1]["length_m"]) >= 0.445
            )
            if not should_refit:
                continue
            prior_depth0 = float(np.polyval(start_model, ridge_index))
            prior_depth1 = float(np.polyval(end_model, ridge_index))
            guided = _family_guided_direct_refit(
                raw_line,
                blob3d,
                intr,
                top,
                prior_depth0,
                prior_depth1,
            )
            if guided is None:
                continue
            previous_status = fit_status[ridge_index]
            fitted[ridge_index] = (raw_line, guided)
            fit_status[ridge_index] = {
                "status": "ok_family_guided_direct_refit",
                "previous_status": previous_status,
                "support_px": int(raw_line["support_count"]),
                "length_px": round(
                    float(
                        np.linalg.norm(raw_line["p1"] - raw_line["p0"])
                    ),
                    1,
                ),
                "length_m": round(float(guided["length_m"]), 3),
                "rms_mm": round(float(guided["rms_m"]) * 1000.0, 2),
                "n_direct_depth": int(guided["n_support"]),
                "direct_coverage": round(
                    float(guided["direct_coverage"]), 3
                ),
                "lift_source": guided["lift_source"],
            }
        debug["ridge_fit_status"] = fit_status

    # 法向 snap 只允许在原五板族附近微调，并在发布前重新检查顺序和等距性。
    # 这和 40 px 连续边缘门共同防止某一条独立吸到板内平行孔边。
    valid_2d = [
        (index, line)
        for index, line in enumerate(lines_2d)
        if line is not None
    ]
    if len(valid_2d) >= 4:
        theta = math.radians(float(best["angle"]))
        family_normal = np.array([-math.sin(theta), math.cos(theta)])
        x, y, width, height = bbox
        family_center = np.array([x + width / 2.0, y + height / 2.0])
        indices = np.asarray(
            [index for index, _ in valid_2d], dtype=np.float64
        )
        snapped_offsets = np.asarray(
            [
                (
                    0.5 * (line["p0"] + line["p1"]) - family_center
                )
                @ family_normal
                for _, line in valid_2d
            ],
            dtype=np.float64,
        )
        expected_offsets = np.asarray(
            [best["offsets"][int(index)] for index in indices],
            dtype=np.float64,
        )
        snapped_gaps = np.diff(snapped_offsets) / np.maximum(
            np.diff(indices), 1.0
        )
        snapped_gap_cv = float(
            np.std(snapped_gaps)
            / max(float(np.mean(snapped_gaps)), 1e-6)
        )
        max_offset_drift = float(
            np.max(np.abs(snapped_offsets - expected_offsets))
        )
        post_snap_ok = bool(
            np.all(snapped_gaps >= 12.0)
            and snapped_gap_cv <= 0.38
            and max_offset_drift <= 12.0
        )
        debug["post_snap_family"] = {
            "offsets_px": [
                round(float(value), 2) for value in snapped_offsets
            ],
            "normalized_gaps_px": [
                round(float(value), 2) for value in snapped_gaps
            ],
            "gap_cv": round(snapped_gap_cv, 3),
            "max_offset_drift_px": round(max_offset_drift, 2),
            "passed": post_snap_ok,
        }
        if not post_snap_ok:
            debug["family_reject"] = "post_snap_five_ridge_regularity"
            return []

    visible_count = sum(item is not None for item in fitted)
    # 五板存在性已经由上面的 3D 硬几何整组确认。端盖可能遮住其中四块板的
    # 大部分自由边；公开接口又只需要一条，所以这里只要求至少一条通过本板
    # 直接深度拟合的安全候选，绝不为了凑数跨板插值。
    if visible_count < 1:
        debug["family_reject"] = "no_direct_visible_ridge"
        return []

    ground_normal = np.asarray(plane[:3], dtype=np.float64)
    ground_normal /= np.linalg.norm(ground_normal) + 1e-12
    ground_offset = float(plane[3]) / (
        np.linalg.norm(np.asarray(plane[:3], dtype=np.float64)) + 1e-12
    )
    annotations = []
    for ridge_index, item in enumerate(fitted):
        if item is None:
            continue
        line, lifted = item
        midpoint = 0.5 * (lifted["p0"] + lifted["p1"])
        height = float(midpoint @ ground_normal + ground_offset)
        annotations.append(
            {
                "type": "line",
                "name": "basket_divider_free_edge",
                "ridge_id": int(ridge_index),
                "segment_index": 0,
                "visible_ridge_count": int(visible_count),
                "pose": top["pose"],
                "endpoints": [
                    lifted["p0"].tolist(),
                    lifted["p1"].tolist(),
                ],
                "endpoints_uv": [
                    line["p0"].tolist(),
                    line["p1"].tolist(),
                ],
                "length_m": float(lifted["length_m"]),
                "confidence": 0.0,
                "orientation": "up",
                "height_above_ground": height,
                "rms_mm": float(lifted["rms_m"] * 1000.0),
                "n_support": int(lifted["n_support"]),
                "grasp_safe": True,
                "inferred": False,
                "debug": {
                    "semantic": (
                        "RGB_supported_visible_subset_of_one_of_five_"
                        "parallel_L_panel_free_edges"
                    ),
                    "family_angle_deg": round(best["angle"], 2),
                    "line_angle_deg": round(line["angle"], 2),
                    "edge_rms_px": round(line["edge_rms_px"], 2),
                    "edge_p90_px": round(line["edge_p90_px"], 2),
                    "edge_within_3px": round(
                        line["edge_within_3px"], 3
                    ),
                    "edge_max_air_run_px": round(
                        line["edge_max_air_run_px"], 2
                    ),
                    "edge_trimmed": bool(line["edge_trimmed"]),
                    "visible_fraction": round(
                        float(line.get("visible_fraction", 1.0)), 3
                    ),
                    "normal_snap_px": round(line["normal_snap_px"], 2),
                    "edge_support": int(line["support_count"]),
                    "depth_valid_fraction": round(
                        lifted["valid_fraction"], 3
                    ),
                    "lift_source": lifted.get("lift_source", "unknown"),
                    "source_segment_index": 0,
                    "direct_depth_coverage": round(
                        float(
                            lifted.get(
                                "direct_coverage",
                                lifted.get("coverage", 1.0),
                            )
                        ),
                        3,
                    ),
                    "forbidden_cap_fraction": (
                        round(
                            _line_mask_fraction(
                                cap_forbidden,
                                line["p0"],
                                line["p1"],
                            ),
                            4,
                        )
                        if cap_forbidden is not None
                        else 0.0
                    ),
                },
            }
        )
    return annotations


def extract_basket_grasps(
    rgb_bgr, blob, depth_mm, intr, plane, blob3d, debug=None
):
    """返回只属于红线语义的抓取线列表。"""
    if debug is None:
        debug = {}
    maps = _geometry_maps(depth_mm, intr, plane)
    top = _top_surface(blob, maps)
    if top is None:
        debug["reject"] = "top_surface_unavailable"
        return []

    plane_norm = np.asarray(plane[:3], dtype=np.float64)
    norm = np.linalg.norm(plane_norm) + 1e-12
    debug["ground_normal"] = (plane_norm / norm).tolist()
    debug["ground_offset"] = float(plane[3]) / norm
    debug["maps"] = maps
    debug["pose"] = top["pose"]
    debug["top_component_ratio"] = round(top["ratio"], 3)
    debug["top_solidity"] = round(top["solidity"], 3)
    debug["top_defect_ratio"] = round(top["defect_ratio"], 3)
    debug["object_height_span_m"] = round(top["span"], 3)

    if top["pose"] == "side_laid":
        annotations = _side_panel_grasps(
            rgb_bgr, blob, top, maps, intr, debug
        )
    else:
        annotations = _divider_family_grasps(
            rgb_bgr,
            blob,
            depth_mm,
            intr,
            plane,
            blob3d,
            top,
            debug,
        )

    annotations = _finalize_top1(annotations, debug)
    # 先按完整语义边选出唯一最高置信候选，再只裁这个 winner 的一端。
    # 这样缩短线不会改变“哪块板最好”的排序，也不会让原本次优的短碎边
    # 因为天生较短而反向抢走 Top-1。
    annotations = _crop_candidates_to_terminal(
        annotations,
        rgb_bgr,
        blob,
        intr,
        plane,
        maps,
        top,
        debug,
    )

    # 大数组不能进入 JSON；保留的全是可复核的标量和短列表。
    debug.pop("maps", None)
    debug.pop("ground_normal", None)
    debug.pop("ground_offset", None)
    debug["published_grasp_line_count"] = len(annotations)
    return annotations
