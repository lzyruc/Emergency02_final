"""轮式办公椅专用识别与椅背上沿抓取线提取。

这条流水线故意不用目标检测网络。它利用赛题给定的真实规格和现场先验：

1. 从稀疏深度点拟合地面，得到每个像素的离地高度。
2. 用 ``椅座高度约 0.43 m + 深色 + 约 0.5 m 尺寸`` 找稳定的椅座种子。
3. 只在椅座周围 0.43 m 内搜索更高的深色结构，排除桌面、机器人本体以及
   纸箱/文件筐/垃圾桶。
4. 在高位结构的真实外轮廓上做概率霍夫直线拟合，只保留近似平行地面的
   物理椅背上沿；侧沿不允许作为降级候选。

输入深度必须已经对齐到 RGB，单位为毫米；内参使用 color 相机内参。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import pca_axes
from .ground import fit_ground
from .result_types import FrameResult, ObjectResult


ALGORITHM_VERSION = "chair_depth_hsv_v4"

# 地面拟合不需要使用 92 万个像素。8 倍抽样后仍有约 1.4 万个地面点，
# 但一帧耗时会从十几秒降到亚秒级。
DEFAULT_GROUND_STRIDE = 8
MAX_DEPTH_MM = 6000

# 椅座先验。实测训练集椅座平面中位高度为 0.42~0.44 m。
SEAT_H_RANGE = (0.38, 0.49)
SEAT_DARK_V_MAX = 155
SEAT_LONG_RANGE = (0.28, 0.68)
SEAT_SHORT_RANGE = (0.08, 0.62)
SEAT_MIN_AREA_PX = 1800
SEAT_BASE_EVIDENCE = 0.58
SEAT_WITH_BACK_EVIDENCE = 0.63
SEAT_WITHOUT_BACK_EVIDENCE = 0.70
SEAT_EDGE_BLACK_RATIO_MIN = 0.80

# 纸箱暗面也可能落在 0.43 m 高度带内，单看 V 通道会伪装成“深色椅座”。
# OpenCV HSV 中纸箱稳定落在棕/黄区；黑色网布椅在训练现场则完全不落入该区。
SEAT_BROWN_RATIO_MAX = 0.28

# 画面左下是机器人旁的近场盲区；窄而贴边的桌下暗面曾在历史集里制造过唯一一次
# “有高位邻居”的假椅子。这个门只否决左下角的窄碎片，不影响远处或右侧裁切椅。
BOTTOM_SIDE_FRAGMENT_SHORT_MAX = 0.18

# 椅背高位结构。标准椅总高 0.85 m；由于裁切和可见面变化，最低放到 0.58 m。
BACK_MIN_H = 0.58
BACK_MAX_H = 1.02
BACK_DARK_V_MAX = 170
BACK_SEARCH_RADIUS_M = 0.43
BACK_MIN_PEAK_H = 0.70
BACK_MIN_AREA_PX = 120

# 椅背抓取线物理长度。0.10 m 以下通常是网眼/噪声，0.65 m 以上不是该椅。
LINE_LEN_RANGE_M = (0.10, 0.65)
LINE_TARGET_M = 0.28

# 抓取边统一为“物理椅背上沿”，不能用左右侧沿补位。候选掩膜先保留顶部
# 约 10 cm 以跨过网布深度空洞，最终硬门只接受距稳健峰值 2.5 cm 内的线；
# 上沿方向应近似平行地面，允许约 10° 的安装/深度误差。
BACK_TOP_SEARCH_BAND_M = 0.105
TOP_MAX_PEAK_OFFSET_M = 0.025
TOP_MAX_VERTICAL_RATIO = 0.18
TOP_MAX_HEIGHT_SPAN_M = 0.025
TOP_BORDER_MARGIN_PX = 10


@dataclass
class _Maps:
    """一帧的逐像素几何量。"""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    height: np.ndarray
    ground_u: np.ndarray
    ground_v: np.ndarray
    valid: np.ndarray
    plane: tuple
    ground_sample_count: int
    ground_inlier_count: int
    fx: float
    fy: float


def _score_peak(x: float, center: float, width: float) -> float:
    """平滑的 0~1 单峰分数。"""
    return float(math.exp(-(((float(x) - center) / max(width, 1e-6)) ** 2)))


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _ground_frame(plane):
    n = np.asarray(plane[:3], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = tmp - float(tmp @ n) * n
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    origin = -float(plane[3]) * n
    return origin, u, v, n


def _build_maps(depth_mm: np.ndarray, intr: dict, ground_stride: int) -> _Maps:
    """用稀疏深度拟合地面，再生成全分辨率高度/地面坐标图。"""
    h, w = depth_mm.shape
    stride = max(3, int(ground_stride))

    vv_s, uu_s = np.mgrid[0:h:stride, 0:w:stride]
    z_s = depth_mm[::stride, ::stride].astype(np.float64) / 1000.0
    ok_s = (z_s > 0.0) & (z_s < MAX_DEPTH_MM / 1000.0)
    if int(ok_s.sum()) < 1000:
        raise ValueError("有效深度点太少，无法稳定拟合地面")

    pts_s = np.column_stack(
        [
            (uu_s[ok_s] - float(intr["cx"])) * z_s[ok_s] / float(intr["fx"]),
            (vv_s[ok_s] - float(intr["cy"])) * z_s[ok_s] / float(intr["fy"]),
            z_s[ok_s],
        ]
    )
    plane, inliers = fit_ground(pts_s)
    plane = tuple(float(v) for v in plane)

    vv, uu = np.mgrid[0:h, 0:w]
    z = depth_mm.astype(np.float64) / 1000.0
    valid = (depth_mm > 0) & (depth_mm < MAX_DEPTH_MM)
    x = (uu.astype(np.float64) - float(intr["cx"])) * z / float(intr["fx"])
    y = (vv.astype(np.float64) - float(intr["cy"])) * z / float(intr["fy"])
    height = plane[0] * x + plane[1] * y + plane[2] * z + plane[3]
    height[~valid] = np.nan

    origin, gu, gv, _ = _ground_frame(plane)
    ground_u = (
        (x - origin[0]) * gu[0] + (y - origin[1]) * gu[1] + (z - origin[2]) * gu[2]
    )
    ground_v = (
        (x - origin[0]) * gv[0] + (y - origin[1]) * gv[1] + (z - origin[2]) * gv[2]
    )
    ground_u[~valid] = np.nan
    ground_v[~valid] = np.nan

    return _Maps(
        x=x,
        y=y,
        z=z,
        height=height,
        ground_u=ground_u,
        ground_v=ground_v,
        valid=valid,
        plane=plane,
        ground_sample_count=int(len(pts_s)),
        ground_inlier_count=int(len(inliers)),
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
    )


def _robust_rect(uv: np.ndarray, keep_percentile: float = 96.0):
    """先剃掉最外圈飞点，再计算地面 OBB。"""
    if len(uv) < 8:
        return None
    center_med = np.median(uv, axis=0)
    radius = np.linalg.norm(uv - center_med, axis=1)
    cutoff = float(np.percentile(radius, keep_percentile))
    trimmed = uv[radius <= cutoff]
    if len(trimmed) < 8:
        trimmed = uv
    rect = cv2.minAreaRect(trimmed.astype(np.float32))
    long_side, short_side = sorted((float(rect[1][0]), float(rect[1][1])), reverse=True)
    return {
        "rect": rect,
        "long": long_side,
        "short": short_side,
        "center": np.asarray(rect[0], dtype=np.float64),
        "median_center": center_med,
    }


def _bbox_from_mask(mask: np.ndarray, margin: int = 0):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w - 1, int(xs.max()) + margin)
    y1 = min(h - 1, int(ys.max()) + margin)
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _find_seat(
    maps: _Maps,
    value_channel: np.ndarray | None,
    hue_channel: np.ndarray | None = None,
    saturation_channel: np.ndarray | None = None,
):
    """找椅座候选并返回得分最高者以及完整候选清单。"""
    lo, hi = SEAT_H_RANGE
    seat_bool = maps.valid & (maps.height >= lo) & (maps.height <= hi)
    if value_channel is not None:
        seat_bool &= value_channel <= SEAT_DARK_V_MAX

    mask = seat_bool.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    h_img, _ = mask.shape
    candidates = []
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < SEAT_MIN_AREA_PX:
            continue
        # 机器人头部/机身稳定地贴在画面底边；真实椅子允许触顶，但不能贴底。
        if y + h >= h_img - 2 or (y + h / 2.0) > 0.88 * h_img:
            continue

        pix = (labels == label) & maps.valid
        if int(pix.sum()) < 500:
            continue
        uv = np.column_stack([maps.ground_u[pix], maps.ground_v[pix]])
        rect = _robust_rect(uv)
        if rect is None:
            continue
        if not (SEAT_LONG_RANGE[0] <= rect["long"] <= SEAT_LONG_RANGE[1]):
            continue
        if not (SEAT_SHORT_RANGE[0] <= rect["short"] <= SEAT_SHORT_RANGE[1]):
            continue

        h_med = float(np.median(maps.height[pix]))
        s_height = _score_peak(h_med, 0.435, 0.060)
        s_long = _score_peak(rect["long"], 0.50, 0.18)
        s_short = _ramp(rect["short"], 0.08, 0.32)
        s_area = _ramp(area, 3000, 26000)
        fill = float(area / max(w * h, 1))
        s_fill = _ramp(fill, 0.20, 0.62)

        if value_channel is None:
            dark_ratio = 0.75
            black_ratio = 0.75
        else:
            dark_ratio = float(np.mean(value_channel[pix] <= SEAT_DARK_V_MAX))
            black_ratio = float(np.mean(value_channel[pix] < 110))

        if hue_channel is None or saturation_channel is None or value_channel is None:
            brown_ratio = 0.0
            non_brown_ratio = 1.0
        else:
            # H: 4~30 对应棕/橙/黄；同时限制 S/V，避免把极暗像素的无意义色相
            # 或灰色桌腿算作纸箱。比例在候选自身掩膜内统计，不受背景影响。
            brown = (
                (hue_channel[pix] >= 4)
                & (hue_channel[pix] <= 30)
                & (saturation_channel[pix] >= 55)
                & (value_channel[pix] >= 45)
            )
            brown_ratio = float(np.mean(brown))
            non_brown_ratio = 1.0 - brown_ratio
            if brown_ratio > SEAT_BROWN_RATIO_MAX:
                continue

        evidence = (
            0.31 * s_height
            + 0.25 * s_long
            + 0.11 * s_short
            + 0.13 * s_area
            + 0.08 * s_fill
            + 0.02 * dark_ratio
            + 0.10 * non_brown_ratio
        )
        # 面积只作轻微排序项，避免一块很小但尺寸碰巧合适的深色碎片胜出。
        rank = evidence * (0.75 + 0.25 * _ramp(area, 2000, 30000))
        candidates.append(
            {
                "label": label,
                "mask": labels == label,
                "bbox": (x, y, w, h),
                "area_px": area,
                "height_median": h_med,
                "rect": rect,
                "dark_ratio": dark_ratio,
                "black_ratio": black_ratio,
                "brown_ratio": brown_ratio,
                "non_brown_ratio": non_brown_ratio,
                "fill_ratio": fill,
                "evidence": float(evidence),
                "rank": float(rank),
                "scores": {
                    "height": s_height,
                    "long": s_long,
                    "short": s_short,
                    "area": s_area,
                    "fill": s_fill,
                    "dark": dark_ratio,
                    "black": black_ratio,
                    "non_brown": non_brown_ratio,
                },
            }
        )

    candidates.sort(key=lambda c: (-c["rank"], -c["area_px"]))
    return (candidates[0] if candidates else None), candidates


def _find_backrest(
    maps: _Maps,
    value_channel: np.ndarray | None,
    seat_center_uv: np.ndarray,
    seat_height: float,
):
    """在椅座的物理邻域里找高位深色椅背。"""
    radial = np.hypot(
        maps.ground_u - float(seat_center_uv[0]),
        maps.ground_v - float(seat_center_uv[1]),
    )
    top_bool = (
        maps.valid
        & (maps.height >= BACK_MIN_H)
        & (maps.height <= BACK_MAX_H)
        & (radial <= BACK_SEARCH_RADIUS_M)
    )
    if value_channel is not None:
        top_bool &= value_channel <= BACK_DARK_V_MAX

    top_mask = top_bool.astype(np.uint8) * 255
    top_mask = cv2.morphologyEx(
        top_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(top_mask, 8)

    candidates = []
    min_peak = max(BACK_MIN_PEAK_H, seat_height + 0.20)
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < BACK_MIN_AREA_PX:
            continue
        pix = (labels == label) & maps.valid
        if int(pix.sum()) < 80:
            continue
        uv = np.column_stack([maps.ground_u[pix], maps.ground_v[pix]])
        rect = _robust_rect(uv, keep_percentile=97.0)
        if rect is None or rect["long"] > 0.72:
            continue
        center_dist = float(np.linalg.norm(rect["median_center"] - seat_center_uv))
        if center_dist > 0.42:
            continue
        peak_h = float(np.percentile(maps.height[pix], 95))
        if peak_h < min_peak:
            continue

        s_height = _ramp(peak_h, min_peak, 0.86)
        s_area = _ramp(math.log1p(area), math.log1p(150), math.log1p(25000))
        s_dist = 1.0 - _ramp(center_dist, 0.18, 0.45)
        s_long = _score_peak(rect["long"], 0.48, 0.24)
        rank = 0.36 * s_height + 0.28 * s_area + 0.22 * s_dist + 0.14 * s_long
        candidates.append(
            {
                "label": label,
                "mask": labels == label,
                "bbox": (x, y, w, h),
                "area_px": area,
                "peak_height": peak_h,
                "center_dist": center_dist,
                "rect": rect,
                "rank": float(rank),
            }
        )

    candidates.sort(key=lambda c: (-c["rank"], -c["peak_height"], -c["area_px"]))
    best = candidates[0] if candidates else None
    if best is not None:
        # 候选搜索先保留最高约 10 cm，最终线仍由更严格的 2.5 cm 高度门限定。
        # 当椅子贴近办公桌时，二者的 2D 连通域偶尔会相连，但 0.75 m 的桌面
        # 明显低于约 0.85 m 的椅背顶沿，会在这里被干净剔除。
        ridge_lo = max(
            BACK_MIN_H,
            float(best["peak_height"]) - BACK_TOP_SEARCH_BAND_M,
        )
        ridge = best["mask"] & maps.valid & (maps.height >= ridge_lo)
        ridge_u8 = ridge.astype(np.uint8) * 255
        ridge_u8 = cv2.morphologyEx(
            ridge_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        best["grasp_mask"] = ridge_u8 > 0
        best["grasp_min_height"] = ridge_lo
    return best, candidates, radial


def _sample_line_3d(
    coords,
    maps: _Maps,
    support_mask: np.ndarray,
    n_samples: int = 36,
    radius: int = 4,
):
    """沿 2D 候选线取高位深度点，并稳健拟合 3D 直线。"""
    x1, y1, x2, y2 = (float(v) for v in coords)
    h_img, w_img = maps.valid.shape
    ts = []
    pts = []
    sampled_h = []
    sampled_z = []

    for t in np.linspace(0.0, 1.0, n_samples):
        px = int(round(x1 + t * (x2 - x1)))
        py = int(round(y1 + t * (y2 - y1)))
        xa, xb = max(0, px - radius), min(w_img, px + radius + 1)
        ya, yb = max(0, py - radius), min(h_img, py + radius + 1)
        local_ok = maps.valid[ya:yb, xa:xb] & support_mask[ya:yb, xa:xb]
        if not np.any(local_ok):
            continue
        local_h = maps.height[ya:yb, xa:xb].copy()
        vals = local_h[local_ok]
        # 轮廓窗口同时可能看到后面的座面；取较高的四分之一，再取其中位点。
        target_h = float(np.percentile(vals, 78))
        yy, xx = np.nonzero(local_ok)
        best_i = int(np.argmin(np.abs(vals - target_h)))
        gy, gx = ya + int(yy[best_i]), xa + int(xx[best_i])
        pts.append([maps.x[gy, gx], maps.y[gy, gx], maps.z[gy, gx]])
        ts.append(float(t))
        sampled_h.append(float(maps.height[gy, gx]))
        sampled_z.append(float(maps.z[gy, gx]))

    valid_fraction = len(pts) / float(n_samples)
    if len(pts) < 8:
        return None

    t_arr = np.asarray(ts, dtype=np.float64)
    p_arr = np.asarray(pts, dtype=np.float64)
    A = np.column_stack([np.ones_like(t_arr), t_arr])
    coef, _, _, _ = np.linalg.lstsq(A, p_arr, rcond=None)
    fit = A @ coef
    residual = np.linalg.norm(p_arr - fit, axis=1)
    keep = residual <= max(0.018, float(np.percentile(residual, 85)))
    if int(keep.sum()) >= 7:
        coef, _, _, _ = np.linalg.lstsq(A[keep], p_arr[keep], rcond=None)
        fit = A[keep] @ coef
        residual_kept = np.linalg.norm(p_arr[keep] - fit, axis=1)
    else:
        residual_kept = residual

    p0 = coef[0]
    p1 = coef[0] + coef[1]
    length_m = float(np.linalg.norm(p1 - p0))
    rms_m = float(np.sqrt(np.mean(np.square(residual_kept))))
    h10, h90 = np.percentile(sampled_h, [10, 90])
    return {
        "endpoints": [p0.tolist(), p1.tolist()],
        "length_m": length_m,
        "rms_m": rms_m,
        "valid_fraction": valid_fraction,
        "height_median": float(np.median(sampled_h)),
        "height_span": float(h90 - h10),
        "depth_median": float(np.median(sampled_z)),
        "n_support": int(len(pts)),
    }


def _line_rgb_edge_score(
    coords,
    gradients: tuple[np.ndarray, np.ndarray] | None,
    n_samples: int = 48,
    radius: int = 3,
):
    """量化候选线沿线的 RGB 法向梯度，实体外框应明显强于网布纹理。"""
    if gradients is None:
        return None, None
    grad_x, grad_y = gradients
    x1, y1, x2, y2 = (float(v) for v in coords)
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0, 0.0
    normal_x, normal_y = -dy / norm, dx / norm
    h_img, w_img = grad_x.shape
    strengths = []
    for t in np.linspace(0.0, 1.0, n_samples):
        px = int(round(x1 + t * dx))
        py = int(round(y1 + t * dy))
        xa, xb = max(0, px - radius), min(w_img, px + radius + 1)
        ya, yb = max(0, py - radius), min(h_img, py + radius + 1)
        normal_gradient = np.abs(
            grad_x[ya:yb, xa:xb] * normal_x + grad_y[ya:yb, xa:xb] * normal_y
        )
        if normal_gradient.size:
            strengths.append(float(normal_gradient.max()))
    if not strengths:
        return 0.0, 0.0
    median_strength = float(np.median(strengths))
    # 约 300 Sobel 灰度单位已经是非常清楚的物体边界。
    score = float(np.clip(median_strength / 300.0, 0.0, 1.0))
    return score, median_strength


def _line_candidates(
    backrest: dict,
    maps: _Maps,
    seat_height: float,
    rgb_gradients: tuple[np.ndarray, np.ndarray] | None = None,
):
    """从椅背外轮廓产生并排序直线候选。"""
    full_mask_u8 = backrest["mask"].astype(np.uint8) * 255
    grasp_mask_u8 = backrest.get("grasp_mask", backrest["mask"]).astype(np.uint8) * 255

    # 必须先取“完整椅背”的外轮廓，再用最高带筛选。反过来直接对最高带
    # findContours 会把高度阈值的下边界也当作轮廓，线可能横跨网布内部。
    contours, _ = cv2.findContours(
        full_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    ridge_contours, _ = cv2.findContours(
        grasp_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    edge = np.zeros_like(full_mask_u8)
    cv2.drawContours(edge, contours, -1, 255, 2)
    # 高位带轮廓能补回被深度空洞切开的真实框边；后续 RGB 法向梯度会压低
    # 仅由高度阈值形成、但图像中并无实体边缘的伪线。
    cv2.drawContours(edge, ridge_contours, -1, 255, 2)
    high_neighborhood = cv2.dilate(
        grasp_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    edge[high_neighborhood == 0] = 0

    hough = cv2.HoughLinesP(
        edge,
        rho=1,
        theta=np.pi / 720.0,
        threshold=18,
        minLineLength=35,
        maxLineGap=20,
    )
    if hough is None:
        return []

    support = cv2.dilate(grasp_mask_u8, np.ones((3, 3), np.uint8)) > 0
    h_img, w_img = full_mask_u8.shape
    peak_h = float(backrest["peak_height"])
    out = []
    for raw in hough.reshape(-1, 4):
        coords = tuple(int(v) for v in raw)
        x1, y1, x2, y2 = coords
        pixel_length = float(math.hypot(x2 - x1, y2 - y1))
        if pixel_length < 35:
            continue

        fit = _sample_line_3d(coords, maps, support)
        if fit is None:
            continue
        length_m = float(fit["length_m"])
        if not (LINE_LEN_RANGE_M[0] <= length_m <= LINE_LEN_RANGE_M[1]):
            continue
        # 轮廓线上最容易出现的坏解是：几十像素的一小段跨到了两个深度层，
        # 3D 拟合却被拉成 0.5~0.6 m。用针孔投影的一阶尺度做一致性门控。
        dx, dy = float(x2 - x1), float(y2 - y1)
        z_med = float(fit["depth_median"])
        apparent_length_m = math.hypot(
            dx * z_med / maps.fx,
            dy * z_med / maps.fy,
        )
        if length_m > max(0.20, 3.0 * apparent_length_m):
            continue
        if fit["rms_m"] > 0.045:
            continue
        h_line = float(fit["height_median"])
        if h_line < max(seat_height + 0.12, peak_h - TOP_MAX_PEAK_OFFSET_M):
            continue

        p0 = np.asarray(fit["endpoints"][0], dtype=np.float64)
        p1 = np.asarray(fit["endpoints"][1], dtype=np.float64)
        plane_n = np.asarray(maps.plane[:3], dtype=np.float64)
        plane_n /= np.linalg.norm(plane_n) + 1e-12
        endpoint_height_delta = float(abs((p1 - p0) @ plane_n))
        vertical_ratio = endpoint_height_delta / max(length_m, 1e-9)
        # 这是区分“上沿”和“侧沿”的硬合同：上沿在物理世界中近似水平。
        # 侧沿即使很长、RGB 梯度很清楚，也绝不能参与抓取候选。
        if vertical_ratio > TOP_MAX_VERTICAL_RATIO:
            continue
        if fit["height_span"] > TOP_MAX_HEIGHT_SPAN_M:
            continue

        delta_h = max(0.0, peak_h - h_line)
        s_height = float(math.exp(-((delta_h / TOP_MAX_PEAK_OFFSET_M) ** 2)))
        if length_m <= LINE_TARGET_M:
            s_length = _ramp(length_m, LINE_LEN_RANGE_M[0], LINE_TARGET_M)
        else:
            s_length = 1.0 - 0.35 * _ramp(length_m, 0.45, LINE_LEN_RANGE_M[1])
        s_valid = float(fit["valid_fraction"])
        s_straight = float(math.exp(-((fit["rms_m"] / 0.030) ** 2)))
        s_level = float(math.exp(-((vertical_ratio / 0.12) ** 2)))
        s_rgb_edge, rgb_edge_strength = _line_rgb_edge_score(coords, rgb_gradients)

        # 只惩罚“整条线就是图像边框”的情况；允许真实边缘的一端恰好被裁切。
        along_border = (
            (y1 <= 2 and y2 <= 2)
            or (x1 <= 2 and x2 <= 2)
            or (x1 >= w_img - 3 and x2 >= w_img - 3)
        )
        border_factor = 0.25 if along_border else 1.0

        base_confidence = (
            0.32 * s_height
            + 0.20 * s_length
            + 0.16 * s_valid
            + 0.17 * s_straight
            + 0.15 * s_level
        )
        if s_rgb_edge is None:
            confidence = base_confidence * border_factor
            rgb_rank_factor = 1.0
        else:
            confidence = (0.90 * base_confidence + 0.10 * s_rgb_edge) * border_factor
            rgb_rank_factor = 0.25 + 0.75 * s_rgb_edge
        # 排名同时看“有多长”和“是不是最高那条”，防止网眼里的短线获胜。
        rank = (
            length_m
            * (0.05 + 0.95 * (s_height**2))
            * (0.45 + 0.55 * s_valid)
            * (0.20 + 0.80 * s_straight)
            * (0.35 + 0.65 * s_level)
            * rgb_rank_factor
            * border_factor
        )
        out.append(
            {
                "coords": coords,
                "pixel_length": pixel_length,
                "rank": float(rank),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "scores": {
                    "height": s_height,
                    "length": s_length,
                    "valid": s_valid,
                    "straight": s_straight,
                    "level": s_level,
                    "rgb_edge": s_rgb_edge,
                },
                "rgb_edge_strength": rgb_edge_strength,
                "endpoint_height_delta": endpoint_height_delta,
                "vertical_ratio": vertical_ratio,
                "apparent_length_m": float(apparent_length_m),
                **fit,
            }
        )

    # 去重：霍夫会在同一根粗边的上下两侧给很多几乎相同的线。
    out.sort(key=lambda c: (-c["rank"], -c["confidence"]))
    deduped = []
    for cand in out:
        mx = (cand["coords"][0] + cand["coords"][2]) / 2.0
        my = (cand["coords"][1] + cand["coords"][3]) / 2.0
        ang = math.atan2(
            cand["coords"][3] - cand["coords"][1],
            cand["coords"][2] - cand["coords"][0],
        )
        duplicate = False
        for kept in deduped:
            kx = (kept["coords"][0] + kept["coords"][2]) / 2.0
            ky = (kept["coords"][1] + kept["coords"][3]) / 2.0
            kang = math.atan2(
                kept["coords"][3] - kept["coords"][1],
                kept["coords"][2] - kept["coords"][0],
            )
            da = abs(math.atan2(math.sin(ang - kang), math.cos(ang - kang)))
            da = min(da, abs(math.pi - da))
            if math.hypot(mx - kx, my - ky) < 18 and da < math.radians(8):
                duplicate = True
                break
        if not duplicate:
            deduped.append(cand)
    return deduped


def _clip_top_edge_to_image_margin(
    candidate: dict,
    image_shape: tuple[int, int],
    margin: int = TOP_BORDER_MARGIN_PX,
):
    """把一端触碰画框的真实上沿裁到安全内边距，并同步插值 3D 端点。"""
    h_img, w_img = image_shape
    x1, y1, x2, y2 = (float(v) for v in candidate["coords"])
    dx, dy = x2 - x1, y2 - y1
    xmin = ymin = float(margin)
    xmax = float(w_img - 1 - margin)
    ymax = float(h_img - 1 - margin)
    if xmax <= xmin or ymax <= ymin:
        return None

    # Liang-Barsky 线段裁剪，保留原始参数 t，便于对 3D 拟合端点做同样插值。
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1 - xmin),
        (dx, xmax - x1),
        (-dy, y1 - ymin),
        (dy, ymax - y1),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None

    uv0 = np.array([x1 + t0 * dx, y1 + t0 * dy], dtype=np.float64)
    uv1 = np.array([x1 + t1 * dx, y1 + t1 * dy], dtype=np.float64)
    p0 = np.asarray(candidate["endpoints"][0], dtype=np.float64)
    p1 = np.asarray(candidate["endpoints"][1], dtype=np.float64)
    p3d0 = p0 + t0 * (p1 - p0)
    p3d1 = p0 + t1 * (p1 - p0)
    length_m = float(np.linalg.norm(p3d1 - p3d0))
    if length_m < LINE_LEN_RANGE_M[0]:
        return None

    out = dict(candidate)
    out["raw_coords"] = tuple(candidate["coords"])
    out["coords"] = (float(uv0[0]), float(uv0[1]), float(uv1[0]), float(uv1[1]))
    out["endpoints"] = [p3d0.tolist(), p3d1.tolist()]
    out["length_m"] = length_m
    out["pixel_length"] = float(np.linalg.norm(uv1 - uv0))
    out["visible_fraction"] = float(max(0.0, t1 - t0))
    out["border_trimmed"] = bool(t0 > 1e-6 or t1 < 1.0 - 1e-6)
    out["apparent_length_m"] = float(
        candidate["apparent_length_m"] * out["visible_fraction"]
    )
    out["endpoint_height_delta"] = float(
        candidate["endpoint_height_delta"] * max(0.0, t1 - t0)
    )
    return out


def _build_chair_mask(
    maps: _Maps,
    value_channel: np.ndarray | None,
    seat: dict,
    backrest: dict | None,
    radial: np.ndarray,
):
    """生成用于 bbox/尺寸统计的椅子可见部分掩膜。"""
    # 普通椅体只收 0.62 m 以下的深色结构；更高部分必须属于已经通过
    # 0.85 m 高度证据验证的椅背顶沿。这样相邻的 0.75 m 办公桌不会撑大 bbox。
    mask = (
        maps.valid
        & (maps.height >= 0.08)
        & (maps.height <= 0.62)
        & (radial <= BACK_SEARCH_RADIUS_M)
    )
    if value_channel is not None:
        mask &= value_channel <= 180
    mask |= seat["mask"]
    if backrest is not None:
        mask |= backrest.get("grasp_mask", backrest["mask"])
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    # 删除小散点，但保留椅背/椅腿等彼此不相连的合法部件。
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, n_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 80:
            cleaned[labels == label] = 255
    cleaned[seat["mask"]] = 255
    if backrest is not None:
        cleaned[backrest.get("grasp_mask", backrest["mask"])] = 255
    return cleaned > 0


def _visible_geometry(chair_mask: np.ndarray, maps: _Maps):
    pix = chair_mask & maps.valid
    pts = np.column_stack([maps.x[pix], maps.y[pix], maps.z[pix]])
    uv = np.column_stack([maps.ground_u[pix], maps.ground_v[pix]])
    rect = _robust_rect(uv, keep_percentile=97.0)
    if rect is None:
        raise ValueError("椅子点云退化，无法计算尺寸")

    # PCA 最多用 3 万点，既稳定也避免对可视化字段做不必要的大矩阵运算。
    if len(pts) > 30000:
        take = np.linspace(0, len(pts) - 1, 30000, dtype=np.int64)
        pca_pts = pts[take]
    else:
        pca_pts = pts
    axes, extents, _ = pca_axes(pca_pts)
    center = np.median(pts, axis=0)
    height = float(np.percentile(maps.height[pix], 99))

    voxel = np.floor(pts / 0.02).astype(np.int32)
    n_vox = len(np.unique(voxel, axis=0))
    visible_box_volume = max(rect["long"] * rect["short"] * max(height, 0.08), 1e-6)
    occupancy = float(np.clip(n_vox * (0.02**3) / visible_box_volume, 0.0, 1.0))
    return {
        "points": pts,
        "center": center,
        "axes": axes,
        "extents": extents,
        "height": height,
        "fp_long": float(rect["long"]),
        "fp_short": float(rect["short"]),
        "fp_area": float(rect["long"] * rect["short"]),
        "occupancy": occupancy,
    }


def detect_chair(
    depth_mm: np.ndarray,
    intr: dict,
    rgb_bgr: np.ndarray | None = None,
    *,
    ground_stride: int = DEFAULT_GROUND_STRIDE,
) -> FrameResult:
    """识别一把轮式办公椅，并返回标准 :class:`FrameResult`。

    没有可靠椅子证据时 ``objects`` 为空。识别到椅子但椅背完全出画时仍返回
    chair 对象，不过 ``grasp_annotations`` 为空；这比在座垫或桌沿上伪造抓取线安全。
    """
    depth_mm = np.asarray(depth_mm)
    if depth_mm.ndim != 2:
        raise ValueError(f"depth_mm 应为二维深度图，实际 shape={depth_mm.shape}")
    if rgb_bgr is not None and rgb_bgr.shape[:2] != depth_mm.shape:
        raise ValueError(
            f"RGB/深度尺寸不一致: rgb={rgb_bgr.shape[:2]}, depth={depth_mm.shape}"
        )
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intr:
            raise ValueError(f"相机内参缺少字段 {key!r}")

    maps = _build_maps(depth_mm, intr, ground_stride)
    hue = saturation = value = None
    rgb_gradients = None
    if rgb_bgr is not None:
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        rgb_gradients = (
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )

    seat, seat_candidates = _find_seat(maps, value, hue, saturation)
    if seat is None or seat["evidence"] < SEAT_BASE_EVIDENCE:
        return FrameResult(
            objects=[],
            ground_plane=maps.plane,
            n_raw_clusters=len(seat_candidates),
            n_kept=0,
            debug={
                "algorithm": ALGORITHM_VERSION,
                "reason": "no_chair_seat_candidate",
                "ground_sample_count": maps.ground_sample_count,
                "ground_inlier_count": maps.ground_inlier_count,
                "seat_candidates": len(seat_candidates),
            },
        )

    seat_center = np.asarray(seat["rect"]["median_center"], dtype=np.float64)
    backrest, back_candidates, radial = _find_backrest(
        maps, value, seat_center, seat["height_median"]
    )

    x_seat, y_seat, w_seat, h_seat = seat["bbox"]
    h_img, w_img = depth_mm.shape
    touches_side = x_seat <= 2 or (x_seat + w_seat) >= w_img - 2
    bottom_side_fragment = (
        x_seat <= 2
        and (y_seat + h_seat / 2.0) > 0.64 * h_img
        and seat["rect"]["short"] < BOTTOM_SIDE_FRAGMENT_SHORT_MAX
    )
    if backrest is not None:
        seat_accepted = seat["evidence"] >= SEAT_WITH_BACK_EVIDENCE
        reject_reason = "weak_seat_evidence_despite_high_neighbor"
    else:
        # 完整座面会得到很高的几何证据；严重裁切时，还允许“贴左右边 +
        # 几乎全黑”的椅座碎片。这覆盖历史集中的半出画椅子，同时不会把
        # 左上角浅灰桌板或纸箱当成椅子。
        seat_accepted = seat["evidence"] >= SEAT_WITHOUT_BACK_EVIDENCE or (
            touches_side
            and seat["black_ratio"] >= SEAT_EDGE_BLACK_RATIO_MIN
            and seat["evidence"] >= SEAT_BASE_EVIDENCE
        )
        reject_reason = "seat_candidate_without_backrest_is_incomplete"

    if bottom_side_fragment or not seat_accepted:
        return FrameResult(
            objects=[],
            ground_plane=maps.plane,
            n_raw_clusters=len(seat_candidates),
            n_kept=0,
            debug={
                "algorithm": ALGORITHM_VERSION,
                "reason": (
                    "bottom_side_fragment" if bottom_side_fragment else reject_reason
                ),
                "ground_sample_count": maps.ground_sample_count,
                "ground_inlier_count": maps.ground_inlier_count,
                "seat_candidates": len(seat_candidates),
                "backrest_candidates": len(back_candidates),
                "seat_short_m": round(float(seat["rect"]["short"]), 4),
                "seat_area_px": int(seat["area_px"]),
                "seat_evidence": round(float(seat["evidence"]), 4),
                "seat_black_ratio": round(float(seat["black_ratio"]), 4),
            },
        )

    chair_mask = _build_chair_mask(maps, value, seat, backrest, radial)
    geometry = _visible_geometry(chair_mask, maps)
    pixel_bbox = _bbox_from_mask(chair_mask, margin=6)

    line_candidates = (
        _line_candidates(
            backrest,
            maps,
            seat["height_median"],
            rgb_gradients=rgb_gradients,
        )
        if backrest
        else []
    )
    grasp_annotations = []
    if line_candidates:
        best_line = next(
            (
                prepared
                for candidate in line_candidates
                if (
                    prepared := _clip_top_edge_to_image_margin(
                        candidate,
                        depth_mm.shape,
                    )
                )
                is not None
            ),
            None,
        )
    else:
        best_line = None
    if best_line is not None:
        p0 = np.asarray(best_line["endpoints"][0], dtype=np.float64)
        p1 = np.asarray(best_line["endpoints"][1], dtype=np.float64)
        mid = (p0 + p1) / 2.0
        h_mid = float(
            maps.plane[0] * mid[0]
            + maps.plane[1] * mid[1]
            + maps.plane[2] * mid[2]
            + maps.plane[3]
        )
        x1, y1, x2, y2 = best_line["coords"]
        grasp_annotations.append(
            {
                "type": "line",
                "name": "backrest_top_edge",
                "endpoints": best_line["endpoints"],
                "endpoints_uv": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "length_m": float(best_line["length_m"]),
                "confidence": float(best_line["confidence"]),
                "orientation": "up",
                "height_above_ground": h_mid,
                "rms_mm": float(best_line["rms_m"] * 1000.0),
                "n_support": int(best_line["n_support"]),
                "debug": {
                    "pixel_length": round(float(best_line["pixel_length"]), 2),
                    "apparent_length_m": round(
                        float(best_line["apparent_length_m"]), 4
                    ),
                    "valid_fraction": round(float(best_line["valid_fraction"]), 3),
                    "height_span_m": round(float(best_line["height_span"]), 4),
                    "endpoint_height_delta_m": round(
                        float(best_line["endpoint_height_delta"]), 4
                    ),
                    "vertical_ratio": round(float(best_line["vertical_ratio"]), 4),
                    "border_trimmed": bool(best_line["border_trimmed"]),
                    "visible_fraction": round(float(best_line["visible_fraction"]), 3),
                    "raw_endpoints_uv": [
                        [
                            float(best_line["raw_coords"][0]),
                            float(best_line["raw_coords"][1]),
                        ],
                        [
                            float(best_line["raw_coords"][2]),
                            float(best_line["raw_coords"][3]),
                        ],
                    ],
                    "rank": round(float(best_line["rank"]), 4),
                    "scores": {
                        k: (round(float(v), 3) if v is not None else None)
                        for k, v in best_line["scores"].items()
                    },
                    "rgb_edge_strength": (
                        round(float(best_line["rgb_edge_strength"]), 1)
                        if best_line["rgb_edge_strength"] is not None
                        else None
                    ),
                    "n_candidates": len(line_candidates),
                },
            }
        )

    scores = seat["scores"]
    s_back = (
        _ramp(backrest["peak_height"], 0.70, 0.86)
        if backrest is not None
        else (0.58 if seat["bbox"][1] <= 2 else 0.0)
    )
    detection_confidence = float(
        np.clip(
            0.15
            + 0.20 * scores["height"]
            + 0.18 * scores["long"]
            + 0.11 * scores["short"]
            + 0.11 * scores["area"]
            + 0.12 * scores["dark"]
            + 0.05 * scores["fill"]
            + 0.08 * s_back,
            0.0,
            0.995,
        )
    )

    backrest_detected = backrest is not None
    top_edge_graspable = bool(grasp_annotations)
    if top_edge_graspable:
        reason = (
            "chair_by_depth_hsv: "
            "seat_height+seat_size+dark_color+high_backrest+level_top_edge"
        )
    elif backrest_detected:
        reason = (
            "chair_by_depth_hsv: seat_height+seat_size+dark_color+high_backrest; "
            "top_edge_clipped_or_no_safe_level_edge"
        )
    else:
        reason = (
            "chair_by_depth_hsv: seat_height+seat_size+dark_color; backrest_clipped"
        )
    obj = ObjectResult(
        id=0,
        cls="chair",
        confidence=detection_confidence,
        position_cam=[float(v) for v in geometry["center"]],
        principal_axes=np.asarray(geometry["axes"]).tolist(),
        principal_extents=np.asarray(geometry["extents"]).tolist(),
        height_above_ground=float(geometry["height"]),
        footprint_area=float(geometry["fp_area"]),
        n_points=int(chair_mask.sum()),
        solid_ratio=float(geometry["occupancy"]),
        grasp_annotations=grasp_annotations,
        in_channel=None,
        debug={
            "reason": reason,
            "algorithm": ALGORITHM_VERSION,
            "pixel_bbox": list(pixel_bbox) if pixel_bbox else None,
            "seat_bbox": list(seat["bbox"]),
            "backrest_bbox": list(backrest["bbox"]) if backrest else None,
            "backrest_visible": backrest_detected,
            "top_edge_graspable": top_edge_graspable,
            "seat_height_median": round(float(seat["height_median"]), 4),
            "seat_fp_long": round(float(seat["rect"]["long"]), 4),
            "seat_fp_short": round(float(seat["rect"]["short"]), 4),
            "seat_area_px": int(seat["area_px"]),
            "seat_dark_ratio": round(float(seat["dark_ratio"]), 4),
            "seat_black_ratio": round(float(seat["black_ratio"]), 4),
            "seat_brown_ratio": round(float(seat["brown_ratio"]), 4),
            "backrest_peak_height": (
                round(float(backrest["peak_height"]), 4) if backrest else None
            ),
            "visible_fp_long": round(float(geometry["fp_long"]), 4),
            "visible_fp_short": round(float(geometry["fp_short"]), 4),
            "n_seat_candidates": len(seat_candidates),
            "n_backrest_candidates": len(back_candidates),
            "n_line_candidates": len(line_candidates),
        },
    )
    return FrameResult(
        objects=[obj],
        ground_plane=maps.plane,
        n_raw_clusters=len(seat_candidates),
        n_kept=1,
        debug={
            "algorithm": ALGORITHM_VERSION,
            "ground_stride": int(max(3, ground_stride)),
            "ground_sample_count": maps.ground_sample_count,
            "ground_inlier_count": maps.ground_inlier_count,
            "seat_candidates": len(seat_candidates),
            "backrest_candidates": len(back_candidates),
            "line_candidates": len(line_candidates),
        },
    )
