"""垃圾桶专用 RGB-D 检测与桶口拟合。

目标垃圾桶的已知规格为：

* 外桶口直径 255 mm；
* 高度 275 mm；
* 底部直径 180 mm。

旧实现从蓝色桶身点云的最高薄层求均值和中值半径，既删掉了真正的白色
桶沿，也没有做圆拟合。本模块改用互相独立的四类证据：

1. Canny 连通弧的椭圆拟合；
2. 白色低饱和桶沿和深色灰蓝内腔的亮度反差；
3. 椭圆下方连续的低饱和灰蓝桶身；
4. 深度和 D=255 mm 先验给出的物理尺寸。

公开结果整帧最多一个垃圾桶、一个 ``top_rim``。图像上保存原始椭圆参数，
机械臂接口仍保存三维圆心、固定物理半径和朝上的法向量。
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares


ALGORITHM_VERSION = "trash_rgbd_outer_rim_ellipse_top1_v2"

MOUTH_DIAMETER_M = 0.255
MOUTH_RADIUS_M = MOUTH_DIAMETER_M / 2.0
HEIGHT_M = 0.275
BOTTOM_DIAMETER_M = 0.180

_CANNY_LEVELS = ((15, 45), (30, 100), (70, 200))
_CONTOUR_SAMPLES = 240
_DRAW_SAMPLES = 180


def _ellipse_points(ellipse, *, count=_CONTOUR_SAMPLES, scale=1.0):
    """按 OpenCV ``RotatedRect`` 约定均匀采样椭圆。"""
    (cx, cy), (width, height), angle_deg = ellipse
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    angle = np.deg2rad(float(angle_deg))
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    local_x = 0.5 * float(width) * float(scale) * np.cos(theta)
    local_y = 0.5 * float(height) * float(scale) * np.sin(theta)
    image_x = cx + local_x * cos_angle - local_y * sin_angle
    image_y = cy + local_x * sin_angle + local_y * cos_angle
    return np.stack([image_x, image_y], axis=1)


def _sample_at(image, points):
    """最近邻采样，并返回落在图内的标记。"""
    height, width = image.shape[:2]
    x = np.rint(points[:, 0]).astype(np.int32)
    y = np.rint(points[:, 1]).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.zeros((len(points),) + image.shape[2:], dtype=image.dtype)
    values[valid] = image[y[valid], x[valid]]
    return values, valid


def _major_axis_angle(ellipse):
    """返回椭圆长轴相对图像 x 轴的角度，范围 [-90, 90)。"""
    (_, _), (width, height), angle = ellipse
    if width < height:
        angle += 90.0
    angle = (float(angle) + 90.0) % 180.0 - 90.0
    return angle


def _local_ellipse_mask(shape, ellipse, *, scale):
    """只创建椭圆附近的小掩膜，避免为每个候选分配整幅 720p 掩膜。"""
    (cx, cy), (width, height), angle = ellipse
    major = max(float(width), float(height))
    pad = max(4, int(np.ceil(0.62 * major * float(scale))))
    x0 = max(0, int(np.floor(cx)) - pad)
    y0 = max(0, int(np.floor(cy)) - pad)
    x1 = min(shape[1], int(np.ceil(cx)) + pad + 1)
    y1 = min(shape[0], int(np.ceil(cy)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1), np.zeros((0, 0), dtype=bool)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    axes = (
        max(1, int(round(0.5 * float(width) * float(scale)))),
        max(1, int(round(0.5 * float(height) * float(scale)))),
    )
    cv2.ellipse(
        mask,
        (int(round(cx - x0)), int(round(cy - y0))),
        axes,
        float(angle),
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return (x0, y0, x1, y1), mask > 127


def _body_statistics(rgb_bgr, hsv, ellipse):
    """统计桶口下方收口梯形中的灰蓝桶身证据。"""
    (cx, cy), (width, height), _ = ellipse
    major = max(float(width), float(height))
    minor = min(float(width), float(height))
    image_h, image_w = rgb_bgr.shape[:2]
    x0 = max(0, int(np.floor(cx - 0.50 * major)))
    x1 = min(image_w, int(np.ceil(cx + 0.50 * major)))
    y0 = max(0, int(np.floor(cy)))
    y1 = min(image_h, int(np.ceil(cy + 1.30 * major)))
    if x1 <= x0 or y1 <= y0:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    progress = (yy - (cy + 0.25 * minor)) / max(major, 1e-6)
    half_width = major * (
        0.40 - 0.10 * np.clip(progress, 0.0, 1.0)
    )
    body_mask = (
        (progress >= 0.0)
        & (progress <= 0.95)
        & (np.abs(xx - cx) <= half_width)
    )
    if int(body_mask.sum()) < 40:
        return None

    body_bgr = rgb_bgr[y0:y1, x0:x1][body_mask]
    body_hsv = hsv[y0:y1, x0:x1][body_mask]
    b_minus_r = (
        body_bgr[:, 0].astype(np.int16)
        - body_bgr[:, 2].astype(np.int16)
    )
    muted_blue = (
        (b_minus_r >= 18)
        & (b_minus_r <= 65)
        & (body_hsv[:, 0] >= 80)
        & (body_hsv[:, 0] <= 135)
        & (body_hsv[:, 1] >= 20)
        & (body_hsv[:, 1] <= 120)
        & (body_hsv[:, 2] >= 35)
        & (body_hsv[:, 2] <= 220)
    )
    return {
        "median_s": float(np.median(body_hsv[:, 1])),
        "median_v": float(np.median(body_hsv[:, 2])),
        "median_b_minus_r": float(np.median(b_minus_r)),
        "muted_blue_fraction": float(np.mean(muted_blue)),
        "pixel_count": int(body_mask.sum()),
    }


def _score_candidate(
    rgb_bgr,
    depth_mm,
    hsv,
    edge_distance,
    ellipse,
    intr,
):
    """计算单个椭圆的可解释证据；不满足硬门时返回 ``None``。"""
    (cx, cy), (width, height), angle = ellipse
    width = float(width)
    height = float(height)
    major = max(width, height)
    minor = min(width, height)
    aspect = minor / max(major, 1e-6)
    image_h, image_w = rgb_bgr.shape[:2]

    # 头部相机画面最下方是机器人自身；该区域不发布物体抓取标注。
    if cy >= 0.76 * image_h:
        return None
    major_angle = _major_axis_angle(ellipse)
    if abs(major_angle) > 38.0:
        return None
    if not (0.42 <= aspect <= 0.90):
        return None

    contour = _ellipse_points(ellipse)
    distances, visible = _sample_at(edge_distance, contour)
    visible_count = int(visible.sum())
    if visible_count < 0.42 * len(contour):
        return None
    visible_distances = distances[visible].astype(np.float32)
    edge_support = float(np.mean(visible_distances <= 2.25))
    edge_mean = float(np.mean(np.exp(-visible_distances / 2.0)))
    edge_p90 = float(np.percentile(visible_distances, 90))
    coverage_parts = np.array_split(np.arange(len(contour)), 16)
    coverage_flags = []
    for indices in coverage_parts:
        valid_indices = indices[visible[indices]]
        if len(valid_indices):
            coverage_flags.append(
                bool(np.any(distances[valid_indices] <= 2.25))
            )
    edge_coverage = (
        float(np.mean(coverage_flags)) if coverage_flags else 0.0
    )
    if edge_support < 0.65 or edge_coverage < 0.80:
        return None

    ring_points = np.concatenate(
        [
            _ellipse_points(ellipse, scale=scale)
            for scale in (0.94, 0.98, 1.02, 1.06)
        ],
        axis=0,
    )
    ring_hsv, ring_visible = _sample_at(hsv, ring_points)
    ring_hsv = ring_hsv[ring_visible]
    if len(ring_hsv) < 80:
        return None
    ring_v = float(np.percentile(ring_hsv[:, 2], 70))
    ring_s = float(np.median(ring_hsv[:, 1]))
    white_support = float(
        np.mean((ring_hsv[:, 1] <= 80) & (ring_hsv[:, 2] >= 105))
    )
    if white_support < 0.38:
        return None

    crop, inner_mask = _local_ellipse_mask(
        hsv.shape,
        ellipse,
        scale=0.68,
    )
    x0, y0, x1, y1 = crop
    inner_hsv = hsv[y0:y1, x0:x1][inner_mask]
    if len(inner_hsv) < 20:
        return None
    inner_v = float(np.median(inner_hsv[:, 2]))
    inner_blue = float(
        np.mean(
            (
                (inner_hsv[:, 0] >= 80)
                & (inner_hsv[:, 0] <= 135)
                & (inner_hsv[:, 1] >= 15)
            )
            | ((inner_hsv[:, 1] < 55) & (inner_hsv[:, 2] < 145))
        )
    )
    contrast = ring_v - inner_v
    if contrast < 55.0:
        return None

    body = _body_statistics(rgb_bgr, hsv, ellipse)
    if body is None:
        return None
    body_s = body["median_s"]
    body_v = body["median_v"]
    body_b_minus_r = body["median_b_minus_r"]
    if body_s < 43.0:
        return None
    if not (2.0 <= body_b_minus_r <= 68.0):
        return None
    if body_v > 115.0 and body_b_minus_r < 27.0:
        return None

    contour_depth, depth_visible = _sample_at(depth_mm, contour)
    depth_good = (
        depth_visible
        & (contour_depth >= 300)
        & (contour_depth <= 5000)
    )
    if int(depth_good.sum()) < 0.45 * visible_count:
        return None
    rim_depth_m = float(np.median(contour_depth[depth_good])) / 1000.0
    effective_focal = float(
        np.hypot(
            float(intr["fx"]) * np.cos(np.deg2rad(major_angle)),
            float(intr["fy"]) * np.sin(np.deg2rad(major_angle)),
        )
    )
    measured_diameter_m = major * rim_depth_m / max(
        effective_focal,
        1e-6,
    )
    if not (0.190 <= measured_diameter_m <= 0.325):
        return None

    physical_score = float(
        np.exp(
            -0.5 * ((measured_diameter_m - MOUTH_DIAMETER_M) / 0.055) ** 2
        )
    )
    edge_score = (
        0.55 * edge_support
        + 0.25 * edge_mean
        + 0.20 * edge_coverage
    )
    appearance_score = (
        0.27 * np.clip((contrast - 10.0) / 60.0, 0.0, 1.0)
        + 0.20 * white_support
        + 0.20 * inner_blue
        + 0.28
        * np.clip(body["muted_blue_fraction"] / 0.30, 0.0, 1.0)
        + 0.05 * np.clip((130.0 - ring_s) / 100.0, 0.0, 1.0)
    )
    aspect_score = float(np.clip((aspect - 0.32) / 0.22, 0.0, 1.0))
    depth_support = float(depth_good.sum() / max(visible_count, 1))
    score = float(
        0.34 * edge_score
        + 0.34 * appearance_score
        + 0.20 * physical_score
        + 0.07 * aspect_score
        + 0.05 * depth_support
    )
    # 椅背遮住桶身时，椭圆弧、白沿和 D255 深度证据仍可很强，但桶身颜色
    # 支持会被黑色椅背稀释。硬门已经逐项成立，这里只拒绝明显低质组合。
    if score < 0.74:
        return None

    return {
        "score": score,
        "ellipse": (
            (float(cx), float(cy)),
            (width, height),
            float(angle),
        ),
        "major": major,
        "minor": minor,
        "major_angle_deg": major_angle,
        "aspect": aspect,
        "edge_support": edge_support,
        "edge_coverage": edge_coverage,
        "edge_distance_p90_px": edge_p90,
        "white_support": white_support,
        "inner_blue_support": inner_blue,
        "rim_inner_contrast": contrast,
        "body": body,
        "rim_depth_m": rim_depth_m,
        "measured_diameter_m": measured_diameter_m,
        "effective_focal_px": effective_focal,
        "visible_fraction": float(visible_count / len(contour)),
    }


def _candidate_rank(candidate):
    """Top-1 排序：总分后依次看像素贴合、可见覆盖和尺寸先验。"""
    return (
        float(candidate["score"]),
        float(candidate["edge_support"]),
        float(candidate["edge_coverage"]),
        -abs(float(candidate["measured_diameter_m"]) - MOUTH_DIAMETER_M),
        float(candidate["major"]),
    )


def _find_candidates(rgb_bgr, depth_mm, intr):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    edge_union = np.zeros(gray.shape, dtype=np.uint8)
    raw_ellipses = []
    for low, high in _CANNY_LEVELS:
        edges = cv2.Canny(gray, low, high, L2gradient=True)
        edge_union = cv2.bitwise_or(edge_union, edges)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        for contour in contours:
            if len(contour) < 18:
                continue
            try:
                ellipse = cv2.fitEllipseAMS(contour)
            except cv2.error:
                continue
            (cx, cy), (width, height), _ = ellipse
            major = max(float(width), float(height))
            minor = min(float(width), float(height))
            if not (34.0 <= major <= 270.0):
                continue
            if not (12.0 <= minor <= 225.0):
                continue
            if minor / max(major, 1e-6) < 0.28:
                continue
            if not (-0.45 * major <= cx <= rgb_bgr.shape[1] + 0.45 * major):
                continue
            if not (-0.45 * major <= cy <= rgb_bgr.shape[0] + 0.45 * major):
                continue
            raw_ellipses.append(ellipse)

    edge_distance = cv2.distanceTransform(
        255 - edge_union,
        cv2.DIST_L2,
        3,
    )
    accepted = []
    for ellipse in raw_ellipses:
        candidate = _score_candidate(
            rgb_bgr,
            depth_mm,
            hsv,
            edge_distance,
            ellipse,
            intr,
        )
        if candidate is None:
            continue
        # 同一条桶沿会在三个 Canny 阈值下重复出现；保留拟合质量最高者。
        old_index = None
        cx, cy = candidate["ellipse"][0]
        for index, old in enumerate(accepted):
            old_cx, old_cy = old["ellipse"][0]
            if (
                np.hypot(cx - old_cx, cy - old_cy)
                < 0.18 * max(candidate["major"], old["major"])
                and abs(candidate["major"] - old["major"])
                < 0.22 * max(candidate["major"], old["major"])
            ):
                old_index = index
                break
        if old_index is None:
            accepted.append(candidate)
        elif _candidate_rank(candidate) > _candidate_rank(
            accepted[old_index]
        ):
            accepted[old_index] = candidate

    accepted.sort(key=_candidate_rank, reverse=True)
    return accepted, {
        "raw_ellipse_count": len(raw_ellipses),
        "accepted_ellipse_count": len(accepted),
    }


def _up_normal(ground_plane):
    normal = np.asarray(ground_plane[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    # 相机 y 轴向下，所以“上”法向在通常姿态下 y 分量为负。
    if normal[1] > 0:
        normal = -normal
    return normal


def _plane_axes(normal):
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis_u = np.cross(normal, reference)
    if np.linalg.norm(axis_u) < 1e-6:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u) + 1e-12
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v) + 1e-12
    return axis_u, axis_v


def _normalized_ellipse_parameters(ellipse):
    """把 OpenCV 椭圆统一成中心、长轴、短轴和长轴角。"""
    (cx, cy), (width, height), angle = ellipse
    if float(width) >= float(height):
        major = float(width)
        minor = float(height)
        major_angle = float(angle)
    else:
        major = float(height)
        minor = float(width)
        major_angle = float(angle) + 90.0
    major_angle = (major_angle + 90.0) % 180.0 - 90.0
    return np.array(
        [float(cx), float(cy), major, minor, major_angle],
        dtype=np.float64,
    )


def _ellipse_angle_delta(first, second):
    return (float(first) - float(second) + 90.0) % 180.0 - 90.0


def _project_circle(center, normal, intr, *, count=_DRAW_SAMPLES):
    """投影给机械臂发布的真实三维圆。"""
    axis_u, axis_v = _plane_axes(normal)
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points_3d = (
        np.asarray(center, dtype=np.float64)[None, :]
        + MOUTH_RADIUS_M * np.cos(theta)[:, None] * axis_u[None, :]
        + MOUTH_RADIUS_M * np.sin(theta)[:, None] * axis_v[None, :]
    )
    if not np.all(np.isfinite(points_3d)) or np.min(points_3d[:, 2]) <= 0.05:
        raise ValueError("三维桶口圆落在相机后方或包含非有限值")
    projected = np.stack(
        [
            float(intr["fx"]) * points_3d[:, 0] / points_3d[:, 2]
            + float(intr["cx"]),
            float(intr["fy"]) * points_3d[:, 1] / points_3d[:, 2]
            + float(intr["cy"]),
        ],
        axis=1,
    )
    return projected


def _projected_ellipse(center, normal, intr):
    projected = _project_circle(center, normal, intr, count=240)
    fitted = cv2.fitEllipseDirect(
        projected.astype(np.float32).reshape(-1, 1, 2)
    )
    return _normalized_ellipse_parameters(fitted), projected


def _fit_3d_circle_to_ellipse(ellipse, initial_center, ground_normal, intr):
    """优化固定半径三维圆，使其重投影与 RGB 椭圆严格一致。

    五个自由度是三维圆心和相对地面法向的两个小倾角。法向、深度均带先验，
    但五个图像椭圆参数是主约束。使用绝对步长数值 Jacobian，避免 OpenCV
    float32 椭圆拟合在零倾角处产生伪零梯度。
    """
    target = _normalized_ellipse_parameters(ellipse)
    initial_center = np.asarray(initial_center, dtype=np.float64)
    ground_normal = np.asarray(ground_normal, dtype=np.float64)
    ground_normal /= np.linalg.norm(ground_normal) + 1e-12
    tangent_u, tangent_v = _plane_axes(ground_normal)

    def unpack(parameters):
        normal = (
            ground_normal
            + float(parameters[3]) * tangent_u
            + float(parameters[4]) * tangent_v
        )
        normal /= np.linalg.norm(normal) + 1e-12
        if normal[1] > 0:
            normal = -normal
        return np.asarray(parameters[:3], dtype=np.float64), normal

    def residual(parameters):
        center, normal = unpack(parameters)
        projected, _ = _projected_ellipse(center, normal, intr)
        return np.array(
            [
                projected[0] - target[0],
                projected[1] - target[1],
                projected[2] - target[2],
                projected[3] - target[3],
                2.0 * _ellipse_angle_delta(projected[4], target[4]),
                float(parameters[3]) / 0.35,
                float(parameters[4]) / 0.35,
                (center[2] - initial_center[2]) / 0.25,
            ],
            dtype=np.float64,
        )

    absolute_steps = np.array(
        [1e-4, 1e-4, 5e-4, 2e-3, 2e-3],
        dtype=np.float64,
    )

    def jacobian(parameters):
        columns = []
        for index, step in enumerate(absolute_steps):
            plus = np.asarray(parameters, dtype=np.float64).copy()
            minus = np.asarray(parameters, dtype=np.float64).copy()
            plus[index] += step
            minus[index] -= step
            columns.append(
                (residual(plus) - residual(minus)) / (2.0 * step)
            )
        return np.stack(columns, axis=1)

    initial_parameters = np.concatenate(
        [initial_center, np.zeros(2, dtype=np.float64)]
    )
    error = None
    try:
        solution = least_squares(
            residual,
            initial_parameters,
            jac=jacobian,
            bounds=(
                np.array([-5.0, -5.0, 0.2, -0.8, -0.8]),
                np.array([5.0, 5.0, 6.0, 0.8, 0.8]),
            ),
            x_scale=np.array([0.2, 0.2, 1.0, 0.1, 0.1]),
            max_nfev=100,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        parameters = solution.x
        optimizer_success = bool(solution.success)
        optimizer_evaluations = int(solution.nfev)
    except Exception as exc:
        parameters = initial_parameters
        optimizer_success = False
        optimizer_evaluations = 0
        error = f"{type(exc).__name__}: {exc}"

    center, normal = unpack(parameters)
    projected_ellipse, projected_contour = _projected_ellipse(
        center,
        normal,
        intr,
    )
    target_contour = _ellipse_points(ellipse, count=240)
    pairwise_distances = np.linalg.norm(
        target_contour[:, None, :] - projected_contour[None, :, :],
        axis=2,
    )
    symmetric_distances = np.concatenate(
        [
            np.min(pairwise_distances, axis=1),
            np.min(pairwise_distances, axis=0),
        ]
    )
    center_error = float(
        np.linalg.norm(projected_ellipse[:2] - target[:2])
    )
    axis_error = float(
        np.max(np.abs(projected_ellipse[2:4] - target[2:4]))
    )
    angle_error = abs(
        _ellipse_angle_delta(projected_ellipse[4], target[4])
    )
    contour_p90 = float(np.percentile(symmetric_distances, 90))
    reprojection_consistent = bool(
        optimizer_success
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(normal))
        and center_error <= 1.0
        and axis_error <= 2.0
        and angle_error <= 2.0
        and contour_p90 <= 1.5
    )
    normal_tilt = float(
        np.degrees(
            np.arccos(
                np.clip(float(normal @ ground_normal), -1.0, 1.0)
            )
        )
    )
    return center, normal, {
        "optimizer_success": optimizer_success,
        "optimizer_evaluations": optimizer_evaluations,
        "optimizer_error": error,
        "reprojection_consistent": reprojection_consistent,
        "reprojection_center_error_px": center_error,
        "reprojection_axis_error_px": axis_error,
        "reprojection_angle_error_deg": angle_error,
        "reprojection_contour_p90_px": contour_p90,
        "normal_tilt_from_ground_deg": normal_tilt,
    }


def _candidate_to_hit(candidate, depth_mm, intr, ground_plane, search_debug):
    ellipse = candidate["ellipse"]
    (cx, cy), (width, height), angle = ellipse
    major = candidate["major"]
    focal = candidate["effective_focal_px"]

    # 由已知外口直径和观测长轴恢复深度。长轴在圆的透视投影中受倾角影响
    # 最小，比桶内/白塑料边缘的混合深度稳定。
    center_z = float(focal * MOUTH_DIAMETER_M / max(major, 1e-6))
    initial_center = np.array(
        [
            (cx - float(intr["cx"])) * center_z / float(intr["fx"]),
            (cy - float(intr["cy"])) * center_z / float(intr["fy"]),
            center_z,
        ],
        dtype=np.float64,
    )
    ground_normal = _up_normal(ground_plane)
    center_3d, normal, reprojection = _fit_3d_circle_to_ellipse(
        ellipse,
        initial_center,
        ground_normal,
        intr,
    )
    if not reprojection["reprojection_consistent"]:
        raise ValueError(
            "固定半径三维圆无法通过 2D 椭圆重投影一致性硬门"
        )
    axis_u, axis_v = _plane_axes(normal)
    object_center = center_3d - 0.5 * HEIGHT_M * normal
    plane_normal = np.asarray(ground_plane[:3], dtype=np.float64)
    plane_norm = np.linalg.norm(plane_normal) + 1e-12
    height_above_ground = abs(
        float(center_3d @ plane_normal + float(ground_plane[3]))
        / plane_norm
    )

    contour_uv = _ellipse_points(
        ellipse,
        count=_DRAW_SAMPLES,
    )
    x_min = max(0, int(np.floor(np.min(contour_uv[:, 0]))))
    x_max = min(
        depth_mm.shape[1] - 1,
        int(np.ceil(np.max(contour_uv[:, 0]))),
    )
    y_min = max(0, int(np.floor(np.min(contour_uv[:, 1]))))
    body_bottom = min(
        depth_mm.shape[0] - 1,
        int(np.ceil(cy + 1.18 * major)),
    )
    confidence = float(
        np.clip(
            0.72 + 0.26 * (float(candidate["score"]) - 0.74) / 0.25,
            0.72,
            0.98,
        )
    )
    visible_fraction = float(candidate["visible_fraction"])
    directly_safe = bool(
        float(candidate["edge_support"]) >= 0.90
        and float(candidate["edge_distance_p90_px"]) <= 4.0
        and visible_fraction >= 0.75
    )
    grasp_safe = bool(
        directly_safe and reprojection["reprojection_consistent"]
    )
    grasp = {
        "type": "circle",
        "name": "top_rim",
        "center": center_3d.tolist(),
        "radius": MOUTH_RADIUS_M,
        "normal": normal.tolist(),
        "axis_u": axis_u.tolist(),
        "axis_v": axis_v.tolist(),
        "orientation": "up",
        "confidence": confidence,
        "grasp_safe": grasp_safe,
        "inferred": not directly_safe,
        "hidden_arc_inferred": not directly_safe,
        "center_uv": [float(cx), float(cy)],
        "ellipse_uv": {
            "center": [float(cx), float(cy)],
            # OpenCV RotatedRect 的 width/height 是完整直径，不是半轴。
            "size": [float(width), float(height)],
            "angle_deg": float(angle),
        },
        "contour_uv": contour_uv.tolist(),
        "visible_arc_fraction": float(candidate["edge_support"]),
        "visible_image_fraction": visible_fraction,
        "rim_diameter_m": MOUTH_DIAMETER_M,
        "fit_method": ALGORITHM_VERSION,
    }
    debug = {
        "algorithm": ALGORITHM_VERSION,
        "selected_frame_top1": True,
        "candidate_score": float(candidate["score"]),
        "edge_support": float(candidate["edge_support"]),
        "edge_coverage": float(candidate["edge_coverage"]),
        "edge_distance_p90_px": float(
            candidate["edge_distance_p90_px"]
        ),
        "white_rim_support": float(candidate["white_support"]),
        "rim_inner_contrast": float(candidate["rim_inner_contrast"]),
        "body_median_s": float(candidate["body"]["median_s"]),
        "body_median_v": float(candidate["body"]["median_v"]),
        "body_median_b_minus_r": float(
            candidate["body"]["median_b_minus_r"]
        ),
        "measured_diameter_m": float(candidate["measured_diameter_m"]),
        "spec_diameter_m": MOUTH_DIAMETER_M,
        "rim_depth_m": float(candidate["rim_depth_m"]),
        "geometry_center_depth_m": center_z,
        "optimized_center_depth_m": float(center_3d[2]),
        "visible_image_fraction": visible_fraction,
        "ellipse_center_uv": [float(cx), float(cy)],
        "ellipse_size_uv": [float(width), float(height)],
        "ellipse_angle_deg": float(angle),
        "pixel_bbox": [
            x_min,
            y_min,
            max(1, x_max - x_min + 1),
            max(1, body_bottom - y_min + 1),
        ],
        "draw_bbox": False,
        **reprojection,
        **search_debug,
    }
    return {
        "confidence": confidence,
        "center_3d": object_center,
        "principal_axes": np.stack([normal, axis_u, axis_v], axis=0),
        "principal_extents": np.array(
            [HEIGHT_M, MOUTH_DIAMETER_M, BOTTOM_DIAMETER_M],
            dtype=np.float64,
        ),
        "height_above_ground": float(height_above_ground),
        "footprint_area": float(np.pi * (0.5 * BOTTOM_DIAMETER_M) ** 2),
        "n_points": int(candidate["body"]["pixel_count"]),
        "solid_ratio": float(candidate["body"]["muted_blue_fraction"]),
        "grasp_annotations": [grasp],
        "debug": debug,
    }


def detect_trash(rgb_bgr, depth_mm, intr, ground_plane):
    """检测整帧唯一垃圾桶并拟合唯一桶口。

    返回 ``(hits, debug)``。``hits`` 只能为 ``[]`` 或单元素列表；桶口没有
    直接视觉/深度证据时宁可不发布圆，也不使用旧点云圆作为降级结果。
    """
    if rgb_bgr is None:
        return [], {
            "algorithm": ALGORITHM_VERSION,
            "reason": "rgb_required",
        }
    if depth_mm.ndim != 2 or rgb_bgr.shape[:2] != depth_mm.shape[:2]:
        raise ValueError("垃圾桶专用分支要求 RGB 与 depth 像素严格对齐")

    candidates, search_debug = _find_candidates(
        rgb_bgr,
        depth_mm,
        intr,
    )
    debug = {
        "algorithm": ALGORITHM_VERSION,
        **search_debug,
    }
    if not candidates:
        debug["reason"] = "no_rim_passed_rgbd_geometry_gates"
        return [], debug

    rejected = []
    for winner in candidates:
        debug["reason"] = "top1_rim_selected"
        debug["top_score"] = float(winner["score"])
        try:
            hit = _candidate_to_hit(
                winner,
                depth_mm,
                intr,
                ground_plane,
                debug,
            )
        except Exception as exc:
            rejected.append(f"{type(exc).__name__}: {exc}")
            continue
        debug["reprojection_rejected_candidates"] = rejected
        return [hit], debug

    debug["reason"] = "all_rgbd_candidates_failed_3d_reprojection"
    debug["reprojection_rejected_candidates"] = rejected
    return [], debug
