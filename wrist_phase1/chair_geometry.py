"""Geometry-first chair detection for the independent wrist RGB-D pipeline.

The detector intentionally does not classify every dark connected component as a
chair.  A hypothesis must contain evidence for a sparse/mesh backrest and a
second chair part (normally the seat), or a particularly strong physical top
rail.  Colour is only a soft cue.  A grasp is published only for the physical
top edge of the backrest; arm rests, side rails, the seat and scene furniture
are never used as fallbacks.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


_EPS = 1.0e-6


def _clip01(value: Any) -> Any:
    return np.clip(value, 0.0, 1.0)


def _ramp(value: Any, low: float, high: float) -> Any:
    if high <= low:
        return np.asarray(value) >= high
    return _clip01((np.asarray(value, dtype=np.float32) - low) / (high - low))


def _bell(value: float, center: float, sigma: float) -> float:
    return float(math.exp(-0.5 * ((float(value) - center) / max(sigma, _EPS)) ** 2))


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def _expand_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    shape: tuple[int, int],
    x_fraction: float,
    y_fraction: float | None = None,
) -> list[int]:
    h, w = shape
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if y_fraction is None:
        y_fraction = x_fraction
    dx = max(4, int(round((x2 - x1) * x_fraction)))
    dy = max(4, int(round((y2 - y1) * y_fraction)))
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w, x2 + dx),
        min(h, y2 + dy),
    ]


def _safe_uv(point: np.ndarray, shape: tuple[int, int]) -> list[int]:
    h, w = shape
    return [
        int(np.clip(round(float(point[0])), 0, w - 1)),
        int(np.clip(round(float(point[1])), 0, h - 1)),
    ]


def _sample(
    image: np.ndarray,
    points: np.ndarray,
    default: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pixels = np.rint(points).astype(np.int32)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    output = np.full(len(points), default, dtype=np.float32)
    if np.any(valid):
        output[valid] = image[pixels[valid, 1], pixels[valid, 0]]
    return output, valid


def _line_points(p0: np.ndarray, p1: np.ndarray, count: int | None = None) -> np.ndarray:
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    length = float(np.linalg.norm(p1 - p0))
    if count is None:
        count = max(24, int(math.ceil(length)))
    alpha = np.linspace(0.0, 1.0, int(count), dtype=np.float32)
    return p0[None, :] * (1.0 - alpha[:, None]) + p1[None, :] * alpha[:, None]


def _line_frame(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    delta = np.asarray(p1, np.float32) - np.asarray(p0, np.float32)
    length = float(np.linalg.norm(delta))
    if length <= _EPS:
        tangent = np.array([1.0, 0.0], np.float32)
    else:
        tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]], np.float32)
    return tangent, normal, length


def _expand_ignore_for_metric_clearance(
    ignore: np.ndarray,
    z: np.ndarray,
    camera: dict[str, Any],
    clearance_m: float = 0.025,
) -> tuple[np.ndarray, int, float | None]:
    """Return a conservative image-space safety halo around the robot.

    The halo is derived from focal length and a foreground scene-depth
    percentile, rather than a fixed pixel count. It is used by every chair
    feature path so morphology cannot reconnect evidence through a masked
    gripper. Final grasp publication performs a second, line-depth-specific
    check in :func:`_make_geometry`.
    """

    raw = (np.asarray(ignore) > 0).astype(np.uint8)
    if not np.any(raw):
        return raw, 0, None

    valid_depth = z[np.isfinite(z) & (z > 0.0) & (raw == 0)]
    reference_depth: float | None
    if valid_depth.size >= 50:
        reference_depth = float(
            np.clip(np.percentile(valid_depth, 25), 0.20, 3.50)
        )
        focal = max(float(camera["fx"]), float(camera["fy"]))
        radius = int(math.ceil(focal * float(clearance_m) / reference_depth))
    else:
        reference_depth = None
        radius = int(math.ceil(0.015 * min(raw.shape)))

    radius = int(np.clip(radius, 3, max(3, round(0.12 * min(raw.shape)))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    expanded = cv2.dilate(raw, kernel)
    return expanded.astype(np.uint8), radius, reference_depth


def _appearance_maps(
    rgb: np.ndarray,
    ignore: np.ndarray,
) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    scale = max(rgb.shape[:2])
    local_kernel = _odd(round(scale / 58), 15)
    fine_kernel = _odd(round(scale / 150), 7)

    low_edge = cv2.Canny(gray, 28, 92)
    high_edge = cv2.Canny(gray, 58, 168)
    edge = cv2.bitwise_or(low_edge, high_edge)
    edge_density = cv2.boxFilter(
        (edge > 0).astype(np.float32),
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    gray_float = gray.astype(np.float32)
    mean = cv2.boxFilter(
        gray_float,
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    mean_square = cv2.boxFilter(
        gray_float * gray_float,
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    local_std = np.sqrt(np.maximum(0.0, mean_square - mean * mean))

    close = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fine_kernel, fine_kernel)),
    )
    thin_dark = cv2.subtract(close, gray).astype(np.float32)
    thin_dark_density = cv2.boxFilter(
        (thin_dark > 9.0).astype(np.float32),
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    diagonal_a = (
        (magnitude > 42.0)
        & (orientation >= 20.0)
        & (orientation <= 75.0)
    )
    diagonal_b = (
        (magnitude > 42.0)
        & (orientation >= 105.0)
        & (orientation <= 160.0)
    )
    diagonal_a_density = cv2.boxFilter(
        diagonal_a.astype(np.float32),
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    diagonal_b_density = cv2.boxFilter(
        diagonal_b.astype(np.float32),
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    crossed_diagonal = np.minimum(diagonal_a_density, diagonal_b_density)
    jxx = cv2.boxFilter(gx * gx, -1, (local_kernel, local_kernel), normalize=True)
    jyy = cv2.boxFilter(gy * gy, -1, (local_kernel, local_kernel), normalize=True)
    jxy = cv2.boxFilter(gx * gy, -1, (local_kernel, local_kernel), normalize=True)
    coherence = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
    coherence /= jxx + jyy + 1.0

    dark = _clip01((150.0 - value.astype(np.float32)) / 105.0)
    local_dark = cv2.boxFilter(
        (value < 132).astype(np.float32),
        -1,
        (local_kernel, local_kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    # Neutral dark plastic/mesh is useful evidence, but never a class gate.
    neutral = 1.0 - _ramp(saturation, 58.0, 168.0)
    blue = (
        _ramp(hue, 82.0, 96.0)
        * (1.0 - _ramp(hue, 132.0, 145.0))
        * _ramp(saturation, 55.0, 155.0)
    )
    warm = (
        (1.0 - _ramp(hue, 32.0, 48.0))
        * _ramp(saturation, 45.0, 150.0)
        * _ramp(value, 55.0, 150.0)
    )

    texture = (
        0.46 * _ramp(crossed_diagonal, 0.018, 0.135)
        + 0.20 * _ramp(edge_density, 0.045, 0.205)
        + 0.15 * _ramp(local_std, 7.0, 28.0)
        + 0.13 * _ramp(thin_dark_density, 0.035, 0.19)
        + 0.06 * (1.0 - 0.60 * _ramp(coherence, 0.55, 0.93))
    )
    appearance = texture * (
        0.48
        + 0.22 * neutral
        + 0.20 * _ramp(local_dark, 0.05, 0.36)
        + 0.10 * dark
    )
    appearance *= 1.0 - 0.40 * blue - 0.28 * warm
    appearance = _clip01(appearance).astype(np.float32)
    appearance[ignore > 0] = 0.0
    part_map = _clip01(
        0.52 * appearance
        + 0.27 * _ramp(local_dark, 0.04, 0.38)
        + 0.21 * dark
    ).astype(np.float32)
    part_map[ignore > 0] = 0.0
    dark_binary = (value < 137).astype(np.uint8)

    # All downstream feature maps are explicitly cleared after their local
    # filters. This prevents blur/box-filter/morphology support from growing
    # back into the expanded gripper safety band.
    blocked = ignore > 0
    for feature in (
        edge,
        edge_density,
        local_std,
        thin_dark_density,
        coherence,
        crossed_diagonal,
        dark,
        dark_binary,
        local_dark,
        neutral,
        blue,
        warm,
    ):
        feature[blocked] = 0

    return {
        "gray": gray,
        "hsv": hsv,
        "edge": edge,
        "edge_density": edge_density,
        "local_std": local_std,
        "thin_dark_density": thin_dark_density,
        "coherence": coherence,
        "crossed_diagonal": crossed_diagonal,
        "dark_score": dark.astype(np.float32),
        "dark_binary": dark_binary,
        "local_dark": local_dark,
        "neutral": neutral.astype(np.float32),
        "blue": blue.astype(np.float32),
        "warm": warm.astype(np.float32),
        "mesh": appearance,
        "part": part_map,
    }


def _extract_segments(
    gray: np.ndarray,
    edge: np.ndarray,
    ignore: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    h, w = gray.shape
    minimum = max(22.0, 0.027 * math.hypot(h, w))
    raw: list[tuple[np.ndarray, np.ndarray]] = []

    # A slight blur removes the D405 speckle/mesh moire without erasing the
    # thick physical frame.  It also keeps Hough from spending its budget on
    # hundreds of one-pixel texture fragments.
    stable_gray = cv2.GaussianBlur(gray, (3, 3), 0.75)
    stable_edge = cv2.Canny(stable_gray, 42, 138)
    stable_edge[ignore > 0] = 0

    hough = cv2.HoughLinesP(
        stable_edge,
        1.0,
        np.pi / 720.0,
        threshold=max(34, int(minimum * 0.92)),
        minLineLength=int(minimum),
        maxLineGap=max(12, int(minimum * 0.42)),
    )
    if hough is not None:
        for item in hough.reshape(-1, 4):
            raw.append(
                (
                    item[:2].astype(np.float32),
                    item[2:].astype(np.float32),
                )
            )

    retained: list[tuple[np.ndarray, np.ndarray]] = []
    for p0, p1 in sorted(raw, key=lambda pair: -float(np.linalg.norm(pair[1] - pair[0]))):
        tangent, _, length = _line_frame(p0, p1)
        points = _line_points(p0, p1, max(28, int(math.ceil(length))))
        blocked, valid = _sample((ignore > 0).astype(np.float32), points, default=1.0)
        # A Hough gap is never allowed to bridge the expanded gripper mask.
        if np.any((blocked > 0.0) | (~valid)):
            continue

        midpoint = 0.5 * (p0 + p1)
        angle = math.atan2(float(tangent[1]), float(tangent[0]))
        duplicate = False
        for q0, q1 in retained[-180:]:
            qmid = 0.5 * (q0 + q1)
            qt, _, qlength = _line_frame(q0, q1)
            qangle = math.atan2(float(qt[1]), float(qt[0]))
            angular = abs(math.sin(angle - qangle))
            if (
                np.linalg.norm(midpoint - qmid) < max(9.0, 0.035 * max(length, qlength))
                and angular < math.sin(math.radians(4.0))
                and min(length, qlength) / max(length, qlength) > 0.62
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append((p0, p1))
        if len(retained) >= 320:
            break
    return retained


def _backrest_regions(
    maps: dict[str, np.ndarray],
    ignore: np.ndarray,
) -> list[dict[str, Any]]:
    mesh = maps["mesh"]
    h, w = mesh.shape
    scale = max(h, w)
    seed = (
        (mesh > 0.47)
        & (maps["edge_density"] > 0.045)
        & (maps["thin_dark_density"] > 0.025)
        & (ignore == 0)
    ).astype(np.uint8)

    close_size = _odd(round(scale / 85), 9)
    merge_size = _odd(round(scale / 62), 13)
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    seed[ignore > 0] = 0
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seed[ignore > 0] = 0
    merged = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_size, merge_size)),
    )
    merged[ignore > 0] = 0

    minimum = max(260, int(round(0.00027 * h * w)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    regions: list[dict[str, Any]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if width < 24 or height < 20:
            continue
        if area > 0.40 * h * w:
            continue

        component = labels == index
        core = component & (seed > 0)
        core_count = int(core.sum())
        if core_count < max(90, int(0.12 * area)):
            continue
        values = mesh[core]
        quality = float(np.mean(values)) if values.size else 0.0
        edge_density = float(np.mean(maps["edge_density"][core]))
        thin_density = float(np.mean(maps["thin_dark_density"][core]))
        crossed_diagonal = float(np.mean(maps["crossed_diagonal"][core]))
        if quality < 0.51 or edge_density < 0.055 or crossed_diagonal < 0.025:
            continue

        regions.append(
            {
                "mask": component,
                "core": core,
                "bbox": [x, y, x + width, y + height],
                "area": area,
                "core_area": core_count,
                "quality": quality,
                "edge_density": edge_density,
                "thin_density": thin_density,
                "crossed_diagonal": crossed_diagonal,
            }
        )

    regions.sort(
        key=lambda item: item["quality"] * math.sqrt(float(item["core_area"])),
        reverse=True,
    )
    return regions[:18]


def _ribbon_depth(
    p0: np.ndarray,
    p1: np.ndarray,
    z: np.ndarray,
    camera: dict[str, Any],
    plane: np.ndarray | None,
    ignore: np.ndarray,
) -> dict[str, Any]:
    tangent, normal_2d, length_px = _line_frame(p0, p1)
    count = int(np.clip(round(length_px * 0.65), 32, 220))
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
    centers = p0[None, :] * (1.0 - alpha[:, None]) + p1[None, :] * alpha[:, None]

    offsets = np.arange(-4, 5, dtype=np.float32)
    profiles = []
    for offset in offsets:
        sample_points = centers + offset * normal_2d
        values, in_frame = _sample(z, sample_points, default=np.nan)
        blocked, _ = _sample(ignore, sample_points, default=1)
        values[(~in_frame) | (blocked > 0)] = np.nan
        profiles.append(values)
    profile = np.stack(profiles, axis=1)

    depth = np.full(count, np.nan, np.float32)
    for index in range(count):
        values = profile[index]
        values = values[np.isfinite(values)]
        if values.size:
            # The top rail normally occludes the background visible through mesh.
            depth[index] = float(np.percentile(values, 32))

    direct_fraction = float(np.mean(np.isfinite(depth)))
    valid = np.isfinite(depth)
    if int(valid.sum()) < 10:
        return {
            "valid_fraction": direct_fraction,
            "length_m": None,
            "residual_p90_m": None,
            "parallel_error": None,
            "height_delta_m": None,
            "endpoints_xyz": None,
        }

    aa = alpha[valid].astype(np.float64)
    zz = depth[valid].astype(np.float64)
    uu = centers[valid, 0].astype(np.float64)
    vv = centers[valid, 1].astype(np.float64)
    xx = (uu - float(camera["cx"])) * zz / max(float(camera["fx"]), _EPS)
    yy = (vv - float(camera["cy"])) * zz / max(float(camera["fy"]), _EPS)
    points = np.column_stack((xx, yy, zz))

    # Fit a 3-D line parameterised by image position.  Two robust passes remove
    # background values leaking through the black mesh.
    design = np.column_stack((aa, np.ones_like(aa)))
    coefficients, _, _, _ = np.linalg.lstsq(design, points, rcond=None)
    predicted = design @ coefficients
    residual = np.linalg.norm(points - predicted, axis=1)
    for threshold in (0.055, 0.035):
        keep = residual <= max(threshold, float(np.percentile(residual, 72)))
        if int(keep.sum()) < 9:
            break
        coefficients, _, _, _ = np.linalg.lstsq(design[keep], points[keep], rcond=None)
        predicted = design @ coefficients
        residual = np.linalg.norm(points - predicted, axis=1)

    robust_limit = max(0.040, float(np.percentile(residual, 75)))
    inliers = residual <= robust_limit
    if int(inliers.sum()) >= 9:
        coefficients, _, _, _ = np.linalg.lstsq(
            design[inliers],
            points[inliers],
            rcond=None,
        )
        predicted = design @ coefficients
        residual = np.linalg.norm(points - predicted, axis=1)
    else:
        inliers = np.ones(len(points), dtype=bool)

    endpoint0 = coefficients[1]
    endpoint1 = coefficients[0] + coefficients[1]
    direction = endpoint1 - endpoint0
    length_m = float(np.linalg.norm(direction))
    if length_m > _EPS:
        direction /= length_m

    parallel_error = None
    height_delta = None
    if plane is not None and plane.shape == (4,):
        normal = plane[:3].astype(np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > _EPS:
            normal /= norm
            parallel_error = float(abs(np.dot(direction, normal)))
            height_delta = float(abs(np.dot(endpoint1 - endpoint0, normal)))

    return {
        "valid_fraction": direct_fraction,
        "inlier_fraction": float(np.mean(inliers)),
        "length_m": length_m,
        "residual_p90_m": float(np.percentile(residual[inliers], 90)),
        "parallel_error": parallel_error,
        "height_delta_m": height_delta,
        "endpoints_xyz": [
            [float(value) for value in endpoint0],
            [float(value) for value in endpoint1],
        ],
    }


def _scene_reliability(scene: dict[str, Any] | None) -> float:
    if not scene:
        return 0.0
    debug = scene.get("debug") or {}
    inlier_fraction = float(debug.get("inlier_fraction", 0.0))
    floor = np.asarray(scene.get("floor_mask", False))
    valid = np.asarray(scene.get("valid", False))
    if floor.shape != valid.shape or floor.size == 0:
        floor_fraction = 0.0
    else:
        denominator = max(1, int(np.sum(valid)))
        floor_fraction = float(np.sum(floor & valid) / denominator)
    return float(
        _clip01(
            0.58 * _ramp(inlier_fraction, 0.08, 0.38)
            + 0.42 * _ramp(floor_fraction, 0.08, 0.52)
        )
    )


def _trusted_ground_scene(scene: dict[str, Any] | None) -> bool:
    """Whether ``scene`` is reliable enough for hard chair-height decisions.

    A close, broad chair surface can itself be the dominant plane in a wrist
    view.  Such a plane often has many inliers, so inlier fraction alone is not
    a safe ground test.  The perpendicular camera-to-plane distance must also
    be large enough to contain an approximately 0.85 m chair below the camera,
    and the fitted plane must have spatially distributed support.
    """

    if not scene:
        return False
    plane = np.asarray(scene.get("plane"), np.float64).reshape(-1)
    valid = np.asarray(scene.get("valid", False))
    floor = np.asarray(scene.get("floor_mask", False))
    height = np.asarray(scene.get("height"))
    if (
        plane.shape != (4,)
        or not np.all(np.isfinite(plane))
        or valid.ndim != 2
        or floor.shape != valid.shape
        or height.shape != valid.shape
    ):
        return False
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm <= _EPS:
        return False
    camera_height = abs(float(plane[3])) / normal_norm
    # A plane closer than this is normally the seat/backrest itself.  This is a
    # physical camera-height condition, not an image-position assumption.
    if not 0.72 <= camera_height <= 5.50:
        return False

    valid_count = int(np.sum(valid))
    floor_count = int(np.sum(floor & valid))
    if valid_count < 600 or floor_count < 180:
        return False
    floor_fraction = float(floor_count / valid_count)
    inlier_fraction = float((scene.get("debug") or {}).get("inlier_fraction", 0.0))
    if floor_fraction < 0.035 or inlier_fraction < 0.045:
        return False

    # A floor plane should not be supported by only one compact object patch.
    ys, xs = np.nonzero(floor & valid)
    if len(xs) < 180:
        return False
    h, w = valid.shape
    occupied = set(
        zip(
            np.clip(ys * 3 // max(h, 1), 0, 2).tolist(),
            np.clip(xs * 4 // max(w, 1), 0, 3).tolist(),
        )
    )
    return len(occupied) >= 5


def _scene_height_summary(
    mask: np.ndarray,
    scene: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return robust ground-relative evidence for a candidate mask."""

    trusted = _trusted_ground_scene(scene)
    if not scene:
        return {
            "trusted": False,
            "sample_count": 0,
            "coverage": 0.0,
            "p50_m": None,
            "p90_m": None,
            "p95_m": None,
            "high_fraction": None,
        }
    height = np.asarray(scene.get("height"))
    valid = np.asarray(scene.get("valid", False))
    selected = (np.asarray(mask) > 0) & valid & np.isfinite(height)
    values = height[selected]
    mask_count = max(1, int(np.sum(np.asarray(mask) > 0)))
    if values.size < 20:
        return {
            "trusted": trusted,
            "sample_count": int(values.size),
            "coverage": float(values.size / mask_count),
            "p50_m": None,
            "p90_m": None,
            "p95_m": None,
            "high_fraction": None,
        }
    p50, p90, p95 = (float(value) for value in np.percentile(values, (50, 90, 95)))
    return {
        "trusted": trusted,
        "sample_count": int(values.size),
        "coverage": float(values.size / mask_count),
        "p50_m": p50,
        "p90_m": p90,
        "p95_m": p95,
        "high_fraction": float(np.mean(values >= 0.58)),
    }


def _passes_scene_chair_height(
    mask: np.ndarray,
    scene: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    """Hard-reject low objects only when the fitted ground is trustworthy."""

    summary = _scene_height_summary(mask, scene)
    if not summary["trusted"]:
        return True, summary
    if summary["sample_count"] < 30 or summary["coverage"] < 0.035:
        # Insufficient metric evidence cannot certify a line, but it also must
        # not erase a visual-only chair under depth holes.
        return True, summary
    p90 = summary["p90_m"]
    p95 = summary["p95_m"]
    high_fraction = summary["high_fraction"]
    accepted = bool(
        p90 is not None
        and p95 is not None
        and high_fraction is not None
        and (p90 >= 0.56 or p95 >= 0.65)
        and high_fraction >= 0.018
    )
    return accepted, summary


def _metric_kernel_radius(
    camera: dict[str, Any],
    depth_m: float | None,
    distance_m: float,
    shape: tuple[int, int],
    *,
    minimum: int = 1,
    maximum_fraction: float = 0.035,
) -> int:
    """Project a physical distance to a bounded image-space radius."""

    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0.05:
        radius = int(round(0.004 * math.hypot(*shape)))
    else:
        focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
        radius = int(math.ceil(focal * float(distance_m) / float(depth_m)))
    maximum = max(minimum, int(round(maximum_fraction * min(shape))))
    return int(np.clip(radius, minimum, maximum))


def _robust_ground_rect(
    mask: np.ndarray,
    scene: dict[str, Any],
) -> dict[str, Any] | None:
    """Metric OBB and robust centre for a possibly perforated scene component."""

    valid = np.asarray(scene["valid"])
    gu = np.asarray(scene["ground_u"])
    gv = np.asarray(scene["ground_v"])
    height = np.asarray(scene["height"])
    selected = (
        (np.asarray(mask) > 0)
        & valid
        & np.isfinite(gu)
        & np.isfinite(gv)
        & np.isfinite(height)
    )
    ys, xs = np.nonzero(selected)
    if len(xs) < 18:
        return None
    ground = np.column_stack((gu[selected], gv[selected])).astype(np.float32)
    center = np.median(ground, axis=0)
    radii = np.linalg.norm(ground - center[None, :], axis=1)
    cutoff = float(np.percentile(radii, 97))
    trimmed = ground[radii <= cutoff]
    if len(trimmed) < 18:
        trimmed = ground
    rect = cv2.minAreaRect(trimmed)
    side_a, side_b = (float(value) for value in rect[1])
    long_side, short_side = max(side_a, side_b), min(side_a, side_b)
    values = height[selected]
    depth = np.asarray(scene["z"])[selected]
    depth = depth[np.isfinite(depth)]
    return {
        "center_ground": center.astype(np.float64),
        "radius_m": float(np.percentile(radii, 92)),
        "long_m": long_side,
        "short_m": short_side,
        "height_p10_m": float(np.percentile(values, 10)),
        "height_median_m": float(np.median(values)),
        "height_p95_m": float(np.percentile(values, 95)),
        "depth_median_m": (
            None if depth.size < 10 else float(np.median(depth))
        ),
        "pixel_center": np.array(
            [float(np.median(xs)), float(np.median(ys))],
            np.float32,
        ),
        "point_count": int(len(xs)),
    }


def _metric_fragments(
    seed: np.ndarray,
    scene: dict[str, Any],
    ignore: np.ndarray,
    *,
    minimum_points: int = 22,
) -> list[dict[str, Any]]:
    """Describe connected fragments without filling across the robot mask."""

    seed_u8 = (np.asarray(seed) > 0).astype(np.uint8)
    seed_u8[np.asarray(ignore) > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed_u8, 8)
    order = sorted(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:96]
    fragments: list[dict[str, Any]] = []
    for index in order:
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_points:
            continue
        mask = labels == index
        metric = _robust_ground_rect(mask, scene)
        if metric is None or metric["point_count"] < minimum_points:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        fragments.append(
            {
                "mask": mask,
                "area_px": area,
                "bbox": [x, y, x + width, y + height],
                **metric,
            }
        )
    return fragments


def _merge_metric_fragments(
    fragments: list[dict[str, Any]],
    scene: dict[str, Any],
    ignore: np.ndarray,
    *,
    maximum_gap_m: float,
    maximum_span_m: float,
    maximum_height_delta_m: float,
    minimum_component_fraction: float = 0.0,
    minimum_component_points: int = 0,
) -> list[dict[str, Any]]:
    """Union fragments by metric gap while refusing to bridge a gripper.

    A sequence of tiny, individually plausible depth islands must not grow an
    object all the way across the image.  When ``minimum_component_fraction``
    is non-zero, the largest connected fragment is the anchor and only
    substantive pieces are retained in the final union.  The threshold is
    relative to the observed anchor, so a backrest split into two sizeable
    pieces remains valid without assuming a fixed pixel size or image region.
    """

    if not fragments:
        return []
    parent = list(range(len(fragments)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    blocked_image = (np.asarray(ignore) > 0).astype(np.float32)
    for first in range(len(fragments)):
        a = fragments[first]
        for second in range(first + 1, len(fragments)):
            b = fragments[second]
            height_delta = abs(
                float(a["height_median_m"]) - float(b["height_median_m"])
            )
            if height_delta > maximum_height_delta_m:
                continue
            distance = float(
                np.linalg.norm(a["center_ground"] - b["center_ground"])
            )
            gap = max(0.0, distance - a["radius_m"] - b["radius_m"])
            if gap > maximum_gap_m:
                continue
            if (
                distance + a["radius_m"] + b["radius_m"]
                > 1.35 * maximum_span_m
            ):
                continue

            # Metric proximity alone must not reconnect a component split by
            # the expanded robot safety mask.
            bridge = _line_points(
                a["pixel_center"],
                b["pixel_center"],
                max(12, int(np.linalg.norm(a["pixel_center"] - b["pixel_center"]))),
            )
            blocked, valid = _sample(blocked_image, bridge, default=1.0)
            if np.any((blocked > 0.0) | (~valid)):
                continue

            trial = a["mask"] | b["mask"]
            metric = _robust_ground_rect(trial, scene)
            if metric is None or metric["long_m"] > maximum_span_m:
                continue
            union(first, second)

    grouped: dict[int, list[int]] = {}
    for index, fragment in enumerate(fragments):
        root = find(index)
        grouped.setdefault(root, []).append(index)

    output: list[dict[str, Any]] = []
    for indices in grouped.values():
        ordered = sorted(
            (fragments[index] for index in indices),
            key=lambda item: item["point_count"],
            reverse=True,
        )
        if minimum_component_fraction > 0.0 and len(ordered) > 1:
            anchor_points = int(ordered[0]["point_count"])
            threshold = max(
                int(minimum_component_points),
                int(math.ceil(anchor_points * minimum_component_fraction)),
            )
            ordered = [
                fragment
                for position, fragment in enumerate(ordered)
                if position == 0 or int(fragment["point_count"]) >= threshold
            ]

        mask = np.zeros_like(np.asarray(ignore), dtype=np.uint8)
        for fragment in ordered:
            mask[np.asarray(fragment["mask"]) > 0] = 1
        mask = mask.astype(np.uint8)
        mask[np.asarray(ignore) > 0] = 0
        metric = _robust_ground_rect(mask, scene)
        bbox = _bbox_from_mask(mask)
        if metric is None or bbox is None or metric["long_m"] > maximum_span_m:
            continue
        output.append(
            {
                "mask": mask,
                "bbox": bbox,
                "area_px": int(np.sum(mask > 0)),
                **metric,
            }
        )
    output.sort(
        key=lambda item: (
            item["point_count"],
            item["height_p95_m"],
            item["long_m"],
        ),
        reverse=True,
    )
    return output


def _side_samples(
    image: np.ndarray,
    midpoint: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    length: float,
    side: float,
    near: float,
    far: float,
    ignore: np.ndarray,
) -> tuple[float, float]:
    along = np.linspace(-0.43 * length, 0.43 * length, 23, dtype=np.float32)
    across = np.linspace(near, far, 9, dtype=np.float32) * float(side)
    points = (
        midpoint[None, None, :]
        + along[:, None, None] * tangent[None, None, :]
        + across[None, :, None] * normal[None, None, :]
    ).reshape(-1, 2)
    values, valid = _sample(image, points, default=np.nan)
    blocked, _ = _sample(ignore, points, default=1)
    values = values[np.isfinite(values) & valid & (blocked == 0)]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(values.size / len(points))


def _score_top_lines(
    region: dict[str, Any],
    segments: list[tuple[np.ndarray, np.ndarray]],
    maps: dict[str, np.ndarray],
    z: np.ndarray,
    camera: dict[str, Any],
    ignore: np.ndarray,
    scene: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    h, w = maps["gray"].shape
    bbox = _expand_bbox(region["bbox"], (h, w), 0.22, 0.28)
    x1, y1, x2, y2 = bbox
    diagonal = math.hypot(x2 - x1, y2 - y1)
    minimum = max(24.0, 0.105 * diagonal)

    ys, xs = np.nonzero(region["core"])
    core_points = np.column_stack((xs, ys)).astype(np.float32)
    plane = None
    if scene and scene.get("plane") is not None:
        candidate_plane = np.asarray(scene["plane"], np.float64).reshape(-1)
        if candidate_plane.shape == (4,) and np.all(np.isfinite(candidate_plane)):
            plane = candidate_plane
    scene_quality = _scene_reliability(scene)

    scored: list[dict[str, Any]] = []
    raw_count = 0
    for raw_p0, raw_p1 in segments:
        midpoint = 0.5 * (raw_p0 + raw_p1)
        if not (x1 <= midpoint[0] < x2 and y1 <= midpoint[1] < y2):
            continue
        tangent, normal, length = _line_frame(raw_p0, raw_p1)
        if length < minimum:
            continue
        raw_count += 1

        points = _line_points(raw_p0, raw_p1)
        gray_line = []
        edge_line = []
        for offset in (-3, -2, -1, 0, 1, 2, 3):
            shifted = points + float(offset) * normal
            gray_values, valid = _sample(maps["gray"], shifted, default=np.nan)
            edge_values, _ = _sample(maps["edge"], shifted, default=0.0)
            gray_values[~valid] = np.nan
            gray_line.append(gray_values)
            edge_line.append(edge_values)
        gray_stack = np.stack(gray_line)
        edge_stack = np.stack(edge_line)
        along_dark = float(np.mean(np.nanmin(gray_stack, axis=0) < 128.0))
        edge_support = float(np.mean(np.max(edge_stack, axis=0) > 0.0))
        if along_dark < 0.28 or edge_support < 0.27:
            continue

        near = max(5.0, min(10.0, 0.030 * length))
        far = max(20.0, min(105.0, 0.42 * length))
        positive, positive_valid = _side_samples(
            maps["mesh"], midpoint, tangent, normal, length, 1.0, near, far, ignore
        )
        negative, negative_valid = _side_samples(
            maps["mesh"], midpoint, tangent, normal, length, -1.0, near, far, ignore
        )
        if positive >= negative:
            back_sign = 1.0
            mesh_behind, mesh_free = positive, negative
            side_valid = min(positive_valid, negative_valid)
        else:
            back_sign = -1.0
            mesh_behind, mesh_free = negative, positive
            side_valid = min(positive_valid, negative_valid)
        if (
            mesh_behind < 0.27
            or mesh_behind - mesh_free < 0.018
            or side_valid < 0.42
        ):
            continue

        signed = (core_points - midpoint[None, :]) @ normal * back_sign
        tangential = np.abs((core_points - midpoint[None, :]) @ tangent)
        near_core = tangential <= 0.68 * length
        if int(np.sum(near_core)) < 30:
            continue
        signed = signed[near_core]
        positive_fraction = float(np.mean(signed > 1.5))
        wrong_side_fraction = float(np.mean(signed < -4.0))
        back_extent = float(np.percentile(signed[signed > 0], 90)) if np.any(signed > 0) else 0.0
        if back_extent < 10.0:
            continue
        if positive_fraction < 0.42 or wrong_side_fraction > 0.48:
            continue

        component_span = float(
            np.percentile(
                np.abs((core_points - midpoint[None, :]) @ tangent),
                92,
            )
            * 2.0
        )
        span_ratio = length / max(component_span, 1.0)

        depth_fit = _ribbon_depth(raw_p0, raw_p1, z, camera, plane, ignore)
        length_m = depth_fit["length_m"]
        if length_m is None:
            physical_score = float(_ramp(length, minimum, 3.2 * minimum))
        else:
            physical_score = max(
                _bell(length_m, 0.46, 0.22),
                0.82 * _bell(length_m, 0.28, 0.16),
            )

        parallel_error = depth_fit["parallel_error"]
        if parallel_error is None:
            parallel_score = 0.42
        else:
            parallel_score = float(1.0 - _ramp(parallel_error, 0.13, 0.42))

        boundary_asymmetry = float(
            _clip01((mesh_behind - mesh_free + 0.08) / 0.36)
        )
        boundary_occupancy = float(
            _ramp(positive_fraction, 0.48, 0.90)
            * (1.0 - _ramp(wrong_side_fraction, 0.08, 0.38))
        )
        span_score = max(_bell(span_ratio, 0.90, 0.48), 0.68 * _bell(span_ratio, 0.48, 0.25))
        semantic_score = float(
            0.18 * _ramp(along_dark, 0.36, 0.90)
            + 0.17 * _ramp(edge_support, 0.34, 0.90)
            + 0.20 * _ramp(mesh_behind, 0.25, 0.68)
            + 0.15 * boundary_asymmetry
            + 0.13 * boundary_occupancy
            + 0.09 * span_score
            + 0.08 * physical_score
        )
        geometry_score = float(
            0.54 * semantic_score
            + 0.15 * _ramp(depth_fit["valid_fraction"], 0.10, 0.62)
            + 0.12 * physical_score
            + 0.11 * parallel_score
            + 0.08
            * (
                1.0
                - _ramp(
                    depth_fit["residual_p90_m"]
                    if depth_fit["residual_p90_m"] is not None
                    else 0.070,
                    0.018,
                    0.070,
                )
            )
        )
        # The plane is deliberately a soft term when the global scene fit is
        # weak.  With a well-supported plane it becomes decisive.
        geometry_score *= float(
            1.0 - scene_quality * 0.28 * (1.0 - parallel_score)
        )

        scored.append(
            {
                "p0": raw_p0,
                "p1": raw_p1,
                "length_px": length,
                "length_m": length_m,
                "back_sign": back_sign,
                "back_extent_px": back_extent,
                "mesh_behind": mesh_behind,
                "mesh_free": mesh_free,
                "along_dark": along_dark,
                "edge_support": edge_support,
                "positive_fraction": positive_fraction,
                "wrong_side_fraction": wrong_side_fraction,
                "span_ratio": span_ratio,
                "semantic_score": semantic_score,
                "geometry_score": geometry_score,
                "depth_fit": depth_fit,
                "side_valid": side_valid,
                "scene_quality": scene_quality,
            }
        )

    scored.sort(
        key=lambda item: (
            item["geometry_score"],
            item["semantic_score"],
            item["length_px"],
        ),
        reverse=True,
    )
    best = scored[0] if scored else None
    return best, {
        "roi_segment_count": raw_count,
        "line_candidate_count": len(scored),
        "best_semantic_score": None if best is None else round(best["semantic_score"], 4),
        "best_geometry_score": None if best is None else round(best["geometry_score"], 4),
        "scene_quality": round(scene_quality, 4),
    }


def _polygon_mask(
    shape: tuple[int, int],
    midpoint: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    tangential_half: float,
    start: float,
    stop: float,
) -> np.ndarray:
    corners = np.array(
        [
            midpoint - tangential_half * tangent + start * normal,
            midpoint + tangential_half * tangent + start * normal,
            midpoint + tangential_half * tangent + stop * normal,
            midpoint - tangential_half * tangent + stop * normal,
        ],
        np.float32,
    )
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1, cv2.LINE_AA)
    return mask


def _seat_relation(
    region: dict[str, Any],
    line: dict[str, Any] | None,
    maps: dict[str, np.ndarray],
    scene: dict[str, Any] | None,
    ignore: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    h, w = maps["gray"].shape
    empty = np.zeros((h, w), np.uint8)
    if line is None:
        bbox = _expand_bbox(region["bbox"], (h, w), 0.20, 0.45)
        x1, y1, x2, y2 = bbox
        candidate = np.zeros((h, w), np.uint8)
        # Image-down is only a fallback used when no rail establishes the
        # backrest direction.  It is not used as a grasp-edge criterion.
        y_mid = int(round(0.55 * y1 + 0.45 * y2))
        candidate[y_mid:y2, x1:x2] = 1
    else:
        p0 = line["p0"]
        p1 = line["p1"]
        tangent, normal, length = _line_frame(p0, p1)
        normal *= float(line["back_sign"])
        midpoint = 0.5 * (p0 + p1)
        extent = float(np.clip(line["back_extent_px"], 20.0, 260.0))
        candidate = _polygon_mask(
            (h, w),
            midpoint,
            tangent,
            normal,
            max(0.44 * length, 36.0),
            max(0.50 * extent, 14.0),
            min(2.15 * extent, extent + max(55.0, 0.62 * length)),
        )

    candidate[ignore > 0] = 0
    dark = (maps["dark_binary"] > 0) & (ignore == 0)
    usable = (candidate > 0) & dark
    if int(np.sum(candidate)) < 80:
        return 0.0, empty, {"candidate_area_px": int(np.sum(candidate))}

    close_size = _odd(round(max(h, w) / 85), 9)
    seat = cv2.morphologyEx(
        usable.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    seat[ignore > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seat, 8)
    minimum = max(100, int(0.003 * np.sum(candidate)))
    kept = np.zeros_like(seat)
    largest = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= minimum:
            kept[labels == index] = 1
            largest = max(largest, area)
    kept[ignore > 0] = 0

    candidate_area = max(1, int(np.sum(candidate)))
    dark_fraction = float(np.sum(usable) / candidate_area)
    contiguous_fraction = float(largest / candidate_area)
    broadness = 0.0
    seat_bbox = _bbox_from_mask(kept)
    if seat_bbox is not None:
        sw = seat_bbox[2] - seat_bbox[0]
        sh = seat_bbox[3] - seat_bbox[1]
        broadness = float(_ramp(max(sw, sh), 24.0, 135.0))

    height_score = 0.45
    if scene is not None:
        height = np.asarray(scene.get("height"))
        valid = np.asarray(scene.get("valid"))
        if height.shape == (h, w) and valid.shape == (h, w):
            seat_values = height[(kept > 0) & valid & np.isfinite(height)]
            back_values = height[region["core"] & valid & np.isfinite(height)]
            if seat_values.size >= 20 and back_values.size >= 20:
                seat_height = float(np.median(seat_values))
                back_height = float(np.percentile(back_values, 62))
                relative = back_height - seat_height
                plausible_absolute = max(
                    _bell(seat_height, 0.46, 0.22),
                    0.65 * _bell(seat_height, 0.60, 0.34),
                )
                height_score = float(
                    0.58 * _ramp(relative, 0.06, 0.32)
                    + 0.42 * plausible_absolute
                )

    score = float(
        0.34 * _ramp(dark_fraction, 0.055, 0.36)
        + 0.31 * _ramp(contiguous_fraction, 0.018, 0.24)
        + 0.20 * broadness
        + 0.15 * height_score
    )
    return score, kept, {
        "candidate_area_px": candidate_area,
        "dark_fraction": round(dark_fraction, 4),
        "contiguous_fraction": round(contiguous_fraction, 4),
        "broadness": round(broadness, 4),
        "height_score": round(height_score, 4),
        "seat_score": round(score, 4),
    }


def _region_scene_features(
    region: dict[str, Any],
    z: np.ndarray,
    camera: dict[str, Any],
    scene: dict[str, Any] | None,
) -> dict[str, Any]:
    core = region["core"]
    valid_z = z[core & np.isfinite(z)]
    median_depth = float(np.median(valid_z)) if valid_z.size >= 20 else None
    x1, y1, x2, y2 = region["bbox"]
    if median_depth is None:
        apparent_width = None
        apparent_height = None
        size_score = float(
            _ramp(max(x2 - x1, y2 - y1), 45.0, 260.0)
        )
    else:
        apparent_width = (x2 - x1) * median_depth / max(float(camera["fx"]), _EPS)
        apparent_height = (y2 - y1) * median_depth / max(float(camera["fy"]), _EPS)
        long_side = max(apparent_width, apparent_height)
        short_side = min(apparent_width, apparent_height)
        size_score = float(
            0.64
            * max(_bell(long_side, 0.48, 0.25), _bell(long_side, 0.68, 0.35))
            + 0.36
            * max(_bell(short_side, 0.30, 0.20), _bell(short_side, 0.48, 0.27))
        )

    height_low = None
    height_high = None
    vertical_extent = None
    height_score = 0.45
    ground_extent = None
    if scene:
        height = np.asarray(scene.get("height"))
        valid = np.asarray(scene.get("valid"))
        gu = np.asarray(scene.get("ground_u"))
        gv = np.asarray(scene.get("ground_v"))
        if height.shape == core.shape and valid.shape == core.shape:
            selected = core & valid & np.isfinite(height)
            values = height[selected]
            if values.size >= 25:
                height_low, height_high = (
                    float(np.percentile(values, percentile))
                    for percentile in (12, 88)
                )
                vertical_extent = height_high - height_low
                height_score = float(
                    max(
                        _bell(vertical_extent, 0.40, 0.28),
                        0.75 * _bell(vertical_extent, 0.62, 0.35),
                    )
                )
                if gu.shape == core.shape and gv.shape == core.shape:
                    ground_valid = (
                        selected
                        & np.isfinite(gu)
                        & np.isfinite(gv)
                    )
                    if int(np.sum(ground_valid)) >= 25:
                        uvalues = gu[ground_valid]
                        vvalues = gv[ground_valid]
                        du = float(np.percentile(uvalues, 90) - np.percentile(uvalues, 10))
                        dv = float(np.percentile(vvalues, 90) - np.percentile(vvalues, 10))
                        ground_extent = math.hypot(du, dv)

    return {
        "median_depth_m": median_depth,
        "apparent_width_m": apparent_width,
        "apparent_height_m": apparent_height,
        "size_score": size_score,
        "height_low_m": height_low,
        "height_high_m": height_high,
        "vertical_extent_m": vertical_extent,
        "height_score": height_score,
        "ground_extent_m": ground_extent,
    }


def _candidate_mask(
    region: dict[str, Any],
    seat_mask: np.ndarray,
    maps: dict[str, np.ndarray],
) -> np.ndarray:
    h, w = maps["gray"].shape
    combined = region["mask"].astype(np.uint8)
    combined[seat_mask > 0] = 1
    bbox = _bbox_from_mask(combined)
    if bbox is None:
        return combined
    roi = _expand_bbox(bbox, (h, w), 0.08, 0.12)
    x1, y1, x2, y2 = roi
    local = (
        (maps["mesh"][y1:y2, x1:x2] > 0.34)
        | (maps["dark_score"][y1:y2, x1:x2] > 0.34)
    ).astype(np.uint8)
    local = cv2.morphologyEx(
        local,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    candidate = np.zeros((h, w), np.uint8)
    candidate[y1:y2, x1:x2] = local
    candidate &= cv2.dilate(
        combined,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    candidate[combined > 0] = 1
    return candidate


def _make_geometry(
    line: dict[str, Any] | None,
    shape: tuple[int, int],
    raw_ignore: np.ndarray,
    camera: dict[str, Any],
    recognition_only: bool = False,
) -> dict[str, Any] | None:
    if line is None:
        return None
    depth = line["depth_fit"]
    length_m = depth["length_m"]
    residual = depth["residual_p90_m"]
    parallel_error = depth["parallel_error"]

    semantic_safe = (
        line["semantic_score"] >= 0.57
        and line["geometry_score"] >= 0.56
        and line["edge_support"] >= 0.34
        and line["mesh_behind"] >= 0.25
        and line["positive_fraction"] >= 0.50
        and line["wrong_side_fraction"] <= 0.34
    )
    metric_safe = (
        length_m is not None
        and 0.220 <= length_m <= 0.82
        and line["length_px"] >= 0.065 * math.hypot(*shape)
        and depth["valid_fraction"] >= 0.10
        and residual is not None
        and residual <= 0.075
    )

    # When a trustworthy floor is available, ground-parallel is a hard safety
    # condition.  A weak global fit cannot turn a semantically valid top rail
    # into a side rail, but it also cannot certify one by itself.
    if parallel_error is None:
        parallel_safe = False
    elif line["scene_quality"] >= 0.33:
        parallel_safe = parallel_error <= 0.40
    else:
        parallel_safe = parallel_error <= 0.52

    if not (semantic_safe and metric_safe and parallel_safe):
        return None

    p0 = np.asarray(line["p0"], np.float32)
    p1 = np.asarray(line["p1"], np.float32)
    # Stay away from the two backrest corner joints.  The remaining single
    # segment is the only chair grasp proposal.
    left = 0.12 * p1 + 0.88 * p0
    right = 0.88 * p1 + 0.12 * p0
    midpoint = 0.5 * (left + right)

    xyz = depth.get("endpoints_xyz")
    if xyz is not None:
        xyz0 = np.asarray(xyz[0], np.float64)
        xyz1 = np.asarray(xyz[1], np.float64)
        left_xyz = 0.88 * xyz0 + 0.12 * xyz1
        right_xyz = 0.12 * xyz0 + 0.88 * xyz1
        midpoint_xyz = 0.5 * (left_xyz + right_xyz)
        endpoints_xyz = [
            [round(float(value), 6) for value in left_xyz],
            [round(float(value), 6) for value in right_xyz],
        ]
        grasp_xyz = [round(float(value), 6) for value in midpoint_xyz]
    else:
        endpoints_xyz = None
        grasp_xyz = None

    # The entire proposed grasp segment, plus a narrow two-sided support band,
    # must keep a metric clearance from the raw robot mask. This check uses the
    # measured rail depth, so it remains valid when resolution or distance
    # changes. A line is never reconstructed across an occluding gripper.
    clearance_m = 0.0 if recognition_only else 0.025
    support_half_width_m = 0.0 if recognition_only else 0.006
    clearance_px = 0
    support_half_width_px = 0
    raw_blocked = np.asarray(raw_ignore) > 0
    if not recognition_only and np.any(raw_blocked):
        if endpoints_xyz is None:
            return None
        depth_reference = float(
            np.mean([endpoints_xyz[0][2], endpoints_xyz[1][2]])
        )
        if not np.isfinite(depth_reference) or depth_reference <= 0.0:
            return None
        focal = max(float(camera["fx"]), float(camera["fy"]))
        clearance_px = max(1, int(math.ceil(focal * clearance_m / depth_reference)))
        support_half_width_px = max(
            1,
            int(math.ceil(focal * support_half_width_m / depth_reference)),
        )
        tangent, normal, proposed_length = _line_frame(left, right)
        sample_count = max(32, int(math.ceil(proposed_length)) + 1)
        centers = _line_points(left, right, sample_count)
        offsets = np.arange(
            -support_half_width_px,
            support_half_width_px + 1,
            dtype=np.float32,
        )
        corridor = (
            centers[:, None, :]
            + offsets[None, :, None] * normal[None, None, :]
        ).reshape(-1, 2)
        free_distance = cv2.distanceTransform(
            (~raw_blocked).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        distances, valid = _sample(free_distance, corridor, default=0.0)
        if (not np.all(valid)) or np.any(distances < float(clearance_px)):
            return None

    confidence = float(
        _clip01(
            0.50
            + 0.28 * line["geometry_score"]
            + 0.12 * line["semantic_score"]
            + 0.10 * (1.0 - min(1.0, parallel_error))
        )
    )
    return {
        "type": "line",
        "name": "backrest_top_edge",
        "endpoints_uv": [_safe_uv(left, shape), _safe_uv(right, shape)],
        "grasp_point_uv": _safe_uv(midpoint, shape),
        "endpoints_xyz": endpoints_xyz,
        "grasp_point_xyz": grasp_xyz,
        "length_m": round(float(length_m) * 0.76, 6),
        "confidence": confidence,
        "grasp_safe": True,
        "corner_margin_fraction": 0.12,
        "ignore_clearance_m": clearance_m,
        "ignore_clearance_px": clearance_px,
        "support_band_half_width_px": support_half_width_px,
    }


def _object_depth_features(
    mask: np.ndarray,
    bbox: list[int],
    z: np.ndarray,
    camera: dict[str, Any],
) -> dict[str, Any]:
    values = z[(mask > 0) & np.isfinite(z)]
    all_values = z[np.isfinite(z)]
    if values.size < 24 or all_values.size < 200:
        return {
            "near_depth_m": None,
            "median_depth_m": None,
            "foreground_rank": None,
            "physical_span_m": None,
            "score": 0.38,
        }
    near = float(np.percentile(values, 24))
    median = float(np.median(values))
    foreground_rank = float(np.mean(all_values <= near))
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    physical_width = width * near / max(float(camera["fx"]), _EPS)
    physical_height = height * near / max(float(camera["fy"]), _EPS)
    physical_span = max(physical_width, physical_height)
    proximity = float(1.0 - _ramp(near, 1.55, 3.10))
    foreground = float(1.0 - _ramp(foreground_rank, 0.42, 0.82))
    size = max(
        _bell(physical_span, 0.58, 0.40),
        0.72 * _bell(physical_span, 0.95, 0.52),
    )
    return {
        "near_depth_m": near,
        "median_depth_m": median,
        "foreground_rank": foreground_rank,
        "physical_span_m": physical_span,
        "score": float(0.44 * proximity + 0.34 * foreground + 0.22 * size),
    }


def _segment_visual_descriptor(
    p0: np.ndarray,
    p1: np.ndarray,
    maps: dict[str, np.ndarray],
    ignore: np.ndarray,
) -> dict[str, Any]:
    tangent, normal, length = _line_frame(p0, p1)
    midpoint = 0.5 * (p0 + p1)
    points = _line_points(p0, p1, max(28, int(min(180.0, length))))
    gray_samples = []
    edge_samples = []
    for offset in (-3, -2, -1, 0, 1, 2, 3):
        shifted = points + float(offset) * normal
        gray, valid = _sample(maps["gray"], shifted, default=np.nan)
        edge, _ = _sample(maps["edge"], shifted, default=0.0)
        blocked, _ = _sample(ignore, shifted, default=1)
        invalid = (~valid) | (blocked > 0)
        gray[invalid] = np.nan
        edge[invalid] = 0.0
        gray_samples.append(gray)
        edge_samples.append(edge)
    gray_stack = np.stack(gray_samples)
    edge_stack = np.stack(edge_samples)
    ribbon_darkness = float(np.mean(np.nanmin(gray_stack, axis=0) < 154.0))
    # The physical top rail itself is black. Merely having a dark object a few
    # pixels to one side is insufficient (floor seams and table boundaries do
    # that frequently), so classification uses the central three samples.
    central_gray = np.nanmedian(gray_stack[2:5], axis=0)
    darkness = float(np.mean(central_gray < 116.0))
    edge_support = float(np.mean(np.max(edge_stack, axis=0) > 0.0))

    band_near = max(5.0, min(11.0, 0.028 * length))
    band_far = max(22.0, min(135.0, 0.55 * length))
    positive, positive_valid = _side_samples(
        maps["part"],
        midpoint,
        tangent,
        normal,
        length,
        1.0,
        band_near,
        band_far,
        ignore,
    )
    negative, negative_valid = _side_samples(
        maps["part"],
        midpoint,
        tangent,
        normal,
        length,
        -1.0,
        band_near,
        band_far,
        ignore,
    )
    if positive >= negative:
        back_sign = 1.0
        behind, free = positive, negative
    else:
        back_sign = -1.0
        behind, free = negative, positive

    return {
        "p0": p0,
        "p1": p1,
        "tangent": tangent,
        "normal": normal,
        "midpoint": midpoint,
        "length": length,
        "darkness": darkness,
        "ribbon_darkness": ribbon_darkness,
        "edge_support": edge_support,
        "back_sign": back_sign,
        "behind": behind,
        "free": free,
        "side_valid": min(positive_valid, negative_valid),
        "boundary_asymmetry": behind - free,
    }


def _line_intersection(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[np.ndarray, float, float] | None:
    t0 = first["tangent"].astype(np.float64)
    t1 = second["tangent"].astype(np.float64)
    matrix = np.column_stack((t0, -t1))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 0.16:
        return None
    rhs = second["p0"].astype(np.float64) - first["p0"].astype(np.float64)
    parameters = np.linalg.solve(matrix, rhs)
    point = first["p0"].astype(np.float64) + parameters[0] * t0
    return point.astype(np.float32), float(parameters[0]), float(parameters[1])


def _frame_support(
    top: dict[str, Any],
    descriptors: list[dict[str, Any]],
) -> dict[str, Any]:
    length = float(top["length"])
    midpoint = top["midpoint"]
    tangent = top["tangent"]
    inward = top["normal"] * float(top["back_sign"])
    supports: list[dict[str, Any]] = []
    for side in descriptors:
        if side is top or side["length"] < max(28.0, 0.24 * length):
            continue
        angular = abs(float(np.cross(tangent, side["tangent"])))
        if angular < 0.28:
            continue
        if side["darkness"] < 0.42 or side["edge_support"] < 0.30:
            continue
        intersection = _line_intersection(top, side)
        if intersection is None:
            continue
        point, top_parameter, side_parameter = intersection
        tangential_offset = float(np.dot(point - midpoint, tangent))
        if abs(tangential_offset) > max(1.45 * length, 185.0):
            continue
        # The side rail must actually extend into the backrest half-plane.
        side_endpoints = np.stack((side["p0"], side["p1"]))
        inward_extent = (side_endpoints - point[None, :]) @ inward
        best_extent = float(np.max(inward_extent))
        if best_extent < max(18.0, 0.16 * length):
            continue
        segment_distance = min(
            abs(side_parameter),
            abs(side_parameter - side["length"]),
            0.0
            if 0.0 <= side_parameter <= side["length"]
            else float("inf"),
        )
        if not np.isfinite(segment_distance):
            segment_distance = min(
                abs(side_parameter),
                abs(side_parameter - side["length"]),
            )
        if segment_distance > max(28.0, 0.20 * length):
            continue
        supports.append(
            {
                "line": side,
                "intersection": point,
                "tangential_offset": tangential_offset,
                "inward_extent": best_extent,
                "score": float(
                    0.36 * _ramp(side["darkness"], 0.40, 0.94)
                    + 0.28 * _ramp(side["edge_support"], 0.30, 0.88)
                    + 0.22 * _ramp(best_extent, 18.0, 0.78 * length)
                    + 0.14 * _ramp(angular, 0.28, 0.88)
                ),
            }
        )

    left = [item for item in supports if item["tangential_offset"] < 0.0]
    right = [item for item in supports if item["tangential_offset"] >= 0.0]
    left_best = max(left, key=lambda item: item["score"], default=None)
    right_best = max(right, key=lambda item: item["score"], default=None)
    chosen = [item for item in (left_best, right_best) if item is not None]
    if not chosen:
        frame_score = 0.0
        back_extent = max(24.0, 0.42 * length)
        half_width = 0.56 * length
    else:
        mean_score = float(np.mean([item["score"] for item in chosen]))
        frame_score = mean_score * (1.0 if len(chosen) == 2 else 0.62)
        back_extent = float(
            np.clip(
                np.percentile([item["inward_extent"] for item in chosen], 75),
                22.0,
                1.45 * length,
            )
        )
        half_width = max(
            0.56 * length,
            max(abs(item["tangential_offset"]) for item in chosen),
        )
    return {
        "score": float(frame_score),
        "side_count": len(chosen),
        "back_extent": back_extent,
        "half_width": float(np.clip(half_width, 0.42 * length, 1.55 * length)),
        "supports": chosen,
    }


def _metric_seat_support(
    high_part: dict[str, Any],
    scene: dict[str, Any],
    maps: dict[str, np.ndarray],
    camera: dict[str, Any],
    ignore: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Find fragmented seat evidence around a high backrest hypothesis."""

    h, w = maps["gray"].shape
    empty = np.zeros((h, w), np.uint8)
    height = np.asarray(scene["height"])
    valid = np.asarray(scene["valid"])
    gu = np.asarray(scene["ground_u"])
    gv = np.asarray(scene["ground_v"])
    center = np.asarray(high_part["center_ground"], np.float64)
    radial = np.hypot(gu - center[0], gv - center[1])

    # Colour remains a soft foreground cue.  Geometry and the two-level
    # seat/backrest relationship do the actual chair classification.
    appearance = np.maximum.reduce(
        [
            maps["dark_score"],
            0.76 * maps["local_dark"],
            0.68 * maps["part"],
            0.42 * maps["neutral"],
        ]
    )
    seat_seed = (
        valid
        & np.isfinite(height)
        & np.isfinite(radial)
        & (height >= 0.36)
        & (height <= 0.57)
        & (radial <= 0.58)
        & (appearance >= 0.16)
        & (ignore == 0)
    )
    depth_values = np.asarray(scene["z"])[seat_seed]
    depth_values = depth_values[np.isfinite(depth_values)]
    depth_reference = (
        float(np.median(depth_values))
        if depth_values.size >= 20
        else high_part.get("depth_median_m")
    )
    radius = _metric_kernel_radius(
        camera,
        depth_reference,
        0.010,
        (h, w),
        minimum=1,
        maximum_fraction=0.018,
    )
    seat_u8 = cv2.morphologyEx(
        seat_seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        ),
    )
    seat_u8[ignore > 0] = 0
    fragments = _metric_fragments(
        seat_u8,
        scene,
        ignore,
        minimum_points=max(18, int(round(0.000020 * h * w))),
    )
    fragments = _merge_metric_fragments(
        fragments,
        scene,
        ignore,
        maximum_gap_m=0.115,
        maximum_span_m=0.84,
        maximum_height_delta_m=0.095,
        minimum_component_fraction=0.020,
        minimum_component_points=60,
    )

    nearby = []
    for fragment in fragments:
        distance = float(
            np.linalg.norm(fragment["center_ground"] - center)
        )
        if distance <= 0.56:
            nearby.append((distance, fragment))
    nearby.sort(key=lambda item: (item[0], -item[1]["point_count"]))

    selected = empty.copy()
    selected_groups = 0
    for distance, fragment in nearby:
        trial = selected | fragment["mask"].astype(np.uint8)
        metric = _robust_ground_rect(trial, scene)
        if metric is None:
            continue
        if metric["long_m"] > 0.84 or metric["short_m"] > 0.72:
            continue
        # A remote tiny patch may be floor paint or a cable at the same height.
        if selected_groups and distance > 0.48 and fragment["point_count"] < 90:
            continue
        selected = trial
        selected_groups += 1
    selected[ignore > 0] = 0

    metric = _robust_ground_rect(selected, scene)
    if metric is None:
        return 0.0, empty, {
            "candidate_count": len(fragments),
            "selected_groups": 0,
            "reason": "no_metric_seat_support",
        }
    center_distance = float(
        np.linalg.norm(metric["center_ground"] - center)
    )
    if (
        metric["point_count"] < max(90, int(round(0.00010 * h * w)))
        or metric["long_m"] < 0.15
        or metric["short_m"] < 0.045
        or center_distance > 0.50
    ):
        return 0.0, empty, {
            "candidate_count": len(fragments),
            "selected_groups": selected_groups,
            "point_count": metric["point_count"],
            "long_m": round(metric["long_m"], 4),
            "short_m": round(metric["short_m"], 4),
            "center_distance_m": round(center_distance, 4),
            "reason": "seat_geometry_outside_contract",
        }

    selected_bool = selected > 0
    dark_mean = float(np.mean(maps["dark_score"][selected_bool]))
    part_mean = float(np.mean(maps["part"][selected_bool]))
    blue = float(np.mean(maps["blue"][selected_bool]))
    warm = float(np.mean(maps["warm"][selected_bool]))
    height_score = _bell(metric["height_median_m"], 0.455, 0.105)
    long_score = max(
        _bell(metric["long_m"], 0.49, 0.25),
        0.78 * _bell(metric["long_m"], 0.34, 0.19),
    )
    short_score = max(
        _bell(metric["short_m"], 0.43, 0.25),
        0.72 * _bell(metric["short_m"], 0.22, 0.16),
    )
    relation_score = float(1.0 - _ramp(center_distance, 0.24, 0.53))
    appearance_score = float(
        0.46 * _ramp(dark_mean, 0.08, 0.58)
        + 0.34 * _ramp(part_mean, 0.16, 0.56)
        + 0.20 * (1.0 - 0.58 * blue - 0.42 * warm)
    )
    score = float(
        0.28 * height_score
        + 0.21 * long_score
        + 0.15 * short_score
        + 0.19 * relation_score
        + 0.17 * appearance_score
    )
    return score, selected, {
        "candidate_count": len(fragments),
        "selected_groups": selected_groups,
        "point_count": metric["point_count"],
        "height_median_m": round(metric["height_median_m"], 4),
        "long_m": round(metric["long_m"], 4),
        "short_m": round(metric["short_m"], 4),
        "center_distance_m": round(center_distance, 4),
        "dark_mean": round(dark_mean, 4),
        "part_mean": round(part_mean, 4),
        "blue_penalty": round(blue, 4),
        "warm_penalty": round(warm, 4),
        "score": round(score, 4),
    }


def _rgb_normal_edge_metrics(
    p0: np.ndarray,
    p1: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    ignore: np.ndarray,
    radius: int,
) -> tuple[float, float, float]:
    """Median normal Sobel response and along-line coverage."""

    tangent, normal, length = _line_frame(p0, p1)
    if length <= _EPS:
        return 0.0, 0.0, 0.0
    points = _line_points(p0, p1, max(36, int(min(260.0, length))))
    responses = []
    valid_columns = np.zeros(len(points), bool)
    for offset in range(-radius, radius + 1):
        shifted = points + float(offset) * normal
        gx, valid = _sample(grad_x, shifted, default=np.nan)
        gy, _ = _sample(grad_y, shifted, default=np.nan)
        blocked, _ = _sample(ignore, shifted, default=1.0)
        response = np.abs(gx * normal[0] + gy * normal[1])
        response[(~valid) | (blocked > 0)] = np.nan
        responses.append(response)
        valid_columns |= np.isfinite(response)
    stack = np.stack(responses)
    with np.errstate(all="ignore"):
        strongest = np.nanmax(stack, axis=0)
    strongest = strongest[np.isfinite(strongest)]
    if strongest.size < 12:
        return 0.0, 0.0, float(np.mean(valid_columns))
    median = float(np.median(strongest))
    coverage = float(np.mean(strongest >= 34.0))
    score = float(
        0.58 * _ramp(median, 18.0, 165.0)
        + 0.42 * _ramp(coverage, 0.22, 0.88)
    )
    return score, median, coverage


def _height_line_metrics(
    p0: np.ndarray,
    p1: np.ndarray,
    support_mask: np.ndarray,
    scene: dict[str, Any],
    ignore: np.ndarray,
    radius: int,
) -> dict[str, Any] | None:
    """Sample the highest locally supported scene height along a 2-D line."""

    _, normal, length = _line_frame(p0, p1)
    count = max(36, int(min(240.0, math.ceil(length))))
    centers = _line_points(p0, p1, count)
    profiles = []
    for offset in range(-radius, radius + 1):
        points = centers + float(offset) * normal
        values, valid = _sample(scene["height"], points, default=np.nan)
        support, _ = _sample(
            (np.asarray(support_mask) > 0).astype(np.float32),
            points,
            default=0.0,
        )
        blocked, _ = _sample(ignore, points, default=1.0)
        values[(~valid) | (support <= 0.0) | (blocked > 0)] = np.nan
        profiles.append(values)
    stack = np.stack(profiles, axis=1)
    sampled = []
    for row in stack:
        values = row[np.isfinite(row)]
        sampled.append(
            np.nan if not len(values) else float(np.percentile(values, 74))
        )
    sampled = np.asarray(sampled, np.float64)
    valid = np.isfinite(sampled)
    if int(np.sum(valid)) < max(10, int(round(0.18 * count))):
        return None
    values = sampled[valid]
    low, high = np.percentile(values, (10, 90))
    return {
        "valid_fraction": float(np.mean(valid)),
        "height_median_m": float(np.median(values)),
        "height_span_m": float(high - low),
        "height_p10_m": float(low),
        "height_p90_m": float(high),
    }


def _highest_outer_line(
    high_part: dict[str, Any],
    scene: dict[str, Any],
    maps: dict[str, np.ndarray],
    z: np.ndarray,
    camera: dict[str, Any],
    ignore: np.ndarray,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Fit only the physical, highest outer contour of a metric backrest."""

    h, w = maps["gray"].shape
    high_mask = (np.asarray(high_part["mask"]) > 0).astype(np.uint8)
    high_mask[ignore > 0] = 0
    depth_reference = high_part.get("depth_median_m")
    close_radius = _metric_kernel_radius(
        camera,
        depth_reference,
        0.006,
        (h, w),
        minimum=1,
        maximum_fraction=0.012,
    )
    contour_mask = cv2.morphologyEx(
        high_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * close_radius + 1, 2 * close_radius + 1),
        ),
    )
    contour_mask[ignore > 0] = 0
    contours, _ = cv2.findContours(
        contour_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None, {"reason": "no_outer_contour"}

    valid_high = (
        (contour_mask > 0)
        & np.asarray(scene["valid"])
        & np.isfinite(scene["height"])
    )
    height_values = np.asarray(scene["height"])[valid_high]
    if height_values.size < 30:
        return None, {"reason": "insufficient_height_support"}
    peak_height = float(np.percentile(height_values, 95))
    ridge = (
        valid_high
        & (np.asarray(scene["height"]) >= max(0.68, peak_height - 0.105))
    )
    ridge_u8 = ridge.astype(np.uint8)
    ridge_u8[ignore > 0] = 0
    ridge_contours, _ = cv2.findContours(
        ridge_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    edge = np.zeros((h, w), np.uint8)
    cv2.drawContours(edge, contours, -1, 255, 2)
    cv2.drawContours(edge, ridge_contours, -1, 255, 1)
    neighborhood_radius = _metric_kernel_radius(
        camera,
        depth_reference,
        0.008,
        (h, w),
        minimum=2,
        maximum_fraction=0.016,
    )
    high_neighborhood = cv2.dilate(
        ridge_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * neighborhood_radius + 1, 2 * neighborhood_radius + 1),
        ),
    )
    edge[high_neighborhood == 0] = 0
    edge[ignore > 0] = 0

    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    if depth_reference is None or not np.isfinite(depth_reference):
        minimum_length = 0.045 * math.hypot(h, w)
        maximum_gap = 0.018 * math.hypot(h, w)
    else:
        minimum_length = focal * 0.085 / max(depth_reference, 0.10)
        maximum_gap = focal * 0.030 / max(depth_reference, 0.10)
    minimum_length = int(
        np.clip(minimum_length, 28, 0.42 * math.hypot(h, w))
    )
    maximum_gap = int(np.clip(maximum_gap, 8, 0.08 * math.hypot(h, w)))
    hough = cv2.HoughLinesP(
        edge,
        1.0,
        np.pi / 720.0,
        threshold=max(16, int(round(0.34 * minimum_length))),
        minLineLength=minimum_length,
        maxLineGap=maximum_gap,
    )
    if hough is None:
        return None, {
            "reason": "no_outer_hough_line",
            "peak_height_m": round(peak_height, 4),
        }

    grad_x = cv2.Sobel(maps["gray"], cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(maps["gray"], cv2.CV_32F, 0, 1, ksize=3)
    grad_x[ignore > 0] = 0.0
    grad_y[ignore > 0] = 0.0
    edge_distance = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 3)
    plane = np.asarray(scene["plane"], np.float64)
    component_y, component_x = np.nonzero(contour_mask)
    component_points = np.column_stack((component_x, component_y)).astype(np.float32)
    candidates: list[dict[str, Any]] = []

    for raw in hough.reshape(-1, 4):
        p0 = raw[:2].astype(np.float32)
        p1 = raw[2:].astype(np.float32)
        tangent, normal, length_px = _line_frame(p0, p1)
        if length_px < minimum_length:
            continue
        probe = _line_points(p0, p1, max(36, int(math.ceil(length_px))))
        blocked, valid_probe = _sample(
            (ignore > 0).astype(np.float32),
            probe,
            default=1.0,
        )
        if np.any((blocked > 0.0) | (~valid_probe)):
            continue

        height_line = _height_line_metrics(
            p0,
            p1,
            contour_mask,
            scene,
            ignore,
            max(2, close_radius),
        )
        if height_line is None:
            continue
        if (
            height_line["height_median_m"] < peak_height - 0.032
            or height_line["height_span_m"] > 0.034
        ):
            continue

        depth_fit = _ribbon_depth(p0, p1, z, camera, plane, ignore)
        length_m = depth_fit["length_m"]
        residual = depth_fit["residual_p90_m"]
        parallel_error = depth_fit["parallel_error"]
        if (
            length_m is None
            or not 0.10 <= length_m <= 0.72
            or residual is None
            or residual > 0.052
            or parallel_error is None
            or parallel_error > 0.22
            or depth_fit["valid_fraction"] < 0.10
        ):
            continue

        depth_median = (
            float(depth_reference)
            if depth_reference is not None
            else float(np.nanmedian(z))
        )
        apparent_length = math.hypot(
            float((p1[0] - p0[0]) * depth_median / camera["fx"]),
            float((p1[1] - p0[1]) * depth_median / camera["fy"]),
        )
        if length_m > max(0.20, 2.8 * apparent_length):
            continue

        distances, _ = _sample(edge_distance, probe, default=99.0)
        contour_tolerance = max(2.0, 0.006 * length_px)
        contour_coverage = float(np.mean(distances <= contour_tolerance))
        rgb_score, rgb_strength, rgb_coverage = _rgb_normal_edge_metrics(
            p0,
            p1,
            grad_x,
            grad_y,
            ignore,
            radius=max(2, min(5, close_radius)),
        )
        if (
            contour_coverage < 0.42
            or rgb_coverage < 0.24
            or rgb_strength < 16.0
        ):
            continue

        midpoint = 0.5 * (p0 + p1)
        signed = (component_points - midpoint[None, :]) @ normal
        along = np.abs((component_points - midpoint[None, :]) @ tangent)
        local = signed[along <= 0.70 * length_px]
        if local.size < 35:
            continue
        positive = float(np.mean(local > 1.5))
        negative = float(np.mean(local < -1.5))
        if negative > positive:
            normal = -normal
            positive, negative = negative, positive
        center_fraction = float(np.mean(np.abs(local) <= 1.5))
        if positive < 0.48 or negative > 0.38:
            continue
        boundary_asymmetry = float(
            _clip01((positive - negative + 0.04) / 0.72)
        )

        top_score = float(
            math.exp(
                -(
                    max(
                        0.0,
                        peak_height - height_line["height_median_m"],
                    )
                    / 0.030
                )
                ** 2
            )
        )
        length_score = max(
            _bell(length_m, 0.42, 0.22),
            0.82 * _bell(length_m, 0.25, 0.14),
        )
        straight_score = float(math.exp(-((residual / 0.030) ** 2)))
        level_score = float(math.exp(-((parallel_error / 0.13) ** 2)))
        valid_score = float(_ramp(depth_fit["valid_fraction"], 0.10, 0.72))
        edge_support = float(max(contour_coverage, rgb_coverage))
        semantic_score = float(
            0.21 * top_score
            + 0.18 * _ramp(contour_coverage, 0.42, 0.93)
            + 0.18 * rgb_score
            + 0.17 * boundary_asymmetry
            + 0.14 * _ramp(positive, 0.48, 0.92)
            + 0.12 * (1.0 - _ramp(center_fraction, 0.18, 0.62))
        )
        geometry_score = float(
            0.22 * top_score
            + 0.18 * length_score
            + 0.22 * straight_score
            + 0.22 * level_score
            + 0.16 * valid_score
        )
        rank = float(
            length_m
            * (0.35 + 0.65 * semantic_score)
            * (0.30 + 0.70 * geometry_score)
            * (0.45 + 0.55 * top_score)
        )
        candidates.append(
            {
                "p0": p0,
                "p1": p1,
                "length_px": length_px,
                "length_m": length_m,
                "back_sign": 1.0,
                "back_extent_px": float(
                    np.percentile(local[local > 0], 90)
                    if np.any(local > 0)
                    else 0.0
                ),
                "mesh_behind": positive,
                "mesh_free": negative,
                "along_dark": float(
                    np.mean(maps["dark_score"][contour_mask > 0])
                ),
                "edge_support": edge_support,
                "positive_fraction": positive,
                "wrong_side_fraction": negative,
                "span_ratio": 1.0,
                "semantic_score": semantic_score,
                "geometry_score": geometry_score,
                "depth_fit": depth_fit,
                "side_valid": 1.0,
                "scene_quality": _scene_reliability(scene),
                "rank": rank,
                "debug": {
                    "height": height_line,
                    "peak_height_m": peak_height,
                    "contour_coverage": contour_coverage,
                    "rgb_normal_score": rgb_score,
                    "rgb_normal_strength": rgb_strength,
                    "rgb_normal_coverage": rgb_coverage,
                    "boundary_asymmetry": boundary_asymmetry,
                    "apparent_length_m": apparent_length,
                },
            }
        )

    candidates.sort(
        key=lambda item: (
            item["rank"],
            item["geometry_score"],
            item["semantic_score"],
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    return best, {
        "peak_height_m": round(peak_height, 4),
        "raw_hough_count": int(len(hough)),
        "accepted_line_count": len(candidates),
        "best": None if best is None else {
            "endpoints_uv": [
                _safe_uv(best["p0"], (h, w)),
                _safe_uv(best["p1"], (h, w)),
            ],
            "length_px": round(float(best["length_px"]), 2),
            "length_m": round(float(best["length_m"]), 5),
            "semantic_score": round(float(best["semantic_score"]), 4),
            "geometry_score": round(float(best["geometry_score"]), 4),
            "parallel_error": round(
                float(best["depth_fit"]["parallel_error"]), 5
            ),
            "residual_p90_m": round(
                float(best["depth_fit"]["residual_p90_m"]), 5
            ),
            **{
                key: (
                    round(float(value), 5)
                    if isinstance(value, (float, np.floating))
                    else value
                )
                for key, value in best["debug"].items()
                if key != "height"
            },
            "height": {
                key: round(float(value), 5)
                for key, value in best["debug"]["height"].items()
            },
        },
    }


def _height_first_candidates(
    rgb: np.ndarray,
    z: np.ndarray,
    camera: dict[str, Any],
    ignore: np.ndarray,
    scene: dict[str, Any] | None,
    maps: dict[str, np.ndarray],
    raw_ignore: np.ndarray,
    recognition_only: bool = False,
) -> list[dict[str, Any]]:
    """Detect a chair from high backrest + lower seat metric topology."""

    if not _trusted_ground_scene(scene):
        return []
    assert scene is not None
    h, w = z.shape
    height = np.asarray(scene["height"])
    valid = np.asarray(scene["valid"])
    core = (
        valid
        & np.isfinite(height)
        & (height >= 0.70)
        & (height <= 1.02)
        & (ignore == 0)
    )
    depth_values = z[core & np.isfinite(z)]
    if depth_values.size < 30:
        return []
    depth_reference = float(np.median(depth_values))
    radius = _metric_kernel_radius(
        camera,
        depth_reference,
        0.006,
        (h, w),
        minimum=1,
        maximum_fraction=0.012,
    )
    core_u8 = cv2.morphologyEx(
        core.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        ),
    )
    core_u8[ignore > 0] = 0
    fragments = _metric_fragments(
        core_u8,
        scene,
        ignore,
        minimum_points=max(20, int(round(0.000022 * h * w))),
    )
    high_parts = _merge_metric_fragments(
        fragments,
        scene,
        ignore,
        maximum_gap_m=0.085,
        maximum_span_m=0.82,
        maximum_height_delta_m=0.12,
        minimum_component_fraction=0.020,
        minimum_component_points=70,
    )
    candidates: list[dict[str, Any]] = []
    for high_part in high_parts[:18]:
        if (
            high_part["point_count"] < max(80, int(round(0.000085 * h * w)))
            or not 0.11 <= high_part["long_m"] <= 0.82
            or high_part["short_m"] > 0.68
            or not 0.70 <= high_part["height_p95_m"] <= 1.06
            or high_part["height_median_m"] < 0.675
        ):
            continue

        high_mask = high_part["mask"] > 0
        dark_mean = float(np.mean(maps["dark_score"][high_mask]))
        part_mean = float(np.mean(maps["part"][high_mask]))
        edge_mean = float(np.mean(maps["edge_density"][high_mask]))
        neutral_mean = float(np.mean(maps["neutral"][high_mask]))
        blue = float(np.mean(maps["blue"][high_mask]))
        warm = float(np.mean(maps["warm"][high_mask]))
        seat_score, seat_mask, seat_debug = _metric_seat_support(
            high_part,
            scene,
            maps,
            camera,
            ignore,
        )
        # The rule-book chair has an approximately 0.55 m wheeled footprint.
        # Allow severe occlusion, but a long shelf paired with a ~10 cm narrow
        # shelf strip is not a physical chair seat.
        if (
            seat_score < 0.34
            or not np.any(seat_mask)
            or float(seat_debug.get("short_m", 0.0)) < 0.18
        ):
            continue

        line, line_debug = _highest_outer_line(
            high_part,
            scene,
            maps,
            z,
            camera,
            ignore,
        )
        geometry = _make_geometry(
            line,
            (h, w),
            raw_ignore,
            camera,
            recognition_only=recognition_only,
        )
        line_quality = (
            0.44
            if line is None
            else 0.50
            * (
                float(line["semantic_score"])
                + float(line["geometry_score"])
            )
        )
        height_score = _bell(high_part["height_p95_m"], 0.86, 0.16)
        high_size_score = max(
            _bell(high_part["long_m"], 0.50, 0.25),
            0.76 * _bell(high_part["long_m"], 0.29, 0.17),
        )
        appearance_score = float(
            0.29 * _ramp(dark_mean, 0.05, 0.62)
            + 0.24 * _ramp(part_mean, 0.13, 0.58)
            + 0.19 * _ramp(edge_mean, 0.015, 0.17)
            + 0.13 * _ramp(neutral_mean, 0.24, 0.92)
            + 0.15
        )
        # A high dark strip over a warm shelf/cabinet patch can mimic the
        # backrest/seat height pair.  Apply the colour veto to the proposed
        # seat as well as the high part; real chair candidates remain neutral,
        # while background wooden furniture is down-weighted.
        seat_warm = float(seat_debug.get("warm_penalty", 0.0))
        colour_factor = float(
            _clip01(1.03 - 0.30 * blue - 0.22 * warm - 0.65 * seat_warm)
        )
        confidence = float(
            colour_factor
            * (
                0.49
                + 0.13 * height_score
                + 0.10 * high_size_score
                + 0.13 * seat_score
                + 0.08 * appearance_score
                + 0.07 * line_quality
            )
        )
        if confidence < 0.755:
            continue

        candidate_mask = high_part["mask"].astype(np.uint8)
        candidate_mask[seat_mask > 0] = 1
        candidate_mask[ignore > 0] = 0
        bbox = _bbox_from_mask(candidate_mask)
        if bbox is None:
            continue
        accepted_height, height_debug = _passes_scene_chair_height(
            candidate_mask,
            scene,
        )
        if not accepted_height:
            continue
        candidates.append(
            {
                "class": "chair",
                "mask": candidate_mask,
                "bbox": bbox,
                "confidence": float(_clip01(confidence)),
                "geometry": geometry,
                "grasp_geometry": geometry,
                "debug": {
                    "mode": "metric_high_backrest_seat",
                    "high_part": {
                        "point_count": high_part["point_count"],
                        "bbox": high_part["bbox"],
                        "height_median_m": round(
                            high_part["height_median_m"], 4
                        ),
                        "height_p95_m": round(
                            high_part["height_p95_m"], 4
                        ),
                        "long_m": round(high_part["long_m"], 4),
                        "short_m": round(high_part["short_m"], 4),
                        "dark_mean": round(dark_mean, 4),
                        "part_mean": round(part_mean, 4),
                        "edge_mean": round(edge_mean, 4),
                        "neutral_mean": round(neutral_mean, 4),
                    },
                    "seat": seat_debug,
                    "line": line_debug,
                    "height_gate": {
                        key: (
                            None
                            if value is None
                            else (
                                round(float(value), 5)
                                if isinstance(
                                    value,
                                    (float, np.floating),
                                )
                                else value
                            )
                        )
                        for key, value in height_debug.items()
                    },
                    "scores": {
                        "height": round(height_score, 4),
                        "high_size": round(high_size_score, 4),
                        "seat": round(seat_score, 4),
                        "appearance": round(appearance_score, 4),
                        "line": round(line_quality, 4),
                        "colour_factor": round(colour_factor, 4),
                    },
                    "ignore_safety": {
                        "clearance_m": (
                            0.0 if recognition_only else 0.025
                        ),
                        "raw_ignore_area_px": int(np.sum(raw_ignore > 0)),
                        "expanded_ignore_area_px": int(np.sum(ignore > 0)),
                    },
                },
            }
        )
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates[:3]


def _line_driven_candidates(
    rgb: np.ndarray,
    z: np.ndarray,
    camera: dict[str, Any],
    ignore: np.ndarray,
    scene: dict[str, Any] | None,
    maps: dict[str, np.ndarray],
    segments: list[tuple[np.ndarray, np.ndarray]],
    raw_ignore: np.ndarray,
    recognition_only: bool = False,
) -> list[dict[str, Any]]:
    h, w = rgb.shape[:2]
    descriptors = [
        _segment_visual_descriptor(p0, p1, maps, ignore)
        for p0, p1 in segments
    ]
    descriptors = [
        item
        for item in descriptors
        if item["length"] >= 30.0
        and item["darkness"] >= 0.36
        and item["edge_support"] >= 0.28
        and item["behind"] >= 0.28
        and item["side_valid"] >= 0.50
    ]
    descriptors.sort(
        key=lambda item: (
            0.36 * item["darkness"]
            + 0.30 * item["behind"]
            + 0.20 * _ramp(item["boundary_asymmetry"], 0.01, 0.24)
            + 0.14 * _ramp(item["length"], 35.0, 360.0)
        ),
        reverse=True,
    )
    descriptors = descriptors[:120]

    plane = None
    if scene and scene.get("plane") is not None:
        candidate_plane = np.asarray(scene["plane"], np.float64).reshape(-1)
        if candidate_plane.shape == (4,) and np.all(np.isfinite(candidate_plane)):
            plane = candidate_plane
    scene_quality = _scene_reliability(scene)
    candidates: list[dict[str, Any]] = []

    for top in descriptors[:72]:
        if top["boundary_asymmetry"] < 0.018:
            continue
        frame = _frame_support(top, descriptors)
        if frame["side_count"] == 0:
            continue

        inward = top["normal"] * float(top["back_sign"])
        back_roi = _polygon_mask(
            (h, w),
            top["midpoint"],
            top["tangent"],
            inward,
            frame["half_width"],
            4.0,
            frame["back_extent"],
        )
        usable = (back_roi > 0) & (ignore == 0)
        if int(np.sum(usable)) < 180:
            continue
        mesh_mean = float(np.mean(maps["mesh"][usable]))
        crossed = float(np.mean(maps["crossed_diagonal"][usable]))
        dark_mean = float(np.mean(maps["dark_score"][usable]))
        blue = float(np.mean(maps["blue"][usable]))
        warm = float(np.mean(maps["warm"][usable]))

        core = usable & (
            (maps["mesh"] > 0.38)
            | (maps["local_dark"] > 0.15)
            | (maps["dark_score"] > 0.38)
        )
        if int(np.sum(core)) < 120:
            continue
        region = {
            "mask": usable,
            "core": core,
            "bbox": _bbox_from_mask(back_roi) or [0, 0, w, h],
            "quality": mesh_mean,
            "edge_density": float(np.mean(maps["edge_density"][usable])),
            "thin_density": float(np.mean(maps["thin_dark_density"][usable])),
        }
        line_for_seat = {
            "p0": top["p0"],
            "p1": top["p1"],
            "back_sign": top["back_sign"],
            "back_extent_px": frame["back_extent"],
        }
        seat_score, seat_mask, seat_debug = _seat_relation(
            region,
            line_for_seat,
            maps,
            scene,
            ignore,
        )
        candidate_mask = back_roi.astype(np.uint8)
        candidate_mask[seat_mask > 0] = 1
        candidate_mask[ignore > 0] = 0
        candidate_bbox = _bbox_from_mask(candidate_mask)
        if candidate_bbox is None:
            continue
        accepted_height, scene_height_debug = _passes_scene_chair_height(
            candidate_mask,
            scene,
        )
        if not accepted_height:
            continue
        object_depth = _object_depth_features(
            candidate_mask,
            candidate_bbox,
            z,
            camera,
        )
        if (
            object_depth["near_depth_m"] is not None
            and object_depth["near_depth_m"] > 3.10
        ):
            continue

        depth_fit = _ribbon_depth(
            top["p0"], top["p1"], z, camera, plane, ignore
        )
        if depth_fit["length_m"] is None:
            physical_score = float(_ramp(top["length"], 34.0, 260.0))
        else:
            physical_score = max(
                _bell(depth_fit["length_m"], 0.42, 0.23),
                0.76 * _bell(depth_fit["length_m"], 0.22, 0.14),
            )
        if depth_fit["parallel_error"] is None:
            parallel_score = 0.35
        else:
            parallel_score = float(
                1.0 - _ramp(depth_fit["parallel_error"], 0.12, 0.46)
            )

        back_score = float(
            0.34 * _ramp(mesh_mean, 0.30, 0.68)
            + 0.22 * _ramp(crossed, 0.018, 0.105)
            + 0.20 * _ramp(dark_mean, 0.05, 0.42)
            + 0.24 * _ramp(top["behind"], 0.28, 0.72)
        )
        top_semantic = float(
            0.18 * _ramp(top["darkness"], 0.38, 0.96)
            + 0.15 * _ramp(top["edge_support"], 0.30, 0.90)
            + 0.17 * _ramp(top["boundary_asymmetry"], 0.018, 0.30)
            + 0.18 * back_score
            + 0.19 * frame["score"]
            + 0.13 * max(seat_score, 0.55 * frame["score"])
        )
        physical_span = object_depth["physical_span_m"]
        if (
            top_semantic < 0.805
            or (physical_span is not None and physical_span < 0.145)
        ):
            continue
        line_geometry_score = float(
            0.57 * top_semantic
            + 0.13 * physical_score
            + 0.11 * parallel_score
            + 0.10 * _ramp(depth_fit["valid_fraction"], 0.08, 0.58)
            + 0.09
            * (
                1.0
                - _ramp(
                    depth_fit["residual_p90_m"]
                    if depth_fit["residual_p90_m"] is not None
                    else 0.075,
                    0.018,
                    0.075,
                )
            )
        )
        line_geometry_score *= float(
            1.0 - 0.30 * scene_quality * (1.0 - parallel_score)
        )

        colour_factor = float(
            _clip01(1.02 - 0.31 * blue - 0.24 * warm)
        )
        object_confidence = float(
            colour_factor
            * (
                0.24 * back_score
                + 0.21 * top_semantic
                + 0.18 * frame["score"]
                + 0.14 * seat_score
                + 0.07 * physical_score
                + 0.06 * max(parallel_score, 0.35)
                + 0.10 * object_depth["score"]
            )
        )
        if (
            object_depth["near_depth_m"] is not None
            and physical_span is not None
            and physical_span < 0.20
            and (
                depth_fit["length_m"] is None
                or depth_fit["length_m"] < 0.12
            )
            and object_confidence < 0.825
        ):
            continue

        structural_gate = (
            frame["side_count"] >= 2
            and frame["score"] >= 0.43
            and back_score >= 0.42
            and top_semantic >= 0.49
        ) or (
            frame["side_count"] >= 1
            and frame["score"] >= 0.32
            and back_score >= 0.48
            and seat_score >= 0.35
            and top_semantic >= 0.52
        )
        depth_available = int(np.sum(np.isfinite(z))) >= 500
        confidence_floor = 0.785 if depth_available else 0.700
        if not structural_gate or object_confidence < confidence_floor:
            continue

        core_y, core_x = np.nonzero(core)
        core_points = np.column_stack((core_x, core_y)).astype(np.float32)
        signed = (
            (core_points - top["midpoint"][None, :])
            @ top["normal"]
            * float(top["back_sign"])
        )
        along = np.abs(
            (core_points - top["midpoint"][None, :]) @ top["tangent"]
        )
        local_support = signed[along <= 0.68 * float(top["length"])]
        if local_support.size:
            positive_fraction = float(np.mean(local_support > 1.5))
            wrong_side_fraction = float(np.mean(local_support < -4.0))
        else:
            positive_fraction = 0.0
            wrong_side_fraction = 1.0

        line = {
            "p0": top["p0"],
            "p1": top["p1"],
            "length_px": top["length"],
            "length_m": depth_fit["length_m"],
            "back_sign": top["back_sign"],
            "back_extent_px": frame["back_extent"],
            "mesh_behind": top["behind"],
            "mesh_free": top["free"],
            "along_dark": top["darkness"],
            "edge_support": top["edge_support"],
            "positive_fraction": positive_fraction,
            "wrong_side_fraction": wrong_side_fraction,
            "span_ratio": 1.0,
            "semantic_score": top_semantic,
            "geometry_score": line_geometry_score,
            "depth_fit": depth_fit,
            "side_valid": top["side_valid"],
            "scene_quality": scene_quality,
        }
        geometry = _make_geometry(
            line,
            (h, w),
            raw_ignore,
            camera,
            recognition_only=recognition_only,
        )

        mask = candidate_mask
        bbox = candidate_bbox
        candidates.append(
            {
                "class": "chair",
                "mask": mask,
                "bbox": bbox,
                "confidence": float(_clip01(object_confidence)),
                "geometry": geometry,
                "grasp_geometry": geometry,
                "debug": {
                    "mode": "line_frame_parts",
                    "top_semantic_score": round(top_semantic, 4),
                    "line_geometry_score": round(line_geometry_score, 4),
                    "back_score": round(back_score, 4),
                    "mesh_mean": round(mesh_mean, 4),
                    "crossed_diagonal": round(crossed, 4),
                    "dark_mean": round(dark_mean, 4),
                    "frame_score": round(float(frame["score"]), 4),
                    "frame_side_count": int(frame["side_count"]),
                    "seat": seat_debug,
                    "blue_penalty": round(blue, 4),
                    "warm_penalty": round(warm, 4),
                    "scene_quality": round(scene_quality, 4),
                    "height_gate": {
                        key: (
                            None
                            if value is None
                            else (
                                round(float(value), 5)
                                if isinstance(value, (float, np.floating))
                                else value
                            )
                        )
                        for key, value in scene_height_debug.items()
                    },
                    "object_depth": {
                        key: None if value is None else round(float(value), 5)
                        for key, value in object_depth.items()
                    },
                    "best_line": {
                        "endpoints_uv": [
                            _safe_uv(top["p0"], (h, w)),
                            _safe_uv(top["p1"], (h, w)),
                        ],
                        "length_px": round(float(top["length"]), 2),
                        "length_m": (
                            None
                            if depth_fit["length_m"] is None
                            else round(float(depth_fit["length_m"]), 5)
                        ),
                        "boundary_asymmetry": round(
                            float(top["boundary_asymmetry"]), 4
                        ),
                        "parallel_error": depth_fit["parallel_error"],
                        "depth_valid_fraction": round(
                            float(depth_fit["valid_fraction"]), 4
                        ),
                        "residual_p90_m": depth_fit["residual_p90_m"],
                    },
                },
            }
        )

    candidates.extend(
        _dark_foreground_fallback(rgb, z, camera, ignore, scene, maps)
    )
    return _deduplicate(candidates)


def _dark_foreground_fallback(
    rgb: np.ndarray,
    z: np.ndarray,
    camera: dict[str, Any],
    ignore: np.ndarray,
    scene: dict[str, Any] | None,
    maps: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Recover strongly oblique chairs whose top rail projects edge-on.

    This branch can identify the object but never publishes a grasp.  It uses a
    foreground-connected dark seat/back silhouette and deliberately strict
    physical/shape checks so a dark floor seam or cabinet cannot substitute for
    the two chair parts.
    """

    valid = np.isfinite(z)
    if int(np.sum(valid)) < 800:
        return []
    h, w = z.shape
    near_limit = float(np.percentile(z[valid], 42))
    dark = maps["dark_score"] > 0.30
    seed = dark & valid & (z <= near_limit) & (ignore == 0)
    seed = cv2.morphologyEx(
        seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    seed[ignore > 0] = 0
    grown = seed
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(8):
        grown = cv2.dilate(grown, kernel) & dark.astype(np.uint8)
        grown[ignore > 0] = 0
    grown = cv2.morphologyEx(
        grown,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    grown[ignore > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, 8)
    minimum_area = max(4200, int(0.0042 * h * w))
    output: list[dict[str, Any]] = []
    hsv = maps["hsv"]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    for index in range(1, count):
        x, y, width, height, area = (
            int(item) for item in stats[index]
        )
        if area < minimum_area or width < 120 or height < 150:
            continue
        fill = area / max(1.0, float(width * height))
        if not 0.18 <= fill <= 0.58:
            continue
        component = labels == index
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = float(cv2.contourArea(contour) / max(hull_area, 1.0))
        if not 0.38 <= solidity <= 0.76:
            continue

        mean_saturation = float(np.mean(saturation[component]))
        mean_value = float(np.mean(value[component]))
        neutral_score = float(1.0 - _ramp(mean_saturation, 54.0, 82.0))
        if neutral_score < 0.36:
            continue
        bbox = [x, y, x + width, y + height]
        depth = _object_depth_features(
            component.astype(np.uint8),
            bbox,
            z,
            camera,
        )
        near_depth = depth["near_depth_m"]
        if near_depth is None or near_depth > 2.35:
            continue
        physical_width = width * near_depth / max(float(camera["fx"]), _EPS)
        physical_height = height * near_depth / max(float(camera["fy"]), _EPS)
        physical_long = max(physical_width, physical_height)
        physical_short = min(physical_width, physical_height)
        size_score = float(
            0.60 * max(
                _bell(physical_long, 0.66, 0.34),
                0.75 * _bell(physical_long, 0.92, 0.45),
            )
            + 0.40 * max(
                _bell(physical_short, 0.36, 0.23),
                0.72 * _bell(physical_short, 0.55, 0.30),
            )
        )
        if size_score < 0.38:
            continue

        mesh_support = float(np.mean(maps["mesh"][component]))
        edge_support = float(np.mean(maps["edge_density"][component]))
        if (
            mean_value > 92.0
            or solidity < 0.42
            or mesh_support < 0.19
            or edge_support < 0.07
            or size_score < 0.56
        ):
            continue
        concavity_score = float(1.0 - _ramp(solidity, 0.58, 0.82))
        fill_score = max(_bell(fill, 0.29, 0.11), 0.72 * _bell(fill, 0.40, 0.12))
        confidence = float(
            0.54
            + 0.10 * neutral_score
            + 0.08 * _ramp(105.0 - mean_value, 22.0, 68.0)
            + 0.07 * _ramp(mesh_support, 0.36, 0.68)
            + 0.05 * _ramp(edge_support, 0.04, 0.16)
            + 0.06 * concavity_score
            + 0.05 * fill_score
            + 0.05 * size_score
        )
        if confidence < 0.785:
            continue

        mask = component.astype(np.uint8)
        accepted_height, scene_height_debug = _passes_scene_chair_height(
            mask,
            scene,
        )
        if not accepted_height:
            continue
        output.append(
            {
                "class": "chair",
                "mask": mask,
                "bbox": bbox,
                "confidence": float(_clip01(confidence)),
                "geometry": None,
                "grasp_geometry": None,
                "debug": {
                    "mode": "dark_foreground_seat_back_fallback",
                    "area_px": area,
                    "fill_ratio": round(fill, 4),
                    "solidity": round(solidity, 4),
                    "neutral_score": round(neutral_score, 4),
                    "mean_saturation": round(mean_saturation, 3),
                    "mean_value": round(mean_value, 3),
                    "mesh_support": round(mesh_support, 4),
                    "edge_support": round(edge_support, 4),
                    "size_score": round(size_score, 4),
                    "height_gate": {
                        key: (
                            None
                            if value is None
                            else (
                                round(float(value), 5)
                                if isinstance(value, (float, np.floating))
                                else value
                            )
                        )
                        for key, value in scene_height_debug.items()
                    },
                    "object_depth": {
                        key: None if item is None else round(float(item), 5)
                        for key, item in depth.items()
                    },
                    "reason_no_grasp": "top_edge_not_metric_safe",
                },
            }
        )
    return output


def _deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        x1, y1, x2, y2 = candidate["bbox"]
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for other in retained:
            a1, b1, a2, b2 = other["bbox"]
            intersection = max(0, min(x2, a2) - max(x1, a1)) * max(
                0, min(y2, b2) - max(y1, b1)
            )
            other_area = max(1, (a2 - a1) * (b2 - b1))
            area_ratio = max(area, other_area) / max(1, min(area, other_area))
            # A tiny high-rail seed inside a complete seat/back silhouette is
            # complementary evidence, not a duplicate object.  Retaining both
            # lets cross-evidence arbitration prefer the physically complete
            # chair when neither candidate has a certified grasp line.
            if (
                intersection / min(area, other_area) > 0.48
                and area_ratio <= 8.0
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return retained[:4]


def detect_chair_candidates(
    rgb: np.ndarray,
    z: np.ndarray,
    camera: dict[str, Any],
    ignore: np.ndarray,
    scene: dict[str, Any] | None = None,
    recognition_only: bool = True,
) -> list[dict[str, Any]]:
    """Return geometry-first chair hypotheses.

    Parameters
    ----------
    rgb:
        OpenCV BGR ``uint8`` image.
    z:
        Aligned depth in metres. Invalid pixels must be ``NaN``.
    camera:
        Mapping containing at least ``fx/fy/cx/cy``.
    ignore:
        Non-zero pixels belong to the robot/gripper and are excluded.
    scene:
        Optional output of :func:`wrist_rgbd.scene.build_scene_geometry`.

    Returns
    -------
    list[dict]
        Every item contains ``mask``, ``bbox``, ``confidence``, ``geometry`` and
        ``debug``. ``geometry`` is either the unique safe physical backrest top
        edge or ``None``. A chair may therefore remain detected when no safe
        grasp line is visible.
    """

    rgb = np.asarray(rgb)
    z = np.asarray(z, np.float32).copy()
    ignore = np.asarray(ignore)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be OpenCV BGR uint8 HxWx3")
    if z.shape != rgb.shape[:2]:
        raise ValueError("z must be aligned HxW depth in metres")
    if ignore.shape != rgb.shape[:2]:
        raise ValueError("ignore must be HxW")

    # Be robust to callers that use zero for invalid depth despite the public
    # NaN contract. Invalid depth can never certify a safe grasp line.
    z[(~np.isfinite(z)) | (z <= 0.0)] = np.nan
    raw_ignore = (ignore > 0).astype(np.uint8)
    if recognition_only:
        safe_ignore = raw_ignore
        halo_radius_px = 0
        halo_reference_depth_m = None
    else:
        safe_ignore, halo_radius_px, halo_reference_depth_m = (
            _expand_ignore_for_metric_clearance(raw_ignore, z, camera)
        )
    safe_z = z.copy()
    safe_z[safe_ignore > 0] = np.nan

    # Cheap content gate rejects degenerate dark/bright/low-texture frames
    # before Hough and part geometry. It uses no absolute image position,
    # training filename, background identity or object colour.
    gate_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    contrast = float(
        np.percentile(gate_gray, 92) - np.percentile(gate_gray, 8)
    )
    gate_edges = cv2.Canny(gate_gray, 35, 115)
    gate_edges[safe_ignore > 0] = 0
    edge_count = int(np.sum(gate_edges > 0))
    if contrast < 5.0 and edge_count < max(80, int(0.00012 * gate_gray.size)):
        return []

    maps = _appearance_maps(rgb, safe_ignore)
    segments = _extract_segments(maps["gray"], maps["edge"], safe_ignore)
    metric_candidates = _height_first_candidates(
        rgb,
        safe_z,
        camera,
        safe_ignore,
        scene,
        maps,
        raw_ignore,
        recognition_only=recognition_only,
    )
    visual_candidates = _line_driven_candidates(
        rgb,
        safe_z,
        camera,
        safe_ignore,
        scene,
        maps,
        segments,
        raw_ignore,
        recognition_only=recognition_only,
    )
    candidates = _deduplicate(metric_candidates + visual_candidates)
    safety_debug = {
        "clearance_m": 0.0 if recognition_only else 0.025,
        "expanded_radius_px": int(halo_radius_px),
        "reference_depth_m": (
            None
            if halo_reference_depth_m is None
            else round(float(halo_reference_depth_m), 5)
        ),
        "raw_ignore_area_px": int(np.sum(raw_ignore > 0)),
        "expanded_ignore_area_px": int(np.sum(safe_ignore > 0)),
    }
    for candidate in candidates:
        candidate.setdefault("debug", {})["ignore_safety"] = safety_debug
    return candidates


__all__ = ["detect_chair_candidates"]
