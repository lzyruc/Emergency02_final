from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


CLASS_COLORS = {
    "basket": (255, 220, 0),
    "trash_can": (0, 255, 255),
    "box": (0, 165, 255),
    "chair": (255, 0, 255),
}


def load_camera(camera):
    if isinstance(camera, (str, Path)):
        camera = json.loads(Path(camera).read_text(encoding="utf-8"))
    if "color" in camera and isinstance(camera["color"], dict):
        merged = dict(camera["color"])
        merged["depth_scale_m"] = camera.get("depth_scale_m", 0.001)
        merged["invalid_depth_values"] = camera.get(
            "invalid_depth_values", [0, 65535]
        )
        camera = merged
    required = ("fx", "fy", "cx", "cy", "width", "height")
    missing = [key for key in required if key not in camera]
    if missing:
        raise ValueError(f"camera config missing: {', '.join(missing)}")
    out = dict(camera)
    for key in ("fx", "fy", "cx", "cy", "depth_scale_m"):
        out[key] = float(out.get(key, 0.0001 if key == "depth_scale_m" else 0))
    out["width"] = int(out["width"])
    out["height"] = int(out["height"])
    out.setdefault("invalid_depth_values", [0, 65535])
    return out


def depth_meters(depth_raw, camera):
    depth = np.asarray(depth_raw)
    z = depth.astype(np.float32) * float(camera["depth_scale_m"])
    valid = np.isfinite(z)
    for value in camera.get("invalid_depth_values", (0, 65535)):
        valid &= depth != value
    valid &= z > 0.025
    valid &= z < 8.0
    z[~valid] = np.nan
    return z, valid


def robust_depth(z, mask, percentile=50.0):
    values = z[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size < 12:
        return None
    lo, hi = np.percentile(values, (10, 80))
    values = values[(values >= lo) & (values <= hi)]
    if values.size < 8:
        return None
    return float(np.percentile(values, percentile))


def bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def expand_bbox(bbox, shape, fraction=0.08):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    pad = int(round(fraction * max(x2 - x1, y2 - y1)))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ]


def component_masks(mask, min_area):
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    order = sorted(
        range(1, n),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    for index in order:
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        yield labels == index, area


def retain_weak_components(weak, strong, min_area, min_strong):
    weak = (weak > 0).astype(np.uint8)
    strong = strong > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    keep = np.zeros(n, dtype=bool)
    if strong.any():
        strong_count = np.bincount(labels[strong], minlength=n)
    else:
        strong_count = np.zeros(n, dtype=np.int64)
    for index in range(1, n):
        keep[index] = (
            stats[index, cv2.CC_STAT_AREA] >= min_area
            and strong_count[index] >= min_strong
        )
    return keep[labels].astype(np.uint8)


def sample_image(image, points, default=0):
    h, w = image.shape[:2]
    pts = np.rint(points).astype(np.int32)
    ok = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )
    if image.ndim == 2:
        out = np.full(len(pts), default, dtype=image.dtype)
        out[ok] = image[pts[ok, 1], pts[ok, 0]]
    else:
        out = np.full((len(pts), image.shape[2]), default, dtype=image.dtype)
        out[ok] = image[pts[ok, 1], pts[ok, 0]]
    return out, ok


def line_points(p0, p1, count=None):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    length = float(np.linalg.norm(p1 - p0))
    if count is None:
        count = max(12, int(math.ceil(length)))
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    return p0[None, :] * (1.0 - t) + p1[None, :] * t


def line_normal(p0, p1):
    delta = np.asarray(p1, np.float32) - np.asarray(p0, np.float32)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.array([0.0, 1.0], np.float32)
    return np.array([-delta[1], delta[0]], np.float32) / norm


def line_mask_support(mask, p0, p1, radius=2):
    points = line_points(p0, p1)
    normal = line_normal(p0, p1)
    values = []
    for offset in range(-radius, radius + 1):
        sampled, ok = sample_image(mask, points + offset * normal)
        values.append((sampled > 0) & ok)
    return float(np.mean(np.any(np.stack(values, axis=0), axis=0)))


def line_gradient_support(gray, p0, p1, radius=2, gradients=None):
    if gradients is None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gx, gy = gradients
    normal = line_normal(p0, p1)
    points = line_points(p0, p1)
    strengths = []
    for offset in range(-radius, radius + 1):
        sx, ok = sample_image(gx, points + offset * normal)
        sy, _ = sample_image(gy, points + offset * normal)
        strengths.append(np.abs(sx * normal[0] + sy * normal[1]) * ok)
    best = np.max(np.stack(strengths, axis=0), axis=0)
    return float(np.percentile(best, 55)), float(np.mean(best > 80.0))


def normalized_angle(p0, p1):
    delta = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
    angle = math.atan2(float(delta[1]), float(delta[0]))
    while angle < 0:
        angle += math.pi
    while angle >= math.pi:
        angle -= math.pi
    return angle


def angle_distance(a, b):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def crop_terminal(p0, p1, centroid, fraction=0.36):
    """Keep the terminal part farther from the object centroid."""
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    centroid = np.asarray(centroid, np.float32)
    if np.linalg.norm(p0 - centroid) >= np.linalg.norm(p1 - centroid):
        outer, inner = p0, p1
    else:
        outer, inner = p1, p0
    end = outer + float(fraction) * (inner - outer)
    return outer, end


def ellipse_points(ellipse, count=None, scale=1.0):
    (cx, cy), (width, height), angle_deg = ellipse
    circumference = math.pi * (
        3 * (width + height)
        - math.sqrt(max(0.0, (3 * width + height) * (width + 3 * height)))
    ) / 2.0
    if count is None:
        count = max(180, int(math.ceil(circumference * 1.3)))
    theta = np.linspace(0, 2 * math.pi, count, endpoint=False)
    x = 0.5 * width * scale * np.cos(theta)
    y = 0.5 * height * scale * np.sin(theta)
    angle = math.radians(angle_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    return np.column_stack(
        [cx + ca * x - sa * y, cy + sa * x + ca * y]
    ).astype(np.float32)


def ellipse_angular_coverage(points, ellipse, bins=36):
    (cx, cy), (width, height), angle_deg = ellipse
    if min(width, height) < 1:
        return 0.0
    angle = -math.radians(angle_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    d = np.asarray(points, np.float64) - np.array([cx, cy])
    x = ca * d[:, 0] - sa * d[:, 1]
    y = sa * d[:, 0] + ca * d[:, 1]
    theta = np.arctan2(y / (0.5 * height), x / (0.5 * width))
    index = np.floor((theta + math.pi) * bins / (2 * math.pi)).astype(int)
    index = np.clip(index, 0, bins - 1)
    return float(len(np.unique(index)) / bins)


def smooth_ramp(value, low, high):
    if high <= low:
        return float(value >= high)
    t = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def bell(value, center, sigma):
    return float(math.exp(-0.5 * ((value - center) / max(sigma, 1e-8)) ** 2))


def safe_point(point, shape):
    if point is None:
        return None
    h, w = shape[:2]
    x = int(np.clip(round(float(point[0])), 0, w - 1))
    y = int(np.clip(round(float(point[1])), 0, h - 1))
    return [x, y]


def _automatic_gripper_mask(rgb, z):
    """Conservative border-connected, relatively-near gripper proposal.

    This is intentionally only a fallback for a missing robot mask.  It uses
    relative depth and local occlusion topology, not a fixed image region or a
    frame-specific colour.  Ambiguous long rails are rejected by the component
    aspect and neighbourhood-depth checks.
    """
    h, w = z.shape
    result = np.zeros((h, w), np.uint8)
    valid = np.isfinite(z) & (z > 0.045) & (z < 6.0)
    values = z[valid]
    if len(values) < 1000:
        return result
    reference = float(np.percentile(values, 20.0))
    near_limit = min(0.78 * reference, reference - 0.080)
    if near_limit <= 0.045:
        return result
    near = valid & (z < near_limit)
    near = cv2.morphologyEx(
        near.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    near[~valid] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(near, 8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    ring_radius = max(13, int(round(0.018 * math.hypot(w, h))))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
    )
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        touches = x <= 2 or y <= 2 or x + width >= w - 2 or y + height >= h - 2
        if not touches:
            continue
        area_fraction = float(area / (h * w))
        extent = float(area / max(1, width * height))
        if not (0.0015 <= area_fraction <= 0.1200) or extent < 0.12:
            continue
        component = labels == index
        points = np.column_stack(np.where(component)[::-1]).astype(np.float32)
        if len(points) < 5:
            continue
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(points)
        aspect = float(max(side_a, side_b) / max(1.0, min(side_a, side_b)))
        if not (1.20 <= aspect <= 8.00):
            continue
        if float(np.mean(hsv[..., 1][component] < 95)) < 0.45:
            continue
        component_depth = z[component]
        depth_median = float(np.median(component_depth))
        depth_mad = float(np.median(np.abs(component_depth - depth_median)))
        if depth_mad > 0.045:
            continue
        ring = cv2.dilate(component.astype(np.uint8), ring_kernel) > 0
        ring &= (~component) & valid
        ring_values = z[ring]
        if len(ring_values) < 50:
            continue
        ring_depth = float(np.median(ring_values))
        if ring_depth - depth_median < 0.080:
            continue
        if depth_median / max(ring_depth, 1e-6) > 0.78:
            continue
        result[component] = 255
    return cv2.dilate(
        result,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )


def make_gripper_mask(rgb, z, supplied=None):
    """Combine a supplied robot mask with a conservative automatic mask."""
    h, w = rgb.shape[:2]
    supplied_safe = np.zeros((h, w), np.uint8)
    if supplied is not None:
        supplied = np.asarray(supplied)
        if supplied.shape != (h, w):
            raise ValueError("ignore_mask must have the same HxW as RGB")
        supplied_safe[supplied > 0] = 255
        radius = int(np.clip(round(0.010 * math.hypot(w, h)), 8, 30))
        supplied_safe = cv2.morphologyEx(
            supplied_safe,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        supplied_safe = cv2.dilate(
            supplied_safe,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            ),
        )

    automatic = _automatic_gripper_mask(rgb, z)
    return cv2.bitwise_or(supplied_safe, automatic)
