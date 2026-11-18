"""Scale-adaptive RGB-D trash-can rim detection for a wrist camera.

The detector deliberately does not use image position, camera pose, filenames,
or a learned detector.  Proposals come from several independent edge and
bright/low-chroma masks.  They are ranked with scale-normalised 2-D evidence,
the known 255 mm mouth diameter, local depth, cavity/body appearance, and
optional ground-relative scene geometry.

Only a point on a *visually observed* part of the selected rim is returned as
``grasp_point``.  An ellipse may bridge an occlusion for visualisation, but an
ignored or inferred arc can never supply the grasp point.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


MOUTH_DIAMETER_M = 0.255
MOUTH_HEIGHT_M = 0.275
GRIPPER_CLEARANCE_M = 0.022


def _smooth(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    value = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _bell(value: float, center: float, sigma: float) -> float:
    return float(math.exp(-0.5 * ((value - center) / max(sigma, 1e-8)) ** 2))


def _ellipse_points(
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
    count: int,
    scale: float = 1.0,
) -> np.ndarray:
    (cx, cy), (width, height), angle_deg = ellipse
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    angle = math.radians(float(angle_deg))
    ca, sa = math.cos(angle), math.sin(angle)
    x = 0.5 * float(width) * float(scale) * np.cos(theta)
    y = 0.5 * float(height) * float(scale) * np.sin(theta)
    return np.column_stack(
        [float(cx) + ca * x - sa * y, float(cy) + sa * x + ca * y]
    ).astype(np.float32)


def _normalised_ellipse(ellipse: Any):
    """Return a finite OpenCV ellipse with positive full-axis lengths."""
    (cx, cy), (width, height), angle = ellipse
    values = np.asarray([cx, cy, width, height, angle], np.float64)
    if not np.all(np.isfinite(values)):
        return None
    width, height = abs(float(width)), abs(float(height))
    if width < 1.0 or height < 1.0:
        return None
    return (
        (float(cx), float(cy)),
        (width, height),
        float(angle) % 180.0,
    )


def _major_geometry(ellipse):
    (_, _), (width, height), angle = ellipse
    if width >= height:
        return float(width), float(height), math.radians(float(angle))
    return float(height), float(width), math.radians(float(angle) + 90.0)


def _sample(image: np.ndarray, points: np.ndarray, default: float = 0.0):
    image_h, image_w = image.shape[:2]
    x = np.rint(points[:, 0]).astype(np.int32)
    y = np.rint(points[:, 1]).astype(np.int32)
    valid = (x >= 0) & (x < image_w) & (y >= 0) & (y < image_h)
    if image.ndim == 2:
        values = np.full(len(points), default, dtype=np.float32)
        values[valid] = image[y[valid], x[valid]]
    else:
        values = np.full(
            (len(points), image.shape[2]), default, dtype=np.float32
        )
        values[valid] = image[y[valid], x[valid]]
    return values, valid


def _circular_bin_coverage(
    supported: np.ndarray, visible: np.ndarray, bins: int = 72
):
    indices = np.array_split(np.arange(len(supported)), bins)
    flags = np.zeros(bins, dtype=bool)
    available = np.zeros(bins, dtype=bool)
    for index, members in enumerate(indices):
        available[index] = bool(np.any(visible[members]))
        flags[index] = bool(np.any(supported[members] & visible[members]))
    available_count = int(available.sum())
    coverage = (
        float(flags.sum() / max(available_count, 1))
        if available_count
        else 0.0
    )
    doubled = np.concatenate([flags, flags])
    longest = current = 0
    for flag in doubled:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    longest = min(longest, bins)
    return coverage, float(longest / bins), flags


def _robust_median(values: np.ndarray, minimum: int = 8):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) < minimum:
        return None
    low, high = np.percentile(values, (12.0, 88.0))
    values = values[(values >= low) & (values <= high)]
    if len(values) < minimum:
        return None
    return float(np.median(values))


def _fit_contour(contour: np.ndarray):
    if len(contour) < 12:
        return []
    out = []
    for fitter in (cv2.fitEllipseAMS, cv2.fitEllipseDirect):
        try:
            ellipse = _normalised_ellipse(fitter(contour))
        except cv2.error:
            ellipse = None
        if ellipse is not None:
            out.append(ellipse)
    return out


def _white_component_proposals(mask: np.ndarray, source: str):
    """Fit both contour ellipses and thin-band rotated-box hypotheses."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    proposals = []
    contours = sorted(
        contours,
        key=lambda contour: cv2.arcLength(contour, True),
        reverse=True,
    )
    for contour in contours[:320]:
        area = float(abs(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if len(contour) < 12 or perimeter < 34.0 or area < 18.0:
            continue
        for ellipse in _fit_contour(contour):
            proposals.append((ellipse, source, contour.reshape(-1, 2)))

        # A highly oblique rim can be only a slender bright band.  Its outer
        # contour remains reliable even when an unconstrained conic fit
        # collapses.  Several minor-axis hypotheses are later resolved by edge
        # and depth evidence, not by the component's pixel thickness.
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        major, minor = max(float(rw), float(rh)), min(float(rw), float(rh))
        if major < 46.0 or minor < 3.0 or major / max(minor, 1e-6) < 2.4:
            continue
        base_angle = float(angle) if rw >= rh else float(angle) + 90.0
        for multiplier in (1.2, 1.8, 2.6, 3.8):
            proposed_minor = min(major, max(6.0, minor * multiplier))
            ellipse = (
                (float(cx), float(cy)),
                (major, proposed_minor),
                base_angle,
            )
            proposals.append(
                (_normalised_ellipse(ellipse), source + "_band", contour.reshape(-1, 2))
            )
    return proposals


def _edge_proposals(edge_maps: list[np.ndarray]):
    proposals = []
    for level, edge in enumerate(edge_maps):
        contours, _ = cv2.findContours(
            edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(
            contours,
            key=lambda contour: cv2.arcLength(contour, False),
            reverse=True,
        )
        for contour in contours[:420]:
            perimeter = float(cv2.arcLength(contour, False))
            if len(contour) < 18 or perimeter < 42.0:
                continue
            for ellipse in _fit_contour(contour):
                proposals.append(
                    (ellipse, f"canny_{level}", contour.reshape(-1, 2))
                )
                major, minor, major_angle = _major_geometry(ellipse)
                if major >= 100.0 and minor / major < 0.14:
                    # A nearly frontal mouth may expose only one long white
                    # arc; fitting that arc alone collapses the minor axis.
                    # Expand it symmetrically and let the radial ring profile,
                    # cavity, body and metric diameter choose the true outer
                    # lip.  No image direction is assumed.
                    center = np.asarray(ellipse[0], np.float64)
                    minor_axis = np.array(
                        [-math.sin(major_angle), math.cos(major_angle)],
                        np.float64,
                    )
                    for multiplier in (1.7, 2.4, 3.2):
                        expanded_minor = min(major, minor * multiplier)
                        for shift_fraction in (-0.20, 0.0, 0.20):
                            expanded_center = (
                                center
                                + shift_fraction
                                * expanded_minor
                                * minor_axis
                            )
                            expanded = _normalised_ellipse(
                                (
                                    tuple(expanded_center.tolist()),
                                    (major, expanded_minor),
                                    math.degrees(major_angle),
                                )
                            )
                            proposals.append(
                                (
                                    expanded,
                                    f"canny_{level}_flat_expand",
                                    contour.reshape(-1, 2),
                                )
                            )
    return proposals


def _body_guided_proposals(appearance, z, camera, ignore):
    """Infer partial flat-rim hypotheses from a solid blue-grey body.

    A large gripper can split both bright rim contours.  The remaining body is
    much larger and still gives an orientation-free symmetry axis, a mouth
    centre near either terminal, and a metric major-axis estimate.  These are
    only proposals; all normal radial-ring and RGB-D semantic gates still run.
    """
    body_mask = (appearance["blue"] >= 0.35).astype(np.uint8)
    body_mask[ignore > 0] = 0
    body_mask = cv2.morphologyEx(
        body_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    body_mask[ignore > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        body_mask, 8
    )
    order = sorted(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:16]
    proposals = []
    mean_focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    for index in order:
        if int(stats[index, cv2.CC_STAT_AREA]) < 800:
            continue
        yy, xx = np.nonzero(labels == index)
        xy = np.column_stack([xx, yy]).astype(np.float64)
        component_depth = _robust_median(z[labels == index], minimum=20)
        if component_depth is None or not (0.10 <= component_depth <= 1.50):
            continue
        center = np.mean(xy, axis=0)
        covariance = np.cov(xy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        body_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        rim_axis = np.array([-body_axis[1], body_axis[0]], np.float64)
        body_projection = (xy - center) @ body_axis
        rim_projection = (xy - center) @ rim_axis
        low_body, high_body = np.percentile(body_projection, (2.0, 98.0))
        low_rim, high_rim = np.percentile(rim_projection, (2.0, 98.0))
        body_length = float(high_body - low_body)
        body_width = float(high_rim - low_rim)
        if body_length < 45.0 or body_width < 40.0:
            continue
        apparent_width_m = body_width * component_depth / mean_focal
        apparent_length_m = body_length * component_depth / mean_focal
        oriented_fill = float(
            len(xy) / max(body_length * body_width, 1.0)
        )
        mean_blue = float(np.mean(appearance["blue"][labels == index]))
        body_aspect = body_length / max(body_width, 1e-6)
        if not (0.12 <= apparent_width_m <= 0.42):
            continue
        if not (0.13 <= apparent_length_m <= 0.58):
            continue
        if not (0.55 <= body_aspect <= 2.80):
            continue
        if oriented_fill < 0.38 or mean_blue < 0.48:
            continue
        focal_along_rim = math.hypot(
            float(camera["fx"]) * float(rim_axis[0]),
            float(camera["fy"]) * float(rim_axis[1]),
        )
        metric_major = (
            focal_along_rim * MOUTH_DIAMETER_M / component_depth
        )
        major_hypotheses = (1.10 * body_width, 1.25 * body_width, metric_major)
        seed = xy[:: max(1, len(xy) // 800)]
        rim_angle = math.degrees(math.atan2(rim_axis[1], rim_axis[0]))
        for sign, terminal in ((-1.0, low_body), (1.0, high_body)):
            end_center = center + terminal * body_axis
            for major in major_hypotheses:
                major = float(major)
                if not (44.0 <= major <= 1.45 * math.hypot(*z.shape)):
                    continue
                for aspect in (0.08, 0.12, 0.18, 0.25):
                    for outward in (0.0, 0.04, 0.08):
                        ellipse_center = (
                            end_center
                            + sign * outward * major * body_axis
                        )
                        ellipse = _normalised_ellipse(
                            (
                                tuple(ellipse_center.tolist()),
                                (major, aspect * major),
                                rim_angle,
                            )
                        )
                        proposals.append(
                            (ellipse, "blue_body_guided", seed)
                        )
    return proposals


def _deduplicate_proposals(proposals, image_shape):
    image_h, image_w = image_shape[:2]
    diagonal = math.hypot(image_w, image_h)
    filtered = []
    for ellipse, source, seed in proposals:
        if ellipse is None:
            continue
        (cx, cy), _, _ = ellipse
        major, minor, _ = _major_geometry(ellipse)
        aspect = minor / max(major, 1e-6)
        if major < 42.0 or major > 1.55 * diagonal:
            continue
        if minor < 4.5 or not (0.065 <= aspect <= 1.02):
            continue
        if not (
            -0.55 * major <= cx <= image_w + 0.55 * major
            and -0.55 * major <= cy <= image_h + 0.55 * major
        ):
            continue
        filtered.append((ellipse, source, seed, major, minor))

    # Prefer proposals with longer source support.  Coarse parameter-space
    # quantisation removes thousands of near-identical Canny contours while
    # retaining different minor-axis hypotheses for a side-view rim.
    filtered.sort(key=lambda item: len(item[2]), reverse=True)
    keys = set()
    kept = []
    for item in filtered:
        ellipse, source, seed, major, minor = item
        (cx, cy), _, angle = ellipse
        _, _, major_angle = _major_geometry(ellipse)
        key = (
            int(round(cx / max(5.0, 0.035 * major))),
            int(round(cy / max(5.0, 0.035 * major))),
            int(round(math.log(max(major, 1.0)) / 0.075)),
            int(round(math.log(max(minor, 1.0)) / 0.11)),
            int(round((major_angle % math.pi) / math.radians(8.0))),
        )
        if key in keys:
            continue
        keys.add(key)
        kept.append((ellipse, source, seed))
        if len(kept) >= 760:
            break
    return kept


def _interior_points(ellipse, count: int = 620, scale: float = 0.72):
    # Deterministic sunflower sampling avoids allocating a full-frame mask per
    # proposal and remains stable for very flat ellipses.
    (cx, cy), (width, height), angle_deg = ellipse
    index = np.arange(count, dtype=np.float64) + 0.5
    radius = np.sqrt(index / count) * float(scale)
    theta = index * (math.pi * (3.0 - math.sqrt(5.0)))
    x = 0.5 * float(width) * radius * np.cos(theta)
    y = 0.5 * float(height) * radius * np.sin(theta)
    angle = math.radians(float(angle_deg))
    ca, sa = math.cos(angle), math.sin(angle)
    return np.column_stack(
        [float(cx) + ca * x - sa * y, float(cy) + sa * x + ca * y]
    ).astype(np.float32)


def _body_lobe_points(ellipse, sign: float):
    """Sample a tapering body hypothesis on either side of the rim."""
    major, minor, angle = _major_geometry(ellipse)
    center = np.asarray(ellipse[0], np.float32)
    major_axis = np.array([math.cos(angle), math.sin(angle)], np.float32)
    minor_axis = np.array([-math.sin(angle), math.cos(angle)], np.float32)
    along = np.linspace(0.24 * minor, 0.95 * major, 24)
    rows = []
    for distance in along:
        progress = float(
            np.clip(
                (distance - 0.24 * minor) / max(0.95 * major, 1.0),
                0.0,
                1.0,
            )
        )
        half_width = 0.46 * major * (1.0 - 0.18 * progress)
        across = np.linspace(-half_width, half_width, 25)
        rows.append(
            center[None, :]
            + float(sign) * distance * minor_axis[None, :]
            + across[:, None] * major_axis[None, :]
        )
    return np.concatenate(rows, axis=0).astype(np.float32)


def _appearance_response(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    b, g, r = cv2.split(rgb)
    b16 = b.astype(np.int16)
    r16 = r.astype(np.int16)
    min_channel = np.minimum(np.minimum(b, g), r)

    value_f = value.astype(np.float32)
    saturation_f = saturation.astype(np.float32)
    min_f = min_channel.astype(np.float32)
    white_response = (
        np.clip((value_f - 88.0) / 125.0, 0.0, 1.0)
        * np.clip((132.0 - saturation_f) / 105.0, 0.0, 1.0)
        * np.clip((min_f - 65.0) / 125.0, 0.0, 1.0)
    )

    blue_advantage = (b16 - r16).astype(np.float32)
    hue_ok = ((hue >= 76) & (hue <= 137)).astype(np.float32)
    blue_response = (
        hue_ok
        * np.clip((blue_advantage - 1.0) / 25.0, 0.0, 1.0)
        * np.clip((180.0 - saturation_f) / 130.0, 0.0, 1.0)
        * np.clip((value_f - 28.0) / 75.0, 0.0, 1.0)
    )
    # The target plastic ranges from blue-grey to moderately blue.  Keep a
    # weak categorical map for adjacency/proposal formation while using the
    # continuous response for scoring.
    muted_blue = (
        (hue >= 76)
        & (hue <= 137)
        & (blue_advantage >= 2.0)
        & (saturation >= 8)
        & (saturation <= 180)
        & (value >= 30)
        & (value <= 235)
    )
    return {
        "hsv": hsv,
        "hue": hue,
        "saturation": saturation,
        "value": value,
        "white": white_response.astype(np.float32),
        "blue": blue_response.astype(np.float32),
        "muted_blue": muted_blue,
        "min_channel": min_channel,
    }


def _proposal_maps(rgb: np.ndarray, ignore: np.ndarray, appearance: dict):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Low-threshold levels recover the softly lit far/side rim; higher levels
    # suppress floor texture and basket perforations.  Edges are combined only
    # for scoring, while each level supplies independent contour proposals.
    smoothed = (
        cv2.GaussianBlur(gray, (3, 3), 0.7),
        cv2.GaussianBlur(gray, (5, 5), 1.25),
        cv2.GaussianBlur(gray, (7, 7), 2.0),
    )
    thresholds = ((18, 55), (38, 115), (72, 205))
    edge_maps = []
    blocked = (ignore > 0).astype(np.uint8)
    for image, (low, high) in zip(smoothed, thresholds):
        edge = cv2.Canny(image, low, high, L2gradient=True)
        edge[blocked > 0] = 0
        edge_maps.append(edge)
    edge_union = np.maximum.reduce(edge_maps)
    edge_distance = cv2.distanceTransform(
        255 - edge_union, cv2.DIST_L2, 3
    )

    white_strict = (
        (appearance["white"] >= 0.34)
        & (appearance["value"] >= 112)
        & (appearance["saturation"] <= 122)
    ).astype(np.uint8)
    white_bright = (
        (appearance["white"] >= 0.24)
        & (appearance["value"] >= 102)
        & (appearance["saturation"] <= 145)
    ).astype(np.uint8)
    white_strict[blocked > 0] = 0
    white_bright[blocked > 0] = 0
    white_strict = cv2.morphologyEx(
        white_strict,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    white_strict[blocked > 0] = 0

    object_like = cv2.morphologyEx(
        appearance["muted_blue"].astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    object_like[blocked > 0] = 0
    adjacency_masks = []
    for radius in (7, 15, 29):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        adjacent = white_bright & cv2.dilate(object_like, kernel)
        adjacent = cv2.morphologyEx(
            adjacent,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        adjacent[blocked > 0] = 0
        adjacency_masks.append(adjacent)
    return gray, edge_maps, edge_distance, white_strict, adjacency_masks


def _radial_samples(ellipse, maps: dict, edge_distance, z, ignore):
    major, _, _ = _major_geometry(ellipse)
    count = int(np.clip(math.ceil(math.pi * major * 1.75), 360, 1800))
    scales = np.asarray((0.94, 0.975, 1.0, 1.025, 1.06), np.float32)
    point_stack = np.stack(
        [_ellipse_points(ellipse, count, float(scale)) for scale in scales],
        axis=0,
    )
    distance_stack = []
    white_stack = []
    value_stack = []
    depth_stack = []
    valid_stack = []
    blocked_stack = []
    for points in point_stack:
        distances, visible = _sample(edge_distance, points, default=99.0)
        white, _ = _sample(maps["white"], points, default=0.0)
        value, _ = _sample(maps["value"], points, default=0.0)
        depth, _ = _sample(z, points, default=np.nan)
        blocked, _ = _sample(ignore, points, default=255.0)
        distance_stack.append(distances)
        white_stack.append(white)
        value_stack.append(value)
        depth_stack.append(depth)
        valid_stack.append(visible)
        blocked_stack.append(blocked)
    distances = np.stack(distance_stack)
    white = np.stack(white_stack)
    values = np.stack(value_stack)
    depths = np.stack(depth_stack)
    visible = np.stack(valid_stack)
    blocked = np.stack(blocked_stack)

    # Select, independently at each angle, the radial sample which best agrees
    # with an actual image edge and a bright/low-chroma rim.
    quality = (
        np.exp(-distances / max(1.5, 0.004 * major))
        * (0.42 + 0.58 * white)
    )
    quality[(~visible) | (blocked > 0)] = -1.0
    best_index = np.argmax(quality, axis=0)
    columns = np.arange(count)
    return {
        "points": point_stack[best_index, columns],
        "distances": distances[best_index, columns],
        "white": white[best_index, columns],
        "value": values[best_index, columns],
        "depth": depths[best_index, columns],
        "visible": visible[best_index, columns],
        "blocked": blocked[best_index, columns] > 0,
        "quality": quality[best_index, columns],
    }


def _radial_white_profile(ellipse, white_response, ignore):
    """Measure whether low-chroma brightness peaks *on* the fitted ring.

    Looking for the best nearby white pixel is useful under small calibration
    errors, but by itself lets a cable, label, or whole pale object masquerade
    as a rim.  A real plastic lip has a radial peak: at least one immediately
    adjacent side is materially less white than the fitted annulus.
    """
    major, _, _ = _major_geometry(ellipse)
    count = int(np.clip(math.ceil(math.pi * major), 300, 1200))
    scales = (0.72, 0.82, 0.88, 0.94, 1.0, 1.06, 1.14, 1.25)
    means = []
    supports = []
    for scale in scales:
        points = _ellipse_points(ellipse, count, scale)
        values, visible = _sample(white_response, points, default=0.0)
        blocked, _ = _sample(ignore, points, default=255.0)
        valid = visible & (blocked == 0)
        if int(valid.sum()) < max(20, int(0.18 * count)):
            means.append(0.0)
            supports.append(0.0)
        else:
            means.append(float(np.mean(values[valid])))
            supports.append(float(np.mean(values[valid] >= 0.24)))
    core = max(means[3:6])
    inner = max(means[:3])
    outer = max(means[6:])
    return {
        "scales": scales,
        "means": means,
        "supports": supports,
        "core": float(core),
        "inner": float(inner),
        "outer": float(outer),
        "peak_both_sides": float(core - 0.5 * (inner + outer)),
        "peak_one_side": float(core - min(inner, outer)),
    }


def _profile_median(
    image: np.ndarray,
    ignore: np.ndarray,
    center: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    across: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Robust cross-rim profile while excluding the known robot mask."""
    rows = [
        center[None, :]
        + float(offset) * minor_axis[None, :]
        + across[:, None] * major_axis[None, :]
        for offset in offsets
    ]
    points = np.concatenate(rows, axis=0).astype(np.float32)
    values, visible = _sample(image, points, default=np.nan)
    blocked, _ = _sample(ignore, points, default=255.0)
    values = np.asarray(values, np.float32).reshape(len(offsets), len(across))
    usable = (
        visible.reshape(len(offsets), len(across))
        & (blocked.reshape(len(offsets), len(across)) == 0)
        & np.isfinite(values)
    )
    minimum = max(18, int(round(0.24 * len(across))))
    profile = np.full(len(offsets), np.nan, np.float32)
    for index in range(len(offsets)):
        if int(usable[index].sum()) >= minimum:
            profile[index] = float(np.median(values[index][usable[index]]))
    return profile


def _smooth_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if int(np.isfinite(values).sum()) < max(12, len(values) // 3):
        return np.full_like(values, np.nan)
    finite = np.isfinite(values)
    filled = values.copy()
    indices = np.arange(len(values), dtype=np.float32)
    filled[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return cv2.GaussianBlur(filled[:, None], (1, 7), 1.2).ravel()


def _true_runs(flags: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, flag in enumerate(np.r_[np.asarray(flags, bool), False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                runs.append((start, index - 1))
            start = None
    return runs


def _refine_oblique_rim(
    ellipse,
    appearance: dict,
    ignore: np.ndarray,
):
    """Refine a side-view rim from its white-lip-to-blue-body topology.

    A generic conic fit to a near-horizontal lip often uses a lower texture edge
    on the bucket body and therefore makes the ellipse too tall.  The physical
    ordering is much stronger: background/cavity, bright lip, a short dark
    underside, then the solid blue-grey body.  This routine measures that
    ordering along the ellipse normal.  It has no image-position or frame-name
    prior and remains valid after rotation, scaling and partial robot occlusion.
    """
    center = np.asarray(ellipse[0], np.float64)
    major, minor, major_angle = _major_geometry(ellipse)
    aspect = minor / max(major, 1e-6)
    if not (0.065 <= aspect <= 0.36) or major < 90.0:
        return None

    major_axis = np.array(
        [math.cos(major_angle), math.sin(major_angle)], np.float64
    )
    minor_axis = np.array(
        [-math.sin(major_angle), math.cos(major_angle)], np.float64
    )
    across_count = int(np.clip(round(0.45 * major), 81, 241))
    across = np.linspace(-0.22 * major, 0.22 * major, across_count)

    # Determine the body direction from two symmetric bands.  A large supplied
    # gripper mask can remove samples but can never vote as blue body evidence.
    side_scores = []
    side_offsets = np.linspace(0.18 * major, 0.30 * major, 7)
    for sign in (-1.0, 1.0):
        profile = _profile_median(
            appearance["blue"],
            ignore,
            center,
            major_axis,
            minor_axis,
            across,
            sign * side_offsets,
        )
        side_scores.append(
            float(np.nanmedian(profile))
            if np.any(np.isfinite(profile))
            else 0.0
        )
    body_index = int(np.argmax(side_scores))
    body_score = side_scores[body_index]
    other_score = side_scores[1 - body_index]
    if body_score < 0.34 or body_score - other_score < 0.16:
        return None
    body_sign = -1.0 if body_index == 0 else 1.0
    body_axis = body_sign * minor_axis

    offset_count = int(np.clip(round(0.47 * major), 161, 281))
    offsets = np.linspace(-0.25 * major, 0.22 * major, offset_count)
    white = _smooth_profile(
        _profile_median(
            appearance["white"],
            ignore,
            center,
            major_axis,
            body_axis,
            across,
            offsets,
        )
    )
    blue = _smooth_profile(
        _profile_median(
            appearance["blue"],
            ignore,
            center,
            major_axis,
            body_axis,
            across,
            offsets,
        )
    )
    value = _smooth_profile(
        _profile_median(
            appearance["value"],
            ignore,
            center,
            major_axis,
            body_axis,
            across,
            offsets,
        )
        / 255.0
    )
    if not (
        np.all(np.isfinite(white))
        and np.all(np.isfinite(blue))
        and np.all(np.isfinite(value))
    ):
        return None
    white_peak = float(np.max(white))
    if white_peak < 0.42:
        return None
    white_threshold = max(0.30, 0.56 * white_peak)
    runs = _true_runs(
        white >= white_threshold,
        max(3, int(round(0.008 * major))),
    )
    if not runs:
        return None

    body_tail = blue[int(round(0.55 * len(blue))) :]
    body_peak = float(np.percentile(body_tail, 90.0))
    if body_peak < 0.34:
        return None
    coarse_blue_threshold = max(0.14, 0.18 * body_peak)
    choices = []
    indices = np.arange(len(offsets))
    for start, end in runs:
        if offsets[start] > 0.04 * major or offsets[end] < -0.24 * major:
            continue
        later = np.flatnonzero(
            (indices > end) & (blue >= coarse_blue_threshold)
        )
        if not len(later):
            continue
        body_start = int(later[0])
        gap = float(offsets[body_start] - offsets[end])
        if not (0.0 <= gap <= 0.16 * major):
            continue
        midpoint = 0.5 * (offsets[start] + offsets[body_start])
        score = (
            float(np.mean(white[start : end + 1]))
            + 0.25 * min(1.0, body_peak)
            + 0.10
            * (1.0 - min(1.0, abs(midpoint) / max(0.18 * major, 1.0)))
        )
        choices.append((score, start, end))
    if not choices:
        return None
    _, run_start, run_end = max(choices)

    # The non-body outer boundary is the positive value transition into the
    # lip.  The body-side boundary is the first repeatable onset of blue-grey
    # material after that lip, i.e. the front outer rim rather than a lower
    # texture edge on the bucket wall.
    value_derivative = np.gradient(value, offsets)
    lead_low = max(0, run_start - int(round(0.025 * major)))
    lead_high = min(
        len(offsets), run_start + int(round(0.015 * major)) + 1
    )
    if lead_high <= lead_low:
        return None
    outer_index = lead_low + int(
        np.argmax(value_derivative[lead_low:lead_high])
    )
    fine_blue_threshold = max(0.11, 0.12 * body_peak)
    body_indices = np.flatnonzero(
        (indices > run_end) & (blue >= fine_blue_threshold)
    )
    if not len(body_indices):
        return None
    body_index = int(body_indices[0])
    outer_offset = float(offsets[outer_index])
    body_offset = float(offsets[body_index])
    refined_minor = body_offset - outer_offset
    if not (0.025 * major <= refined_minor <= 0.22 * major):
        return None
    # This is a corrective refinement.  If the original conic is already as
    # tight as the semantic profile, leave it untouched.
    if refined_minor >= 0.92 * minor:
        return None

    refined_center = center + 0.5 * (outer_offset + body_offset) * body_axis
    refined = _normalised_ellipse(
        (
            tuple(refined_center.tolist()),
            (float(major), float(refined_minor)),
            math.degrees(major_angle),
        )
    )
    if refined is None:
        return None
    return refined, {
        "old_minor_px": float(minor),
        "new_minor_px": float(refined_minor),
        "outer_offset_px": outer_offset,
        "body_offset_px": body_offset,
        "white_run_px": [
            float(offsets[run_start]),
            float(offsets[run_end]),
        ],
        "white_peak": white_peak,
        "body_peak": body_peak,
        "body_side_scores": [float(value) for value in side_scores],
    }


def _observed_grasp_on_ellipse(
    ellipse,
    appearance: dict,
    edge_distance: np.ndarray,
    z: np.ndarray,
    ignore: np.ndarray,
    ignore_clearance: np.ndarray,
    rim_depth: float | None,
    recognition_only: bool = False,
):
    """Choose an exact, directly observed point on the published ellipse."""
    image_h, image_w = z.shape
    major, _, _ = _major_geometry(ellipse)
    count = int(np.clip(math.ceil(math.pi * major * 1.75), 360, 1800))
    points = _ellipse_points(ellipse, count)
    distances, visible = _sample(edge_distance, points, default=99.0)
    white, _ = _sample(appearance["white"], points, default=0.0)
    depth, _ = _sample(z, points, default=np.nan)
    blocked, _ = _sample(ignore, points, default=255.0)
    if recognition_only:
        clearance = None
    else:
        clearance, _ = _sample(ignore_clearance, points, default=0.0)
    border = np.minimum.reduce(
        [
            points[:, 0],
            points[:, 1],
            image_w - 1.0 - points[:, 0],
            image_h - 1.0 - points[:, 1],
        ]
    )
    tolerance = max(2.0, 0.008 * major)
    observed = (
        visible
        & (blocked == 0)
        & np.isfinite(depth)
        & (distances <= tolerance)
        & (white >= 0.20)
        & (border >= max(3.0, 0.006 * major))
    )
    if not recognition_only:
        required_clearance = max(
            8.0, GRIPPER_CLEARANCE_M * major / MOUTH_DIAMETER_M
        )
        observed &= clearance >= required_clearance
    else:
        required_clearance = None
    if int(observed.sum()) < max(8, int(round(0.012 * count))):
        return None
    if rim_depth is None or not np.isfinite(rim_depth):
        depth_term = np.full(count, 0.45, np.float32)
    else:
        depth_term = np.exp(-np.abs(depth - float(rim_depth)) / 0.035)
    score = (
        0.43 * np.exp(-distances / max(1.0, 0.0045 * major))
        + 0.31 * white
        + 0.16 * depth_term
        + 0.10 * np.clip(border / max(16.0, 0.055 * major), 0.0, 1.0)
    )
    if not recognition_only:
        score = (
            0.88 * score
            + 0.12
            * np.clip(clearance / max(20.0, 0.08 * major), 0.0, 1.0)
        )
    score[~observed] = -1.0
    index = int(np.argmax(score))
    point = [
        int(np.clip(round(float(points[index, 0])), 0, image_w - 1)),
        int(np.clip(round(float(points[index, 1])), 0, image_h - 1)),
    ]
    return point, {
        "exact_ellipse_observed_fraction": float(np.mean(observed)),
        "grasp_score": float(score[index]),
        "grasp_ignore_clearance_px": (
            None if clearance is None else float(clearance[index])
        ),
        "required_gripper_clearance_px": (
            None
            if required_clearance is None
            else float(required_clearance)
        ),
    }


def _candidate_features(
    proposal,
    rgb,
    z,
    camera,
    ignore,
    ignore_clearance,
    maps,
    edge_distance,
    scene,
    recognition_only=False,
):
    ellipse, source, seed = proposal
    image_h, image_w = z.shape
    major, minor, major_angle = _major_geometry(ellipse)
    aspect = minor / max(major, 1e-6)

    samples = _radial_samples(ellipse, maps, edge_distance, z, ignore)
    tolerance = max(2.0, 0.0075 * major)
    visible = samples["visible"] & (~samples["blocked"])
    if float(np.mean(visible)) < 0.24:
        return None
    edge_supported = samples["distances"] <= tolerance
    white_supported = samples["white"] >= 0.24
    observed = visible & edge_supported & white_supported
    edge_support = float(np.mean(edge_supported[visible]))
    white_support = float(np.mean(white_supported[visible]))
    observed_support = float(np.mean(observed[visible]))
    edge_coverage, edge_longest, _ = _circular_bin_coverage(
        edge_supported, visible
    )
    observed_coverage, observed_longest, _ = _circular_bin_coverage(
        observed, visible
    )
    if edge_support < 0.20 or edge_coverage < 0.20:
        return None
    if white_support < 0.15 or observed_coverage < 0.10:
        return None

    radial_profile = _radial_white_profile(
        ellipse, maps["white"], ignore
    )
    # Hard semantic guard: brightness must be concentrated on the proposed
    # annulus rather than merely somewhere in its neighbourhood.  This rejects
    # a complete pale body ellipse, long floor markings, and most box labels.
    if radial_profile["core"] < 0.38:
        return None
    if radial_profile["peak_both_sides"] < 0.060:
        return None
    if radial_profile["peak_one_side"] < 0.095:
        return None

    observed_depth = samples["depth"][
        observed & np.isfinite(samples["depth"])
    ]
    rim_depth = _robust_median(observed_depth, minimum=6)
    if rim_depth is None:
        all_depth = samples["depth"][
            visible & edge_supported & np.isfinite(samples["depth"])
        ]
        rim_depth = _robust_median(all_depth, minimum=8)
    if rim_depth is None:
        return None
    focal = math.hypot(
        float(camera["fx"]) * math.cos(major_angle),
        float(camera["fy"]) * math.sin(major_angle),
    )
    diameter_m = float(major * rim_depth / max(focal, 1e-6))
    # A deliberately broad plausibility guard rejects tiny basket holes and
    # labels before expensive appearance work.  The ranking remains a soft
    # bell around the measured 255 mm specification.
    if not (0.135 <= diameter_m <= 0.405):
        return None
    if not (0.10 <= rim_depth <= 1.20):
        return None
    diameter_score = _bell(diameter_m, MOUTH_DIAMETER_M, 0.075)

    if len(observed_depth) >= 8:
        depth_mad = float(
            np.median(np.abs(observed_depth - np.median(observed_depth)))
        )
        depth_consistency = float(math.exp(-depth_mad / 0.035))
        depth_fraction = float(
            np.mean(observed & np.isfinite(samples["depth"]))
            / max(np.mean(observed), 1e-6)
        )
    else:
        depth_mad = None
        depth_consistency = 0.30
        depth_fraction = 0.0

    interior = _interior_points(ellipse)
    inner_visible, inner_ok = _sample(
        (ignore == 0).astype(np.uint8), interior, default=0
    )
    inner_ok &= inner_visible > 0
    inner_v, _ = _sample(maps["value"], interior)
    inner_blue, _ = _sample(maps["blue"], interior)
    inner_s, _ = _sample(maps["saturation"], interior)
    inner_z, _ = _sample(z, interior, default=np.nan)
    if int(inner_ok.sum()) < 25:
        return None
    inner_value = float(np.median(inner_v[inner_ok]))
    cavity_blue = float(np.mean(inner_blue[inner_ok]))
    cavity_dark = float(
        np.mean(
            (inner_v[inner_ok] <= max(145.0, 0.84 * np.median(samples["value"][visible])))
            & (inner_s[inner_ok] <= 190)
        )
    )
    ring_value = float(np.percentile(samples["value"][visible], 72))
    contrast = ring_value - inner_value
    valid_inner_z = inner_z[inner_ok & np.isfinite(inner_z)]
    inner_depth = _robust_median(valid_inner_z, minimum=12)
    depth_step = (
        None if inner_depth is None else float(inner_depth - rim_depth)
    )
    depth_step_score = (
        0.35
        if depth_step is None
        else max(
            _smooth(depth_step, 0.004, 0.065),
            0.55 * _smooth(abs(depth_step), 0.008, 0.075),
        )
    )

    body_results = []
    for sign in (-1.0, 1.0):
        body_points = _body_lobe_points(ellipse, sign)
        body_blue, body_ok = _sample(maps["blue"], body_points)
        body_v, _ = _sample(maps["value"], body_points)
        body_blocked, _ = _sample(ignore, body_points, default=255)
        body_z, _ = _sample(z, body_points, default=np.nan)
        body_ok &= body_blocked == 0
        if int(body_ok.sum()) < 35:
            body_results.append(
                {
                    "score": 0.0,
                    "blue": 0.0,
                    "valid_fraction": 0.0,
                    "depth": None,
                    "sign": sign,
                }
            )
            continue
        blue_fraction = float(np.mean(body_blue[body_ok]))
        valid_z = body_z[body_ok & np.isfinite(body_z)]
        valid_fraction = float(len(valid_z) / max(int(body_ok.sum()), 1))
        body_depth = _robust_median(valid_z, minimum=12)
        value_spread = float(
            np.percentile(body_v[body_ok], 85)
            - np.percentile(body_v[body_ok], 15)
        )
        solid_score = 1.0 - _smooth(value_spread, 70.0, 150.0)
        score = (
            0.68 * _smooth(blue_fraction, 0.025, 0.28)
            + 0.20 * _smooth(valid_fraction, 0.12, 0.72)
            + 0.12 * solid_score
        )
        body_results.append(
            {
                "score": float(score),
                "blue": blue_fraction,
                "valid_fraction": valid_fraction,
                "depth": body_depth,
                "sign": sign,
            }
        )
    body = max(body_results, key=lambda result: result["score"])
    body_other = min(body_results, key=lambda result: result["score"])
    body_blue_asymmetry = float(body["blue"] - body_other["blue"])
    object_support = max(
        _smooth(cavity_blue, 0.015, 0.20),
        body["score"],
    )
    if object_support < 0.13:
        return None
    contrast_score = _smooth(contrast, 7.0, 82.0)
    if contrast < 4.0 and body["score"] < 0.40:
        return None

    # Perspective supplies two complementary semantic cases.  In a flat
    # side-view ellipse the visible evidence is a strong, solid grey-blue body
    # lobe on one normal side.  In a rounder overhead ellipse the inner cavity
    # itself must be grey-blue/dark.  This is deliberately orientation-free:
    # both normal directions are evaluated above.
    if contrast < 30.0:
        return None
    if aspect < 0.42:
        if body["blue"] < 0.48 or body["score"] < 0.70:
            return None
        if body_blue_asymmetry < 0.24 or cavity_dark < 0.30:
            return None
    else:
        if cavity_blue < 0.34:
            return None

    scene_height = None
    scene_height_score = None
    if isinstance(scene, dict) and scene.get("height") is not None:
        heights, height_ok = _sample(
            np.asarray(scene["height"], np.float32),
            samples["points"],
            default=np.nan,
        )
        height_values = heights[observed & height_ok & np.isfinite(heights)]
        scene_height = _robust_median(height_values, minimum=8)
        if scene_height is not None:
            scene_height_score = max(
                _bell(scene_height, MOUTH_HEIGHT_M, 0.085),
                0.35 * _bell(abs(scene_height), MOUTH_HEIGHT_M, 0.13),
            )

    edge_score = (
        0.52 * _smooth(edge_support, 0.20, 0.82)
        + 0.30 * _smooth(edge_coverage, 0.20, 0.82)
        + 0.18 * _smooth(edge_longest, 0.07, 0.38)
    )
    bright_score = (
        0.40 * _smooth(white_support, 0.15, 0.72)
        + 0.25 * _smooth(observed_coverage, 0.10, 0.66)
        + 0.20 * _smooth(radial_profile["core"], 0.38, 0.78)
        + 0.15
        * _smooth(radial_profile["peak_both_sides"], 0.06, 0.42)
    )
    cavity_score = max(
        0.62 * _smooth(cavity_blue, 0.015, 0.22)
        + 0.38 * _smooth(cavity_dark, 0.20, 0.82),
        body["score"],
    )
    score = (
        0.25 * edge_score
        + 0.20 * bright_score
        + 0.18 * diameter_score
        + 0.13 * contrast_score
        + 0.12 * cavity_score
        + 0.07 * depth_consistency
        + 0.05 * depth_step_score
    )
    if scene_height_score is not None:
        # Scene geometry is useful corroboration, never an obligatory camera
        # pose assumption.  A poor plane fit therefore cannot veto RGB-D.
        score = 0.91 * score + 0.09 * scene_height_score
    # Very small, barely observed conics are the dominant false-positive mode
    # on basket perforations and printed box labels.
    score *= 0.78 + 0.22 * _smooth(major, 52.0, 150.0)
    if score < 0.535:
        return None

    border = np.minimum.reduce(
        [
            samples["points"][:, 0],
            samples["points"][:, 1],
            image_w - 1.0 - samples["points"][:, 0],
            image_h - 1.0 - samples["points"][:, 1],
        ]
    )
    graspable = (
        observed
        & np.isfinite(samples["depth"])
        & (border >= max(3.0, 0.006 * major))
    )
    if recognition_only:
        clearance_values = None
        required_gripper_clearance = None
    else:
        clearance_values, _ = _sample(
            ignore_clearance, samples["points"], default=0.0
        )
        required_gripper_clearance = max(
            8.0, GRIPPER_CLEARANCE_M * major / MOUTH_DIAMETER_M
        )
        graspable &= clearance_values >= required_gripper_clearance
    grasp_point = None
    grasp_score_value = None
    grasp_clearance = None
    if np.any(graspable):
        depth_delta = np.abs(samples["depth"] - rim_depth)
        grasp_score = (
            0.39 * np.exp(-samples["distances"] / max(1.0, 0.0045 * major))
            + 0.29 * samples["white"]
            + 0.21 * np.exp(-depth_delta / 0.035)
            + 0.11 * np.clip(border / max(16.0, 0.055 * major), 0.0, 1.0)
        )
        if not recognition_only:
            grasp_score = (
                0.87 * grasp_score
                + 0.13
                * np.clip(
                    clearance_values / max(20.0, 0.08 * major),
                    0.0,
                    1.0,
                )
            )
        grasp_score[~graspable] = -1.0
        grasp_index = int(np.argmax(grasp_score))
        grasp_point = [
            int(
                np.clip(
                    round(float(samples["points"][grasp_index, 0])),
                    0,
                    image_w - 1,
                )
            ),
            int(
                np.clip(
                    round(float(samples["points"][grasp_index, 1])),
                    0,
                    image_h - 1,
                )
            ),
        ]
        grasp_score_value = float(grasp_score[grasp_index])
        if clearance_values is not None:
            grasp_clearance = float(clearance_values[grasp_index])

    return {
        "ellipse": ellipse,
        "score": float(np.clip(score, 0.0, 1.0)),
        "rim_depth": float(rim_depth),
        "diameter_m": diameter_m,
        "grasp_point": grasp_point,
        "debug": {
            "proposal_source": source,
            "major_px": float(major),
            "minor_px": float(minor),
            "aspect": float(aspect),
            "edge_support": edge_support,
            "edge_coverage": edge_coverage,
            "edge_longest_arc": edge_longest,
            "white_support": white_support,
            "radial_white_core": radial_profile["core"],
            "radial_white_inner": radial_profile["inner"],
            "radial_white_outer": radial_profile["outer"],
            "radial_white_peak": radial_profile["peak_both_sides"],
            "radial_white_one_side_peak": radial_profile["peak_one_side"],
            "radial_white_profile": radial_profile["means"],
            "observed_support": observed_support,
            "observed_coverage": observed_coverage,
            "observed_longest_arc": observed_longest,
            "rim_inner_contrast": float(contrast),
            "cavity_blue": cavity_blue,
            "cavity_dark": cavity_dark,
            "body_score": float(body["score"]),
            "body_blue": float(body["blue"]),
            "opposite_body_blue": float(body_other["blue"]),
            "body_blue_asymmetry": body_blue_asymmetry,
            "body_side": int(body["sign"]),
            "depth_valid_fraction": depth_fraction,
            "depth_mad_m": depth_mad,
            "inner_depth_m": inner_depth,
            "inner_minus_rim_depth_m": depth_step,
            "scene_rim_height_m": scene_height,
            "scene_height_score": scene_height_score,
            "grasp_score": grasp_score_value,
            "grasp_is_observed": grasp_point is not None,
            "grasp_ignore_clearance_px": grasp_clearance,
            "required_gripper_clearance_px": (
                None
                if required_gripper_clearance is None
                else float(required_gripper_clearance)
            ),
            "seed_point_count": int(len(seed)),
        },
    }


def _candidate_duplicate(first: dict, second: dict) -> bool:
    first_center = np.asarray(first["ellipse"][0], np.float64)
    second_center = np.asarray(second["ellipse"][0], np.float64)
    first_major, first_minor, _ = _major_geometry(first["ellipse"])
    second_major, second_minor, _ = _major_geometry(second["ellipse"])
    scale = max(first_major, second_major)
    return bool(
        np.linalg.norm(first_center - second_center) <= 0.13 * scale
        and abs(first_major - second_major) <= 0.28 * scale
        and abs(first_minor - second_minor) <= 0.38 * scale
    )


def detect_trash_rims(
    rgb: np.ndarray,
    z: np.ndarray,
    camera: dict,
    ignore: np.ndarray,
    scene: dict | None = None,
    recognition_only: bool = True,
) -> list[dict]:
    """Detect the single best 255 mm trash-can rim in aligned wrist RGB-D.

    Parameters
    ----------
    rgb:
        ``uint8`` BGR image.
    z:
        Aligned depth in metres. Invalid pixels must be ``NaN``.
    camera:
        Mapping containing at least ``fx``, ``fy``, ``cx`` and ``cy``.
    ignore:
        Nonzero pixels belong to the robot/gripper and cannot provide evidence
        or a grasp point.
    scene:
        Optional output of :func:`wrist_rgbd.scene.build_scene_geometry`.

    Returns
    -------
    list[dict]
        Empty, or a one-element Top-1 list.  The ellipse may geometrically
        complete hidden arcs; ``grasp_point`` is always on a directly observed,
        depth-valid, non-ignored section of the rim.
    """
    rgb = np.asarray(rgb)
    z = np.asarray(z, np.float32)
    ignore = np.asarray(ignore)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 HxWx3 BGR")
    if z.ndim != 2 or z.shape != rgb.shape[:2]:
        raise ValueError("z must be aligned HxW depth in metres")
    if ignore.shape != z.shape:
        raise ValueError("ignore must have the same HxW shape as rgb")
    for key in ("fx", "fy", "cx", "cy"):
        if key not in camera or not np.isfinite(float(camera[key])):
            raise ValueError(f"camera missing finite {key}")
    if float(camera["fx"]) <= 0.0 or float(camera["fy"]) <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    ignore = (ignore > 0).astype(np.uint8)
    valid_z = np.isfinite(z) & (z > 0.045) & (z < 6.0)
    z = z.copy()
    z[~valid_z] = np.nan
    appearance = _appearance_response(rgb)
    (
        _,
        edge_maps,
        edge_distance,
        white_strict,
        adjacency_masks,
    ) = _proposal_maps(rgb, ignore, appearance)

    proposals = _edge_proposals(edge_maps)
    proposals.extend(
        _body_guided_proposals(appearance, z, camera, ignore)
    )
    proposals.extend(_white_component_proposals(white_strict, "white"))
    for index, mask in enumerate(adjacency_masks):
        proposals.extend(
            _white_component_proposals(mask, f"white_blue_adj_{index}")
        )
    proposals = _deduplicate_proposals(proposals, rgb.shape)

    if recognition_only:
        ignore_clearance = None
    else:
        free = (ignore == 0).astype(np.uint8)
        ignore_clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    accepted = []
    for proposal in proposals:
        item = _candidate_features(
            proposal,
            rgb,
            z,
            camera,
            ignore,
            ignore_clearance,
            appearance,
            edge_distance,
            scene,
            recognition_only=recognition_only,
        )
        if item is None:
            continue
        accepted.append(item)

    if not accepted:
        return []
    accepted.sort(
        key=lambda item: (
            item["score"],
            item["debug"]["observed_coverage"],
            -abs(item["diameter_m"] - MOUTH_DIAMETER_M),
            item["debug"]["major_px"],
        ),
        reverse=True,
    )
    # Cluster only after scoring.  Inside each concentric cluster, choose the
    # largest strongly supported member: the task specification is the 255 mm
    # *outer* lip, not the smaller inner cavity boundary.  A tight score window
    # prevents a weak whole-body conic from winning merely because it is large.
    clusters = []
    for item in accepted:
        destination = None
        for cluster in clusters:
            if _candidate_duplicate(item, cluster[0]):
                destination = cluster
                break
        if destination is None:
            clusters.append([item])
        else:
            destination.append(item)
    unique = []
    for cluster in clusters:
        best_score = max(item["score"] for item in cluster)
        eligible = [
            item for item in cluster if item["score"] >= best_score - 0.018
        ]
        representative = max(
            eligible,
            key=lambda item: (
                item["debug"]["major_px"] * item["debug"]["minor_px"],
                -abs(item["diameter_m"] - MOUTH_DIAMETER_M),
                item["score"],
            ),
        )
        representative["debug"]["concentric_cluster_size"] = len(cluster)
        representative["debug"]["cluster_best_score"] = float(best_score)
        unique.append(representative)
    accepted = sorted(
        unique,
        key=lambda item: (
            item["score"],
            item["debug"]["observed_coverage"],
            -abs(item["diameter_m"] - MOUTH_DIAMETER_M),
        ),
        reverse=True,
    )
    # The task scene contains at most one relevant trash can and the downstream
    # arm consumes one grasp point.  Returning Top-1 here prevents a secondary
    # label, cable loop, or inner rim from leaking into the public result.
    winner = dict(accepted[0])
    winner["debug"] = dict(winner["debug"])
    refinement = _refine_oblique_rim(
        winner["ellipse"], appearance, ignore
    )
    if refinement is not None:
        refined_ellipse, refinement_debug = refinement
        refined_grasp = _observed_grasp_on_ellipse(
            refined_ellipse,
            appearance,
            edge_distance,
            z,
            ignore,
            ignore_clearance,
            winner.get("rim_depth"),
            recognition_only=recognition_only,
        )
        if refined_grasp is not None:
            grasp_point, grasp_debug = refined_grasp
            major, minor, _ = _major_geometry(refined_ellipse)
            winner["ellipse"] = refined_ellipse
            winner["grasp_point"] = grasp_point
            winner["debug"]["geometry_refinement"] = {
                "method": "white_lip_to_blue_body_profile",
                **refinement_debug,
            }
            winner["debug"]["major_px"] = float(major)
            winner["debug"]["minor_px"] = float(minor)
            winner["debug"]["aspect"] = float(
                minor / max(major, 1e-6)
            )
            winner["debug"].update(grasp_debug)
            winner["debug"]["grasp_is_observed"] = True
    winner["debug"]["raw_proposal_count"] = int(len(proposals))
    winner["debug"]["accepted_candidate_count"] = int(len(accepted))
    return [winner]


__all__ = ["detect_trash_rims"]
