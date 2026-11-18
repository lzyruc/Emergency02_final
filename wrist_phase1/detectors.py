from __future__ import annotations

import math

import cv2
import numpy as np

from .basket_grasp import find_basket_grasp
from .chair_geometry import detect_chair_candidates
from .trash_rim import detect_trash_rims
from .utils import (
    angle_distance,
    bbox_area,
    bbox_from_mask,
    bell,
    component_masks,
    crop_terminal,
    ellipse_angular_coverage,
    ellipse_points,
    expand_bbox,
    line_gradient_support,
    line_mask_support,
    line_normal,
    line_points,
    normalized_angle,
    retain_weak_components,
    robust_depth,
    safe_point,
    sample_image,
    smooth_ramp,
)


def _base_candidate(cls, confidence, bbox, mask, depth, geometry, debug):
    h, w = mask.shape
    area_score = float(np.clip(math.sqrt(max(1, bbox_area(bbox)) / (h * w)), 0, 1))
    if depth is None:
        depth_score = 0.45
    else:
        depth_score = float(np.clip(math.exp(-max(0.0, depth - 0.25) / 1.35), 0, 1))
    # Object identity must dominate cross-class arbitration.  A geometrically
    # convenient line is not evidence that the underlying class is correct
    # (for example, a cabinet or a floor marking can contain excellent
    # straight edges).  Depth retains enough weight to prefer the nearer
    # wrist-camera target when several real scene objects are visible.  Grasp
    # quality is deliberately excluded from object classification.
    selection = (
        0.76 * float(confidence)
        + 0.08 * area_score
        + 0.16 * depth_score
    )
    return {
        "class": cls,
        "confidence": float(np.clip(confidence, 0, 1)),
        "selection_score": float(np.clip(selection, 0, 1)),
        "bbox": [int(v) for v in bbox],
        "mask": mask,
        "depth_m": None if depth is None else float(depth),
        "grasp_geometry": geometry,
        "debug": debug,
    }


def _hough_segments(edge, min_length, max_gap=18, threshold=20):
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 720.0,
        threshold=max(10, int(threshold)),
        minLineLength=max(12, int(min_length)),
        maxLineGap=max(4, int(max_gap)),
    )
    if lines is None:
        return []
    segments = [tuple(float(v) for v in item) for item in lines.reshape(-1, 4)]
    segments.sort(
        key=lambda line: (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2,
        reverse=True,
    )
    return segments[:420]


def _dedupe_lines(items, midpoint_px=24, angle_deg=6):
    kept = []
    for item in sorted(items, key=lambda value: value["score"], reverse=True):
        mid = 0.5 * (item["p0"] + item["p1"])
        duplicate = False
        for other in kept:
            omid = 0.5 * (other["p0"] + other["p1"])
            if (
                np.linalg.norm(mid - omid) < midpoint_px
                and angle_distance(item["angle"], other["angle"])
                < math.radians(angle_deg)
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def _basket_grasp(rgb, mask, z, bbox, camera, ignore):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gradients = (
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    object_depth = robust_depth(z, mask)
    if object_depth is None:
        min_length = max(28, int(0.045 * math.hypot(w, h)))
    else:
        min_length = max(24, int(0.055 * camera["fx"] / object_depth))
    min_length = int(np.clip(min_length, 24, 170))

    color_edge = cv2.Canny((mask * 255).astype(np.uint8), 40, 120)
    rgb_edge = cv2.Canny(gray, 45, 135)
    edge = cv2.bitwise_or(color_edge, rgb_edge)
    roi = np.zeros_like(edge)
    roi[y1:y2, x1:x2] = 255
    edge[(roi == 0) | (ignore > 0)] = 0
    segments = _hough_segments(
        edge,
        min_length=min_length,
        max_gap=max(10, int(0.025 * max(x2 - x1, y2 - y1))),
        threshold=max(16, min_length // 3),
    )
    candidates = []
    normal_offset = max(4, int(round(0.008 * max(x2 - x1, y2 - y1))))
    for raw in segments:
        p0 = np.array(raw[:2], np.float32)
        p1 = np.array(raw[2:], np.float32)
        length = float(np.linalg.norm(p1 - p0))
        if length < min_length:
            continue
        points = line_points(p0, p1)
        line_blue, ok = sample_image(mask, points)
        if float(np.mean((line_blue > 0) & ok)) < 0.42:
            continue
        normal = line_normal(p0, p1)
        side_a, oka = sample_image(mask, points + normal_offset * normal)
        side_b, okb = sample_image(mask, points - normal_offset * normal)
        a = float(np.mean((side_a > 0) & oka))
        b = float(np.mean((side_b > 0) & okb))
        asym = abs(a - b)
        edge_strength, edge_coverage = line_gradient_support(
            gray, p0, p1, 2, gradients=gradients
        )
        zline, zok = sample_image(z, points, default=np.nan)
        valid_z = np.isfinite(zline) & zok
        zmed = float(np.nanmedian(zline[valid_z])) if valid_z.sum() >= 8 else object_depth
        length_m = (
            None
            if zmed is None
            else length * zmed / max(float(camera["fx"]), 1e-6)
        )
        if length_m is not None and not (0.045 <= length_m <= 0.44):
            continue
        angle = normalized_angle(p0, p1)
        score = (
            0.22 * smooth_ramp(length, min_length, 2.8 * min_length)
            + 0.24 * smooth_ramp(asym, 0.08, 0.62)
            + 0.19 * smooth_ramp(edge_coverage, 0.35, 0.92)
            + 0.14 * smooth_ramp(edge_strength, 55, 230)
            + 0.12 * float(np.mean(valid_z))
            + 0.09 * (1.0 - float(np.mean(ignore[
                np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1),
                np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1),
            ] > 0)))
        )
        candidates.append(
            {
                "p0": p0,
                "p1": p1,
                "length": length,
                "length_m": length_m,
                "angle": angle,
                "asymmetry": asym,
                "edge_strength": edge_strength,
                "edge_coverage": edge_coverage,
                "score": float(score),
            }
        )
    if not candidates:
        return None, {"raw_line_count": len(segments), "line_candidate_count": 0}

    # Divider free edges occur as a repeated parallel family. A single long
    # perpendicular beam is not an allowed fallback.
    for item in candidates:
        family = [
            other
            for other in candidates
            if angle_distance(item["angle"], other["angle"]) < math.radians(11)
        ]
        item["family_count"] = len(_dedupe_lines(family, midpoint_px=18, angle_deg=5))
        item["score"] += 0.18 * smooth_ramp(item["family_count"], 1.0, 4.0)
        if item["family_count"] <= 1:
            item["score"] *= 0.72
    candidates = _dedupe_lines(candidates)
    candidates.sort(
        key=lambda item: (
            item["family_count"] >= 2,
            item["score"],
            item["length"],
        ),
        reverse=True,
    )
    best = candidates[0]
    if best["score"] < 0.42:
        return None, {
            "raw_line_count": len(segments),
            "line_candidate_count": len(candidates),
            "best_score": round(best["score"], 4),
        }
    ys, xs = np.nonzero(mask)
    centroid = [float(np.mean(xs)), float(np.mean(ys))]
    p0, p1 = crop_terminal(best["p0"], best["p1"], centroid, fraction=0.36)
    if np.linalg.norm(p1 - p0) < 18:
        return None, {
            "raw_line_count": len(segments),
            "line_candidate_count": len(candidates),
            "reason": "terminal_too_short",
        }
    confidence = float(np.clip(0.50 + 0.48 * best["score"], 0, 0.99))
    geometry = {
        "type": "line",
        "name": "basket_divider_free_edge",
        "endpoints_uv": [safe_point(p0, mask.shape), safe_point(p1, mask.shape)],
        "confidence": confidence,
        "grasp_safe": True,
        "terminal_fraction": 0.36,
    }
    return geometry, {
        "raw_line_count": len(segments),
        "line_candidate_count": len(candidates),
        "family_count": best["family_count"],
        "full_line_length_px": round(best["length"], 2),
        "full_line_length_m": (
            None if best["length_m"] is None else round(best["length_m"], 4)
        ),
        "boundary_asymmetry": round(best["asymmetry"], 3),
        "best_line_score": round(best["score"], 4),
    }


def detect_baskets(rgb, z, camera, ignore, scene=None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = [channel.astype(np.int16) for channel in cv2.split(rgb)]
    bmr = b - r
    strong = (
        (h >= 88)
        & (h <= 132)
        & (s >= 78)
        & (v >= 42)
        & (bmr >= 24)
    )
    rel = bmr.astype(np.float32) / np.maximum(v.astype(np.float32), 1)
    weak = (
        (h >= 82)
        & (h <= 138)
        & (s >= 35)
        & (v >= 28)
        & (bmr >= 10)
        & (rel >= 0.16)
    )
    groundish = None
    colour_cast = False
    if scene is not None:
        scene_valid = np.asarray(scene["valid"])
        scene_height = np.asarray(scene["height"])
        groundish = (
            scene_valid
            & np.isfinite(scene_height)
            & (np.abs(scene_height) <= 0.040)
        )
        usable_global = ignore == 0
        global_weak_fraction = (
            float(np.mean(weak[usable_global]))
            if np.any(usable_global)
            else 0.0
        )
        usable_ground = groundish & usable_global
        ground_weak_fraction = (
            float(np.mean(weak[usable_ground]))
            if np.any(usable_ground)
            else 0.0
        )
        colour_cast = bool(
            global_weak_fraction > 0.65
            or ground_weak_fraction > 0.48
        )
        # White-balance shifts can make a large grey floor look weakly blue.
        # The basket is a raised 303 x 257 x 308 mm five-divider structure;
        # planar ground pixels are removed only when that cast is measured.
        # Ordinary frames retain the original v5 HSV component path exactly.
        if colour_cast:
            strong[groundish] = False
            weak[groundish] = False

        # Anchor colour support to a physically raised blue component.  Invalid
        # depth holes in the perforated plastic are recovered within a metric
        # halo only when the image is demonstrably under a global cool cast.
        # Ordinary images keep their original v5 colour component extent.
        if colour_cast:
            raised = (
                weak
                & scene_valid
                & np.isfinite(scene_height)
                & (scene_height >= 0.065)
                & (scene_height <= 0.52)
                & (ignore == 0)
            ).astype(np.uint8)
        else:
            raised = np.zeros_like(weak, dtype=np.uint8)
        raised = cv2.morphologyEx(
            raised,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        raised[ignore > 0] = 0
        anchors = np.zeros_like(raised)
        for component, area in component_masks(
            raised,
            max(180, int(0.00020 * raised.size)),
        ):
            bbox = bbox_from_mask(component)
            depth = robust_depth(z, component, 35)
            if bbox is None or depth is None:
                continue
            apparent_long_m = (
                max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                * depth
                / camera["fx"]
            )
            strong_count = int(np.sum(strong[component]))
            if (
                strong_count >= max(45, int(0.00005 * raised.size))
                and 0.08 <= apparent_long_m <= 0.85
            ):
                anchors[component] = 1
        if np.any(anchors):
            anchor_depth = robust_depth(z, anchors > 0, 35)
            focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
            radius = int(
                np.clip(
                    math.ceil(
                        0.050
                        * focal
                        / max(
                            0.08,
                            float(anchor_depth)
                            if anchor_depth is not None
                            else 0.65,
                        )
                    ),
                    6,
                    140,
                )
            )
            distance = cv2.distanceTransform(
                (anchors == 0).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            allowed = distance <= float(radius)
            strong[~allowed] = False
            weak[~allowed] = False
    strong[ignore > 0] = False
    weak[ignore > 0] = False
    weak = cv2.morphologyEx(
        weak.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    # Closing can paint across a supplied/automatic robot mask. Reapply it
    # before and after retention so bbox/depth/colour never use gripper or
    # white-balance-shifted ground.
    weak[ignore > 0] = 0
    if groundish is not None and colour_cast:
        weak[groundish] = 0
    mask = retain_weak_components(
        weak,
        strong,
        min_area=max(220, int(0.00035 * rgb.shape[0] * rgb.shape[1])),
        min_strong=max(45, int(0.00006 * rgb.shape[0] * rgb.shape[1])),
    )
    mask[ignore > 0] = 0
    if groundish is not None and colour_cast:
        mask[groundish] = 0
    out = []
    for component, area in component_masks(
        mask, max(220, int(0.00035 * mask.size))
    ):
        bbox = bbox_from_mask(component)
        if bbox is None:
            continue
        color_purity = float(np.mean(strong[component]))
        blue_rel = float(np.median(rel[component]))
        depth = robust_depth(z, component, 35)
        width_px = bbox[2] - bbox[0]
        height_px = bbox[3] - bbox[1]
        if depth is None:
            apparent_m = None
            size_score = smooth_ramp(max(width_px, height_px), 35, 180)
        else:
            apparent_m = max(width_px, height_px) * depth / camera["fx"]
            size_score = max(
                bell(apparent_m, 0.31, 0.13),
                bell(apparent_m, 0.43, 0.19),
            )
        contours, hierarchy = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        holes = 0
        if hierarchy is not None:
            holes = int(np.sum(hierarchy[0, :, 3] >= 0))
        hole_score = smooth_ramp(holes, 2, 18)
        confidence = (
            0.42 * smooth_ramp(color_purity, 0.18, 0.72)
            + 0.22 * smooth_ramp(blue_rel, 0.28, 0.70)
            + 0.20 * size_score
            + 0.11 * hole_score
            + 0.05 * smooth_ramp(area / mask.size, 0.001, 0.03)
        )
        if confidence < 0.48:
            continue
        geometry, grasp_debug = find_basket_grasp(
            rgb,
            z,
            component.astype(np.uint8),
            bbox,
            camera,
            ignore,
            scene=scene,
        )
        out.append(
            _base_candidate(
                "basket",
                confidence,
                bbox,
                component.astype(np.uint8),
                depth,
                geometry,
                {
                    "area_px": int(area),
                    "color_purity": round(color_purity, 4),
                    "blue_relative": round(blue_rel, 4),
                    "hole_count": holes,
                    "apparent_size_m": (
                        None if apparent_m is None else round(apparent_m, 4)
                    ),
                    "grasp": grasp_debug,
                },
            )
        )
    return sorted(out, key=lambda item: item["selection_score"], reverse=True)[:3]


def _ellipse_mask(shape, ellipse, scale):
    mask = np.zeros(shape, np.uint8)
    (cx, cy), (width, height), angle = ellipse
    cv2.ellipse(
        mask,
        ((float(cx), float(cy)), (float(width * scale), float(height * scale)), float(angle)),
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask > 0


def _trash_grasp_point(ellipse, rgb, z, ignore):
    points = ellipse_points(ellipse, count=max(360, int(max(ellipse[1]) * 2.5)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad = cv2.magnitude(gx, gy)
    edge, ok = sample_image(grad, points)
    depth, _ = sample_image(z, points, default=np.nan)
    blocked, _ = sample_image(ignore, points, default=255)
    h, w = gray.shape
    border = np.minimum.reduce(
        [points[:, 0], points[:, 1], w - 1 - points[:, 0], h - 1 - points[:, 1]]
    )
    valid_depth = np.isfinite(depth)
    if valid_depth.any():
        d0 = float(np.nanpercentile(depth, 20))
        depth_score = np.exp(-np.maximum(0.0, depth - d0) / 0.08)
        depth_score[~valid_depth] = 0.35
    else:
        depth_score = np.full(len(points), 0.45)
    score = (
        0.40 * np.clip(edge / 260.0, 0, 1)
        + 0.32 * depth_score
        + 0.18 * np.clip(border / 35.0, 0, 1)
        + 0.10 * (points[:, 1] / max(1, h - 1))
    )
    score[(blocked > 0) | (~ok)] = -1
    index = int(np.argmax(score))
    if score[index] < 0:
        return None
    return safe_point(points[index], gray.shape)


def detect_trash(rgb, z, camera, ignore):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(rgb)
    min_channel = np.minimum(np.minimum(b, g), r)
    white = (
        (v >= 135)
        & (s <= 92)
        & (min_channel >= 105)
    ).astype(np.uint8)
    white[ignore > 0] = 0
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 45, 145)
    edge_dist = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 3)
    candidates = []
    diag = math.hypot(*rgb.shape[:2])
    for contour in contours:
        if len(contour) < 28:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 0.055 * diag:
            continue
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (ew, eh), angle = ellipse
        major, minor = max(ew, eh), min(ew, eh)
        if major < 28 or major > 1.35 * diag or minor < 10:
            continue
        aspect = minor / max(major, 1e-6)
        if not (0.08 <= aspect <= 1.01):
            continue
        points = ellipse_points(ellipse)
        white_values, ok = sample_image(white, points)
        white_support = float(np.mean((white_values > 0) & ok))
        distances, _ = sample_image(edge_dist, points, default=99)
        edge_support = float(np.mean(distances <= max(2.2, 0.006 * major)))
        coverage = ellipse_angular_coverage(contour.reshape(-1, 2), ellipse, 48)
        inner = _ellipse_mask(white.shape, ellipse, 0.76)
        ring = _ellipse_mask(white.shape, ellipse, 1.03) & ~_ellipse_mask(
            white.shape, ellipse, 0.86
        )
        inner &= ignore == 0
        ring &= ignore == 0
        if inner.sum() < 80 or ring.sum() < 50:
            continue
        inner_v = float(np.median(v[inner]))
        ring_v = float(np.median(v[ring]))
        contrast = ring_v - inner_v
        inner_h = h[inner]
        inner_s = s[inner]
        muted_blue = float(
            np.mean(
                ((inner_h >= 78) & (inner_h <= 128) & (inner_s <= 155))
                | (inner_s <= 48)
            )
        )
        rim_depth = robust_depth(z, ring, 35)
        if rim_depth is None:
            diameter_m = None
            size_score = 0.48
        else:
            effective_f = math.hypot(
                camera["fx"] * math.cos(math.radians(angle)),
                camera["fy"] * math.sin(math.radians(angle)),
            )
            diameter_m = major * rim_depth / max(effective_f, 1e-6)
            size_score = bell(diameter_m, 0.255, 0.085)
        score = (
            0.24 * smooth_ramp(white_support, 0.24, 0.78)
            + 0.24 * smooth_ramp(edge_support, 0.38, 0.90)
            + 0.14 * smooth_ramp(coverage, 0.48, 0.92)
            + 0.16 * smooth_ramp(contrast, 18, 95)
            + 0.12 * muted_blue
            + 0.10 * size_score
        )
        if score < 0.53:
            continue
        candidates.append(
            {
                "ellipse": ellipse,
                "score": score,
                "white_support": white_support,
                "edge_support": edge_support,
                "coverage": coverage,
                "contrast": contrast,
                "muted_blue": muted_blue,
                "rim_depth": rim_depth,
                "diameter_m": diameter_m,
            }
        )
    # Prefer the outer member of near-concentric duplicates when its evidence is close.
    candidates.sort(
        key=lambda item: (
            item["score"] + 0.04 * smooth_ramp(max(item["ellipse"][1]), 80, 420)
        ),
        reverse=True,
    )
    kept = []
    for item in candidates:
        center = np.array(item["ellipse"][0])
        duplicate = False
        for other in kept:
            oc = np.array(other["ellipse"][0])
            scale = max(max(item["ellipse"][1]), max(other["ellipse"][1]))
            if np.linalg.norm(center - oc) < 0.06 * scale:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    out = []
    for item in kept[:3]:
        ellipse = item["ellipse"]
        points = ellipse_points(ellipse)
        bbox = [
            int(np.clip(np.floor(points[:, 0].min()), 0, rgb.shape[1] - 1)),
            int(np.clip(np.floor(points[:, 1].min()), 0, rgb.shape[0] - 1)),
            int(np.clip(np.ceil(points[:, 0].max()) + 1, 1, rgb.shape[1])),
            int(np.clip(np.ceil(points[:, 1].max()) + 1, 1, rgb.shape[0])),
        ]
        point = _trash_grasp_point(ellipse, rgb, z, ignore)
        confidence = float(np.clip(0.48 + 0.50 * item["score"], 0, 0.995))
        geometry = {
            "type": "ellipse",
            "name": "top_rim",
            "ellipse": {
                "center": [round(float(v), 3) for v in ellipse[0]],
                "axes": [round(float(v), 3) for v in ellipse[1]],
                "angle_deg": round(float(ellipse[2]), 4),
            },
            "grasp_point_uv": point,
            "confidence": confidence,
            "grasp_safe": point is not None,
        }
        mask = _ellipse_mask(z.shape, ellipse, 1.08).astype(np.uint8)
        out.append(
            _base_candidate(
                "trash_can",
                confidence,
                bbox,
                mask,
                item["rim_depth"],
                geometry,
                {
                    "white_support": round(item["white_support"], 4),
                    "edge_support": round(item["edge_support"], 4),
                    "angular_coverage": round(item["coverage"], 4),
                    "rim_inner_contrast": round(item["contrast"], 2),
                    "muted_blue_fraction": round(item["muted_blue"], 4),
                    "diameter_m": (
                        None
                        if item["diameter_m"] is None
                        else round(item["diameter_m"], 4)
                    ),
                },
            )
        )
    return out


def detect_trash_metric(
    rgb,
    z,
    camera,
    ignore,
    scene=None,
    recognition_only=True,
):
    """Wrap the dedicated partial-rim detector in the common object schema."""
    out = []
    # Local rim/body depth is authoritative.  A ground plane is deliberately
    # not injected here because close wrist views can make the bucket or a box
    # face the dominant plane; the dedicated detector already validates the
    # 255 mm diameter without relying on camera pose.
    for item in detect_trash_rims(
        rgb,
        z,
        camera,
        ignore,
        scene=None,
        recognition_only=recognition_only,
    ):
        ellipse = item["ellipse"]
        points = ellipse_points(ellipse)
        bbox = [
            int(np.clip(np.floor(points[:, 0].min()), 0, rgb.shape[1] - 1)),
            int(np.clip(np.floor(points[:, 1].min()), 0, rgb.shape[0] - 1)),
            int(np.clip(np.ceil(points[:, 0].max()) + 1, 1, rgb.shape[1])),
            int(np.clip(np.ceil(points[:, 1].max()) + 1, 1, rgb.shape[0])),
        ]
        confidence = float(np.clip(item["score"], 0.0, 0.995))
        point = item.get("grasp_point")
        geometry = {
            "type": "ellipse",
            "name": "top_rim",
            "ellipse": {
                "center": [round(float(v), 3) for v in ellipse[0]],
                "axes": [round(float(v), 3) for v in ellipse[1]],
                "angle_deg": round(float(ellipse[2]), 4),
            },
            "grasp_point_uv": point,
            "confidence": confidence,
            "grasp_safe": point is not None,
            "inferred": False,
        }
        mask = _ellipse_mask(z.shape, ellipse, 1.06).astype(np.uint8)
        out.append(
            _base_candidate(
                "trash_can",
                confidence,
                bbox,
                mask,
                item.get("rim_depth"),
                geometry,
                {
                    "dedicated_partial_rim": True,
                    "diameter_m": (
                        None
                        if item.get("diameter_m") is None
                        else round(float(item["diameter_m"]), 5)
                    ),
                    **item.get("debug", {}),
                },
            )
        )
    return out


def _box_grasp(
    rgb,
    mask,
    bbox,
    z,
    camera,
    ignore,
    scene=None,
    recognition_only=True,
):
    mask = mask.copy()
    mask[ignore > 0] = 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gradients = (
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    x1, y1, x2, y2 = bbox
    edge = cv2.Canny(gray, 45, 140)
    roi = np.zeros_like(edge)
    roi[y1:y2, x1:x2] = 255
    edge[(roi == 0) | (ignore > 0)] = 0
    min_len = max(28, int(0.16 * min(x2 - x1, y2 - y1)))
    lines = _hough_segments(edge, min_len, max_gap=20, threshold=20)
    scored = []
    object_depth = robust_depth(z, mask)
    if recognition_only:
        ignore_clearance = None
        required_clearance = None
    else:
        ignore_clearance = cv2.distanceTransform(
            (ignore == 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        if object_depth is None:
            required_clearance = max(10.0, 0.012 * math.hypot(*mask.shape))
        else:
            mean_focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
            required_clearance = float(
                np.clip(0.022 * mean_focal / object_depth, 8.0, 55.0)
            )
    if scene is not None:
        object_heights = np.asarray(scene["height"])[
            (mask > 0) & np.asarray(scene["valid"])
        ]
        object_heights = object_heights[np.isfinite(object_heights)]
        object_top = (
            float(np.percentile(object_heights, 96))
            if len(object_heights) >= 40
            else None
        )
    else:
        object_top = None
    for raw in lines:
        p0 = np.array(raw[:2], np.float32)
        p1 = np.array(raw[2:], np.float32)
        length = float(np.linalg.norm(p1 - p0))
        probe = line_points(p0, p1, max(32, int(round(length))))
        clearance_p10 = None
        if not recognition_only:
            clearance, clearance_ok = sample_image(
                ignore_clearance, probe, default=0.0
            )
            clearance = clearance[clearance_ok & np.isfinite(clearance)]
            if len(clearance) < max(12, int(round(0.40 * len(probe)))):
                continue
            clearance_p10 = float(np.percentile(clearance, 10.0))
            if clearance_p10 < required_clearance:
                continue
        support = line_mask_support(mask, p0, p1, 4)
        strength, coverage = line_gradient_support(
            gray, p0, p1, 2, gradients=gradients
        )
        mid = 0.5 * (p0 + p1)
        topness = 1.0 - np.clip((mid[1] - y1) / max(1, y2 - y1), 0, 1)
        if object_top is not None:
            line_height, line_ok = sample_image(
                scene["height"], line_points(p0, p1), default=np.nan
            )
            line_height = line_height[np.isfinite(line_height) & line_ok]
            if len(line_height) >= 8:
                height_median = float(np.median(line_height))
                height_span = float(
                    np.percentile(line_height, 90)
                    - np.percentile(line_height, 10)
                )
                height_score = bell(height_median - object_top, 0.0, 0.045)
                level_score = bell(height_span, 0.0, 0.035)
            else:
                height_median = None
                height_span = None
                height_score = 0.30
                level_score = 0.35
        else:
            height_median = None
            height_span = None
            height_score = float(topness)
            level_score = 0.50
        if object_depth is None:
            length_m = None
            size_score = smooth_ramp(length, min_len, 3 * min_len)
        else:
            length_m = length * object_depth / camera["fx"]
            size_score = max(
                bell(length_m, 0.35, 0.20),
                bell(length_m, 0.50, 0.22),
            )
        score = (
            0.25 * smooth_ramp(support, 0.28, 0.82)
            + 0.21 * smooth_ramp(coverage, 0.30, 0.88)
            + 0.13 * smooth_ramp(strength, 50, 220)
            + 0.15 * size_score
            + 0.18 * height_score
            + 0.08 * level_score
        )
        if score >= 0.43:
            scored.append(
                {
                    "p0": p0,
                    "p1": p1,
                    "score": float(score),
                    "length": length,
                    "length_m": length_m,
                    "angle": normalized_angle(p0, p1),
                    "height_m": height_median,
                    "height_span_m": height_span,
                    "height_score": height_score,
                    "ignore_clearance_px": clearance_p10,
                }
            )
    scored = _dedupe_lines(scored)
    if not scored:
        return None, {"raw_line_count": len(lines), "line_candidate_count": 0}
    best = max(scored, key=lambda item: item["score"])
    p0 = best["p0"] * 0.78 + best["p1"] * 0.22
    p1 = best["p0"] * 0.22 + best["p1"] * 0.78
    geometry = {
        "type": "line",
        "name": "box_visible_free_edge",
        "endpoints_uv": [safe_point(p0, mask.shape), safe_point(p1, mask.shape)],
        "confidence": float(np.clip(0.48 + 0.49 * best["score"], 0, 0.98)),
        "grasp_safe": True,
    }
    return geometry, {
        "raw_line_count": len(lines),
        "line_candidate_count": len(scored),
        "best_line_score": round(best["score"], 4),
        "full_line_length_px": round(best["length"], 2),
        "full_line_length_m": (
            None if best["length_m"] is None else round(best["length_m"], 4)
        ),
        "line_height_m": (
            None if best["height_m"] is None else round(best["height_m"], 4)
        ),
        "line_height_span_m": (
            None
            if best["height_span_m"] is None
            else round(best["height_span_m"], 4)
        ),
        "line_height_score": round(float(best["height_score"]), 4),
        "ignore_clearance_px": (
            None
            if best["ignore_clearance_px"] is None
            else round(float(best["ignore_clearance_px"]), 3)
        ),
        "required_clearance_px": (
            None
            if required_clearance is None
            else round(float(required_clearance), 3)
        ),
    }


def detect_boxes(
    rgb,
    z,
    camera,
    ignore,
    scene=None,
    recognition_only=True,
):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = [channel.astype(np.int16) for channel in cv2.split(rgb)]
    warm = (
        (h >= 4)
        & (h <= 39)
        & (s >= 18)
        & (v >= 48)
        & (r >= b + 5)
        & (g >= b - 4)
    ).astype(np.uint8)
    warm[ignore > 0] = 0
    if scene is not None:
        groundish = (
            np.asarray(scene["valid"])
            & np.isfinite(scene["height"])
            & (np.abs(scene["height"]) <= 0.035)
        )
        warm[groundish] = 0
    warm = cv2.morphologyEx(
        warm,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Morphology may bridge across a removed gripper silhouette.  Re-apply the
    # safety mask before connected components and all depth statistics.
    warm[ignore > 0] = 0
    out = []
    for component_index, (component, area) in enumerate(
        component_masks(warm, max(500, int(0.00065 * warm.size)))
    ):
        if component_index >= 8:
            break
        bbox = bbox_from_mask(component)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        width_px, height_px = x2 - x1, y2 - y1
        if min(width_px, height_px) < 18:
            continue
        depth = robust_depth(z, component, 30)
        scene_debug = None
        height_score = 0.52
        footprint_score = None
        if scene is not None:
            metric = component & np.asarray(scene["valid"])
            heights = np.asarray(scene["height"])[metric]
            heights = heights[np.isfinite(heights)]
            if len(heights) >= 40:
                height_p95 = float(np.percentile(heights, 95))
                height_score = bell(height_p95, 0.30, 0.14)
            else:
                height_p95 = None
            gu = np.asarray(scene["ground_u"])[metric]
            gv = np.asarray(scene["ground_v"])[metric]
            finite_ground = np.isfinite(gu) & np.isfinite(gv)
            if int(finite_ground.sum()) >= 40:
                points_ground = np.column_stack(
                    [gu[finite_ground], gv[finite_ground]]
                ).astype(np.float32)
                ground_center = np.median(points_ground, axis=0)
                ground_radius = np.linalg.norm(
                    points_ground - ground_center[None, :], axis=1
                )
                cutoff = float(np.percentile(ground_radius, 94))
                trimmed_ground = points_ground[ground_radius <= cutoff]
                if len(trimmed_ground) >= 40:
                    points_ground = trimmed_ground
                (_, _), (side_a, side_b), _ = cv2.minAreaRect(points_ground)
                footprint = [float(max(side_a, side_b)), float(min(side_a, side_b))]
                # The rule-book carton is 0.50 x 0.35 x 0.30 m.  Ground
                # projection is deliberately allowed a very wide margin for
                # occlusion and plane-fit noise, but multi-metre furniture is
                # not a parcel hypothesis.
                if footprint[0] > 1.45:
                    continue
                if footprint[0] > 1.35:
                    footprint_score = None
                else:
                    footprint_score = (
                        0.62
                        * max(
                            bell(footprint[0], 0.50, 0.24),
                            bell(footprint[0], 0.35, 0.20),
                        )
                        + 0.38 * bell(footprint[1], 0.30, 0.20)
                    )
            else:
                footprint = None
            scene_debug = {
                "height_p95_m": None if height_p95 is None else round(height_p95, 4),
                "height_score": round(float(height_score), 4),
                "footprint_m": (
                    None if footprint is None else [round(v, 4) for v in footprint]
                ),
                "footprint_score": (
                    None if footprint_score is None else round(float(footprint_score), 4)
                ),
            }
            if height_p95 is not None and not (0.08 <= height_p95 <= 0.58):
                continue
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        solidity = float(cv2.contourArea(contour) / hull_area)
        fill = float(area / max(1, width_px * height_px))
        if depth is None:
            physical = None
            size_score = smooth_ramp(max(width_px, height_px), 55, 240)
        else:
            physical = [
                width_px * depth / camera["fx"],
                height_px * depth / camera["fy"],
            ]
            long_m, short_m = max(physical), min(physical)
            size_score = max(
                0.65 * bell(long_m, 0.50, 0.24)
                + 0.35 * bell(short_m, 0.30, 0.18),
                0.55 * bell(long_m, 0.36, 0.20)
                + 0.45 * bell(short_m, 0.30, 0.18),
            )
        if footprint_score is not None:
            size_score = 0.45 * size_score + 0.55 * footprint_score
        warm_purity = float(np.mean(warm[component] > 0))
        confidence = (
            0.22 * smooth_ramp(fill, 0.22, 0.70)
            + 0.17 * smooth_ramp(solidity, 0.50, 0.94)
            + 0.28 * size_score
            + 0.13 * smooth_ramp(area / warm.size, 0.001, 0.04)
            + 0.08 * warm_purity
            + 0.12 * height_score
        )
        # Without aligned depth, pixel size cannot separate the specified
        # carton from a warm cabinet, desk or floor patch.  Keep a conservative
        # RGB-only fallback candidate, but do not let it outrank a class with
        # verified part topology.
        if depth is None:
            confidence *= 0.72
        if confidence < 0.48:
            continue
        geometry, grasp_debug = _box_grasp(
            rgb,
            component.astype(np.uint8),
            bbox,
            z,
            camera,
            ignore,
            scene=scene,
            recognition_only=recognition_only,
        )
        if (
            geometry is not None
            and scene is not None
            and grasp_debug.get("line_height_score") is not None
            and float(grasp_debug["line_height_score"]) < 0.30
        ):
            geometry = None
            grasp_debug["reason_no_grasp"] = (
                "visible_edge_inconsistent_with_rule_height"
            )
        out.append(
            _base_candidate(
                "box",
                confidence,
                bbox,
                component.astype(np.uint8),
                depth,
                geometry,
                {
                    "area_px": int(area),
                    "fill_ratio": round(fill, 4),
                    "solidity": round(solidity, 4),
                    "apparent_size_m": (
                        None if physical is None else [round(v, 4) for v in physical]
                    ),
                    "grasp": grasp_debug,
                    "scene": scene_debug,
                },
            )
        )
    return sorted(out, key=lambda item: item["selection_score"], reverse=True)[:4]


def _mesh_edge_density(gray, region):
    ys, xs = np.nonzero(region)
    if xs.size < 50:
        return 0.0
    edge = cv2.Canny(gray, 35, 115)
    return float(np.mean(edge[region] > 0))


def _chair_top_line(rgb, chair_mask, bbox, z, camera, ignore):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gradients = (
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    x1, y1, x2, y2 = bbox
    edge = cv2.Canny(gray, 35, 115)
    roi = np.zeros_like(edge)
    roi[y1:y2, x1:x2] = 255
    edge[(roi == 0) | (ignore > 0)] = 0
    min_len = max(32, int(0.13 * max(x2 - x1, y2 - y1)))
    lines = _hough_segments(edge, min_len, max_gap=24, threshold=18)
    object_depth = robust_depth(z, chair_mask, 30)
    candidates = []
    normal_offset = max(5, int(0.018 * max(x2 - x1, y2 - y1)))
    for raw in lines:
        p0 = np.array(raw[:2], np.float32)
        p1 = np.array(raw[2:], np.float32)
        length = float(np.linalg.norm(p1 - p0))
        points = line_points(p0, p1)
        normal = line_normal(p0, p1)
        dark0, ok0 = sample_image(gray, points)
        along_dark = float(np.mean((dark0 < 115) & ok0))
        side_a, oka = sample_image(gray, points + normal_offset * normal)
        side_b, okb = sample_image(gray, points - normal_offset * normal)
        da = float(np.mean((side_a < 150) & oka))
        db = float(np.mean((side_b < 150) & okb))
        side_contrast = abs(da - db)
        support = line_mask_support(chair_mask, p0, p1, 5)
        strength, coverage = line_gradient_support(
            gray, p0, p1, 2, gradients=gradients
        )
        mid = 0.5 * (p0 + p1)
        relative_y = float(np.clip((mid[1] - y1) / max(1, y2 - y1), 0, 1))
        high_score = 1.0 - smooth_ramp(relative_y, 0.18, 0.70)
        zline, zok = sample_image(z, points, default=np.nan)
        zv = zline[np.isfinite(zline) & zok]
        zmed = float(np.median(zv)) if len(zv) >= 8 else object_depth
        valid_fraction = float(len(zv) / len(points))
        if zmed is None:
            length_m = None
            physical_score = smooth_ramp(length, min_len, 3.0 * min_len)
        else:
            length_m = length * zmed / camera["fx"]
            physical_score = max(
                bell(length_m, 0.48, 0.22),
                bell(length_m, 0.30, 0.18),
            )
        score = (
            0.19 * smooth_ramp(along_dark, 0.25, 0.86)
            + 0.17 * smooth_ramp(side_contrast, 0.06, 0.55)
            + 0.17 * smooth_ramp(support, 0.25, 0.82)
            + 0.15 * smooth_ramp(coverage, 0.28, 0.86)
            + 0.10 * smooth_ramp(strength, 45, 220)
            + 0.13 * physical_score
            + 0.06 * high_score
            + 0.03 * smooth_ramp(valid_fraction, 0.05, 0.55)
        )
        if score >= 0.43:
            candidates.append(
                {
                    "p0": p0,
                    "p1": p1,
                    "length": length,
                    "length_m": length_m,
                    "angle": normalized_angle(p0, p1),
                    "score": float(score),
                    "relative_y": relative_y,
                    "support": support,
                    "coverage": coverage,
                    "along_dark": along_dark,
                    "side_contrast": side_contrast,
                }
            )
    candidates = _dedupe_lines(candidates, midpoint_px=18, angle_deg=5)
    if not candidates:
        return None, {"raw_line_count": len(lines), "line_candidate_count": 0}
    candidates.sort(
        key=lambda item: (
            item["score"] * (1.1 - 0.20 * item["relative_y"]),
            item["length"],
        ),
        reverse=True,
    )
    best = candidates[0]
    p0 = 0.96 * best["p0"] + 0.04 * best["p1"]
    p1 = 0.04 * best["p0"] + 0.96 * best["p1"]
    confidence = float(np.clip(0.48 + 0.50 * best["score"], 0, 0.99))
    return {
        "type": "line",
        "name": "backrest_top_edge",
        "endpoints_uv": [safe_point(p0, chair_mask.shape), safe_point(p1, chair_mask.shape)],
        "confidence": confidence,
        "grasp_safe": True,
    }, {
        "raw_line_count": len(lines),
        "line_candidate_count": len(candidates),
        "best_line_score": round(best["score"], 4),
        "full_line_length_px": round(best["length"], 2),
        "full_line_length_m": (
            None if best["length_m"] is None else round(best["length_m"], 4)
        ),
        "mask_support": round(best["support"], 4),
        "edge_coverage": round(best["coverage"], 4),
        "along_dark": round(best["along_dark"], 4),
        "side_contrast": round(best["side_contrast"], 4),
    }


def detect_chairs(rgb, z, camera, ignore):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    brown = (h >= 3) & (h <= 35) & (s >= 45)
    dark = (v <= 132) & ~brown
    dark[ignore > 0] = False
    dark = cv2.morphologyEx(
        dark.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    out = []
    for component_index, (component, area) in enumerate(
        component_masks(dark, max(700, int(0.0009 * dark.size)))
    ):
        if component_index >= 8:
            break
        bbox = bbox_from_mask(component)
        if bbox is None:
            continue
        bbox = expand_bbox(bbox, dark.shape, 0.04)
        x1, y1, x2, y2 = bbox
        width_px, height_px = x2 - x1, y2 - y1
        if width_px < 35 or height_px < 35:
            continue
        roi_mask = np.zeros_like(component, np.uint8)
        roi_mask[y1:y2, x1:x2] = 1
        component = component & (roi_mask > 0)
        depth = robust_depth(z, component, 30)
        dark_ratio = float(np.mean(v[component] <= 105))
        brown_ratio = float(np.mean(brown[component]))
        edge_density = _mesh_edge_density(gray, component)
        fill = float(area / max(1, width_px * height_px))
        if depth is None:
            physical = None
            size_score = smooth_ramp(max(width_px, height_px), 80, 310)
        else:
            physical = [
                width_px * depth / camera["fx"],
                height_px * depth / camera["fy"],
            ]
            long_m, short_m = max(physical), min(physical)
            size_score = (
                0.62 * max(bell(long_m, 0.62, 0.34), bell(long_m, 0.85, 0.42))
                + 0.38 * max(bell(short_m, 0.48, 0.27), bell(short_m, 0.30, 0.22))
            )
        confidence = (
            0.25 * smooth_ramp(dark_ratio, 0.30, 0.82)
            + 0.17 * smooth_ramp(edge_density, 0.025, 0.16)
            + 0.16 * smooth_ramp(fill, 0.10, 0.55)
            + 0.29 * size_score
            + 0.13 * smooth_ramp(area / dark.size, 0.001, 0.045)
        )
        confidence *= 1.0 - 0.55 * smooth_ramp(brown_ratio, 0.20, 0.72)
        if confidence < 0.44:
            continue
        geometry, grasp_debug = _chair_top_line(
            rgb, component.astype(np.uint8), bbox, z, camera, ignore
        )
        if geometry is not None:
            confidence = max(confidence, 0.52 + 0.25 * geometry["confidence"])
        out.append(
            _base_candidate(
                "chair",
                confidence,
                bbox,
                component.astype(np.uint8),
                depth,
                geometry,
                {
                    "area_px": int(area),
                    "dark_ratio": round(dark_ratio, 4),
                    "brown_ratio": round(brown_ratio, 4),
                    "edge_density": round(edge_density, 4),
                    "fill_ratio": round(fill, 4),
                    "apparent_size_m": (
                        None if physical is None else [round(v, 4) for v in physical]
                    ),
                    "grasp": grasp_debug,
                },
            )
        )
    return sorted(out, key=lambda item: item["selection_score"], reverse=True)[:5]


def run_all_detectors(rgb, z, camera, ignore, scene=None):
    chair_candidates = []
    for candidate in detect_chair_candidates(
        rgb,
        z,
        camera,
        ignore,
        scene=scene,
        recognition_only=False,
    ):
        mask = np.asarray(candidate["mask"], np.uint8)
        geometry = candidate.get("grasp_geometry", candidate.get("geometry"))
        chair_candidates.append(
            _base_candidate(
                "chair",
                candidate["confidence"],
                candidate["bbox"],
                mask,
                robust_depth(z, mask, 30),
                geometry,
                candidate.get("debug", {}),
            )
        )
    return (
        detect_baskets(rgb, z, camera, ignore, scene=scene)
        + detect_trash_metric(
            rgb,
            z,
            camera,
            ignore,
            scene=scene,
            recognition_only=False,
        )
        + detect_boxes(
            rgb,
            z,
            camera,
            ignore,
            scene=scene,
            recognition_only=False,
        )
        + chair_candidates
    )
