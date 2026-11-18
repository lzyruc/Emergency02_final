"""Fast, explainable wrist RGB-D routing and one-class refinement.

The router intentionally uses only frame content: broad HSV colour families,
metric depth, connected structure, and edge density.  It does not use file
names, frame order, a fixed image ROI, or content hashes.  Once routed, only
one of the existing class-specific refiners is executed.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import cv2
import numpy as np

from .detectors import (
    _base_candidate,
    detect_baskets,
    detect_boxes,
    detect_chairs,
    detect_trash,
    detect_trash_metric,
)
from .chair_geometry import _appearance_maps as _chair_appearance_maps
from .chair_geometry import detect_chair_candidates
from .scene import build_scene_geometry
from .utils import bell, robust_depth, smooth_ramp


ROUTER_WIDTH = 320
REFINE_SCALE = {
    "basket": 1.0,
    "box": 0.75,
    "chair": 0.375,
    "trash_can": 0.60,
}


def _scaled_camera(camera: dict[str, Any], width: int, height: int):
    sx = float(width) / float(camera["width"])
    sy = float(height) / float(camera["height"])
    result = dict(camera)
    result.update(
        {
            "width": int(width),
            "height": int(height),
            "fx": float(camera["fx"]) * sx,
            "fy": float(camera["fy"]) * sy,
            "cx": float(camera["cx"]) * sx,
            "cy": float(camera["cy"]) * sy,
        }
    )
    return result


def _resize_inputs(rgb, z, ignore, camera, scale):
    height, width = rgb.shape[:2]
    work_width = max(160, int(round(width * float(scale))))
    work_height = max(90, int(round(height * float(scale))))
    size = (work_width, work_height)
    work_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    work_z = cv2.resize(z, size, interpolation=cv2.INTER_NEAREST)
    work_ignore = cv2.resize(ignore, size, interpolation=cv2.INTER_NEAREST)
    work_ignore = (work_ignore > 0).astype(np.uint8)
    work_z[work_ignore > 0] = np.nan
    return (
        work_rgb,
        work_z,
        work_ignore,
        _scaled_camera(camera, work_width, work_height),
    )


def _largest_component_fraction(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0
    return float(np.max(stats[1:, cv2.CC_STAT_AREA]) / mask.size)


def _component_structure_edge_density(rgb, component, ignore):
    """Measure repeated internal structure inside one blue object hull."""

    component = (np.asarray(component) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    contour_points = np.vstack(contours)
    if len(contour_points) < 3:
        return 0.0
    hull = cv2.convexHull(contour_points)
    region = np.zeros(component.shape, np.uint8)
    cv2.fillConvexPoly(region, hull, 1)
    region = cv2.erode(region, np.ones((3, 3), np.uint8))
    region[np.asarray(ignore) > 0] = 0
    usable = region > 0
    if int(np.count_nonzero(usable)) < 80:
        return 0.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 120)
    edge[np.asarray(ignore) > 0] = 0
    return float(np.mean(edge[usable] > 0))


def _basket_route(rgb, z, camera, ignore):
    """Find a physically plausible, saturated-blue perforated object.

    The hard gate is deliberately object-centric.  A blue floor cast or a
    distant blue prop can be large in pixels, but its metric component size
    and colour purity do not match a nearby file basket.
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue, _, red = [channel.astype(np.int16) for channel in cv2.split(rgb)]
    blue_minus_red = blue - red
    relative_blue = blue_minus_red.astype(np.float32) / np.maximum(
        value.astype(np.float32), 1.0
    )
    strong = (
        (hue >= 88)
        & (hue <= 132)
        & (saturation >= 78)
        & (value >= 42)
        & (blue_minus_red >= 24)
    )
    weak = (
        (hue >= 82)
        & (hue <= 138)
        & (saturation >= 35)
        & (value >= 28)
        & (blue_minus_red >= 10)
        & (relative_blue >= 0.16)
    )
    weak[ignore > 0] = False
    strong[ignore > 0] = False
    weak = cv2.morphologyEx(
        weak.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    weak[ignore > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    best = None
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    for index in range(1, count):
        x, y, width, height, area = [
            int(value) for value in stats[index]
        ]
        area_fraction = float(area / weak.size)
        if area_fraction < 0.018:
            continue
        component = labels == index
        purity = float(np.mean(strong[component]))
        relative = float(np.median(relative_blue[component]))
        structure_edge_density = _component_structure_edge_density(
            rgb, component, ignore
        )
        bbox_fill = float(area / max(1, width * height))
        component_u8 = component.astype(np.uint8)
        contours, _ = cv2.findContours(
            component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            hull_solidity = float(area / max(hull_area, 1.0))
        else:
            hull_solidity = 0.0
        valid_depth = z[
            component & np.isfinite(z) & (z > 0.05) & (z < 6.0)
        ]
        depth = (
            None
            if valid_depth.size < max(12, int(0.02 * area))
            else float(np.median(valid_depth))
        )
        metric_long = (
            None
            if depth is None
            else float(max(width, height) * depth / max(focal, 1e-6))
        )
        metric_score = (
            smooth_ramp(area_fraction, 0.055, 0.18)
            if metric_long is None
            else bell(metric_long, 0.50, 0.22)
        )
        score = float(
            0.40 * smooth_ramp(purity, 0.72, 0.94)
            + 0.22 * smooth_ramp(relative, 0.22, 0.72)
            + 0.20 * metric_score
            + 0.18 * smooth_ramp(area_fraction, 0.045, 0.20)
        )
        hard_pass = (
            area_fraction >= 0.075
            and purity >= 0.80
            and structure_edge_density >= 0.10
            and (
                metric_long is None
                or 0.30 <= float(metric_long) <= 0.90
            )
        )
        item = {
            "score": score,
            "hard_pass": bool(hard_pass),
            "area_fraction": area_fraction,
            "purity": purity,
            "relative_blue": relative,
            "structure_edge_density": structure_edge_density,
            "bbox_fill": bbox_fill,
            "hull_solidity": hull_solidity,
            "metric_long_m": metric_long,
            "bbox": [x, y, x + width, y + height],
        }
        if best is None or item["score"] > best["score"]:
            best = item

    if best is None:
        best = {
            "score": 0.0,
            "hard_pass": False,
            "area_fraction": 0.0,
            "purity": 0.0,
            "relative_blue": 0.0,
            "structure_edge_density": 0.0,
            "bbox_fill": 0.0,
            "hull_solidity": 0.0,
            "metric_long_m": None,
            "bbox": None,
        }
    return best


def _box_chair_route(rgb, ignore):
    """Separate filled cardboard faces from dark/open chair structure."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue, _, red = [channel.astype(np.int16) for channel in cv2.split(rgb)]
    usable = ignore == 0
    cardboard = (
        (hue >= 4)
        & (hue <= 32)
        & (saturation >= 25)
        & (saturation <= 190)
        & (value >= 45)
        & ((red - blue) >= 8)
        & usable
    )
    cardboard = cv2.morphologyEx(
        cardboard.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cardboard[~usable] = 0
    cardboard_fraction = _largest_component_fraction(cardboard)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 120) > 0
    edge[ignore > 0] = False
    edge_density = float(np.mean(edge[usable])) if np.any(usable) else 1.0
    cardboard_score = smooth_ramp(cardboard_fraction, 0.035, 0.13)
    # Filled carton faces remain locally smooth even when illumination shifts
    # their hue.  Chair backs contribute a denser set of frame, mesh, slot,
    # and silhouette edges.  Colour remains supporting evidence rather than a
    # hard requirement, which keeps pale and cool-cast cardboard routable.
    box_score = float(
        0.72 * smooth_ramp(0.102 - edge_density, 0.006, 0.040)
        + 0.28 * cardboard_score
    )
    chair_score = float(
        0.82 * smooth_ramp(edge_density, 0.086, 0.125)
        + 0.18 * (1.0 - cardboard_score)
    )
    selected = "box" if edge_density < 0.090 else "chair"
    return {
        "selected": selected,
        "box_score": box_score,
        "chair_score": chair_score,
        "cardboard_component_fraction": cardboard_fraction,
        "edge_density": edge_density,
    }


def _sample_route_map(image, points, default=0.0):
    points = np.asarray(points, np.float32)
    x = np.rint(points[:, 0]).astype(np.int32)
    y = np.rint(points[:, 1]).astype(np.int32)
    height, width = image.shape[:2]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.full(len(points), default, np.float32)
    values[inside] = image[y[inside], x[inside]]
    return values, inside


def _quick_flat_white_rim(rgb, z, camera, ignore, appearance):
    """Recognise a very oblique bright rim above a coherent blue-grey body."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.7), 35, 110)
    edge[ignore > 0] = 0
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 360.0,
        threshold=14,
        minLineLength=15,
        maxLineGap=7,
    )
    if lines is None:
        return None

    hue, saturation, value = cv2.split(appearance["hsv"])
    blue, _, red = cv2.split(rgb)
    blue_gray = (
        (hue >= 75)
        & (hue <= 140)
        & (saturation >= 8)
        & (saturation <= 190)
        & (value >= 35)
        & (value <= 230)
        & ((blue.astype(np.int16) - red.astype(np.int16)) >= 5)
        & (ignore == 0)
    )
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    best = None
    for raw in lines.reshape(-1, 4)[:350]:
        p0 = raw[:2].astype(np.float32)
        p1 = raw[2:].astype(np.float32)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 30.0:
            continue
        normal = np.array([-delta[1], delta[0]], np.float32) / length
        alpha = np.linspace(0.0, 1.0, max(20, int(length)), dtype=np.float32)
        centers = (
            p0[None, :] * (1.0 - alpha[:, None])
            + p1[None, :] * alpha[:, None]
        )
        depth_rows = []
        strict_white_rows = []
        for offset in (-3, -2, -1, 0, 1, 2, 3):
            points = centers + float(offset) * normal
            depth, visible = _sample_route_map(z, points, default=np.nan)
            sample_s, _ = _sample_route_map(
                saturation, points, default=255.0
            )
            sample_v, _ = _sample_route_map(value, points, default=0.0)
            blocked, _ = _sample_route_map(ignore, points, default=1.0)
            depth_rows.append(depth)
            strict_white_rows.append(
                (sample_s <= 55.0)
                & (sample_v >= 195.0)
                & visible
                & (blocked == 0)
            )

        depth_stack = np.stack(depth_rows)
        valid_depth = (
            np.isfinite(depth_stack)
            & (depth_stack > 0.05)
            & (depth_stack < 6.0)
        )
        if int(np.count_nonzero(valid_depth)) < max(
            12, int(0.20 * depth_stack.size)
        ):
            continue
        depth_m = float(np.median(depth_stack[valid_depth]))
        length_m = float(length * depth_m / max(focal, 1e-6))
        white_coverage = float(
            np.mean(np.any(np.stack(strict_white_rows), axis=0))
        )
        if not (0.14 <= length_m <= 0.36 and white_coverage >= 0.80):
            continue

        along = np.linspace(0.08, 0.92, 18, dtype=np.float32)
        body_centers = (
            p0[None, :] * (1.0 - along[:, None])
            + p1[None, :] * along[:, None]
        )
        distances = np.linspace(0.10 * length, 1.25 * length, 24)
        sides = []
        for sign in (-1.0, 1.0):
            points = (
                body_centers[:, None, :]
                + sign
                * distances[None, :, None]
                * normal[None, None, :]
            ).reshape(-1, 2)
            body_depth, visible = _sample_route_map(
                z, points, default=np.nan
            )
            body_colour, _ = _sample_route_map(
                blue_gray.astype(np.uint8), points, default=0.0
            )
            blocked, _ = _sample_route_map(ignore, points, default=1.0)
            same_surface = (
                visible
                & (blocked == 0)
                & np.isfinite(body_depth)
                & (
                    np.abs(body_depth - depth_m)
                    <= max(0.12, 0.12 * depth_m)
                )
            )
            colour_surface = (body_colour > 0) & same_surface
            sides.append(
                {
                    "colour_surface_fraction": float(
                        np.mean(colour_surface)
                    ),
                    "surface_fraction": float(np.mean(same_surface)),
                    "colour_purity": float(
                        np.count_nonzero(colour_surface)
                        / max(1, np.count_nonzero(same_surface))
                    ),
                }
            )

        body_index = int(
            sides[1]["colour_surface_fraction"]
            > sides[0]["colour_surface_fraction"]
        )
        body_side = sides[body_index]
        other_side = sides[1 - body_index]
        passed = (
            body_side["colour_surface_fraction"] >= 0.28
            and body_side["surface_fraction"] >= 0.25
            and body_side["colour_purity"] >= 0.80
            and other_side["surface_fraction"] <= 0.06
        )
        if not passed:
            continue
        score = float(
            0.46 * white_coverage
            + 0.34 * min(
                1.0, body_side["colour_surface_fraction"] / 0.34
            )
            + 0.20
            * math.exp(-0.5 * ((length_m - 0.255) / 0.10) ** 2)
        )
        item = {
            "score": score,
            "line_uv": raw.astype(int).tolist(),
            "length_m": length_m,
            "depth_m": depth_m,
            "white_coverage": white_coverage,
            "body_side": body_side,
            "other_side": other_side,
        }
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def _quick_trash_route(rgb, z, camera, ignore, blue_component):
    """Route a trash can from a smooth metric body or a strict white rim."""

    metric_long = blue_component.get("metric_long_m")
    smooth_body = bool(
        float(blue_component.get("area_fraction", 0.0)) >= 0.018
        and float(blue_component.get("bbox_fill", 0.0)) >= 0.74
        and float(blue_component.get("hull_solidity", 0.0)) >= 0.89
        and float(
            blue_component.get("structure_edge_density", 1.0)
        )
        <= 0.06
        and metric_long is not None
        and 0.20 <= float(metric_long) <= 0.42
    )

    rim_candidate = None
    if not smooth_body:
        for candidate in detect_trash(rgb, z, camera, ignore):
            debug = candidate.get("debug") or {}
            diameter_m = debug.get("diameter_m")
            ellipse = (
                candidate.get("grasp_geometry", {})
                .get("ellipse", {})
            )
            axes = ellipse.get("axes")
            ellipse_fraction = (
                0.0
                if axes is None or len(axes) != 2
                else float(
                    math.pi
                    * float(axes[0])
                    * float(axes[1])
                    / (4.0 * rgb.shape[0] * rgb.shape[1])
                )
            )
            if (
                float(debug.get("white_support", 0.0)) >= 0.50
                and float(debug.get("rim_inner_contrast", 0.0)) >= 15.0
                and ellipse_fraction >= 0.012
                and diameter_m is not None
                and 0.10 <= float(diameter_m) <= 0.42
            ):
                rim_candidate = {
                    "white_support": float(debug["white_support"]),
                    "rim_inner_contrast": float(
                        debug["rim_inner_contrast"]
                    ),
                    "diameter_m": float(diameter_m),
                    "ellipse_fraction": ellipse_fraction,
                }
                break

    hard_pass = smooth_body or rim_candidate is not None
    return {
        "hard_pass": bool(hard_pass),
        "score": 1.0 if smooth_body else float(rim_candidate is not None),
        "smooth_body": smooth_body,
        "body": {
            key: blue_component.get(key)
            for key in (
                "area_fraction",
                "bbox_fill",
                "hull_solidity",
                "structure_edge_density",
                "metric_long_m",
            )
        },
        "rim": rim_candidate,
    }


def _quick_chair_route(rgb, z, camera, ignore):
    """Separate chair structure from cardboard using one dark component."""

    del z, camera
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    brown = (hue >= 3) & (hue <= 35) & (saturation >= 45)
    dark = (value <= 132) & ~brown & (ignore == 0)
    dark = cv2.morphologyEx(
        dark.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    dark = cv2.dilate(
        dark,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    dark[ignore > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if count <= 1:
        return {
            "hard_pass": False,
            "score": 0.0,
            "dark_ratio": 0.0,
            "edge_density": 0.0,
            "component_fraction": 0.0,
        }
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == index
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 35, 115)
    edge[ignore > 0] = 0
    dark_ratio = float(np.mean(value[component] <= 105))
    edge_density = float(np.mean(edge[component] > 0))
    dark_structure = dark_ratio >= 0.65 or edge_density >= 0.12
    chair_maps = _chair_appearance_maps(rgb, ignore)
    mesh = (chair_maps["mesh"] >= 0.20).astype(np.uint8)
    mesh[ignore > 0] = 0
    mesh_count, _, mesh_stats, _ = cv2.connectedComponentsWithStats(
        mesh, 8
    )
    mesh_fraction = (
        0.0
        if mesh_count <= 1
        else float(
            np.max(mesh_stats[1:, cv2.CC_STAT_AREA]) / mesh.size
        )
    )
    # Strong contiguous chair structure is decisive.  Near the lower mesh
    # boundary, require the independent dark/edge cue as confirmation.
    hard_pass = mesh_fraction >= 0.34 or (
        mesh_fraction >= 0.27 and dark_structure
    )
    score = float(
        max(
            smooth_ramp(mesh_fraction, 0.25, 0.42),
            smooth_ramp(dark_ratio, 0.54, 0.72),
            smooth_ramp(edge_density, 0.105, 0.14),
        )
    )
    return {
        "hard_pass": bool(hard_pass),
        "score": score,
        "dark_ratio": dark_ratio,
        "edge_density": edge_density,
        "mesh_component_fraction": mesh_fraction,
        "component_fraction": float(
            stats[index, cv2.CC_STAT_AREA] / dark.size
        ),
    }


def _route(rgb, z, camera, ignore):
    basket = _basket_route(rgb, z, camera, ignore)
    if basket["hard_pass"]:
        return "basket", {
            "selected": "basket",
            "basket": basket,
            "trash": None,
            "chair": None,
            "box_chair": None,
        }

    trash = _quick_trash_route(rgb, z, camera, ignore, basket)
    if trash["hard_pass"]:
        return "trash_can", {
            "selected": "trash_can",
            "basket": basket,
            "trash": trash,
            "chair": None,
            "box_chair": None,
        }

    chair = _quick_chair_route(rgb, z, camera, ignore)
    if chair["hard_pass"]:
        return "chair", {
            "selected": "chair",
            "basket": basket,
            "trash": trash,
            "chair": chair,
            "box_chair": None,
        }

    box_chair = _box_chair_route(rgb, ignore)
    return "box", {
        "selected": "box",
        "basket": basket,
        "trash": trash,
        "chair": chair,
        "box_chair": box_chair,
    }


def _line_support_in_mask(geometry, mask, bbox):
    endpoints = np.asarray(geometry.get("endpoints_uv"), np.float32)
    if endpoints.shape != (2, 2):
        return False, {"reason": "invalid_endpoints"}
    delta = endpoints[1] - endpoints[0]
    if abs(float(delta[0])) < 1.4 * abs(float(delta[1])):
        return False, {"reason": "not_backrest_top_orientation"}
    length = float(np.linalg.norm(delta))
    if length < 18.0:
        return False, {"reason": "line_too_short"}

    count = max(24, int(math.ceil(length)))
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
    points = (
        endpoints[0][None, :] * (1.0 - alpha[:, None])
        + endpoints[1][None, :] * alpha[:, None]
    )
    pixels = np.rint(points).astype(np.int32)
    height, width = mask.shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    radius = int(np.clip(round(0.012 * math.hypot(width, height)), 5, 13))
    expanded = cv2.dilate(
        (mask > 0).astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        ),
    )
    support = np.zeros(count, dtype=bool)
    valid_pixels = pixels[inside]
    support[inside] = expanded[
        valid_pixels[:, 1], valid_pixels[:, 0]
    ] > 0
    support_fraction = float(np.mean(support))

    x1, y1, x2, y2 = [float(value) for value in bbox]
    midpoint = np.mean(endpoints, axis=0)
    relative_y = float((midpoint[1] - y1) / max(1.0, y2 - y1))
    passed = support_fraction >= 0.55 and relative_y <= 0.32
    return passed, {
        "reason": None if passed else "outside_backrest_top",
        "mask_support": support_fraction,
        "relative_y": relative_y,
        "line_length_px": length,
    }


def _detect_chairs_fast(rgb, z, camera, ignore):
    """Use geometry-first chair masks and a tightly validated top-rail aid."""

    scene = build_scene_geometry(z, camera, ignore)
    raw_candidates = detect_chair_candidates(
        rgb, z, camera, ignore, scene=scene
    )
    if not raw_candidates:
        # The geometry-first branch is intentionally strict and may reject a
        # partially visible chair.  Preserve class recognition with the cheap
        # dark/mesh detector, but do not publish its less-specific line as a
        # backrest top edge.
        fallback = []
        for item in detect_chairs(rgb, z, camera, ignore):
            semantic_mask = (
                np.asarray(item["mask"]) > 0
            ).astype(np.uint8)
            debug = copy.deepcopy(item.get("debug") or {})
            debug["fast_top_rail_supplement"] = {
                "accepted": False,
                "trials": [],
                "reason": "class_only_legacy_fallback",
            }
            fallback.append(
                _base_candidate(
                    "chair",
                    float(item["confidence"]),
                    item["bbox"],
                    semantic_mask,
                    robust_depth(z, semantic_mask, 30),
                    None,
                    debug,
                )
            )
        fallback.sort(
            key=lambda item: float(item.get("selection_score", 0.0)),
            reverse=True,
        )
        return fallback

    needs_supplement = any(
        item.get("grasp_geometry", item.get("geometry")) is None
        for item in raw_candidates
    )
    if needs_supplement:
        legacy = detect_chairs(rgb, z, camera, ignore)
        legacy_lines = [
            geometry
            for item in legacy
            if isinstance(
                geometry := item.get("grasp_geometry"), dict
            )
            and geometry.get("type") == "line"
        ]
    else:
        legacy_lines = []

    output = []
    for raw in raw_candidates:
        semantic_mask = (
            np.asarray(raw["mask"]) > 0
        ).astype(np.uint8)
        geometry = raw.get("grasp_geometry", raw.get("geometry"))
        debug = copy.deepcopy(raw.get("debug") or {})
        supplement_trials = []
        if geometry is None:
            for trial in legacy_lines:
                passed, evidence = _line_support_in_mask(
                    trial, semantic_mask, raw["bbox"]
                )
                supplement_trials.append(evidence)
                if passed:
                    geometry = copy.deepcopy(trial)
                    geometry["name"] = "backrest_top_edge"
                    break

        debug["fast_top_rail_supplement"] = {
            "accepted": geometry is not None
            and raw.get("grasp_geometry", raw.get("geometry")) is None,
            "trials": supplement_trials,
        }
        output.append(
            _base_candidate(
                "chair",
                float(raw["confidence"]),
                raw["bbox"],
                semantic_mask,
                robust_depth(z, semantic_mask, 30),
                geometry,
                debug,
            )
        )
    output.sort(
        key=lambda item: float(item.get("selection_score", 0.0)),
        reverse=True,
    )
    return output


def _run_selected_detector(cls, rgb, z, camera, ignore):
    if cls == "basket":
        return detect_baskets(rgb, z, camera, ignore, scene=None)
    if cls == "box":
        scene = build_scene_geometry(z, camera, ignore)
        return detect_boxes(rgb, z, camera, ignore, scene=scene)
    if cls == "chair":
        return _detect_chairs_fast(rgb, z, camera, ignore)
    if cls == "trash_can":
        return detect_trash_metric(rgb, z, camera, ignore, scene=None)
    raise ValueError(f"unsupported routed class: {cls}")


def _scale_point(point, sx, sy):
    if point is None or len(point) != 2:
        return point
    return [float(point[0]) / sx, float(point[1]) / sy]


def _restore_geometry(geometry, sx, sy):
    if geometry is None:
        return None
    result = copy.deepcopy(geometry)
    if result.get("endpoints_uv") is not None:
        result["endpoints_uv"] = [
            _scale_point(point, sx, sy)
            for point in result["endpoints_uv"]
        ]
    if result.get("grasp_point_uv") is not None:
        result["grasp_point_uv"] = _scale_point(
            result["grasp_point_uv"], sx, sy
        )
    for key in ("ellipse", "ellipse_uv"):
        ellipse = result.get(key)
        if not isinstance(ellipse, dict):
            continue
        ellipse["center"] = _scale_point(ellipse.get("center"), sx, sy)
        axes = ellipse.get("axes")
        if axes is not None and len(axes) == 2:
            ellipse["axes"] = [
                float(axes[0]) / sx,
                float(axes[1]) / sy,
            ]
    return result


def _refine_line_full_resolution(rgb, geometry, ignore):
    endpoints = np.asarray(geometry.get("endpoints_uv"), np.float32)
    if endpoints.shape != (2, 2):
        return geometry
    delta = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(delta))
    if length < 20.0:
        return geometry
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]], np.float32)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 45, 135) > 0
    edge[ignore > 0] = False
    ys, xs = np.nonzero(edge)
    if xs.size < 20:
        return geometry
    points = np.column_stack([xs, ys]).astype(np.float32)
    relative = points - endpoints[0][None, :]
    along = relative @ tangent
    across = np.abs(relative @ normal)
    band_px = float(np.clip(0.006 * math.hypot(*rgb.shape[:2]), 5.0, 10.0))
    selected = (
        (along >= -0.04 * length)
        & (along <= 1.04 * length)
        & (across <= band_px)
    )
    support = points[selected]
    if len(support) < max(18, int(0.10 * length)):
        return geometry

    direction_x, direction_y, center_x, center_y = [
        float(value)
        for value in cv2.fitLine(
            support, cv2.DIST_HUBER, 0, 0.01, 0.01
        ).reshape(-1)
    ]
    fitted_direction = np.array(
        [direction_x, direction_y], np.float32
    )
    if float(fitted_direction @ tangent) < 0.0:
        fitted_direction *= -1.0
    angle_cosine = float(
        np.clip(fitted_direction @ tangent, -1.0, 1.0)
    )
    if math.degrees(math.acos(angle_cosine)) > 8.0:
        return geometry
    fitted_center = np.array([center_x, center_y], np.float32)
    refined = []
    for endpoint in endpoints:
        projected = fitted_center + fitted_direction * float(
            (endpoint - fitted_center) @ fitted_direction
        )
        if float(np.linalg.norm(projected - endpoint)) > 1.5 * band_px:
            return geometry
        refined.append(projected)

    result = copy.deepcopy(geometry)
    result["endpoints_uv"] = [
        [float(point[0]), float(point[1])] for point in refined
    ]
    result["grasp_point_uv"] = [
        float(0.5 * (refined[0][0] + refined[1][0])),
        float(0.5 * (refined[0][1] + refined[1][1])),
    ]
    result["full_resolution_refined"] = True
    return result


def _trim_chair_line_to_bbox(geometry, bbox, image_shape):
    """Keep the published chair line inside the detected backrest region."""

    endpoints = np.asarray(geometry.get("endpoints_uv"), np.float32)
    if endpoints.shape != (2, 2) or bbox is None or len(bbox) != 4:
        return geometry
    x1, y1, x2, y2 = [float(value) for value in bbox]
    pad_x = max(8.0, 0.08 * max(1.0, x2 - x1))
    pad_y = max(8.0, 0.08 * max(1.0, y2 - y1))
    height, width = image_shape[:2]
    bounds = (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width - 1), x2 - 1.0 + pad_x),
        min(float(height - 1), y2 - 1.0 + pad_y),
    )
    alpha = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    points = (
        endpoints[0][None, :] * (1.0 - alpha[:, None])
        + endpoints[1][None, :] * alpha[:, None]
    )
    inside = (
        (points[:, 0] >= bounds[0])
        & (points[:, 0] <= bounds[2])
        & (points[:, 1] >= bounds[1])
        & (points[:, 1] <= bounds[3])
    )
    indices = np.nonzero(inside)[0]
    if indices.size < 2:
        return geometry
    start = float(alpha[int(indices[0])])
    stop = float(alpha[int(indices[-1])])
    if stop - start < 0.35:
        return geometry
    trimmed = [
        endpoints[0] * (1.0 - start) + endpoints[1] * start,
        endpoints[0] * (1.0 - stop) + endpoints[1] * stop,
    ]
    if float(np.linalg.norm(trimmed[1] - trimmed[0])) < 20.0:
        return geometry
    result = copy.deepcopy(geometry)
    result["endpoints_uv"] = [
        [float(point[0]), float(point[1])] for point in trimmed
    ]
    result["grasp_point_uv"] = [
        float(0.5 * (trimmed[0][0] + trimmed[1][0])),
        float(0.5 * (trimmed[0][1] + trimmed[1][1])),
    ]
    result["bbox_trimmed"] = bool(start > 0.0 or stop < 1.0)
    return result


def _ellipse_point(ellipse, theta):
    center = np.asarray(ellipse["center"], np.float64)
    axes = 0.5 * np.asarray(ellipse["axes"], np.float64)
    angle = math.radians(float(ellipse["angle_deg"]))
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        np.float64,
    )
    local = np.array(
        [axes[0] * math.cos(theta), axes[1] * math.sin(theta)],
        np.float64,
    )
    return center + rotation @ local


def _refine_ellipse_full_resolution(rgb, geometry, ignore):
    ellipse = geometry.get("ellipse")
    if not isinstance(ellipse, dict):
        return geometry
    center = np.asarray(ellipse.get("center"), np.float64)
    axes = np.asarray(ellipse.get("axes"), np.float64)
    if center.shape != (2,) or axes.shape != (2,) or np.min(axes) < 12.0:
        return geometry

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 45, 135) > 0
    edge[ignore > 0] = False
    ys, xs = np.nonzero(edge)
    if xs.size < 30:
        return geometry
    points = np.column_stack([xs, ys]).astype(np.float64)
    angle = math.radians(float(ellipse.get("angle_deg", 0.0)))
    rotation = np.array(
        [
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ],
        np.float64,
    )
    local = (points - center[None, :]) @ rotation.T
    radii = 0.5 * axes
    normalized_radius = np.sqrt(
        (local[:, 0] / max(radii[0], 1.0)) ** 2
        + (local[:, 1] / max(radii[1], 1.0)) ** 2
    )
    tolerance = float(
        np.clip(8.0 / max(np.min(radii), 8.0), 0.045, 0.22)
    )
    support = points[np.abs(normalized_radius - 1.0) <= tolerance]
    if len(support) < 40:
        return geometry

    fitted_center, fitted_axes, fitted_angle = cv2.fitEllipse(
        support.astype(np.float32).reshape(-1, 1, 2)
    )
    fitted_center = np.asarray(fitted_center, np.float64)
    fitted_axes = np.asarray(fitted_axes, np.float64)
    if float(np.linalg.norm(fitted_center - center)) > 12.0:
        return geometry
    old_sorted = np.sort(axes)
    new_sorted = np.sort(fitted_axes)
    relative_axis_error = float(
        np.max(np.abs(new_sorted - old_sorted) / np.maximum(old_sorted, 1.0))
    )
    if relative_axis_error > 0.12:
        return geometry

    fitted = {
        "center": [float(value) for value in fitted_center],
        "axes": [float(value) for value in fitted_axes],
        "angle_deg": float(fitted_angle),
    }
    result = copy.deepcopy(geometry)
    result["ellipse"] = fitted
    old_point = result.get("grasp_point_uv")
    if old_point is not None:
        old_point = np.asarray(old_point, np.float64)
        new_angle = math.radians(float(fitted["angle_deg"]))
        inverse_rotation = np.array(
            [
                [math.cos(new_angle), math.sin(new_angle)],
                [-math.sin(new_angle), math.cos(new_angle)],
            ],
            np.float64,
        )
        point_local = inverse_rotation @ (old_point - fitted_center)
        half_axes = 0.5 * fitted_axes
        theta = math.atan2(
            point_local[1] / max(half_axes[1], 1.0),
            point_local[0] / max(half_axes[0], 1.0),
        )
        result["grasp_point_uv"] = [
            float(value) for value in _ellipse_point(fitted, theta)
        ]
    result["full_resolution_refined"] = True
    return result


def _restore_candidate(
    candidate,
    source_rgb,
    source_ignore,
    work_shape,
    route_debug,
    refine_scale,
):
    source_height, source_width = source_rgb.shape[:2]
    work_height, work_width = work_shape[:2]
    sx = float(work_width) / float(source_width)
    sy = float(work_height) / float(source_height)
    result = copy.deepcopy(candidate)
    bbox = result.get("bbox")
    if bbox is not None and len(bbox) == 4:
        result["bbox"] = [
            int(np.clip(math.floor(float(bbox[0]) / sx), 0, source_width)),
            int(np.clip(math.floor(float(bbox[1]) / sy), 0, source_height)),
            int(np.clip(math.ceil(float(bbox[2]) / sx), 0, source_width)),
            int(np.clip(math.ceil(float(bbox[3]) / sy), 0, source_height)),
        ]
    if result.get("mask") is not None:
        result["mask"] = cv2.resize(
            np.asarray(result["mask"], np.uint8),
            (source_width, source_height),
            interpolation=cv2.INTER_NEAREST,
        )
    geometry = _restore_geometry(
        result.get("grasp_geometry"), sx, sy
    )
    if geometry is not None and result.get("class") in {"box", "chair"}:
        geometry = _refine_line_full_resolution(
            source_rgb, geometry, source_ignore
        )
        if result.get("class") == "chair":
            geometry = _trim_chair_line_to_bbox(
                geometry,
                result.get("bbox"),
                source_rgb.shape,
            )
    elif geometry is not None and result.get("class") == "trash_can":
        geometry = _refine_ellipse_full_resolution(
            source_rgb, geometry, source_ignore
        )
    result["grasp_geometry"] = geometry
    debug = copy.deepcopy(result.get("debug") or {})
    debug["fast_pipeline"] = {
        "route": route_debug,
        "refine_scale": float(refine_scale),
        "working_resolution": [int(work_width), int(work_height)],
    }
    result["debug"] = debug
    return result


def run_fast_pipeline(
    rgb,
    z,
    camera,
    supplied_ignore=None,
    target_class=None,
):
    """Route or force one class, then return full-resolution candidates."""

    rgb = np.asarray(rgb)
    z = np.asarray(z, np.float32)
    height, width = rgb.shape[:2]
    ignore = np.zeros((height, width), np.uint8)
    if supplied_ignore is not None:
        supplied_ignore = np.asarray(supplied_ignore)
        if supplied_ignore.shape != (height, width):
            raise ValueError("ignore_mask must have the same HxW as RGB")
        ignore[supplied_ignore > 0] = 1

    if target_class is None:
        route_scale = min(1.0, float(ROUTER_WIDTH) / float(width))
        route_rgb, route_z, route_ignore, route_camera = _resize_inputs(
            rgb, z, ignore, camera, route_scale
        )
        routed_class, route_debug = _route(
            route_rgb, route_z, route_camera, route_ignore
        )
        route_resolution = [int(route_rgb.shape[1]), int(route_rgb.shape[0])]
    else:
        if target_class not in REFINE_SCALE:
            raise ValueError(f"unsupported target class: {target_class}")
        routed_class = target_class
        route_debug = {
            "selected": routed_class,
            "forced_by_head_yolo": True,
        }
        route_resolution = None

    refine_scale = float(REFINE_SCALE[routed_class])
    work_rgb, work_z, work_ignore, work_camera = _resize_inputs(
        rgb, z, ignore, camera, refine_scale
    )
    detections = _run_selected_detector(
        routed_class, work_rgb, work_z, work_camera, work_ignore
    )
    restored = [
        _restore_candidate(
            item,
            rgb,
            ignore,
            work_rgb.shape,
            route_debug,
            refine_scale,
        )
        for item in detections
    ]
    restored.sort(
        key=lambda item: float(item.get("selection_score", 0.0)),
        reverse=True,
    )
    debug = {
        "mode": "fast_single_class",
        "routed_class": routed_class,
        "forced_by_head_yolo": target_class is not None,
        "route_resolution": route_resolution,
        "refine_resolution": [
            int(work_rgb.shape[1]),
            int(work_rgb.shape[0]),
        ],
        "route": route_debug,
        "candidate_count": len(restored),
    }
    return restored, ignore, debug
