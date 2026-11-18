"""Metric, semantic grasp-line extraction for the blue five-divider basket.

Only the exposed free upper edge of a divider plate is an allowed grasp edge.
The following visually tempting features are deliberately *not* fallbacks:

* front or rear transverse beams;
* base/frame edges;
* hole/grid edges;
* a single arbitrary outer silhouette.

The extractor is intentionally independent from object classification.  Its
``mask`` and ``bbox`` arguments are proposals only; a line is published after
it independently passes metric height, repeated-divider-family, one-sided
free-space, RGB-boundary and 3-D straightness checks.
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from .scene import build_scene_geometry


ALGORITHM_VERSION = "wrist-basket-five-ridge-rgbd-v6"

BASKET_DIVIDER_SPACING_M = 0.303 / 4.0
FULL_EDGE_MIN_M = 0.052
FULL_EDGE_MAX_M = 0.46
GRASP_MIN_M = 0.060
GRASP_MAX_M = 0.160
GRASP_TARGET_M = 0.105
# The 308 mm basket's divider free edge remains above the bottom grid even
# when the basket is tilted.  A terminal below this world-height band is a
# floor/base/hole feature, not an exposed upper edge.  Apply only when the
# scene has a valid ground plane; RGB-D without one stays conservatively
# governed by the remaining topology gates.
GRASP_MIN_MEAN_HEIGHT_M = 0.120
FREE_EDGE_MAX_PAIR_GAP_M = 0.170
FAMILY_MIN_HEIGHT_SPAN_M = 0.080
TERMINAL_FAMILY_LIMIT = 6
PAIR_FAMILY_MIN_SCORE = 0.60
MIN_SELECTED_FREE_EDGE_SCORE = 0.45


def _ramp(value, lo, hi):
    if hi <= lo:
        return float(value >= hi)
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def _bell(value, center, sigma):
    return float(
        math.exp(-0.5 * ((float(value) - center) / max(float(sigma), 1e-9)) ** 2)
    )


def _safe_uv(point, shape):
    h, w = shape[:2]
    return [
        int(np.clip(round(float(point[0])), 0, w - 1)),
        int(np.clip(round(float(point[1])), 0, h - 1)),
    ]


def _line_points(p0, p1, count=None):
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    length = float(np.linalg.norm(p1 - p0))
    if count is None:
        count = max(18, int(math.ceil(length * 1.15)))
    alpha = np.linspace(0.0, 1.0, int(count), dtype=np.float32)
    points = p0[None, :] + alpha[:, None] * (p1 - p0)[None, :]
    return points, alpha


def _sample(image, points, default=0):
    h, w = image.shape[:2]
    pixels = np.rint(points).astype(np.int32)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    if image.ndim == 2:
        out = np.full(len(points), default, dtype=image.dtype)
        out[valid] = image[pixels[valid, 1], pixels[valid, 0]]
    else:
        out = np.full((len(points), image.shape[2]), default, dtype=image.dtype)
        out[valid] = image[pixels[valid, 1], pixels[valid, 0]]
    return out, valid


def _normal(p0, p1):
    delta = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        return np.array([0.0, 1.0], np.float64)
    return np.array([-delta[1], delta[0]], np.float64) / norm


def _angle(p0, p1):
    delta = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
    angle = math.atan2(float(delta[1]), float(delta[0])) % math.pi
    return float(angle)


def _angle_distance(a, b):
    delta = abs(float(a) - float(b)) % math.pi
    return min(delta, math.pi - delta)


def _object_depth(z, mask):
    values = np.asarray(z)[
        (mask > 0)
        & np.isfinite(z)
        & (np.asarray(z) > 0.05)
        & (np.asarray(z) < 8.0)
    ]
    if len(values) < 12:
        return None
    lo, hi = np.percentile(values, (8, 82))
    trimmed = values[(values >= lo) & (values <= hi)]
    if len(trimmed) < 8:
        trimmed = values
    return float(np.median(trimmed))


def _metric_pixels(camera, depth, distance_m, lo, hi):
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    value = focal * float(distance_m) / max(float(depth), 0.04)
    return int(np.clip(round(value), int(lo), int(hi)))


def _metric_ignore_clearance(ignore, camera, depth, clearance_m=0.022):
    """Dilate the robot mask by a metric safety clearance.

    The gripper is a foreground occluder, not free space.  A depth-scaled
    dilation prevents its silhouette, fingers, and registration halo from
    becoming candidate basket edges at different camera distances.
    """
    radius = _metric_pixels(camera, depth, clearance_m, 4, 76)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    clear = cv2.dilate((np.asarray(ignore) > 0).astype(np.uint8), kernel)
    return clear, radius


def _unproject(uv, depth, camera):
    uv = np.asarray(uv, np.float64)
    depth = np.asarray(depth, np.float64)
    x = (uv[:, 0] - float(camera["cx"])) * depth / float(camera["fx"])
    y = (uv[:, 1] - float(camera["cy"])) * depth / float(camera["fy"])
    return np.column_stack([x, y, depth])


def _edge_support(distance_map, points, tolerance_px):
    values, valid = _sample(distance_map, points, default=1e6)
    return float(np.mean(valid & (values <= float(tolerance_px))))


def _longest_true_run(values):
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _gradient_support(gray, p0, p1, normal):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    points, _ = _line_points(p0, p1)
    strengths = []
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        sx, ok = _sample(gx, points + offset * normal, default=0)
        sy, _ = _sample(gy, points + offset * normal, default=0)
        strengths.append(np.abs(sx * normal[0] + sy * normal[1]) * ok)
    strength = np.max(np.stack(strengths, axis=0), axis=0)
    return float(np.percentile(strength, 55)), float(np.mean(strength >= 55.0))


def _make_occupancy(mask, depth, camera):
    """Bridge metric-size grid holes without filling the large L-shaped notch."""
    radius = _metric_pixels(camera, depth, 0.0065, 2, 19)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    occupancy = cv2.morphologyEx(
        (mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel
    )
    # A very small closing joins sensor pinholes left after the metric close.
    occupancy = cv2.morphologyEx(
        occupancy,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return occupancy, radius


def _extract_segments(edge, min_length_px, max_gap_px):
    segments = []
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 720.0,
        threshold=max(14, int(round(min_length_px * 0.24))),
        minLineLength=max(16, int(round(min_length_px))),
        maxLineGap=max(5, int(max_gap_px)),
    )
    if lines is not None:
        for raw in lines.reshape(-1, 4):
            segments.append(
                (
                    np.asarray(raw[:2], np.float32),
                    np.asarray(raw[2:], np.float32),
                    "hough",
                )
            )

    # LSD gives much tighter endpoints than Hough on partially occluded edges.
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(edge)[0]
    if found is not None:
        for raw in found.reshape(-1, 4):
            p0 = np.asarray(raw[:2], np.float32)
            p1 = np.asarray(raw[2:], np.float32)
            if float(np.linalg.norm(p1 - p0)) >= 0.72 * min_length_px:
                segments.append((p0, p1, "lsd"))
    segments.sort(key=lambda item: float(np.linalg.norm(item[1] - item[0])), reverse=True)
    return segments[:900]


def _depth_line_fit(
    p0,
    p1,
    material_sign,
    z,
    camera,
    px_per_m,
    ignore,
):
    """Fit the physical edge from depth on the material side of its boundary."""
    points, alpha = _line_points(p0, p1)
    normal = _normal(p0, p1)
    # Do not sample exactly on the silhouette: registered depth there often
    # belongs to the background.  The first samples are just inside the plate.
    offsets_m = (0.0025, 0.0045, 0.0070, 0.0100, 0.0140)
    # Vectorized closest-valid-offset selection is both faster and makes the
    # gripper exclusion apply to every depth support sample, not just the
    # visible centreline.
    selected_uv = np.full((len(points), 2), np.nan, np.float64)
    selected_z = np.full(len(points), np.nan, np.float64)
    for offset_m in offsets_m:
        shifted = points + material_sign * offset_m * px_per_m * normal
        values, in_frame = _sample(z, shifted, default=np.nan)
        blocked, _ = _sample(ignore, shifted, default=255)
        use = (
            np.isnan(selected_z)
            & in_frame
            & np.isfinite(values)
            & (blocked == 0)
        )
        selected_uv[use] = shifted[use]
        selected_z[use] = values[use]

    selected = np.isfinite(selected_z)
    if int(selected.sum()) < 10:
        return None
    uv = selected_uv[selected]
    depths = selected_z[selected]
    alphas = np.asarray(alpha[selected], np.float64)

    # Remove flying pixels using a robust local depth trend, not a constant-Z gate.
    design = np.column_stack([np.ones(len(alphas)), alphas])
    coefficients, _, _, _ = np.linalg.lstsq(design, depths, rcond=None)
    residual_z = depths - design @ coefficients
    median = float(np.median(residual_z))
    mad = float(np.median(np.abs(residual_z - median)))
    tolerance = max(0.008, 3.5 * 1.4826 * mad)
    keep = np.abs(residual_z - median) <= tolerance
    if int(keep.sum()) < 9:
        return None
    uv = uv[keep]
    depths = depths[keep]
    alphas = alphas[keep]
    xyz = _unproject(uv, depths, camera)

    center = np.median(xyz, axis=0)
    matrix = xyz - center
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    direction = vh[0]
    if float(direction[2]) < 0:
        direction = -direction
    along = matrix @ direction
    perpendicular = matrix - along[:, None] * direction[None, :]
    residual = np.linalg.norm(perpendicular, axis=1)
    rms = float(np.sqrt(np.mean(residual**2)))

    low, high = np.percentile(along, (3, 97))
    length_m = float(high - low)
    endpoints_xyz = np.stack(
        [center + low * direction, center + high * direction], axis=0
    )

    # A linear alpha->XYZ model lets the later terminal crop retain metric points.
    design = np.column_stack([np.ones(len(alphas)), alphas])
    model, _, _, _ = np.linalg.lstsq(design, xyz, rcond=None)
    return {
        "direction": direction,
        "center": center,
        "rms_m": rms,
        "length_m": length_m,
        "support_count": int(len(xyz)),
        "valid_fraction": float(len(xyz) / max(len(points), 1)),
        "endpoints_xyz": endpoints_xyz,
        "alpha_xyz_model": model,
        "depth_span_m": float(np.percentile(depths, 90) - np.percentile(depths, 10)),
    }


def _height_evidence(scene, mask, points, material_sign, normal, px_per_m):
    if not scene or "height" not in scene:
        return {
            "available": False,
            "rank": 0.55,
            "height_m": None,
            "base_reject": False,
            "flatness_m": None,
            "inward_drop_m": None,
            "endpoint_heights_m": None,
        }
    height = np.asarray(scene["height"])
    valid_map = np.asarray(scene.get("valid", np.isfinite(height)))
    object_values = height[
        (mask > 0) & valid_map & np.isfinite(height) & (height > -0.04)
    ]
    if len(object_values) < 30:
        return {
            "available": False,
            "rank": 0.55,
            "height_m": None,
            "base_reject": False,
            "flatness_m": None,
            "inward_drop_m": None,
            "endpoint_heights_m": None,
        }
    samples = []
    near_samples = []
    far_samples = []
    for offset_m in (0.0025, 0.005, 0.008, 0.011, 0.020, 0.030, 0.040):
        values, in_frame = _sample(
            height,
            points + material_sign * offset_m * px_per_m * normal,
            default=np.nan,
        )
        row = np.where(in_frame, values, np.nan)
        samples.append(row)
        if offset_m <= 0.008:
            near_samples.append(row)
        if offset_m >= 0.020:
            far_samples.append(row)
    stacked = np.stack(samples[:4], axis=0)
    line_height = np.full(stacked.shape[1], np.nan, np.float32)
    for index in range(stacked.shape[1]):
        column = stacked[:, index]
        column = column[np.isfinite(column)]
        if len(column):
            line_height[index] = float(np.median(column))
    finite = line_height[np.isfinite(line_height)]
    if len(finite) < 8:
        return {
            "available": False,
            "rank": 0.55,
            "height_m": None,
            "base_reject": False,
            "flatness_m": None,
            "inward_drop_m": None,
            "endpoint_heights_m": None,
        }
    lo, hi = np.percentile(object_values, (8, 94))
    median = float(np.median(finite))
    rank = float(np.clip((median - lo) / max(float(hi - lo), 0.025), 0.0, 1.0))
    flatness = float(np.percentile(finite, 90) - np.percentile(finite, 10))
    terminal_count = max(3, int(round(0.16 * len(line_height))))
    first_terminal = line_height[:terminal_count]
    second_terminal = line_height[-terminal_count:]
    first_terminal = first_terminal[np.isfinite(first_terminal)]
    second_terminal = second_terminal[np.isfinite(second_terminal)]
    endpoint_heights = None
    if len(first_terminal) >= 2 and len(second_terminal) >= 2:
        endpoint_heights = [
            float(np.median(first_terminal)),
            float(np.median(second_terminal)),
        ]
    near_values = np.stack(near_samples, axis=0)
    far_values = np.stack(far_samples, axis=0)
    near_values = near_values[np.isfinite(near_values)]
    far_values = far_values[np.isfinite(far_values)]
    inward_drop = None
    if len(near_values) >= 8 and len(far_values) >= 8:
        inward_drop = float(np.median(near_values) - np.median(far_values))
    return {
        "available": True,
        "rank": rank,
        "height_m": median,
        "base_reject": bool(median < 0.028 or rank < 0.20),
        "flatness_m": flatness,
        "inward_drop_m": inward_drop,
        "endpoint_heights_m": endpoint_heights,
    }


def _candidate_from_segment(
    segment,
    gray,
    z,
    mask,
    occupancy,
    boundary_distance,
    external_distance,
    combined_distance,
    bbox,
    camera,
    ignore,
    scene,
    object_depth,
):
    p0, p1, source = segment
    pixel_length = float(np.linalg.norm(p1 - p0))
    if pixel_length < 12:
        return None, "pixel_length"
    normal = _normal(p0, p1)
    points, _ = _line_points(p0, p1)
    px_per_m = math.sqrt(float(camera["fx"]) * float(camera["fy"])) / object_depth
    tolerance_px = float(np.clip(0.0035 * px_per_m, 1.8, 9.0))
    mask_edge_support = _edge_support(boundary_distance, points, tolerance_px)
    external_edge_support = _edge_support(
        external_distance, points, tolerance_px
    )
    combined_support = _edge_support(combined_distance, points, tolerance_px)
    # Interior dividers may project over another blue panel.  They can have
    # weak binary-mask contrast while still being a strong RGB/depth ridge.
    if combined_support < 0.60:
        return None, "not_mask_boundary"

    # A genuine free upper edge has plastic consistently on one side and open
    # space on the other.  Grid/hole edges have only a narrow local asymmetry.
    # The far samples are essential: a circular grid hole looks one-sided for
    # a few millimetres, but plastic reappears across the hole.  A true exposed
    # panel edge stays free until close to the next 75.8-mm divider.
    band_offsets_m = (0.004, 0.008, 0.014, 0.022, 0.032, 0.044)
    side_values = []
    side_depth_values = []
    side_profiles = []
    side_depth_profiles = []
    line_z, _ = _sample(z, points, default=np.nan)
    line_z_valid = line_z[np.isfinite(line_z)]
    line_z_median = (
        float(np.median(line_z_valid)) if len(line_z_valid) >= 6 else object_depth
    )
    for sign in (1.0, -1.0):
        occupancy_rows = []
        same_depth_rows = []
        for offset_m in band_offsets_m:
            shifted = points + sign * offset_m * px_per_m * normal
            values, in_frame = _sample(occupancy, shifted, default=0)
            occupancy_rows.append((values > 0) & in_frame)
            depth_values, _ = _sample(z, shifted, default=np.nan)
            same_depth_rows.append(
                np.isfinite(depth_values)
                & (np.abs(depth_values - line_z_median) <= 0.024)
            )
        occupancy_stack = np.stack(occupancy_rows, axis=0)
        depth_stack = np.stack(same_depth_rows, axis=0)
        side_profiles.append(np.mean(occupancy_stack, axis=0))
        side_depth_profiles.append(np.mean(depth_stack, axis=0))
        side_values.append(float(np.mean(occupancy_stack)))
        side_depth_values.append(float(np.mean(depth_stack)))
    preliminary_index = int(np.argmax(side_values))
    preliminary_material = side_values[preliminary_index]
    preliminary_free = side_values[1 - preliminary_index]
    preliminary_sidedness = preliminary_material - preliminary_free
    preliminary_free_depth = 1.0 - side_depth_values[1 - preliminary_index]
    mask_one_sided = (
        preliminary_material >= 0.26
        and preliminary_sidedness >= 0.07
        and preliminary_free <= 0.62
        and preliminary_free_depth >= 0.32
    )
    depth_one_sided = (
        preliminary_material >= 0.34
        and preliminary_sidedness >= 0.015
        and preliminary_free_depth >= 0.70
    )
    if not (mask_one_sided or depth_one_sided):
        return None, "not_one_sided_free_edge"

    gradient_strength, gradient_coverage = _gradient_support(
        gray, p0, p1, normal
    )
    if gradient_coverage < 0.30 and combined_support < 0.76:
        return None, "weak_rgb_edge"

    # When two blue plates overlap in the image, occupancy alone cannot tell
    # which normal side owns the edge.  Fit both and keep the straighter,
    # better-supported physical ridge.
    fits = []
    for index, sign in enumerate((1.0, -1.0)):
        trial = _depth_line_fit(
            p0, p1, sign, z, camera, px_per_m, ignore
        )
        if trial is None:
            continue
        trial_merit = (
            1.6 * trial["valid_fraction"]
            - 70.0 * trial["rms_m"]
            + 0.25 * side_values[index]
        )
        fits.append((trial_merit, index, sign, trial))
    if not fits:
        return None, "no_depth_line"
    _, material_index, material_sign, fitted = max(fits, key=lambda row: row[0])
    material_occupancy = side_values[material_index]
    free_occupancy = side_values[1 - material_index]
    sidedness = material_occupancy - free_occupancy
    free_depth = 1.0 - side_depth_values[1 - material_index]
    mask_one_sided = (
        material_occupancy >= 0.26
        and sidedness >= 0.07
        and free_occupancy <= 0.62
        and free_depth >= 0.32
    )
    depth_one_sided = (
        material_occupancy >= 0.34
        and sidedness >= 0.015
        and free_depth >= 0.70
    )
    if not (mask_one_sided or depth_one_sided):
        return None, "depth_side_disagrees"
    if not (FULL_EDGE_MIN_M <= fitted["length_m"] <= FULL_EDGE_MAX_M):
        return None, "physical_length"
    rms_gate = max(0.0045, 0.009 * object_depth)
    if fitted["rms_m"] > rms_gate or fitted["valid_fraction"] < 0.16:
        return None, "depth_not_straight"

    material_profile = side_profiles[material_index]
    free_profile = side_profiles[1 - material_index]
    free_depth_profile = 1.0 - side_depth_profiles[1 - material_index]
    boundary_values, boundary_valid = _sample(
        combined_distance, points, default=1e6
    )
    continuous = (
        boundary_valid
        & (boundary_values <= tolerance_px)
        & (material_profile >= 0.34)
        & ((free_profile <= 0.42) | (free_depth_profile >= 0.72))
    )
    longest_run = _longest_true_run(continuous)
    continuous_fraction = float(longest_run / max(len(points), 1))
    continuous_run_m = float(pixel_length * continuous_fraction / px_per_m)
    # Perspective, narrow moulded lips and partial depth dropout can split a
    # true ridge into short image runs.  Require a physical 18-mm contiguous
    # run here; the selected 60--160-mm terminal is revalidated separately.
    required_run_m = min(0.018, 0.20 * fitted["length_m"])
    if continuous_run_m < required_run_m:
        return None, "discontinuous_or_grid_edge"

    height = _height_evidence(
        scene, mask, points, material_sign, normal, px_per_m
    )
    if height["base_reject"]:
        return None, "base_or_low_frame_edge"

    pixels = np.rint(points).astype(np.int32)
    in_frame = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < mask.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < mask.shape[0])
    )
    blocked = np.ones(len(points), dtype=bool)
    blocked[in_frame] = (
        ignore[pixels[in_frame, 1], pixels[in_frame, 0]] > 0
    )
    clear_fraction = float(np.mean(~blocked))
    if clear_fraction < 0.98:
        return None, "robot_occlusion"

    score = (
        0.17 * _ramp(mask_edge_support, 0.08, 0.86)
        + 0.08 * _ramp(external_edge_support, 0.10, 0.88)
        + 0.10 * _ramp(combined_support, 0.58, 0.95)
        + 0.15 * _ramp(sidedness, 0.02, 0.62)
        + 0.09 * _ramp(material_occupancy, 0.26, 0.78)
        + 0.09 * _ramp(free_depth, 0.32, 0.92)
        + 0.08 * _ramp(gradient_coverage, 0.25, 0.82)
        + 0.12 * _ramp(fitted["valid_fraction"], 0.16, 0.70)
        + 0.10 * _ramp(rms_gate - fitted["rms_m"], 0.0, rms_gate)
        + 0.06 * _ramp(height["rank"], 0.24, 0.92)
        + 0.02 * clear_fraction
        + 0.02 * _ramp(continuous_run_m, 0.012, 0.070)
    )
    return {
        "p0": p0,
        "p1": p1,
        "pixel_length": pixel_length,
        "angle": _angle(p0, p1),
        "normal": normal,
        "material_sign": material_sign,
        "mask_edge_support": mask_edge_support,
        "external_edge_support": external_edge_support,
        "combined_edge_support": combined_support,
        "material_occupancy": material_occupancy,
        "free_occupancy": free_occupancy,
        "sidedness": sidedness,
        "free_depth": free_depth,
        "gradient_strength": gradient_strength,
        "gradient_coverage": gradient_coverage,
        "clear_fraction": clear_fraction,
        "continuous_free_run_m": continuous_run_m,
        "continuous_free_run_fraction": continuous_fraction,
        "required_free_run_m": required_run_m,
        "height": height,
        "fit": fitted,
        "base_score": float(np.clip(score, 0.0, 1.0)),
        "source": source,
    }, None


def _line_distance_3d(first, second, direction):
    delta = second["fit"]["center"] - first["fit"]["center"]
    return float(np.linalg.norm(delta - float(delta @ direction) * direction))


def _deduplicate(candidates):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["base_score"], reverse=True):
        duplicate = False
        for other in kept:
            direction = candidate["fit"]["direction"]
            parallel = abs(float(direction @ other["fit"]["direction"]))
            if parallel < math.cos(math.radians(12.0)):
                continue
            if _line_distance_3d(candidate, other, direction) < 0.020:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _lattice_support(anchor, candidates):
    """Find a compact observed family at the nominal divider pitch.

    The basket has at most five parallel plate sites.  Families therefore
    have to fit inside four consecutive pitch intervals and must retain the
    anchor itself.  This prevents the two outer walls, which are roughly one
    basket width apart, from masquerading as an adjacent-divider pair.
    """
    direction = np.asarray(anchor["fit"]["direction"], np.float64)
    related = []
    for candidate in candidates:
        candidate_direction = np.asarray(candidate["fit"]["direction"], np.float64)
        if abs(float(direction @ candidate_direction)) < math.cos(math.radians(19.0)):
            continue
        distance = _line_distance_3d(anchor, candidate, direction)
        if distance <= 0.38:
            related.append(candidate)
    if not related:
        return {
            "count": 1,
            "regularity": 0.0,
            "spacing_m": BASKET_DIVIDER_SPACING_M,
            "span_m": 0.0,
            "members": [anchor],
            "indices": [0],
            "bilateral": False,
            "adjacent_bilateral": False,
            "mean_error_m": 0.0,
        }

    centers = np.stack([item["fit"]["center"] for item in related], axis=0)
    anchor_index = next(
        (index for index, item in enumerate(related) if item is anchor), 0
    )
    centered = centers - centers[anchor_index]
    perpendicular = centered - (centered @ direction)[:, None] * direction[None, :]
    if len(perpendicular) >= 2:
        _, _, vh = np.linalg.svd(perpendicular, full_matrices=False)
        lateral_axis = vh[0]
    else:
        lateral_axis = np.array([1.0, 0.0, 0.0])
    lateral_axis -= float(lateral_axis @ direction) * direction
    lateral_axis /= np.linalg.norm(lateral_axis) + 1e-12
    coordinates = centered @ lateral_axis

    best = None
    spacing_grid = np.linspace(
        BASKET_DIVIDER_SPACING_M - 0.012,
        BASKET_DIVIDER_SPACING_M + 0.012,
        25,
    )
    for spacing in spacing_grid:
        bins = np.rint(coordinates / spacing).astype(int)
        residual = np.abs(coordinates - bins * spacing)
        tolerance = min(0.011, 0.15 * spacing)
        unique = {0: anchor_index}
        for index in np.flatnonzero(residual <= tolerance):
            key = int(bins[index])
            if abs(key) > 4:
                continue
            if key not in unique or residual[index] < residual[unique[key]]:
                unique[key] = int(index)
        for window_low in range(-4, 1):
            window_high = window_low + 4
            selected_bins = sorted(
                key for key in unique if window_low <= key <= window_high
            )
            if 0 not in selected_bins:
                continue
            selected = [unique[key] for key in selected_bins]
            count = len(selected)
            errors = residual[selected]
            bilateral = any(key < 0 for key in selected_bins) and any(
                key > 0 for key in selected_bins
            )
            adjacent_bilateral = -1 in selected_bins and 1 in selected_bins
            prior = abs(spacing - BASKET_DIVIDER_SPACING_M) / 0.012
            merit = (
                2.0 * min(count, 5)
                + 0.75 * float(bilateral)
                + 0.45 * float(adjacent_bilateral)
                - 24.0 * float(np.mean(errors))
                - 0.30 * prior
            )
            row = (
                merit,
                count,
                spacing,
                selected,
                selected_bins,
                float(np.mean(errors)),
                bilateral,
                adjacent_bilateral,
                tolerance,
            )
            if best is None or row[0] > best[0]:
                best = row
    if best is None:
        return {
            "count": 1,
            "regularity": 0.0,
            "spacing_m": None,
            "span_m": 0.0,
            "members": [anchor],
            "indices": [0],
            "bilateral": False,
            "adjacent_bilateral": False,
            "mean_error_m": 0.0,
        }
    (
        _,
        count,
        spacing,
        selected,
        selected_bins,
        mean_error,
        bilateral,
        adjacent_bilateral,
        tolerance,
    ) = best
    members = [related[index] for index in selected]
    selected_coordinates = coordinates[selected]
    regularity = (
        _ramp(count, 1.5, 4.0)
        * _ramp(tolerance - mean_error, 0.0, tolerance)
        * _bell(spacing, BASKET_DIVIDER_SPACING_M, 0.010)
    )
    return {
        "count": int(count),
        "regularity": float(regularity),
        "spacing_m": float(spacing),
        "span_m": float(np.ptp(selected_coordinates)) if count > 1 else 0.0,
        "members": members,
        "indices": [int(value) for value in selected_bins],
        "bilateral": bool(bilateral),
        "adjacent_bilateral": bool(adjacent_bilateral),
        "mean_error_m": float(mean_error),
    }


def _image_lattice_support(
    anchor,
    mask,
    occupancy,
    boundary_distance,
    combined_distance,
    object_depth,
    camera,
    scene,
    ignore,
):
    """Verify missing/occluded divider members directly in image evidence.

    A nearby divider can have too little valid depth to become a publishable
    line itself.  It still contributes family evidence if a shifted version of
    the anchor meets the same RGB boundary, blue-material, one-sidedness and
    high-object-region tests at the known metric divider pitch.
    """
    px_per_m = math.sqrt(float(camera["fx"]) * float(camera["fy"])) / object_depth
    spacing_px = BASKET_DIVIDER_SPACING_M * px_per_m
    if spacing_px < 10.0:
        return {
            "count": 1,
            "regularity": 0.0,
            "spacing_m": BASKET_DIVIDER_SPACING_M,
            "span_m": 0.0,
            "members": [],
            "mean_evidence": 0.0,
            "indices": [0],
            "bilateral": False,
            "adjacent_bilateral": False,
        }
    p0 = np.asarray(anchor["p0"], np.float64)
    p1 = np.asarray(anchor["p1"], np.float64)
    midpoint = 0.5 * (p0 + p1)
    half = 0.5 * (p1 - p0)
    normal = _normal(p0, p1)
    edge_tolerance = float(np.clip(0.0045 * px_per_m, 2.0, 10.0))

    object_height = None
    if scene and "height" in scene:
        height_map = np.asarray(scene["height"])
        values = height_map[(mask > 0) & np.isfinite(height_map)]
        if len(values) >= 30:
            object_height = (
                height_map,
                float(np.percentile(values, 8)),
                float(np.percentile(values, 94)),
            )

    evidence_rows = []
    for sign in (-1.0, 1.0):
        for step in range(1, 5):
            expected = sign * step * spacing_px
            best = None
            for residual_fraction in (-0.30, -0.15, 0.0, 0.15, 0.30):
                shift = expected + residual_fraction * spacing_px
                shifted_midpoint = midpoint + shift * normal
                for rotation_deg in (-15.0, -7.0, 0.0, 7.0, 15.0):
                    angle = math.radians(rotation_deg)
                    ca, sa = math.cos(angle), math.sin(angle)
                    rotated_half = np.array(
                        [ca * half[0] - sa * half[1], sa * half[0] + ca * half[1]],
                        np.float64,
                    )
                    q0 = shifted_midpoint - rotated_half
                    q1 = shifted_midpoint + rotated_half
                    points, _ = _line_points(q0, q1)
                    in_frame = (
                        (points[:, 0] >= 0)
                        & (points[:, 0] < mask.shape[1])
                        & (points[:, 1] >= 0)
                        & (points[:, 1] < mask.shape[0])
                    )
                    if float(np.mean(in_frame)) < 0.58:
                        continue
                    blocked, _ = _sample(ignore, points, default=255)
                    robot_clear = float(np.mean(in_frame & (blocked == 0)))
                    if robot_clear < 0.94:
                        continue
                    combined = _edge_support(
                        combined_distance, points, edge_tolerance
                    )
                    mask_edge = _edge_support(
                        boundary_distance, points, edge_tolerance
                    )
                    near_blue_rows = []
                    trial_normal = _normal(q0, q1)
                    for offset_m in (-0.006, -0.003, 0.0, 0.003, 0.006):
                        values, valid = _sample(
                            mask,
                            points + offset_m * px_per_m * trial_normal,
                            default=0,
                        )
                        near_blue_rows.append((values > 0) & valid)
                    near_blue = float(
                        np.mean(np.any(np.stack(near_blue_rows, axis=0), axis=0))
                    )
                    side = []
                    for direction in (-1.0, 1.0):
                        rows = []
                        for offset_m in (0.006, 0.011, 0.017):
                            values, valid = _sample(
                                occupancy,
                                points
                                + direction
                                * offset_m
                                * px_per_m
                                * trial_normal,
                                default=0,
                            )
                            rows.append((values > 0) & valid)
                        side.append(float(np.mean(np.stack(rows, axis=0))))
                    asymmetry = abs(side[0] - side[1])
                    height_score = 0.58
                    height_rank = None
                    if object_height is not None:
                        height_map, low_height, high_height = object_height
                        height_rows = []
                        for offset_m in (-0.005, 0.0, 0.005):
                            values, valid = _sample(
                                height_map,
                                points
                                + offset_m * px_per_m * trial_normal,
                                default=np.nan,
                            )
                            height_rows.append(np.where(valid, values, np.nan))
                        stacked = np.stack(height_rows, axis=0)
                        finite = stacked[np.isfinite(stacked)]
                        if len(finite) >= 8:
                            line_height = float(np.percentile(finite, 72))
                            height_rank = float(
                                np.clip(
                                    (line_height - low_height)
                                    / max(high_height - low_height, 0.025),
                                    0.0,
                                    1.0,
                                )
                            )
                            height_score = _ramp(height_rank, 0.22, 0.72)
                    evidence = (
                        0.34 * combined
                        + 0.18 * mask_edge
                        + 0.18 * near_blue
                        + 0.13 * _ramp(asymmetry, 0.04, 0.48)
                        + 0.17 * height_score
                    )
                    row = {
                        "step": int(step),
                        "sign": int(sign),
                        "residual_fraction": float(residual_fraction),
                        "rotation_deg": float(rotation_deg),
                        "evidence": float(evidence),
                        "combined_edge": combined,
                        "mask_edge": mask_edge,
                        "near_blue": near_blue,
                        "asymmetry": asymmetry,
                        "height_rank": height_rank,
                        "robot_clear_fraction": robot_clear,
                    }
                    if best is None or row["evidence"] > best["evidence"]:
                        best = row
            if best is not None and best["evidence"] >= 0.52:
                anchor_rank = float(anchor["height"]["rank"])
                minimum_member_rank = max(0.34, anchor_rank - 0.42)
                if (
                    object_height is None
                    or (
                        best["height_rank"] is not None
                        and best["height_rank"] >= minimum_member_rank
                    )
                ):
                    evidence_rows.append(best)

    # Opposite scan directions can hit the same physical plate.  A lattice
    # index relative to the anchor is unique, so retain one observation each.
    unique = {}
    for row in evidence_rows:
        key = row["sign"] * row["step"]
        if key not in unique or row["evidence"] > unique[key]["evidence"]:
            unique[key] = row
    # Pick one compact window containing the anchor.  Five physical plate
    # sites span at most four pitch intervals; evidence on both sides proves
    # that the anchor is an internal divider rather than an outer wall.
    best_window = None
    for window_low in range(-4, 1):
        window_high = window_low + 4
        rows = [
            row
            for index, row in unique.items()
            if window_low <= index <= window_high
        ]
        indices = sorted(row["sign"] * row["step"] for row in rows)
        bilateral = any(index < 0 for index in indices) and any(
            index > 0 for index in indices
        )
        adjacent_bilateral = -1 in indices and 1 in indices
        mean_evidence = (
            float(np.mean([row["evidence"] for row in rows])) if rows else 0.0
        )
        mean_residual = (
            float(np.mean([abs(row["residual_fraction"]) for row in rows]))
            if rows
            else 1.0
        )
        merit = (
            1.8 * len(rows)
            + 0.85 * float(bilateral)
            + 0.60 * float(adjacent_bilateral)
            + 0.45 * mean_evidence
            - 0.30 * mean_residual
        )
        row = (
            merit,
            rows,
            indices,
            bilateral,
            adjacent_bilateral,
            mean_evidence,
            mean_residual,
        )
        if best_window is None or row[0] > best_window[0]:
            best_window = row
    if best_window is None:
        members = []
        member_indices = []
        bilateral = False
        adjacent_bilateral = False
        mean_evidence = 0.0
        residual = 1.0
    else:
        (
            _,
            members,
            member_indices,
            bilateral,
            adjacent_bilateral,
            mean_evidence,
            residual,
        ) = best_window
        members = sorted(
            members, key=lambda row: row["sign"] * row["step"]
        )
    count = 1 + len(members)
    if members:
        regularity = float(
            0.38 * _ramp(count, 1.8, 4.0)
            + 0.22 * _ramp(0.30 - residual, 0.0, 0.30)
            + 0.24 * _ramp(mean_evidence, 0.52, 0.82)
            + 0.16 * float(adjacent_bilateral)
        )
        span_indices = [0, *member_indices]
        span_m = float(
            (max(span_indices) - min(span_indices))
            * BASKET_DIVIDER_SPACING_M
        )
    else:
        mean_evidence = 0.0
        regularity = 0.0
        span_m = 0.0
    return {
        "count": int(count),
        "regularity": float(regularity),
        "spacing_m": BASKET_DIVIDER_SPACING_M,
        "span_m": span_m,
        "members": members,
        "mean_evidence": mean_evidence,
        "indices": [0, *[int(value) for value in member_indices]],
        "bilateral": bool(bilateral),
        "adjacent_bilateral": bool(adjacent_bilateral),
    }


def _window_metrics(candidate, start, end, boundary_distance, z, ignore, camera):
    p0 = candidate["p0"] + start * (candidate["p1"] - candidate["p0"])
    p1 = candidate["p0"] + end * (candidate["p1"] - candidate["p0"])
    points, alpha = _line_points(p0, p1)
    depth = max(float(candidate["fit"]["center"][2]), 0.04)
    px_per_m = math.sqrt(float(camera["fx"]) * float(camera["fy"])) / depth
    edge_tolerance = float(np.clip(0.004 * px_per_m, 2.0, 9.0))
    edge = _edge_support(boundary_distance, points, edge_tolerance)
    normal = candidate["normal"]
    material_sign = candidate["material_sign"]
    sampled = np.full(len(points), np.nan, np.float64)
    for offset_m in (0.0025, 0.005, 0.008, 0.012):
        shifted = points + material_sign * offset_m * px_per_m * normal
        values, in_frame = _sample(
            z,
            shifted,
            default=np.nan,
        )
        blocked, _ = _sample(ignore, shifted, default=255)
        use = (
            np.isnan(sampled)
            & in_frame
            & np.isfinite(values)
            & (blocked == 0)
        )
        sampled[use] = values[use]
    valid = np.isfinite(sampled)
    valid_values = sampled[valid]
    valid_fraction = float(len(valid_values) / max(len(points), 1))
    if len(valid_values) >= 6:
        design = np.column_stack([np.ones(int(valid.sum())), alpha[valid]])
        model, _, _, _ = np.linalg.lstsq(design, valid_values, rcond=None)
        residual = valid_values - design @ model
        depth_range = float(
            np.percentile(residual, 90) - np.percentile(residual, 10)
        )
    else:
        depth_range = 1.0
    blocked, _ = _sample(ignore, points, default=255)
    clear = float(np.mean(blocked == 0))
    height_preference = 0.5
    endpoint_heights = candidate["height"].get("endpoint_heights_m")
    if endpoint_heights is not None:
        first_height, second_height = endpoint_heights
        start_height = first_height + start * (second_height - first_height)
        end_height = first_height + end * (second_height - first_height)
        mean_height = 0.5 * (start_height + end_height)
        low_height = min(first_height, second_height)
        high_height = max(first_height, second_height)
        height_preference = float(
            np.clip(
                (mean_height - low_height) / max(high_height - low_height, 0.006),
                0.0,
                1.0,
            )
        )
    score = (
        0.36 * edge
        + 0.22 * valid_fraction
        + 0.16 * _ramp(0.035 - depth_range, 0.0, 0.035)
        + 0.11 * clear
        + 0.15 * height_preference
    )
    return score, p0, p1, {
        "edge_support": edge,
        "valid_depth_fraction": valid_fraction,
        "depth_p90_p10_m": depth_range,
        "clear_fraction": clear,
        "height_preference": height_preference,
    }


def _terminal_crop(candidate, boundary_distance, z, ignore, camera):
    full_length = float(candidate["fit"]["length_m"])
    if full_length < GRASP_MIN_M:
        return None
    if full_length > 0.22:
        target = full_length / 3.0
        nominal_fraction = 1.0 / 3.0
    elif full_length > 0.13:
        target = full_length / 2.0
        nominal_fraction = 0.5
    else:
        target = full_length
        nominal_fraction = 1.0
    target = float(np.clip(target, GRASP_MIN_M, GRASP_MAX_M))
    # Prefer about 10.5 cm while preserving the user's 1/3 or 1/2 policy.
    if full_length >= GRASP_TARGET_M:
        target = min(target, max(GRASP_TARGET_M, 0.60 * target))
    fraction = float(np.clip(target / full_length, 0.0, 1.0))

    alternatives = []
    alternatives.append(
        (
            "p0_terminal",
            *_window_metrics(
                candidate, 0.0, fraction, boundary_distance, z, ignore, camera
            ),
            0.0,
            fraction,
        )
    )
    if fraction < 0.999:
        alternatives.append(
            (
                "p1_terminal",
                *_window_metrics(
                    candidate,
                    1.0 - fraction,
                    1.0,
                    boundary_distance,
                    z,
                    ignore,
                    camera,
                ),
                1.0 - fraction,
                1.0,
            )
        )
    alternatives.sort(key=lambda row: row[1], reverse=True)
    name, score, p0, p1, debug, alpha0, alpha1 = alternatives[0]
    if (
        debug["edge_support"] < 0.52
        or debug["clear_fraction"] < 0.995
        or debug["valid_depth_fraction"] < 0.12
    ):
        return None

    model = candidate["fit"]["alpha_xyz_model"]
    xyz0 = np.array([1.0, alpha0]) @ model
    xyz1_raw = np.array([1.0, alpha1]) @ model
    direction = xyz1_raw - xyz0
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-7:
        return None
    # Robust PCA length uses the supported 3--97% interval whereas the image
    # segment spans alpha 0--1.  Publish the explicitly selected metric target
    # along the observed direction, rather than leaking that estimator offset.
    length_m = target
    xyz1 = xyz0 + length_m * direction / direction_norm
    return {
        "p0": p0,
        "p1": p1,
        "xyz0": xyz0,
        "xyz1": xyz1,
        "length_m": length_m,
        "terminal": name,
        "terminal_fraction": fraction,
        "nominal_fraction": nominal_fraction,
        "score": float(score),
        "debug": debug,
    }


def _candidate_debug(candidate, rank=None):
    family = candidate.get("family", {})
    row = {
        "rank": rank,
        "score": round(float(candidate.get("score", 0.0)), 5),
        "base_score": round(float(candidate["base_score"]), 5),
        "endpoints_uv": [
            [round(float(value), 2) for value in candidate["p0"]],
            [round(float(value), 2) for value in candidate["p1"]],
        ],
        "full_length_px": round(float(candidate["pixel_length"]), 2),
        "full_length_m": round(float(candidate["fit"]["length_m"]), 5),
        "rms_mm": round(1000.0 * float(candidate["fit"]["rms_m"]), 3),
        "depth_support_fraction": round(
            float(candidate["fit"]["valid_fraction"]), 4
        ),
        "mask_edge_support": round(float(candidate["mask_edge_support"]), 4),
        "external_edge_support": round(
            float(candidate["external_edge_support"]), 4
        ),
        "combined_edge_support": round(
            float(candidate["combined_edge_support"]), 4
        ),
        "sidedness": round(float(candidate["sidedness"]), 4),
        "material_occupancy": round(float(candidate["material_occupancy"]), 4),
        "free_occupancy": round(float(candidate["free_occupancy"]), 4),
        "free_depth": round(float(candidate["free_depth"]), 4),
        "gradient_coverage": round(float(candidate["gradient_coverage"]), 4),
        "height_rank": round(float(candidate["height"]["rank"]), 4),
        "height_m": (
            None
            if candidate["height"]["height_m"] is None
            else round(float(candidate["height"]["height_m"]), 5)
        ),
        "inward_height_drop_m": (
            None
            if candidate["height"].get("inward_drop_m") is None
            else round(float(candidate["height"]["inward_drop_m"]), 5)
        ),
        "family_count": int(family.get("count", 0)),
        "family_regularity": round(float(family.get("regularity", 0.0)), 4),
        "family_spacing_m": (
            None
            if family.get("spacing_m") is None
            else round(float(family["spacing_m"]), 5)
        ),
        "family_span_m": round(float(family.get("span_m", 0.0)), 5),
        "family_source": candidate.get("family_source"),
        "gate_reason": candidate.get("gate_reason"),
        "continuous_free_run_m": round(
            float(candidate.get("continuous_free_run_m", 0.0)), 5
        ),
        "continuous_free_run_fraction": round(
            float(candidate.get("continuous_free_run_fraction", 0.0)), 4
        ),
    }
    return row




# ---------------------------------------------------------------------------
# Whole-object five-ridge solver
# ---------------------------------------------------------------------------
#
# The public function is redefined below to use this solver exclusively.  A
# local-line-first strategy is unsafe for this perforated object: diagonals
# through circular holes can look straight, blue and depth-consistent in a
# short window.  Here a line is not even eligible until the image contains a
# complete five-ridge projective family and at least three of those ridges form
# the specified metric 3-D lattice.


def _sf_angle_distance_deg(first, second):
    delta = abs((float(first) - float(second)) % 180.0)
    return min(delta, 180.0 - delta)


def _sf_smooth_profile(values, sigma=2.0):
    radius = max(2, int(math.ceil(3.0 * float(sigma))))
    locations = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (locations / max(float(sigma), 1e-6)) ** 2)
    kernel /= float(kernel.sum())
    return np.convolve(np.asarray(values, np.float64), kernel, mode="same")


def _sf_profile_peaks(profile, min_distance, min_prominence):
    profile = np.asarray(profile, np.float64)
    distance = max(1, int(min_distance))
    padded = np.pad(
        profile,
        (distance, distance),
        mode="constant",
        constant_values=np.inf,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, distance + 1
    )
    left_min = np.min(windows[: len(profile)], axis=1)
    right_min = np.min(
        windows[distance : distance + len(profile)], axis=1
    )
    local_maximum = np.zeros(len(profile), dtype=bool)
    local_maximum[1:-1] = (
        (profile[1:-1] >= profile[:-2])
        & (profile[1:-1] > profile[2:])
    )
    prominent = (
        profile - np.maximum(left_min, right_min)
        >= float(min_prominence)
    )
    raw = np.flatnonzero(local_maximum & prominent)
    raw = raw[np.argsort(profile[raw])[::-1]].tolist()
    selected = []
    for index in raw:
        if all(abs(index - other) >= distance for other in selected):
            selected.append(index)
    return np.asarray(sorted(selected), np.int32)


def _sf_edge_evidence(rgb, mask, ignore_clearance, object_depth, camera):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    blue, _, red = cv2.split(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blue_score = np.clip(
        2 * (blue.astype(np.int16) - red.astype(np.int16))
        + hsv[:, :, 1].astype(np.int16) // 2,
        0,
        255,
    ).astype(np.uint8)
    blue_score = cv2.GaussianBlur(blue_score, (3, 3), 0)
    region_radius = _metric_pixels(camera, object_depth, 0.008, 3, 18)
    region = cv2.dilate(
        (mask > 0).astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * region_radius + 1, 2 * region_radius + 1),
        ),
    )
    gray_gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gray_gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    blue_gx = cv2.Sobel(blue_score, cv2.CV_32F, 1, 0, ksize=3)
    blue_gy = cv2.Sobel(blue_score, cv2.CV_32F, 0, 1, ksize=3)
    gray_magnitude = cv2.magnitude(gray_gx, gray_gy)
    blue_magnitude = cv2.magnitude(blue_gx, blue_gy)

    usable = (region > 0) & (ignore_clearance == 0)

    def adaptive_canny(image, magnitude):
        values = magnitude[usable & np.isfinite(magnitude) & (magnitude > 1.0)]
        if values.size < 80:
            return np.zeros(mask.shape, np.uint8)
        high = float(np.clip(np.percentile(values, 68), 42.0, 155.0))
        low = float(np.clip(0.38 * high, 14.0, 62.0))
        return cv2.Canny(image, low, high)

    edges = adaptive_canny(gray, gray_magnitude) | adaptive_canny(
        blue_score, blue_magnitude
    )
    edges &= region
    edges[ignore_clearance > 0] = 0

    # A neutral metal gripper has almost no blue-chroma gradient.  Using only
    # that gradient makes atan2(0, 0) look like a valid 90 degree tangent.
    # Select the stronger measured gradient at every pixel and invalidate
    # genuinely directionless responses instead.
    use_blue = blue_magnitude > gray_magnitude
    gx = np.where(use_blue, blue_gx, gray_gx)
    gy = np.where(use_blue, blue_gy, gray_gy)
    orientation_strength = np.maximum(gray_magnitude, blue_magnitude)
    direction_valid = np.isfinite(orientation_strength) & (
        orientation_strength >= 12.0
    )
    tangent = np.full(mask.shape, np.nan, np.float64)
    tangent[direction_valid] = (
        np.degrees(np.arctan2(gy[direction_valid], gx[direction_valid]))
        + 90.0
    ) % 180.0
    edges[~direction_valid] = 0
    return edges.astype(np.uint8), tangent


def _sf_family_candidates(edges, mask, bbox, object_depth, camera):
    edge_v, edge_u = np.nonzero(edges > 0)
    blob_v, blob_u = np.nonzero(mask > 0)
    if len(edge_u) < 80 or len(blob_u) < 120:
        return []
    # The detector bbox may be loose or split differently after a lighting
    # change.  Normalize by the proposal mask's own tight extent instead.
    x1 = int(blob_u.min())
    y1 = int(blob_v.min())
    x2 = int(blob_u.max()) + 1
    y2 = int(blob_v.max()) + 1
    center = np.array([0.5 * (x1 + x2 - 1), 0.5 * (y1 + y2 - 1)])
    edge_xy = np.column_stack([edge_u, edge_v]).astype(np.float64) - center
    # Projection extrema over a binary component are attained on its convex
    # hull.  Using only those vertices preserves every angle's exact support
    # span while avoiding 180 projections of every interior mask pixel.
    blob_points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if blob_points is None:
        return []
    blob_hull = cv2.convexHull(blob_points).reshape(-1, 2)
    blob_xy = blob_hull.astype(np.float64) - center
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    candidates = []
    # A one-degree sweep duplicates nearly identical Hough profiles while
    # doubling the most expensive part of the basket search.  Two-degree
    # sampling still bounds orientation quantisation to one degree; the
    # subsequent per-ridge fit snaps the family back to observed edges.
    for angle in range(0, 180):
        theta = math.radians(float(angle))
        normal = np.array([-math.sin(theta), math.cos(theta)])
        rho = edge_xy @ normal
        blob_rho = blob_xy @ normal
        low = float(math.floor(float(blob_rho.min())) - 2)
        high = float(math.ceil(float(blob_rho.max())) + 2)
        count = int(high - low + 1)
        if count < 24:
            continue
        bins = np.clip(np.rint(rho - low).astype(np.int32), 0, count - 1)
        profile = np.bincount(bins, minlength=count).astype(np.float64)
        profile = _sf_smooth_profile(profile, 2.0)
        span = float(blob_rho.max() - blob_rho.min())
        min_distance = max(6, int(round(0.075 * span)))
        peaks = _sf_profile_peaks(
            profile,
            min_distance=min_distance,
            min_prominence=max(2.0, 0.040 * float(profile.max())),
        )
        if len(peaks) < 5:
            continue
        best = None
        for start, end in itertools.combinations(peaks, 2):
            spread = float(end - start)
            if not (0.52 * span <= spread <= 1.08 * span):
                continue
            apparent_span_m = spread * float(object_depth) / max(focal, 1e-6)
            if not (0.15 <= apparent_span_m <= 0.47):
                continue
            step = spread / 4.0
            chosen = []
            regularity_error = 0.0
            for ridge_index in range(5):
                target = float(start) + ridge_index * step
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
            strong_count = int(
                sum(profile[index] > 0.45 * median_strength for index in peaks)
            )
            if strong_count < 5:
                continue
            balance = float(
                np.percentile(strengths, 20)
                / max(float(np.percentile(strengths, 80)), 1e-6)
            )
            physical_prior = _bell(apparent_span_m, BASKET_DIVIDER_SPACING_M * 4.0, 0.105)
            structural = (
                float(strengths.mean())
                / max(float(max(x2 - x1, y2 - y1)), 1.0)
                * (spread / max(span, 1.0))
                * math.exp(-regularity_error)
                * (0.70 + 0.30 * balance)
                * (0.55 + 0.45 * physical_prior)
                / (1.0 + 0.22 * max(0, strong_count - 5))
            )
            item = {
            "angle": float(angle),
                "offsets": [float(low + index) for index in chosen],
                "structural": float(structural),
                "regularity_error": float(regularity_error),
                "strong_peak_count": int(strong_count),
                "balance": balance,
                "apparent_span_m": apparent_span_m,
                "center": center,
            }
            if best is None or item["structural"] > best["structural"]:
                best = item
        if best is not None:
            candidates.append(best)
    candidates.sort(key=lambda item: item["structural"], reverse=True)
    # Adjacent integer angles describe the same family.  Retain diverse
    # hypotheses before the more expensive per-ridge depth fit.
    diverse = []
    for candidate in candidates:
        if any(
            _sf_angle_distance_deg(candidate["angle"], other["angle"]) < 2.5
            for other in diverse
        ):
            continue
        diverse.append(candidate)
        if len(diverse) >= 18:
            break
    return diverse


def _sf_line_metrics(distance, p0, p1, tolerance_px):
    points, _ = _sf_search_line_points(p0, p1)
    values, valid = _sample(distance, points, default=1e6)
    values = np.asarray(values, np.float64)
    good = valid & (values <= float(tolerance_px))
    far = (~valid) | (values > max(4.5, 1.55 * float(tolerance_px)))
    longest_far = _longest_true_run(far)
    length_px = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    return {
        "values": values,
        "valid": valid,
        "good": good,
        "length_px": length_px,
        "p90_px": float(np.percentile(values, 90)),
        "within_tolerance": float(np.mean(good)),
        "max_air_run_px": float(longest_far * length_px / max(len(points) - 1, 1)),
    }


def _sf_search_line_points(p0, p1):
    """Dense samples used by the repeated-family edge-continuity gates."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    length = float(np.linalg.norm(p1 - p0))
    count = max(18, int(math.ceil(1.15 * length)))
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return p0[None, :] + alpha[:, None] * (p1 - p0)[None, :], alpha


def _sf_snap_and_trim(
    distance,
    p0,
    p1,
    tolerance_px,
    min_run_px,
    max_gap_px,
    ignore_clearance,
):
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length < float(min_run_px):
        return None
    direction = vector / length
    normal = np.array([-direction[1], direction[0]])
    max_offset = int(np.clip(round(1.8 * tolerance_px), 4, 14))
    offsets = np.arange(-max_offset, max_offset + 1, dtype=np.float64)
    base_points, _ = _sf_search_line_points(p0, p1)
    offset_points = (
        base_points[None, :, :]
        + offsets[:, None, None] * normal[None, None, :]
    )
    sampled, sampled_valid = _sample(
        distance,
        offset_points.reshape(-1, 2),
        default=1e6,
    )
    values_by_offset = np.asarray(sampled, np.float64).reshape(
        len(offsets), len(base_points)
    )
    valid_by_offset = sampled_valid.reshape(len(offsets), len(base_points))
    good_by_offset = valid_by_offset & (
        values_by_offset <= float(tolerance_px)
    )
    far_by_offset = (~valid_by_offset) | (
        values_by_offset > max(4.5, 1.55 * float(tolerance_px))
    )
    current_run = np.zeros(len(offsets), np.int32)
    longest_run = np.zeros(len(offsets), np.int32)
    for column in range(far_by_offset.shape[1]):
        current_run = np.where(
            far_by_offset[:, column], current_run + 1, 0
        )
        longest_run = np.maximum(longest_run, current_run)
    p90_by_offset = np.percentile(values_by_offset, 90, axis=1)
    within_by_offset = np.mean(good_by_offset, axis=1)
    max_air_by_offset = (
        longest_run.astype(np.float64)
        * length
        / max(len(base_points) - 1, 1)
    )
    scores = (
        1.5 * p90_by_offset
        + 7.0 * (1.0 - within_by_offset)
        + 0.16 * max_air_by_offset
        + 0.05 * np.abs(offsets)
    )
    best_index = int(np.argmin(scores))
    normal_offset = float(offsets[best_index])
    q0 = p0 + normal_offset * normal
    q1 = p1 + normal_offset * normal
    metrics = {
        "values": values_by_offset[best_index],
        "valid": valid_by_offset[best_index],
        "good": good_by_offset[best_index],
        "length_px": length,
        "p90_px": float(p90_by_offset[best_index]),
        "within_tolerance": float(within_by_offset[best_index]),
        "max_air_run_px": float(max_air_by_offset[best_index]),
    }
    points, alpha = _sf_search_line_points(q0, q1)
    blocked, _ = _sample(ignore_clearance, points, default=255)
    good = metrics["good"] & (blocked == 0)
    index = 0
    while index < len(good):
        if good[index]:
            index += 1
            continue
        end = index
        while end < len(good) and not good[end]:
            end += 1
        if index > 0 and end < len(good) and end - index <= int(max_gap_px):
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
        run_px = float((alpha[end - 1] - alpha[index]) * length)
        if run_px >= float(min_run_px):
            runs.append((index, end - 1, run_px))
        index = end
    if not runs:
        return None
    start, end, _ = max(runs, key=lambda item: item[2])
    r0 = q0 + float(alpha[start]) * (q1 - q0)
    r1 = q0 + float(alpha[end]) * (q1 - q0)
    final_metrics = _sf_line_metrics(distance, r0, r1, tolerance_px)
    final_points, _ = _sf_search_line_points(r0, r1)
    final_blocked, _ = _sample(ignore_clearance, final_points, default=255)
    if (
        final_metrics["within_tolerance"] < 0.82
        or final_metrics["p90_px"] > 1.55 * float(tolerance_px)
        or final_metrics["max_air_run_px"] > max(8.0, 2.2 * float(max_gap_px))
        or float(np.mean(final_blocked == 0)) < 0.995
    ):
        return None
    return {
        "p0": r0,
        "p1": r1,
        "edge_metrics": final_metrics,
        "normal_offset_px": normal_offset,
    }


def _sf_fit_ridge(
    family,
    ridge_index,
    family_edge_points,
    family_edge_rho,
    edge_distance,
    tolerance_px,
    min_run_px,
    max_gap_px,
    ignore_clearance,
):
    angle = float(family["angle"])
    theta = math.radians(angle)
    expected_direction = np.array([math.cos(theta), math.sin(theta)])
    offset = float(family["offsets"][ridge_index])
    family_gaps = np.diff(np.asarray(family["offsets"], np.float64))
    search_width = float(
        np.clip(0.12 * float(np.median(family_gaps)), 5.0, 22.0)
    )
    selected = np.abs(family_edge_rho - offset) <= search_width
    points = family_edge_points[selected]
    if len(points) < 16:
        return None

    keep = np.ones(len(points), dtype=bool)
    fitted_center = points.mean(axis=0)
    fitted_direction = expected_direction.copy()
    for _ in range(3):
        sample = points[keep]
        if len(sample) < 12:
            return None
        fitted_center = sample.mean(axis=0)
        covariance = (sample - fitted_center).T @ (sample - fitted_center)
        _, vectors = np.linalg.eigh(covariance)
        fitted_direction = vectors[:, -1]
        fitted_angle = (
            math.degrees(
                math.atan2(float(fitted_direction[1]), float(fitted_direction[0]))
            )
            % 180.0
        )
        if _sf_angle_distance_deg(fitted_angle, angle) > 24.0:
            return None
        fitted_normal = np.array([-fitted_direction[1], fitted_direction[0]])
        residual = np.abs((points - fitted_center) @ fitted_normal)
        threshold = float(
            np.clip(np.percentile(residual[keep], 68), 2.0, 4.2)
        )
        new_keep = residual <= threshold
        if int(new_keep.sum()) < 12:
            break
        keep = new_keep
    support = points[keep]
    if len(support) < 12:
        return None
    if float(fitted_direction @ expected_direction) < 0:
        fitted_direction = -fitted_direction
    locations = (support - fitted_center) @ fitted_direction
    low, high = np.percentile(locations, (2, 98))
    if float(high - low) < float(min_run_px):
        return None
    p0 = fitted_center + float(low) * fitted_direction
    p1 = fitted_center + float(high) * fitted_direction
    snapped = _sf_snap_and_trim(
        edge_distance,
        p0,
        p1,
        tolerance_px,
        min_run_px,
        max_gap_px,
        ignore_clearance,
    )
    if snapped is None:
        return None
    p0 = snapped["p0"]
    p1 = snapped["p1"]
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    direction = vector / max(length, 1e-9)
    support_along = (support - p0) @ direction
    local = (support_along >= -3.0) & (support_along <= length + 3.0)
    if int(local.sum()) >= 10:
        support = support[local]
    residual = np.abs((support - p0) @ np.array([-direction[1], direction[0]]))
    return {
        "index": int(ridge_index),
        "p0": p0,
        "p1": p1,
        "angle": float(
            math.degrees(math.atan2(float(direction[1]), float(direction[0])))
            % 180.0
        ),
        "edge_rms_px": float(np.sqrt(np.mean(residual**2))),
        "edge_metrics": snapped["edge_metrics"],
        "normal_offset_px": float(snapped["normal_offset_px"]),
        "support_count": int(len(support)),
    }


def _sf_build_cloud(z, mask, ignore_clearance, camera, scene):
    valid = (
        (mask > 0)
        & np.isfinite(z)
        & (z > 0.05)
        & (ignore_clearance == 0)
    )
    if int(np.count_nonzero(valid)) < 80:
        return None
    # Keep aligned dense arrays.  Each ridge accesses only its narrow bounding
    # strip, so building and repeatedly projecting a full-object point cloud is
    # unnecessary and disproportionately expensive at wrist-camera resolution.
    return {"z": np.asarray(z, np.float32), "valid": valid}


def _sf_lift_ridge(line, cloud, camera):
    p0 = np.asarray(line["p0"], np.float64)
    p1 = np.asarray(line["p1"], np.float64)
    vector = p1 - p0
    length_px = float(np.linalg.norm(vector))
    if length_px < 18.0:
        return None
    direction2 = vector / length_px
    normal2 = np.array([-direction2[1], direction2[0]])
    # Stay on the physical ridge/material lip.  A wider symmetric strip often
    # contains the floor behind a perforation on one side and a plate face on
    # the other, causing different ridges to acquire incompatible 3-D slopes.
    band_px = float(np.clip(0.025 * length_px, 3.0, 6.0))
    wide_band_px = min(9.0, 1.45 * band_px)
    extended_p0 = p0 - 0.06 * vector
    extended_p1 = p1 + 0.06 * vector
    margin = int(math.ceil(wide_band_px)) + 2
    valid_map = cloud["valid"]
    image_height, image_width = valid_map.shape
    x0 = max(
        0,
        int(math.floor(min(float(extended_p0[0]), float(extended_p1[0]))))
        - margin,
    )
    y0 = max(
        0,
        int(math.floor(min(float(extended_p0[1]), float(extended_p1[1]))))
        - margin,
    )
    x1 = min(
        image_width,
        int(math.ceil(max(float(extended_p0[0]), float(extended_p1[0]))))
        + margin
        + 1,
    )
    y1 = min(
        image_height,
        int(math.ceil(max(float(extended_p0[1]), float(extended_p1[1]))))
        + margin
        + 1,
    )
    local_v, local_u = np.nonzero(valid_map[y0:y1, x0:x1])
    if len(local_u) < 12:
        return None
    sample_u = local_u + x0
    sample_v = local_v + y0
    sample_uv = np.column_stack([sample_u, sample_v]).astype(np.float64)
    sample_depth = cloud["z"][sample_v, sample_u].astype(np.float64)
    relative = sample_uv - p0
    along_px = relative @ direction2
    lateral_px = np.abs(relative @ normal2)
    selected = (
        (along_px >= -0.05 * length_px)
        & (along_px <= 1.05 * length_px)
        & (lateral_px <= band_px)
    )
    if int(selected.sum()) < 18:
        selected = (
            (along_px >= -0.06 * length_px)
            & (along_px <= 1.06 * length_px)
            & (lateral_px <= wide_band_px)
        )
    if int(selected.sum()) < 12:
        return None
    raw_alpha = along_px[selected] / length_px
    raw_uv = sample_uv[selected]
    raw_points = _unproject(raw_uv, sample_depth[selected], camera)

    # Build one near-material depth representative per longitudinal bin.  Raw
    # point counts are dominated by whichever perforated face occupies more
    # pixels, which can make adjacent ridges choose different depth layers.
    # Equal-width bins plus a robust near quantile follow the exposed plastic
    # lip while isolated flying pixels are removed by the later line RANSAC.
    bin_count = int(np.clip(round(length_px / 8.0), 24, 72))
    bin_ids = np.floor(
        (raw_alpha + 0.06) * bin_count / 1.12
    ).astype(np.int32)
    usable_bins = (bin_ids >= 0) & (bin_ids < bin_count)
    order = np.argsort(bin_ids[usable_bins], kind="stable")
    sorted_bin = bin_ids[usable_bins][order]
    sorted_alpha = raw_alpha[usable_bins][order]
    sorted_depth = raw_points[usable_bins, 2][order]
    bin_counts = np.bincount(
        sorted_bin, minlength=bin_count
    )
    occupied = bin_counts >= 2
    if int(np.count_nonzero(occupied)) < 10:
        return None
    max_bin_count = int(bin_counts.max())
    bin_starts = np.cumsum(bin_counts) - bin_counts
    within_bin = np.arange(len(sorted_bin)) - np.repeat(
        bin_starts, bin_counts
    )
    alpha_grid = np.full(
        (bin_count, max_bin_count), np.inf, np.float64
    )
    depth_grid = np.full_like(alpha_grid, np.inf)
    alpha_grid[sorted_bin, within_bin] = sorted_alpha
    depth_grid[sorted_bin, within_bin] = sorted_depth
    present = (
        np.arange(max_bin_count)[None, :] < bin_counts[:, None]
    )
    ordered_depth = np.sort(depth_grid, axis=1)
    near_limit = np.full(bin_count, np.inf, np.float64)
    quantile_rows = np.flatnonzero(bin_counts >= 5)
    quantile_position = 0.38 * (bin_counts[quantile_rows] - 1)
    quantile_low = np.floor(quantile_position).astype(np.int32)
    quantile_high = np.ceil(quantile_position).astype(np.int32)
    quantile_fraction = quantile_position - quantile_low
    near_limit[quantile_rows] = (
        ordered_depth[quantile_rows, quantile_low]
        + quantile_fraction
        * (
            ordered_depth[quantile_rows, quantile_high]
            - ordered_depth[quantile_rows, quantile_low]
        )
    )
    near = present & (depth_grid <= near_limit[:, None])
    near_count = np.sum(near, axis=1).astype(np.int32)
    near_alpha = np.sort(np.where(near, alpha_grid, np.inf), axis=1)
    median_low = (near_count - 1) // 2
    median_high = near_count // 2
    rows = np.arange(bin_count)
    representative_alpha_all = 0.5 * (
        near_alpha[rows, median_low] + near_alpha[rows, median_high]
    )
    representative_depth_all = 0.5 * (
        ordered_depth[rows, median_low]
        + ordered_depth[rows, median_high]
    )
    alpha = representative_alpha_all[occupied]
    representative_depth = representative_depth_all[occupied]
    representative_weight = near_count[occupied]
    representative_uv = p0[None, :] + alpha[:, None] * vector[None, :]
    points = _unproject(representative_uv, representative_depth, camera)
    inv_z = 1.0 / np.maximum(points[:, 2], 1e-6)
    seed = int(round(p0[0] * 13.0 + p0[1] * 17.0)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    # The longitudinal bins already regularise the observations.  Two
    # deterministic pairs per bin are enough to seed the subsequent three
    # robust least-squares refinements; the former four-per-bin cap mostly
    # duplicated nearly identical inverse-depth hypotheses.
    trial_count = min(96, max(45, 2 * len(points)))
    # Generate the same deterministic hypotheses as the former scalar RANSAC,
    # then score them as one small matrix.  This keeps regression outputs stable
    # while removing Python work from every hypothesis/point comparison.
    pairs = []
    for _ in range(trial_count):
        pair = rng.choice(len(points), 2, replace=False)
        if abs(float(alpha[pair[1]]) - float(alpha[pair[0]])) >= 0.16:
            pairs.append(pair)
    if not pairs:
        return None
    pairs = np.asarray(pairs, np.int32)
    pair_first = pairs[:, 0]
    pair_second = pairs[:, 1]

    first_alpha = alpha[pair_first]
    second_alpha = alpha[pair_second]
    slopes = (
        inv_z[pair_second] - inv_z[pair_first]
    ) / (second_alpha - first_alpha)
    intercepts = inv_z[pair_first] - slopes * first_alpha
    predicted_inverse = (
        slopes[:, None] * alpha[None, :] + intercepts[:, None]
    )
    predicted_depth = np.full(predicted_inverse.shape, np.inf, np.float64)
    positive = predicted_inverse > 0.15
    predicted_depth[positive] = 1.0 / predicted_inverse[positive]
    inlier_matrix = (
        np.abs(points[None, :, 2] - predicted_depth) <= 0.014
    )
    inlier_count = np.sum(inlier_matrix, axis=1)
    alpha_low = np.min(
        np.where(inlier_matrix, alpha[None, :], np.inf), axis=1
    )
    alpha_high = np.max(
        np.where(inlier_matrix, alpha[None, :], -np.inf), axis=1
    )
    coverage_by_model = alpha_high - alpha_low

    endpoint_inverse = (
        slopes[:, None] * np.array([0.0, 1.0])[None, :]
        + intercepts[:, None]
    )
    endpoint_depth = 1.0 / np.maximum(endpoint_inverse, 0.15)
    endpoint_u = np.column_stack(
        [
            np.full(len(slopes), float(p0[0])),
            np.full(len(slopes), float(p1[0])),
        ]
    )
    endpoint_v = np.column_stack(
        [
            np.full(len(slopes), float(p0[1])),
            np.full(len(slopes), float(p1[1])),
        ]
    )
    endpoint_x = (
        (endpoint_u - float(camera["cx"]))
        * endpoint_depth
        / float(camera["fx"])
    )
    endpoint_y = (
        (endpoint_v - float(camera["cy"]))
        * endpoint_depth
        / float(camera["fy"])
    )
    length_by_model = np.sqrt(
        np.diff(endpoint_x, axis=1)[:, 0] ** 2
        + np.diff(endpoint_y, axis=1)[:, 0] ** 2
        + np.diff(endpoint_depth, axis=1)[:, 0] ** 2
    )
    model_valid = (
        (inlier_count >= 10)
        & (coverage_by_model >= 0.34)
        & (length_by_model >= FULL_EDGE_MIN_M)
        & (length_by_model <= FULL_EDGE_MAX_M)
    )
    if not bool(np.any(model_valid)):
        return None
    model_score = (
        inlier_count.astype(np.float64)
        + 22.0 * np.minimum(coverage_by_model, 1.0)
    )
    model_score[~model_valid] = -np.inf
    best_index = int(np.argmax(model_score))
    keep = inlier_matrix[best_index]
    slope = float(slopes[best_index])
    intercept = float(intercepts[best_index])
    for _ in range(3):
        design = np.column_stack([alpha[keep], np.ones(int(keep.sum()))])
        slope, intercept = np.linalg.lstsq(design, inv_z[keep], rcond=None)[0]
        predicted_inv = slope * alpha + intercept
        predicted_z = np.full(len(points), np.inf, np.float64)
        usable = predicted_inv > 0.15
        predicted_z[usable] = 1.0 / predicted_inv[usable]
        residual = np.abs(points[:, 2] - predicted_z)
        threshold = float(
            np.clip(np.percentile(residual[keep], 78) * 1.8, 0.006, 0.018)
        )
        new_keep = residual <= threshold
        if int(new_keep.sum()) < 10:
            break
        keep = new_keep
    coverage = float(np.ptp(alpha[keep]))
    if int(keep.sum()) < 10 or coverage < 0.34:
        return None
    endpoint_depths = 1.0 / np.maximum(
        slope * np.array([0.0, 1.0]) + intercept, 0.15
    )
    endpoints = _unproject(np.stack([p0, p1]), endpoint_depths, camera)
    length_m = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    if not (FULL_EDGE_MIN_M <= length_m <= FULL_EDGE_MAX_M):
        return None
    direction3 = endpoints[1] - endpoints[0]
    direction3 /= np.linalg.norm(direction3) + 1e-12
    residual3 = np.linalg.norm(
        (points[keep] - endpoints[0])
        - np.outer((points[keep] - endpoints[0]) @ direction3, direction3),
        axis=1,
    )
    rms = float(np.sqrt(np.mean(residual3**2)))
    if rms > 0.018:
        return None
    return {
        "p0": endpoints[0],
        "p1": endpoints[1],
        "center": 0.5 * (endpoints[0] + endpoints[1]),
        "direction": direction3,
        "length_m": length_m,
        "rms_m": rms,
        "n_support": int(representative_weight[keep].sum()),
        "valid_fraction": float(
            representative_weight[keep].sum() / max(int(selected.sum()), 1)
        ),
        "coverage": coverage,
        "inverse_depth_model": [float(slope), float(intercept)],
    }


def _sf_validate_family(family, fitted):
    def reject(reason, **values):
        family["metric_failure"] = {"reason": reason, **values}
        return None

    visible = [item for item in fitted if item.get("lift") is not None]
    if len(visible) < 3:
        return reject("fewer_than_three_direct_3d", visible=len(visible))
    fitted = sorted(fitted, key=lambda item: int(item["index"]))
    fitted_indices = np.asarray(
        [int(item["index"]) for item in fitted], np.int32
    )
    expected_offsets = np.asarray(family["offsets"], np.float64)[
        fitted_indices
    ]
    theta = math.radians(float(family["angle"]))
    normal2 = np.array([-math.sin(theta), math.cos(theta)])
    center2 = np.asarray(family["center"], np.float64)
    snapped_offsets = np.asarray(
        [
            (0.5 * (item["p0"] + item["p1"]) - center2) @ normal2
            for item in fitted
        ],
        np.float64,
    )
    index_steps = np.diff(fitted_indices).astype(np.float64)
    raw_gap2 = np.diff(snapped_offsets)
    if np.any(index_steps <= 0) or np.any(raw_gap2 <= 8.0 * index_steps):
        return reject("pixel_offsets_not_ordered", gaps=raw_gap2.tolist())
    gap2 = raw_gap2 / index_steps
    gap2_cv = float(np.std(gap2) / max(float(np.mean(gap2)), 1e-6))
    max_drift = float(np.max(np.abs(snapped_offsets - expected_offsets)))
    if gap2_cv > 0.38 or max_drift > max(15.0, 0.13 * float(np.median(gap2))):
        return reject(
            "pixel_lattice_irregular",
            gap_cv=gap2_cv,
            max_drift=max_drift,
            gaps=gap2.tolist(),
        )

    # A perforated face can occasionally give one ridge the background depth
    # layer.  Keep the largest mutually parallel direct-3D consensus instead
    # of allowing that single bad lift to veto four physically coherent lines.
    parallel_cosine = math.cos(math.radians(28.0))
    consensus = None
    consensus_score = -float("inf")
    for count in range(len(visible), 2, -1):
        for subset in itertools.combinations(visible, count):
            reference = np.asarray(subset[0]["lift"]["direction"], np.float64)
            directions_here = []
            for item in subset:
                direction = np.asarray(item["lift"]["direction"], np.float64)
                if float(direction @ reference) < 0:
                    direction = -direction
                directions_here.append(direction)
            pair_cosines = [
                float(first @ second)
                for first, second in itertools.combinations(directions_here, 2)
            ]
            if pair_cosines and min(pair_cosines) < parallel_cosine:
                continue
            score = float(np.mean(pair_cosines)) if pair_cosines else 1.0
            if score > consensus_score:
                consensus = (list(subset), directions_here)
                consensus_score = score
        if consensus is not None:
            break
    if consensus is None:
        return reject("fewer_than_three_parallel_direct_3d_ridges")
    visible, directions = consensus
    family_direction = np.mean(np.stack(directions, axis=0), axis=0)
    family_direction /= np.linalg.norm(family_direction) + 1e-12
    visible.sort(key=lambda item: int(item["index"]))
    normalized_gaps = []
    for first, second in zip(visible[:-1], visible[1:]):
        index_step = int(second["index"]) - int(first["index"])
        if index_step <= 0:
            return reject("ridge_indices_not_ordered")
        delta = second["lift"]["center"] - first["lift"]["center"]
        lateral = delta - float(delta @ family_direction) * family_direction
        normalized_gaps.append(float(np.linalg.norm(lateral) / index_step))
    gaps = np.asarray(normalized_gaps, np.float64)
    if len(gaps) < 2:
        return reject("fewer_than_two_direct_3d_gaps")
    pair_gap_cv = float(np.std(gaps) / max(float(np.mean(gaps)), 1e-6))
    # A plausible first/last span must not hide a zig-zag set of unrelated
    # depth lines.  Every directly observed pitch remains physically bounded.
    if (
        pair_gap_cv > 0.42
        or float(np.min(gaps)) < 0.040
        or float(np.max(gaps)) > 0.145
        or float(np.max(np.abs(gaps - BASKET_DIVIDER_SPACING_M))) > 0.070
    ):
        return reject(
            "individual_3d_pitch_irregular",
            pair_gap_cv=pair_gap_cv,
            gaps=gaps.tolist(),
        )
    first_center = visible[0]["lift"]["center"]
    last_center = visible[-1]["lift"]["center"]
    span_delta = last_center - first_center
    span_lateral = span_delta - float(span_delta @ family_direction) * family_direction
    physical_span = float(np.linalg.norm(span_lateral))
    index_span = int(visible[-1]["index"]) - int(visible[0]["index"])
    expected_span = float(index_span * BASKET_DIVIDER_SPACING_M)
    span_tolerance = max(0.045, 0.18 * expected_span)
    mean_gap = physical_span / max(index_span, 1)
    if (
        not (0.045 <= mean_gap <= 0.105)
        or abs(physical_span - expected_span) > span_tolerance
    ):
        return reject(
            "metric_span_or_mean_pitch_failed",
            mean_gap=mean_gap,
            physical_span=physical_span,
            expected_span=expected_span,
        )
    regularity = float(
        _bell(mean_gap, BASKET_DIVIDER_SPACING_M, 0.028)
        * math.exp(-gap2_cv / 0.55)
        * math.exp(-pair_gap_cv / 0.55)
        * _ramp(len(visible), 2.0, 5.0)
    )
    return {
        "visible_count": int(len(visible)),
        "validated_indices": [int(item["index"]) for item in visible],
        "direction_consensus": consensus_score,
        "mean_gap_m": mean_gap,
        "normalized_gaps_m": normalized_gaps,
        "gap_cv": gap2_cv,
        "pair_center_gap_cv": pair_gap_cv,
        "physical_span_m": physical_span,
        "expected_span_m": expected_span,
        "family_direction": family_direction,
        "snapped_offsets_px": snapped_offsets.tolist(),
        "pixel_gap_cv": gap2_cv,
        "max_offset_drift_px": max_drift,
        "regularity": regularity,
    }


def _sf_select_family(
    edges,
    tangent,
    mask,
    bbox,
    object_depth,
    camera,
    ignore_clearance,
    cloud,
    resume_state=None,
    accepted_limit=TERMINAL_FAMILY_LIMIT,
):
    if resume_state is None:
        edge_distance = cv2.distanceTransform(
            (edges == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
        px_per_m = focal / max(float(object_depth), 0.04)
        tolerance_px = float(np.clip(0.0032 * px_per_m, 2.4, 6.0))
        min_run_px = float(np.clip(0.052 * px_per_m, 24.0, 120.0))
        max_gap_px = int(np.clip(round(0.0030 * px_per_m), 3, 8))
        hypotheses = _sf_family_candidates(
            edges, mask, bbox, object_depth, camera
        )
        edge_v, edge_u = np.nonzero(edges > 0)
        edge_points = np.column_stack([edge_u, edge_v]).astype(np.float64)
        edge_tangent = tangent[edge_v, edge_u]
        next_hypothesis = 0
    else:
        edge_distance = resume_state["edge_distance"]
        tolerance_px = resume_state["tolerance_px"]
        min_run_px = resume_state["min_run_px"]
        max_gap_px = resume_state["max_gap_px"]
        hypotheses = resume_state["hypotheses"]
        edge_points = resume_state["edge_points"]
        edge_tangent = resume_state["edge_tangent"]
        next_hypothesis = int(resume_state["next_hypothesis"])
    reviewed = []
    accepted = []
    for hypothesis_index in range(next_hypothesis, len(hypotheses)):
        family = hypotheses[hypothesis_index]
        angle = float(family["angle"])
        theta = math.radians(angle)
        expected_normal = np.array([-math.sin(theta), math.cos(theta)])
        rho = (
            edge_points - np.asarray(family["center"], np.float64)
        ) @ expected_normal
        angle_delta = np.abs((edge_tangent - angle) % 180.0)
        angle_delta = np.minimum(angle_delta, 180.0 - angle_delta)
        oriented = angle_delta <= 22.0
        family_edge_points = edge_points[oriented]
        family_edge_rho = rho[oriented]
        fitted = []
        for index in range(5):
            line = _sf_fit_ridge(
                family,
                index,
                family_edge_points,
                family_edge_rho,
                edge_distance,
                tolerance_px,
                min_run_px,
                max_gap_px,
                ignore_clearance,
            )
            if line is None:
                continue
            fitted.append(line)
        if len(fitted) < 4:
            reviewed.append(
                {
                    "angle_deg": family["angle"],
                    "structural": round(float(family["structural"]), 5),
                    "observed_rgb_ridges": int(len(fitted)),
                    "reason": "fewer_than_four_rgb_ridges_fit",
                }
            )
            continue
        for line in fitted:
            line["lift"] = _sf_lift_ridge(line, cloud, camera)
        validation = _sf_validate_family(family, fitted)
        if validation is None:
            reviewed.append(
                {
                    "angle_deg": family["angle"],
                    "structural": round(float(family["structural"]), 5),
                    "visible_3d": int(
                        sum(item["lift"] is not None for item in fitted)
                    ),
                    "ridge_fits": [
                        {
                            "index": int(item["index"]),
                            "uv": [
                                [round(float(value), 2) for value in item["p0"]],
                                [round(float(value), 2) for value in item["p1"]],
                            ],
                            "direction_3d": (
                                None
                                if item["lift"] is None
                                else [
                                    round(float(value), 4)
                                    for value in item["lift"]["direction"]
                                ]
                            ),
                        }
                        for item in fitted
                    ],
                    "metric_failure": family.get("metric_failure"),
                    "reason": "metric_five_ridge_lattice_failed",
                }
            )
            continue
        validated_indices = set(validation["validated_indices"])
        for item in fitted:
            item["lift_metric_valid"] = (
                int(item["index"]) in validated_indices
            )
        mean_edge = float(
            np.mean(
                [item["edge_metrics"]["within_tolerance"] for item in fitted]
            )
        )
        score = float(
            0.34 * _ramp(family["structural"], 0.012, 0.075)
            + 0.34 * validation["regularity"]
            + 0.20 * _ramp(validation["visible_count"], 2.0, 5.0)
            + 0.12 * _ramp(mean_edge, 0.82, 0.98)
        )
        accepted.append(
            {
                "family": family,
                "fitted": fitted,
                "validation": validation,
                "edge_distance": edge_distance,
                "tolerance_px": tolerance_px,
                "score": score,
            }
        )
        reviewed.append(
            {
                "angle_deg": family["angle"],
                "structural": round(float(family["structural"]), 5),
                "visible_3d": int(validation["visible_count"]),
                "mean_gap_m": round(float(validation["mean_gap_m"]), 5),
                "score": round(score, 5),
                "reason": "accepted",
            }
        )
        if (
            accepted_limit is not None
            and len(accepted) >= int(accepted_limit)
        ):
            next_hypothesis = hypothesis_index + 1
            break
    else:
        next_hypothesis = len(hypotheses)
    accepted.sort(key=lambda item: item["score"], reverse=True)
    state = {
        "edge_distance": edge_distance,
        "tolerance_px": tolerance_px,
        "min_run_px": min_run_px,
        "max_gap_px": max_gap_px,
        "hypotheses": hypotheses,
        "edge_points": edge_points,
        "edge_tangent": edge_tangent,
        "next_hypothesis": int(next_hypothesis),
    }
    return accepted, {
        "hypothesis_count": int(len(hypotheses)),
        "evaluated_hypothesis_count": int(next_hypothesis),
        "search_exhausted": bool(next_hypothesis >= len(hypotheses)),
        "reviewed": reviewed[:18],
        "edge_tolerance_px": round(tolerance_px, 3),
        "minimum_continuous_edge_px": round(min_run_px, 2),
        "maximum_closed_gap_px": int(max_gap_px),
    }, state


def _sf_project(points, camera):
    points = np.asarray(points, np.float64)
    depth = np.maximum(points[..., 2], 1e-9)
    u = float(camera["fx"]) * points[..., 0] / depth + float(camera["cx"])
    v = float(camera["fy"]) * points[..., 1] / depth + float(camera["cy"])
    return np.stack([u, v], axis=-1)


def _sf_line_clear_fraction(ignore_clearance, p0, p1):
    points, _ = _line_points(p0, p1)
    blocked, in_frame = _sample(ignore_clearance, points, default=255)
    return float(np.mean(in_frame & (blocked == 0)))


def _sf_single_plate_depth_continuity(depth_stack, alpha):
    """Reject a line assembled from several divider faces/depth layers.

    A true free upper edge has one coherent material surface immediately on
    its material side.  A transverse line can look continuous in RGB while
    successively crossing several dividers and the gaps between them.  Fit
    the near material-side depth as one perspective 3-D line and require most
    of that short terminal window to agree with the fit.
    """
    depth_stack = np.asarray(depth_stack, np.float64)
    alpha = np.asarray(alpha, np.float64)
    if (
        depth_stack.ndim != 2
        or depth_stack.shape[1] != len(alpha)
        or len(alpha) < 12
    ):
        return {
            "passed": False,
            "reason": "invalid_material_depth_profile",
        }

    valid_depth = np.isfinite(depth_stack) & (depth_stack > 0.05)
    column_count = np.sum(valid_depth, axis=0)
    usable_column = column_count >= 2
    column_depth = np.full(len(alpha), np.nan, np.float64)
    if bool(np.any(usable_column)):
        # Follow the near plastic lip rather than background seen through
        # perforations.  The depth stack has only a few material-side offsets,
        # so an explicit vectorized linear quantile is much cheaper than the
        # generic nanpercentile machinery and is numerically equivalent.
        depth_for_quantile = np.sort(
            np.where(valid_depth, depth_stack, np.inf), axis=0
        )
        columns = np.flatnonzero(usable_column)
        quantile_position = 0.20 * (column_count[columns] - 1)
        quantile_low = np.floor(quantile_position).astype(np.int32)
        quantile_high = np.ceil(quantile_position).astype(np.int32)
        quantile_fraction = quantile_position - quantile_low
        low_value = depth_for_quantile[quantile_low, columns]
        high_value = depth_for_quantile[quantile_high, columns]
        interpolated = (
            low_value
            + quantile_fraction * (high_value - low_value)
        )
        column_depth[columns] = interpolated

    valid = np.isfinite(column_depth)
    valid_fraction = float(np.mean(valid))
    if int(valid.sum()) < max(12, int(math.ceil(0.55 * len(alpha)))):
        return {
            "passed": False,
            "reason": "insufficient_single_plate_depth_support",
            "valid_fraction": valid_fraction,
        }

    inverse_depth = 1.0 / np.maximum(column_depth[valid], 1e-6)
    design = np.column_stack(
        [alpha[valid], np.ones(int(valid.sum()), np.float64)]
    )
    slope, intercept = np.linalg.lstsq(
        design, inverse_depth, rcond=None
    )[0]
    predicted_inverse = slope * alpha[valid] + intercept
    positive = predicted_inverse > 0.15
    if int(positive.sum()) < max(12, int(math.ceil(0.55 * len(alpha)))):
        return {
            "passed": False,
            "reason": "invalid_single_plate_depth_fit",
            "valid_fraction": valid_fraction,
        }

    residual = np.abs(
        column_depth[valid][positive]
        - 1.0 / predicted_inverse[positive]
    )
    median_residual = float(np.median(residual))
    p75_residual = float(np.percentile(residual, 75))
    inlier_fraction = float(np.mean(residual <= 0.030))
    passed = bool(
        valid_fraction >= 0.55
        and median_residual <= 0.030
        and p75_residual <= 0.045
        and inlier_fraction >= 0.52
    )
    return {
        "passed": passed,
        "reason": (
            "single_plate_depth_continuity_passed"
            if passed
            else "crosses_multiple_material_depth_layers"
        ),
        "valid_fraction": valid_fraction,
        "median_residual_m": median_residual,
        "p75_residual_m": p75_residual,
        "inlier_30mm_fraction": inlier_fraction,
    }


def _sf_free_edge_evidence(
    p0,
    p1,
    xyz0,
    xyz1,
    occupancy,
    z,
    camera,
    ignore_clearance,
):
    """Verify material on one side and genuinely deeper space on the other.

    Missing depth and closer foreground are unknown/occluded, never free.
    Pixels explicitly marked by the caller's ignore mask are unavailable
    evidence; no clearance or downstream gripper-safety margin is applied.
    """
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    points, alpha = _line_points(p0, p1)
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length < 12.0:
        return {"passed": False, "reason": "short_semantic_window"}
    normal = np.array([-vector[1], vector[0]], np.float64) / length
    z0 = max(float(np.asarray(xyz0)[2]), 0.04)
    z1 = max(float(np.asarray(xyz1)[2]), 0.04)
    predicted_z = 1.0 / (
        (1.0 - alpha) / z0 + alpha / z1
    )
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    px_per_m = focal / max(float(np.median(predicted_z)), 0.04)
    offsets_m = (0.004, 0.008, 0.014, 0.022, 0.032)
    side_occupancy = []
    side_same_depth = []
    side_deeper = []
    side_closer = []
    side_unknown = []
    side_robot = []
    side_profiles = []
    side_depth_samples = []
    for sign in (1.0, -1.0):
        occupancy_rows = []
        same_rows = []
        deeper_rows = []
        closer_rows = []
        unknown_rows = []
        robot_rows = []
        paired_depth_rows = []
        for offset_m in offsets_m:
            shifted = points + sign * offset_m * px_per_m * normal
            occupied, valid = _sample(occupancy, shifted, default=1)
            occupancy_rows.append((occupied > 0) & valid)
            depth, in_frame = _sample(z, shifted, default=np.nan)
            blocked, robot_in_frame = _sample(
                ignore_clearance, shifted, default=0
            )
            robot = robot_in_frame & (blocked > 0)
            depth_valid = in_frame & np.isfinite(depth) & (depth > 0.05)
            tolerance = np.maximum(0.012, 0.035 * predicted_z)
            same_rows.append(
                depth_valid & (np.abs(depth - predicted_z) <= tolerance)
            )
            deeper_rows.append(depth_valid & (depth > predicted_z + tolerance))
            closer_rows.append(depth_valid & (depth < predicted_z - tolerance))
            unknown_rows.append(in_frame & ~depth_valid)
            robot_rows.append(robot)
            paired_depth_rows.append(
                np.where(depth_valid & ~robot, depth, np.nan)
            )
        occupancy_stack = np.stack(occupancy_rows, axis=0)
        same_stack = np.stack(same_rows, axis=0)
        deeper_stack = np.stack(deeper_rows, axis=0)
        closer_stack = np.stack(closer_rows, axis=0)
        unknown_stack = np.stack(unknown_rows, axis=0)
        robot_stack = np.stack(robot_rows, axis=0)
        paired_depth_stack = np.stack(paired_depth_rows, axis=0)
        side_occupancy.append(float(np.mean(occupancy_stack)))
        side_same_depth.append(float(np.mean(same_stack)))
        side_deeper.append(float(np.mean(deeper_stack)))
        side_closer.append(float(np.mean(closer_stack)))
        side_unknown.append(float(np.mean(unknown_stack)))
        side_robot.append(float(np.mean(robot_stack)))
        side_depth_samples.append(paired_depth_stack)
        side_profiles.append(
            {
                "occupancy": np.mean(occupancy_stack, axis=1).tolist(),
                "same": np.mean(same_stack, axis=1).tolist(),
                "deeper": np.mean(deeper_stack, axis=1).tolist(),
                "closer": np.mean(closer_stack, axis=1).tolist(),
                "unknown": np.mean(unknown_stack, axis=1).tolist(),
            }
        )
    robot_fraction = float(max(side_robot))
    if robot_fraction > 0.005:
        return {
            "passed": False,
            "reason": "robot_clearance_intersects_normal_band",
            "score": 0.0,
            "robot_fraction": robot_fraction,
        }
    first_depth, second_depth = side_depth_samples
    pair_valid = np.isfinite(first_depth) & np.isfinite(second_depth)
    pair_tolerance = np.maximum(0.012, 0.035 * predicted_z)[None, :]
    first_in_front = pair_valid & (
        first_depth + pair_tolerance < second_depth
    )
    second_in_front = pair_valid & (
        second_depth + pair_tolerance < first_depth
    )
    side_frontness = [
        float(np.mean(first_in_front)),
        float(np.mean(second_in_front)),
    ]
    paired_depth_coverage = float(np.mean(pair_valid))
    material_index = int(
        np.argmax(
            np.asarray(side_occupancy)
            + 0.75 * np.asarray(side_frontness)
            + 0.18 * np.asarray(side_same_depth)
            - 0.18 * np.asarray(side_deeper)
        )
    )
    free_index = 1 - material_index
    material_occupancy = side_occupancy[material_index]
    free_occupancy = side_occupancy[free_index]
    material_depth = side_same_depth[material_index]
    free_same_depth = side_same_depth[free_index]
    sidedness = material_occupancy - free_occupancy
    depth_sidedness = material_depth - free_same_depth
    free_depth = side_deeper[free_index]
    free_closer = side_closer[free_index]
    free_unknown = side_unknown[free_index]
    depth_frontness = side_frontness[material_index]
    reverse_frontness = side_frontness[free_index]
    if material_index == 0:
        signed_pair_gap = second_depth - first_depth
    else:
        signed_pair_gap = first_depth - second_depth
    positive_pair_gap = signed_pair_gap[
        pair_valid & (signed_pair_gap > pair_tolerance)
    ]
    pair_gap_m = (
        float(np.median(positive_pair_gap))
        if positive_pair_gap.size
        else 0.0
    )
    mask_gate = bool(
        material_occupancy >= 0.38
        and sidedness >= 0.14
        and free_occupancy <= 0.68
        and reverse_frontness <= 0.10
        and (
            depth_frontness >= reverse_frontness
            or paired_depth_coverage < 0.18
        )
    )
    depth_gate = bool(
        material_occupancy >= 0.72
        and depth_frontness >= 0.24
        and reverse_frontness <= 0.12
        and pair_gap_m >= 0.014
        and free_occupancy <= 1.0
    )
    # A gap larger than the local basket topology is an arbitrary outer
    # silhouette against distant background, not a divider free edge.
    local_pair_gap_gate = bool(pair_gap_m <= FREE_EDGE_MAX_PAIR_GAP_M)
    single_plate_continuity = _sf_single_plate_depth_continuity(
        side_depth_samples[material_index],
        alpha,
    )
    passed = bool(
        (mask_gate or depth_gate)
        and local_pair_gap_gate
        and single_plate_continuity.get("passed", False)
    )
    score = float(
        0.34 * _ramp(sidedness, 0.08, 0.55)
        + 0.26 * _ramp(depth_frontness - reverse_frontness, 0.10, 0.55)
        + 0.20 * _ramp(pair_gap_m, 0.012, 0.080)
        + 0.20 * _ramp(material_occupancy, 0.30, 0.82)
    )
    return {
        "passed": passed,
        "score": score,
        "material_side": int(1 if material_index == 0 else -1),
        "material_occupancy": material_occupancy,
        "free_occupancy": free_occupancy,
        "sidedness": sidedness,
        "material_same_depth": material_depth,
        "free_same_depth": free_same_depth,
        "depth_sidedness": depth_sidedness,
        "free_depth": free_depth,
        "free_closer": free_closer,
        "free_unknown": free_unknown,
        "depth_frontness": depth_frontness,
        "reverse_frontness": reverse_frontness,
        "paired_depth_coverage": paired_depth_coverage,
        "pair_gap_m": pair_gap_m,
        "local_pair_gap_gate": local_pair_gap_gate,
        "single_plate_continuity": single_plate_continuity,
        "robot_fraction": robot_fraction,
        "offsets_m": list(offsets_m),
        "material_profile": side_profiles[material_index],
        "free_profile": side_profiles[free_index],
    }


def _sf_terminal_window(selected_line, family_result, camera, scene):
    lift = selected_line["lift"]
    full_length = float(lift["length_m"])
    if full_length < GRASP_MIN_M:
        return None
    if full_length > 0.22:
        nominal_fraction = 1.0 / 3.0
        target = full_length * nominal_fraction
    elif full_length > 0.13:
        nominal_fraction = 0.5
        target = full_length * nominal_fraction
    else:
        nominal_fraction = 1.0
        target = full_length
    target = float(np.clip(target, GRASP_MIN_M, GRASP_MAX_M))
    first = np.asarray(lift["p0"], np.float64)
    second = np.asarray(lift["p1"], np.float64)
    direction = (second - first) / max(full_length, 1e-9)
    inset = min(0.010, 0.025 * full_length)
    plane = None
    if scene is not None and "plane" in scene:
        plane = np.asarray(scene["plane"], np.float64)
    alternatives = []
    for name, origin, vector in (
        ("p0_terminal", first, direction),
        ("p1_terminal", second, -direction),
    ):
        start_xyz = origin + inset * vector
        end_xyz = start_xyz + target * vector
        endpoints_uv = _sf_project(
            np.stack([start_xyz, end_xyz], axis=0), camera
        )
        metrics = _sf_line_metrics(
            family_result["edge_distance"],
            endpoints_uv[0],
            endpoints_uv[1],
            family_result["tolerance_px"],
        )
        clear = _sf_line_clear_fraction(
            family_result["ignore_clearance"],
            endpoints_uv[0],
            endpoints_uv[1],
        )
        semantic = _sf_free_edge_evidence(
            endpoints_uv[0],
            endpoints_uv[1],
            start_xyz,
            end_xyz,
            family_result["occupancy"],
            family_result["z"],
            camera,
            family_result["ignore_clearance"],
        )
        full_uv_direction = np.asarray(
            selected_line["p1"], np.float64
        ) - np.asarray(selected_line["p0"], np.float64)
        window_uv_direction = endpoints_uv[1] - endpoints_uv[0]
        orientation_sign = (
            1 if float(window_uv_direction @ full_uv_direction) >= 0.0 else -1
        )
        if "material_side" in semantic:
            semantic["family_material_side"] = int(
                orientation_sign * int(semantic["material_side"])
            )
        if plane is None:
            mean_height = None
            height_score = 0.5
        else:
            heights = (
                np.stack([start_xyz, end_xyz], axis=0) @ plane[:3]
                + float(plane[3])
            )
            mean_height = float(np.mean(heights))
            height_score = _ramp(mean_height, 0.02, 0.28)
        height_gate = bool(
            mean_height is None or mean_height >= GRASP_MIN_MEAN_HEIGHT_M
        )
        score = float(
            0.48 * metrics["within_tolerance"]
            + 0.20
            * _ramp(
                1.6 * family_result["tolerance_px"] - metrics["p90_px"],
                0.0,
                1.6 * family_result["tolerance_px"],
            )
            + 0.14 * clear
            + 0.10 * _ramp(0.024 - lift["rms_m"], 0.0, 0.024)
            + 0.08 * height_score
            + 0.16 * float(semantic.get("score", 0.0))
        )
        hard_pass = bool(
            metrics["within_tolerance"] >= 0.84
            and metrics["p90_px"]
            <= 1.6 * float(family_result["tolerance_px"])
            and metrics["max_air_run_px"] <= 8.0
            and clear >= 0.995
            and semantic.get("passed", False)
            and height_gate
        )
        alternatives.append(
            {
                "name": name,
                "xyz0": start_xyz,
                "xyz1": end_xyz,
                "uv0": endpoints_uv[0],
                "uv1": endpoints_uv[1],
                "metrics": metrics,
                "clear_fraction": clear,
                "mean_height_m": mean_height,
                "height_gate": height_gate,
                "score": score,
                "free_edge": semantic,
                "hard_pass": hard_pass,
            }
        )
    selected_line["terminal_trials"] = alternatives
    eligible = [item for item in alternatives if item["hard_pass"]]
    if not eligible:
        return None
    for item in eligible:
        item["length_m"] = target
        item["nominal_fraction"] = nominal_fraction
        item["actual_fraction"] = target / full_length
    eligible.sort(key=lambda item: item["score"], reverse=True)
    best = eligible[0]
    return best


def _sf_choose_top1(family_result, camera, scene):
    candidates = []
    plane = None
    if scene is not None and "plane" in scene:
        plane = np.asarray(scene["plane"], np.float64)
    for line in family_result["fitted"]:
        lift = line.get("lift")
        if (
            lift is None
            or not line.get("lift_metric_valid", False)
            or float(lift["length_m"]) < GRASP_MIN_M
        ):
            continue
        edge = line["edge_metrics"]
        if plane is None:
            line["height_span_m"] = None
            line["mean_height_m"] = None
        else:
            full_heights = (
                np.stack([lift["p0"], lift["p1"]], axis=0) @ plane[:3]
                + float(plane[3])
            )
            line["height_span_m"] = float(np.ptp(full_heights))
            line["mean_height_m"] = float(np.mean(full_heights))
        base_score = float(
            0.34 * edge["within_tolerance"]
            + 0.22 * _ramp(0.024 - lift["rms_m"], 0.0, 0.024)
            + 0.20 * _ramp(lift["coverage"], 0.34, 0.92)
            + 0.15 * _ramp(lift["n_support"], 10, 80)
            + 0.09 * _ramp(lift["length_m"], 0.060, 0.24)
        )
        line["base_selection_score"] = base_score
        terminal = _sf_terminal_window(line, family_result, camera, scene)
        line["terminal"] = terminal
        if terminal is None:
            line["selection_score"] = 0.62 * base_score
        else:
            line["selection_score"] = float(
                0.62 * base_score + 0.38 * terminal["score"]
            )
        candidates.append(line)
    candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    direct_indices = set()
    side_options = {1: [], -1: []}
    for line in candidates:
        best_by_side = {}
        for trial in line.get("terminal_trials", []):
            if not trial.get("hard_pass", False):
                continue
            side = trial["free_edge"].get("family_material_side")
            if side not in (-1, 1):
                continue
            direct_indices.add(int(line["index"]))
            previous = best_by_side.get(int(side))
            if previous is None or trial["score"] > previous["score"]:
                best_by_side[int(side)] = trial
        for side, trial in best_by_side.items():
            side_options[side].append((line, trial))

    ranked_sides = sorted(
        side_options,
        key=lambda side: (
            len(side_options[side]),
            sum(item[1]["score"] for item in side_options[side]),
        ),
        reverse=True,
    )
    consistent_side = int(ranked_sides[0])
    consistent = side_options[consistent_side]
    height_spans = [
        float(item["height_span_m"])
        for item in candidates
        if item.get("height_span_m") is not None
    ]
    median_height_span = (
        float(np.median(height_spans)) if height_spans else None
    )
    family_result["semantic_validation"] = {
        "passed_ridge_count": int(len(direct_indices)),
        "consistent_material_side": consistent_side,
        "consistent_count": int(len(consistent)),
        "passed_ridge_indices": sorted(direct_indices),
        "consistent_ridge_indices": [
            int(line["index"]) for line, _ in consistent
        ],
        "median_height_span_m": median_height_span,
    }
    if (
        median_height_span is not None
        and median_height_span < FAMILY_MIN_HEIGHT_SPAN_M
    ):
        family_result["semantic_validation"]["reason"] = (
            "insufficient_3d_height_span_for_free_upper_edges"
        )
        return None, None, candidates
    direct_count = len(direct_indices)
    # Two isolated lines are too easily manufactured by perforations or a
    # back-panel texture.  The accepted training corpus does not require that
    # former occlusion exception: every real published divider edge is backed
    # by at least three independently lifted ridges, including tilted baskets.
    if (
        direct_count < 3
        or len(consistent) < 2
        or 2 * len(consistent) <= direct_count
    ):
        family_result["semantic_validation"]["reason"] = (
            "no_material_side_majority_across_three_direct_free_edges"
        )
        return None, None, candidates

    for line, trial in consistent:
        line["terminal"] = trial
        line["selection_score"] = float(
            0.62 * line["base_selection_score"] + 0.38 * trial["score"]
        )
    internal = [
        item for item in consistent if int(item[0]["index"]) in (1, 2, 3)
    ]
    if not internal:
        family_result["semantic_validation"]["reason"] = (
            "no_internal_divider_free_edge"
        )
        return None, None, candidates
    consistent = [
        item
        for item in internal
        if float(item[1].get("free_edge", {}).get("score", 0.0))
        >= MIN_SELECTED_FREE_EDGE_SCORE
    ]
    if not consistent:
        family_result["semantic_validation"]["reason"] = (
            "internal_free_edge_score_below_publish_threshold"
        )
        return None, None, candidates
    consistent.sort(key=lambda item: item[0]["selection_score"], reverse=True)
    best_line, best_terminal = consistent[0]
    candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    return best_line, best_terminal, candidates


def find_basket_grasp(
    rgb,
    z,
    mask,
    bbox,
    camera,
    ignore,
    scene=None,
):
    """Publish one short grasp line only from a verified five-ridge family."""
    rgb = np.asarray(rgb)
    z = np.asarray(z, np.float32)
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    ignore = (np.asarray(ignore) > 0).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[:2] != z.shape or z.shape != mask.shape:
        raise ValueError("rgb, z and mask must be aligned HxW arrays")
    if ignore.shape != mask.shape:
        raise ValueError("ignore must have the same HxW shape as mask")
    for key in ("fx", "fy", "cx", "cy"):
        if key not in camera:
            raise ValueError(f"camera missing {key}")
    scene_source = "provided"
    if scene is None:
        # The fast router deliberately skips global scene construction.  The
        # basket branch still needs a metric ground plane to distinguish true
        # raised divider edges from repeated slots on the low front/base frame.
        # Fitting it here keeps the safeguard class-local and content-driven.
        scene = build_scene_geometry(z, camera, ignore)
        scene_source = "basket_local" if scene is not None else "unavailable"
    debug = {
        "algorithm": ALGORITHM_VERSION,
        "allowed_semantic": "one observed free edge from the complete five-ridge divider family",
        "explicitly_forbidden": [
            "front_transverse_beam",
            "rear_transverse_beam",
            "base_or_frame_edge",
            "hole_or_grid_edge",
            "arbitrary_outer_silhouette",
            "single_line_fallback",
        ],
        "bbox": [int(value) for value in bbox],
        "scene_geometry_source": scene_source,
    }
    mask[ignore > 0] = 0
    if int(mask.sum()) < 180:
        debug["reason"] = "insufficient_unoccluded_basket_mask"
        return None, debug
    object_depth = _object_depth(z, mask)
    if object_depth is None:
        debug["reason"] = "metric_depth_required"
        return None, debug
    # Detection excludes only pixels explicitly marked as robot.  A 22-mm
    # metric dilation belonged to downstream gripper-clearance planning, not
    # object recognition or free-edge localisation, and dominated runtime on
    # full-resolution close views.
    ignore_clearance = ignore
    debug["object_depth_m"] = round(float(object_depth), 5)
    debug["robot_clearance_radius_px"] = 0
    debug["robot_clearance_m"] = 0.0
    edges, tangent = _sf_edge_evidence(
        rgb, mask, ignore_clearance, object_depth, camera
    )
    cloud = _sf_build_cloud(z, mask, ignore_clearance, camera, scene)
    if cloud is None:
        debug["reason"] = "insufficient_direct_basket_depth"
        return None, debug
    family_results, family_debug, family_search_state = _sf_select_family(
        edges,
        tangent,
        mask,
        bbox,
        object_depth,
        camera,
        ignore_clearance,
        cloud,
    )
    debug["family_search"] = family_debug
    if not family_results:
        debug["reason"] = "no_complete_metric_five_ridge_family"
        return None, debug
    occupancy, _ = _make_occupancy(mask, object_depth, camera)
    occupancy[ignore_clearance > 0] = 1
    safe_options = []
    semantic_review = []
    evaluated_family_count = 0
    candidate_index = 0
    while True:
        if candidate_index >= len(family_results):
            if (
                safe_options
                or family_search_state["next_hypothesis"]
                >= len(family_search_state["hypotheses"])
            ):
                break
            # No publishable terminal was found in the initial six validated
            # families.  Resume the same search state one accepted family at a
            # time.  This preserves late rank-10--14 hypotheses without paying
            # for every remaining 3-D lift on the normal successful path.
            additional, additional_debug, family_search_state = (
                _sf_select_family(
                    edges,
                    tangent,
                    mask,
                    bbox,
                    object_depth,
                    camera,
                    ignore_clearance,
                    cloud,
                    resume_state=family_search_state,
                    accepted_limit=1,
                )
            )
            family_results.extend(additional)
            family_debug["evaluated_hypothesis_count"] = (
                additional_debug["evaluated_hypothesis_count"]
            )
            family_debug["search_exhausted"] = additional_debug[
                "search_exhausted"
            ]
            family_debug["reviewed"].extend(additional_debug["reviewed"])
            if not additional and family_debug["search_exhausted"]:
                break
            continue
        candidate_family = family_results[candidate_index]
        candidate_index += 1
        if evaluated_family_count >= TERMINAL_FAMILY_LIMIT and safe_options:
            break
        evaluated_family_count += 1
        candidate_family["ignore_clearance"] = ignore_clearance
        candidate_family["occupancy"] = occupancy
        candidate_family["z"] = z
        selected_line, terminal_window, ridge_ranking = _sf_choose_top1(
            candidate_family, camera, scene
        )
        semantic_review.append(
            {
                "angle_deg": round(
                    float(candidate_family["family"]["angle"]), 2
                ),
                "family_score": round(float(candidate_family["score"]), 5),
                "strong_peak_count": int(
                    candidate_family["family"]["strong_peak_count"]
                ),
                "validation": candidate_family.get("semantic_validation"),
                "ridges": [
                    {
                        "index": int(line["index"]),
                        "has_direct_3d": line.get("lift") is not None,
                        "full_length_m": (
                            None
                            if line.get("lift") is None
                            else round(float(line["lift"]["length_m"]), 5)
                        ),
                        "height_span_m": (
                            None
                            if line.get("height_span_m") is None
                            else round(float(line["height_span_m"]), 5)
                        ),
                        "mean_height_m": (
                            None
                            if line.get("mean_height_m") is None
                            else round(float(line["mean_height_m"]), 5)
                        ),
                        "full_endpoints_uv": [
                            [round(float(value), 2) for value in line["p0"]],
                            [round(float(value), 2) for value in line["p1"]],
                        ],
                        "terminal_passed": line.get("terminal") is not None,
                        "terminal_trials": [
                            {
                                "end": trial["name"],
                                "hard_pass": bool(trial["hard_pass"]),
                                "edge_support": round(
                                    float(
                                        trial["metrics"]["within_tolerance"]
                                    ),
                                    4,
                                ),
                                "p90_px": round(
                                    float(trial["metrics"]["p90_px"]), 3
                                ),
                                "mean_height_m": (
                                    None
                                    if trial["mean_height_m"] is None
                                    else round(
                                        float(trial["mean_height_m"]), 5
                                    )
                                ),
                                "height_gate": bool(trial["height_gate"]),
                                "free_edge": {
                                    key: (
                                        round(float(value), 4)
                                        if isinstance(
                                            value, (float, np.floating)
                                        )
                                        else value
                                    )
                                    for key, value in trial["free_edge"].items()
                                },
                            }
                            for trial in line.get("terminal_trials", [])
                        ],
                    }
                    for line in ridge_ranking
                ],
            }
        )
        if selected_line is None or terminal_window is None:
            continue
        semantic_validation = candidate_family["semantic_validation"]
        semantic_quality = 0.5 * _ramp(
            semantic_validation["consistent_count"], 2.0, 5.0
        ) + 0.5 * _ramp(
            semantic_validation["passed_ridge_count"], 2.0, 5.0
        )
        height_span = semantic_validation.get("median_height_span_m")
        height_quality = (
            0.5
            if height_span is None
            else _ramp(float(height_span), 0.020, 0.180)
        )
        combined_score = float(
            0.31 * candidate_family["score"]
            + 0.24 * selected_line["selection_score"]
            + 0.21 * terminal_window["score"]
            + 0.14 * semantic_quality
            + 0.10 * height_quality
        )
        candidate_family["combined_score"] = combined_score
        candidate_family["semantic_quality"] = semantic_quality
        candidate_family["height_quality"] = height_quality
        safe_options.append(
            (
                combined_score,
                candidate_family,
                selected_line,
                terminal_window,
                ridge_ranking,
            )
        )
    debug["family_semantic_review"] = semantic_review
    debug["terminal_family_search"] = {
        "evaluated": int(evaluated_family_count),
        "available": int(len(family_results)),
        "expanded_after_initial_six": bool(
            evaluated_family_count > TERMINAL_FAMILY_LIMIT
        ),
    }
    if not safe_options:
        debug["reason"] = "no_safe_terminal_on_verified_ridge"
        return None, debug
    safe_options.sort(key=lambda item: item[0], reverse=True)
    _, family_result, selected, terminal, ranked = safe_options[0]
    lift = selected["lift"]
    validation = family_result["validation"]
    midpoint_xyz = 0.5 * (terminal["xyz0"] + terminal["xyz1"])
    midpoint_uv = _sf_project(midpoint_xyz[None, :], camera)[0]
    confidence = float(
        np.clip(
            0.50
            + 0.22 * family_result["score"]
            + 0.16 * selected["selection_score"]
            + 0.12 * terminal["score"],
            0.0,
            0.995,
        )
    )
    geometry = {
        "type": "line",
        "name": "basket_divider_free_edge",
        "endpoints_uv": [
            _safe_uv(terminal["uv0"], mask.shape),
            _safe_uv(terminal["uv1"], mask.shape),
        ],
        "endpoints": [
            [round(float(value), 6) for value in terminal["xyz0"]],
            [round(float(value), 6) for value in terminal["xyz1"]],
        ],
        "grasp_point_uv": _safe_uv(midpoint_uv, mask.shape),
        "grasp_point_xyz": [round(float(value), 6) for value in midpoint_xyz],
        "length_m": round(float(terminal["length_m"]), 6),
        "rms_mm": round(1000.0 * float(lift["rms_m"]), 3),
        "confidence": confidence,
        "grasp_safe": True,
        "inferred": False,
        "n_support": int(lift["n_support"]),
        "terminal_fraction": round(float(terminal["actual_fraction"]), 5),
        "family_count": 5,
        "observed_family_count": int(validation["visible_count"]),
        "family_source": "complete_five_ridge_rgbd_lattice",
    }
    debug.update(
        {
            "selected_top1": True,
            "family_angle_deg": round(
                float(family_result["family"]["angle"]), 2
            ),
            "family_offsets_px": [
                round(float(value), 2)
                for value in family_result["family"]["offsets"]
            ],
            "family_metric": {
                "visible_3d_count": int(validation["visible_count"]),
                "mean_gap_m": round(float(validation["mean_gap_m"]), 5),
                "normalized_gaps_m": [
                    round(float(value), 5)
                    for value in validation["normalized_gaps_m"]
                ],
                "gap_cv": round(float(validation["gap_cv"]), 4),
                "pair_center_gap_cv": round(
                    float(validation["pair_center_gap_cv"]), 4
                ),
                "regularity": round(float(validation["regularity"]), 4),
            },
            "family_semantic": family_result["semantic_validation"],
            "selected_ridge_index": int(selected["index"]),
            "selected_full_line": {
                "endpoints_uv": [
                    [round(float(value), 2) for value in selected["p0"]],
                    [round(float(value), 2) for value in selected["p1"]],
                ],
                "length_m": round(float(lift["length_m"]), 5),
                "rms_mm": round(1000.0 * float(lift["rms_m"]), 3),
                "depth_coverage": round(float(lift["coverage"]), 4),
                "edge_within_tolerance": round(
                    float(selected["edge_metrics"]["within_tolerance"]), 4
                ),
                "selection_score": round(
                    float(selected["selection_score"]), 5
                ),
            },
            "ridge_ranking": [
                {
                    "index": int(item["index"]),
                    "score": round(float(item["selection_score"]), 5),
                    "has_direct_3d": item.get("lift") is not None,
                }
                for item in ranked
            ],
            "terminal_policy": {
                "selected_end": terminal["name"],
                "nominal_fraction": terminal["nominal_fraction"],
                "actual_fraction": round(
                    float(terminal["actual_fraction"]), 5
                ),
                "mean_height_m": (
                    None
                    if terminal["mean_height_m"] is None
                    else round(float(terminal["mean_height_m"]), 5)
                ),
                "edge_within_tolerance": round(
                    float(terminal["metrics"]["within_tolerance"]), 4
                ),
                "clear_fraction": round(
                    float(terminal["clear_fraction"]), 4
                ),
                "free_edge": {
                    key: (
                        round(float(value), 4)
                        if isinstance(value, (float, np.floating))
                        else value
                    )
                    for key, value in terminal["free_edge"].items()
                },
            },
            "semantic_guards_passed": [
                "complete_five_ridge_rgb_family",
                "metric_303mm_span_and_75p75mm_pitch",
                "at_least_three_direct_3d_ridges",
                "at_least_three_direct_free_edge_topologies",
                "consistent_material_side_across_family",
                "continuous_rgb_edge_support",
                "above_bottom_grid_height_band",
                "local_pair_gap_below_170mm",
                "family_height_span_above_80mm",
                "explicit_ignore_pixels_excluded",
                "single_top1_terminal_window",
            ],
        }
    )
    return geometry, debug
