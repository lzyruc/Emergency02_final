from __future__ import annotations

import cv2
import numpy as np


def _camera_points(z, camera, stride=1):
    h, w = z.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    zz = z[::stride, ::stride]
    valid = np.isfinite(zz)
    x = (uu.astype(np.float64) - float(camera["cx"])) * zz / float(camera["fx"])
    y = (vv.astype(np.float64) - float(camera["cy"])) * zz / float(camera["fy"])
    return np.column_stack([x[valid], y[valid], zz[valid]]), valid, uu, vv


def _fit_plane(points, seed=20260729):
    """Fit the dominant scene plane with deterministic batched RANSAC."""
    points = np.asarray(points, np.float64)
    if len(points) < 900:
        return None
    if len(points) > 14000:
        take = np.linspace(0, len(points) - 1, 14000, dtype=np.int64)
        fit_points = points[take]
    else:
        fit_points = points

    rng = np.random.default_rng(seed)
    triples = rng.integers(0, len(fit_points), size=(512, 3))
    p0 = fit_points[triples[:, 0]]
    p1 = fit_points[triples[:, 1]]
    p2 = fit_points[triples[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normals, axis=1)
    good = norm > 1e-5
    normals = normals[good] / norm[good, None]
    p0 = p0[good]
    if not len(normals):
        return None
    offsets = -np.sum(normals * p0, axis=1)
    flip = normals[:, 2] > 0.0
    normals[flip] *= -1.0
    offsets[flip] *= -1.0

    best_count = -1
    best_normal = None
    best_offset = None
    for start in range(0, len(normals), 32):
        batch_n = normals[start : start + 32]
        batch_d = offsets[start : start + 32]
        distance = np.abs(fit_points @ batch_n.T + batch_d[None, :])
        counts = np.sum(distance < 0.020, axis=0)
        # A downward-looking wrist camera sees the upward floor normal with
        # negative optical Z; camera Y may change sign as the wrist rolls.
        # This broad pose prior
        # prevents a close cardboard face or cabinet wall from becoming the
        # "ground" solely because it occupies more pixels.
        pose_weight = np.where(
            (batch_n[:, 2] < -0.04) & (batch_n[:, 1] < -0.04),
            1.0,
            0.0,
        )
        weighted = counts.astype(np.float64) * pose_weight
        index = int(np.argmax(weighted))
        weighted_count = float(weighted[index])
        if weighted_count > best_count:
            best_count = weighted_count
            best_normal = batch_n[index]
            best_offset = float(batch_d[index])
    if best_normal is None or best_count < 160:
        return None

    normal = np.asarray(best_normal, np.float64)
    offset = float(best_offset)
    seed_normal = normal.copy()
    inliers = np.zeros(len(points), dtype=bool)
    previous = None
    for threshold in (0.025, 0.022, 0.020, 0.018, 0.018):
        inliers = np.abs(points @ normal + offset) < threshold
        if int(inliers.sum()) < 240:
            break
        if previous is not None and np.array_equal(inliers, previous):
            break
        previous = inliers.copy()
        sample = points[inliers]
        center = np.mean(sample, axis=0)
        covariance = (sample - center).T @ (sample - center)
        _, vectors = np.linalg.eigh(covariance)
        refined = vectors[:, 0]
        if float(refined @ normal) < 0:
            refined = -refined
        if float(refined @ seed_normal) < 0.96 or refined[1] >= -0.02:
            break
        normal = refined / (np.linalg.norm(refined) + 1e-12)
        offset = -float(normal @ center)

    # OpenCV camera coordinates use +Y down. The ground normal therefore
    # normally points upward when its Y component is negative.
    if normal[2] > 0:
        normal = -normal
        offset = -offset
    residual = np.abs(points @ normal + offset)
    inliers = residual < 0.020
    return normal, offset, inliers, residual


def build_scene_geometry(z, camera, ignore=None):
    """Build metric maps shared by all wrist-camera object branches.

    A failure to fit the ground is non-fatal: callers can continue with RGB
    and local depth evidence by checking for ``None``.
    """
    z = np.asarray(z, np.float32)
    h, w = z.shape
    valid = np.isfinite(z) & (z > 0.05) & (z < 6.0)
    if ignore is not None:
        valid &= np.asarray(ignore) == 0
    stride = max(6, int(round(min(h, w) / 90.0)))
    zz = z.copy()
    zz[~valid] = np.nan
    points, sampled_valid, sampled_u, sampled_v = _camera_points(
        zz, camera, stride=stride
    )
    fitted = _fit_plane(points)
    if fitted is None:
        return None
    normal, offset, inliers, residual = fitted
    # The camera must sit above the 0.30 m-class target objects.  A dominant
    # plane only a few centimetres away is therefore an object face (box,
    # seat, bin wall), not the floor.  Returning no scene geometry is safer
    # than assigning physically wrong heights; every detector has a local
    # RGB-D fallback for this case.
    if abs(float(offset)) < 0.29:
        return None

    vv, uu = np.mgrid[0:h, 0:w]
    x = (uu.astype(np.float32) - float(camera["cx"])) * z / float(camera["fx"])
    y = (vv.astype(np.float32) - float(camera["cy"])) * z / float(camera["fy"])
    height = normal[0] * x + normal[1] * y + normal[2] * z + offset
    height[~valid] = np.nan

    origin = -offset * normal
    reference = (
        np.array([1.0, 0.0, 0.0], np.float64)
        if abs(normal[0]) < 0.90
        else np.array([0.0, 0.0, 1.0], np.float64)
    )
    axis_u = reference - float(reference @ normal) * normal
    axis_u /= np.linalg.norm(axis_u) + 1e-12
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v) + 1e-12
    ground_u = (
        (x - origin[0]) * axis_u[0]
        + (y - origin[1]) * axis_u[1]
        + (z - origin[2]) * axis_u[2]
    )
    ground_v = (
        (x - origin[0]) * axis_v[0]
        + (y - origin[1]) * axis_v[1]
        + (z - origin[2]) * axis_v[2]
    )
    ground_u[~valid] = np.nan
    ground_v[~valid] = np.nan

    floor_mask = valid & (np.abs(height) <= 0.025)
    floor_mask = cv2.morphologyEx(
        floor_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    return {
        "z": z,
        "valid": valid,
        "x": x,
        "y": y,
        "height": height.astype(np.float32),
        "ground_u": ground_u.astype(np.float32),
        "ground_v": ground_v.astype(np.float32),
        "plane": [
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
            float(offset),
        ],
        "normal": normal.astype(np.float32),
        "floor_mask": floor_mask,
        "debug": {
            "stride": stride,
            "sample_count": int(len(points)),
            "inlier_count": int(inliers.sum()),
            "inlier_fraction": round(float(np.mean(inliers)), 5),
            "residual_p90_m": round(float(np.percentile(residual[inliers], 90)), 6),
        },
    }
