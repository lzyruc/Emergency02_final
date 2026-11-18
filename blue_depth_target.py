#!/usr/bin/env python3
"""Select a stable 3-D point from overlapping blue and depth-edge regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetDetection:
    pixel_uv: tuple[int, int]
    point_camera_m: np.ndarray
    depth_m: float
    confidence: float
    blue_mask: np.ndarray
    depth_edges: np.ndarray
    selected_mask: np.ndarray


class BlueDepthTargetDetector:
    """Blue-region detector whose component score is confirmed by depth edges."""

    MIN_DEPTH_MM = 80
    MAX_DEPTH_MM = 1500
    MIN_BLUE_AREA_PX = 150
    MIN_CONFIDENCE = 0.30
    BLUE_HSV_LOWER = np.array([101, 150, 90], dtype=np.uint8)
    BLUE_HSV_UPPER = np.array([132, 255, 255], dtype=np.uint8)
    MIN_BLUE_CHANNEL = 120
    MIN_BLUE_OVER_GREEN = 40
    MIN_BLUE_OVER_RED = 50

    def __init__(self, camera_matrix: np.ndarray) -> None:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        self.camera_matrix = matrix

    @staticmethod
    def _blue_mask(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(
            hsv,
            BlueDepthTargetDetector.BLUE_HSV_LOWER,
            BlueDepthTargetDetector.BLUE_HSV_UPPER,
        )
        blue = bgr[:, :, 0].astype(np.int16)
        green = bgr[:, :, 1].astype(np.int16)
        red = bgr[:, :, 2].astype(np.int16)
        channel_mask = (
            (blue >= BlueDepthTargetDetector.MIN_BLUE_CHANNEL)
            & (
                blue - green
                >= BlueDepthTargetDetector.MIN_BLUE_OVER_GREEN
            )
            & (
                blue - red
                >= BlueDepthTargetDetector.MIN_BLUE_OVER_RED
            )
        )
        mask = (
            (hsv_mask > 0) & channel_mask
        ).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @classmethod
    def _depth_edges(cls, depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = (
            (depth_mm >= cls.MIN_DEPTH_MM)
            & (depth_mm <= cls.MAX_DEPTH_MM)
        )
        if not np.any(valid):
            return np.zeros(depth_mm.shape, np.uint8), valid

        filled = depth_mm.astype(np.float32)
        median_depth = float(np.median(filled[valid]))
        filled[~valid] = median_depth
        filled = cv2.medianBlur(filled, 5)

        grad_x = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        valid_magnitude = magnitude[valid]
        adaptive = float(np.percentile(valid_magnitude, 90.0))
        threshold = max(40.0, adaptive)

        edges = ((magnitude >= threshold) & valid).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges, valid

    def detect(self, bgr: np.ndarray, depth_mm: np.ndarray) -> TargetDetection:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr must have shape HxWx3")
        if depth_mm.shape != bgr.shape[:2]:
            raise ValueError(
                f"RGB/depth are not aligned: {bgr.shape[:2]} != {depth_mm.shape}"
            )

        blue_mask = self._blue_mask(bgr)
        depth_edges, valid_depth = self._depth_edges(depth_mm)
        contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        image_area = float(bgr.shape[0] * bgr.shape[1])
        boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.MIN_BLUE_AREA_PX:
                continue

            component = np.zeros(blue_mask.shape, dtype=np.uint8)
            cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
            boundary = cv2.morphologyEx(
                component,
                cv2.MORPH_GRADIENT,
                boundary_kernel,
            )
            boundary_count = int(np.count_nonzero(boundary))
            if boundary_count == 0:
                continue

            overlap_count = int(
                np.count_nonzero((boundary > 0) & (depth_edges > 0))
            )
            overlap_ratio = overlap_count / boundary_count
            component_pixels = component > 0
            valid_ratio = float(
                np.count_nonzero(component_pixels & valid_depth)
                / np.count_nonzero(component_pixels)
            )
            area_score = min(1.0, area / (0.08 * image_area))
            confidence = (
                0.65 * min(1.0, overlap_ratio / 0.20)
                + 0.25 * valid_ratio
                + 0.10 * area_score
            )

            candidates.append((confidence, area, component))

        if not candidates:
            raise RuntimeError("no blue component with valid area was found")

        eligible = [
            candidate
            for candidate in candidates
            if candidate[0] >= self.MIN_CONFIDENCE
        ]
        if not eligible:
            best_confidence = max(candidate[0] for candidate in candidates)
            raise RuntimeError(
                f"best blue/depth confidence {best_confidence:.3f} is below "
                f"{self.MIN_CONFIDENCE:.3f}"
            )

        eligible_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
        for _, _, component in eligible:
            eligible_mask = cv2.bitwise_or(eligible_mask, component)

        # Reject isolated low-depth speckles before selecting the nearest point.
        robust_depth = depth_mm.astype(np.uint16, copy=True)
        robust_depth[~valid_depth] = np.iinfo(np.uint16).max
        robust_depth = cv2.medianBlur(robust_depth, 5)
        nearest_candidates = (
            (eligible_mask > 0)
            & (robust_depth >= self.MIN_DEPTH_MM)
            & (robust_depth <= self.MAX_DEPTH_MM)
        )
        if not np.any(nearest_candidates):
            raise RuntimeError("eligible blue regions have no valid depth")

        nearest_depth_mm = int(np.min(robust_depth[nearest_candidates]))
        equally_near = (
            nearest_candidates
            & (robust_depth <= nearest_depth_mm + 1)
        )

        # When several pixels have the same nearest depth, prefer the one
        # farther from the color-region boundary for a stable 3-D point.
        interior = cv2.distanceTransform(eligible_mask, cv2.DIST_L2, 5)
        interior[~equally_near] = 0.0
        _, max_distance, _, max_location = cv2.minMaxLoc(interior)
        if max_distance <= 0:
            ys, xs = np.nonzero(equally_near)
            if len(xs) == 0:
                raise RuntimeError("could not select the nearest blue point")
            u, v = int(xs[0]), int(ys[0])
        else:
            u, v = int(max_location[0]), int(max_location[1])

        selected_candidates = [
            candidate
            for candidate in eligible
            if candidate[2][v, u] > 0
        ]
        confidence, _, selected_mask = max(
            selected_candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        row_start = max(0, v - 2)
        row_stop = min(depth_mm.shape[0], v + 3)
        column_start = max(0, u - 2)
        column_stop = min(depth_mm.shape[1], u + 3)
        depth_window = depth_mm[
            row_start:row_stop,
            column_start:column_stop,
        ]
        valid_window = (
            (depth_window >= self.MIN_DEPTH_MM)
            & (depth_window <= self.MAX_DEPTH_MM)
        )
        if not np.any(valid_window):
            raise RuntimeError("selected blue point has no valid 5x5 depth")
        depth_value_mm = float(
            np.mean(depth_window[valid_window], dtype=np.float64)
        )

        z = depth_value_mm / 1000.0
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        point_camera = np.array(
            [(u - cx) * z / fx, (v - cy) * z / fy, z],
            dtype=np.float64,
        )
        return TargetDetection(
            pixel_uv=(u, v),
            point_camera_m=point_camera,
            depth_m=z,
            confidence=float(confidence),
            blue_mask=blue_mask,
            depth_edges=depth_edges,
            selected_mask=selected_mask,
        )


def save_debug_images(
    output_dir: Path,
    bgr: np.ndarray,
    detection: TargetDetection,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    u, v = detection.pixel_uv
    contours, _ = cv2.findContours(
        detection.selected_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.circle(overlay, (u, v), 7, (0, 0, 255), -1)
    cv2.putText(
        overlay,
        f"conf={detection.confidence:.3f} z={detection.depth_m:.3f}m",
        (max(0, u - 180), max(30, v - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(str(output_dir / "rgb_target.jpg"), overlay)
    cv2.imwrite(str(output_dir / "blue_mask.png"), detection.blue_mask)
    cv2.imwrite(str(output_dir / "depth_edges.png"), detection.depth_edges)
