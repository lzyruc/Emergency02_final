#!/usr/bin/env python3
"""Detect a dark cylindrical object in aligned RGB-D images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    pixel_uv: tuple[int, int]
    point_camera_m: np.ndarray
    point_tcp_m: np.ndarray
    distance_to_tcp_m: float
    local_width_px: float


@dataclass(frozen=True)
class BlackCylinderDetection:
    pixel_uv: tuple[int, int]
    point_camera_m: np.ndarray
    depth_m: float
    confidence: float
    angle_deg: float
    major_axis_px: float
    minor_axis_px: float
    gripper_mask: np.ndarray
    black_mask: np.ndarray
    selected_mask: np.ndarray
    grasp_candidates: tuple[GraspCandidate, ...]
    selection_strategy: str


class BlackCylinderDetector:
    """RGB darkness proposal confirmed by depth consistency and elongation."""

    MIN_DEPTH_MM = 80
    MAX_DEPTH_MM = 800
    MIN_AREA_PX = 500
    MIN_MAJOR_AXIS_PX = 45.0
    # A shaft can be connected to a round base, making the full RGB component
    # close to square. The axial grasp-point selection below still isolates
    # the near, narrow shaft using depth-supported pixels.
    MIN_ASPECT_RATIO = 1.00
    MAX_VALUE = 72
    MAX_CHANNEL_SPREAD = 32
    MIN_CONFIDENCE = 0.43

    def __init__(self, camera_matrix: np.ndarray) -> None:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        self.camera_matrix = matrix

    @classmethod
    def _black_mask(cls, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        channels = bgr.astype(np.int16)
        spread = channels.max(axis=2) - channels.min(axis=2)
        mask = (
            (value <= cls.MAX_VALUE)
            & (spread <= cls.MAX_CHANNEL_SPREAD)
        ).astype(np.uint8) * 255
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7)

    @staticmethod
    def _rect_axes(contour: np.ndarray) -> tuple[float, float, float]:
        (_, _), (width, height), angle = cv2.minAreaRect(contour)
        if width >= height:
            return float(width), float(height), float(angle)
        return float(height), float(width), float(angle + 90.0)

    def detect(
        self,
        bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        roi: tuple[float, float, float, float] = (0.08, 0.08, 0.92, 0.92),
        camera_to_tcp: np.ndarray | None = None,
        exclusion_mask: np.ndarray | None = None,
        component_selection: str = "nearest_tcp",
    ) -> BlackCylinderDetection:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr must have shape HxWx3")
        if depth_mm.shape != bgr.shape[:2]:
            raise ValueError("RGB and aligned depth shapes differ")
        if not (np.issubdtype(depth_mm.dtype, np.integer)):
            raise ValueError("depth_mm must contain integer millimetres")
        if camera_to_tcp is None:
            camera_to_tcp_matrix = np.eye(4, dtype=np.float64)
        else:
            camera_to_tcp_matrix = np.asarray(camera_to_tcp, dtype=np.float64)
            if (
                camera_to_tcp_matrix.shape != (4, 4)
                or not np.all(np.isfinite(camera_to_tcp_matrix))
            ):
                raise ValueError("camera_to_tcp must be a finite 4x4 matrix")

        height, width = depth_mm.shape
        x0f, y0f, x1f, y1f = roi
        if not (0.0 <= x0f < x1f <= 1.0 and 0.0 <= y0f < y1f <= 1.0):
            raise ValueError("roi must be normalized x0,y0,x1,y1")
        x0, x1 = round(x0f * width), round(x1f * width)
        y0, y1 = round(y0f * height), round(y1f * height)

        gripper_mask = (
            np.zeros((height, width), dtype=bool)
            if exclusion_mask is None
            else np.asarray(exclusion_mask, dtype=bool)
        )
        if gripper_mask.shape != (height, width):
            raise ValueError("exclusion_mask shape must match RGB-D")
        if component_selection not in ("nearest_tcp", "leftmost"):
            raise ValueError(
                "component_selection must be 'nearest_tcp' or 'leftmost'"
            )
        valid_depth = (
            (depth_mm >= self.MIN_DEPTH_MM)
            & (depth_mm <= self.MAX_DEPTH_MM)
            & ~gripper_mask
        )
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y0:y1, x0:x1] = 255
        black_mask = cv2.bitwise_and(self._black_mask(bgr), roi_mask)
        black_mask[gripper_mask] = 0
        black_valid = (black_mask > 0) & valid_depth
        if np.count_nonzero(black_valid) < self.MIN_AREA_PX:
            raise RuntimeError("black ROI has insufficient valid depth")

        # Shadows and dark background often join the shaft in RGB. The target
        # is the near dark object in a wrist-camera view, so isolate its robust
        # foreground depth band before connected-component scoring.
        foreground_ceiling_mm = min(
            self.MAX_DEPTH_MM,
            float(np.percentile(depth_mm[black_valid], 20.0)) + 80.0,
        )
        foreground_valid = (
            valid_depth
            & (depth_mm <= foreground_ceiling_mm)
        )

        contours, _ = cv2.findContours(
            black_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: list[
            tuple[float, float, np.ndarray, float, float, float]
        ] = []
        image_area = float(height * width)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.MIN_AREA_PX:
                continue
            major, minor, angle = self._rect_axes(contour)
            if minor < 5.0 or major < self.MIN_MAJOR_AXIS_PX:
                continue
            aspect = major / minor
            if aspect < self.MIN_ASPECT_RATIO:
                continue

            component = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
            pixels = component > 0
            valid_pixels = pixels & foreground_valid
            valid_count = int(np.count_nonzero(valid_pixels))
            if valid_count < max(50, round(area * 0.005)):
                continue

            depths = depth_mm[valid_pixels].astype(np.float64)
            median_depth = float(np.median(depths))
            pixel_count = max(1, int(np.count_nonzero(pixels)))
            all_depth_ratio = float(
                np.count_nonzero(pixels & valid_depth) / pixel_count
            )
            foreground_support = min(
                1.0,
                valid_count / max(1.0, area * 0.02),
            )
            aspect_score = min(1.0, (aspect - 1.0) / 2.0)
            area_score = min(1.0, area / (0.025 * image_area))
            confidence = (
                0.35 * aspect_score
                + 0.25 * all_depth_ratio
                + 0.20 * foreground_support
                + 0.20 * area_score
            )
            candidates.append(
                (confidence, -median_depth, component, angle, major, minor)
            )

        if not candidates:
            raise RuntimeError("no dark elongated RGB-D component was found")
        best_confidence = max(item[0] for item in candidates)
        if best_confidence < self.MIN_CONFIDENCE:
            raise RuntimeError(
                f"best black-cylinder confidence {best_confidence:.3f} is below "
                f"{self.MIN_CONFIDENCE:.3f}"
            )

        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        component_results: list[
            tuple[
                float,
                float,
                float,
                np.ndarray,
                float,
                float,
                float,
                list[GraspCandidate],
            ]
        ] = []
        for confidence, _, component_mask, angle, major, minor in candidates:
            if confidence < self.MIN_CONFIDENCE:
                continue

            # Erosion rejects mixed foreground/background depth at boundaries.
            erode_size = min(
                max(3, int(round(minor * 0.20)) | 1),
                19,
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erode_size, erode_size),
            )
            usable = (cv2.erode(component_mask, kernel) > 0) & foreground_valid
            if np.count_nonzero(usable) < 20:
                usable = (component_mask > 0) & foreground_valid
            ys, xs = np.nonzero(usable)
            if len(xs) == 0:
                continue

            angle_rad = np.deg2rad(angle)
            axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            if abs(axis[1]) >= 0.35 and axis[1] < 0.0:
                axis = -axis
            perpendicular = np.array([-axis[1], axis[0]])
            usable_points = np.column_stack((xs, ys)).astype(np.float64)
            axial = usable_points @ axis
            band_half_width = max(3.0, major * 0.025)
            radius = max(3, min(9, int(round(minor * 0.12))))
            component_grasp_candidates: list[GraspCandidate] = []
            seen_pixels: set[tuple[int, int]] = set()
            for quantile in np.linspace(0.05, 0.95, 11):
                target_axial = float(np.quantile(axial, quantile))
                band = np.abs(axial - target_axial) <= band_half_width
                if np.count_nonzero(band) < 10:
                    continue
                band_points = usable_points[band]
                transverse = band_points @ perpendicular
                local_width = float(
                    np.quantile(transverse, 0.95)
                    - np.quantile(transverse, 0.05)
                )
                u = int(round(float(np.median(band_points[:, 0]))))
                v = int(round(float(np.median(band_points[:, 1]))))
                if (u, v) in seen_pixels:
                    continue
                seen_pixels.add((u, v))
                yy0, yy1 = max(0, v - radius), min(height, v + radius + 1)
                xx0, xx1 = max(0, u - radius), min(width, u + radius + 1)
                patch_mask = usable[yy0:yy1, xx0:xx1]
                patch_depth = depth_mm[yy0:yy1, xx0:xx1][patch_mask]
                if patch_depth.size < 5:
                    continue
                z = float(np.median(patch_depth)) / 1000.0
                point_camera = np.array(
                    [(u - cx) * z / fx, (v - cy) * z / fy, z],
                    dtype=np.float64,
                )
                point_tcp = (
                    camera_to_tcp_matrix @ np.append(point_camera, 1.0)
                )[:3]
                component_grasp_candidates.append(
                    GraspCandidate(
                        pixel_uv=(u, v),
                        point_camera_m=point_camera,
                        point_tcp_m=point_tcp,
                        distance_to_tcp_m=float(np.linalg.norm(point_tcp)),
                        local_width_px=local_width,
                    )
                )
            if not component_grasp_candidates:
                continue

            # Reject a broad connected base before comparing TCP distances.
            widths = np.asarray(
                [
                    candidate.local_width_px
                    for candidate in component_grasp_candidates
                ],
                dtype=np.float64,
            )
            width_limit = max(12.0, float(np.quantile(widths, 0.10)) * 2.50)
            shaft_candidates = [
                candidate
                for candidate in component_grasp_candidates
                if candidate.local_width_px <= width_limit
            ]
            if len(shaft_candidates) >= 2:
                component_grasp_candidates = shaft_candidates
            nearest_distance = min(
                candidate.distance_to_tcp_m
                for candidate in component_grasp_candidates
            )
            component_results.append(
                (
                    nearest_distance,
                    -confidence,
                    float(np.median(xs)),
                    component_mask,
                    angle,
                    major,
                    minor,
                    component_grasp_candidates,
                )
            )

        if not component_results:
            raise RuntimeError("no accepted cylinder has valid grasp candidates")
        if component_selection == "leftmost":
            selected_component = min(
                component_results,
                key=lambda item: (item[2], item[0], item[1]),
            )
        else:
            selected_component = min(
                component_results,
                key=lambda item: (item[0], item[1]),
            )
        (
            _,
            negative_confidence,
            _,
            selected_mask,
            angle,
            major,
            minor,
            grasp_candidates,
        ) = selected_component
        confidence = -negative_confidence
        selected_candidate = min(
            grasp_candidates,
            key=lambda candidate: candidate.distance_to_tcp_m,
        )
        u, v = selected_candidate.pixel_uv
        point_camera = selected_candidate.point_camera_m
        z = float(point_camera[2])
        return BlackCylinderDetection(
            pixel_uv=(u, v),
            point_camera_m=point_camera,
            depth_m=z,
            confidence=float(confidence),
            angle_deg=angle,
            major_axis_px=major,
            minor_axis_px=minor,
            gripper_mask=gripper_mask.astype(np.uint8) * 255,
            black_mask=black_mask,
            selected_mask=selected_mask,
            grasp_candidates=tuple(grasp_candidates),
            selection_strategy=(
                "leftmost_component_then_nearest_point_to_tcp"
                if component_selection == "leftmost"
                else "nearest_component_then_nearest_point_to_tcp"
            ),
        )


def save_debug_images(
    output_dir: Path,
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    detection: BlackCylinderDetection,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    contours, _ = cv2.findContours(
        detection.selected_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    candidate_pixels = [
        candidate.pixel_uv for candidate in detection.grasp_candidates
    ]
    if len(candidate_pixels) >= 2:
        cv2.polylines(
            overlay,
            [np.asarray(candidate_pixels, dtype=np.int32)],
            False,
            (255, 180, 0),
            2,
        )
    for index, candidate in enumerate(detection.grasp_candidates, start=1):
        selected = candidate.pixel_uv == detection.pixel_uv
        color = (0, 255, 0) if selected else (0, 165, 255)
        radius = 9 if selected else 6
        cv2.circle(overlay, candidate.pixel_uv, radius, color, -1)
        label = f"C{index} {candidate.distance_to_tcp_m * 100:.1f}cm"
        u_candidate, v_candidate = candidate.pixel_uv
        cv2.putText(
            overlay,
            label,
            (u_candidate + 8, max(18, v_candidate - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    u, v = detection.pixel_uv
    cv2.rectangle(overlay, (12, 10), (505, 76), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        "orange: candidates   green: selected",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    selected_distance = min(
        candidate.distance_to_tcp_m
        for candidate in detection.grasp_candidates
        if candidate.pixel_uv == detection.pixel_uv
    )
    cv2.putText(
        overlay,
        f"nearest TCP distance: {selected_distance * 100:.1f} cm",
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "rgb_raw.jpg"), bgr)
    cv2.imwrite(str(output_dir / "depth_raw_mm.png"), depth_mm)
    cv2.imwrite(str(output_dir / "gripper_static_mask.png"), detection.gripper_mask)
    cv2.imwrite(str(output_dir / "black_mask.png"), detection.black_mask)
    cv2.imwrite(str(output_dir / "rgb_target.jpg"), overlay)
    cv2.imwrite(str(output_dir / "candidate_points.jpg"), overlay)
