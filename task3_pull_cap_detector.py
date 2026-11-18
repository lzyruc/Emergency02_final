"""Detect the task3 pull cap first and optionally infer three reel slots.

Input is aligned wrist BGR plus uint16 millimetre depth. Returned 3-D points
use the OpenCV camera frame (+x right, +y down, +z forward), in metres. The
small cap is the primary target; failure to find the large reel circle does
not reject an otherwise valid cap detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SlotCandidate:
    direction_quarter_turns_from_cap: int
    pixel_uv: tuple[int, int]
    point_camera_m: np.ndarray


@dataclass(frozen=True)
class PullCapDetection:
    cap_pixel_uv: tuple[int, int]
    point_camera_m: np.ndarray
    depth_m: float
    confidence: float
    cap_circle_uvr: tuple[int, int, int]
    large_circle_uvr: tuple[int, int, int] | None
    slot_candidates: tuple[SlotCandidate, ...]
    large_mask: np.ndarray
    cap_mask: np.ndarray
    selection_strategy: str


class PullCapDetector:
    """Cap-first Hough detector with optional loose large-circle support."""

    MIN_DEPTH_MM = 80
    MAX_DEPTH_MM = 800
    MAX_DARK_VALUE = 115
    MAX_CHANNEL_SPREAD = 60
    MIN_CAP_RADIUS_PX = 10
    MAX_CAP_RADIUS_PX = 70
    MIN_CAP_RADIUS_M = 0.0035
    MAX_CAP_RADIUS_M = 0.020
    TARGET_CAP_RADIUS_M = 0.009
    MIN_CAP_DARK_RATIO = 0.25
    MIN_CAP_DEPTH_RATIO = 0.35
    MIN_LARGE_RADIUS_PX = 42
    MAX_LARGE_DEPTH_DIFFERENCE_MM = 100.0

    def __init__(self, camera_matrix: np.ndarray) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape 3x3")

    @staticmethod
    def _circle_mask(
        shape: tuple[int, int],
        circle: tuple[float, float, float],
        radius_scale: float,
    ) -> np.ndarray:
        height, width = shape
        u, v, radius = circle
        yy, xx = np.ogrid[:height, :width]
        return (xx - u) ** 2 + (yy - v) ** 2 <= (radius * radius_scale) ** 2

    def _point_camera(self, u: int, v: int, depth_m: float) -> np.ndarray:
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        return np.asarray(
            [(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m],
            dtype=np.float64,
        )

    def _large_for_cap(
        self,
        cap: dict,
        circles: list[dict],
    ) -> dict | None:
        cap_u, cap_v, cap_radius = cap["circle"]
        supported: list[tuple[float, dict]] = []
        for candidate in circles:
            large_u, large_v, large_radius = candidate["circle"]
            if large_radius < max(self.MIN_LARGE_RADIUS_PX, cap_radius * 1.8):
                continue
            distance_ratio = float(
                np.hypot(cap_u - large_u, cap_v - large_v) / large_radius
            )
            depth_difference = abs(candidate["depth_mm"] - cap["depth_mm"])
            if not (
                0.80 <= distance_ratio <= 2.15
                and candidate["dark_ratio"] >= 0.20
                and candidate["depth_ratio"] >= 0.25
                and depth_difference <= self.MAX_LARGE_DEPTH_DIFFERENCE_MM
            ):
                continue
            score = (
                0.35 * min(1.0, candidate["rgb_contrast"] / 70.0)
                + 0.30 * min(1.0, candidate["edge_ratio"] / 0.10)
                + 0.15 * candidate["dark_ratio"]
                + 0.10 * candidate["depth_ratio"]
                + 0.05
                * (1.0 - depth_difference / self.MAX_LARGE_DEPTH_DIFFERENCE_MM)
                + 0.05 * max(0.0, 1.0 - abs(distance_ratio - 1.45) / 0.70)
            )
            supported.append((float(score), candidate))
        return max(supported, key=lambda item: item[0])[1] if supported else None

    def detect(
        self,
        bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        roi: tuple[float, float, float, float] = (0.44, 0.03, 0.98, 0.97),
        exclusion_mask: np.ndarray | None = None,
        slot_radius_scale: float = 1.0,
    ) -> PullCapDetection:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr must have shape HxWx3")
        if depth_mm.shape != bgr.shape[:2]:
            raise ValueError("RGB and aligned depth shapes differ")
        if not np.issubdtype(depth_mm.dtype, np.integer):
            raise ValueError("depth_mm must contain integer millimetres")

        height, width = depth_mm.shape
        x0f, y0f, x1f, y1f = roi
        x0, x1 = round(x0f * width), round(x1f * width)
        y0, y1 = round(y0f * height), round(y1f * height)
        ignored = (
            np.zeros((height, width), dtype=bool)
            if exclusion_mask is None
            else np.asarray(exclusion_mask, dtype=bool)
        )
        if ignored.shape != depth_mm.shape:
            raise ValueError("exclusion_mask shape must match RGB-D")

        channels = bgr.astype(np.int16)
        value = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark = (
            (value <= self.MAX_DARK_VALUE)
            & ((channels.max(axis=2) - channels.min(axis=2)) <= self.MAX_CHANNEL_SPREAD)
            & ~ignored
        )
        valid_depth = (
            (depth_mm >= self.MIN_DEPTH_MM)
            & (depth_mm <= self.MAX_DEPTH_MM)
            & ~ignored
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi_blurred = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 1.8)
        raw_circles = cv2.HoughCircles(
            roi_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=90,
            param2=26,
            minRadius=self.MIN_CAP_RADIUS_PX,
            maxRadius=min(height, width) // 4,
        )
        if raw_circles is None:
            raise RuntimeError("small cylindrical cap circle was not found")

        # Large-circle support uses a separate, stricter RGB edge vote. It is
        # optional for cap success, but required before producing slot points.
        raw_large_circles = cv2.HoughCircles(
            roi_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=45,
            param1=100,
            param2=36,
            minRadius=self.MIN_LARGE_RADIUS_PX,
            maxRadius=min(height, width) // 4,
        )

        focal = float(np.sqrt(self.camera_matrix[0, 0] * self.camera_matrix[1, 1]))
        circles: list[dict] = []
        for u_roi, v_roi, radius in raw_circles[0]:
            circle = (float(u_roi + x0), float(v_roi + y0), float(radius))
            inner = self._circle_mask(depth_mm.shape, circle, 0.68)
            pixel_count = int(np.count_nonzero(inner))
            supported = inner & valid_depth
            if pixel_count == 0 or not np.any(supported):
                continue
            median_depth_mm = float(np.median(depth_mm[supported]))
            circles.append(
                {
                    "circle": circle,
                    "dark_ratio": float(np.count_nonzero(inner & dark) / pixel_count),
                    "depth_ratio": float(np.count_nonzero(supported) / pixel_count),
                    "depth_mm": median_depth_mm,
                    "radius_m": float(radius * median_depth_mm / (1000.0 * focal)),
                }
            )

        large_circles: list[dict] = []
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 120) > 0
        if raw_large_circles is not None:
            yy, xx = np.ogrid[:height, :width]
            for u_roi, v_roi, radius in raw_large_circles[0]:
                circle = (float(u_roi + x0), float(v_roi + y0), float(radius))
                u, v, measured_radius = circle
                distance = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
                inner = distance <= 0.68 * measured_radius
                supported = inner & valid_depth
                if not np.any(supported):
                    continue
                ring = np.abs(distance - measured_radius) <= 3.0
                outer = (
                    (distance >= 1.03 * measured_radius)
                    & (distance <= 1.18 * measured_radius)
                )
                large_circles.append(
                    {
                        "circle": circle,
                        "dark_ratio": float(
                            np.count_nonzero(inner & dark) / np.count_nonzero(inner)
                        ),
                        "depth_ratio": float(
                            np.count_nonzero(supported) / np.count_nonzero(inner)
                        ),
                        "depth_mm": float(np.median(depth_mm[supported])),
                        "edge_ratio": float(
                            np.count_nonzero(ring & edges) / np.count_nonzero(ring)
                        ),
                        "rgb_contrast": max(
                            0.0,
                            float(np.median(gray[outer]) - np.median(gray[inner])),
                        ),
                    }
                )

        cap_candidates: list[tuple[float, dict, dict | None]] = []
        minimum_depth = min(
            (candidate["depth_mm"] for candidate in circles),
            default=float(self.MAX_DEPTH_MM),
        )
        for candidate in circles:
            _, _, radius = candidate["circle"]
            if not (
                self.MIN_CAP_RADIUS_PX <= radius <= self.MAX_CAP_RADIUS_PX
                and self.MIN_CAP_RADIUS_M
                <= candidate["radius_m"]
                <= self.MAX_CAP_RADIUS_M
                and candidate["dark_ratio"] >= self.MIN_CAP_DARK_RATIO
                and candidate["depth_ratio"] >= self.MIN_CAP_DEPTH_RATIO
            ):
                continue
            large = self._large_for_cap(candidate, large_circles)
            radius_score = max(
                0.0,
                1.0
                - abs(candidate["radius_m"] - self.TARGET_CAP_RADIUS_M) / 0.010,
            )
            near_score = max(
                0.0,
                1.0 - (candidate["depth_mm"] - minimum_depth) / 500.0,
            )
            score = (
                0.30 * radius_score
                + 0.25 * candidate["dark_ratio"]
                + 0.25 * candidate["depth_ratio"]
                + 0.10 * near_score
                + (0.10 if large is not None else 0.0)
            )
            cap_candidates.append((float(score), candidate, large))
        if not cap_candidates:
            raise RuntimeError("no cap-sized dark RGB-D circle was found")

        confidence, cap, large = max(cap_candidates, key=lambda item: item[0])
        cap_u, cap_v, _ = cap["circle"]
        cap_depth_area = (
            self._circle_mask(depth_mm.shape, cap["circle"], 0.45)
            & valid_depth
        )
        if np.count_nonzero(cap_depth_area) < 8:
            raise RuntimeError("small cap centre has insufficient valid depth")
        depth_m = float(np.median(depth_mm[cap_depth_area])) / 1000.0
        cap_pixel = (int(round(cap_u)), int(round(cap_v)))
        point_camera_m = self._point_camera(*cap_pixel, depth_m)

        slot_candidates: list[SlotCandidate] = []
        large_circle: tuple[int, int, int] | None = None
        large_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
        if large is not None:
            large_u, large_v, large_radius = large["circle"]
            cap_direction = np.asarray(
                [cap_u - large_u, cap_v - large_v],
                dtype=np.float64,
            )
            direction_norm = float(np.linalg.norm(cap_direction))
            if direction_norm > 1.0:
                cap_direction /= direction_norm
                large_depth_m = float(large["depth_mm"]) / 1000.0
                for quarter_turns in (1, 2, 3):
                    angle = quarter_turns * np.pi / 2.0
                    rotation = np.asarray(
                        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                    )
                    direction = rotation @ cap_direction
                    pixel = (
                        int(round(large_u + direction[0] * large_radius * slot_radius_scale)),
                        int(round(large_v + direction[1] * large_radius * slot_radius_scale)),
                    )
                    if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                        slot_candidates.append(
                            SlotCandidate(
                                direction_quarter_turns_from_cap=quarter_turns,
                                pixel_uv=pixel,
                                point_camera_m=self._point_camera(
                                    *pixel,
                                    large_depth_m,
                                ),
                            )
                        )
            large_circle = tuple(int(round(value)) for value in large["circle"])
            large_mask = np.uint8(
                self._circle_mask(depth_mm.shape, large["circle"], 1.0)
            ) * 255

        return PullCapDetection(
            cap_pixel_uv=cap_pixel,
            point_camera_m=point_camera_m,
            depth_m=depth_m,
            confidence=confidence,
            cap_circle_uvr=tuple(int(round(value)) for value in cap["circle"]),
            large_circle_uvr=large_circle,
            slot_candidates=tuple(slot_candidates),
            large_mask=large_mask,
            cap_mask=np.uint8(
                self._circle_mask(depth_mm.shape, cap["circle"], 1.0)
            ) * 255,
            selection_strategy=(
                "cap_metric_rgbd_with_optional_large_support"
                if large is not None
                else "cap_metric_rgbd_without_large_circle"
            ),
        )


def save_pull_cap_debug_images(
    output_dir: Path,
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    detection: PullCapDetection,
) -> None:
    """Save raw RGB-D and cap/optional-large/slot overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    cap_u, cap_v, cap_radius = detection.cap_circle_uvr
    if detection.large_circle_uvr is not None:
        large_u, large_v, large_radius = detection.large_circle_uvr
        cv2.circle(overlay, (large_u, large_v), large_radius, (0, 255, 255), 3)
        cv2.line(overlay, (large_u, large_v), (cap_u, cap_v), (255, 0, 255), 2)
    cv2.circle(overlay, (cap_u, cap_v), cap_radius, (0, 255, 0), 3)
    cv2.circle(overlay, detection.cap_pixel_uv, 7, (0, 0, 255), -1)
    for slot in detection.slot_candidates:
        cv2.circle(overlay, slot.pixel_uv, 9, (255, 255, 0), 2)
        cv2.putText(
            overlay,
            f"S{slot.direction_quarter_turns_from_cap}",
            (slot.pixel_uv[0] + 8, slot.pixel_uv[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / "rgb_raw.png"), bgr)
    cv2.imwrite(str(output_dir / "depth_raw_mm.png"), depth_mm)
    cv2.imwrite(str(output_dir / "large_circle_mask.png"), detection.large_mask)
    cv2.imwrite(str(output_dir / "cap_mask.png"), detection.cap_mask)
    cv2.imwrite(str(output_dir / "rgb_target.png"), overlay)
