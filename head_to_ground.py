"""Project one aligned head-camera depth pixel into the robot base frame."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
JOINT_CONTRACT = (
    ("first_leg_pitch_joint", "base", "first_leg_pitch_link", (0.0, 1.0, 0.0)),
    (
        "second_leg_pitch_joint",
        "first_leg_pitch_link",
        "second_leg_pitch_link",
        (0.0, 1.0, 0.0),
    ),
    (
        "waist_pitch_joint",
        "second_leg_pitch_link",
        "waist_pitch_link",
        (0.0, 1.0, 0.0),
    ),
    (
        "waist_yaw_joint",
        "waist_pitch_link",
        "waist_yaw_link",
        (0.0, 0.0, 1.0),
    ),
    ("head_yaw_joint", "waist_yaw_link", "head_yaw_link", (0.0, 0.0, 1.0)),
    (
        "head_pitch_joint",
        "head_yaw_link",
        "head_pitch_link",
        (0.0, 1.0, 0.0),
    ),
    (
        "head_roll_joint",
        "head_pitch_link",
        "head_roll_link",
        (1.0, 0.0, 0.0),
    ),
)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class RobotJoints:
    """Radians in leg-pitch, waist-pitch/yaw, head-yaw/pitch/roll order."""

    first_leg_pitch: float = 0.0
    second_leg_pitch: float = 0.0
    waist_pitch: float = 0.0
    waist_yaw: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    pixel_uv: tuple[int, int]
    depth_m: float
    depth_source: str
    fallback_radius_px: int
    camera_xyz_m: tuple[float, float, float]
    base_xyz_m: tuple[float, float, float]


def _axis_rotation(
    axis: tuple[float, float, float],
    angle_rad: float,
) -> FloatArray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    transform = np.eye(4, dtype=np.float64)
    if axis == (1.0, 0.0, 0.0):
        transform[1:3, 1:3] = ((cosine, -sine), (sine, cosine))
    elif axis == (0.0, 1.0, 0.0):
        transform[0, 0], transform[0, 2] = cosine, sine
        transform[2, 0], transform[2, 2] = -sine, cosine
    elif axis == (0.0, 0.0, 1.0):
        transform[0:2, 0:2] = ((cosine, -sine), (sine, cosine))
    else:
        raise ValueError(f"unsupported joint axis: {axis}")
    return transform


def _origin_transform(
    xyz_m: Sequence[float],
    rpy_rad: Sequence[float],
) -> FloatArray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz_m
    return (
        transform
        @ _axis_rotation((0.0, 0.0, 1.0), rpy_rad[2])
        @ _axis_rotation((0.0, 1.0, 0.0), rpy_rad[1])
        @ _axis_rotation((1.0, 0.0, 0.0), rpy_rad[0])
    )


class HeadToGroundProjector:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        head_roll_from_camera: FloatArray,
        joint_origins: tuple[FloatArray, ...],
        *,
        station_camera_key: str,
        image_size: tuple[int, int],
        depth_scale_m: float,
        depth_range_m: tuple[float, float],
        fallback_radii_px: tuple[int, ...],
        fallback_min_valid: int,
    ) -> None:
        self.intrinsics = intrinsics
        self.head_roll_from_camera = head_roll_from_camera
        self.joint_origins = joint_origins
        self.station_camera_key = station_camera_key
        self.image_size = image_size
        self.depth_scale_m = depth_scale_m
        self.min_depth_m, self.max_depth_m = depth_range_m
        self.fallback_radii_px = fallback_radii_px
        self.fallback_min_valid = fallback_min_valid
        self.has_distortion = any(intrinsics.distortion)
        self.fallback_disks = {}
        for radius in fallback_radii_px:
            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            self.fallback_disks[radius] = xx * xx + yy * yy <= radius * radius

    def base_from_camera(self, joints: RobotJoints) -> FloatArray:
        angles = (
            joints.first_leg_pitch,
            joints.second_leg_pitch,
            joints.waist_pitch,
            joints.waist_yaw,
            joints.head_yaw,
            joints.head_pitch,
            joints.head_roll,
        )
        transform = np.eye(4, dtype=np.float64)
        for origin, contract, angle in zip(
            self.joint_origins,
            JOINT_CONTRACT,
            angles,
            strict=True,
        ):
            transform = transform @ origin @ _axis_rotation(contract[3], angle)
        return transform @ self.head_roll_from_camera

    def sample_depth(
        self,
        depth_image: npt.NDArray[np.generic],
        u: int,
        v: int,
    ) -> tuple[float, str, int]:
        width, height = self.image_size
        if not isinstance(depth_image, np.ndarray) or depth_image.shape != (height, width):
            raise ValueError(f"depth image must have shape {(height, width)}")
        if not 0 <= u < width or not 0 <= v < height:
            raise ValueError(f"pixel {(u, v)} is outside {width}x{height}")

        center_m = float(depth_image[v, u]) * self.depth_scale_m
        if math.isfinite(center_m) and self.min_depth_m <= center_m <= self.max_depth_m:
            return center_m, "center", 0

        for radius in self.fallback_radii_px:
            x0, x1 = max(0, u - radius), min(width, u + radius + 1)
            y0, y1 = max(0, v - radius), min(height, v + radius + 1)
            patch_m = (
                depth_image[y0:y1, x0:x1].astype(np.float64, copy=False)
                * self.depth_scale_m
            )
            disk = self.fallback_disks[radius][
                y0 - (v - radius) : y1 - (v - radius),
                x0 - (u - radius) : x1 - (u - radius),
            ]
            valid = (
                disk
                & np.isfinite(patch_m)
                & (patch_m >= self.min_depth_m)
                & (patch_m <= self.max_depth_m)
            )
            if int(np.count_nonzero(valid)) >= self.fallback_min_valid:
                return float(np.mean(patch_m[valid])), "neighborhood_mean", radius
        raise ValueError(f"pixel {(u, v)} has no valid nearby depth")

    def project(
        self,
        depth_image: npt.NDArray[np.generic],
        pixel_uv: tuple[int, int],
        joints: RobotJoints = RobotJoints(),
    ) -> ProjectionResult:
        u, v = int(pixel_uv[0]), int(pixel_uv[1])
        depth_m, source, radius = self.sample_depth(depth_image, u, v)
        x_distorted = (u - self.intrinsics.cx) / self.intrinsics.fx
        y_distorted = (v - self.intrinsics.cy) / self.intrinsics.fy
        x_ray, y_ray = x_distorted, y_distorted
        if self.has_distortion:
            k1, k2, p1, p2, k3 = self.intrinsics.distortion
            for _ in range(5):
                radius2 = x_ray * x_ray + y_ray * y_ray
                radial = (
                    1.0
                    + k1 * radius2
                    + k2 * radius2 * radius2
                    + k3 * radius2 * radius2 * radius2
                )
                delta_x = (
                    2.0 * p1 * x_ray * y_ray
                    + p2 * (radius2 + 2.0 * x_ray * x_ray)
                )
                delta_y = (
                    p1 * (radius2 + 2.0 * y_ray * y_ray)
                    + 2.0 * p2 * x_ray * y_ray
                )
                x_ray = (x_distorted - delta_x) / radial
                y_ray = (y_distorted - delta_y) / radial

        camera_xyz = np.asarray(
            (x_ray * depth_m, y_ray * depth_m, depth_m),
            dtype=np.float64,
        )
        base_xyz = (
            self.base_from_camera(joints)
            @ np.asarray((*camera_xyz, 1.0), dtype=np.float64)
        )[:3]
        return ProjectionResult(
            pixel_uv=(u, v),
            depth_m=depth_m,
            depth_source=source,
            fallback_radius_px=radius,
            camera_xyz_m=tuple(float(value) for value in camera_xyz),
            base_xyz_m=tuple(float(value) for value in base_xyz),
        )


def load_projector(calibration_path: Path) -> HeadToGroundProjector:
    config = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera = config["camera"]
    width, height = (int(value) for value in camera["image_size"])
    depth = config["depth"]
    depth_scale_m = float(depth["scale_to_m"])
    depth_range_m = tuple(float(value) for value in depth["valid_range_m"])
    fallback_radii = tuple(int(value) for value in depth["fallback_radii_px"])
    fallback_min_valid = int(depth["fallback_min_valid"])
    raw_intrinsics = config["intrinsics"]
    distortion = tuple(float(value) for value in raw_intrinsics["dist_coeffs"])
    intrinsics = CameraIntrinsics(
        fx=float(raw_intrinsics["fx"]),
        fy=float(raw_intrinsics["fy"]),
        cx=float(raw_intrinsics["cx"]),
        cy=float(raw_intrinsics["cy"]),
        distortion=distortion,
    )
    extrinsic = config["extrinsic"]
    head_roll_from_camera = np.asarray(extrinsic["matrix"], dtype=np.float64)
    raw_chain = config["kinematics"]["joint_chain"]
    joint_origins = []
    for raw_joint in raw_chain:
        xyz_m = tuple(float(value) for value in raw_joint["origin_xyz_m"])
        rpy_rad = tuple(float(value) for value in raw_joint["origin_rpy_rad"])
        joint_origins.append(_origin_transform(xyz_m, rpy_rad))

    return HeadToGroundProjector(
        intrinsics,
        head_roll_from_camera,
        tuple(joint_origins),
        station_camera_key=camera["station_key"],
        image_size=(width, height),
        depth_scale_m=depth_scale_m,
        depth_range_m=(depth_range_m[0], depth_range_m[1]),
        fallback_radii_px=fallback_radii,
        fallback_min_valid=fallback_min_valid,
    )
