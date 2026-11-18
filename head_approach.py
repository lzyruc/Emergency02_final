"""Locate a target (basket/box/chair) with head RGB-D and plan one base-frame move."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from head_to_ground import (
    HeadToGroundProjector,
    ProjectionResult,
    RobotJoints,
    load_projector,
)
from task2_phase1 import Phase1Model


class TargetNotFoundError(RuntimeError):
    """No requested target was present in a valid RGB-D frame."""


@dataclass(frozen=True, slots=True)
class ApproachPlan:
    target_class: str
    target_pixel_uv: tuple[int, int]
    projection: ProjectionResult
    x_scale: float
    y_scale: float
    turn_scale: float

    @property
    def move_by(self) -> tuple[float, float, float]:
        """Return metres/metres/degrees in +x forward, +y left."""
        x_m = self.projection.base_xyz_m[0] * self.x_scale
        y_m = self.projection.base_xyz_m[1] * self.y_scale
        heading_deg = math.degrees(math.atan2(y_m, x_m))
        return x_m, y_m, heading_deg * self.turn_scale


def read_projection_joints(station: Any) -> RobotJoints:
    """Read the live body/head joints required by the head-to-base chain."""
    state = station.get_robot_state()
    head = state["head_joints"]
    waist = state["waist_joints"]
    leg = state["leg_joints"]
    return RobotJoints(
        first_leg_pitch=float(leg[0]),
        second_leg_pitch=float(leg[1]),
        waist_yaw=float(waist[0]),
        waist_pitch=float(waist[1]),
        head_yaw=float(head[0]),
        head_pitch=float(head[1]),
        head_roll=float(head[2]),
    )


def plan_head_approach(
    station: Any,
    projector: HeadToGroundProjector,
    phase1_model: Phase1Model,
    task_config: dict[str, Any],
    color: Any,
    depth: Any,
    *,
    target_class: str = "basket",
    mode: str | None = None,
    output_dir: Path | None = None,
) -> ApproachPlan:
    """Turn one head RGB-D frame into a guarded base-frame approach.

    color/depth: camera_io.read_head_camera 解码后的 BGR 图与毫米深度图。
    target_class: "basket"/"box"/"chair"。
    mode: 不传时按 target_class 自动选（basket 用 navigation，其余用 grasp）。
    """
    if color is None or depth is None:
        raise TargetNotFoundError("head RGB-D frame is unavailable")
    if mode is None:
        mode = "navigation" if target_class == "basket" else "grasp"
    bgr = np.asarray(color)
    depth_mm = np.asarray(depth).astype(np.uint16)

    joints = read_projection_joints(station)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "head_rgb_raw.jpg"), bgr)
        cv2.imwrite(str(output_dir / "head_depth_mm.png"), depth_mm)

    detection = phase1_model.detect(
        {
            "rgb": bgr,
            "depth": depth_mm,
        },
        target_class=target_class,
        mode=mode,
    )
    if detection is None:
        raise TargetNotFoundError(f"Phase1Model did not detect a {target_class}")
    if detection["x"] is None or detection["y"] is None:
        raise TargetNotFoundError(f"Phase1Model {target_class} has no candidate pixel")
    target_pixel = (int(detection["x"]), int(detection["y"]))
    projection = projector.project(depth_mm, target_pixel, joints)
    navigation = task_config["navigation"]
    # 各类别暂共用 blue_approach_*_scale，需要单独调时按 target_class 加配置键。
    plan = ApproachPlan(
        target_class=detection["class"],
        target_pixel_uv=target_pixel,
        projection=projection,
        x_scale=float(navigation["blue_approach_x_scale"]),
        y_scale=float(navigation["blue_approach_y_scale"]),
        turn_scale=float(navigation["blue_approach_turn_scale"]),
    )
    x_m, y_m, theta_deg = plan.move_by
    print(
        "head target projection: "
        f"x={projection.base_xyz_m[0]:.3f} m, "
        f"y={projection.base_xyz_m[1]:.3f} m, "
        f"z={projection.base_xyz_m[2]:.3f} m; "
        "base approach uses x/y only",
        flush=True,
    )

    if output_dir is not None:
        overlay = np.ascontiguousarray(bgr).copy()
        cv2.circle(overlay, target_pixel, 10, (0, 255, 255), 3)
        cv2.imwrite(str(output_dir / "head_target.jpg"), overlay)
        report = {
            "target_class": detection["class"],
            "target_pixel_uv": list(target_pixel),
            "projection": asdict(projection),
            "move_by": {
                "x_m_forward": x_m,
                "y_m_left": y_m,
                "theta_deg_ccw": theta_deg,
            },
        }
        (output_dir / "head_approach.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return plan


_perception_cache: tuple[HeadToGroundProjector, Phase1Model] | None = None


def _perception() -> tuple[HeadToGroundProjector, Phase1Model]:
    """按 head_calibration.json 构建投影器与检测模型，仅构建一次。"""
    global _perception_cache
    if _perception_cache is None:
        intrinsics = settings.HEAD_CALIBRATION["intrinsics"]
        width, height = settings.HEAD_CALIBRATION["camera"]["image_size"]
        _perception_cache = (
            load_projector(settings.HEAD_CALIBRATION_PATH),
            Phase1Model(
                {
                    "fx": intrinsics["fx"],
                    "fy": intrinsics["fy"],
                    "cx": intrinsics["cx"],
                    "cy": intrinsics["cy"],
                    "width": width,
                    "height": height,
                }
            ),
        )
    return _perception_cache


def get_chassis_pose_from_rgbd(
    station: Any,
    rgb: Any,
    depth: Any,
    *,
    target_class: str = "basket",
    mode: str | None = None,
    output_dir: Path | None = None,
) -> list[float] | None:
    """识别指定类别目标，返回底盘相对位姿 [x 前, y 左, yaw 度]；识别不到返回 None。

    target_class: "basket"（默认）/"box"/"chair"。
    """
    if output_dir is None:
        output_dir = settings.RUNTIME_DIR / "head_approach" / target_class
    projector, phase1_model = _perception()
    try:
        plan = plan_head_approach(
            station,
            projector,
            phase1_model,
            settings.TASK,
            rgb,
            depth,
            target_class=target_class,
            mode=mode,
            output_dir=output_dir,
        )
    except TargetNotFoundError as exc:
        print(f"未识别到目标（target_class={target_class}）：{exc}", flush=True)
        return None
    return list(plan.move_by)
