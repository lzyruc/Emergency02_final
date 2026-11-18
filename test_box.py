#!/usr/bin/env python3
"""从startpoint循环识别、接近、搬运并放置最近的纸箱。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from agent.navigation import AvoidCollision, NavigationGraph
from arm_motion import moveit_joints
from camera_io import read_head_camera
from task1 import BOX_HOLD_HORIZONTAL_PATH, BOX_PREGRASP_PATH
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


FRAME_COUNT = 5
FRAME_INTERVAL_S = 0.1
BOX_STANDOFF_M = 0.6
BOX_FINE_DEADBAND_M = 0.02
INITIAL_PLACE_DISTANCE_M = 1.0
PLACE_DISTANCE_DECREMENT_M = 0.25
INITIAL_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_emergency/应急_初始姿态.json"
)
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)


def joint_spd_from_delta(
    current_pos,
    target_pos,
    *,
    min_spd: float = 0.2,
    max_spd: float = 0.45,
) -> list[float]:
    """按关节位移同步缩放腰腿速度。"""
    if current_pos is None:
        return [min_spd] * len(target_pos)
    if len(current_pos) != len(target_pos):
        raise ValueError(
            f"current_pos与target_pos长度不一致："
            f"{len(current_pos)} vs {len(target_pos)}"
        )
    delta = [abs(float(target) - float(current)) for current, target in zip(
        current_pos,
        target_pos,
        strict=True,
    )]
    moving = [value for value in delta if value > 0.0]
    if not moving:
        return [min_spd] * len(target_pos)
    sync_time = max(moving) / max_spd
    return [
        min_spd
        if value == 0.0
        else min(max_spd, max(min_spd, value / sync_time))
        for value in delta
    ]


def execute_synced_waist_leg_action(
    station: Any,
    action: dict[str, Any],
    *,
    min_spd: float = 0.3,
    max_spd: float = 0.6,
) -> None:
    """同步腰腿速度执行预设，并恢复默认速度。"""
    waist_target = (action.get("waist") or {}).get("position")
    leg_target = (action.get("leg") or {}).get("position")
    if waist_target is None and leg_target is None:
        raise ValueError("action必须包含waist.position或leg.position")

    robot = station.get_robot_handle().get("robot")
    waist_dof = len(waist_target) if waist_target is not None else 0
    leg_dof = len(leg_target) if leg_target is not None else 0
    current_pos = []
    target_pos = []
    current_missing = False
    if waist_target is not None:
        current_waist = robot.state.get_waist_pos_list()
        current_missing = current_waist is None
        if current_waist is not None:
            current_pos.extend(current_waist)
        target_pos.extend(waist_target)
    if leg_target is not None:
        current_leg = robot.state.get_leg_pos_list()
        current_missing = current_missing or current_leg is None
        if current_leg is not None:
            current_pos.extend(current_leg)
        target_pos.extend(leg_target)

    all_spd = joint_spd_from_delta(
        None if current_missing else current_pos,
        target_pos,
        min_spd=min_spd,
        max_spd=max_spd,
    )
    try:
        if waist_target is not None:
            robot.waist_ctrler.set_motor_config(spd=all_spd[:waist_dof])
        if leg_target is not None:
            robot.leg_ctrler.set_motor_config(spd=all_spd[waist_dof:])
        station.execute_action(action)
        time.sleep(0.2)
    finally:
        if waist_target is not None:
            robot.waist_ctrler.set_motor_config(spd=[0.5, 0.3])
        if leg_target is not None:
            robot.leg_ctrler.set_motor_config(spd=[0.3] * leg_dof)


def execute_object_pose(station: Any, pose: dict[str, Any]) -> bool:
    """先执行双臂关节预设，再按确认执行腰腿预设。"""
    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        return False
    input("双臂已到位；确认腰腿路径安全后按 Enter 执行腰腿姿态：")
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        },
    )
    time.sleep(1.0)
    return True


def apply_startpoint_pose(station: Any, pose: dict[str, Any]) -> None:
    """腰腿回正后恢复应急初始腰腿和双臂姿态。"""
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": [0.0, 0.001]},
            "leg": {"position": [-0.5, 0.5]},
        },
    )
    input("腰腿回正完成；按 Enter 恢复应急初始腰腿和双臂姿态：")
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        },
    )
    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        raise RuntimeError("双臂恢复应急初始姿态失败")


def center_depth_m(
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> float | None:
    """返回检测框中心30%区域的有效深度中位数，单位米。"""
    x_min, y_min, x_max, y_max = bbox_xyxy
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    half_width = 0.15 * (x_max - x_min)
    half_height = 0.15 * (y_max - y_min)
    left = max(0, int(center_x - half_width))
    right = min(depth_mm.shape[1], int(center_x + half_width) + 1)
    top = max(0, int(center_y - half_height))
    bottom = min(depth_mm.shape[0], int(center_y + half_height) + 1)
    values = depth_mm[top:bottom, left:right]
    valid = values[(values > 0) & np.isfinite(values)]
    return None if valid.size == 0 else float(np.median(valid)) / 1000.0


def recognize_box_frame(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    frame_index: int,
) -> list[dict[str, Any]]:
    """读取一帧头部RGB-D，返回所有有效纸箱的base_footprint坐标。"""
    color, depth_mm = read_head_camera(station)
    if color is None or depth_mm is None:
        raise RuntimeError("未获取到头部RGB-D帧")
    detections = [
        detection
        for detection in detector.detect(color)
        if detection["type"] == "box"
    ]
    print(
        f"第{frame_index}/{FRAME_COUNT}帧：识别到{len(detections)}个纸箱",
        flush=True,
    )
    valid_detections = []
    samples = []
    for detection in detections:
        depth_m = center_depth_m(depth_mm, detection["bbox_xyxy"])
        if depth_m is None:
            print(
                f"  confidence={detection['confidence']:.3f}，深度无效",
                flush=True,
            )
            continue
        center_u, center_v = detection["center_uv"]
        valid_detections.append({**detection, "depth_m": depth_m})
        samples.append([center_u, center_v, depth_m])

    if not samples:
        return []
    poses_waist = deprojector.deproject_many(samples, face_target=False)
    targets = []
    for detection, pose_waist in zip(
        valid_detections,
        poses_waist,
        strict=True,
    ):
        point_base = deprojector.transform_to_ground(pose_waist[:3])
        target = {
            **detection,
            "point_base_m": point_base,
            "distance_xy_m": math.hypot(
                float(point_base[0]),
                float(point_base[1]),
            ),
        }
        targets.append(target)
        print(
            f"  confidence={target['confidence']:.3f}，"
            f"depth={target['depth_m']:.3f} m，"
            f"base x={point_base[0]:.3f} m，y={point_base[1]:.3f} m，"
            f"距离={target['distance_xy_m']:.3f} m",
            flush=True,
        )
    return targets


def recognize_nearest_box(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    prompt: str,
) -> dict[str, Any] | None:
    """五帧各取最近纸箱，至少命中4帧后返回坐标中位数。"""
    input(prompt)
    samples = []
    for frame_index in range(1, FRAME_COUNT + 1):
        targets = recognize_box_frame(
            station,
            detector,
            deprojector,
            frame_index,
        )
        if targets:
            samples.append(min(targets, key=lambda item: item["distance_xy_m"]))
        if frame_index < FRAME_COUNT:
            time.sleep(FRAME_INTERVAL_S)
    if len(samples) < 4:
        print(
            f"最近纸箱只命中{len(samples)}/{FRAME_COUNT}帧，结果不稳定",
            flush=True,
        )
        return None

    point_base = np.median(
        np.asarray([sample["point_base_m"] for sample in samples]),
        axis=0,
    )
    target = {
        "hits": len(samples),
        "confidence": float(np.mean([
            sample["confidence"] for sample in samples
        ])),
        "depth_m": float(np.mean([sample["depth_m"] for sample in samples])),
        "point_base_m": point_base,
        "distance_xy_m": math.hypot(float(point_base[0]), float(point_base[1])),
    }
    print(
        "最近纸箱五帧融合："
        f"命中={target['hits']}/{FRAME_COUNT}，"
        f"平均置信度={target['confidence']:.3f}，"
        f"平均深度={target['depth_m']:.3f} m，"
        f"base x={point_base[0]:.3f} m，y={point_base[1]:.3f} m，"
        f"距离={target['distance_xy_m']:.3f} m",
        flush=True,
    )
    return target


def plan_standoff_move(target: dict[str, Any]) -> tuple[float, float]:
    """沿目标方向平移，使目标与base原点相距0.60 m。"""
    x_m, y_m = (float(value) for value in target["point_base_m"][:2])
    distance_m = math.hypot(x_m, y_m)
    if distance_m == 0.0:
        raise ValueError("纸箱与base_footprint原点重合")
    travel_m = max(0.0, distance_m - BOX_STANDOFF_M)
    return x_m * travel_m / distance_m, y_m * travel_m / distance_m


def refine_box_approach(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
) -> bool:
    """再次识别最近纸箱，以x/y平移对齐；不旋转且不后退。"""
    target = recognize_nearest_box(
        station,
        detector,
        deprojector,
        "第一次接近完成；按 Enter 再识别5帧最近纸箱并计算微调：",
    )
    if target is None:
        return False
    target_x_m, target_y_m = (
        float(value) for value in target["point_base_m"][:2]
    )
    fine_x_m = max(0.0, target_x_m - BOX_STANDOFF_M)
    fine_y_m = target_y_m
    fine_distance_m = math.hypot(fine_x_m, fine_y_m)
    if target_x_m < BOX_STANDOFF_M:
        print("纸箱已近于base x=0.60 m，不后退，只修正y", flush=True)
    print(
        f"计划微调x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m、"
        "theta=0.0°",
        flush=True,
    )
    if fine_distance_m <= BOX_FINE_DEADBAND_M:
        print("微调距离不超过2 cm，跳过第二次moveBy", flush=True)
        return True
    if avoid.checkCollision(fine_x_m, fine_y_m, 0.0):
        print("纸箱微调路径与固定障碍物碰撞，不执行移动", flush=True)
        return False
    input(
        f"确认微调路径安全，按 Enter 执行moveBy："
        f"x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m、不旋转："
    )
    navigation.moveBy(fine_x_m, fine_y_m, 0.0)
    return True


def run_box_after_approach(
    station: Any,
    navigation: NavigationGraph,
    initial_pose: dict[str, Any],
    placement_distance_m: float,
) -> None:
    """抱持最近纸箱、移出通道、释放并恢复初始姿态和startpoint。"""
    pregrasp_pose = json.loads(
        BOX_PREGRASP_PATH.read_text(encoding="utf-8")
    )
    hold_pose = json.loads(
        BOX_HOLD_HORIZONTAL_PATH.read_text(encoding="utf-8")
    )
    input("按 Enter 张开双夹爪并切换纸箱预备姿态：")
    station.execute_action(
        {
            "hand": {
                "position": {
                    "left": [settings.TASK["hands"]["left_open_position"]],
                    "right": [settings.TASK["hands"]["right_open_position"]],
                }
            }
        }
    )
    time.sleep(1.0)
    if not execute_object_pose(station, pregrasp_pose):
        raise RuntimeError("纸箱预备姿态执行失败")

    input("确认双臂与纸箱位置安全，按 Enter 切换横纸箱抱持姿态：")
    if not execute_object_pose(station, hold_pose):
        raise RuntimeError("纸箱横向抱持姿态执行失败")

    input("按 Enter 执行底盘旋转-90°：")
    navigation.moveBy(0.0, 0.0, -90.0, correct=False)
    input("按 Enter 携带纸箱恢复startpoint：")
    navigation.recoverState("startpoint", method="move_by")
    input("确认右侧0.30 m安全，按 Enter 向右横移0.30 m：")
    navigation.moveBy(0.0, -0.3, 0.0)
    input("按 Enter 旋转90°：")
    navigation.moveBy(0.0, 0.0, 90.0, correct=True)
    input(
        f"确认前方{placement_distance_m:.2f} m安全，按 Enter 拖动纸箱："
    )
    navigation.moveBy(placement_distance_m, 0.0, 0.0)

    input("确认纸箱已在通道外，按 Enter 切换预备姿态释放纸箱：")
    if not execute_object_pose(station, pregrasp_pose):
        raise RuntimeError("纸箱释放姿态执行失败")
    input("确认纸箱已脱离双臂，按 Enter 恢复应急初始姿态：")
    apply_startpoint_pose(station, initial_pose)
    input("应急初始姿态已到达，按 Enter 恢复startpoint：")
    navigation.recoverState("startpoint", method="move_by")
    input("按 Enter 转到0°并结束本轮纸箱流程：")
    navigation.turnTo(0.0)


def main() -> None:
    """重复处理当前最近纸箱，稳定识别不到纸箱时结束。"""
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    detector = ObstacleYolo()
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    navigation_config = settings.TASK["navigation"]
    removed_count = 0
    try:
        time.sleep(1.0)
        deprojector = deprojector_module.Deprojector(
            station,
            CALIBRATION_PATH,
            "head",
        )
        with NavigationGraph(
            settings.prepare_route(),
            initial_node="startpoint",
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            if not navigation.checkState():
                raise RuntimeError("机器人实测位置不在startpoint")
            input("确认机器人周围安全，按 Enter 恢复应急初始姿态：")
            apply_startpoint_pose(station, initial_pose)
            avoid = AvoidCollision(
                navigation,
                "emergency.json",
                collision_radius_m=0.2,
            )

            round_index = 1
            while True:
                target = recognize_nearest_box(
                    station,
                    detector,
                    deprojector,
                    f"\n第{round_index}轮：确认机器人位于startpoint，"
                    "按 Enter 识别5帧并选择最近纸箱：",
                )
                if target is None:
                    print("没有稳定识别到纸箱，多纸箱测试结束", flush=True)
                    break

                move_x_m, move_y_m = plan_standoff_move(target)
                if avoid.checkCollision(move_x_m, move_y_m, 0.0):
                    print("最近纸箱第一段接近路径与固定障碍物碰撞，测试结束", flush=True)
                    break
                input(
                    f"确认第一段路径安全，按 Enter 平移到纸箱前："
                    f"x={move_x_m:+.3f} m、y={move_y_m:+.3f} m、"
                    "theta=0.0°："
                )
                navigation.moveBy(move_x_m, move_y_m, 0.0)

                if not refine_box_approach(
                    station,
                    navigation,
                    avoid,
                    detector,
                    deprojector,
                ):
                    input("纸箱复识别或微调失败；按 Enter 恢复startpoint重试：")
                    navigation.recoverState("startpoint", method="move_by")
                    round_index += 1
                    continue

                placement_distance_m = (
                    INITIAL_PLACE_DISTANCE_M
                    - PLACE_DISTANCE_DECREMENT_M * removed_count
                    + 0.30
                )
                run_box_after_approach(
                    station,
                    navigation,
                    initial_pose,
                    placement_distance_m,
                )
                removed_count += 1
                print(
                    f"第{round_index}轮完成，累计搬运{removed_count}个纸箱；"
                    "下一轮将重新识别全部纸箱",
                    flush=True,
                )
                round_index += 1
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
