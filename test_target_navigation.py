#!/usr/bin/env python3
"""从startpoint循环执行五帧识别、目标接近和任务1分物体测试。

头部RGB-D通过xrocs读取，底盘只通过NavigationGraph控制；纸箱目标会继续执行双臂
抱持和拖行，其余目标只执行预抓取姿态和腕部识别。坐标为base_footprint：+x前、
+y左，单位米。
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from agent.navigation import AvoidCollision, NavigationGraph
from arm_motion import moveit_cartesians, moveit_joints
from camera_io import read_head_camera
from emergency import execute_synced_waist_leg_action
from task1 import (
    BIN_LIFT_PATH,
    BIN_PREGRASP_PATHS,
    CHAIR_PREGRASP_PATHS,
    FILEBOX_LIFT_PATH,
    _execute_object_pose,
)
from test_box import run_box_after_approach
from wrist_phase1 import Phase1Model
from wrist_target import WRIST_DEPTH_SCALE_M, wrist_grasp_candidates
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TARGET_NAMES = {
    "chair": "办公转椅",
    "filebox": "文件筐",
    "bin": "垃圾桶",
    "box": "纸箱（3 kg）",
}
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
FRAME_COUNT = 5
FRAME_INTERVAL_S = 0.1
DEFAULT_STANDOFF_M = 0.5
STANDOFF_M_BY_TYPE = {"box": 0.35}
MIN_HITS_EXCLUSIVE = 4
INITIAL_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_emergency/应急_初始姿态.json"
)


def apply_startpoint_pose(station: Any, pose: dict[str, Any]) -> None:
    """腰腿先回正，再恢复双臂以及初始头、腰、腿姿态。"""
    print("startpoint姿态恢复：腰腿先回正并等待5秒", flush=True)
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": [0.0, 0.0]},
            "leg": {"position": [0.0, 0.0]},
        },
    )
    time.sleep(5.0)
    print("startpoint姿态恢复：移动双臂，再同步移动头、腰和腿", flush=True)
    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        raise RuntimeError("双臂恢复应急初始姿态失败")
    execute_synced_waist_leg_action(
        station,
        {
            "head": {"position": pose["head"]},
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        },
    )
    time.sleep(1.0)


def center_depth_m(
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> float | None:
    """返回检测框中心 30% 区域的有效深度中位数，单位为米。"""
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


def recognize_frame(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    frame_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """读取并打印一帧检测，返回具有有效 base_footprint 坐标的目标。"""
    color, depth_mm = read_head_camera(station)
    if color is None or depth_mm is None:
        raise RuntimeError("未获取到头部 RGB-D 帧，不能把相机故障视为无目标")
    detections = detector.detect(color)
    print(f"\n第 {frame_index}/{FRAME_COUNT} 帧：{len(detections)} 个检测", flush=True)
    if not detections:
        print("  未识别到目标", flush=True)
        return [], 0

    valid_detections = []
    samples = []
    for detection in detections:
        depth_m = center_depth_m(depth_mm, detection["bbox_xyxy"])
        if depth_m is None:
            print(
                f"  {TARGET_NAMES[detection['type']]}（{detection['type']}）："
                f"confidence={detection['confidence']:.3f}，"
                "depth=无有效值，base x/y=--",
                flush=True,
            )
            continue
        center_u, center_v = detection["center_uv"]
        valid_detections.append({**detection, "depth_m": depth_m})
        samples.append([center_u, center_v, depth_m])

    if not samples:
        return [], len(detections)
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
            "distance_xy_m": math.hypot(float(point_base[0]), float(point_base[1])),
        }
        targets.append(target)
        print(
            f"  {TARGET_NAMES[target['type']]}（{target['type']}）："
            f"confidence={target['confidence']:.3f}，"
            f"depth={target['depth_m']:.3f} m，"
            f"base x={point_base[0]:.3f} m，y={point_base[1]:.3f} m",
            flush=True,
        )
    return targets, len(detections)


def fuse_targets(
    frames: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """按类别融合五帧结果：base XYZ 取中位数，置信度和深度取均值。"""
    grouped = defaultdict(list)
    for frame_targets in frames:
        nearest_by_type = {}
        for target in frame_targets:
            target_type = target["type"]
            previous = nearest_by_type.get(target_type)
            if previous is None or target["distance_xy_m"] < previous["distance_xy_m"]:
                nearest_by_type[target_type] = target
        for target_type, target in nearest_by_type.items():
            grouped[target_type].append(target)

    fused = []
    for target_type, samples in grouped.items():
        point_base = np.median(
            np.asarray([sample["point_base_m"] for sample in samples]),
            axis=0,
        )
        fused.append(
            {
                "type": target_type,
                "name": TARGET_NAMES[target_type],
                "hits": len(samples),
                "confidence": float(
                    np.mean([sample["confidence"] for sample in samples])
                ),
                "depth_m": float(np.mean([sample["depth_m"] for sample in samples])),
                "point_base_m": point_base,
                "distance_xy_m": math.hypot(
                    float(point_base[0]),
                    float(point_base[1]),
                ),
            }
        )
    return fused


def plan_standoff_move(
    target: dict[str, Any],
    standoff_m: float = DEFAULT_STANDOFF_M,
) -> tuple[float, float, float]:
    """返回相对平移 x/y 和面向目标的相对旋转角度。"""
    x_m, y_m = (float(value) for value in target["point_base_m"][:2])
    distance_m = math.hypot(x_m, y_m)
    if distance_m == 0.0:
        raise ValueError("目标与 base_footprint 原点重合，无法计算接近方向")
    travel_m = max(0.0, distance_m - standoff_m)
    return (
        x_m * travel_m / distance_m,
        y_m * travel_m / distance_m,
        math.degrees(math.atan2(y_m, x_m)),
    )


def select_safe_approach(
    avoid: AvoidCollision,
    targets: list[dict[str, Any]],
) -> tuple[dict[str, Any], tuple[float, float, float]] | None:
    """按平面距离选择第一条无固定障碍物碰撞的目标接近动作。"""
    for target in sorted(targets, key=lambda item: item["distance_xy_m"]):
        standoff_m = STANDOFF_M_BY_TYPE.get(
            target.get("type"),
            DEFAULT_STANDOFF_M,
        )
        motion = plan_standoff_move(target, standoff_m)
        if avoid.checkCollision(*motion):
            print(
                f"跳过{target['name']}：{standoff_m:.2f} m接近直线路径"
                "与固定障碍物碰撞",
                flush=True,
            )
            continue
        return target, motion
    return None


def grasp_filebox_or_bin_from_bin_pose(
    station: Any,
    navigation: NavigationGraph,
    deprojectors: dict[str, deprojector_module.Deprojector],
    models: dict[str, Phase1Model],
    target_type: str,
) -> bool:
    """用垃圾桶远观察姿态抓取文件框或垃圾桶并移出通道。"""
    if target_type not in {"filebox", "bin"}:
        raise ValueError(f"不支持的抓取目标：{target_type}")
    target_name = TARGET_NAMES[target_type]
    prepare_path = BIN_PREGRASP_PATHS[0]
    lift_path = FILEBOX_LIFT_PATH if target_type == "filebox" else BIN_LIFT_PATH
    prepare_pose = json.loads(prepare_path.read_text(encoding="utf-8"))
    lift_pose = json.loads(lift_path.read_text(encoding="utf-8"))

    input(
        f"步骤1：确认安全后按 Enter 张开双夹爪并切换到"
        f"{prepare_path.name}；Ctrl+C 终止："
    )
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
    if not _execute_object_pose(station, prepare_pose):
        raise RuntimeError(f"{target_name}抓取预备姿态执行失败")

    input(f"步骤2：按 Enter 使用左右腕判断{target_name}抓取点；Ctrl+C 终止：")
    candidates = wrist_grasp_candidates(
        station,
        target_type,
        deprojectors,
        models,
    )
    if target_type == "filebox":
        candidates = [
            candidate
            for candidate in candidates
            if candidate["point_source"] == "grasp_line"
        ]
    succeeded = False
    if not candidates:
        print(f"左右腕均未识别到可靠的{target_name}抓取点", flush=True)
    else:
        candidate = min(candidates, key=lambda item: item["tcp_distance_m"])
        print(
            f"选择{candidate['arm']}臂：像素={candidate['pixel_uv']}，"
            f"TCP距离={candidate['tcp_distance_m']:.3f} m（不设距离上限），"
            f"置信度={candidate['confidence']:.3f}，"
            f"目标Pose={candidate['target_pose_waist_xyz_xyzw'].tolist()}",
            flush=True,
        )
        input(
            f"确认{candidate['arm']}臂路径安全，按 Enter 执行MoveIt抓取并"
            "闭合对应夹爪；Ctrl+C 终止："
        )
        target_pose = candidate["target_pose_waist_xyz_xyzw"].tolist()
        if candidate["arm"] == "left":
            succeeded = moveit_cartesians(station, left=[target_pose])
        else:
            succeeded = moveit_cartesians(station, right=[target_pose])
        if succeeded:
            hand_positions = {
                "left": settings.TASK["hands"]["left_open_position"],
                "right": settings.TASK["hands"]["right_open_position"],
            }
            hand_positions[candidate["arm"]] = settings.TASK["hands"][
                f"{candidate['arm']}_close_position"
            ]
            station.execute_action(
                {
                    "hand": {
                        "position": {
                            "left": [hand_positions["left"]],
                            "right": [hand_positions["right"]],
                        }
                    }
                }
            )
            time.sleep(1.0)
        else:
            print(f"{target_name}抓取点MoveIt执行失败，不闭合夹爪", flush=True)

    if succeeded:
        input(
            f"步骤3：按 Enter 切换到{lift_path.name}；"
            "先动头腰腿，等待3秒后再动双臂；Ctrl+C 终止："
        )
        execute_synced_waist_leg_action(
            station,
            {
                "head": {"position": lift_pose["head"]},
                "waist": {"position": lift_pose["waist"]},
                "leg": {"position": lift_pose["leg"]},
            },
        )
        time.sleep(3.0)
        succeeded = moveit_joints(
            station,
            left=lift_pose["left_arm"],
            right=lift_pose["right_arm"],
        )
        if not succeeded:
            print(f"{target_name}拿起姿态双臂执行失败", flush=True)

    input(
        f"步骤4：确认底盘返回路径安全，按 Enter 带{target_name}恢复startpoint；"
        "Ctrl+C 保持当前位置："
    )
    navigation.recoverState("startpoint", method="move_by")
    if not succeeded:
        print(f"{target_name}未成功拿起，已恢复startpoint并结束本次流程", flush=True)
        return False

    input(
        "步骤5：确认右侧0.3 m安全，按 Enter 向右横移0.3 m；"
        "Ctrl+C 终止："
    )
    navigation.moveBy(0.0, -0.3, 0.0)

    input("步骤6：按 Enter 执行 turnTo(90)；Ctrl+C 终止：")
    navigation.turnTo(90.0)

    input(
        "步骤7：确认前方0.8 m安全，按 Enter 向前移动0.8 m；"
        "Ctrl+C 终止："
    )
    navigation.moveBy(0.8, 0.0, 0.0)

    input(f"步骤8：确认{target_name}位于通道外，按 Enter 张开双夹爪释放；")
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

    input(
        f"步骤9：确认机械臂路径安全，按 Enter 切换回{prepare_path.name}；"
        "Ctrl+C 终止："
    )
    if not _execute_object_pose(station, prepare_pose):
        raise RuntimeError(f"{target_name}释放后预备姿态执行失败")

    input(
        "步骤10：确认返回路径安全，按 Enter 恢复startpoint；"
        "Ctrl+C 保持当前位置："
    )
    navigation.recoverState("startpoint", method="move_by")

    input(f"步骤11：按 Enter 转到0°并结束{target_name}流程；Ctrl+C 终止：")
    navigation.turnTo(0.0)
    return succeeded


def recognize_target_from_task1_poses(
    station: Any,
    target_type: str,
    pose_paths: tuple[Path, ...],
    deprojectors: dict[str, deprojector_module.Deprojector],
    models: dict[str, Phase1Model],
) -> dict[str, Any] | None:
    """依次测试目标自己的预抓取姿态和双腕识别，不执行抓取。"""
    for index, path in enumerate(pose_paths):
        input(
            f"{TARGET_NAMES[target_type]}已在目标前：确认双臂、腰腿路径安全后"
            f"按 Enter 切换预抓取姿态 {index}（{path.name}）并执行双腕识别；"
            "Ctrl+C 终止："
        )
        pose = json.loads(path.read_text(encoding="utf-8"))
        if not _execute_object_pose(station, pose):
            raise RuntimeError(
                f"{TARGET_NAMES[target_type]}预抓取姿态 {index} 执行失败"
            )
        candidates = wrist_grasp_candidates(
            station,
            target_type,
            deprojectors,
            models,
        )
        if not candidates:
            print(
                f"{TARGET_NAMES[target_type]}预抓取姿态 {index}："
                "左右腕均无候选点",
                flush=True,
            )
            continue
        candidates.sort(key=lambda item: item["tcp_distance_m"])
        for candidate in candidates:
            print(
                f"{TARGET_NAMES[target_type]}预抓取姿态 {index}："
                f"{candidate['arm']}腕候选，像素={candidate['pixel_uv']}，"
                f"深度={candidate['depth_m']:.3f} m"
                f"（{candidate['depth_source']}），"
                f"点来源={candidate['point_source']}，"
                f"TCP距离={candidate['tcp_distance_m']:.3f} m，"
                f"置信度={candidate['confidence']:.3f}，"
                f"选择分={candidate['selection_score']:.3f}，"
                f"目标Pose={candidate['target_pose_waist_xyz_xyzw'].tolist()}",
                flush=True,
            )
        nearest = candidates[0]
        if nearest["tcp_distance_m"] < 0.4:
            print(
                f"按 task1 规则将选择{nearest['arm']}臂候选；"
                "本测试不执行MoveIt抓取，也不闭合夹爪",
                flush=True,
            )
            return nearest
    print(
        f"{len(pose_paths)}个{TARGET_NAMES[target_type]}预抓取姿态均无 "
        "0.4 m 内候选点",
        flush=True,
    )
    return None


def filter_targets_by_hits(
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只保留五帧中命中数严格大于4的目标。"""
    return [
        target
        for target in targets
        if int(target["hits"]) > MIN_HITS_EXCLUSIVE
    ]


def main() -> None:
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
    try:
        time.sleep(1.0)
        deprojector = deprojector_module.Deprojector(
            station,
            CALIBRATION_PATH,
            "head",
        )
        wrist_deprojectors = {
            side: deprojector_module.Deprojector(
                station,
                CALIBRATION_PATH,
                side,
            )
            for side in ("left", "right")
        }
        wrist_models = {
            side: Phase1Model(
                {
                    "fx": float(camera["camera_matrix"][0][0]),
                    "fy": float(camera["camera_matrix"][1][1]),
                    "cx": float(camera["camera_matrix"][0][2]),
                    "cy": float(camera["camera_matrix"][1][2]),
                    "width": int(camera["image_width"]),
                    "height": int(camera["image_height"]),
                    "depth_scale_m": WRIST_DEPTH_SCALE_M,
                    "invalid_depth_values": [0, 65535],
                }
            )
            for side, camera in (
                ("left", settings.LEFT_CAMERA),
                ("right", settings.RIGHT_CAMERA),
            )
        }
        with NavigationGraph(
            settings.prepare_route(),
            initial_node="startpoint",
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            if not navigation.checkState():
                raise RuntimeError("机器人实测位置不在 startpoint，停止长程测试")
            input(
                "初始化已确认位于startpoint；确认双臂、头、腰和腿路径安全后"
                "按 Enter 恢复应急初始姿态："
            )
            apply_startpoint_pose(station, initial_pose)
            avoid = AvoidCollision(navigation, "emergency.json")

            round_index = 1
            while True:
                input(
                    f"\n第 {round_index} 轮：确认机器人位于 startpoint，"
                    "按 Enter 连续识别5帧；Ctrl+C 终止："
                )
                frames = []
                raw_detection_count = 0
                for frame_index in range(1, FRAME_COUNT + 1):
                    targets, detection_count = recognize_frame(
                        station,
                        detector,
                        deprojector,
                        frame_index,
                    )
                    frames.append(targets)
                    raw_detection_count += detection_count
                    if frame_index < FRAME_COUNT:
                        time.sleep(FRAME_INTERVAL_S)

                if raw_detection_count == 0:
                    print("连续5帧均未识别到物体，长程测试结束", flush=True)
                    break

                fused_targets = fuse_targets(frames)
                if not fused_targets:
                    print(
                        "检测到了物体，但深度均无效；不移动，重新开始下一轮",
                        flush=True,
                    )
                    round_index += 1
                    continue

                print("\n五帧融合结果：", flush=True)
                for target in sorted(
                    fused_targets,
                    key=lambda item: item["distance_xy_m"],
                ):
                    point_base = target["point_base_m"]
                    print(
                        f"  {target['name']}（{target['type']}）："
                        f"命中={target['hits']}/{FRAME_COUNT}，"
                        f"平均置信度={target['confidence']:.3f}，"
                        f"平均深度={target['depth_m']:.3f} m，"
                        f"base x={point_base[0]:.3f} m，"
                        f"y={point_base[1]:.3f} m，"
                        f"平面距离={target['distance_xy_m']:.3f} m",
                        flush=True,
                    )

                reliable_targets = filter_targets_by_hits(fused_targets)
                for target in fused_targets:
                    if int(target["hits"]) <= MIN_HITS_EXCLUSIVE:
                        print(
                            f"过滤{target['name']}：命中="
                            f"{target['hits']}/{FRAME_COUNT}，要求命中>4/5",
                            flush=True,
                        )
                if not reliable_targets:
                    print(
                        "本轮没有5/5稳定命中的目标；不移动，重新开始下一轮",
                        flush=True,
                    )
                    round_index += 1
                    continue

                approach = select_safe_approach(avoid, reliable_targets)
                if approach is None:
                    print(
                        "本轮所有目标的直线接近路径均会碰撞；"
                        "planApproach尚未实现，停止测试且不发送底盘命令",
                        flush=True,
                    )
                    break
                nearest, (move_x, move_y, turn_deg) = approach
                standoff_m = STANDOFF_M_BY_TYPE.get(
                    nearest["type"],
                    DEFAULT_STANDOFF_M,
                )
                input(
                    f"最近目标为{nearest['name']}，计划平移 "
                    f"x={move_x:.3f} m、y={move_y:.3f} m，"
                    f"停在目标前 {standoff_m:.2f} m 后旋转 {turn_deg:.1f}°；"
                    "确认安全后按 Enter 执行："
                )
                navigation.moveBy(move_x, move_y, 0.0)
                navigation.moveBy(0.0, 0.0, turn_deg)

                if nearest["type"] == "box":
                    print(
                        "稳定识别目标为纸箱，进入横纸箱抱持、拖行和释放流程",
                        flush=True,
                    )
                    run_box_after_approach(station, navigation)
                elif nearest["type"] in {"filebox", "bin"}:
                    print(
                        f"稳定识别目标为{nearest['name']}，使用垃圾桶远观察姿态"
                        "执行单臂抓取、拿起和移出通道",
                        flush=True,
                    )
                    grasp_filebox_or_bin_from_bin_pose(
                        station,
                        navigation,
                        wrist_deprojectors,
                        wrist_models,
                        nearest["type"],
                    )
                elif nearest["type"] == "chair":
                    pose_paths = CHAIR_PREGRASP_PATHS
                    print(
                        f"稳定识别目标为{nearest['name']}，进入 task1 "
                        f"{len(pose_paths)}姿态双腕识别测试；不会执行抓取",
                        flush=True,
                    )
                    recognize_target_from_task1_poses(
                        station,
                        nearest["type"],
                        pose_paths,
                        wrist_deprojectors,
                        wrist_models,
                    )
                    input(
                        f"{nearest['name']}腕部识别测试结束；确认机械臂路径安全后"
                        "按 Enter 先恢复应急初始姿态，再恢复 startpoint："
                    )
                    apply_startpoint_pose(station, initial_pose)
                    navigation.recoverState("startpoint", method="move_by")
                else:
                    input(
                        "机器人已到目标前并朝向目标；确认返回路径安全后按 Enter "
                        "恢复 startpoint："
                    )
                    navigation.recoverState()

                print("已恢复 startpoint", flush=True)
                if nearest["type"] not in {"filebox", "chair", "bin"}:
                    input(
                        "确认机器人已在startpoint且姿态恢复路径安全，按 Enter恢复"
                        "应急初始姿态："
                    )
                    apply_startpoint_pose(station, initial_pose)
                round_index += 1
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
