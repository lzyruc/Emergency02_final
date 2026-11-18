#!/usr/bin/env python3
"""依次清理四类障碍物，然后执行任务2窗口视觉操作。

头部RGB-D通过xrocs读取，底盘只通过NavigationGraph控制。坐标为
base_footprint：+x前、+y左，单位米。
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from agent.navigation import AvoidCollision, NavigationGraph
from arm_motion import moveit_cartesians, moveit_joints
from bin_filebox_grasp import (
    candidate_motion_info,
    select_candidate,
    wrist_yolo_grasp_candidates,
)
from black_grasp_single import grasp_black_single
from camera_io import read_head_camera, read_wrist_camera
from task1 import (
    BIN_LIFT_PATH,
    BIN_PREGRASP_PATHS,
    BOX_HOLD_HORIZONTAL_PATH,
    BOX_PREGRASP_PATH,
    CHAIR_PREGRASP_PATHS,
    FILEBOX_LIFT_PATH,
)
from waist_leg_motion import execute_synced_waist_leg_action
from window_handle_recognition import recognize_window_handle
from wrist_phase1 import Phase1Model
from wrist_target import WRIST_DEPTH_SCALE_M, wrist_grasp_candidates
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager

# 四类障碍物中文名
TARGET_NAMES = {
    "chair": "办公转椅",
    "filebox": "文件筐",
    "bin": "垃圾桶",
    "box": "纸箱（3 kg）",
}
# 相机标定文件路径
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
# 多帧识别与接近停靠参数
FRAME_COUNT = 5
FRAME_INTERVAL_S = 0.1
DEFAULT_STANDOFF_M = 0.4
STANDOFF_M_BY_TYPE = {"box": 0.45}
BOX_FINAL_STANDOFF_M = 0.45
BOX_FINE_DEADBAND_M = 0.02
TURN_REDUCTION_DEG_BY_TYPE = {"bin": 0.0, "filebox": 0.0}
COLLISION_RADIUS_M = 0.25
MIN_HITS_EXCLUSIVE = 3
INITIAL_PLACE_DISTANCE_M = 1.0
PLACE_DISTANCE_DECREMENT_M = 0.25
# 每个视角重拍2帧；原位失败后相对首次TCP沿waist_yaw_link Y轴左右搜索3 cm。
WINDOW_HANDLE_VIEW_OFFSETS_Y_M = (0.0, 0.03, -0.03)
WINDOW_HANDLE_FRAMES_PER_VIEW = 2
MAX_TASK_ATTEMPTS = 3  # 任务1 单类目标最大尝试次数
MAX_STAGE_ATTEMPTS = 3  # 任务阶段失败后最大重试次数
WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M = 0.35
WINDOW2_HANDLE_MAX_TCP_DELTA_Z_M = 0.25
# 推窗最多尝试3次，每次保留上一次90%的总偏移；剩余前向距离不足1 cm时不再补推。
WINDOW_PUSH_MAX_ATTEMPTS = 7
WINDOW_PUSH_RETRY_SCALE = 0.9
WINDOW_PUSH_MIN_RETRY_TRAVEL_M = 0.01
WINDOW_PUSH_INITIAL_DELTA_TCP_M = (0.00, 0.0, -0.35)
# 应急初始姿态 JSON
INITIAL_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_emergency/应急_初始姿态.json"
)

def run_with_retry_or_skip(
    task_name: str,
    action: Any,
    *,
    max_attempts: int = MAX_STAGE_ATTEMPTS,
) -> bool:
    """执行子任务；失败则询问重试，用尽次数后跳过，不退出整机流程。

    action 返回 False 视为失败；返回 None/其他非 False 视为成功。
    成功返回 True，跳过返回 False。
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = action()
            if result is False:
                raise RuntimeError(f"{task_name}执行返回失败")
            print(f"{task_name}完成", flush=True)
            return True
        except Exception as error:
            print(f'{task_name}失败（第{attempt}/{max_attempts}次）：{error}', flush=True)
            if attempt >= max_attempts:
                input(f"{task_name}已达最大重试次数，按 Enter 跳过并继续后续任务：")
                print(f"已跳过{task_name}", flush=True)
                return False
            choice = input(f'{task_name}：按 Enter 再次尝试，输入 s 跳过该任务：').strip().lower()
            if choice in {"s", "skip", "跳过"}:
                print(f"已跳过{task_name}", flush=True)
                return False
    return False

# 执行物体预设姿态（双臂→腰腿）
def _execute_object_pose(
    station: Any,
    pose: dict[str, Any],
) -> bool:
    """串行执行双臂、Body腰腿和头部预设，不改变夹爪状态。"""
    moveit_success = moveit_joints(station, left=pose['left_arm'], right=pose['right_arm'])
    if not moveit_success:
        return False
    input('确认手臂已到位，确认腰腿路径安全后按 Enter 执行Body腰腿和头部姿态：')
    execute_synced_waist_leg_action(station, {'waist': {'position': pose['waist']}, 'leg': {'position': pose['leg']}})
    time.sleep(1.0)
    return True

# 恢复应急 startpoint 姿态
def apply_startpoint_pose(
    station: Any,
    pose: dict[str, Any],
) -> bool:
    """腰腿先回正，再恢复双臂以及初始头、腰、腿姿态。失败返回 False。"""
    # 先把腰腿回到安全位
    print("startpoint姿态恢复：移动双臂，再同步移动头、腰和腿", flush=True)
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [0.0, 0.001]},
        'leg': {'position': [-0.5, 0.5]},
        'head': {'position': [0.0, 0.2, 0.0]},
    })
    time.sleep(5.0)
    input('[start]腰腿回正后按enter继续')
    execute_synced_waist_leg_action(station, {'waist': {'position': pose['waist']}, 'leg': {'position': pose['leg']}})
    time.sleep(0.5)
    # 再恢复双臂到应急初始姿态
    moveit_success = moveit_joints(station, left=pose['left_arm'], right=pose['right_arm'])
    if not moveit_success:
        print("双臂恢复应急初始姿态失败", flush=True)
        return False
    return True

# 纸箱：抱持、拖出通道、释放并回位
def run_box_after_approach(
    station: Any,
    navigation: NavigationGraph,
    initial_pose: dict[str, Any],
    placement_distance_m: float,
    approach_origin_node: str,
) -> bool:
    """机器人已在纸箱前时抱持、拖行、释放，恢复初始姿态并回位。失败返回 False。"""
    # 读取纸箱预备与横向抱持姿态
    pregrasp_pose = json.loads(
        BOX_PREGRASP_PATH.read_text(encoding='utf-8'),)
    hold_pose = json.loads(
        BOX_HOLD_HORIZONTAL_PATH.read_text(encoding='utf-8'),)
    input('纸箱步骤1：确认安全后按 Enter 张开夹爪并切换纸箱预备姿态：')
    station.execute_action({'hand': {'position': {'left': [settings.TASK['hands']['left_open_position']], 'right': [settings.TASK['hands']['right_open_position']]}}})
    time.sleep(1.0)
    pose_success = _execute_object_pose(station, pregrasp_pose)
    if not pose_success:
        print("纸箱预备姿态执行失败", flush=True)
        return False
    input('纸箱步骤2：确认双臂与纸箱位置安全，按 Enter 切换横纸箱抱持姿态：')
    pose_success = _execute_object_pose(station, hold_pose)
    if not pose_success:
        print("纸箱横向抱持姿态执行失败", flush=True)
        return False
    # input("纸箱步骤3：按 Enter 执行旋转-90度")
    # navigation.moveBy(0,0,theta=-90, correct=False)
    input('纸箱步骤4：按 Enter 恢复startpoint')
    # 抱箱后先回到观察点
    return_from_approach(navigation, approach_origin_node)
    input('纸箱步骤6：确认右侧0.3 m无障碍，按 Enter 向右横移0.3 m：')
    navigation.moveBy(0.0, -0.3, 0.0)
    input("纸箱步骤7：按 Enter 执行 turnTo(90)")
    navigation.moveBy(0.0,0.0,90.0,correct=True)
    input(f'纸箱步骤8：确认前方{placement_distance_m:.1f} m安全，按 Enter 向前拖动{placement_distance_m:.1f} m')
    navigation.moveBy(placement_distance_m, 0.0, 0.0)
    input('纸箱步骤9：确认纸箱已在通道外，按 Enter 切换预备姿态释放纸箱：')
    pose_success = _execute_object_pose(station, pregrasp_pose)
    if not pose_success:
        print("纸箱释放姿态执行失败", flush=True)
        return False
    input('纸箱步骤10：确认纸箱已脱离双臂且机械臂路径安全，按 Enter 执行完整应急初始姿态恢复')
    pose_success = apply_startpoint_pose(station, initial_pose)
    if not pose_success:
        return False
    input('纸箱步骤11：应急初始姿态已到达，按 Enter 恢复startpoint：')
    navigation.recoverState("startpoint", method="move_by")
    input("纸箱步骤12：按 Enter 执行 turnTo(0)并结束")
    navigation.turnTo(0.0)
    print("横纸箱抱持、拖行、初始姿态恢复和回位完成", flush=True)
    return True

# 检测框中心区域深度中位数（米）
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

# 单帧头部识别并投影到 base_footprint
def recognize_frame(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    frame_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """读取并打印一帧检测，返回具有有效 base_footprint 坐标的目标。"""
    # 读头部 RGB-D 并 YOLO 检测
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
            print(f"  {TARGET_NAMES[detection['type']]}（{detection['type']}）：confidence={detection['confidence']:.3f}，depth=无有效值，base x/y=--", flush=True)
            continue
        center_u, center_v = detection["center_uv"]
        valid_detections.append({**detection, "depth_m": depth_m})
        samples.append([center_u, center_v, depth_m])
    if not samples:
        return [], len(detections)
    # 像素+深度 → waist → base
    poses_waist = deprojector.deproject_many(samples, face_target=False)
    targets = []
    for detection, pose_waist in zip(
        valid_detections, poses_waist, strict=True,):
        point_base = deprojector.transform_to_ground(pose_waist[:3])
        target = {
            **detection,
            "point_base_m": point_base,
            "distance_xy_m": math.hypot(float(point_base[0]), float(point_base[1])),
        }
        targets.append(target)
        print(f"  {TARGET_NAMES[target['type']]}（{target['type']}）：confidence={target['confidence']:.3f}，depth={target['depth_m']:.3f} m，base x={point_base[0]:.3f} m，y={point_base[1]:.3f} m", flush=True)
    return targets, len(detections)

# 五帧按类别融合（中位数/均值）
def fuse_targets(
    frames: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """按类别融合五帧结果：base XYZ 取中位数，置信度和深度取均值。"""
    # 每帧每类只保留最近目标再融合
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
            np.asarray([sample['point_base_m'] for sample in samples]), axis=0,)
        fused.append(
            {'type': target_type, 'name': TARGET_NAMES[target_type], 'hits': len(samples), 'confidence': float(np.mean([sample['confidence'] for sample in samples])), 'depth_m': float(np.mean([sample['depth_m'] for sample in samples])), 'point_base_m': point_base, 'distance_xy_m': math.hypot(float(point_base[0]), float(point_base[1]))},)
    return fused

# 计算停在目标前 standoff 的相对位移与朝向
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

# 选最近且无碰撞的接近动作
def select_safe_approach(
    avoid: AvoidCollision,
    targets: list[dict[str, Any]],
    *,
    max_abs_y_m: float | None = None,
) -> tuple[dict[str, Any], tuple[float, float, float]] | None:
    """按平面距离选择第一条无固定障碍物碰撞的目标接近动作。"""
    # 由近到远尝试；可截断横向位移
    for target in sorted(targets, key=lambda item: item["distance_xy_m"]):
        standoff_m = STANDOFF_M_BY_TYPE.get(
            target.get('type'), DEFAULT_STANDOFF_M,)
        motion = plan_standoff_move(target, standoff_m)
        reduction_deg = TURN_REDUCTION_DEG_BY_TYPE.get(target.get("type"), 0.0)
        if motion[2] > 0.0:
            motion = (motion[0], motion[1], motion[2] + reduction_deg)
        elif motion[2] < 0.0:
            motion = (motion[0], motion[1], motion[2] - reduction_deg)
        if max_abs_y_m is not None and abs(motion[1]) > max_abs_y_m:
            original_y_m = motion[1]
            motion = (
                motion[0],
                math.copysign(max_abs_y_m, original_y_m),
                motion[2],
            )
            print(f"{target['name']}接近delta-y={original_y_m:+.3f} m超过限制，截断为{motion[1]:+.3f} m，继续检查和移动", flush=True)
        has_collision = avoid.checkCollision(*motion)
        if has_collision:
            print(f"跳过{target['name']}：{standoff_m:.2f} m接近直线路径与固定障碍物碰撞", flush=True)
            continue
        return target, motion
    return None

# 任务1：startpoint / h0 观察并选路
def find_task1_approach(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    remaining_types: set[str],
    round_index: int,
    *,
    start_from_h0: bool = False,
) -> tuple[
    str,
    list[dict[str, Any]],
    tuple[dict[str, Any], tuple[float, float, float]] | None,
]:
    """先在startpoint识别和选路；无稳定目标或接近路线碰撞时再去h0复识别。h0仍无目标则结束选路，不回startpoint。"""
    if not remaining_types:
        origin = "h0_track_turn_2" if start_from_h0 else "startpoint"
        print("没有剩余未处理类别，跳过观察点识别和移动", flush=True)
        return origin, [], None
    reliable_targets: list[dict[str, Any]] = []
    observation_nodes = (
        ("h0_track_turn_2",)
        if start_from_h0
        else ("startpoint", "h0_track_turn_2")
    )
    last_node = observation_nodes[0]
    for observation_node in observation_nodes:
        last_node = observation_node
        if observation_node == "h0_track_turn_2":
            if start_from_h0:
                print("上一轮目标来自h0_track_turn_2，保持在h0继续识别", flush=True)
            else:
                print("startpoint无稳定目标或所有目标接近路线碰撞，直接到h0_track_turn_2重新识别", flush=True)
                navigation.recoverState("h0_track_turn_2", method="move_by")
        fused_targets = recognize_task1_batch(
            station, detector, deprojector, f'\n第 {round_index} 轮：确认机器人位于 startpoint，按 Enter 连续识别5帧' if observation_node == 'startpoint' else '机器人已到h0_track_turn_2；按 Enter 连续识别5帧',)
        unprocessed_targets = [
            target for target in fused_targets if target["type"] in remaining_types]
        reliable_targets = filter_targets_by_hits(unprocessed_targets)
        for target in fused_targets:
            if int(target["hits"]) <= MIN_HITS_EXCLUSIVE:
                print(f"过滤{target['name']}：命中={target['hits']}/{FRAME_COUNT}，要求命中>=4/5", flush=True)
        if not reliable_targets:
            print(f"{observation_node}没有稳定命中的未处理目标", flush=True)
            if observation_node == "h0_track_turn_2":
                print("h0无稳定未处理目标，不再返回startpoint", flush=True)
                return observation_node, [], None
            continue
        print(f"当前观察点={observation_node}，开始检查稳定目标的接近路线", flush=True)
        approach = select_safe_approach(
            avoid, reliable_targets, max_abs_y_m=0.1 if observation_node == "h0_track_turn_2" else None,)
        if approach is not None:
            return observation_node, reliable_targets, approach
        if observation_node == "startpoint":
            print("startpoint识别到稳定目标，但所有直线接近路线均碰撞；不进入任务2，改从h0_track_turn_2复识别", flush=True)
    return last_node, reliable_targets, None

# 从目标前回到本轮观察点（必要时再回 startpoint）
def return_from_approach(
    navigation: NavigationGraph,
    approach_origin_node: str,
) -> None:
    """从目标前恢复本轮观察点；第二观察点任务再回到startpoint。"""
    navigation.recoverState(approach_origin_node, method="move_by")
    if approach_origin_node != "startpoint":
        navigation.recoverState("startpoint", method="move_by")

# 从任务1观察点进入任务2起点
def move_from_observation_to_task2_start(
    navigation: NavigationGraph,
    observation_node: str,
) -> None:
    """从任务1当前观察点沿固定路线进入任务2起点。"""
    if observation_node == "startpoint":
        navigation.recoverState(
            'h0_track_turn_2', method='move_by',)
    navigation.moveTo("h0_track_end", correct=True)
    navigation.turnTo(270.0)
    navigation.moveTo(
        'h270_task2_start', correct=True, align=False,)

# 纸箱近距离复识别并平移微调
def refine_box_approach(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
) -> bool:
    """近距离复识别纸箱，并以base_footprint下的x/y平移到最终操作位置。"""
    input('纸箱第一次接近完成；按 Enter 再连续识别5帧并计算平移微调：')
    # 近距离再采 5 帧
    frames = []
    for frame_index in range(1, FRAME_COUNT + 1):
        try:
            targets, _ = recognize_frame(
                station, detector, deprojector, frame_index,)
        except RuntimeError as error:
            print(f"纸箱近距离第{frame_index}帧识别失败：{error}", flush=True)
            return False
        frames.append(targets)
        if frame_index < FRAME_COUNT:
            time.sleep(FRAME_INTERVAL_S)
    boxes = [
        target
        for target in filter_targets_by_hits(fuse_targets(frames))
        if target["type"] == "box"
    ]
    if not boxes:
        print("纸箱近距离5帧复识别不稳定，不使用startpoint旧坐标继续移动", flush=True)
        return False
    target = min(boxes, key=lambda item: item["distance_xy_m"])
    target_x_m, target_y_m = (
        float(value) for value in target["point_base_m"][:2]
    )
    # 只前进到最终停靠，不后退
    fine_x_m = max(0.0, target_x_m - BOX_FINAL_STANDOFF_M)
    fine_y_m = target_y_m
    fine_distance_m = math.hypot(fine_x_m, fine_y_m)
    if target_x_m < BOX_FINAL_STANDOFF_M:
        print('纸箱已近于最终base x基准，第二次微调不后退，只修正y', flush=True)
    print(f"纸箱近距离融合结果：命中={target['hits']}/{FRAME_COUNT}，base x={target_x_m:.3f} m，y={target_y_m:.3f} m；计划微调x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m、theta=0.0°", flush=True)
    if fine_distance_m <= BOX_FINE_DEADBAND_M:
        print("纸箱平移误差不超过2 cm，跳过第二次moveBy", flush=True)
        return True
    has_collision = avoid.checkCollision(fine_x_m, fine_y_m, 0.0)
    if has_collision:
        print("纸箱第二次平移路径与固定障碍物碰撞，不执行移动", flush=True)
        return False
    input(f'确认纸箱微调路径安全，按 Enter 执行moveBy：x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m，不旋转：')
    navigation.moveBy(fine_x_m, fine_y_m, 0.0)
    return True

# 文件框/垃圾桶：腕部识别抓取并移出通道
def grasp_filebox_or_bin_from_bin_pose(
    station: Any,
    navigation: NavigationGraph,
    detector: ObstacleYolo,
    deprojectors: dict[str, deprojector_module.Deprojector],
    target_type: str,
    placement_distance_m: float,
    approach_origin_node: str,
) -> bool:
    """用垃圾桶远观察姿态抓取文件框或垃圾桶并移出通道。"""
    if target_type not in {"filebox", "bin"}:
        raise ValueError(f"不支持的抓取目标：{target_type}")
    target_name = TARGET_NAMES[target_type]
    # 统一用垃圾桶远观察预备姿态
    prepare_path = BIN_PREGRASP_PATHS[0]
    lift_path = FILEBOX_LIFT_PATH if target_type == "filebox" else BIN_LIFT_PATH
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    prepare_pose = json.loads(prepare_path.read_text(encoding="utf-8"))
    lift_pose = json.loads(lift_path.read_text(encoding="utf-8"))
    input(f'步骤1：确认安全后按 Enter 张开双夹爪并切换到{prepare_path.name}')
    station.execute_action({'hand': {'position': {'left': [settings.TASK['hands']['left_open_position']], 'right': [settings.TASK['hands']['right_open_position']]}}})
    time.sleep(1.0)
    pose_success = _execute_object_pose(station, prepare_pose)
    if not pose_success:
        print(f"{target_name}抓取预备姿态执行失败", flush=True)
        return False
    # 左右腕估计抓取候选
    input(f"步骤2：按 Enter 使用左右腕判断{target_name}抓取点")
    output_dir = (
        settings.ROOT
        / "output/bin_filebox_grasp"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target_type}"
    )
    candidates = wrist_yolo_grasp_candidates(
        station, target_type, detector, deprojectors, output_dir, read_wrist_camera,)
    serializable = []
    for candidate in candidates:
        candidate.update(candidate_motion_info(station, candidate))
        serializable.append(
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in candidate.items()},)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'candidates.json').write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
    succeeded = False
    # 选最优臂与抓取点
    candidate = select_candidate(target_type, candidates)
    if candidate is None:
        print(f"左右腕均未识别到可靠的{target_name}抓取点", flush=True)
    else:
        motion = candidate_motion_info(station, candidate)
        point_tcp = motion["point_tcp_xyz_m"]
        print(f"选择{candidate['arm']}臂：像素={candidate['pixel_uv']}，ROI={candidate['roi_source']}，来源={candidate['point_source']}，深度={candidate['depth_m']:.3f} m，TCP局部点=({point_tcp[0]:+.4f}, {point_tcp[1]:+.4f}, {point_tcp[2]:+.4f}) m，TCP距离={motion['distance_m']:.3f} m（不设距离上限），置信度={candidate['confidence']:.3f}，目标Pose={candidate['target_pose_waist_xyz_xyzw'].tolist()}", flush=True)
        robot = station.get_robot_handle().get("robot")
        state = station.get_robot_state()
        if candidate["arm"] == "left":
            T_waist_tcp = robot.pin_kine.get_left_tcp_pose(state['arm_joints']['left']).get_affine_matrix_m_ndarray()
        else:
            T_waist_tcp = robot.pin_kine.get_right_tcp_pose(state['arm_joints']['right']).get_affine_matrix_m_ndarray()
        current_pose = state["arm_pose"][candidate["arm"]]
        current_pose_waist = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64,)
        # 第一段：TCP 局部 x/y 对齐
        first_delta_tcp_m = np.asarray(
            [point_tcp[0], point_tcp[1], 0.0], dtype=np.float64,)
        first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
        first_target_pose_waist = current_pose_waist.copy()
        first_target_pose_waist[:3] = (
            T_waist_tcp[:3, 3] + first_delta_waist_m
        )
        input(f"确认{candidate['arm']}臂第一段TCP局部x/y路径安全，按 Enter 执行MoveIt")
        if candidate["arm"] == "left":
            succeeded = moveit_cartesians(station, left=[first_target_pose_waist.tolist()], use_cartesian_path=True, left_motion_id=1)
        else:
            succeeded = moveit_cartesians(station, right=[first_target_pose_waist.tolist()], use_cartesian_path=True, right_motion_id=1)
        if succeeded:
            state = station.get_robot_state()
            if candidate["arm"] == "left":
                T_waist_tcp = robot.pin_kine.get_left_tcp_pose(state['arm_joints']['left']).get_affine_matrix_m_ndarray()
            else:
                T_waist_tcp = robot.pin_kine.get_right_tcp_pose(state['arm_joints']['right']).get_affine_matrix_m_ndarray()
            current_pose = state["arm_pose"][candidate["arm"]]
            current_pose_waist = np.asarray(
                current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64,)
            # 第二段：TCP 局部 z + 配置额外量
            extra_z_tcp_m = float(
                settings.TASK['bin_filebox_grasp']['second_approach_extra_z_tcp_m'][target_type],)
            second_delta_tcp_m = np.asarray(
                [0.0, 0.0, point_tcp[2] + extra_z_tcp_m], dtype=np.float64,)
            second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
            second_target_pose_waist = current_pose_waist.copy()
            second_target_pose_waist[:3] = (
                T_waist_tcp[:3, 3] + second_delta_waist_m
            )
            (output_dir / 'motion_plan.json').write_text(
                json.dumps({'selected_arm': candidate['arm'], 'point_tcp_xyz_m': point_tcp.tolist(), 'first_delta_tcp_xyz_m': first_delta_tcp_m.tolist(), 'first_delta_waist_xyz_m': first_delta_waist_m.tolist(), 'second_delta_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_delta_waist_xyz_m': second_delta_waist_m.tolist(), 'second_approach_extra_z_tcp_m': extra_z_tcp_m, 'z_rule': 'recognized_tcp_z_m + configured_extra_z_tcp_m'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
            input(f"确认{candidate['arm']}臂第二段TCP局部z路径安全：识别z={point_tcp[2] * 100:+.2f} cm，额外偏移={extra_z_tcp_m * 100:+.2f} cm，最终z={second_delta_tcp_m[2] * 100:+.2f} cm；按 Enter 执行MoveIt")
            if candidate["arm"] == "left":
                succeeded = moveit_cartesians(station, left=[second_target_pose_waist.tolist()], use_cartesian_path=True, left_motion_id=2)
            else:
                succeeded = moveit_cartesians(station, right=[second_target_pose_waist.tolist()], use_cartesian_path=True, right_motion_id=2)
        if succeeded:
            # 接近成功后闭合对应夹爪
            hand_positions = {
                "left": settings.TASK["hands"]["left_open_position"],
                "right": settings.TASK["hands"]["right_open_position"],
            }
            hand_positions[candidate["arm"]] = settings.TASK["hands"][
                f"{candidate['arm']}_close_position"
            ]
            station.execute_action({'hand': {'position': {'left': [hand_positions['left']], 'right': [hand_positions['right']]}}})
            time.sleep(1.0)
        else:
            print(f"{target_name}TCP对齐失败，不闭合夹爪", flush=True)
    if succeeded:
        input(f'步骤3：按 Enter 切换到{lift_path.name}；先动头腰腿，等待3秒后再动双臂')
        execute_synced_waist_leg_action(station, {'waist': {'position': lift_pose['waist']}, 'leg': {'position': lift_pose['leg']}})
        time.sleep(3.0)
        succeeded = moveit_joints(station, left=lift_pose['left_arm'], right=lift_pose['right_arm'])
        # 失败则恢复姿态并回 startpoint
        if not succeeded:
            print(f"{target_name}拿起姿态双臂执行失败", flush=True)
    if not succeeded:
        input(f'{target_name}检测或抓取失败；确认恢复路径安全后按 Enter 切回应急初始姿态')
        apply_startpoint_pose(station, initial_pose)
        input('初始姿态已恢复，确认底盘路径安全后按 Enter 恢复startpoint：')
        return_from_approach(navigation, approach_origin_node)
        print(f"{target_name}未成功拿起，已恢复初始姿态和startpoint", flush=True)
        return False
    input(f'步骤4：确认底盘返回路径安全，按 Enter 带{target_name}恢复startpoint：')
    return_from_approach(navigation, approach_origin_node)
    # input(
    #     "步骤5：确认右侧0.3 m安全，按 Enter 向右横移0.3 m；"
    #     ""
    # )
    # navigation.moveBy(0.0, -0.3, 0.0)
    input("步骤6：按 Enter 执行 turnTo(90)")
    navigation.turnTo(90.0)
    input(f'步骤7：确认前方{placement_distance_m:.1f} m安全，按 Enter 向前移动{placement_distance_m:.1f} m')
    navigation.moveBy(placement_distance_m, 0.0, 0.0)
    input(f"步骤8：确认{target_name}位于通道外，按 Enter 张开双夹爪释放：")
    station.execute_action({'hand': {'position': {'left': [settings.TASK['hands']['left_open_position']], 'right': [settings.TASK['hands']['right_open_position']]}}})
    time.sleep(1.0)
    # input(
    #     f"步骤9：确认机械臂路径安全，按 Enter 切换回{prepare_path.name}；"
    #     ""
    # )
    # if not _execute_object_pose(station, prepare_pose):
    #     raise RuntimeError(f"{target_name}释放后预备姿态执行失败")
    input('步骤10：确认机械臂、腰腿路径安全，按 Enter 执行完整应急初始姿态恢复')
    pose_success = apply_startpoint_pose(station, initial_pose)
    if not pose_success:
        return False
    input('步骤11：应急初始姿态已到达，确认返回路径安全后按 Enter 恢复startpoint：')
    navigation.recoverState("startpoint", method="move_by")
    input(f"步骤12：按 Enter 转到0°并结束{target_name}流程")
    navigation.turnTo(0.0)
    return succeeded

# 转椅等：预抓取姿态下双腕识别（不抓取）
def recognize_target_from_task1_poses(
    station: Any,
    target_type: str,
    pose_paths: tuple[Path, ...],
    deprojectors: dict[str, deprojector_module.Deprojector],
    models: dict[str, Phase1Model],
) -> dict[str, Any] | None:
    """依次测试目标自己的预抓取姿态和双腕识别，不执行抓取。"""
    for index, path in enumerate(pose_paths):
        input(f'{TARGET_NAMES[target_type]}已在目标前：确认双臂、腰腿路径安全后按 Enter 切换预抓取姿态 {index}（{path.name}）并执行双腕识别：')
        pose = json.loads(path.read_text(encoding="utf-8"))
        pose_success = _execute_object_pose(station, pose)
        if not pose_success:
            print(f'{TARGET_NAMES[target_type]}预抓取姿态 {index} 执行失败，尝试下一姿态', flush=True)
            continue
        candidates = wrist_grasp_candidates(
            station, target_type, deprojectors, models,)
        if not candidates:
            print(f'{TARGET_NAMES[target_type]}预抓取姿态 {index}：左右腕均无候选点', flush=True)
            continue
        candidates.sort(key=lambda item: item["tcp_distance_m"])
        for candidate in candidates:
            print(f"{TARGET_NAMES[target_type]}预抓取姿态 {index}：{candidate['arm']}腕候选，像素={candidate['pixel_uv']}，深度={candidate['depth_m']:.3f} m（{candidate['depth_source']}），点来源={candidate['point_source']}，TCP距离={candidate['tcp_distance_m']:.3f} m，置信度={candidate['confidence']:.3f}，选择分={candidate['selection_score']:.3f}，目标Pose={candidate['target_pose_waist_xyz_xyzw'].tolist()}", flush=True)
        nearest = candidates[0]
        if nearest["tcp_distance_m"] < 0.4:
            print(f"按 task1 规则将选择{nearest['arm']}臂候选；本测试不执行MoveIt抓取，也不闭合夹爪", flush=True)
            return nearest
    print(f'{len(pose_paths)}个{TARGET_NAMES[target_type]}预抓取姿态均无 0.4 m 内候选点', flush=True)
    return None

# 过滤命中不足的目标
def filter_targets_by_hits(
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只保留五帧中命中数严格大于4的目标。"""
    return [
        target
        for target in targets
        if int(target["hits"]) > MIN_HITS_EXCLUSIVE
    ]

# 连续识别五帧并融合
def recognize_task1_batch(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    prompt: str,
) -> list[dict[str, Any]]:
    """按确认连续识别五帧，打印并返回融合后的全部任务1目标。"""
    input(prompt)
    frames = []
    for frame_index in range(1, FRAME_COUNT + 1):
        try:
            targets, _ = recognize_frame(
                station, detector, deprojector, frame_index,)
        except RuntimeError as error:
            print(f"第{frame_index}帧识别失败，本批识别中止：{error}", flush=True)
            return []
        frames.append(targets)
        if frame_index < FRAME_COUNT:
            time.sleep(FRAME_INTERVAL_S)
    fused_targets = fuse_targets(frames)
    if not fused_targets:
        print("本观察点五帧内没有具有有效深度的目标", flush=True)
        return []
    print("\n五帧融合结果：", flush=True)
    for target in sorted(
        fused_targets, key=lambda item: item['distance_xy_m'],):
        point_base = target["point_base_m"]
        print(f"  {target['name']}（{target['type']}）：命中={target['hits']}/{FRAME_COUNT}，平均置信度={target['confidence']:.3f}，平均深度={target['depth_m']:.3f} m，base x={point_base[0]:.3f} m，y={point_base[1]:.3f} m，平面距离={target['distance_xy_m']:.3f} m", flush=True)
    return fused_targets

# 窗把手多视角识别，校验 TCP 前向距离
def recognize_window_handle_with_tcp_z_retry(
    station: Any,
    output_dir: Path,
    max_delta_z_m: float,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None:
    """原位重拍失败后横移左TCP搜索，只接受前向距离有效的目标点。

    返回识别结果、waist_yaw_link目标Pose、
    ``waist_yaw_link <- left_tcp_link`` 齐次矩阵和目标在
    left_tcp_link下的XYZ，长度单位均为米。所有视角均失败时恢复首次
    TCP位姿并返回None；搜索位移位于waist_yaw_link，TCP方向保持不变。
    """
    deprojector = deprojector_module.Deprojector(
        station, CALIBRATION_PATH, 'left',)
    robot = station.get_robot_handle().get("robot")
    initial_state = station.get_robot_state()
    initial_pose = initial_state["arm_pose"]["left"]
    initial_pose_waist = np.asarray(
        initial_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(initial_pose, "get_xyz_m_xyzw_ndarray")
        else initial_pose,
        dtype=np.float64,
    ).copy()
    moved_from_initial = False
    for view_index, offset_y_m in enumerate(
        WINDOW_HANDLE_VIEW_OFFSETS_Y_M,
        start=1,
    ):
        if offset_y_m != 0.0:
            view_pose_waist = initial_pose_waist.copy()
            view_pose_waist[1] += offset_y_m
            input(
                f"窗把手原位识别失败；确认左臂横移路径安全，按 Enter 将TCP"
                f"移动到相对首次位姿waist-Y={offset_y_m * 100:+.1f} cm："
            )
            if not moveit_cartesians(
                station,
                left=view_pose_waist,
                use_cartesian_path=True,
                left_motion_id=19 + view_index,
            ):
                print("窗把手视角横移失败，停止搜索", flush=True)
                break
            moved_from_initial = True

        for frame_index in range(1, WINDOW_HANDLE_FRAMES_PER_VIEW + 1):
            attempt_dir = (
                output_dir
                / f"view_{view_index}_y_{offset_y_m:+.3f}"
                / f"frame_{frame_index}"
            )
            try:
                result = recognize_window_handle(station, attempt_dir)
            except RuntimeError as error:
                print(
                    f"窗把手视角{view_index}第{frame_index}/"
                    f"{WINDOW_HANDLE_FRAMES_PER_VIEW}帧未检出：{error}",
                    flush=True,
                )
                continue
            target_pose_waist = deprojector.deproject(
                [*result['pixel_uv'], result['depth_m']],
                face_target=False,
            )
            state = station.get_robot_state()
            T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                state['arm_joints']['left']
            ).get_affine_matrix_m_ndarray()
            point_left_tcp_m = (
                np.linalg.inv(T_waist_tcp)
                @ np.append(target_pose_waist[:3], 1.0)
            )[:3]
            delta_z_m = float(point_left_tcp_m[2])
            result.update(
                {
                    'recognition_view': view_index,
                    'recognition_frame': frame_index,
                    'view_offset_waist_y_m': offset_y_m,
                    'point_left_tcp_xyz_m': point_left_tcp_m.tolist(),
                    'tcp_delta_z_accepted_range_m': [-max_delta_z_m, 0.0],
                }
            )
            (attempt_dir / 'result.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            print(
                f'窗把手视角{view_index}第{frame_index}/'
                f'{WINDOW_HANDLE_FRAMES_PER_VIEW}帧：'
                f'waist-Y={offset_y_m * 100:+.1f} cm，'
                f'left_tcp_link delta-x={point_left_tcp_m[0] * 100:+.2f} cm，'
                f'delta-y={point_left_tcp_m[1] * 100:+.2f} cm，'
                f'delta-z={delta_z_m * 100:+.2f} cm，'
                f'前向上限={max_delta_z_m * 100:.1f} cm',
                flush=True,
            )
            if -max_delta_z_m <= delta_z_m <= 0.0:
                return result, target_pose_waist, T_waist_tcp, point_left_tcp_m
            print("识别点前向距离超限，本视角继续重拍", flush=True)

    if moved_from_initial:
        input(
            "所有窗把手搜索视角均失败；确认恢复路径安全，按 Enter 将左TCP"
            "恢复到首次识别位姿："
        )
        if not moveit_cartesians(
            station,
            left=initial_pose_waist,
            use_cartesian_path=True,
            left_motion_id=23,
        ):
            print("窗把手搜索失败且左TCP恢复首次位姿失败", flush=True)
            return None
    print("所有窗把手搜索视角均未获得有效结果", flush=True)
    return None

# 窗口2：识别、对齐、扭窗、底盘开窗
def run_window2_action(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """在h270_window2_action执行左腕识别、TCP对齐和开窗动作。"""
    output_dir = (
        settings.ROOT
        / "output/task2_recognize"
        / datetime.now().strftime("%Y%m%d_%H%M%S_window2")
    )
    input('任务2窗口2操作步骤1：机器人已到h270_window2_action，按 Enter 读取左腕RGB-D、保存图像并识别窗把手')
    try:
        recognized = recognize_window_handle_with_tcp_z_retry(
            station, output_dir, WINDOW2_HANDLE_MAX_TCP_DELTA_Z_M,)
    except RuntimeError as error:
        print(f'窗口2把手识别失败，跳过后续对齐和扭窗步骤：{error}', flush=True)
        return False
    if recognized is None:
        print("窗口2开窗判定超出阈值，跳过后续操作并进入下一项目", flush=True)
        return False
    result, target_pose_waist, T_waist_tcp, point_left_tcp_m = recognized
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    robot = station.get_robot_handle().get("robot")
    # 先修 TCP x/y，再沿 z 靠近把手
    first_delta_tcp_m = np.asarray(
        [point_left_tcp_m[0], point_left_tcp_m[1], 0.0], dtype=np.float64,)
    first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(f'窗口2识别点（left_tcp_link）：x={point_left_tcp_m[0] * 100:+.2f} cm，y={point_left_tcp_m[1] * 100:+.2f} cm，z={point_left_tcp_m[2] * 100:+.2f} cm', flush=True)
    print(f'窗口2第一次TCP局部偏移：x={first_delta_tcp_m[0] * 100:+.2f} cm，y={first_delta_tcp_m[1] * 100:+.2f} cm，z=+0.00 cm；方向保持不变', flush=True)
    input('任务2窗口2操作步骤2：确认路径安全，按 Enter 只修正TCP局部x、y：')
    moveit_success = moveit_cartesians(station, left=first_target_pose_waist, use_cartesian_path=True, left_motion_id=8)
    if not moveit_success:
        print("窗口2左臂第一次x/y修正失败", flush=True)
        return False
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(state['arm_joints']['left']).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    current_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64,)
    second_delta_tcp_m = np.asarray(
        [0.0, 0.0, point_left_tcp_m[2] + 0.19], dtype=np.float64,)
    second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
    second_target_pose_waist = current_pose_waist.copy()
    second_target_pose_waist[:3] = T_waist_tcp[:3, 3] + second_delta_waist_m
    print(f'窗口2第二次TCP局部z偏移：识别z={point_left_tcp_m[2] * 100:+.2f} cm + 19.00 cm = {second_delta_tcp_m[2] * 100:+.2f} cm；方向保持不变', flush=True)
    result.update(
        {'point_left_tcp_xyz_m': point_left_tcp_m.tolist(), 'first_delta_left_tcp_xyz_m': first_delta_tcp_m.tolist(), 'second_delta_left_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_z_rule': 'recognized_left_tcp_z_m + 0.19 m'},)
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
    input('任务2窗口2操作步骤3：确认路径安全，按 Enter 只沿TCP局部z移动：')
    moveit_success = moveit_cartesians(station, left=second_target_pose_waist, use_cartesian_path=True, left_motion_id=9)
    if not moveit_success:
        print("窗口2左臂第二次z修正失败", flush=True)
        return False
    # 腰系平移靠近把手侧
    for step, motion_id, label, delta_waist_m in (
        (4, 10, "沿waist_yaw_link -Y向视觉右侧移动11 cm", [0.0, -0.11, 0.0]),
        (5, 11, "沿waist_yaw_link +Z向上移动3 cm", [0.0, 0.0, 0.03]),
        (6, 12, "沿waist_yaw_link -Y向视觉右侧移动3 cm", [0.0, -0.03, 0.0]),
    ):
        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
        target_pose_waist[:3] += np.asarray(delta_waist_m, dtype=np.float64)
        input(f'任务2窗口2操作步骤{step}：确认路径安全，按 Enter {label}：')
        moveit_success = moveit_cartesians(station, left=target_pose_waist, use_cartesian_path=True, left_motion_id=motion_id)
        if not moveit_success:
            print(f"窗口2左臂{label}失败", flush=True)
            return False
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    rotation_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    # 绕夹爪局部 z 顺时针 90°
    qx, qy, qz, qw = rotation_target_pose_waist[3:7]
    cos_h = math.cos(math.pi / 4.0)
    sin_h = math.sin(math.pi / 4.0)
    qw_new = qw * cos_h - qz * sin_h
    qx_new = qx * cos_h + qy * sin_h
    qy_new = qy * cos_h - qx * sin_h
    qz_new = qz * cos_h + qw * sin_h
    norm = math.sqrt(
        qw_new ** 2 + qx_new ** 2 + qy_new ** 2 + qz_new ** 2,)
    rotation_target_pose_waist[3:7] = (
        qx_new / norm,
        qy_new / norm,
        qz_new / norm,
        qw_new / norm,
    )
    input('任务2窗口2操作步骤7：确认路径安全，按 Enter 保持TCP位置不变，绕当前夹爪局部z轴顺时针旋转90°')
    moveit_success = moveit_cartesians(station, left=rotation_target_pose_waist, use_cartesian_path=True, left_motion_id=13)
    if not moveit_success:
        print("窗口2左臂TCP绕局部z轴顺时针旋转90°失败", flush=True)
        return False
    for step, motion_id, label, delta_waist_m in (
        (8, 14, "沿waist_yaw_link +Z向上移动2 cm", [0.0, 0.0, 0.02]),
        (9, 15, "沿waist_yaw_link +Y向视觉左侧移动5 cm", [0.0, 0.05, 0.0]),
    ):
        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
        target_pose_waist[:3] += np.asarray(delta_waist_m, dtype=np.float64)
        input(f'任务2窗口2操作步骤{step}：确认路径安全，按 Enter {label}：')
        moveit_success = moveit_cartesians(station, left=target_pose_waist, use_cartesian_path=True, left_motion_id=motion_id)
        if not moveit_success:
            print(f"窗口2左臂{label}失败", flush=True)
            return False
    input('任务2窗口2操作步骤10：确认把手位于夹爪内，按 Enter 闭合左夹爪：')
    station.execute_action({'hand': {'position': {'left': [1.0], 'right': [0.0]}}})
    input('任务2窗口2操作步骤11：确认底盘、窗户和机械臂路径安全，按 Enter 执行底盘开窗：dx=-32 cm，dy=-15 cm')
    # 底盘拉开窗户
    navigation.moveBy(-0.32, -0.15, 0.0, linear_speed=0.1)
    input('任务2窗口2操作步骤12：检查开窗结果，按 Enter 松开左夹爪：')
    station.execute_action({'hand': {'position': {'left': [0.0], 'right': [0.0]}}})
    input('任务2窗口2操作步骤13：保持左夹爪张开，确认机械臂路径安全，按 Enter 恢复双臂到应急_窗户2_准备姿态')
    prepare_pose = settings.WINDOW2_PREPARE_POSE
    moveit_success = moveit_joints(station, left=prepare_pose['left_arm'], right=prepare_pose['right_arm'])
    if not moveit_success:
        print("窗口2操作结束后双臂恢复准备姿态失败", flush=True)
        return False
    return True

# 推窗失败则缩短偏移重试
def _push_window_with_reduced_offset(
    station: Any,
    robot: Any,
    *,
    first_motion_id: int,
    label: str,
    initial_delta_tcp_m: np.ndarray,
) -> bool:
    """以首次TCP位姿为基准缩短总偏移重试，不叠加每次位移。"""
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
        state["arm_joints"]["left"]
    ).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    start_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    ).copy()
    initial_delta_tcp_m = np.asarray(
        initial_delta_tcp_m,
        dtype=np.float64,
    ).copy()
    initial_delta_waist_m = T_waist_tcp[:3, :3] @ initial_delta_tcp_m
    initial_distance_m = float(np.linalg.norm(initial_delta_waist_m))
    if initial_distance_m == 0.0:
        raise ValueError("推窗TCP偏移不能为零")
    push_direction_waist = initial_delta_waist_m / initial_distance_m

    for attempt in range(1, WINDOW_PUSH_MAX_ATTEMPTS + 1):
        scale = WINDOW_PUSH_RETRY_SCALE ** (attempt - 1)
        delta_tcp_m = initial_delta_tcp_m * scale
        delta_waist_m = initial_delta_waist_m * scale
        target_pose_waist = start_pose_waist.copy()
        target_pose_waist[:3] = start_pose_waist[:3] + delta_waist_m
        attempt_label = "首次" if attempt == 1 else f"第{attempt}次缩距重试"
        input(
            f"任务2推窗：确认机械臂已停止且路径安全，"
            f"按 Enter {attempt_label}沿左TCP{label}移动："
            f"相对首次起点的TCP总偏移="
            f"({delta_tcp_m[0] * 100:+.1f}, "
            f"{delta_tcp_m[1] * 100:+.1f}, "
            f"{delta_tcp_m[2] * 100:+.1f}) cm；Ctrl+C 终止："
        )
        if attempt > 1:
            latest_state = station.get_robot_state()
            latest_T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                latest_state["arm_joints"]["left"]
            ).get_affine_matrix_m_ndarray()
            remaining_forward_m = float(
                np.dot(
                    target_pose_waist[:3] - latest_T_waist_tcp[:3, 3],
                    push_direction_waist,
                )
            )
            if remaining_forward_m <= WINDOW_PUSH_MIN_RETRY_TRAVEL_M:
                print(
                    f"当前TCP已达到或越过"
                    f"{np.linalg.norm(delta_tcp_m) * 100:.1f} cm缩距目标，"
                    "视为推窗成功，不执行可能反向的重试",
                    flush=True,
                )
                return True

        motion_id = first_motion_id + attempt - 1
        if moveit_cartesians(
            station,
            left=target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=motion_id,
        ):
            print(
                f"左臂TCP{label}推窗成功：第{attempt}次尝试，"
                f"偏移={np.linalg.norm(delta_tcp_m) * 100:.1f} cm，"
                f"motion_id={motion_id}",
                flush=True,
            )
            return True
        if attempt < WINDOW_PUSH_MAX_ATTEMPTS:
            next_distance_cm = (
                np.linalg.norm(initial_delta_tcp_m)
                * WINDOW_PUSH_RETRY_SCALE**attempt
                * 100.0
            )
            print(
                f"左臂TCP{label}第{attempt}次推窗失败，"
                f"下一次总偏移缩短为{next_distance_cm:.1f} cm",
                flush=True,
            )
    print(
        f"左臂TCP{label}推窗连续"
        f"{WINDOW_PUSH_MAX_ATTEMPTS}次失败",
        flush=True,
    )
    return False

# ---------------------------------------------------------------------------
# 任务1 杂物：垃圾桶 / 椅子 / 文件框 / 箱子
# ---------------------------------------------------------------------------
def run_task1_bin(
    station: Any,
    navigation: NavigationGraph,
    detector: ObstacleYolo,
    wrist_deprojectors: dict[str, deprojector_module.Deprojector],
    placement_distance_m: float,
    approach_origin_node: str,
) -> bool:
    """任务1-垃圾桶。"""
    grasp_success = grasp_filebox_or_bin_from_bin_pose(station, navigation, detector, wrist_deprojectors, 'bin', placement_distance_m, approach_origin_node)
    return grasp_success

def run_task1_filebox(
    station: Any,
    navigation: NavigationGraph,
    detector: ObstacleYolo,
    wrist_deprojectors: dict[str, deprojector_module.Deprojector],
    placement_distance_m: float,
    approach_origin_node: str,
) -> bool:
    """任务1-文件框。"""
    grasp_success = grasp_filebox_or_bin_from_bin_pose(station, navigation, detector, wrist_deprojectors, 'filebox', placement_distance_m, approach_origin_node)
    return grasp_success

def run_task1_chair(
    station: Any,
    navigation: NavigationGraph,
    wrist_deprojectors: dict[str, deprojector_module.Deprojector],
    wrist_models: dict[str, Phase1Model],
    initial_pose: dict[str, Any],
    approach_origin_node: str,
    attempt_index: int,
) -> bool:
    """任务1-椅子（预抓取姿态双腕识别，不执行抓取）。"""
    pose_paths = CHAIR_PREGRASP_PATHS
    print(f"稳定识别目标为{TARGET_NAMES['chair']}，第{attempt_index}/{MAX_TASK_ATTEMPTS}次尝试；进入 task1 {len(pose_paths)}姿态双腕识别测试；不会执行抓取", flush=True)
    recognize_target_from_task1_poses(
        station, 'chair', pose_paths, wrist_deprojectors, wrist_models,)
    input(f"{TARGET_NAMES['chair']}腕部识别测试结束；确认机械臂路径安全后按 Enter 先恢复应急初始姿态，再恢复 startpoint：")
    pose_success = apply_startpoint_pose(station, initial_pose)
    if not pose_success:
        return False
    return_from_approach(navigation, approach_origin_node)
    return True

def run_task1_box(
    station: Any,
    navigation: NavigationGraph,
    initial_pose: dict[str, Any],
    placement_distance_m: float,
    approach_origin_node: str,
    attempt_index: int,
) -> bool:
    """任务1-纸箱：抱持拖行（调用前需已完成近距微调）。"""
    print(f'稳定识别目标为纸箱，第{attempt_index}/{MAX_TASK_ATTEMPTS}次尝试；进入横纸箱抱持、拖行和释放流程', flush=True)
    box_success = run_box_after_approach(station, navigation, initial_pose, placement_distance_m + 0.3, approach_origin_node)
    return box_success

def run_task1_debris(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    wrist_deprojectors: dict[str, deprojector_module.Deprojector],
    wrist_models: dict[str, Phase1Model],
    initial_pose: dict[str, Any],
) -> bool:
    """任务1杂物：循环清理四类障碍物，结束后进入任务2起点。"""
    round_index = 1
    remaining_types = set(TARGET_NAMES)
    type_attempt_counts: dict[str, int] = defaultdict(int)
    removed_object_count = 0
    next_round_from_h0 = False
    last_observation = "startpoint"
    moved_to_task2 = False
    while remaining_types:
        try:
            (
                approach_origin_node,
                reliable_targets,
                approach,
            ) = find_task1_approach(
                station, navigation, avoid, detector, deprojector, remaining_types, round_index, start_from_h0=next_round_from_h0,)
        except Exception as approach_error:
            print(f"任务1接近选路异常：{approach_error}", flush=True)
            choice = input("按 Enter 重试选路，输入 s 放弃任务1进入任务2：").strip().lower()
            if choice in {"s", "skip", "跳过"}:
                try:
                    move_from_observation_to_task2_start(navigation, last_observation)
                except Exception as move_error:
                    print(f"进入任务2起点失败：{move_error}", flush=True)
                moved_to_task2 = True
                break
            continue
        next_round_from_h0 = False
        last_observation = approach_origin_node
        if approach is None:
            if reliable_targets:
                print('h0_track_turn_2识别到稳定目标，但将delta-y限制为±0.10 m后所有直线接近路线仍会碰撞', flush=True)
                input('确认放弃本轮未清理目标，且通往任务2的固定路线安全，按 Enter 继续到h270_task2_start')
            else:
                print(f'{approach_origin_node}无稳定未处理目标，进入任务2起点', flush=True)
            move_from_observation_to_task2_start(navigation, approach_origin_node)
            moved_to_task2 = True
            break
        nearest, (move_x, move_y, turn_deg) = approach
        standoff_m = STANDOFF_M_BY_TYPE.get(nearest["type"], DEFAULT_STANDOFF_M)
        input(f"最近目标为{nearest['name']}，计划平移 x={move_x:.3f} m、y={move_y:.3f} m，停在目标前 {standoff_m:.2f} m 后旋转 {turn_deg:.1f}°；确认安全后按 Enter 执行：")
        navigation.moveBy(move_x, move_y, 0.0)
        if nearest["type"] != "box":
            print("都不转")
        placement_distance_m = (
            INITIAL_PLACE_DISTANCE_M
            - PLACE_DISTANCE_DECREMENT_M * removed_object_count
        )
        target_type = nearest["type"]
        attempt_index = type_attempt_counts[target_type] + 1
        task1_name = f"任务1-{nearest['name']}"
        # 纸箱近距微调失败则回位续跑，不进入重试封装
        if target_type == "box":
            refine_success = refine_box_approach(
                station, navigation, avoid, detector, deprojector,)
            if not refine_success:
                input('纸箱近距离复识别或微调失败；确认返回路径安全后按 Enter 恢复startpoint：')
                return_from_approach(navigation, approach_origin_node)
                round_index += 1
                continue

        def run_current_target() -> bool:
            if target_type == "box":
                box_success = run_task1_box(station, navigation, initial_pose, placement_distance_m, approach_origin_node, attempt_index)
                return box_success
            if target_type == "bin":
                print(f"稳定识别目标为{nearest['name']}，第{attempt_index}/{MAX_TASK_ATTEMPTS}次尝试；使用垃圾桶远观察姿态执行单臂抓取、拿起和移出通道", flush=True)
                bin_success = run_task1_bin(station, navigation, detector, wrist_deprojectors, placement_distance_m, approach_origin_node)
                return bin_success
            if target_type == "filebox":
                print(f"稳定识别目标为{nearest['name']}，第{attempt_index}/{MAX_TASK_ATTEMPTS}次尝试；使用垃圾桶远观察姿态执行单臂抓取、拿起和移出通道", flush=True)
                filebox_success = run_task1_filebox(station, navigation, detector, wrist_deprojectors, placement_distance_m, approach_origin_node)
                return filebox_success
            if target_type == "chair":
                chair_success = run_task1_chair(station, navigation, wrist_deprojectors, wrist_models, initial_pose, approach_origin_node, attempt_index)
                return chair_success
            input("机器人已到目标前并朝向目标；确认返回路径安全后按 Enter 恢复 startpoint：")
            return_from_approach(navigation, approach_origin_node)
            return True

        target_ok = run_with_retry_or_skip(task1_name, run_current_target)
        if not target_ok:
            print(f"{task1_name}未完成，已跳过本轮操作；尝试回 startpoint 后继续", flush=True)
            try:
                apply_startpoint_pose(station, initial_pose)
                return_from_approach(navigation, approach_origin_node)
            except Exception as recover_error:
                print(f"失败后回位异常（忽略并继续）：{recover_error}", flush=True)
        print("已恢复 startpoint", flush=True)
        if target_type not in {"filebox", "chair", "bin"}:
            input("确认机器人已在startpoint且姿态恢复路径安全，按 Enter继续")
        type_attempt_counts[target_type] = attempt_index
        if approach_origin_node != "startpoint":
            input("目标来自h0_track_turn_2；按 Enter 返回该观察点执行抓取后复核：")
            navigation.recoverState("h0_track_turn_2", method="move_by")
        verify_targets = recognize_task1_batch(
            station, detector, deprojector, f"机器人已到{approach_origin_node}；按 Enter 复核5帧{nearest['name']}：",)
        still_present = any(
            (target['type'] == target_type for target in filter_targets_by_hits(verify_targets)),)
        if approach_origin_node != "startpoint":
            next_round_from_h0 = True
        if not still_present:
            remaining_types.discard(target_type)
            removed_object_count += 1
            print(f"{nearest['name']}图像复核未稳定检出，视为已清除；剩余类别={sorted(remaining_types)}", flush=True)
        elif attempt_index >= MAX_TASK_ATTEMPTS:
            remaining_types.discard(target_type)
            print(f"{nearest['name']}已尝试{attempt_index}遍仍存在，忽略该物体；剩余类别={sorted(remaining_types)}", flush=True)
        else:
            print(f"{nearest['name']}第{attempt_index}/{MAX_TASK_ATTEMPTS}次尝试后仍存在，保留重试；剩余类别={sorted(remaining_types)}", flush=True)
        round_index += 1
    if not moved_to_task2:
        print("任务1已无剩余未处理类别，进入任务2起点", flush=True)
        move_from_observation_to_task2_start(navigation, last_observation)
    print("任务1循环结束", flush=True)
    return True

# ---------------------------------------------------------------------------
# 任务2 开窗：窗户1 / 窗户2
# ---------------------------------------------------------------------------
def run_task2_window1(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """窗户1：导航作业点、识别把手、对齐、推窗并恢复姿态。"""
    input("任务2步骤3：按 Enter 导航到 h270_window1_action：")
    navigation.moveTo("h270_window1_action", correct=True, align=True)
    input("垂直抬升腰和腿，确认机器人安全，按Enter执行")
    execute_synced_waist_leg_action(
        station, {'waist': {'position': [0.0, 0.02]}, 'leg': {'position': [-0.02, 0.02]}},)
    for index in range(2):
        input(f"任务2步骤{index + 4}：确认机械臂路径安全，按 Enter 执行窗口迫近 step_{index}")
        window_pose = settings.WINDOW1_POSES[index]
        moveit_success = moveit_joints(station, left=window_pose["left_arm"])
        if not moveit_success:
            print(f"窗口迫近 step_{index} 执行失败", flush=True)
            return False
    output_dir = (
        settings.ROOT
        / "output/task2_recognize"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    input("任务2步骤7：step_0～1已完成，按 Enter 读取左腕 RGB-D、保存图像并识别窗把手")
    recognized = recognize_window_handle_with_tcp_z_retry(
        station, output_dir, WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M,)
    if recognized is None:
        print("任务2开窗判定超出阈值", flush=True)
        return False
    result, target_pose_waist, T_waist_tcp, point_left_tcp_m = recognized
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    robot = station.get_robot_handle().get("robot")
    first_delta_tcp_m = np.asarray(
        [point_left_tcp_m[0], point_left_tcp_m[1], 0.0], dtype=np.float64,)
    first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(f'识别点（left_tcp_link）：x={point_left_tcp_m[0] * 100:+.2f} cm，y={point_left_tcp_m[1] * 100:+.2f} cm，z={point_left_tcp_m[2] * 100:+.2f} cm', flush=True)
    print(f'第一次TCP局部偏移：x={first_delta_tcp_m[0] * 100:+.2f} cm，y={first_delta_tcp_m[1] * 100:+.2f} cm，z=+0.00 cm；方向保持不变', flush=True)
    input("任务2步骤8：确认路径安全，按 Enter 只修正TCP局部x、y：")
    moveit_success = moveit_cartesians(
        station, left=first_target_pose_waist, use_cartesian_path=True, left_motion_id=1,)
    if not moveit_success:
        print("左臂第一次x/y修正失败", flush=True)
        return False
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(state['arm_joints']['left']).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    current_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64,)
    second_delta_tcp_m = np.asarray(
        [0.0, 0.0, point_left_tcp_m[2] + 0.2], dtype=np.float64,)
    second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
    second_target_pose_waist = current_pose_waist.copy()
    second_target_pose_waist[:3] = T_waist_tcp[:3, 3] + second_delta_waist_m
    print(f'第二次TCP局部z偏移：识别z={point_left_tcp_m[2] * 100:+.2f} cm + 20.00 cm = {second_delta_tcp_m[2] * 100:+.2f} cm；方向保持不变', flush=True)
    result.update(
        {'point_left_tcp_xyz_m': point_left_tcp_m.tolist(), 'first_delta_left_tcp_xyz_m': first_delta_tcp_m.tolist(), 'second_delta_left_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_z_rule': 'recognized_left_tcp_z_m + 0.20 m'},)
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
    input("任务2步骤9：确认路径安全，按 Enter 只沿TCP局部z移动：")
    moveit_success = moveit_cartesians(
        station, left=second_target_pose_waist, use_cartesian_path=True, left_motion_id=2,)
    if not moveit_success:
        print("左臂第二次z修正失败", flush=True)
        return False
    input("任务2步骤10：TCP的x/y/z对齐已完成，本步骤不闭合夹爪；确认状态后按 Enter 继续：")
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    right_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    right_target_pose_waist[1] -= 0.10
    input("任务2步骤11：确认路径安全，按 Enter 沿waist_yaw_link -Y向视觉右侧移动10 cm")
    moveit_success = moveit_cartesians(
        station, left=right_target_pose_waist, use_cartesian_path=True, left_motion_id=3,)
    if not moveit_success:
        print("左臂向视觉右侧移动10 cm失败", flush=True)
        return False
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    upward_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    upward_target_pose_waist[2] += 0.04
    input("任务2步骤12：确认路径安全，按 Enter 沿waist_yaw_link +Z向上移动4 cm")
    moveit_success = moveit_cartesians(
        station, left=upward_target_pose_waist, use_cartesian_path=True, left_motion_id=4,)
    if not moveit_success:
        print("左臂向上移动4 cm失败", flush=True)
        return False
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    rotate_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    qx, qy, qz, qw = rotate_target_pose_waist[3:]
    half_angle_rad = math.radians(-5.0) / 2.0
    sin_half = math.sin(half_angle_rad)
    cos_half = math.cos(half_angle_rad)
    rotate_target_pose_waist[3:] = np.asarray(
        [qx * cos_half - qz * sin_half, qw * sin_half + qy * cos_half, qx * sin_half + qz * cos_half, qw * cos_half - qy * sin_half], dtype=np.float64,)
    rotate_target_pose_waist[3:] /= np.linalg.norm(rotate_target_pose_waist[3:])
    input("底盘右移动15cm")
    navigation.moveBy(0.0, -0.15, 0.0, correct=False)
    input("任务2推窗：确认左臂原地旋转路径安全，按 Enter 绕TCP局部-Y轴向右旋转5°；位置保持不变：")
    moveit_success = moveit_cartesians(
        station, left=rotate_target_pose_waist, use_cartesian_path=True, left_motion_id=5,)
    if not moveit_success:
        print("左臂TCP向左旋转5°失败", flush=True)
        return False

    push_succeeded = _push_window_with_reduced_offset(
        station,
        robot,
        first_motion_id=7,
        label="向前",
        initial_delta_tcp_m=np.asarray(
            WINDOW_PUSH_INITIAL_DELTA_TCP_M,
            dtype=np.float64,
        ),
    )
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [-0.3, 0.1]}})
    time.sleep(0.7)
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [0.0, 0.0]}})
    input("底盘左移动15cm")
    navigation.moveBy(0.0, 0.15, 0.0, correct=False)
    for index in range(9, 14):
        input(f"任务2恢复：确认机械臂路径安全，按 Enter 执行窗口恢复 step_{index}")
        window_pose = settings.WINDOW1_POSES[index]
        moveit_success = moveit_joints(station, left=window_pose["left_arm"])
        if not moveit_success:
            print(f"窗口恢复 step_{index} 执行失败", flush=True)
            return False
    
    if not push_succeeded:
        print(
            "左臂TCP向前推窗缩距重试仍失败；"
            "窗口恢复姿态已执行，继续现有任务2流程",
            flush=True,
        )
    return True

def run_task2_window2(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """窗户2：过渡导航、准备姿态，再执行现场开窗动作。"""
    input("任务2窗口2步骤1：窗口恢复step_11～12已完成，按 Enter 导航到h270_task2_check")
    navigation.moveTo("h270_task2_check", correct=False)
    input("任务2窗口2步骤2：确认姿态切换路径安全，按 Enter 切换到应急_窗户2_准备姿态")
    prepare_pose = settings.WINDOW2_PREPARE_POSE
    moveit_success = moveit_joints(
        station, left=prepare_pose['left_arm'], right=prepare_pose['right_arm'],)
    if not moveit_success:
        print("双臂切换应急_窗户2_准备姿态失败", flush=True)
        return False
    input("任务2窗口2步骤3：窗户2准备姿态已到达，按 Enter 导航到h270_window2_action")
    navigation.moveTo("h270_window2_action", correct=False)
    window2_success = run_window2_action(station, navigation)
    if window2_success:
        print("任务2窗口视觉操作完成", flush=True)
        return True
    print("任务2窗口2识别失败，已跳过窗口2操作", flush=True)
    return True

def run_task2_windows(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """任务2开窗：依次窗户1、窗户2。"""
    window1_success = run_task2_window1(station, navigation)
    if not window1_success:
        return False
    window2_success = run_task2_window2(station, navigation)
    return window2_success

def run_task2_window(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """兼容旧名：等同 run_task2_windows。"""
    windows_success = run_task2_windows(station, navigation)
    return windows_success

# ---------------------------------------------------------------------------
# 任务3 隔离带：抓取（带1/带2）+ 拉隔离带
# ---------------------------------------------------------------------------
def run_task3_grasp_belt1(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """隔离带1（右侧）识别与抓取。"""
    input('任务3步骤1：确认机器人实际位于h270_task3_check且右侧安全，按 Enter 执行右侧黑色隔离带识别与抓取')
    task3_right = settings.TASK3_RIGHT_PREPARE_POSE
    moveit_success = moveit_joints(
        station, left=task3_right['left_arm'], right=task3_right['right_arm'],)
    if not moveit_success:
        print("双臂到右侧黑色隔离带抓取姿态失败", flush=True)
        return False
    execute_synced_waist_leg_action(
        station, {'waist': {'position': task3_right['waist']}, 'leg': {'position': task3_right['leg']}},)
    time.sleep(3.0)
    print("任务3右侧夹爪预置：left=0.000，right=0.000", flush=True)
    station.execute_action({"hand": {"position": {"left": [0.0], "right": [0.0]}}})
    time.sleep(2.0)
    right_result = grasp_black_single(station, "right", {"left": 0.0, "right": 0.0})
    if right_result is None or not right_result["hand_executed"]:
        print("右侧黑色隔离带抓取失败")
    return True

def run_task3_grasp_belt2(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """隔离带2（左侧）识别与抓取。"""
    input('任务3步骤2：确认右侧目标已抓稳且移动路径安全，按 Enter 移动到左右任务点并执行左侧识别与抓取')
    navigation.moveTo("h270_task3_right", correct=True)
    task3_left = settings.TASK3_LEFT_PREPARE_POSE
    moveit_success = moveit_joints(station, left=task3_left["left_arm"])
    if not moveit_success:
        print("左臂到左侧黑色隔离带预备姿态失败", flush=True)
        return False
    execute_synced_waist_leg_action(
        station, {'waist': {'position': task3_left['waist']}},)
    time.sleep(5.0)
    navigation.moveTo("h270_task3_left", correct=True)
    print("任务3左侧夹爪预置：left=0.000，right=1.000", flush=True)
    station.execute_action({"hand": {"position": {"left": [0.0], "right": [1.0]}}})
    time.sleep(2.0)
    left_result = grasp_black_single(station, "left", {"left": 0.0, "right": 1.0})
    if left_result is None or not left_result["hand_executed"]:
        print("左侧黑色隔离带抓取失败，本任务失败", flush=True)
        return False
    return True

def run_task3_grasp_belts(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """抓取隔离带：依次隔离带1、隔离带2。"""
    belt1_success = run_task3_grasp_belt1(station, navigation)
    if not belt1_success:
        return False
    belt2_success = run_task3_grasp_belt2(station, navigation)
    return belt2_success

def run_task3_pull_belts(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """拉隔离带：携带目标返回放置点、松爪并恢复姿态。"""
    input('任务3步骤3：确认两个目标均已抓稳且返回路线安全，按 Enter 携带目标返回h270_task3_place')
    navigation.moveTo("h270_task3_back", correct=True)
    navigation.moveTo("h270_task3_return", correct=True)
    navigation.moveTo("h270_task3_place", correct=True)
    print("任务3已到达放置点，张开双夹爪", flush=True)
    station.execute_action({"hand": {"position": {"left": [0.0], "right": [0.0]}}})
    time.sleep(2.0)
    recover_pose = settings.TASK3_PLACE_RECOVER_POSE
    print("任务3双臂切换到放置后恢复姿态，然后恢复腰腿", flush=True)
    moveit_success = moveit_joints(
        station, left=recover_pose['left_arm'], right=recover_pose['right_arm'],)
    if not moveit_success:
        print("双臂切换到放置后恢复姿态失败", flush=True)
        return False
    navigation.moveBy(0, 0.3, 0)
    execute_synced_waist_leg_action(
        station, {'waist': {'position': recover_pose['waist']}, 'leg': {'position': recover_pose['leg']}},)
    print("任务3黑色隔离带抓取、放置和恢复完成", flush=True)
    return True

def run_task3_belts(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """任务3隔离带：抓取后拉放。进入前导航到 h270_task3_check。"""
    input('任务2已结束；确认机器人当前位置及路径安全后按 Enter 导航到h270_task3_check')
    try:
        navigation.moveTo("h270_task3_check", correct=True)
    except Exception as nav_error:
        print(f"前往任务3检查点失败：{nav_error}", flush=True)
        stage_success = run_with_retry_or_skip(
            '导航到h270_task3_check', lambda: (navigation.moveTo('h270_task3_check', correct=True), True)[1],)
        if not stage_success:
            print("无法到达任务3检查点，跳过任务3", flush=True)
            return False
    grasp_success = run_task3_grasp_belts(station, navigation)
    if not grasp_success:
        return False
    pull_success = run_task3_pull_belts(station, navigation)
    return pull_success

# ---------------------------------------------------------------------------
# 主流程：初始化后仅调用三个任务
# ---------------------------------------------------------------------------
def main() -> None:
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    detector = ObstacleYolo()
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get('robot', {}), camera_dict=config.get('camera', {}), hand_dict=config.get('hand', {}), force_sensor_dict={}, control_rate_hz=config.get('basic', {}).get('control_rate_hz', 100),)
    navigation_config = settings.TASK["navigation"]
    try:
        time.sleep(1.0)
        deprojector = deprojector_module.Deprojector(
            station, CALIBRATION_PATH, 'head',)
        wrist_deprojectors = {
            side: deprojector_module.Deprojector(station, CALIBRATION_PATH, side)
            for side in ("left", "right")
        }
        wrist_models = {
            side: Phase1Model(
                {'fx': float(camera['camera_matrix'][0][0]), 'fy': float(camera['camera_matrix'][1][1]), 'cx': float(camera['camera_matrix'][0][2]), 'cy': float(camera['camera_matrix'][1][2]), 'width': int(camera['image_width']), 'height': int(camera['image_height']), 'depth_scale_m': WRIST_DEPTH_SCALE_M, 'invalid_depth_values': [0, 65535]},)
            for side, camera in (
                ("left", settings.LEFT_CAMERA),
                ("right", settings.RIGHT_CAMERA),
            )
        }
        with NavigationGraph(
            settings.prepare_route(), initial_node='startpoint', navigation_mode='move_by', linear_speed=float(navigation_config['linear_speed_m_s']), angular_speed=float(navigation_config['angular_speed_rad_s']), compress_path=False,) as navigation:
            def ensure_at_startpoint() -> bool:
                state_ok = navigation.checkState()
                if not state_ok:
                    raise RuntimeError("机器人实测位置不在 startpoint")
                return True
            stage_success = run_with_retry_or_skip(
                '确认位于startpoint', ensure_at_startpoint,)
            if not stage_success:
                print("未确认startpoint，跳过长程测试后续任务", flush=True)
                return
            input('初始化已确认位于startpoint；确认双臂、头、腰和腿路径安全后按 Enter 恢复应急初始姿态：')
            stage_success = run_with_retry_or_skip(
                '恢复应急初始姿态', lambda: apply_startpoint_pose(station, initial_pose),)
            if not stage_success:
                print("初始姿态恢复失败且已跳过，继续尝试后续流程", flush=True)
            avoid = AvoidCollision(
                navigation,
                settings.CONFIG_DIR / "scene.json",
                collision_radius_m=0.2,
            )
            run_with_retry_or_skip(
                '任务1杂物', lambda: run_task1_debris(station, navigation, avoid, detector, deprojector, wrist_deprojectors, wrist_models, initial_pose),)
            run_with_retry_or_skip(
                '任务2开窗', lambda: run_task2_windows(station, navigation),)
            run_with_retry_or_skip(
                '任务3隔离带', lambda: run_task3_belts(station, navigation),)
    finally:
        station.shutdown()

if __name__ == "__main__":
    main()
