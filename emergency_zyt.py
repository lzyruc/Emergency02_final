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
from box_orientation_depth import detect_box_orientation_from_horizontal_span
from camera_io import read_head_camera, read_wrist_camera
from chair_depth_grasp import recognize_chair_grasp
from channel_region import ChannelRegion
from task1 import (
    BIN_LIFT_PATH,
    BIN_PREGRASP_PATHS,
    BOX_HOLD_HORIZONTAL_PATH,
    BOX_HOLD_VERTICAL_30_PATH,
    BOX_HOLD_VERTICAL_PATH,
    BOX_PREGRASP_PATH,
    CHAIR_PREGRASP_PATH_BY_VIEW,
    FILEBOX_LIFT_PATH,
)
from waist_leg_motion import execute_synced_waist_leg_action
from window_handle_recognition import recognize_window_handle
from yolo import ChairViewYolo, ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager

# ---------------------------------------------------------------------------
# 公共参数
# ---------------------------------------------------------------------------
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
INITIAL_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_emergency/应急_初始姿态.json"
)
MAX_STAGE_ATTEMPTS = 3  # 单轮最大重试次数；用尽后由上层继续重试同一任务


# ---------------------------------------------------------------------------
# 任务1：通道障碍物识别、接近、抓取与放置参数
# ---------------------------------------------------------------------------
TARGET_NAMES = {
    "chair": "办公转椅",
    "filebox": "文件筐",
    "bin": "垃圾桶",
    "box": "纸箱（3 kg）",
}
TASK1_IGNORED_TYPES: set[str] = set()
FRAME_COUNT = 5
FRAME_INTERVAL_S = 0.1
TASK1_YOLO_SETTLE_S = 3.0
# 第一段接近的 standoff 是目标到机器人最终中心的平面径向距离。
# plan_standoff_move 会把目标 base_footprint (x, y) 按同一比例缩短：
# move_x=x*(distance-standoff)/distance，move_y同理，因此方向保持不变。
DEFAULT_STANDOFF_M = 0.4
# 纸箱第一段固定使用0.45 m径向停靠，然后进行第二次头部识别微调。
STANDOFF_M_BY_TYPE = {"box": 0.45}
# 椅子先由专用模型区分朝向；第一次径向接近和第二次X保距使用同一参数。
CHAIR_STANDOFF_M_BY_VIEW = {"front": 0.80, "back": 0.60}
# 垃圾桶和文件框分别配置第一次接近的径向停靠距离，均按从近到远尝试；
# 每个候选都使用弯腰矩形扫掠检查，首个安全候选立即采用。
BENT_APPROACH_STANDOFFS_M_BY_TYPE = {
    "bin": (0.50, 0.60, 0.70),
    "filebox": (0.40,0.50, 0.60, 0.70),
}
STARTPOINT_RETRY_OBSERVATION_MOTION = (1.5, -1.5, 45.0)
H0_APPROACH_MAX_ABS_Y_M = 0.25
# 第二段复识别发生在底盘不旋转的前提下：X只前进到目标前保留该距离，
# 即 move_x=max(0, target_x-final_standoff)；Y不设standoff，直接用target_y对中。
FINAL_STANDOFF_M_BY_TYPE = {"box": 0.45}
TASK1_FINE_DEADBAND_M = 0.02
CHAIR_VIEW_FRAME_COUNT = 3
CHAIR_VIEW_MIN_HITS = 2
BOX_ORIENTATION_MIN_HEIGHT_M = 0.15
BOX_HORIZONTAL_SPAN_THRESHOLD_M = 0.40
BOX_ORIENTATION_RETRY_INTERVAL_S = 0.20
BOX_ORIENTATION_MAX_YOLO_FRAMES = 3
# 前两次方向识别失败默认横抱；第三次起改用竖抱，尝试抓纸箱窄边。
BOX_NARROW_FALLBACK_ATTEMPT = 3
TURN_REDUCTION_DEG_BY_TYPE = {"bin": 0.0, "filebox": 0.0}
COLLISION_RADIUS_M = 0.10
BENT_COLLISION_FRONT_M = 0.65
MIN_HITS_EXCLUSIVE = 3
CHANNEL_OUTSIDE_MARGIN_M_BY_TYPE = {
    target_type: float(distance_m)
    for target_type, distance_m in settings.TASK["channel_region"][
        "outside_ignore_distance_m_by_type"
    ].items()
}
CHANNEL_MAX_OBJECT_Z_M = 0.80
STARTPOINT_DEFER_LOGICAL_X_M = 2.60
INITIAL_PLACE_DISTANCE_M = 1.12
PLACE_DISTANCE_DECREMENT_M = 0.35
CHAIR_GRASP_OFFSET_TCP_M = np.asarray([0.0, 0.0, 0.10], dtype=np.float64)
# waist_yaw_link下的腕部识别前预移动(x前、y左)，单位米。
# 正面参数已经实机验证；背面切完姿态后直接识别，不执行预移动。
CHAIR_PREMOVE_WAIST_XY_M_BY_VIEW = {
    "front": {"left": (0.05, 0.05), "right": (0.05, -0.05)},
    "back": {"left": (0.0, 0.0), "right": (0.0, 0.0)},
}


# ---------------------------------------------------------------------------
# 任务2：窗户把手识别、机械臂对齐与底盘开窗参数
# ---------------------------------------------------------------------------
# 每个视角重拍2帧；原位失败后相对首次TCP沿waist_yaw_link Y轴左右搜索3 cm。
WINDOW_HANDLE_VIEW_OFFSETS_Y_M = (0.0, 0.03, -0.03)
WINDOW_HANDLE_FRAMES_PER_VIEW = 2
WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M = 0.50
WINDOW2_HANDLE_MAX_TCP_DELTA_Z_M = 0.40
# 窗2按识别TCP-z接近后保留的前向距离；执行时保持waist_yaw_link Z不变。
WINDOW2_HANDLE_STANDOFF_TCP_Z_M = 0.20
# 窗1首次x/y对齐额外向视觉右侧移动；视觉右侧为waist_yaw_link -Y。
WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M = 0.02
# 推窗最多尝试7次，每次保留上一次90%的总偏移；剩余前向距离不足1 cm时不再补推。
WINDOW_PUSH_MAX_ATTEMPTS = 7
WINDOW_PUSH_RETRY_SCALE = 0.9
WINDOW_PUSH_MIN_RETRY_TRAVEL_M = 0.01
WINDOW_PUSH_INITIAL_DELTA_TCP_M = (0.00, 0.0, -0.30)
# 窗2闭爪后底盘拉窗，以及松爪后与窗户分离的平移。
# base_footprint：+x前、+y左，单位米；两段移动均不旋转。
WINDOW2_OPEN_BASE_MOVE_XY_M = (-0.32, -0.15)
WINDOW2_RELEASE_BASE_MOVE_XY_M = (0.0, -0.10)
WINDOW2_BASE_MOVE_LINEAR_SPEED_M_S = 0.10


# ---------------------------------------------------------------------------
# 任务3：黑色隔离带抓取与重抓参数
# ---------------------------------------------------------------------------
# 单侧MoveIt失败后的底盘二次抓取平移。
# base_footprint：+x前、+y左，单位米。
TASK3_RIGHT_REGRASP_MOVE_XY_M = (0.10, 0.0)
TASK3_LEFT_REGRASP_MOVE_XY_M = (0.05, 0.05)


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """将TCP的XYZW四元数转换为TCP到waist_yaw_link的旋转矩阵。"""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

def run_with_retry_or_skip(
    task_name: str,
    action: Any,
    *,
    max_attempts: int = MAX_STAGE_ATTEMPTS,
    automatic: bool = False,
) -> bool:
    """执行子任务并在失败后重试。不允许跳过进入后续任务。

    action 返回 False 视为失败；返回 None/其他非 False 视为成功。
    成功返回 True；本轮次数用尽返回 False，由调用方继续重试同一任务。
    任务内步骤（如窗户1→窗户2）仍允许跳过。
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
                if not automatic:
                    input(
                        f"{task_name}本轮重试已用尽，按 Enter 继续重试本任务"
                        "（不允许进入后续任务）："
                    )
                print(f"{task_name}未完成，继续重试", flush=True)
                return False
            if automatic:
                print(f"{task_name}将自动重试", flush=True)
                continue
            input(f'{task_name}：按 Enter 再次尝试（不允许跳过）：')
    return False


def run_until_success(
    task_name: str,
    action: Any,
    *,
    automatic: bool = False,
) -> None:
    """重复执行直到成功；完成前不进入后续任务。"""
    while not run_with_retry_or_skip(
        task_name, action, automatic=automatic,
    ):
        print(f"{task_name}未完成，继续重试，不进入后续任务", flush=True)

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
    time.sleep(4.0)
    input('[start]腰腿回正后按 Enter 继续')
    execute_synced_waist_leg_action(station, {'waist': {'position': pose['waist']}, 'leg': {'position': pose['leg']}})
    # 腰腿为下降，这里不可以sleep
    # 再恢复双臂到应急初始姿态
    moveit_success = moveit_joints(station, left=pose['left_arm'], right=pose['right_arm'])
    if not moveit_success:
        print("双臂恢复应急初始姿态失败", flush=True)
        return False
    return True

# 纸箱二次微调后：持续识别并判断横竖方向
def recognize_box_orientation_until_ready(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    fallback_orientation: str,
) -> dict[str, Any]:
    """最多采集3帧头部RGB-D；方向不可判定时使用本轮回退方向。"""
    print(
        "纸箱二次微调已结束；开始循环采集头部RGB-D，"
        f"最多{BOX_ORIENTATION_MAX_YOLO_FRAMES}帧",
        flush=True,
    )
    for frame_index in range(1, BOX_ORIENTATION_MAX_YOLO_FRAMES + 1):
        color, depth_mm = read_head_camera(station)
        if color is None or depth_mm is None:
            raise RuntimeError("未获取到头部RGB-D帧")
        boxes = detector.detect(color, target_type="box")
        if not boxes:
            print(
                f"纸箱方向识别：第{frame_index}/"
                f"{BOX_ORIENTATION_MAX_YOLO_FRAMES}帧YOLO未检出纸箱",
                flush=True,
            )
            if frame_index < BOX_ORIENTATION_MAX_YOLO_FRAMES:
                time.sleep(BOX_ORIENTATION_RETRY_INTERVAL_S)
            continue
        detection = max(boxes, key=lambda item: item["confidence"])
        result = detect_box_orientation_from_horizontal_span(
            depth_mm,
            detection["bbox_xyxy"],
            deprojector,
            min_height_m=BOX_ORIENTATION_MIN_HEIGHT_M,
            horizontal_span_threshold_m=BOX_HORIZONTAL_SPAN_THRESHOLD_M,
        )
        if result["valid_row_count"] == 0:
            print(
                f"纸箱方向识别：第{frame_index}/"
                f"{BOX_ORIENTATION_MAX_YOLO_FRAMES}帧没有有效深度跨度，"
                "继续采集",
                flush=True,
            )
            if frame_index < BOX_ORIENTATION_MAX_YOLO_FRAMES:
                time.sleep(BOX_ORIENTATION_RETRY_INTERVAL_S)
            continue
        print(
            f"纸箱方向识别：confidence={detection['confidence']:.3f}，"
            f"orientation={result['orientation']}，"
            f"span={result['span_m'] if result['span_m'] is not None else '--'} m，"
            f"有效扫描行={result['valid_row_count']}，"
            f"reason={result['reason']}",
            flush=True,
        )
        return result
    print(
        f"纸箱方向识别：连续{BOX_ORIENTATION_MAX_YOLO_FRAMES}帧"
        "均未获得有效方向结果，默认使用"
        f"{'竖箱窄边' if fallback_orientation == 'vertical' else '横箱宽边'}姿态",
        flush=True,
    )
    return {
        "orientation": fallback_orientation,
        "reason": f"box_orientation_unavailable_default_{fallback_orientation}",
        "span_m": None,
        "row_spans_m": [],
        "rows": [],
        "valid_row_count": 0,
    }


# 纸箱：抱持、拖出通道、释放并回位
def run_box_after_approach(
    station: Any,
    navigation: NavigationGraph,
    orientation_result: dict[str, Any],
    initial_pose: dict[str, Any],
    placement_distance_m: float,
    approach_origin_node: str,
    vertical_hold_path: Path = BOX_HOLD_VERTICAL_PATH,
) -> bool:
    """机器人已在纸箱前时抱持、拖行、释放，恢复初始姿态并回位。失败返回 False。"""
    # 读取纸箱预备与横向抱持姿态
    pregrasp_pose = json.loads(
        BOX_PREGRASP_PATH.read_text(encoding='utf-8'),)
    hold_pose = json.loads(
        BOX_HOLD_HORIZONTAL_PATH.read_text(encoding='utf-8'),)
    vertical_hold_pose = json.loads(
        vertical_hold_path.read_text(encoding="utf-8"),
    )
    input('纸箱步骤1：确认安全后按 Enter 张开夹爪并切换纸箱预备姿态：')
    station.execute_action({'hand': {'position': {'left': [settings.TASK['hands']['left_open_position']], 'right': [settings.TASK['hands']['right_open_position']]}}})
    execute_synced_waist_leg_action(station, {'waist': {'position': pregrasp_pose['waist']}, 'leg': {'position': pregrasp_pose['leg']}},min_spd=0.05,max_spd=0.25)

    """串行执行双臂、Body腰腿和头部预设，不改变夹爪状态。"""
    moveit_success = moveit_joints(station, left=pregrasp_pose['left_arm'], right=pregrasp_pose['right_arm'])
    if not moveit_success:
        return False

    input('纸箱步骤2：确认双臂与纸箱位置安全，按 Enter 切换横纸箱抱持姿态：')
    pose_success = _execute_object_pose(station, hold_pose)
    if not pose_success:
        print("纸箱横向抱持姿态执行失败", flush=True)
        return False
    if orientation_result["orientation"] == "vertical":
        input(
            f"纸箱判断为竖箱；确认双臂路径安全后按 Enter "
            f"切换{vertical_hold_path.name}："
        )
        pose_success = _execute_object_pose(station, vertical_hold_pose)
        if not pose_success:
            print("纸箱竖向抱持姿态执行失败", flush=True)
            return False
    else:
        print("纸箱判断为横箱，保持当前横纸箱抱持姿态", flush=True)
    # navigation.moveBy(0,0,theta=-90, correct=False)
    input('纸箱步骤4：确认抱箱回退路径安全，按 Enter 先恢复观察点和startpoint，再恢复box_point：')
    return_from_approach(navigation, approach_origin_node)
    navigation.recoverState("box_point", method="move_by")
    input("纸箱步骤5：按 Enter 执行 turnTo(90)")
    navigation.moveBy(0.0,0.0,90.0,correct=True)
    input(f'纸箱步骤6：确认前方{placement_distance_m:.1f} m安全，按 Enter 向前拖动{placement_distance_m:.1f} m')
    navigation.moveBy(placement_distance_m, 0.0, 0.0)
    input('纸箱步骤7：确认纸箱已在通道外，按 Enter 切换预备姿态释放纸箱：')
    pose_success = _execute_object_pose(station, pregrasp_pose)
    if not pose_success:
        print("纸箱释放姿态执行失败", flush=True)
        return False
    input('纸箱步骤8：确认纸箱已脱离双臂且机械臂路径安全，按 Enter 执行完整应急初始姿态恢复')
    pose_success = apply_startpoint_pose(station, initial_pose)
    if not pose_success:
        return False
    input('纸箱步骤9：应急初始姿态已到达，按 Enter 恢复startpoint：')
    navigation.recoverState("startpoint", method="move_by")
    input("纸箱步骤10：按 Enter 执行 turnTo(0)并结束")
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
        fused_target = {
            'type': target_type,
            'name': TARGET_NAMES[target_type],
            'hits': len(samples),
            'confidence': float(np.mean([sample['confidence'] for sample in samples])),
            'depth_m': float(np.mean([sample['depth_m'] for sample in samples])),
            'point_base_m': point_base,
            'distance_xy_m': math.hypot(float(point_base[0]), float(point_base[1])),
        }
        # 只用于在新拍全图的椅子朝向结果中关联同一目标，不用于裁剪推理图像。
        if all("bbox_xyxy" in sample for sample in samples):
            fused_target["bbox_xyxy"] = np.median(
                np.asarray([sample["bbox_xyxy"] for sample in samples]),
                axis=0,
            ).tolist()
        fused.append(fused_target)
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
) -> tuple[
    dict[str, Any],
    tuple[float, float, float],
    list[tuple[float, float, float]],
] | None:
    """按平面距离选择第一条无固定障碍物碰撞的目标接近动作。"""
    # 由近到远尝试；可截断横向位移
    for target in sorted(targets, key=lambda item: item["distance_xy_m"]):
        if target.get("type") == "chair":
            standoff_m = CHAIR_STANDOFF_M_BY_VIEW[target["chair_view"]]
        else:
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
        target_type = target.get("type")
        if target_type in BENT_APPROACH_STANDOFFS_M_BY_TYPE:
            # 保留x/y沿目标方向同比例缩短，且抓取前不旋转。
            # 当前类别的近距离候选碰撞时，依次增加停靠距离。
            standoff_candidates = BENT_APPROACH_STANDOFFS_M_BY_TYPE[target_type]
            for candidate_standoff_m in standoff_candidates:
                candidate_motion = plan_standoff_move(
                    target,
                    candidate_standoff_m,
                )
                candidate_motion = (
                    candidate_motion[0],
                    candidate_motion[1],
                    0.0,
                )
                if (
                    max_abs_y_m is not None
                    and abs(candidate_motion[1]) > max_abs_y_m
                ):
                    candidate_motion = (
                        candidate_motion[0],
                        math.copysign(max_abs_y_m, candidate_motion[1]),
                        0.0,
                    )
                approach_motions = avoid.planApproach(*candidate_motion)
                if approach_motions is not None:
                    selected_target = dict(target)
                    selected_target["approach_standoff_m"] = (
                        candidate_standoff_m
                    )
                    return selected_target, candidate_motion, approach_motions
                print(
                    f"{target['name']}{candidate_standoff_m:.2f} m停靠不可行，"
                    "继续增加停靠距离",
                    flush=True,
                )
            print(
                f"跳过{target['name']}：{standoff_candidates[0]:.2f}～"
                f"{standoff_candidates[-1]:.2f} m的接近终点或"
                "弯腰矩形扫掠路径均与固定障碍物碰撞",
                flush=True,
            )
            continue
        if avoid.checkCollision(*motion):
            print(f"跳过{target['name']}：{standoff_m:.2f} m接近直线路径与固定障碍物碰撞", flush=True)
            continue
        selected_target = dict(target)
        selected_target["approach_standoff_m"] = standoff_m
        return selected_target, motion, [(motion[0], motion[1], 0.0)]
    return None


def classify_chairs_in_channel(
    station: Any,
    detector: ChairViewYolo,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """重拍三张完整头部图像，以多数票区分通道内椅子的朝向。"""
    classified_targets = []
    for target in targets:
        if target["type"] != "chair":
            classified_targets.append(target)
            continue
        bbox = target.get("bbox_xyxy")
        if bbox is None:
            print("跳过办公转椅：五帧融合结果没有检测框，无法关联全图结果", flush=True)
            continue
        target_center_u = 0.5 * (float(bbox[0]) + float(bbox[2]))
        target_center_v = 0.5 * (float(bbox[1]) + float(bbox[3]))
        views: dict[str, list[float]] = defaultdict(list)
        for frame_index in range(1, CHAIR_VIEW_FRAME_COUNT + 1):
            color, depth_mm = read_head_camera(station)
            if color is None or depth_mm is None:
                raise RuntimeError("椅子朝向二次分类未获取到头部RGB-D帧")
            detections = detector.detect(color)
            if detections:
                best = min(
                    detections,
                    key=lambda item: (
                        math.hypot(
                            0.5 * (item["bbox_xyxy"][0] + item["bbox_xyxy"][2])
                            - target_center_u,
                            0.5 * (item["bbox_xyxy"][1] + item["bbox_xyxy"][3])
                            - target_center_v,
                        ),
                        -item["confidence"],
                    ),
                )
                views[best["view"]].append(float(best["confidence"]))
                print(
                    f"椅子朝向第{frame_index}/{CHAIR_VIEW_FRAME_COUNT}帧："
                    f"{best['view']}，confidence={best['confidence']:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"椅子朝向第{frame_index}/{CHAIR_VIEW_FRAME_COUNT}帧：未检出",
                    flush=True,
                )
            if frame_index < CHAIR_VIEW_FRAME_COUNT:
                time.sleep(FRAME_INTERVAL_S)
        if not views:
            print("跳过办公转椅：朝向模型三帧均未检出", flush=True)
            continue
        view, confidences = max(
            views.items(),
            key=lambda item: (len(item[1]), float(np.mean(item[1]))),
        )
        if len(confidences) < CHAIR_VIEW_MIN_HITS:
            print(
                f"跳过办公转椅：朝向分类不稳定，结果="
                f"{ {name: len(scores) for name, scores in views.items()} }",
                flush=True,
            )
            continue
        mean_confidence = float(np.mean(confidences))
        if view in {"lean", "back"}:
            print(
                f"忽略{view}椅子：命中={len(confidences)}/"
                f"{CHAIR_VIEW_FRAME_COUNT}，平均置信度={mean_confidence:.3f}",
                flush=True,
            )
            continue
        classified_target = dict(target)
        classified_target["chair_view"] = view
        classified_target["chair_view_confidence"] = mean_confidence
        classified_targets.append(classified_target)
        print(
            f"保留{view}椅子：命中={len(confidences)}/{CHAIR_VIEW_FRAME_COUNT}，"
            f"平均置信度={mean_confidence:.3f}，两段停靠距离="
            f"{CHAIR_STANDOFF_M_BY_VIEW[view]:.2f} m",
            flush=True,
        )
    return classified_targets

# 任务1：startpoint / h0 观察并选路
def find_task1_approach(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    channel_region: dict[str, ChannelRegion],
    chair_view_detector: ChairViewYolo,
    remaining_types: set[str],
    round_index: int,
    *,
    start_from_h0: bool = False,
    startpoint_is_offset: bool = False,
    blocked_targets_from_start: list[dict[str, Any]] | None = None,
) -> tuple[
    str,
    list[dict[str, Any]],
    tuple[
        dict[str, Any],
        tuple[float, float, float],
        list[tuple[float, float, float]],
    ] | None,
]:
    """先在startpoint识别和选路；无稳定目标或接近路线碰撞时再去h0复识别。h0仍无目标则结束选路，不回startpoint。"""
    if not remaining_types:
        origin = "h0_track_turn_2" if start_from_h0 else "startpoint"
        print("没有剩余未处理类别，跳过观察点识别和移动", flush=True)
        return origin, [], None
    reliable_targets: list[dict[str, Any]] = []
    blocked_targets = list(blocked_targets_from_start or ())
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
                input("startpoint无稳定目标；确认整条底盘清场路线安全，按 Enter 开始两轮清场：")
                input("清场步骤1：确认右侧0.40 m安全，按 Enter 向右移动0.40 m：")
                navigation.moveBy(0.0, -0.40, 0.0, correct=True)
                input("清场步骤2：确认前方2.30 m及左侧0.40 m安全，按 Enter 执行斜向清场：")
                navigation.moveBy(2.10, 0.40, 0.0, correct=False)
                input("清场步骤3：确认返回路径安全，按 Enter 恢复startpoint：")
                navigation.recoverState("startpoint", method="move_by")
                input("清场步骤4：确认左侧0.25 m安全，按 Enter 向左移动0.25 m：")
                navigation.moveBy(0.0, 0.25, 0.0, correct=True)
                input("清场步骤5：确认前方2.30 m安全，按 Enter 向前清场：")
                navigation.moveBy(2.10, 0.0, 0.0, correct=False)
                input(
                    "清场步骤6：确认先恢复h270_track_turn_1、"
                    "再恢复h0_track_turn_2的路径安全，"
                    "按 Enter 继续："
                )
                navigation.recoverState(
                    "h270_track_turn_1",
                    method="move_by",
                )
                navigation.recoverState(
                    "h0_track_turn_2",
                    method="move_by",
                )
                print(
                    "两次底盘清场轨迹执行完成，"
                    "已经h270_track_turn_1恢复h0_track_turn_2",
                    flush=True,
                )
        print(
            f"第 {round_index} 轮：机器人位于{observation_node}，"
            "准备连续识别5帧",
            flush=True,
        )
        input("确认机器人当前观察位置和腰腿姿态已稳定，按 Enter 连续识别5帧：")
        fused_targets = recognize_task1_batch(
            station, detector, deprojector,)
        for target in fused_targets:
            if target["type"] in TASK1_IGNORED_TYPES:
                print(
                    f"忽略{target['name']}：当前任务1暂不处理该类别",
                    flush=True,
                )
        channel_targets = filter_targets_in_channel(
            navigation,
            channel_region,
            [
                target for target in fused_targets
                if target["type"] not in TASK1_IGNORED_TYPES
            ],
        )
        if observation_node == "startpoint":
            startpoint_targets = []
            for target in channel_targets:
                logical_x_m = float(target["point_logical_m"][0])
                if logical_x_m > STARTPOINT_DEFER_LOGICAL_X_M:
                    print(
                        f"延后{target['name']}：logical x={logical_x_m:.3f} m "
                        f"> {STARTPOINT_DEFER_LOGICAL_X_M:.2f} m，"
                        "交给h0_track_turn_2重新识别和处理",
                        flush=True,
                    )
                    continue
                startpoint_targets.append(target)
            channel_targets = startpoint_targets
        unprocessed_targets = [
            target for target in channel_targets if target["type"] in remaining_types]
        reliable_targets = filter_targets_by_hits(unprocessed_targets)
        reliable_targets = classify_chairs_in_channel(
            station,
            chair_view_detector,
            reliable_targets,
        )
        for target in fused_targets:
            if int(target["hits"]) <= MIN_HITS_EXCLUSIVE:
                print(f"过滤{target['name']}：命中={target['hits']}/{FRAME_COUNT}，要求命中>=4/5", flush=True)
        if not reliable_targets:
            print(f"{observation_node}没有稳定命中的未处理目标", flush=True)
            if observation_node == "h0_track_turn_2":
                if blocked_targets:
                    print(
                        "h0无稳定目标，但startpoint仍有无法安全接近的障碍物，"
                        "不得判定清场",
                        flush=True,
                    )
                    return observation_node, blocked_targets, None
                print("h0无稳定未处理目标，不再返回startpoint", flush=True)
                return observation_node, [], None
            if startpoint_is_offset:
                print("换观察位无稳定目标，先恢复startpoint再检查h0", flush=True)
                input("确认备用观察位返回startpoint的路径安全，按 Enter 继续：")
                navigation.recoverState("startpoint", method="move_by")
            continue
        print(f"当前观察点={observation_node}，开始检查稳定目标的接近路线", flush=True)
        approach = select_safe_approach(
            avoid,
            reliable_targets,
            max_abs_y_m=(
                H0_APPROACH_MAX_ABS_Y_M
                if observation_node == "h0_track_turn_2"
                else None
            ),
        )
        if approach is not None:
            return observation_node, reliable_targets, approach
        blocked_targets = reliable_targets
        if observation_node == "startpoint":
            if startpoint_is_offset:
                print(
                    "换观察位仍无安全接近路线，恢复startpoint后改从h0复识别",
                    flush=True,
                )
                input("确认备用观察位返回startpoint的路径安全，按 Enter 继续：")
                navigation.recoverState("startpoint", method="move_by")
                continue
            print(
                "机器人仍位于startpoint；直接执行"
                "moveBy(x=1.5 m, y=-1.5 m, theta=45°)后重新识别",
                flush=True,
            )
            input(
                "确认从startpoint到备用观察位的路径安全，"
                "按 Enter 执行moveBy(1.5, -1.5, 45°)："
            )
            navigation.moveBy(
                *STARTPOINT_RETRY_OBSERVATION_MOTION,
                correct=False,
            )
            return find_task1_approach(
                station,
                navigation,
                avoid,
                detector,
                deprojector,
                channel_region,
                chair_view_detector,
                remaining_types,
                round_index,
                startpoint_is_offset=True,
                blocked_targets_from_start=reliable_targets,
            )
    return last_node, blocked_targets or reliable_targets, None

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
) -> bool:
    """从任务1当前观察点沿固定路线进入任务2起点。"""
    if observation_node == "startpoint":
        navigation.recoverState(
            'h0_track_turn_2', method='move_by',)
    #侧向那边，多撞点东西
    navigation.moveBy(0.0, - 0.15, 0.0, correct=False)
    navigation.moveTo("h0_track_end", correct=False)
    #这里可能撞到转椅、垃圾箱等，后退避免move_to失败
    #navigation.moveBy(-0.15,0.0,0.0,correct=False)
    #navigation.turnTo(270.0)
    #navigation.moveTo('h270_task2_start', correct=True, align=False)
    # 到任务1判定点1：使用底盘move_by按实测位姿恢复到任务2起点。
    navigation.recoverState("h270_task2_start", method="move_by")
    return True
# 纸箱/转椅近距离复识别并平移微调
def refine_task1_approach(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
    target_type: str,
    chair_view: str | None = None,
    fallback_point_base_m: tuple[float, float] | None = None,
) -> bool:
    """近距离复识别纸箱或转椅，并平移到该类别最终X停靠距离。

    ``fallback_point_base_m`` 是根据第一次识别坐标减去第一段实际平移后，
    估计出的当前base_footprint坐标；仅供纸箱复识别不稳定时继续第二段移动。
    """
    if target_type == "chair":
        if chair_view not in CHAIR_STANDOFF_M_BY_VIEW:
            raise ValueError(f"椅子二段接近缺少有效朝向：{chair_view}")
        final_standoff_m = CHAIR_STANDOFF_M_BY_VIEW[chair_view]
    elif target_type in FINAL_STANDOFF_M_BY_TYPE:
        final_standoff_m = FINAL_STANDOFF_M_BY_TYPE[target_type]
    else:
        raise ValueError(f"不支持二段接近的目标类型：{target_type}")
    target_name = TARGET_NAMES[target_type]
    input(
        f'{target_name}第一次接近完成；按 Enter 再连续识别5帧并计算平移微调：'
    )
    # 近距离再采 5 帧
    frames = []
    for frame_index in range(1, FRAME_COUNT + 1):
        try:
            targets, _ = recognize_frame(
                station, detector, deprojector, frame_index,)
        except RuntimeError as error:
            print(
                f"{target_name}近距离第{frame_index}帧识别失败：{error}",
                flush=True,
            )
            return False
        frames.append(targets)
        if frame_index < FRAME_COUNT:
            time.sleep(FRAME_INTERVAL_S)
    matching_targets = [
        target
        for target in filter_targets_by_hits(fuse_targets(frames))
        if target["type"] == target_type
    ]
    if not matching_targets:
        if target_type != "box" or fallback_point_base_m is None:
            print(f"{target_name}近距离5帧复识别不稳定", flush=True)
            return False
        target_x_m, target_y_m = fallback_point_base_m
        print(
            f"纸箱近距离5帧复识别不稳定，使用第一次识别和第一段实际平移"
            f"估计当前base x={target_x_m:.3f} m、y={target_y_m:.3f} m",
            flush=True,
        )
    else:
        target = min(matching_targets, key=lambda item: item["distance_xy_m"])
        target_x_m, target_y_m = (
            float(value) for value in target["point_base_m"][:2]
        )
    # 只前进到最终停靠，不后退
    fine_x_m = max(0.0, target_x_m - final_standoff_m)
    fine_y_m = target_y_m
    fine_distance_m = math.hypot(fine_x_m, fine_y_m)
    if target_x_m < final_standoff_m:
        print(
            f'{target_name}已近于最终base x基准，第二次微调不后退，只修正y',
            flush=True,
        )
    if matching_targets:
        print(
            f"{target_name}近距离融合结果：命中={target['hits']}/{FRAME_COUNT}，"
            f"base x={target_x_m:.3f} m，y={target_y_m:.3f} m；"
            f"最终X保距={final_standoff_m:.2f} m，计划微调"
            f"x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m、theta=0.0°",
            flush=True,
        )
    else:
        print(
            f"纸箱第一次识别回退结果：最终X保距={final_standoff_m:.2f} m，"
            f"计划第二段x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m、"
            "theta=0.0°",
            flush=True,
        )
    if fine_distance_m <= TASK1_FINE_DEADBAND_M:
        print(
            f"{target_name}平移误差不超过2 cm，跳过第二次moveBy",
            flush=True,
        )
        return True
    has_collision = avoid.checkCollision(fine_x_m, fine_y_m, 0.0)
    if has_collision:
        print(
            f"{target_name}第二次平移路径与固定障碍物碰撞，不执行移动",
            flush=True,
        )
        return False
    input(
        f'确认{target_name}微调路径安全，按 Enter 执行moveBy：'
        f'x={fine_x_m:+.3f} m、y={fine_y_m:+.3f} m，不旋转：'
    )
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
        print(
            f"{target_name}未成功拿起；"
            "交由任务1外层统一恢复本轮观察点并重新识别",
            flush=True,
        )
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

# 任务1目标从base_footprint转换到Navigation logical并过滤通道外目标
def filter_targets_in_channel(
    navigation: NavigationGraph,
    channel_region: dict[str, ChannelRegion],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """保留通道内、边界上或外侧缓冲带内且高度不超过阈值的目标。"""
    logical_pose, _ = navigation.getPose()
    if logical_pose is None:
        raise RuntimeError("任务1通道判断需要机器人位于NavigationGraph命名节点")
    theta_rad = math.radians(float(logical_pose["theta_deg"]))
    cosine, sine = math.cos(theta_rad), math.sin(theta_rad)
    # T_logical_base：base_footprint中的目标点左乘后转换到Navigation logical。
    T_logical_base = np.asarray(
        [
            [cosine, -sine, 0.0, float(logical_pose["x_m"])],
            [sine, cosine, 0.0, float(logical_pose["y_m"])],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    accepted = []
    for target in targets:
        point_base = np.asarray(target["point_base_m"], dtype=np.float64)
        point_logical = T_logical_base @ np.asarray(
            [point_base[0], point_base[1], point_base[2], 1.0],
            dtype=np.float64,
        )
        target["point_logical_m"] = point_logical[:3]
        logical_text = (
            f"logical=({point_logical[0]:.3f}, "
            f"{point_logical[1]:.3f}, {point_logical[2]:.3f}) m"
        )
        if point_logical[2] > CHANNEL_MAX_OBJECT_Z_M:
            print(
                f"忽略{target['name']}：{logical_text}，"
                f"高度超过{CHANNEL_MAX_OBJECT_Z_M:.2f} m",
                flush=True,
            )
            continue
        outside_margin_m = CHANNEL_OUTSIDE_MARGIN_M_BY_TYPE[target["type"]]
        if not channel_region[target["type"]].requires_clearing(
            float(point_logical[0]), float(point_logical[1])
        ):
            print(
                f"忽略{target['name']}：{logical_text}，已在通道及"
                f"{outside_margin_m:.2f} m外侧缓冲区之外",
                flush=True,
            )
            continue
        print(
            f"保留{target['name']}：{logical_text}，位于通道内、边界上或"
            f"{outside_margin_m:.2f} m外侧缓冲区内",
            flush=True,
        )
        accepted.append(target)
    return accepted

# 连续识别五帧并融合
def recognize_task1_batch(
    station: Any,
    detector: ObstacleYolo,
    deprojector: deprojector_module.Deprojector,
) -> list[dict[str, Any]]:
    """自动连续识别五帧，打印并返回融合后的全部任务1目标。"""
    time.sleep(TASK1_YOLO_SETTLE_S)
    frames = []
    for frame_index in range(1, FRAME_COUNT + 1):
        targets, _ = recognize_frame(
            station, detector, deprojector, frame_index,)
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
        #待测试
        #navigation.moveBy(0.04,0.0,0.0,correct=True)
        recognized = recognize_window_handle_with_tcp_z_retry(
            station, output_dir, WINDOW2_HANDLE_MAX_TCP_DELTA_Z_M,)
    except RuntimeError as error:
        print(f'窗口2把手识别失败，跳过后续对齐和扭窗步骤：{error}', flush=True)
        return False
    if recognized is None:
        print("窗口2开窗判定超出阈值，跳过后续操作并进入下一项目", flush=True)
        return False
    result, target_pose_waist, T_waist_tcp, point_left_tcp_m = recognized
    recognized_tcp_z_m = float(point_left_tcp_m[2])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    robot = station.get_robot_handle().get("robot")
    # 先修TCP x/y，再按识别TCP-z计算水平接近量。
    first_delta_tcp_m = np.asarray(
        [point_left_tcp_m[0], point_left_tcp_m[1], 0.0], dtype=np.float64,)
    first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(f'窗口2识别点（left_tcp_link）：x={point_left_tcp_m[0] * 100:+.2f} cm，y={point_left_tcp_m[1] * 100:+.2f} cm，z={recognized_tcp_z_m * 100:+.2f} cm', flush=True)
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
        [
            0.0,
            0.0,
            recognized_tcp_z_m + WINDOW2_HANDLE_STANDOFF_TCP_Z_M,
        ],
        dtype=np.float64,
    )
    second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
    second_delta_waist_m[2] = 0.0
    second_target_pose_waist = current_pose_waist.copy()
    second_target_pose_waist[:3] = T_waist_tcp[:3, 3] + second_delta_waist_m
    print(f'窗口2第二段：识别TCP-z={recognized_tcp_z_m * 100:+.2f} cm + 保留距离{WINDOW2_HANDLE_STANDOFF_TCP_Z_M * 100:.2f} cm = TCP-z计算量{second_delta_tcp_m[2] * 100:+.2f} cm；最终waist位移x={second_delta_waist_m[0] * 100:+.2f} cm，y={second_delta_waist_m[1] * 100:+.2f} cm，z=+0.00 cm', flush=True)
    result.update(
        {'point_left_tcp_xyz_m': point_left_tcp_m.tolist(), 'recognized_left_tcp_z_raw_m': recognized_tcp_z_m, 'first_delta_left_tcp_xyz_m': first_delta_tcp_m.tolist(), 'second_delta_left_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_delta_waist_xyz_m': second_delta_waist_m.tolist(), 'second_z_rule': 'recognized_left_tcp_z_m + 0.19 m, then keep waist_yaw_link z unchanged'},)
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
    input('任务2窗口2操作步骤3：确认路径安全，按 Enter 执行水平接近（waist_yaw_link Z保持不变）：')
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
        (9, 15, "沿waist_yaw_link +Y向视觉左侧移动5 cm", [0.0, 0.08, 0.0]),
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
    input(
        '任务2窗口2操作步骤11：确认底盘、窗户和机械臂路径安全，'
        f'按 Enter 执行底盘开窗：dx={WINDOW2_OPEN_BASE_MOVE_XY_M[0] * 100:+.0f} cm，'
        f'dy={WINDOW2_OPEN_BASE_MOVE_XY_M[1] * 100:+.0f} cm'
    )
    # 底盘拉开窗户
    navigation.moveBy(
        *WINDOW2_OPEN_BASE_MOVE_XY_M,
        0.0,
        linear_speed=WINDOW2_BASE_MOVE_LINEAR_SPEED_M_S,
        correct=True,
    )
    input('任务2窗口2操作步骤12：检查开窗结果，按 Enter 松开左夹爪：')
    station.execute_action({'hand': {'position': {'left': [0.0], 'right': [0.0]}}})
    time.sleep(0.5)
    # navigation.moveBy(
    #     *WINDOW2_RELEASE_BASE_MOVE_XY_M,
    #     0.0,
    #     linear_speed=WINDOW2_BASE_MOVE_LINEAR_SPEED_M_S,
    #     correct=False,
    # )
    # input('任务2窗口2操作步骤13：保持左夹爪张开，确认机械臂路径安全，按 Enter 恢复双臂到应急初始姿态（腰腿不变）')
    # initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    # moveit_success = moveit_joints(
    #     station, left=initial_pose['left_arm'], right=initial_pose['right_arm'],)
    if not moveit_success:
        print("窗口2操作结束后双臂恢复应急初始姿态失败", flush=True)
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
    initial_pose: dict[str, Any],
    placement_distance_m: float,
    attempt_index: int,
    chair_view: str,
    approach_origin_node: str,
) -> bool:
    """任务1-椅子：按正背面姿态双腕识别、两段TCP对齐并拖出通道。"""
    prepare_path = CHAIR_PREGRASP_PATH_BY_VIEW[chair_view]
    prepare_pose = json.loads(prepare_path.read_text(encoding="utf-8"))
    print(
        f"稳定识别目标为{TARGET_NAMES['chair']}，"
        f"第{attempt_index}次尝试；"
        f"执行{chair_view}姿态双腕抓取和拖行",
        flush=True,
    )
    input(
        f"转椅步骤1：确认路径安全，按 Enter 张开双夹爪并切换到"
        f"{prepare_path.name}："
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
    if not apply_startpoint_pose(station, prepare_pose):
        print(f"转椅{chair_view}抓取姿态执行失败", flush=True)
        return False

    available_arms = []
    premoves = CHAIR_PREMOVE_WAIST_XY_M_BY_VIEW[chair_view]
    for arm in ("left", "right"):
        delta_x_m, delta_y_m = premoves[arm]
        if delta_x_m == 0.0 and delta_y_m == 0.0:
            print(
                f"转椅{chair_view}姿态：{arm}臂不执行腕部识别前预移动",
                flush=True,
            )
            available_arms.append(arm)
            continue
        current_pose = station.get_robot_state()["arm_pose"][arm]
        pre_target_pose = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        pre_target_pose[0] += delta_x_m
        pre_target_pose[1] += delta_y_m
        input(
            f"转椅步骤2：确认{arm}臂路径安全，按 Enter 在waist_yaw_link下"
            f"预移动x={delta_x_m * 100:+.0f} cm、"
            f"y={delta_y_m * 100:+.0f} cm："
        )
        succeeded = (
            moveit_cartesians(
                station,
                left=[pre_target_pose.tolist()],
                use_cartesian_path=True,
                left_motion_id=1,
            )
            if arm == "left"
            else moveit_cartesians(
                station,
                right=[pre_target_pose.tolist()],
                use_cartesian_path=True,
                right_motion_id=1,
            )
        )
        if succeeded:
            available_arms.append(arm)
        else:
            print(f"{arm}臂预移动失败，跳过该侧", flush=True)

    output_dir = (
        settings.ROOT
        / "output/chair_depth_grasp"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    hand_positions = {
        "left": settings.TASK["hands"]["left_open_position"],
        "right": settings.TASK["hands"]["right_open_position"],
    }
    grasped_arms = []
    for arm in available_arms:
        input(
            f"转椅步骤3：确认机器人静止，按 Enter 使用{arm}腕RGB-D识别扶手："
        )
        try:
            grasp = recognize_chair_grasp(
                station,
                arm,
                wrist_deprojectors[arm],
                output_dir,
            )
        except RuntimeError as error:
            print(
                f"{arm}腕扶手识别失败：{error}；诊断图已保存到{output_dir}",
                flush=True,
            )
            continue
        target_pose = np.asarray(
            grasp["target_pose_waist_xyz_xyzw"],
            dtype=np.float64,
        ).copy()
        target_pose[:3] += (
            rotation_xyzw(target_pose[3:]) @ CHAIR_GRASP_OFFSET_TCP_M
        )
        current_pose = station.get_robot_state()["arm_pose"][arm]
        current_pose = np.asarray(
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else current_pose,
            dtype=np.float64,
        ).copy()
        tcp_rotation = rotation_xyzw(current_pose[3:])
        total_delta_tcp_m = tcp_rotation.T @ (
            target_pose[:3] - current_pose[:3]
        )
        xy_delta_tcp_m = np.asarray(
            [total_delta_tcp_m[0], total_delta_tcp_m[1], 0.0],
            dtype=np.float64,
        )
        xy_target_pose = current_pose.copy()
        xy_target_pose[:3] += tcp_rotation @ xy_delta_tcp_m
        print(
            f"{arm}腕扶手像素={grasp['pixel_uv']}，深度={grasp['depth_m']:.3f} m；"
            f"总TCP位移={np.round(total_delta_tcp_m, 6).tolist()} m，"
            f"第一段XY={np.round(xy_delta_tcp_m, 6).tolist()} m",
            flush=True,
        )
        input(f"转椅步骤4：确认{arm}臂路径安全，按 Enter 执行TCP X/Y对齐：")
        succeeded = (
            moveit_cartesians(
                station,
                left=[xy_target_pose.tolist()],
                use_cartesian_path=True,
                left_motion_id=2,
            )
            if arm == "left"
            else moveit_cartesians(
                station,
                right=[xy_target_pose.tolist()],
                use_cartesian_path=True,
                right_motion_id=2,
            )
        )
        if not succeeded:
            print(f"{arm}臂TCP X/Y对齐失败", flush=True)
            continue
        print(
            f"{arm}臂第二段TCP Z="
            f"{total_delta_tcp_m[2] * 100:+.2f} cm，"
            f"最终目标XYZ={np.round(target_pose[:3], 6).tolist()} m",
            flush=True,
        )
        input(f"转椅步骤5：确认{arm}臂路径安全，按 Enter 执行TCP Z移动：")
        succeeded = (
            moveit_cartesians(
                station,
                left=[target_pose.tolist()],
                use_cartesian_path=True,
                left_motion_id=3,
            )
            if arm == "left"
            else moveit_cartesians(
                station,
                right=[target_pose.tolist()],
                use_cartesian_path=True,
                right_motion_id=3,
            )
        )
        if not succeeded:
            print(f"{arm}臂TCP Z移动失败", flush=True)
            continue
        input(f"转椅步骤6：确认{arm}夹爪位于扶手处，按 Enter 闭合夹爪：")
        hand_positions[arm] = settings.TASK["hands"][f"{arm}_close_position"]
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
        grasped_arms.append(arm)

    if not grasped_arms:
        print(
            "转椅左右臂均未抓取成功；松开双夹爪并返回失败",
            flush=True,
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
        return False

    if len(grasped_arms) == 1:
        print(
            f"转椅仅{grasped_arms[0]}臂抓取成功，仍按单臂抓取继续拖行",
            flush=True,
        )
    else:
        print("转椅左右臂均抓取成功，继续拖行", flush=True)
    input("转椅步骤7：确认抱椅回退路径安全，按 Enter 先恢复观察点和startpoint，再恢复box_point：")
    return_from_approach(navigation, approach_origin_node)
    navigation.recoverState("box_point", method="move_by")
    input("转椅步骤8：按 Enter 在box_point原地旋转90°：")
    navigation.moveBy(0.0, 0.0, 90.0, correct=True)
    input(
        f"转椅步骤9：确认前方{placement_distance_m:.2f} m安全，"
        "按 Enter 将转椅拖出通道："
    )
    navigation.moveBy(placement_distance_m, 0.0, 0.0)
    input("转椅步骤10：确认转椅位于通道外，按 Enter 松开双夹爪：")
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
    input("转椅步骤11：确认机械臂和腰腿路径安全，按 Enter 恢复应急初始姿态：")
    if not apply_startpoint_pose(station, initial_pose):
        return False
    navigation.recoverState("startpoint", method="move_by")
    navigation.turnTo(0.0)
    return True

def run_task1_box(
    station: Any,
    navigation: NavigationGraph,
    orientation_result: dict[str, Any],
    initial_pose: dict[str, Any],
    placement_distance_m: float,
    attempt_index: int,
    approach_origin_node: str,
    vertical_hold_path: Path = BOX_HOLD_VERTICAL_PATH,
) -> bool:
    """任务1-纸箱：抱持拖行（调用前需已完成近距微调）。"""
    print(f'稳定识别目标为纸箱，第{attempt_index}次尝试；进入横纸箱抱持、拖行和释放流程', flush=True)
    box_success = run_box_after_approach(
        station,
        navigation,
        orientation_result,
        initial_pose,
        placement_distance_m + 0.3,
        approach_origin_node,
        vertical_hold_path,
    )
    return box_success

def run_task1_debris(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    channel_region: dict[str, ChannelRegion],
    detector: ObstacleYolo,
    chair_view_detector: ChairViewYolo,
    deprojector: deprojector_module.Deprojector,
    wrist_deprojectors: dict[str, deprojector_module.Deprojector],
    initial_pose: dict[str, Any],
) -> str:
    """无限循环清障；仅在完整观察轮次无稳定目标时结束。"""
    round_index = 1
    target_types = set(TARGET_NAMES) - TASK1_IGNORED_TYPES
    type_attempt_counts: dict[str, int] = defaultdict(int)
    successful_removal_count = 0
    vertical_box_attempt_count = 0
    next_round_from_h0 = False
    last_observation = "startpoint"
    while True:
        started_from_h0 = next_round_from_h0
        (
            approach_origin_node,
            reliable_targets,
            approach,
        ) = find_task1_approach(
            station, navigation, avoid, detector, deprojector,
            channel_region, chair_view_detector, target_types,
            round_index, start_from_h0=next_round_from_h0,)
        next_round_from_h0 = False
        last_observation = approach_origin_node
        if approach is None:
            if reliable_targets:
                print(
                    "当前观察流程仍有稳定障碍物，但全部停靠距离均无"
                    "固定障碍安全接近路线；保持当前观察点并无限重试",
                    flush=True,
                )
                next_round_from_h0 = approach_origin_node == "h0_track_turn_2"
                round_index += 1
                continue
            if started_from_h0:
                print(
                    "局部重试观察流程无稳定目标；返回startpoint执行完整清场确认",
                    flush=True,
                )
                input("确认从h0返回startpoint的路径安全，按 Enter 继续：")
                navigation.recoverState("startpoint", method="move_by")
                round_index += 1
                continue
            print("startpoint与h0_track_turn_2均无稳定障碍物，任务1清场确认完成", flush=True)
            break
        nearest, (move_x, move_y, turn_deg), approach_motions = approach
        standoff_m = float(
            nearest.get(
                "approach_standoff_m",
                STANDOFF_M_BY_TYPE.get(
                    nearest["type"],
                    DEFAULT_STANDOFF_M,
                ),
            )
        )
        input(
            f"最近目标为{nearest['name']}，名义平移 x={move_x:.3f} m、"
            f"y={move_y:.3f} m，停在目标前 {standoff_m:.2f} m；"
            f"弯腰矩形规划为{len(approach_motions)}段且不旋转。"
            "确认安全后按 Enter 执行："
        )
        for motion_index, approach_motion in enumerate(approach_motions, 1):
            print(
                f"执行接近第{motion_index}/{len(approach_motions)}段："
                f"x={approach_motion[0]:+.3f} m，y={approach_motion[1]:+.3f} m，"
                f"theta={approach_motion[2]:+.1f}°",
                flush=True,
            )
            navigation.moveBy(*approach_motion, correct=False)
        if nearest["type"] != "box":
            print("都不转")
        placement_distance_m = (
            INITIAL_PLACE_DISTANCE_M
            - PLACE_DISTANCE_DECREMENT_M * successful_removal_count
        )
        target_type = nearest["type"]
        attempt_index = type_attempt_counts[target_type] + 1
        task1_name = f"任务1-{nearest['name']}"
        # 纸箱/转椅先用头部YOLO近距复识别并完成第二段平移。
        box_orientation_result = None
        box_vertical_hold_path = BOX_HOLD_VERTICAL_PATH
        if target_type in {"box", "chair"}:
            fallback_point_base_m = None
            if target_type == "box":
                # 第一段不旋转：当前目标坐标=第一次识别坐标-已执行的base平移。
                fallback_point_base_m = (
                    float(nearest["point_base_m"][0])
                    - sum(motion[0] for motion in approach_motions),
                    float(nearest["point_base_m"][1])
                    - sum(motion[1] for motion in approach_motions),
                )
            refine_success = refine_task1_approach(
                station,
                navigation,
                avoid,
                detector,
                deprojector,
                target_type,
                nearest.get("chair_view"),
                fallback_point_base_m,
            )
            if not refine_success:
                input(
                    f'{nearest["name"]}近距离复识别或微调失败；'
                    '确认返回路径安全后按 Enter 恢复startpoint：'
                )
                return_from_approach(navigation, approach_origin_node)
                round_index += 1
                continue
        if target_type == "box":
            fallback_orientation = (
                "vertical"
                if attempt_index >= BOX_NARROW_FALLBACK_ATTEMPT
                else "horizontal"
            )
            print(
                f"纸箱第{attempt_index}次尝试："
                "方向识别失败时回退为"
                f"{'竖抱窄边' if fallback_orientation == 'vertical' else '横抱宽边'}",
                flush=True,
            )
            box_orientation_result = recognize_box_orientation_until_ready(
                station,
                detector,
                deprojector,
                fallback_orientation,
            )
            if box_orientation_result["orientation"] == "vertical":
                vertical_box_attempt_count += 1
                if vertical_box_attempt_count >= 2:
                    box_vertical_hold_path = BOX_HOLD_VERTICAL_30_PATH
                print(
                    f"竖纸箱处理计数={vertical_box_attempt_count}，"
                    f"本轮使用{box_vertical_hold_path.name}",
                    flush=True,
                )

        def run_current_target() -> bool:
            if target_type == "box":
                box_success = run_task1_box(
                    station,
                    navigation,
                    box_orientation_result,
                    initial_pose,
                    placement_distance_m,
                    attempt_index,
                    approach_origin_node,
                    box_vertical_hold_path,
                )
                return box_success
            if target_type == "bin":
                print(f"稳定识别目标为{nearest['name']}，第{attempt_index}次尝试；使用垃圾桶远观察姿态执行单臂抓取、拿起和移出通道", flush=True)
                bin_success = run_task1_bin(station, navigation, detector, wrist_deprojectors, placement_distance_m, approach_origin_node)
                return bin_success
            if target_type == "filebox":
                print(f"稳定识别目标为{nearest['name']}，第{attempt_index}次尝试；使用垃圾桶远观察姿态执行单臂抓取、拿起和移出通道", flush=True)
                filebox_success = run_task1_filebox(station, navigation, detector, wrist_deprojectors, placement_distance_m, approach_origin_node)
                return filebox_success
            if target_type == "chair":
                return run_task1_chair(
                    station,
                    navigation,
                    wrist_deprojectors,
                    initial_pose,
                    placement_distance_m,
                    attempt_index,
                    nearest["chair_view"],
                    approach_origin_node,
                )
            input("机器人已到目标前并朝向目标；确认返回路径安全后按 Enter 恢复 startpoint：")
            return_from_approach(navigation, approach_origin_node)
            return True

        target_ok = run_with_retry_or_skip(
            task1_name,
            run_current_target,
            max_attempts=1,
            automatic=True,
        )
        if not target_ok:
            print(
                f"{task1_name}未完成，恢复本轮观察点后重新识别，"
                "不在旧接近状态下直接重试抓取",
                flush=True,
            )
            try:
                apply_startpoint_pose(station, initial_pose)
                navigation.recoverState(
                    approach_origin_node,
                    method="move_by",
                )
            except Exception as recover_error:
                print(f"失败后回位异常（忽略并继续）：{recover_error}", flush=True)
        else:
            successful_removal_count += 1
            if target_type == "box":
                vertical_box_attempt_count = 0
            print(
                f"{task1_name}已完成第{successful_removal_count}次成功搬运；"
                "该计数只用于递减投放距离，不屏蔽后续同类识别",
                flush=True,
            )
        print(
            f"已恢复{'startpoint' if target_ok else approach_origin_node}",
            flush=True,
        )
        if target_ok and target_type not in {"filebox", "chair", "bin"}:
            input("确认机器人已在startpoint且姿态恢复路径安全，按 Enter 继续")
        type_attempt_counts[target_type] = attempt_index
        if target_ok and approach_origin_node != "startpoint":
            input("目标来自h0_track_turn_2；按 Enter 返回该观察点执行抓取后复核：")
            navigation.recoverState("h0_track_turn_2", method="move_by")
        print(
            f"机器人已到{approach_origin_node}；自动复核5帧"
            f"{nearest['name']}",
            flush=True,
        )
        input("确认机器人和腰腿姿态已稳定，按 Enter 开始复核5帧：")
        verify_targets = recognize_task1_batch(
            station, detector, deprojector,)
        verify_targets = filter_targets_in_channel(
            navigation, channel_region, verify_targets,
        )
        stable_verify_targets = filter_targets_by_hits(verify_targets)
        still_present = any(
            target['type'] == target_type
            for target in stable_verify_targets
        )
        if not still_present:
            if approach_origin_node != "startpoint":
                next_round_from_h0 = True
                print(
                    "h0复核未检出当前类别；不据此剔除任何类别，"
                    "保持在h0继续完整识别",
                    flush=True,
                )
            print(
                f"{nearest['name']}本次复核未稳定检出；"
                "返回完整观察流程继续扫描全部类别",
                flush=True,
            )
        else:
            next_round_from_h0 = approach_origin_node != "startpoint"
            print(
                f"{nearest['name']}第{attempt_index}次尝试后仍存在，"
                "保留该目标并无限重试",
                flush=True,
            )
        round_index += 1
    print("任务1循环结束：已确认通道内无稳定障碍物", flush=True)
    return last_observation


def run_task1_final_chassis_clean(
    navigation: NavigationGraph,
    observation_node: str,
) -> str:
    """任务1视觉清场后，依次执行堆放处、通道1和通道2的底盘清理。

    当前尚未配置三段实测导航动作，因此不发送底盘指令，并返回机器人当前
    所在的Navigation节点。后续导航动作应按“堆放处→通道1→通道2”顺序
    添加在本函数内，返回值必须改为最后实际到达的命名节点。
    """
    print(
        f"任务1最终底盘清理入口已到达；当前保持在{observation_node}，"
        "尚未执行堆放处、通道1和通道2清理动作",
        flush=True,
    )
    return observation_node


def run_task1_and_go_to_task2_start(
    station: Any,
    navigation: NavigationGraph,
    avoid: AvoidCollision,
    channel_region: dict[str, ChannelRegion],
    detector: ObstacleYolo,
    chair_view_detector: ChairViewYolo,
    deprojector: deprojector_module.Deprojector,
    wrist_deprojectors: dict[str, deprojector_module.Deprojector],
    initial_pose: dict[str, Any],
) -> None:
    """无限重试任务1；仅在清场确认后进入任务2起点。"""
    exit_node = "startpoint"
    def run_debris() -> bool:
        nonlocal exit_node
        exit_node = run_task1_debris(
            station, navigation, avoid, channel_region, detector,
            chair_view_detector, deprojector, wrist_deprojectors,
            initial_pose,)
        return True
    run_until_success('任务1杂物', run_debris, automatic=True)

    def run_final_chassis_clean() -> bool:
        nonlocal exit_node
        exit_node = run_task1_final_chassis_clean(navigation, exit_node)
        return True

    run_until_success(
        "任务1最终底盘清理",
        run_final_chassis_clean,
        automatic=True,
    )

    def enter_task2_start() -> bool:
        input("任务1已确认清场；确认前往任务2起点的路径安全，按 Enter 继续：")
        return move_from_observation_to_task2_start(navigation, exit_node)

    run_until_success('进入任务2起点', enter_task2_start, automatic=True)

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
    for index, window_pose in enumerate(settings.WINDOW1_APPROACH_POSES):
        input(f"任务2步骤{index + 4}：确认机械臂路径安全，按 Enter 执行窗口迫近 new_step{index}")
        moveit_success = moveit_joints(station, left=window_pose["left_arm"])
        if not moveit_success:
            print(f"窗口迫近 new_step{index} 执行失败", flush=True)
            return False
    output_dir = (
        settings.ROOT
        / "output/task2_recognize"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    input("任务2步骤7：new_step0～2已完成，按 Enter 读取左腕 RGB-D、保存图像并识别窗把手")
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
    first_delta_waist_m[1] -= WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(f'识别点（left_tcp_link）：x={point_left_tcp_m[0] * 100:+.2f} cm，y={point_left_tcp_m[1] * 100:+.2f} cm，z={point_left_tcp_m[2] * 100:+.2f} cm', flush=True)
    print(f'第一次TCP局部偏移：x={first_delta_tcp_m[0] * 100:+.2f} cm，y={first_delta_tcp_m[1] * 100:+.2f} cm，z=+0.00 cm；另沿waist_yaw_link -Y向视觉右侧移动{WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M * 100:.2f} cm，方向保持不变', flush=True)
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
        {'point_left_tcp_xyz_m': point_left_tcp_m.tolist(), 'first_delta_left_tcp_xyz_m': first_delta_tcp_m.tolist(), 'first_extra_right_waist_y_m': -WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M, 'second_delta_left_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_z_rule': 'recognized_left_tcp_z_m + 0.20 m'},)
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
        'waist': {'position': [-0.08, 0.1]}},
        min_spd=0.10,
        max_spd=0.20,
        )
    time.sleep(0.5)
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
    input("任务2窗口2步骤1：窗口1恢复已完成，按 Enter 经h270_window1_turn、h0_track_end返回h0_track_turn_2")
    navigation.moveTo("h270_window1_turn", correct=True)
    navigation.moveTo(
        "h0_track_end",
        correct=False,
        rotate=False,
    )
    navigation.moveTo(
        "h0_track_turn_2",
        correct=False,
        rotate=False,
    )

    input("任务2窗口2步骤2：确认姿态切换路径安全，按 Enter 切换到应急_窗户2_准备姿态")
    prepare_pose = settings.WINDOW2_PREPARE_POSE
    moveit_success = moveit_joints(
        station, left=prepare_pose['left_arm'], right=prepare_pose['right_arm'],)
    if not moveit_success:
        print("双臂切换应急_窗户2_准备姿态失败", flush=True)
        return False
    input("任务2窗口2步骤3：窗户2准备姿态已到达，按 Enter 导航到h270_window2_action")
    navigation.moveTo("h270_window2_action", correct=True)
    window2_success = run_window2_action(station, navigation)
    # 向右平移脱离把手
    navigation.moveBy(-0.20, -0.20, 0.0, correct=False) # 向右后
    # 确保窗户打开角度
    navigation.moveBy(0.0, 0.45, 0.0, correct=False) # 向左
    # 左转腰避让
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [0.1, 0.0]}},
        min_spd=0.10,
        max_spd=0.20,
        )
    time.sleep(0.5)
    navigation.moveBy(0.20, 0.0, 0.0, correct=False) # 向前
    time.sleep(1)
    # 右转腰开窗
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [-0.4, 0.2]}},
        min_spd=0.10,
        max_spd=0.10,
        )
    time.sleep(7)
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [0.0, 0.0]}},
		min_spd=0.2,
		max_spd=0.3,
		)
    time.sleep(2)
    navigation.moveBy(-0.20, 0.0, 0.0, correct=False) # 向后
    navigation.moveBy(0.0, -0.20, 0.0, correct=False) # 向右
    # 双臂恢复位置
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    moveit_success = moveit_joints(
        station, left=initial_pose['left_arm'], right=initial_pose['right_arm'],)
    if window2_success:
        print("任务2窗口视觉操作完成", flush=True)
        return True
    print("任务2窗口2识别失败，已跳过窗口2操作", flush=True)
    return True

def run_task2_windows(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """任务2开窗：依次窗户1、窗户2。窗户1失败不阻断窗户2。"""
    try:
        window1_success = run_task2_window1(station, navigation)
    except Exception as error:
        print(f"任务2窗户1失败：{error}，继续窗户2", flush=True)
        window1_success = False
    if not window1_success:
        print("任务2窗户1未完成，继续窗户2", flush=True)
    try:
        window2_success = run_task2_window2(station, navigation)
    except Exception as error:
        print(f"任务2窗户2失败：{error}，前往判定点2", flush=True)
        window2_success = False
    try:
        navigation.moveTo("h270_task2_check", correct=True, rotate=False)
    except Exception as check_error:
        print(f"前往任务2判定点失败：{check_error}", flush=True)
    else:
        apply_startpoint_pose(station, json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8")))
    return window2_success


# ---------------------------------------------------------------------------
# 任务3 隔离带：抓取（带1/带2）+ 拉隔离带
# ---------------------------------------------------------------------------
def retry_task3_grasp_after_base_move(
    station: Any,
    navigation: NavigationGraph,
    arm: str,
    hand_positions: dict[str, float],
    first_result: dict[str, Any] | None,
    retry_move_xy_m: tuple[float, float],
    recovery_node: str,
) -> bool:
    """单侧TCP MoveIt失败时，张开当前夹爪、平移底盘并再抓一次。"""
    if first_result is not None and first_result["motion_succeeded"]:
        return True

    print(f"任务3 {arm}臂首次TCP MoveIt失败，张开当前夹爪准备底盘微调重抓", flush=True)
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
    time.sleep(2.0)
    retry_x_m, retry_y_m = retry_move_xy_m
    input(
        f"确认底盘微调路径安全，按 Enter 执行 "
        f"moveBy(x={retry_x_m:+.3f}, y={retry_y_m:+.3f}, theta=0) 后重新抓取"
    )
    navigation.moveBy(retry_x_m, retry_y_m, 0.0, correct=False)
    retry_result = grasp_black_single(station, arm, hand_positions)
    retry_success = retry_result is not None and retry_result["motion_succeeded"]
    navigation.recoverState(recovery_node, method="move_by")
    if not retry_success:
        print(f"任务3 {arm}臂底盘微调后重抓仍失败", flush=True)
    return retry_success


def run_task3_grasp_belt1(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """隔离带1（右侧）识别与抓取。"""
    input('任务3步骤1：确认机器人实际位于h270_task3_check且右侧安全，按 Enter 执行右侧黑色隔离带识别与抓取')
    #navigation.moveBy(-0.10,0.10,0,correct=True)
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
    right_success = retry_task3_grasp_after_base_move(
        station,
        navigation,
        "right",
        {"left": 0.0, "right": 0.0},
        right_result,
        TASK3_RIGHT_REGRASP_MOVE_XY_M,
        "h270_task3_check",
    )
    if not right_success:
        print("右侧黑色隔离带抓取失败，本任务失败", flush=True)
        return False
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
    left_success = retry_task3_grasp_after_base_move(
        station,
        navigation,
        "left",
        {"left": 0.0, "right": 1.0},
        left_result,
        TASK3_LEFT_REGRASP_MOVE_XY_M,
        "h270_task3_left",
    )
    if not left_success:
        print("左侧黑色隔离带抓取失败，本任务失败", flush=True)
        return False
    return True

def run_task3_grasp_belts(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """抓取隔离带：局部重抓仍失败时回任务3起点重做。"""
    for attempt_index in range(1, MAX_STAGE_ATTEMPTS + 1):
        print(
            f"任务3隔离带抓取第{attempt_index}/{MAX_STAGE_ATTEMPTS}轮",
            flush=True,
        )
        belt1_success = run_task3_grasp_belt1(station, navigation)
        belt2_success = (
            run_task3_grasp_belt2(station, navigation)
            if belt1_success
            else False
        )
        if belt1_success and belt2_success:
            return True
        print("任务3局部重抓仍失败，张开双夹爪并返回h270_task3_check后重试", flush=True)
        station.execute_action(
            {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
        )
        time.sleep(2.0)
        input("确认返回任务3起点的直线路径安全，按 Enter 继续：")
        navigation.recoverState("h270_task3_check", method="move_by")
        if attempt_index < MAX_STAGE_ATTEMPTS:
            input("已回到h270_task3_check，按 Enter 从右侧隔离带开始重做任务3抓取：")
    return False

def run_task3_pull_belts(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """拉隔离带：携带目标返回放置点、松爪并恢复姿态。"""
    input('任务3步骤3：确认两个目标均已抓稳且返回路线安全，按 Enter 携带目标返回h270_task3_place')
    navigation.moveTo("h270_task3_back", correct=False)
    navigation.moveTo("h270_task3_return", correct=True)
    navigation.moveTo("h270_task3_place", correct=False)
    print("任务3已到达放置点，张开双夹爪", flush=True)
    station.execute_action({"hand": {"position": {"left": [0.0], "right": [0.0]}}})
    time.sleep(2.0)
    navigation.moveTo("h270_task3_return", correct=False)
    recover_pose = settings.TASK3_PLACE_RECOVER_POSE
    print("任务3双臂切换到放置后恢复姿态，然后恢复腰腿", flush=True)
    moveit_success = moveit_joints(
        station, left=recover_pose['left_arm'], right=recover_pose['right_arm'],)
    if not moveit_success:
        print("双臂切换到放置后恢复姿态失败", flush=True)
        return False
    execute_synced_waist_leg_action(
        station, {'waist': {'position': recover_pose['waist']}, 'leg': {'position': recover_pose['leg']}},)
    print("任务3黑色隔离带抓取、放置和恢复完成", flush=True)
    return True

def run_task3_belts(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """任务3隔离带：抓取后拉放。进入前导航到 h270_task3_check。任务内步骤可跳过。"""
    input('任务2已结束；确认机器人当前位置及路径安全后按 Enter 导航到h270_task3_check')
    try:
        navigation.moveTo("h270_task3_check", correct=True)
    except Exception as nav_error:
        print(f"前往任务3检查点失败：{nav_error}", flush=True)
        if not run_with_retry_or_skip(
            '导航到h270_task3_check',
            lambda: (navigation.moveTo('h270_task3_check', correct=True), True)[1],
        ):
            print("尚未到达任务3检查点，交由上层重试任务3", flush=True)
            return False
    grasp_success = False
    pull_success = False
    try:
        grasp_success = run_task3_grasp_belts(station, navigation)
        if grasp_success:
            pull_success = run_task3_pull_belts(station, navigation)
        else:
            print("任务3抓取失败，跳过拉放，前往判定点3", flush=True)
    except Exception as error:
        print(f"任务3中途失败：{error}，前往判定点3", flush=True)
    try:
        navigation.moveTo("h270_task3_finish", correct=False)
    except Exception as finish_error:
        print(f"前往任务3判定点失败：{finish_error}", flush=True)
    else:
        apply_startpoint_pose(station, json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8")))
    return grasp_success and pull_success

# ---------------------------------------------------------------------------
# 主流程：初始化后仅调用三个任务
# ---------------------------------------------------------------------------
def main() -> None:
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    detector = ObstacleYolo()
    chair_view_detector = ChairViewYolo()
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
        with NavigationGraph(
            settings.prepare_route(), initial_node='startpoint', navigation_mode='move_by', linear_speed=float(navigation_config['linear_speed_m_s']), angular_speed=float(navigation_config['angular_speed_rad_s']), compress_path=False,) as navigation:
            def ensure_at_startpoint() -> bool:
                state_ok = navigation.checkState()
                if not state_ok:
                    raise RuntimeError("机器人实测位置不在 startpoint")
                return True
            run_until_success(
                '确认位于startpoint', ensure_at_startpoint, automatic=True,
            )
            input(
                '初始化已确认位于startpoint；确认双臂、头、腰和腿路径安全后'
                '按 Enter 恢复应急初始姿态：'
            )
            run_until_success(
                '恢复应急初始姿态',
                lambda: apply_startpoint_pose(station, initial_pose),
                automatic=True,
            )
            #等待恢复到应急初始姿态后再开始识别和抓取
            time.sleep(TASK1_YOLO_SETTLE_S)
            avoid = AvoidCollision(
                navigation,
                settings.CONFIG_DIR / "scene.json",
                collision_radius_m=COLLISION_RADIUS_M,
                bent_front_m=BENT_COLLISION_FRONT_M,
            )
            channel_region = {
                target_type: ChannelRegion(
                    settings.TASK["channel_region"]["polygon_xy_m"],
                    outside_margin_m=outside_margin_m,
                )
                for target_type, outside_margin_m
                in CHANNEL_OUTSIDE_MARGIN_M_BY_TYPE.items()
            }
            # 任务1
            run_task1_and_go_to_task2_start(
                station, navigation, avoid, channel_region, detector,
                chair_view_detector, deprojector, wrist_deprojectors,
                initial_pose,)
            # 任务2：完成前不进入任务3
            run_until_success(
                '任务2开窗', lambda: run_task2_windows(station, navigation),)
            # 任务3：完成前不结束流程
            run_until_success(
                '任务3隔离带', lambda: run_task3_belts(station, navigation),)
    finally:
        station.shutdown()

if __name__ == "__main__":
    main()
