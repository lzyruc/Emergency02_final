"""应急任务 1：选择并清理逃生通道内的障碍物。

本模块只负责任务编排。station 和 NavigationGraph 由比赛入口创建后传入，
本模块不创建硬件资源。感知边界统一使用以下目标格式：

    {
        "type": "chair" | "filebox" | "bin" | "box",
        "pose": [x_m_forward, y_m_left, yaw_deg_ccw],
        "confidence": 0.0,
    }

``pose`` 位于当前机器人 ``base_footprint`` 坐标系，长度单位为米，角度单位为度。
初版只按平面距离选择目标，confidence 留给感知和后续排序策略使用。

离线最小示例：

    target = select_nearest_obstacle([
        {"type": "box", "pose": [1.2, 0.1, 0.0], "confidence": 0.9},
        {"type": "filebox", "pose": [0.8, -0.1, 0.0], "confidence": 0.8},
    ])
    assert target["type"] == "filebox"
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np

import settings
from agent.perception.deprojector import Deprojector
from arm_motion import moveit_cartesians, moveit_joints
from blue_depth_target import BlueDepthTargetDetector
from camera_io import read_head_camera, read_wrist_camera
from task2_phase1 import Phase1Model


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = ROOT / "agent/perception/config/camera_calibration.json"
TASK_DIR = Path(__file__).resolve().parent
FILEBOX_POSE_DIR = TASK_DIR / "poses_filebox"
CHAIR_POSE_DIR = TASK_DIR / "poses_chair"
BIN_POSE_DIR = TASK_DIR / "poses_bin"
BOX_POSE_DIR = TASK_DIR / "poses_box"
FILEBOX_PREGRASP_PATHS = tuple(
    FILEBOX_POSE_DIR / f"应急_文件框{index}.json" for index in range(3)
)
FILEBOX_LIFT_PATH = FILEBOX_POSE_DIR / "应急_文件框_拿起.json"
CHAIR_PREGRASP_PATH_BY_VIEW = {
    "front": CHAIR_POSE_DIR / "应急_转椅_正面.json",
    "back": CHAIR_POSE_DIR / "应急_转椅_背面.json",
}
CHAIR_BACK_POSE_PATHS = (
    CHAIR_POSE_DIR / "应急_右手抓椅子1_挺身准备.json",
    CHAIR_POSE_DIR / "应急_右手抓椅子2_单手动3.json",
    CHAIR_POSE_DIR / "应急_右手抓椅子3_准备抓取4.json",
)
CHAIR_SIDE_POSE_PATHS = (
    CHAIR_POSE_DIR / "应急_转椅_侧面_1起身准备.json",
    CHAIR_POSE_DIR / "应急_转椅_侧面_2准备抓取.json",
)
# 旧的姿态遍历测试仍只使用正面姿态；正式流程按上面的视角映射选择。
CHAIR_PREGRASP_PATHS = (CHAIR_PREGRASP_PATH_BY_VIEW["front"],)
BIN_PREGRASP_PATHS = (
    BIN_POSE_DIR / "应急_垃圾桶_1远观察.json",
)
BIN_LIFT_PATH = BIN_POSE_DIR / "应急_垃圾桶_3提起.json"
BOX_PREGRASP_PATH = BOX_POSE_DIR / "应急_纸箱_1预备.json"
BOX_HOLD_HORIZONTAL_PATH = BOX_POSE_DIR / "应急_纸箱_2横箱子.json"
BOX_HOLD_VERTICAL_PATH = BOX_POSE_DIR / "应急_纸箱_2竖箱子.json"
BOX_HOLD_VERTICAL_30_PATH = BOX_POSE_DIR / "应急_纸箱_2竖箱子_30.json"
TARGET_NAMES = {
    "chair": "办公转椅",
    "filebox": "文件筐",
    "bin": "垃圾桶",
    "box": "纸箱（3 kg）",
}
PHASE1_TO_YOLO_TYPE = {
    "chair": "chair",
    "basket": "filebox",
    "trash_can": "bin",
    "box": "box",
}


def recognize_obstacles(station: Any) -> list[dict[str, Any]]:
    """读取一帧头部 RGB-D，返回四类障碍物的底盘相对位姿。

    ``pose`` 为 ``base_footprint`` 下的 ``[x前/米, y左/米, yaw度]``；
    ``point_base_m`` 为同坐标系 XYZ，``distance_3d_m`` 为三维欧氏距离。
    """
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))[
        "cameras"
    ]["head"]
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    width, height = calibration["image_size"]
    intrinsics = {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "width": int(width),
        "height": int(height),
    }
    color, depth_mm = read_head_camera(station)
    if color is None or depth_mm is None:
        raise RuntimeError("未获取到头部 RGB-D 帧")

    model = Phase1Model(intrinsics)
    model.detect({"rgb": color, "depth": depth_mm})
    objects = [
        obj
        for obj in model.last_frame_result.objects
        if obj.cls in PHASE1_TO_YOLO_TYPE
    ]
    if not objects:
        return []

    samples = []
    for obj in objects:
        x_camera_m, y_camera_m, depth_m = (
            float(value) for value in obj.position_cam
        )
        samples.append(
            [
                x_camera_m * intrinsics["fx"] / depth_m + intrinsics["cx"],
                y_camera_m * intrinsics["fy"] / depth_m + intrinsics["cy"],
                depth_m,
            ]
        )

    deprojector = Deprojector(station, CALIBRATION_PATH, "head")
    poses_waist = deprojector.deproject_many(samples, face_target=False)
    targets = []
    for obj, pose_waist in zip(objects, poses_waist, strict=True):
        target_type = PHASE1_TO_YOLO_TYPE[obj.cls]
        point_base = deprojector.transform_to_ground(pose_waist[:3])
        targets.append(
            {
                "type": target_type,
                "name": TARGET_NAMES[target_type],
                "confidence": float(obj.confidence),
                "pose": [
                    float(point_base[0]),
                    float(point_base[1]),
                    math.degrees(
                        math.atan2(float(point_base[1]), float(point_base[0]))
                    ),
                ],
                "point_base_m": point_base.tolist(),
                "distance_3d_m": float(np.linalg.norm(point_base)),
            }
        )
    return targets


def _move_to_standoff(
    navigation: Any,
    pose: Sequence[float],
    standoff_m: float,
) -> None:
    """沿目标方向平移并转向，在目标前指定距离停止。"""
    x_m, y_m = float(pose[0]), float(pose[1])
    distance_m = math.hypot(x_m, y_m)
    if distance_m == 0.0:
        raise ValueError("目标与 base_footprint 原点重合，无法计算接近方向")
    travel_m = max(0.0, distance_m - standoff_m)
    if travel_m > 0.0:
        navigation.moveBy(
            x_m * travel_m / distance_m,
            y_m * travel_m / distance_m,
            0.0,
        )
    navigation.moveBy(0.0, 0.0, math.degrees(math.atan2(y_m, x_m)))


def _execute_object_pose(station: Any, pose: dict[str, Any]) -> bool:
    """串行执行双臂和头、腰、腿预设，不改变夹爪状态。"""
    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        return False
    input(
        "确认手臂已回正，确认机械臂路径安全后按 Enter "
        "恢复双臂以及初始头、腰、腿姿态："
    )
    station.execute_action(
        {
            "head": {"position": pose["head"]},
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        }
    )
    time.sleep(1.0)
    return True


def _filebox_wrist_candidates(
    station: Any,
    deprojectors: dict[str, Deprojector],
    detectors: dict[str, BlueDepthTargetDetector],
) -> list[dict[str, Any]]:
    """用左右腕识别文件筐，返回抓取Pose及其到对应当前TCP的距离。"""
    candidates = []
    for arm, camera_config in (
        ("left", settings.LEFT_CAMERA),
        ("right", settings.RIGHT_CAMERA),
    ):
        try:
            color, depth_mm = read_wrist_camera(station, camera_config)
            detection = detectors[arm].detect(color, depth_mm)
        except RuntimeError as exc:
            print(f"{arm}腕未识别到文件筐：{exc}", flush=True)
            continue

        u, v = detection.pixel_uv
        target_pose = deprojectors[arm].deproject(
            [float(u), float(v), float(detection.point_camera_m[2])],
            face_target=False,
        )
        current_pose = station.get_robot_state()["arm_pose"][arm]
        current_pose = (
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else np.asarray(current_pose, dtype=np.float64)
        )
        candidates.append(
            {
                "arm": arm,
                "confidence": float(detection.confidence),
                "target_pose_waist_xyz_xyzw": target_pose,
                "tcp_distance_m": float(
                    np.linalg.norm(target_pose[:3] - current_pose[:3])
                ),
            }
        )
    return candidates


def _chair_bin_wrist_candidates(
    station: Any,
    target_type: str,
    deprojectors: dict[str, Deprojector],
    models: dict[str, Phase1Model],
) -> list[dict[str, Any]]:
    """用双腕 RGB-D 精定位椅背上沿或垃圾桶桶口，不执行抓取。"""
    phase1_type = {"chair": "chair", "bin": "trash_can"}[target_type]
    candidates = []
    for arm, camera_config in (
        ("left", settings.LEFT_CAMERA),
        ("right", settings.RIGHT_CAMERA),
    ):
        try:
            color, depth_mm = read_wrist_camera(station, camera_config)
        except RuntimeError as exc:
            print(f"{arm}腕相机读取失败：{exc}", flush=True)
            continue
        output = models[arm].detect(
            {"rgb": color, "depth": depth_mm},
            target_class=phase1_type,
        )
        if output is None or output["x"] is None or output["y"] is None:
            print(f"{arm}腕未识别到{TARGET_NAMES[target_type]}抓取点", flush=True)
            continue
        u, v = int(output["x"]), int(output["y"])
        row_start = max(0, v - 2)
        row_stop = min(depth_mm.shape[0], v + 3)
        column_start = max(0, u - 2)
        column_stop = min(depth_mm.shape[1], u + 3)
        depth_window = depth_mm[row_start:row_stop, column_start:column_stop]
        valid_depth = depth_window[
            (depth_window > 0) & np.isfinite(depth_window)
        ]
        if valid_depth.size == 0:
            print(f"{arm}腕抓取点附近没有有效深度", flush=True)
            continue
        depth_m = float(np.median(valid_depth)) / 1000.0
        target_pose = deprojectors[arm].deproject(
            [float(u), float(v), depth_m],
            face_target=False,
        )
        current_pose = station.get_robot_state()["arm_pose"][arm]
        current_pose = (
            current_pose.get_xyz_m_xyzw_ndarray()
            if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
            else np.asarray(current_pose, dtype=np.float64)
        )
        selected_object = max(
            (
                obj
                for obj in models[arm].last_frame_result.objects
                if obj.cls == phase1_type
            ),
            key=lambda obj: float(obj.confidence),
        )
        candidates.append(
            {
                "arm": arm,
                "confidence": float(selected_object.confidence),
                "pixel_uv": [u, v],
                "depth_m": depth_m,
                "target_pose_waist_xyz_xyzw": target_pose,
                "tcp_distance_m": float(
                    np.linalg.norm(target_pose[:3] - current_pose[:3])
                ),
            }
        )
    return candidates


def select_nearest_obstacle(
    obstacles: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """返回当前机器人平面距离最近的障碍物；列表为空时返回 None。

    输入目标的 ``pose`` 为 ``base_footprint`` 下的
    ``[x 前/米, y 左/米, yaw 逆时针/度]``。本函数不修改输入列表。
    """
    if not obstacles:
        return None
    return min(
        obstacles,
        key=lambda obstacle: math.hypot(
            float(obstacle["pose"][0]),
            float(obstacle["pose"][1]),
        ),
    )


def main(
    station: Any,
    navigation: Any,
    observe_obstacles: Callable[[Any], Sequence[dict[str, Any]]],
) -> bool:
    """持续观察并清理通道；全部清理完成返回 True，子任务失败返回 False。

    ``observe_obstacles(station)`` 每次都应重新采集图像，只返回仍位于通道内的
    障碍物。这样每清理一个物体后都会重新定位和排序，不使用旧检测结果。
    """
    while True:
        target = select_nearest_obstacle(observe_obstacles(station))
        if target is None:
            return True
        if not remove_object(station, navigation, target):
            return False


def remove_object(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """按照感知类别分派清理策略，并返回该物体是否已移出通道。"""
    target_type = target["type"]
    if target_type == "chair":
        return remove_chair(station, navigation, target)
    if target_type == "box":
        return remove_box(station, navigation, target)
    if target_type == "bin":
        return remove_bin(station, navigation, target)
    if target_type == "filebox":
        return remove_filebox(station, navigation, target)
    raise ValueError(f"不支持的障碍物类别: {target_type}")


def remove_bin(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理垃圾桶：接近、预备、夹持、少量抬起、移出通道、释放并回位。"""
    # 待接入：垃圾桶三段预设姿态和经用户确认的放置路线。
    raise NotImplementedError("垃圾桶清理动作尚未接入")


def remove_box(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理 3 kg 横放纸箱：接近、双臂抱持、拖动、释放并回位。"""
    pregrasp_pose = json.loads(
        BOX_PREGRASP_PATH.read_text(encoding="utf-8")
    )
    hold_pose = json.loads(
        BOX_HOLD_HORIZONTAL_PATH.read_text(encoding="utf-8")
    )

    _move_to_standoff(navigation, target["pose"], 1.0)
    boxes = [
        item for item in recognize_obstacles(station) if item["type"] == "box"
    ]
    if not boxes:
        print("第一段接近后未重新识别到纸箱，恢复接近前位置", flush=True)
        navigation.recoverState()
        return False
    target = min(
        boxes,
        key=lambda item: math.hypot(item["pose"][0], item["pose"][1]),
    )
    _move_to_standoff(navigation, target["pose"], 0.4)

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
    if not _execute_object_pose(station, pregrasp_pose):
        print("纸箱预备姿态执行失败", flush=True)
        return False
    if not _execute_object_pose(station, hold_pose):
        print("纸箱横向抱持姿态执行失败", flush=True)
        return False

    navigation.turnTo(270.0)
    navigation.moveTo("startpoint", correct=True, rotate=False)
    navigation.turnTo(0.0)
    navigation.moveBy(0.0, -0.3, 0.0)
    navigation.turnTo(90.0)
    navigation.moveBy(0.8, 0.0, 0.0)

    # 先张开双臂留住纸箱，再恢复底盘到 startpoint。
    if not _execute_object_pose(station, pregrasp_pose):
        print("纸箱释放姿态执行失败", flush=True)
        return False
    navigation.recoverState("startpoint", method="move_by")
    navigation.turnTo(0.0)
    return True


def remove_chair(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理办公转椅：接近、夹持椅背、侧移或拖行、释放并回位。"""
    # 待接入：椅背朝向判断、正反面预设姿态和拖行路线。
    raise NotImplementedError("办公转椅清理动作尚未接入")


def remove_filebox(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理文件筐：两段视觉接近、腕部精定位、抓取、移出通道并回位。"""
    pregrasp_poses = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in FILEBOX_PREGRASP_PATHS
    ]
    lift_pose = json.loads(FILEBOX_LIFT_PATH.read_text(encoding="utf-8"))
    deprojectors = {
        "left": Deprojector(station, CALIBRATION_PATH, "left"),
        "right": Deprojector(station, CALIBRATION_PATH, "right"),
    }
    detectors = {
        "left": BlueDepthTargetDetector(
            np.asarray(settings.LEFT_CAMERA["camera_matrix"], dtype=np.float64)
        ),
        "right": BlueDepthTargetDetector(
            np.asarray(settings.RIGHT_CAMERA["camera_matrix"], dtype=np.float64)
        ),
    }

    _move_to_standoff(navigation, target["pose"], 1.0)
    fileboxes = [
        item for item in recognize_obstacles(station) if item["type"] == "filebox"
    ]
    if not fileboxes:
        print("第一段接近后未重新识别到文件筐，恢复接近前位置", flush=True)
        navigation.recoverState()
        return False
    target = min(
        fileboxes,
        key=lambda item: math.hypot(item["pose"][0], item["pose"][1]),
    )
    _move_to_standoff(navigation, target["pose"], 0.4)

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

    grasped = False
    for index, pose in enumerate(pregrasp_poses):
        if not _execute_object_pose(station, pose):
            print(f"文件筐预抓取姿态 {index} 执行失败", flush=True)
            return False
        candidates = _filebox_wrist_candidates(
            station,
            deprojectors,
            detectors,
        )
        if not candidates:
            continue
        candidate = min(candidates, key=lambda item: item["tcp_distance_m"])
        print(
            f"文件筐预抓取姿态 {index}：选择{candidate['arm']}臂，"
            f"TCP距离={candidate['tcp_distance_m']:.3f} m，"
            f"置信度={candidate['confidence']:.3f}",
            flush=True,
        )
        if candidate["tcp_distance_m"] >= 0.4:
            continue

        target_pose = candidate["target_pose_waist_xyz_xyzw"].tolist()
        if candidate["arm"] == "left":
            succeeded = moveit_cartesians(station, left=[target_pose])
        else:
            succeeded = moveit_cartesians(station, right=[target_pose])
        if not succeeded:
            print("文件筐抓取点 MoveIt 执行失败", flush=True)
            return False

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
        grasped = True
        break

    if not grasped:
        print("三个文件筐预抓取姿态均无 0.4 m 内候选点", flush=True)
        return False
    if not _execute_object_pose(station, lift_pose):
        print("文件筐拿起姿态执行失败", flush=True)
        return False

    navigation.turnTo(0.0)
    navigation.moveTo("startpoint", correct=True, rotate=False)
    navigation.turnTo(90.0)
    navigation.moveBy(0.5, 0.0, 0.0)
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
    navigation.recoverState("startpoint", method="move_by")
    navigation.turnTo(0.0)
    return True
