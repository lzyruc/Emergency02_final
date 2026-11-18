#!/usr/bin/env python3
"""手动输入底盘相对运动，随后测试任务1姿态和腕部抓取点识别。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import agent.perception.deprojector as deprojector_module
import settings
from agent.navigation import AvoidCollision, NavigationGraph
from task1 import (
    BIN_PREGRASP_PATHS,
    CHAIR_PREGRASP_PATHS,
    FILEBOX_PREGRASP_PATHS,
    _execute_object_pose,
)
from test_box import run_box_after_approach
from wrist_phase1 import Phase1Model
from wrist_target import (
    TARGET_NAMES,
    WRIST_DEPTH_SCALE_M,
    wrist_grasp_candidates,
)
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TASK_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
INITIAL_POSE_PATH = TASK_DIR / "poses_emergency/应急_初始姿态.json"
POSE_PATHS_BY_TARGET = {
    "chair": CHAIR_PREGRASP_PATHS,
    "filebox": FILEBOX_PREGRASP_PATHS,
    "bin": BIN_PREGRASP_PATHS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="跳过头部YOLO，手动移动到任务1目标前进行测试"
    )
    parser.add_argument(
        "target",
        choices=("chair", "filebox", "bin", "box"),
        help="准备测试的物体类别",
    )
    return parser.parse_args()


def recognize_from_poses(
    station: Any,
    target_type: str,
    deprojectors: dict[str, deprojector_module.Deprojector],
    models: dict[str, Phase1Model],
) -> dict[str, Any] | None:
    """逐个切换目标预抓取姿态，双腕识别但不执行抓取。"""
    pose_paths = POSE_PATHS_BY_TARGET[target_type]
    for index, path in enumerate(pose_paths):
        input(
            f"确认机械臂、腰腿路径安全，按 Enter 切换{TARGET_NAMES[target_type]}"
            f"预抓取姿态 {index}（{path.name}）并执行双腕识别；Ctrl+C 终止："
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
        candidates.sort(key=lambda item: item["tcp_distance_m"])
        if not candidates:
            print(
                f"{TARGET_NAMES[target_type]}预抓取姿态 {index}："
                "左右腕均无候选点",
                flush=True,
            )
            continue
        for candidate in candidates:
            print(
                f"{candidate['arm']}腕：类别={candidate['class']}，"
                f"像素={candidate['pixel_uv']}，"
                f"点来源={candidate['point_source']}，"
                f"置信度={candidate['confidence']:.3f}，"
                f"选择分={candidate['selection_score']:.3f}，"
                f"深度={candidate['depth_m']:.3f} m"
                f"（{candidate['depth_source']}），"
                f"TCP距离={candidate['tcp_distance_m']:.3f} m，"
                f"目标Pose={candidate['target_pose_waist_xyz_xyzw'].tolist()}",
                flush=True,
            )
        if candidates[0]["tcp_distance_m"] < 0.4:
            print(
                f"按task1规则选择{candidates[0]['arm']}臂候选；"
                "本脚本不执行MoveIt抓取，也不闭合夹爪",
                flush=True,
            )
            return candidates[0]
    print(
        f"全部{len(pose_paths)}个姿态均无0.4 m内候选点",
        flush=True,
    )
    return None


def main() -> None:
    args = parse_args()
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
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
        deprojectors = {}
        models = {}
        if args.target != "box":
            deprojectors = {
                side: deprojector_module.Deprojector(
                    station,
                    CALIBRATION_PATH,
                    side,
                )
                for side in ("left", "right")
            }
            models = {
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
                raise RuntimeError("机器人实测位置不在startpoint，停止手动测试")
            input(
                "确认机器人位于startpoint且姿态路径安全，按 Enter 恢复"
                "应急初始姿态："
            )
            if not _execute_object_pose(station, initial_pose):
                raise RuntimeError("应急初始姿态执行失败")
            avoid = AvoidCollision(navigation, "emergency.json")

            while True:
                command = input(
                    "输入 dx dy dtheta（base_footprint：米、米、度），"
                    "或 done / return / quit："
                ).strip()
                if command == "done":
                    break
                if command == "quit":
                    print("未执行后续目标流程，测试结束", flush=True)
                    return
                if command == "return":
                    input("确认返回路径安全，按 Enter 恢复startpoint：")
                    navigation.recoverState()
                    print("已恢复startpoint，可继续输入移动命令", flush=True)
                    continue
                try:
                    dx, dy, dtheta = map(float, command.split())
                except ValueError:
                    print("格式错误，应输入三个数，或 done / return / quit", flush=True)
                    continue
                if avoid.checkCollision(dx, dy, dtheta):
                    print("该相对运动与固定障碍物碰撞，未发送底盘命令", flush=True)
                    continue
                input(
                    f"将执行 moveBy({dx:.3f}, {dy:.3f}, {dtheta:.1f})；"
                    "确认路径安全后按 Enter："
                )
                navigation.moveBy(dx, dy, dtheta)

            if args.target == "box":
                run_box_after_approach(station, navigation)
                input(
                    "确认机器人姿态恢复路径安全，按 Enter 恢复应急初始姿态："
                )
                if not _execute_object_pose(station, initial_pose):
                    raise RuntimeError("应急初始姿态执行失败")
            else:
                recognize_from_poses(
                    station,
                    args.target,
                    deprojectors,
                    models,
                )
                input(
                    "腕部识别结束；确认姿态及返回路径安全后按 Enter 恢复"
                    "应急初始姿态和startpoint："
                )
                if not _execute_object_pose(station, initial_pose):
                    raise RuntimeError("应急初始姿态执行失败")
                navigation.recoverState()
            print("手动导航目标测试完成", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
