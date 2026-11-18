#!/usr/bin/env python3
"""实时显示任务1中需要清理的目标框，不发送任何机器人运动命令。

头部RGB-D目标先从base_footprint转换到Navigation logical，再使用
``config/task.json``中的通道多边形、外侧忽略距离和任务1高度阈值过滤。
只有需要清理且具有有效深度的目标会出现在MJPEG页面中。
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from agent.navigation import NavigationGraph
from camera_io import read_head_camera
from channel_region import ChannelRegion
from test_target_live import (
    StreamState,
    ViewerHandler,
    annotate_and_compress,
    center_depth_m,
)
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


CHANNEL_MAX_OBJECT_Z_M = 0.80
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)


def measured_navigation_logical_pose(
    navigation: NavigationGraph,
) -> tuple[dict[str, float], bool]:
    """用实测SLAM位姿计算Navigation logical位姿和命名点容差状态。"""
    expected_logical, _ = navigation.getPose()
    diagnostics = navigation.getStateError()
    if expected_logical is None or diagnostics is None:
        raise RuntimeError("NavigationGraph没有可用的位姿诊断")

    expected_map = diagnostics["expected_map_pose"]
    actual_map = diagnostics["actual_map_pose"]
    logical_yaw_rad = math.radians(float(expected_logical["theta_deg"]))
    expected_map_yaw_rad = float(expected_map["yaw_rad"])
    actual_map_yaw_rad = float(actual_map["yaw_rad"])
    # T_map_logical = T_map_expected @ inv(T_logical_expected)，再将
    # 实测T_map_actual左乘inv(T_map_logical)得到T_logical_actual。
    T_logical_expected = np.asarray(
        [
            [math.cos(logical_yaw_rad), -math.sin(logical_yaw_rad), 0.0, float(expected_logical["x_m"])],
            [math.sin(logical_yaw_rad), math.cos(logical_yaw_rad), 0.0, float(expected_logical["y_m"])],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_map_expected = np.asarray(
        [
            [math.cos(expected_map_yaw_rad), -math.sin(expected_map_yaw_rad), 0.0, float(expected_map["x_m"])],
            [math.sin(expected_map_yaw_rad), math.cos(expected_map_yaw_rad), 0.0, float(expected_map["y_m"])],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_map_actual = np.asarray(
        [
            [math.cos(actual_map_yaw_rad), -math.sin(actual_map_yaw_rad), 0.0, float(actual_map["x_m"])],
            [math.sin(actual_map_yaw_rad), math.cos(actual_map_yaw_rad), 0.0, float(actual_map["y_m"])],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_map_logical = T_map_expected @ np.linalg.inv(T_logical_expected)
    T_logical_actual = np.linalg.inv(T_map_logical) @ T_map_actual
    return (
        {
            "x_m": float(T_logical_actual[0, 3]),
            "y_m": float(T_logical_actual[1, 3]),
            "theta_deg": math.degrees(math.atan2(
                float(T_logical_actual[1, 0]),
                float(T_logical_actual[0, 0]),
            )),
        },
        bool(diagnostics["within_tolerance"]),
    )


def select_cleaning_detections(
    detections: list[dict[str, Any]],
    depth_mm: np.ndarray,
    deprojector: deprojector_module.Deprojector,
    logical_pose: dict[str, float],
    channel_region: dict[str, ChannelRegion],
) -> list[dict[str, Any]]:
    """返回需要清理的检测，坐标为Navigation logical，长度单位为米。"""
    depths_m = [
        center_depth_m(depth_mm, detection["bbox_xyxy"])
        for detection in detections
    ]
    valid_indices = [
        index
        for index, depth_m in enumerate(depths_m)
        if depth_m is not None
    ]
    if not valid_indices:
        return []

    poses_waist = deprojector.deproject_many(
        [
            [
                detections[index]["center_uv"][0],
                detections[index]["center_uv"][1],
                depths_m[index],
            ]
            for index in valid_indices
        ],
        face_target=False,
    )
    points_base = deprojector.transform_many_to_ground(poses_waist[:, :3])
    theta_rad = math.radians(float(logical_pose["theta_deg"]))
    cosine, sine = math.cos(theta_rad), math.sin(theta_rad)
    # T_logical_base：base_footprint目标点左乘后转换到Navigation logical。
    T_logical_base = np.asarray(
        [
            [cosine, -sine, 0.0, float(logical_pose["x_m"])],
            [sine, cosine, 0.0, float(logical_pose["y_m"])],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    selected = []
    for index, point_base in zip(valid_indices, points_base, strict=True):
        point_logical = T_logical_base @ np.asarray(
            [point_base[0], point_base[1], point_base[2], 1.0],
            dtype=np.float64,
        )
        if point_logical[2] > CHANNEL_MAX_OBJECT_Z_M:
            continue
        target_type = detections[index]["type"]
        if not channel_region[target_type].requires_clearing(
            float(point_logical[0]),
            float(point_logical[1]),
        ):
            continue
        detection = {
            **detections[index],
            "depth_m": depths_m[index],
            "point_base_m": point_base,
            "point_logical_m": point_logical[:3],
        }
        selected.append(detection)
        print(
            f"需要清理 {detection['type']}：confidence="
            f"{detection['confidence']:.3f}，logical="
            f"({point_logical[0]:.3f}, {point_logical[1]:.3f}, "
            f"{point_logical[2]:.3f}) m",
            flush=True,
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实时显示任务1通道内需要清理的目标框",
    )
    parser.add_argument("--node", default="startpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8112)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--quality", type=int, default=55)
    parser.add_argument(
        "--depth",
        action="store_true",
        help="在目标框标签中显示中心深度",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        deprojector = deprojector_module.Deprojector(
            station,
            CALIBRATION_PATH,
            "head",
        )
        channel_config = settings.TASK["channel_region"]
        channel_region = {
            target_type: ChannelRegion(
                channel_config["polygon_xy_m"],
                outside_margin_m=float(outside_margin_m),
            )
            for target_type, outside_margin_m in channel_config[
                "outside_ignore_distance_m_by_type"
            ].items()
        }
        with NavigationGraph(
            settings.prepare_route(),
            initial_node=args.node,
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            logical_pose, within_node_tolerance = (
                measured_navigation_logical_pose(navigation)
            )
            if not within_node_tolerance:
                print(
                    f"提示：机器人实测位置不在{args.node}容差内；"
                    "已改用实测SLAM换算的Navigation logical位姿",
                    flush=True,
                )

            state = StreamState()
            server = ThreadingHTTPServer(
                (args.host, args.port),
                ViewerHandler,
            )
            server.stream_state = state
            server_thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
            )
            server_thread.start()
            print(f"仅显示需清理目标：http://<机器人IP>:{args.port}", flush=True)
            print(
                f"实测logical=({logical_pose['x_m']:.3f}, "
                f"{logical_pose['y_m']:.3f}, {logical_pose['theta_deg']:.1f}°)，"
                "不会发送运动命令",
                flush=True,
            )

            interval_s = 1.0 / args.fps
            last_publish = time.monotonic()
            try:
                while True:
                    loop_started = time.monotonic()
                    color, depth_mm = read_head_camera(station)
                    if color is None or depth_mm is None:
                        continue
                    inference_started = time.monotonic()
                    detections = detector.detect(color)
                    inference_ms = (
                        time.monotonic() - inference_started
                    ) * 1000.0
                    logical_pose, _ = measured_navigation_logical_pose(
                        navigation,
                    )
                    selected = select_cleaning_detections(
                        detections,
                        depth_mm,
                        deprojector,
                        logical_pose,
                        channel_region,
                    )
                    display_detections = (
                        selected
                        if args.depth
                        else [
                            {
                                key: value
                                for key, value in detection.items()
                                if key != "depth_m"
                            }
                            for detection in selected
                        ]
                    )
                    jpeg = annotate_and_compress(
                        color,
                        display_detections,
                        width=args.width,
                        quality=args.quality,
                    )
                    now = time.monotonic()
                    elapsed = now - last_publish
                    depths_m = [
                        float(detection["depth_m"])
                        for detection in selected
                    ]
                    state.publish(
                        jpeg,
                        {
                            "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
                            "inference_ms": inference_ms,
                            "kbps": len(jpeg) * 8.0 / 1000.0
                            / max(elapsed, interval_s),
                            "objects": len(selected),
                            "depth_enabled": args.depth,
                            "nearest_depth_m": (
                                min(depths_m)
                                if args.depth and depths_m
                                else None
                            ),
                        },
                    )
                    last_publish = now
                    time.sleep(max(
                        0.0,
                        interval_s - (time.monotonic() - loop_started),
                    ))
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join()
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
