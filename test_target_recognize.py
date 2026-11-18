#!/usr/bin/env python3
"""只读头部 RGB-D，用应急 YOLO 打印四类障碍物并选出最近目标。
'''
cd /home/nvidia/tienyi2_pro_move/Emergency
source /home/nvidia/xos/setup.bash
source /home/nvidia/XARM/install/setup.bash
source /home/nvidia/eai/env/bin/activate
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/usr/lib/python3.12/dist-packages" \
python test_target_recognize.py
'''
本测试不会调用 Navigation、MoveIt 或 ``station.execute_action()``，不会发送运动命令。
当前不使用相机外参；距离取检测框中心区域的有效深度中位数，单位为米。
"""

from __future__ import annotations

import time

import numpy as np

from camera_io import read_head_camera
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


def center_depth_m(
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> float | None:
    """返回检测框中心 30% 区域的有效深度中位数。"""
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


def main() -> None:
    """创建唯一 station，执行一次只读识别并打印结果。"""
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        time.sleep(1.0)
        color, depth_mm = read_head_camera(station)
        if color is None or depth_mm is None:
            raise RuntimeError("未获取到头部 RGB-D 帧")

        detector = ObstacleYolo()
        detections = detector.detect(color)
        if not detections:
            print("未识别到办公转椅、纸箱、垃圾桶或文件筐", flush=True)
            return

        targets = []
        for detection in detections:
            depth_m = center_depth_m(depth_mm, detection["bbox_xyxy"])
            target = {**detection, "depth_m": depth_m}
            targets.append(target)
            x_min, y_min, x_max, y_max = target["bbox_xyxy"]
            center_u, center_v = target["center_uv"]
            depth_text = "无有效深度" if depth_m is None else f"{depth_m:.3f} m"
            print(
                f"物体：{TARGET_NAMES[target['type']]}（{target['type']}），"
                f"置信度={target['confidence']:.3f}，"
                f"bbox=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]，"
                f"中心=({center_u:.1f}, {center_v:.1f})，"
                f"中心深度={depth_text}",
                flush=True,
            )

        targets_with_depth = [
            target for target in targets if target["depth_m"] is not None
        ]
        if not targets_with_depth:
            print("所有检测目标的中心区域均无有效深度，无法判断最近目标", flush=True)
            return
        nearest = min(targets_with_depth, key=lambda target: target["depth_m"])
        print(
            f"最近物体：{TARGET_NAMES[nearest['type']]}（{nearest['type']}），"
            f"中心深度={nearest['depth_m']:.3f} m",
            flush=True,
        )
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
