#!/usr/bin/env python3
"""按 Enter 采集左腕 RGB-D 原始帧，不发送任何机器人控制命令。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

import settings
from camera_io import read_wrist_camera
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


OUTPUT_ROOT = Path(__file__).resolve().parent / "output/left_wrist_capture"


def save_frame(station: TienYi2ProRos2Station) -> Path:
    """采集一帧并保存BGR、毫米深度、深度预览和元数据。"""
    bgr, depth_mm = read_wrist_camera(station, settings.LEFT_CAMERA)
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)

    valid = (depth_mm > 0) & (depth_mm < 65535)
    valid_depth = depth_mm[valid]
    depth_gray = np.zeros(depth_mm.shape, dtype=np.uint8)
    if valid_depth.size:
        near_mm, far_mm = np.percentile(valid_depth, (2.0, 98.0))
        if far_mm > near_mm:
            depth_gray[valid] = np.uint8(
                np.clip(
                    (depth_mm[valid].astype(np.float32) - near_mm)
                    * 255.0
                    / (far_mm - near_mm),
                    0.0,
                    255.0,
                )
            )
    depth_preview = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)
    depth_preview[~valid] = 0

    files = {
        "rgb_raw.png": bgr,
        "depth_raw_mm.png": depth_mm,
        "depth_preview.png": depth_preview,
    }
    for name, image in files.items():
        path = output_dir / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"图像保存失败：{path}")

    metadata = {
        "camera_key": settings.LEFT_CAMERA["station_camera_key"],
        "color_encoding": "BGR uint8",
        "depth_encoding": "uint16",
        "depth_unit": "mm",
        "image_width": int(bgr.shape[1]),
        "image_height": int(bgr.shape[0]),
        "valid_depth_pixels": int(valid_depth.size),
        "valid_depth_min_mm": int(valid_depth.min()) if valid_depth.size else None,
        "valid_depth_max_mm": int(valid_depth.max()) if valid_depth.size else None,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    left_camera = config.get("camera", {})["left"]
    camera_name = left_camera["camera_name"].strip("/")
    # 当前 RealSense 发布 rect 图，仅重映射本进程的左腕订阅名。
    sys.argv.extend(
        [
            "--ros-args",
            "-r",
            f"/{camera_name}/color/image_raw:=/{camera_name}/color/image_rect_raw",
        ]
    )
    station = TienYi2ProRos2Station(
        robot_dict={},
        camera_dict={"left": left_camera},
        hand_dict={},
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        while True:
            command = input("按 Enter 采集一帧左腕 RGB-D；输入 q 后回车退出：")
            if command.strip().lower() == "q":
                break
            output_dir = save_frame(station)
            print(f"左腕 RGB-D 已保存：{output_dir}", flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
