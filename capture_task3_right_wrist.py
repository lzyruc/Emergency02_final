#!/usr/bin/env python3
"""采集一次任务3右腕RGB-D图像，不发送任何机器人控制命令。

运行：
    python capture_task3_right_wrist.py

输出目录为 ``output/task3_right_wrist/<时间戳>``：
``rgb_raw.png`` 是BGR彩色图，``depth_raw_mm.png`` 是毫米深度原图，
``depth_preview.png`` 仅用于快速查看有效深度分布。
"""

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


OUTPUT_ROOT = Path(__file__).resolve().parent / "output/task3_right_wrist"


def main() -> None:
    """创建唯一station，采集右腕一帧RGB-D并保存后关闭。"""
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    right_camera = config.get("camera", {})["right"]
    camera_name = right_camera["camera_name"].strip("/")
    # 当前RealSense驱动发布rect图，xrocs订阅raw图；仅重映射本进程的订阅名。
    sys.argv.extend(
        [
            "--ros-args",
            "-r",
            f"/{camera_name}/color/image_raw:=/{camera_name}/color/image_rect_raw",
        ]
    )
    station = TienYi2ProRos2Station(
        # 本脚本只采集图像，不初始化机器人和夹爪控制器。
        robot_dict={},
        camera_dict={"right": right_camera},
        hand_dict={},
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        bgr, depth_mm = read_wrist_camera(station, settings.RIGHT_CAMERA)
        output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True)

        valid_depth = depth_mm[(depth_mm > 0) & (depth_mm < 65535)]
        depth_preview = np.zeros(depth_mm.shape, dtype=np.uint8)
        if valid_depth.size:
            near_mm, far_mm = np.percentile(valid_depth, (2.0, 98.0))
            if far_mm > near_mm:
                depth_preview = np.uint8(
                    np.clip((depth_mm - near_mm) * 255.0 / (far_mm - near_mm), 0, 255)
                )
        depth_preview = cv2.applyColorMap(depth_preview, cv2.COLORMAP_TURBO)

        files = {
            "rgb_raw": output_dir / "rgb_raw.png",
            "depth_raw_mm": output_dir / "depth_raw_mm.png",
            "depth_preview": output_dir / "depth_preview.png",
        }
        for name, path in files.items():
            image = bgr if name == "rgb_raw" else (
                depth_mm if name == "depth_raw_mm" else depth_preview
            )
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"failed to write image: {path}")

        metadata = {
            "camera_key": settings.RIGHT_CAMERA["station_camera_key"],
            "color_encoding": "BGR uint8",
            "depth_unit": "mm",
            "image_size": [int(bgr.shape[1]), int(bgr.shape[0])],
            "valid_depth_pixels": int(valid_depth.size),
            "valid_depth_min_mm": int(valid_depth.min()) if valid_depth.size else None,
            "valid_depth_max_mm": int(valid_depth.max()) if valid_depth.size else None,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(output_dir, flush=True)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
