"""统一的相机读取：重试取帧，并解码成 BGR 彩色图 + 毫米深度图。"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

import settings
from xrocs.utils.camera_utils import CameraUtils


def read_camera(
    station: Any,
    camera_key: str,
    *,
    color_encoding: str = "bgr",
    depth_unit: str = "mm",
    attempts: int = 5,
    interval_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """重试读取一路相机，返回 (BGR uint8, 毫米深度 uint16)；重试用尽抛 RuntimeError。"""
    for attempt in range(attempts):
        images, depths = station.get_camera_state()
        color, depth = images[camera_key], depths[camera_key]
        if color is not None and depth is not None:
            break
        if attempt < attempts - 1:
            time.sleep(interval_s)
    else:
        raise RuntimeError(
            f"RGB-D frame is unavailable after {attempts} attempts "
            f"for camera {camera_key!r}"
        )
    color = np.asarray(color)
    depth = np.asarray(depth)
    if color.ndim == 1:
        color = CameraUtils.decode_color_image(color, output_format="bgr")
    elif color_encoding == "rgb":
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    if depth.ndim == 1:
        depth = CameraUtils.decode_depth_image(depth)
    elif depth_unit == "m":
        depth = np.rint(depth.astype(np.float64) * 1000.0)
    return np.ascontiguousarray(color), depth.astype(np.uint16)


def read_head_camera(station: Any) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """读取头部相机；重试用尽返回 (None, None)，由调用方按未识别处理。"""
    station_config = settings.TASK["station"]
    try:
        return read_camera(
            station,
            station_config["head_camera_key"],
            color_encoding=station_config["head_color_encoding"],
            depth_unit=station_config["head_depth_unit"],
        )
    except RuntimeError as exc:
        print(f"未获取到头部相机图像：{exc}", flush=True)
        return None, None


def read_wrist_camera(station: Any, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """按腕部相机标定读取左/右腕相机；重试用尽抛 RuntimeError。"""
    return read_camera(
        station,
        config["station_camera_key"],
        color_encoding=config["color_encoding"],
        depth_unit=config["depth_unit"],
    )
