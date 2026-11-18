"""加载深度图 + 内参"""
import json
import cv2
import numpy as np


def load_depth(path):
    """加载 16-bit 深度图 (毫米单位), 返回 uint16 numpy array"""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"深度图应该是 uint16, 得到 {depth.dtype}")
    return depth


def load_intrinsics(path):
    """加载 intrinsics.json, 返回 dict (color 通道的内参)
    因为深度已经对齐到 color 视角
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    color = data.get("color", data)  # 兼容单一相机的情况
    return {
        "fx": float(color["fx"]),
        "fy": float(color["fy"]),
        "cx": float(color["cx"]),
        "cy": float(color["cy"]),
        "width": int(color["width"]),
        "height": int(color["height"]),
    }
