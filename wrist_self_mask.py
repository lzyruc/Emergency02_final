"""加载比赛固定腕部夹爪遮挡区域，供各RGB-D规则识别器复用。

输入为当前腕部图像的 ``(height, width)``，输出同尺寸布尔mask；True表示
机器人自身张开夹爪区域，调用方必须从颜色和/或深度候选中排除。

最小调用：``mask = gripper_mask(depth_mm.shape); depth_mm[mask] = 0``。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


GRIPPER_MASK_PATH = Path(__file__).resolve().parent / "config/gripper_mask.png"
_GRIPPER_MASK = cv2.imread(str(GRIPPER_MASK_PATH), cv2.IMREAD_GRAYSCALE)
if _GRIPPER_MASK is None:
    raise FileNotFoundError(GRIPPER_MASK_PATH)


def gripper_mask(image_shape: tuple[int, int]) -> np.ndarray:
    """返回与腕部RGB-D对齐的静态张开夹爪布尔mask。"""
    height, width = image_shape
    mask = _GRIPPER_MASK
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0
