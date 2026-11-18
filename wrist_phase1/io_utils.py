"""RGB-D 文件与腕部相机内参加载。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy", "width", "height")
_DEFAULT_INVALID_DEPTH_VALUES = (0, 65535)


def load_depth(path: str | Path) -> np.ndarray:
    """加载单通道 16-bit 原始深度图。

    返回值保持相机原始单位不变；米制换算仅由内参中的
    ``depth_scale_m`` 决定。
    """

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {path}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(
            "深度图必须是单通道 uint16 (H,W)，"
            f"实际为 dtype={depth.dtype}, shape={depth.shape}"
        )
    return depth


def load_intrinsics(path: str | Path) -> dict[str, Any]:
    """加载并校验腕部相机内参。

    支持字段直接位于 JSON 顶层，或六个成像内参位于 ``color`` 对象中。
    ``depth_scale_m`` 必须由配置明确给出，绝不根据相机类型或深度数值
    自动猜测。
    """

    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到内参文件: {config_path}") from None
    if not isinstance(data, dict):
        raise ValueError("intrinsics.json 顶层必须是 JSON object")
    return _normalise_intrinsics(data)


def _normalise_intrinsics(data: Mapping[str, Any]) -> dict[str, Any]:
    """将已加载配置规范为核心模型使用的扁平相机字典。"""

    if not isinstance(data, Mapping):
        raise TypeError("intrinsics 必须是 dict")

    color = data.get("color")
    if color is None:
        source: Mapping[str, Any] = data
    elif isinstance(color, Mapping):
        source = color
    else:
        raise ValueError("intrinsics['color'] 必须是 JSON object")

    missing = [key for key in _INTRINSIC_KEYS if key not in source]
    if missing:
        raise ValueError(f"intrinsics 缺少字段: {', '.join(missing)}")

    scale_marker = object()
    depth_scale = data.get(
        "depth_scale_m",
        source.get("depth_scale_m", scale_marker),
    )
    if depth_scale is scale_marker:
        raise ValueError(
            "intrinsics 缺少明确的 depth_scale_m；"
            "本包 D405 样例配置应为 0.0001 m/unit"
        )

    result: dict[str, Any] = {
        "fx": _finite_float(source["fx"], "fx"),
        "fy": _finite_float(source["fy"], "fy"),
        "cx": _finite_float(source["cx"], "cx"),
        "cy": _finite_float(source["cy"], "cy"),
        "width": _positive_int(source["width"], "width"),
        "height": _positive_int(source["height"], "height"),
        "depth_scale_m": _finite_float(depth_scale, "depth_scale_m"),
    }
    if result["fx"] <= 0.0 or result["fy"] <= 0.0:
        raise ValueError("intrinsics 的 fx 和 fy 必须大于 0")
    if result["depth_scale_m"] <= 0.0:
        raise ValueError("intrinsics 的 depth_scale_m 必须大于 0")

    invalid_values = data.get(
        "invalid_depth_values",
        source.get(
            "invalid_depth_values",
            _DEFAULT_INVALID_DEPTH_VALUES,
        ),
    )
    result["invalid_depth_values"] = _invalid_depth_values(invalid_values)
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"intrinsics['{name}'] 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"intrinsics['{name}'] 必须是有限数值"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"intrinsics['{name}'] 必须是有限数值")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"intrinsics['{name}'] 必须是正整数")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"intrinsics['{name}'] 必须是正整数") from None
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"intrinsics['{name}'] 必须是正整数")
    return int(number)


def _invalid_depth_values(values: Any) -> list[int]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, (list, tuple))
    ):
        raise ValueError("invalid_depth_values 必须是整数数组")

    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "invalid_depth_values 中的值必须是 0..65535 的整数"
            )
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "invalid_depth_values 中的值必须是 0..65535 的整数"
            ) from None
        if (
            not math.isfinite(number)
            or not number.is_integer()
            or not 0 <= number <= 65535
        ):
            raise ValueError(
                "invalid_depth_values 中的值必须是 0..65535 的整数"
            )
        integer = int(number)
        if integer not in result:
            result.append(integer)
    return result
