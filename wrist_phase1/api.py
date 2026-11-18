"""与头部相机交付包一致的腕部相机公共接口。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import _normalise_intrinsics, load_intrinsics
from .model import WristRGBDModel


PIPELINE_VERSION = "phase1_wrist_rgbd_fast_v1"
_EMPTY_RESULT = {"class": None, "x": None, "y": None}
_INPUT_KEYS = frozenset(("rgb", "depth"))


class Phase1Model:
    """单帧腕部 RGB-D 识别的团队接口。

    用法::

        model = Phase1Model(load_intrinsics("intrinsics.json"))
        result = model.detect({
            "rgb": rgb_bgr,
            "depth": depth_raw,
        })

    ``detect`` 始终返回且仅返回 ``class/x/y`` 三个字段。验收程序可在
    成功推理后读取 ``last_frame_result`` 获取类别框与抓取几何详情。
    """

    def __init__(self, intrinsics: dict[str, Any] | str | Path):
        if isinstance(intrinsics, (str, Path)):
            camera = load_intrinsics(intrinsics)
        elif isinstance(intrinsics, dict):
            camera = _normalise_intrinsics(intrinsics)
        else:
            raise TypeError(
                "intrinsics 必须是内参 dict 或 intrinsics.json 路径"
            )

        self.intrinsics = dict(camera)
        self._model = WristRGBDModel(self.intrinsics)
        self.last_frame_result: dict[str, Any] | None = None

    def detect(self, img: dict[str, np.ndarray]) -> dict[str, Any]:
        """识别一帧，并返回主目标的类别与唯一候选像素。

        Args:
            img: 必须恰好包含 ``rgb`` 与 ``depth``。RGB 是 OpenCV BGR
                ``uint8 (H,W,3)``；depth 是对齐到 RGB 的原始
                ``uint16 (H,W)`` 深度。
        """

        self.last_frame_result = None
        rgb_bgr, depth_raw = self._validate_img(img)
        detailed = self._model.detect_detailed(
            {
                "rgb": rgb_bgr,
                "depth": depth_raw,
            }
        )
        self.last_frame_result = detailed
        return self._public_result(detailed, rgb_bgr.shape[:2])

    def detect_for_class(
        self,
        img: dict[str, np.ndarray],
        target_class: str,
    ) -> dict[str, Any]:
        """跳过四分类路由，直接运行指定类别的抓取点检测器。

        ``target_class`` 必须是 ``box/chair/basket/trash_can`` 之一；返回格式
        与 :meth:`detect` 相同，仍然只有 ``class/x/y`` 三个字段。
        """
        self.last_frame_result = None
        rgb_bgr, depth_raw = self._validate_img(img)
        detailed = self._model.detect_detailed(
            {
                "rgb": rgb_bgr,
                "depth": depth_raw,
            },
            target_class=target_class,
        )
        self.last_frame_result = detailed
        return self._public_result(detailed, rgb_bgr.shape[:2])

    def _validate_img(
        self,
        img: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(img, dict):
            raise TypeError("img 必须是包含 rgb 和 depth 的 dict")

        keys = set(img)
        missing = sorted(_INPUT_KEYS - keys)
        extra = keys - _INPUT_KEYS
        if missing:
            raise ValueError(
                "img 必须同时包含 rgb 和 depth；"
                f"缺少: {', '.join(missing)}"
            )
        if extra:
            raise ValueError(
                "公共接口仅接受 rgb 和 depth；"
                "不支持额外字段: "
                f"{', '.join(sorted(repr(key) for key in extra))}"
            )

        rgb_bgr = np.asarray(img["rgb"])
        depth_raw = np.asarray(img["depth"])
        if (
            rgb_bgr.dtype != np.uint8
            or rgb_bgr.ndim != 3
            or rgb_bgr.shape[2] != 3
        ):
            raise ValueError("img['rgb'] 必须是 uint8 (H,W,3) BGR 图")
        if depth_raw.dtype != np.uint16 or depth_raw.ndim != 2:
            raise ValueError(
                "img['depth'] 必须是 uint16 (H,W) 原始深度图"
            )
        if rgb_bgr.shape[:2] != depth_raw.shape:
            raise ValueError("RGB 与 depth 必须同分辨率并逐像素对齐")

        height, width = depth_raw.shape
        expected_width = self.intrinsics["width"]
        expected_height = self.intrinsics["height"]
        if width != expected_width or height != expected_height:
            raise ValueError(
                "输入分辨率与 intrinsics 不一致："
                f"输入={width}x{height}, "
                f"内参={expected_width}x{expected_height}"
            )
        return rgb_bgr, depth_raw

    @staticmethod
    def _public_result(
        detailed: Any,
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        if not isinstance(detailed, dict):
            raise RuntimeError("核心模型返回了无效的详细结果")
        raw = detailed.get("result")
        if raw is None:
            return dict(_EMPTY_RESULT)
        if not isinstance(raw, dict):
            raise RuntimeError("核心模型返回了无效的公共结果")

        object_class = raw.get("class")
        x = raw.get("x")
        y = raw.get("y")
        if object_class is None:
            if x is not None or y is not None:
                raise RuntimeError("空类别不能带有坐标")
            return dict(_EMPTY_RESULT)
        if not isinstance(object_class, str) or not object_class:
            raise RuntimeError("核心模型返回了无效类别")

        if x is None or y is None:
            if x is not None or y is not None:
                raise RuntimeError("x/y 必须同时有值或同时为 None")
            return {"class": object_class, "x": None, "y": None}

        x_int = Phase1Model._coordinate(x, "x")
        y_int = Phase1Model._coordinate(y, "y")
        height, width = image_shape
        if not 0 <= x_int < width or not 0 <= y_int < height:
            raise RuntimeError(
                f"核心模型返回了越界坐标: ({x_int}, {y_int})"
            )
        return {"class": object_class, "x": x_int, "y": y_int}

    @staticmethod
    def _coordinate(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError(f"核心模型返回了无效 {name} 坐标")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError(
                f"核心模型返回了无效 {name} 坐标"
            ) from None
        if not math.isfinite(number) or not number.is_integer():
            raise RuntimeError(f"核心模型返回了无效 {name} 坐标")
        return int(number)
