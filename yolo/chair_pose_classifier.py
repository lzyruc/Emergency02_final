"""裁剪椅子图像的 front/back/lean 分类入口。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

__all__ = ["ChairPoseClassifier"]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models/chair_pose_classifier_best.pt"
)
CHAIR_VIEWS = {"front", "back", "lean"}


class ChairPoseClassifier:
    """模型加载一次；输入椅子BGR裁剪图，返回姿态类别和置信度。"""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        *,
        image_size: int = 224,
        device: int | str = 0,
        half: bool = True,
        model: Any | None = None,
    ) -> None:
        self.image_size = int(image_size)
        self.device = device
        self.half = half
        self.model = (
            YOLO(str(model_path), task="classify")
            if model is None
            else model
        )
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.names = {
            int(index): str(name).strip().lower()
            for index, name in names.items()
        }

    def classify(self, chair_bgr_crop: np.ndarray) -> dict[str, Any] | None:
        """返回 ``view/confidence``；没有分类概率时返回 ``None``。"""
        image = np.asarray(chair_bgr_crop)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("chair_bgr_crop 必须是非空 uint8 HxWx3 BGR 图像")
        result = self.model.predict(
            source=image,
            imgsz=self.image_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        if result.probs is None:
            return None
        class_index = int(result.probs.top1)
        view = self.names.get(class_index, "")
        if view not in CHAIR_VIEWS:
            return None
        confidence = result.probs.top1conf
        if hasattr(confidence, "item"):
            confidence = confidence.item()
        return {"view": view, "confidence": float(confidence)}

    def classify_bbox(
        self,
        bgr_image: np.ndarray,
        bbox_xyxy: list[float],
        *,
        padding_ratio: float,
    ) -> dict[str, Any] | None:
        """将检测框每侧按宽高比例外扩、裁到图像边界后执行分类。"""
        image = np.asarray(bgr_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr_image 必须是 uint8 HxWx3 BGR 图像")
        x_min, y_min, x_max, y_max = (
            float(value) for value in bbox_xyxy
        )
        padding_x = (x_max - x_min) * padding_ratio
        padding_y = (y_max - y_min) * padding_ratio
        crop_left = max(0, int(math.floor(x_min - padding_x)))
        crop_top = max(0, int(math.floor(y_min - padding_y)))
        crop_right = min(
            image.shape[1],
            int(math.ceil(x_max + padding_x)),
        )
        crop_bottom = min(
            image.shape[0],
            int(math.ceil(y_max + padding_y)),
        )
        prediction = self.classify(
            image[crop_top:crop_bottom, crop_left:crop_right]
        )
        if prediction is None:
            return None
        return {
            **prediction,
            "crop_bbox_xyxy": [
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            ],
        }
