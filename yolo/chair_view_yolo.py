"""转椅正面、背面和侧倒状态的 Ultralytics 二维检测入口。

最小调用示例::

    detector = ChairViewYolo()
    detections = detector.detect(chair_bgr_crop)

输入是 ``uint8 HxWx3`` BGR 图像；输出项包含 ``view``（front/back/lean）、
``confidence`` 和像素坐标 ``bbox_xyxy``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

__all__ = ["ChairViewYolo"]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models/chair_yolo26l_200e_best.pt"
)
CHAIR_VIEWS = {"front", "back", "lean"}


class ChairViewYolo:
    """模型加载一次；每帧返回转椅视角检测框。"""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        *,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: int | str = 0,
        half: bool = True,
        model: Any | None = None,
    ) -> None:
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.image_size = int(image_size)
        self.device = device
        self.half = half
        self.model = YOLO(str(model_path), task="detect") if model is None else model
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.names = {
            int(index): str(name).strip().lower()
            for index, name in names.items()
        }

    def detect(self, bgr_image: np.ndarray) -> list[dict[str, Any]]:
        """返回 ``view/confidence/bbox_xyxy``，框坐标单位为输入图像像素。"""
        image = np.asarray(bgr_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr_image 必须是 uint8 HxWx3 BGR 图像")
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(np.int64)
        detections = []
        for box, score, class_index in zip(boxes, scores, classes, strict=True):
            view = self.names.get(int(class_index), "")
            if view not in CHAIR_VIEWS:
                continue
            detections.append(
                {
                    "view": view,
                    "confidence": float(score),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )
        return detections
