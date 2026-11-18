"""椅子、文件筐、垃圾桶和纸箱的 Ultralytics 二维检测入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

__all__ = ["ObstacleYolo"]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models/emerency_yolo26l_20260804_cls1234_best.pt"
)
MODEL_NAME_TO_TYPE = {
    "1": "chair",
    "2": "filebox",
    "3": "bin",
    "4": "box",
}


class ObstacleYolo:
    """模型加载一次；每帧返回四类物体的二维检测框。"""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        *,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 1280,
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
        self.names = {int(key): str(value) for key, value in names.items()}

    def detect(
        self,
        bgr_image: np.ndarray,
        *,
        target_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回 ``type/confidence/bbox_xyxy/center_uv``，坐标单位为像素。"""
        image = np.asarray(bgr_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr_image 必须是 uint8 H×W×3 BGR 图像")
        if target_type is not None and target_type not in MODEL_NAME_TO_TYPE.values():
            raise ValueError("target_type 必须是 chair、filebox、bin 或 box")

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
            obstacle_type = MODEL_NAME_TO_TYPE.get(self.names.get(int(class_index), ""))
            if obstacle_type is None or (
                target_type is not None and obstacle_type != target_type
            ):
                continue
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            detections.append(
                {
                    "type": obstacle_type,
                    "confidence": float(score),
                    "bbox_xyxy": [x_min, y_min, x_max, y_max],
                    "center_uv": [
                        0.5 * (x_min + x_max),
                        0.5 * (y_min + y_max),
                    ],
                }
            )
        return detections
