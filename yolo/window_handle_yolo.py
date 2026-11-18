"""窗把手YOLO检测入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

__all__ = ["WindowHandleYolo"]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models/window_handle_yolo26n_20260818_best.pt"
)


class WindowHandleYolo:
    """模型加载一次；每帧返回按置信度降序排列的窗把手框。"""

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

    def detect(self, bgr_image: np.ndarray) -> list[dict[str, Any]]:
        """返回 ``confidence/bbox_xyxy/center_uv``，坐标单位为像素。"""
        image = np.asarray(bgr_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr_image 必须是 uint8 H×W×3 BGR 图像")
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
        detections = []
        for box, score in zip(boxes, scores, strict=True):
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            detections.append(
                {
                    "confidence": float(score),
                    "bbox_xyxy": [x_min, y_min, x_max, y_max],
                    "center_uv": [
                        0.5 * (x_min + x_max),
                        0.5 * (y_min + y_max),
                    ],
                }
            )
        return sorted(
            detections,
            key=lambda detection: detection["confidence"],
            reverse=True,
        )
