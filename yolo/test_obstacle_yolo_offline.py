"""ObstacleYolo 的无权重、无 GPU 离线测试。"""

from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


ultralytics = types.ModuleType("ultralytics")
ultralytics.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics)
ObstacleYolo = importlib.import_module("obstacle_yolo").ObstacleYolo


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self):
        self.xyxy = _Tensor(
            [
                [10.0, 20.0, 30.0, 60.0],
                [40.0, 50.0, 80.0, 90.0],
                [90.0, 10.0, 130.0, 50.0],
                [20.0, 100.0, 70.0, 160.0],
            ]
        )
        self.conf = _Tensor([0.9, 0.8, 0.7, 0.6])
        self.cls = _Tensor([0, 1, 2, 3])


class _Result:
    boxes = _Boxes()


class _Model:
    names = {0: "1", 1: "2", 2: "3", 3: "4"}

    def predict(self, **kwargs):
        assert kwargs["source"].shape == (480, 640, 3)
        assert kwargs["imgsz"] == 1280
        return [_Result()]


class ObstacleYoloTests(unittest.TestCase):
    def setUp(self):
        self.detector = ObstacleYolo(model=_Model())
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_class_mapping_and_bbox(self):
        detections = self.detector.detect(self.image)
        self.assertEqual(
            [item["type"] for item in detections],
            ["chair", "filebox", "bin", "box"],
        )
        self.assertEqual(detections[0]["bbox_xyxy"], [10.0, 20.0, 30.0, 60.0])
        self.assertEqual(detections[0]["center_uv"], [20.0, 40.0])

    def test_target_filter(self):
        detections = self.detector.detect(self.image, target_type="filebox")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["type"], "filebox")


if __name__ == "__main__":
    unittest.main()
