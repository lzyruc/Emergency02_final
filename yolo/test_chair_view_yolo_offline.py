"""ChairViewYolo 的无权重、无 GPU 离线测试。"""

from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


ultralytics = types.ModuleType("ultralytics")
ultralytics.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics)
ChairViewYolo = importlib.import_module("chair_view_yolo").ChairViewYolo


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
            [[10.0, 20.0, 110.0, 220.0], [120.0, 30.0, 210.0, 200.0]]
        )
        self.conf = _Tensor([0.91, 0.72])
        self.cls = _Tensor([2, 0])


class _Result:
    boxes = _Boxes()


class _Model:
    # 故意使用非固定类别顺序，验证代码按model.names动态映射。
    names = {0: "lean", 1: "back", 2: "front"}

    def predict(self, **kwargs):
        assert kwargs["source"].shape == (360, 640, 3)
        assert kwargs["imgsz"] == 640
        return [_Result()]


class ChairViewYoloTests(unittest.TestCase):
    def test_dynamic_name_mapping(self):
        detector = ChairViewYolo(model=_Model())
        detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertEqual([item["view"] for item in detections], ["front", "lean"])
        self.assertEqual(
            detections[0]["bbox_xyxy"],
            [10.0, 20.0, 110.0, 220.0],
        )

    def test_invalid_image_rejected(self):
        detector = ChairViewYolo(model=_Model())
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((360, 640), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
