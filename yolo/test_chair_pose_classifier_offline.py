"""ChairPoseClassifier 的无权重、无 GPU 离线测试。"""

from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


ultralytics = types.ModuleType("ultralytics")
ultralytics.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics)
ChairPoseClassifier = importlib.import_module(
    "chair_pose_classifier"
).ChairPoseClassifier


class _Confidence:
    def __init__(self, value: float):
        self.value = value

    def item(self) -> float:
        return self.value


class _Probs:
    top1 = 1
    top1conf = _Confidence(0.91)


class _Result:
    probs = _Probs()


class _Model:
    names = {0: "back", 1: "front", 2: "lean"}

    def __init__(self, expected_shape=(180, 240, 3)):
        self.expected_shape = expected_shape

    def predict(self, **kwargs):
        assert kwargs["source"].shape == self.expected_shape
        assert kwargs["imgsz"] == 224
        return [_Result()]


class ChairPoseClassifierTests(unittest.TestCase):
    def test_classification_result(self):
        classifier = ChairPoseClassifier(model=_Model())
        result = classifier.classify(
            np.zeros((180, 240, 3), dtype=np.uint8)
        )
        self.assertEqual(result["view"], "front")
        self.assertAlmostEqual(result["confidence"], 0.91)

    def test_empty_crop_rejected(self):
        classifier = ChairPoseClassifier(model=_Model())
        with self.assertRaises(ValueError):
            classifier.classify(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_bbox_padding_and_image_boundary(self):
        classifier = ChairPoseClassifier(
            model=_Model(expected_shape=(100, 120, 3))
        )
        result = classifier.classify_bbox(
            np.zeros((100, 200, 3), dtype=np.uint8),
            [0.0, 10.0, 100.0, 90.0],
            padding_ratio=0.20,
        )
        self.assertEqual(result["crop_bbox_xyxy"], [0, 0, 120, 100])


if __name__ == "__main__":
    unittest.main()
