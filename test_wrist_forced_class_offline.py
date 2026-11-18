"""头部类别强制腕部检测与调试图保存的离线测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import wrist_target
from wrist_phase1 import fast_pipeline


class WristForcedClassTests(unittest.TestCase):
    def test_head_class_bypasses_wrist_router(self):
        rgb = np.zeros((720, 1280, 3), np.uint8)
        depth_m = np.full((720, 1280), np.nan, np.float32)
        camera = {
            "width": 1280,
            "height": 720,
            "fx": 658.0,
            "fy": 658.0,
            "cx": 640.0,
            "cy": 360.0,
        }
        with (
            patch.object(
                fast_pipeline,
                "_route",
                side_effect=AssertionError("强制类别时不应运行四分类路由"),
            ),
            patch.object(fast_pipeline, "_run_selected_detector", return_value=[]),
        ):
            for target_class in ("basket", "chair", "trash_can", "box"):
                _, _, debug = fast_pipeline.run_fast_pipeline(
                    rgb,
                    depth_m,
                    camera,
                    target_class=target_class,
                )
                self.assertEqual(debug["routed_class"], target_class)
                self.assertTrue(debug["forced_by_head_yolo"])

    def test_rgb_depth_and_annotation_are_saved(self):
        rgb = np.zeros((720, 1280, 3), np.uint8)
        depth_mm = np.zeros((720, 1280), np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                wrist_target,
                "WRIST_OUTPUT_DIR",
                Path(directory),
            ):
                prefix = wrist_target._save_wrist_images(
                    "filebox",
                    "left",
                    rgb,
                    depth_mm,
                    {"primary": None, "result": None},
                )
            self.assertEqual(len(list(Path(directory).glob(f"{prefix.name}_*"))), 3)


if __name__ == "__main__":
    unittest.main()
