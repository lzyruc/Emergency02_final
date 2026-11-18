"""倾斜转椅侧框深度算法的离线测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("chair_side_grasp.py")
camera_io = types.ModuleType("camera_io")
camera_io.read_wrist_camera = lambda station, config: None
previous_camera_io = sys.modules.get("camera_io")
sys.modules["camera_io"] = camera_io
spec = importlib.util.spec_from_file_location(
    "chair_side_grasp_subject",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if previous_camera_io is None:
    sys.modules.pop("camera_io", None)
else:
    sys.modules["camera_io"] = previous_camera_io


def _depth_with_vertical_rail() -> np.ndarray:
    depth = np.full((480, 640), 800, dtype=np.uint16)
    depth[130:350, 300:340] = 300
    return depth


class _Deprojector:
    def deproject(self, sample, face_target=True):
        return np.asarray(
            [sample[2], sample[0] / 1000.0, sample[1] / 1000.0, 0, 0, 0, 1],
            dtype=np.float64,
        )


class ChairSideGraspTests(unittest.TestCase):
    def test_detects_vertical_side_rail(self) -> None:
        result = module.detect_chair_side_grasp(
            _depth_with_vertical_rail(),
            (0.12, 0.08, 0.92, 0.72),
            module.settings.TASK["chair_side_grasp"],
            600.0,
            600.0,
        )

        self.assertEqual(result["grasp_mode"], "side_rail")
        self.assertGreaterEqual(result["axis_from_horizontal_deg"], 35.0)
        self.assertAlmostEqual(result["depth_m"], 0.3)
        self.assertGreaterEqual(result["width_m"], 0.015)

    def test_rejects_horizontal_back_board(self) -> None:
        depth = np.full((480, 640), 800, dtype=np.uint16)
        depth[220:260, 180:460] = 300
        with self.assertRaises(RuntimeError):
            module.detect_chair_side_grasp(
                depth,
                (0.12, 0.08, 0.92, 0.72),
                module.settings.TASK["chair_side_grasp"],
                600.0,
                600.0,
            )

    def test_interface_saves_diagnostics_and_applies_waist_x_offset(self) -> None:
        depth = _depth_with_vertical_rail()
        color = np.full((*depth.shape, 3), 180, dtype=np.uint8)
        original_reader = module.read_wrist_camera
        original_mask = module.gripper_mask
        module.read_wrist_camera = lambda station, config: (
            color.copy(),
            depth.copy(),
        )
        module.gripper_mask = lambda shape: np.zeros(shape, dtype=bool)
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = module.recognize_chair_side_grasp(
                    object(),
                    "left",
                    _Deprojector(),
                    Path(directory),
                )
                self.assertEqual(result["grasp_mode"], "side_rail")
                for path in result["output_files"].values():
                    self.assertTrue(Path(path).is_file())
                mask = cv2.imread(
                    result["output_files"]["side_mask"],
                    cv2.IMREAD_GRAYSCALE,
                )
                self.assertGreater(np.count_nonzero(mask), 700)
                self.assertAlmostEqual(
                    result["target_pose_waist_xyz_xyzw"][0],
                    result["depth_m"] - 0.08,
                )
        finally:
            module.read_wrist_camera = original_reader
            module.gripper_mask = original_mask


if __name__ == "__main__":
    unittest.main()
