"""正放转椅扶手深度连通域算法的离线测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("chair_depth_grasp.py")
camera_io = types.ModuleType("camera_io")
camera_io.read_wrist_camera = lambda station, config: None
previous_camera_io = sys.modules.get("camera_io")
sys.modules["camera_io"] = camera_io
spec = importlib.util.spec_from_file_location("chair_depth_grasp_subject", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if previous_camera_io is None:
    sys.modules.pop("camera_io", None)
else:
    sys.modules["camera_io"] = previous_camera_io
detect_armrest_grasp = module.detect_armrest_grasp


def _parameters() -> dict:
    return {
        "min_depth_mm": 150,
        "max_depth_mm": 320,
        "close_kernel_px": 9,
        "min_component_area_px": 1000,
        "min_major_span_px": 80,
        "min_elongation_ratio": 1.5,
        "cross_section_band_px": 4,
        "side_sample_offset_px": 10,
        "min_side_jump_mm": 60.0,
        "min_grasp_width_m": 0.015,
        "max_grasp_width_m": 0.10,
        "stability_patch_px": 9,
        "min_patch_valid_ratio": 0.5,
        "max_patch_depth_range_mm": 80.0,
        "camera_fx_px": 650.0,
        "camera_fy_px": 650.0,
    }


def _armrest_depth() -> np.ndarray:
    depth = np.full((240, 320), 500, dtype=np.uint16)
    depth[30:220, 135:185] = 240
    depth[80:85, 150:155] = 0
    return depth


class _Deprojector:
    def deproject(self, sample, face_target=True):
        return np.asarray(
            [sample[2], sample[0] / 1000.0, sample[1] / 1000.0, 0, 0, 0, 1],
            dtype=np.float64,
        )


class ChairDepthGraspTests(unittest.TestCase):
    def test_selects_armrest_middle_cross_section(self) -> None:
        result = detect_armrest_grasp(
            _armrest_depth(),
            (0.20, 0.05, 0.80, 0.95),
            **_parameters(),
        )

        self.assertTrue(145 <= result["pixel_uv"][0] <= 175)
        self.assertTrue(100 <= result["pixel_uv"][1] <= 150)
        self.assertAlmostEqual(result["depth_m"], 0.240)
        self.assertGreaterEqual(result["width_m"], 0.015)
        self.assertGreater(result["elongation_ratio"], 1.5)

    def test_rejects_roi_without_near_armrest(self) -> None:
        with self.assertRaises(RuntimeError):
            detect_armrest_grasp(
                _armrest_depth(),
                (0.00, 0.05, 0.30, 0.95),
                **_parameters(),
            )

    def test_interface_masks_gripper_and_saves_diagnostics(self) -> None:
        depth = _armrest_depth()
        color = np.full((*depth.shape, 3), 180, dtype=np.uint8)
        original_reader = module.read_wrist_camera
        original_mask = module.gripper_mask
        original_config = module.settings.TASK["chair_grasp"]
        module.read_wrist_camera = lambda station, config: (color.copy(), depth.copy())
        static_mask = np.zeros(depth.shape, dtype=bool)
        static_mask[220:, :] = True
        module.gripper_mask = lambda shape: static_mask.copy()
        config = _parameters()
        config.pop("camera_fx_px")
        config.pop("camera_fy_px")
        config.update(
            {
                "left_roi_normalized_xyxy": [0.20, 0.05, 0.80, 0.95],
                "right_roi_normalized_xyxy": [0.20, 0.05, 0.80, 0.95],
                "grasp_target_x_offset_m": -0.08,
            }
        )
        module.settings.TASK["chair_grasp"] = config
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = module.recognize_chair_grasp(
                    object(),
                    "left",
                    _Deprojector(),
                    Path(directory),
                )
                for path in result["output_files"].values():
                    self.assertTrue(Path(path).is_file())
                mask = cv2.imread(
                    result["output_files"]["armrest_mask"],
                    cv2.IMREAD_GRAYSCALE,
                )
                self.assertGreater(np.count_nonzero(mask), 1000)
                self.assertAlmostEqual(
                    result["target_pose_waist_xyz_xyzw"][0],
                    result["depth_m"] - 0.08,
                )
        finally:
            module.read_wrist_camera = original_reader
            module.gripper_mask = original_mask
            module.settings.TASK["chair_grasp"] = original_config


if __name__ == "__main__":
    unittest.main()
