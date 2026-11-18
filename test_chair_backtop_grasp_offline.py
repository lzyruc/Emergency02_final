"""椅背顶端抓取运动分段的离线测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("chair_backtop_grasp.py")
camera_io = types.ModuleType("camera_io")
camera_io.read_wrist_camera = lambda station, config: None
previous_camera_io = sys.modules.get("camera_io")
sys.modules["camera_io"] = camera_io
spec = importlib.util.spec_from_file_location(
    "chair_backtop_grasp_subject",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if previous_camera_io is None:
    sys.modules.pop("camera_io", None)
else:
    sys.modules["camera_io"] = previous_camera_io


class ChairBacktopGraspTests(unittest.TestCase):
    def test_point_selection_avoids_dark_endpoint_and_prefers_near_middle(
        self,
    ) -> None:
        component = np.zeros((120, 80), dtype=bool)
        component[10:110, 35:45] = True
        gray = np.full(component.shape, 120, dtype=np.uint8)
        gray[10:20, 35:45] = 5
        gray[55:65, 35:45] = 50
        depth_mm = np.full(component.shape, 250, dtype=np.uint16)
        depth_mm[55:65, 35:45] = 210

        u, v = module._central_near_ridge_uv(
            component,
            gray,
            depth_mm,
            central_axis_fraction=0.60,
            depth_score_weight=0.55,
            center_score_weight=0.30,
            dark_score_weight=0.15,
            best_score_fraction=0.01,
        )

        self.assertGreaterEqual(v, 55)
        self.assertLess(v, 65)
        self.assertGreaterEqual(u, 35)
        self.assertLess(u, 45)

    def test_rejects_grasp_point_at_bottom_of_roi(self) -> None:
        depth_mm = np.zeros((200, 200), dtype=np.uint16)
        depth_mm[170:200, 0:200] = 220
        color = np.zeros((200, 200, 3), dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "ROI过低区域"):
            module.detect_chair_backtop_grasp(
                depth_mm,
                color,
                (0.0, 0.0, 1.0, 1.0),
                600.0,
                600.0,
            )

    def test_preserves_full_xy_alignment(self) -> None:
        first_delta, second_delta = module.plan_chair_backtop_grasp_steps(
            np.asarray([0.30, -0.40, -0.50], dtype=np.float64),
            approach_m=0.40,
            standoff_m=0.05,
            z_compensation_m=-0.15,
            second_stage_z_m=0.08,
        )

        np.testing.assert_allclose(first_delta, [0.30, -0.40, -0.17])
        np.testing.assert_allclose(second_delta, [0.0, 0.0, -0.08])

    def test_keeps_standoff_without_z_motion(self) -> None:
        first_delta, second_delta = module.plan_chair_backtop_grasp_steps(
            np.asarray([0.12, 0.08, 0.04], dtype=np.float64),
            approach_m=0.40,
            standoff_m=0.05,
            z_compensation_m=-0.15,
            second_stage_z_m=0.08,
        )

        np.testing.assert_allclose(first_delta, [0.12, 0.08, 0.0])
        np.testing.assert_allclose(second_delta, [0.0, 0.0, 0.0])

    def test_short_z_is_entirely_deferred_to_second_stage(self) -> None:
        first_delta, second_delta = module.plan_chair_backtop_grasp_steps(
            np.asarray([0.12, 0.08, -0.25], dtype=np.float64),
            approach_m=0.40,
            standoff_m=0.05,
            z_compensation_m=-0.15,
            second_stage_z_m=0.08,
        )

        np.testing.assert_allclose(first_delta, [0.12, 0.08, 0.0])
        np.testing.assert_allclose(second_delta, [0.0, 0.0, -0.05])


if __name__ == "__main__":
    unittest.main()
