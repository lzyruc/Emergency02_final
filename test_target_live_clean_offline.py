"""离线验证实时清理目标框的通道、缓冲区和高度过滤。"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import unittest

import numpy as np

from channel_region import ChannelRegion


SCRIPT = Path(__file__).resolve().parent / "test_target_live_clean.py"


def load_functions():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    selected = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "measured_navigation_logical_pose",
            "select_cleaning_detections",
        }
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            selected["measured_navigation_logical_pose"],
            selected["select_cleaning_detections"],
        ],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "math": math,
        "CHANNEL_MAX_OBJECT_Z_M": 0.80,
        "center_depth_m": lambda depth_mm, bbox: 1.0,
    }
    exec(compile(ast.fix_missing_locations(module), str(SCRIPT), "exec"), namespace)
    return namespace


class TargetLiveCleanTests(unittest.TestCase):
    def test_task_outside_margin_is_selected_by_type(self) -> None:
        task = json.loads(
            (SCRIPT.parent / "config/task.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            task["channel_region"]["outside_ignore_distance_m_by_type"],
            {"chair": 0.3, "box": 0.3, "bin": 0.15, "filebox": 0.15},
        )

    def test_same_outer_point_keeps_chair_but_ignores_bin(self) -> None:
        class Deprojector:
            def deproject_many(self, samples, face_target=False):
                return np.zeros((len(samples), 7), dtype=np.float64)

            def transform_many_to_ground(self, points):
                return np.asarray([
                    [1.20, 0.50, 0.20],
                    [1.20, 0.50, 0.20],
                ])

        detections = [
            {
                "type": target_type,
                "confidence": 0.9,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "center_uv": [5.0, 5.0],
            }
            for target_type in ("chair", "bin")
        ]
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        selected = load_functions()["select_cleaning_detections"](
            detections,
            np.ones((12, 12), dtype=np.float32),
            Deprojector(),
            {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
            {
                "chair": ChannelRegion(polygon, outside_margin_m=0.3),
                "bin": ChannelRegion(polygon, outside_margin_m=0.15),
            },
        )
        self.assertEqual([detection["type"] for detection in selected], ["chair"])

    def test_live_loop_refreshes_measured_pose_before_filtering(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        live_loop = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
        )
        calls = [
            node.func.id
            for node in ast.walk(live_loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {
                "measured_navigation_logical_pose",
                "select_cleaning_detections",
            }
        ]
        self.assertIn("measured_navigation_logical_pose", calls)
        self.assertIn("select_cleaning_detections", calls)

    def test_only_inside_or_outer_margin_low_targets_are_selected(self) -> None:
        class Deprojector:
            def deproject_many(self, samples, face_target=False):
                return np.zeros((len(samples), 7), dtype=np.float64)

            def transform_many_to_ground(self, points):
                return np.asarray([
                    [0.50, 0.50, 0.20],
                    [1.20, 0.50, 0.20],
                    [1.30, 0.50, 0.20],
                    [0.50, 0.50, 0.81],
                ])

        detections = [
            {
                "type": name,
                "confidence": 0.9,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "center_uv": [5.0, 5.0],
            }
            for name in ("inside", "margin", "boundary", "high")
        ]
        region = ChannelRegion(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            outside_margin_m=0.3,
        )
        selected = load_functions()["select_cleaning_detections"](
            detections,
            np.ones((12, 12), dtype=np.float32),
            Deprojector(),
            {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
            {detection["type"]: region for detection in detections},
        )
        self.assertEqual(
            [detection["type"] for detection in selected],
            ["inside", "margin"],
        )

    def test_measured_slam_pose_is_converted_to_logical(self) -> None:
        class Navigation:
            def getPose(self):
                return ({"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0}, {})

            def getStateError(self):
                return {
                    "expected_map_pose": {
                        "x_m": 10.0,
                        "y_m": 20.0,
                        "yaw_rad": math.pi / 2.0,
                    },
                    "actual_map_pose": {
                        "x_m": 10.0,
                        "y_m": 21.0,
                        "yaw_rad": math.pi / 2.0,
                    },
                    "within_tolerance": False,
                }

        logical_pose, within_tolerance = load_functions()[
            "measured_navigation_logical_pose"
        ](Navigation())
        self.assertAlmostEqual(logical_pose["x_m"], 1.0)
        self.assertAlmostEqual(logical_pose["y_m"], 0.0)
        self.assertAlmostEqual(logical_pose["theta_deg"], 0.0)
        self.assertFalse(within_tolerance)


if __name__ == "__main__":
    unittest.main()
