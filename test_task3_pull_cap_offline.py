#!/usr/bin/env python3
"""Offline tests for task3 pull-cap detection and split TCP motion."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np

from task3_pull_cap_detector import PullCapDetector


MODULE_PATH = Path(__file__).with_name("task3_pull_cap_grasp.py")


class FakeStation:
    def __init__(self) -> None:
        self.actions: list[dict] = []
        self.right_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]

    def get_robot_state(self) -> dict:
        return {
            "arm_pose": {
                "right": self.right_pose,
            }
        }

    def execute_action(self, action: dict) -> None:
        self.actions.append(action)


def load_grasp_module(
    motion_results: list[bool],
    *,
    with_slots: bool = False,
    detector_failure: bool = False,
) -> tuple[types.ModuleType, list[dict]]:
    settings = types.ModuleType("settings")
    settings.RIGHT_CAMERA = {
        "station_camera_key": "right",
        "camera_matrix": np.eye(3).tolist(),
        "extrinsics_cam_to_right_tcp_link_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    settings.RUNTIME_DIR = Path("runtime")
    settings.TASK = {
        "task3_pull_cap": {
            "right_roi_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
            "final_grasp_offset_right_tcp_m": [0.0, 0.0, 0.18],
            "slot_radius_scale": 1.0,
            "slot_insert_z_tcp_m": -0.07,
        },
        "hands": {"right_close_position": 1.0},
    }

    calls: list[dict] = []
    arm_motion = types.ModuleType("arm_motion")

    def moveit_cartesians(station: object, **kwargs: object) -> bool:
        calls.append(kwargs)
        succeeded = motion_results[len(calls) - 1]
        if succeeded:
            station.right_pose = list(kwargs["right"])
        return succeeded

    arm_motion.moveit_cartesians = moveit_cartesians
    camera_io = types.ModuleType("camera_io")
    camera_io.read_wrist_camera = lambda station, config: (
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.full((20, 20), 300, dtype=np.uint16),
    )
    detector = types.ModuleType("task3_pull_cap_detector")

    class FakeDetector:
        def __init__(self, camera_matrix: object) -> None:
            pass

        def detect(self, *args: object, **kwargs: object) -> object:
            if detector_failure:
                raise RuntimeError("small cylindrical cap circle was not found")
            slots = (
                (
                    types.SimpleNamespace(
                        direction_quarter_turns_from_cap=1,
                        pixel_uv=(15, 10),
                        point_camera_m=np.asarray([0.2, 0.1, 0.3]),
                    ),
                    types.SimpleNamespace(
                        direction_quarter_turns_from_cap=2,
                        pixel_uv=(10, 15),
                        point_camera_m=np.asarray([0.05, -0.02, 0.3]),
                    ),
                    types.SimpleNamespace(
                        direction_quarter_turns_from_cap=3,
                        pixel_uv=(5, 10),
                        point_camera_m=np.asarray([-0.1, 0.1, 0.3]),
                    ),
                )
                if with_slots
                else ()
            )
            return types.SimpleNamespace(
                cap_pixel_uv=(10, 10),
                large_circle_uvr=(10, 14, 5),
                cap_circle_uvr=(10, 5, 2),
                confidence=0.9,
                depth_m=0.3,
                point_camera_m=np.asarray([0.1, -0.2, 0.3]),
                selection_strategy="cap_only",
                slot_candidates=slots,
            )

    detector.PullCapDetector = FakeDetector
    detector.PullCapDetection = object
    detector.save_pull_cap_debug_images = lambda *args: None
    wrist_mask = types.ModuleType("wrist_self_mask")
    wrist_mask.gripper_mask = lambda shape: np.zeros(shape, dtype=bool)
    sys.modules.update(
        {
            "settings": settings,
            "arm_motion": arm_motion,
            "camera_io": camera_io,
            "task3_pull_cap_detector": detector,
            "wrist_self_mask": wrist_mask,
        }
    )
    spec = importlib.util.spec_from_file_location("task3_pull_cap_grasp_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.time = types.SimpleNamespace(sleep=lambda _: None)
    module.input = lambda _: ""
    return module, calls


class PullCapDetectorTest(unittest.TestCase):
    def test_large_circle_prefers_rgb_edge_over_larger_radius(self) -> None:
        detector = PullCapDetector(np.eye(3))
        cap = {"circle": (662.0, 192.0, 36.0), "depth_mm": 157.0}
        correct = {
            "circle": (693.0, 371.0, 130.0),
            "depth_mm": 162.0,
            "dark_ratio": 1.0,
            "depth_ratio": 1.0,
            "edge_ratio": 0.094,
            "rgb_contrast": 73.0,
        }
        larger_but_wrong = {
            "circle": (669.0, 404.0, 139.0),
            "depth_mm": 163.0,
            "dark_ratio": 0.99,
            "depth_ratio": 0.99,
            "edge_ratio": 0.040,
            "rgb_contrast": 66.0,
        }
        self.assertIs(
            detector._large_for_cap(cap, [larger_but_wrong, correct]),
            correct,
        )

    def test_two_stage_circle_detection(self) -> None:
        bgr = np.full((720, 1280, 3), 170, dtype=np.uint8)
        depth = np.full((720, 1280), 1000, dtype=np.uint16)
        cv2.circle(bgr, (700, 380), 125, (25, 25, 25), -1)
        cv2.circle(depth, (700, 380), 125, 160, -1)
        cv2.circle(bgr, (665, 195), 36, (20, 20, 20), -1)
        cv2.circle(depth, (665, 195), 36, 157, -1)
        detection = PullCapDetector(
            np.asarray(
                [[657.0, 0.0, 640.0], [0.0, 657.0, 338.0], [0.0, 0.0, 1.0]]
            )
        ).detect(bgr, depth)
        self.assertLess(np.linalg.norm(np.asarray(detection.cap_pixel_uv) - [665, 195]), 5)
        self.assertLess(
            np.linalg.norm(np.asarray(detection.large_circle_uvr[:2]) - [700, 380]),
            5,
        )
        self.assertAlmostEqual(detection.depth_m, 0.157, places=3)
        self.assertEqual(len(detection.slot_candidates), 3)
        self.assertEqual(
            detection.selection_strategy,
            "cap_metric_rgbd_with_optional_large_support",
        )

    def test_cap_succeeds_without_large_circle(self) -> None:
        bgr = np.full((720, 1280, 3), 170, dtype=np.uint8)
        depth = np.full((720, 1280), 1000, dtype=np.uint16)
        cv2.circle(bgr, (665, 195), 36, (20, 20, 20), -1)
        cv2.circle(depth, (665, 195), 36, 157, -1)
        detection = PullCapDetector(
            np.asarray(
                [[657.0, 0.0, 640.0], [0.0, 657.0, 338.0], [0.0, 0.0, 1.0]]
            )
        ).detect(bgr, depth)
        self.assertLess(np.linalg.norm(np.asarray(detection.cap_pixel_uv) - [665, 195]), 5)
        self.assertIsNone(detection.large_circle_uvr)
        self.assertEqual(detection.slot_candidates, ())
        self.assertEqual(detection.selection_strategy, "cap_metric_rgbd_without_large_circle")


class PullCapGraspTest(unittest.TestCase):
    def test_detector_failure_uses_recognition_exception(self) -> None:
        module, calls = load_grasp_module([], detector_failure=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(module.PullCapRecognitionError):
                module.grasp_task3_pull_cap(
                    FakeStation(),
                    {"left": 0.0, "right": 0.0},
                    output_dir=Path(directory),
                )
        self.assertEqual(calls, [])

    def test_moves_xy_then_z_then_closes(self) -> None:
        module, calls = load_grasp_module([True, True])
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            result = module.grasp_task3_pull_cap(
                station,
                {"left": 0.25, "right": 0.0},
                output_dir=Path(directory),
            )
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0]["right"][:3], [1.1, 1.8, 3.0])
        np.testing.assert_allclose(calls[1]["right"][:3], [1.1, 1.8, 2.88])
        self.assertEqual(
            station.actions,
            [{"hand": {"position": {"left": [0.25], "right": [1.0]}}}],
        )
        self.assertEqual(result["first_delta_right_tcp_xyz_m"], [0.1, -0.2, 0.0])
        np.testing.assert_allclose(
            result["second_delta_right_tcp_xyz_m"],
            [0.0, 0.0, -0.12],
        )
        self.assertTrue(result["hand_executed"])

    def test_xy_failure_stops_before_z_and_hand(self) -> None:
        module, calls = load_grasp_module([False])
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            result = module.grasp_task3_pull_cap(
                station,
                {"left": 0.0, "right": 0.0},
                output_dir=Path(directory),
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(station.actions, [])
        self.assertFalse(result["hand_executed"])

    def test_z_failure_does_not_close_hand(self) -> None:
        module, calls = load_grasp_module([True, False])
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            result = module.grasp_task3_pull_cap(
                station,
                {"left": 0.0, "right": 0.0},
                output_dir=Path(directory),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(station.actions, [])
        self.assertFalse(result["hand_executed"])

    def test_slot_selects_nearest_xy_then_moves_z(self) -> None:
        module, calls = load_grasp_module([True, True], with_slots=True)
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            result = module.align_task3_reel_slot(
                station,
                output_dir=Path(directory),
            )
        self.assertEqual(result["selected_slot_quarter_turns_from_cap"], 2)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0]["right"][:3], [1.05, 1.98, 3.0])
        np.testing.assert_allclose(calls[1]["right"][:3], [1.05, 1.98, 2.93])
        self.assertEqual(result["slot_insert_z_right_tcp_m"], -0.07)
        self.assertTrue(result["z_motion_succeeded"])


if __name__ == "__main__":
    unittest.main()
