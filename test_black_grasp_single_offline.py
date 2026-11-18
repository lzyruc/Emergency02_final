#!/usr/bin/env python3
"""离线验证单次三维视觉修正、手臂选择和夹爪行为。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


MODULE_PATH = Path(__file__).with_name("black_grasp_single.py")


class FakeStation:
    def __init__(self) -> None:
        self.actions: list[dict] = []

    def execute_action(self, action: dict) -> None:
        self.actions.append(action)

    def get_robot_state(self) -> dict:
        pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        return {"arm_pose": {"left": pose, "right": pose}}


def load_module(motion_succeeded: bool = True) -> tuple[types.ModuleType, list]:
    """加载被测模块，并注入不依赖机器人环境的感知假实现。"""
    settings = types.ModuleType("settings")
    settings.RIGHT_CAMERA = {
        "arm_name": "right",
        "station_camera_key": "right",
        "image_width": 1280,
        "image_height": 720,
        "color_encoding": "bgr",
        "depth_unit": "mm",
        "camera_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "extrinsics_cam_to_right_tcp_link_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    settings.LEFT_CAMERA = {
        "arm_name": "left",
        "station_camera_key": "left",
        "image_width": 1280,
        "image_height": 720,
        "color_encoding": "bgr",
        "depth_unit": "mm",
        "camera_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "extrinsics_cam_to_left_tcp_link_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    settings.TASK = {
        "hands": {"left_close_position": 1.0, "right_close_position": 1.0},
        "black_grasp": {
            "left_roi_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
            "right_roi_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
            "final_grasp_offset_left_tcp_m": [0.0, 0.0, 0.12],
            "final_grasp_offset_right_tcp_m": [0.0, 0.0, 0.12],
        },
    }
    settings.RUNTIME_DIR = Path("runtime")

    detector_module = types.ModuleType("black_cylinder_detector")
    detect_calls: list[dict] = []

    class Detector:
        def __init__(self, camera_matrix: object) -> None:
            pass

        def detect(self, *args: object, **kwargs: object) -> object:
            detect_calls.append(kwargs)
            return types.SimpleNamespace(
                pixel_uv=(640, 360),
                confidence=0.8,
                depth_m=0.1,
                angle_deg=90.0,
                major_axis_px=200.0,
                minor_axis_px=50.0,
                grasp_candidates=(object(),),
                selection_strategy="nearest",
                point_camera_m=(0.1, 0.2, 0.3),
            )

    def save_debug_images(output_dir: Path, *args: object) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    detector_module.BlackCylinderDetector = Detector
    detector_module.save_debug_images = save_debug_images
    camera_io = types.ModuleType("camera_io")
    camera_io.read_wrist_camera = lambda station, config: (
        object(),
        types.SimpleNamespace(shape=(720, 1280)),
    )
    wrist_self_mask = types.ModuleType("wrist_self_mask")
    wrist_self_mask.gripper_mask = lambda shape: object()
    moveit_calls: list = []
    arm_motion = types.ModuleType("arm_motion")

    def moveit_cartesians(
        station: object,
        *,
        left: list | None = None,
        right: list | None = None,
        use_cartesian_path: bool = False,
        left_motion_id: int = 0,
        right_motion_id: int = 0,
    ) -> bool:
        moveit_calls.append(
            {
                "left": left,
                "right": right,
                "use_cartesian_path": use_cartesian_path,
                "left_motion_id": left_motion_id,
                "right_motion_id": right_motion_id,
            }
        )
        return motion_succeeded

    arm_motion.moveit_cartesians = moveit_cartesians
    blue_grasp = types.ModuleType("blue_grasp")
    blue_grasp.pose_xyz_xyzw = lambda pose: pose

    sys.modules["settings"] = settings
    sys.modules["arm_motion"] = arm_motion
    sys.modules["black_cylinder_detector"] = detector_module
    sys.modules["blue_grasp"] = blue_grasp
    sys.modules["camera_io"] = camera_io
    sys.modules["wrist_self_mask"] = wrist_self_mask
    spec = importlib.util.spec_from_file_location("black_grasp_single_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._test_detect_calls = detect_calls
    return module, moveit_calls


class BlackGraspSingleTest(unittest.TestCase):
    def test_right_arm_moves_full_xyz_and_closes_only_right(self) -> None:
        module, moveit_calls = load_module()
        module.time = types.SimpleNamespace(sleep=lambda _: None)
        module.input = lambda _: ""
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            result = module.grasp_black_single(
                station,
                "right",
                {"left": 0.0, "right": 0.0},
                output_dir=Path(directory),
            )
        self.assertEqual(
            station.actions,
            [{"hand": {"position": {"left": [0.0], "right": [1.0]}}}],
        )
        self.assertEqual(len(moveit_calls), 1)
        target = moveit_calls[0]["right"]
        self.assertIsNone(moveit_calls[0]["left"])
        self.assertAlmostEqual(target[0], 1.1)
        self.assertAlmostEqual(target[1], 2.2)
        self.assertAlmostEqual(target[2], 3.42)
        self.assertFalse(moveit_calls[0]["use_cartesian_path"])
        self.assertAlmostEqual(result["delta_waist_m"][2], 0.42)
        self.assertTrue(result["tcp_correction_executed"])
        self.assertTrue(result["hand_executed"])

    def test_moveit_failure_still_closes_hand(self) -> None:
        module, moveit_calls = load_module(False)
        module.time = types.SimpleNamespace(sleep=lambda _: None)
        module.input = lambda _: ""
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            result = module.grasp_black_single(
                station,
                "left",
                {"left": 0.0, "right": 1.0},
                output_dir=Path(directory),
            )
        self.assertEqual(len(moveit_calls), 1)
        self.assertEqual(
            station.actions,
            [{"hand": {"position": {"left": [1.0], "right": [1.0]}}}],
        )
        self.assertFalse(result["motion_succeeded"])
        self.assertTrue(result["hand_executed"])

    def test_component_selection_is_forwarded_to_detector(self) -> None:
        module, _ = load_module()
        module.time = types.SimpleNamespace(sleep=lambda _: None)
        station = FakeStation()
        with tempfile.TemporaryDirectory() as directory:
            module.grasp_black_single(
                station,
                "left",
                {"left": 0.0, "right": 1.0},
                output_dir=Path(directory),
                component_selection="leftmost",
            )
        self.assertEqual(
            module._test_detect_calls[-1]["component_selection"],
            "leftmost",
        )


if __name__ == "__main__":
    unittest.main()
