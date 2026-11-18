#!/usr/bin/env python3
"""Offline verification of task3 pull, slot insertion and recovery order."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("test_task3_pull_cap_cycle.py")


class FakeStation:
    INITIAL_RIGHT_JOINTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    POST_GRASP_RIGHT_JOINTS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.actions: list[dict] = []
        self.right_joints = np.asarray(self.INITIAL_RIGHT_JOINTS)

    def get_robot_state(self) -> dict:
        return {
            "arm_joints": {"right": self.right_joints.copy()},
            "waist_joints": np.asarray([0.2, -0.1]),
            "leg_joints": np.asarray([0.3, -0.2]),
        }

    def execute_action(self, action: dict) -> None:
        if "hand" in action:
            self.events.append("open")
        elif "leg" in action:
            self.events.append("observe_waist_leg")
        else:
            self.events.append("waist_rotate")
        self.actions.append(action)


def load_module(
    events: list[str],
    grasp_succeeded: bool = True,
    pull_arm_succeeded: bool = True,
    recognition_failure_stage: str | None = None,
) -> types.ModuleType:
    settings = types.ModuleType("settings")
    settings.TASK = {
        "hands": {
            "left_open_position": 0.0,
            "right_open_position": 0.0,
        },
        "task3_pull_cap": {
            "waist_yaw_delta_rad": np.deg2rad(32.0),
            "pull_right_arm_joint_delta_rad": {
                "shoulder_pitch": np.deg2rad(-9.0),
                "elbow_pitch": np.deg2rad(12.0),
                "elbow_yaw": np.deg2rad(2.0),
            },
            "post_grasp_retreat_z_tcp_m": 0.07,
        },
    }
    arm_motion = types.ModuleType("arm_motion")
    joint_calls: list[dict[str, list[float] | None]] = []

    def moveit_joints(
        station: object,
        *,
        left: list[float] | None = None,
        right: list[float] | None = None,
    ) -> bool:
        joint_calls.append({"left": left, "right": right})
        events.append("observe_arms" if left is not None else "pull_arm")
        if left is not None:
            return left == [1.0] * 7 and right == [2.0] * 7
        return pull_arm_succeeded

    arm_motion.moveit_joints = moveit_joints
    waist_leg_motion = types.ModuleType("waist_leg_motion")
    waist_leg_motion.execute_synced_waist_leg_action = (
        lambda station, action: station.execute_action(action)
    )
    grasp = types.ModuleType("task3_pull_cap_grasp")

    class PullCapRecognitionError(RuntimeError):
        pass

    def grasp_task3_pull_cap(*args: object, **kwargs: object) -> dict:
        events.append("grasp")
        if recognition_failure_stage == "initial":
            raise PullCapRecognitionError("small cylindrical cap circle was not found")
        station = args[0]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output_dir / "rgb_raw.png"),
            np.full((120, 160, 3), 32, dtype=np.uint8),
        ):
            raise RuntimeError("failed to create fake cap frame")
        if grasp_succeeded:
            station.right_joints = np.asarray(FakeStation.POST_GRASP_RIGHT_JOINTS)
        return {
            "cap_pixel_uv": [48, 42],
            "hand_executed": grasp_succeeded,
        }

    def move_right_tcp_local_z(*args: object, **kwargs: object) -> tuple:
        events.append("retreat_z")
        return True, [1.0, 2.0, 3.07, 0.0, 0.0, 0.0, 1.0]

    def align_task3_reel_slot(*args: object, **kwargs: object) -> dict:
        events.append("slot_xy_z")
        if recognition_failure_stage == "slot":
            raise PullCapRecognitionError("未获得大圆其余三个方向的完整插槽候选")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output_dir / "rgb_raw.png"),
            np.full((120, 160, 3), 64, dtype=np.uint8),
        ):
            raise RuntimeError("failed to create fake slot frame")
        return {
            "selected_slot_pixel_uv": [112, 78],
            "z_motion_succeeded": True,
        }

    grasp.grasp_task3_pull_cap = grasp_task3_pull_cap
    grasp.move_right_tcp_local_z = move_right_tcp_local_z
    grasp.align_task3_reel_slot = align_task3_reel_slot
    grasp.PullCapRecognitionError = PullCapRecognitionError
    config_loader = types.ModuleType("xrocs.core.config_loader")
    config_loader.ConfigLoader = object
    station_module = types.ModuleType(
        "xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station"
    )
    station_module.TienYi2ProRos2Station = object
    path_module = types.ModuleType("xrocs.utils.path_manager")
    path_module.path_manager = object()
    sys.modules.update(
        {
            "settings": settings,
            "arm_motion": arm_motion,
            "task3_pull_cap_grasp": grasp,
            "waist_leg_motion": waist_leg_motion,
            "xrocs": types.ModuleType("xrocs"),
            "xrocs.core": types.ModuleType("xrocs.core"),
            "xrocs.core.config_loader": config_loader,
            "xrocs.entity": types.ModuleType("xrocs.entity"),
            "xrocs.entity.station": types.ModuleType("xrocs.entity.station"),
            "xrocs.entity.station.TienYi2Pro": types.ModuleType(
                "xrocs.entity.station.TienYi2Pro"
            ),
            "xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station": (
                station_module
            ),
            "xrocs.utils": types.ModuleType("xrocs.utils"),
            "xrocs.utils.path_manager": path_module,
        }
    )
    spec = importlib.util.spec_from_file_location(
        "task3_pull_cap_cycle_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.time = types.SimpleNamespace(sleep=lambda _: None)
    module.input = lambda _: ""
    module.joint_calls = joint_calls
    return module


class PullCapCycleTest(unittest.TestCase):
    OBSERVE_POSE = {
        "head": [0.1, 0.2, 0.3],
        "left_arm": [1.0] * 7,
        "right_arm": [2.0] * 7,
        "waist": [0.4, 0.5],
        "leg": [0.6, 0.7],
    }

    def test_observe_pose_uses_moveit_then_station_waist_leg(self) -> None:
        events: list[str] = []
        module = load_module(events)
        station = FakeStation(events)
        module.move_to_task3_observe_pose(
            station,
            self.OBSERVE_POSE,
        )
        self.assertEqual(events, ["observe_arms", "observe_waist_leg"])
        self.assertEqual(
            station.actions,
            [
                {
                    "waist": {"position": [0.4, 0.5]},
                    "leg": {"position": [0.6, 0.7]},
                }
            ],
        )

    def test_complete_sequence_and_saved_joint_recovery(self) -> None:
        events: list[str] = []
        module = load_module(events)
        station = FakeStation(events)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = module.run_test_sequence(
                station,
                output_dir,
                self.OBSERVE_POSE,
            )
            saved = json.loads((output_dir / "cycle_summary.json").read_text())
            initial_point_image = (
                output_dir / "initial_cap_grasp/selected_grasp_point.png"
            )
            slot_point_image = output_dir / "slot_recognition/selected_grasp_point.png"
            self.assertTrue(initial_point_image.is_file())
            self.assertTrue(slot_point_image.is_file())
            initial_marked = cv2.imread(str(initial_point_image))
            slot_marked = cv2.imread(str(slot_point_image))
            self.assertEqual(initial_marked[42, 48].tolist(), [0, 255, 255])
            self.assertEqual(slot_marked[78, 112].tolist(), [0, 255, 255])
            self.assertFalse(
                np.array_equal(
                    initial_marked,
                    cv2.imread(str(output_dir / "initial_cap_grasp/rgb_raw.png")),
                )
            )
        self.assertEqual(
            events,
            [
                "grasp",
                "waist_rotate",
                "pull_arm",
                "retreat_z",
                "slot_xy_z",
                "open",
                "observe_arms",
                "observe_waist_leg",
            ],
        )
        np.testing.assert_allclose(
            module.joint_calls[0]["right"],
            [
                0.15 + np.deg2rad(-9.0),
                0.25,
                0.35,
                0.45 + np.deg2rad(12.0),
                0.55 + np.deg2rad(2.0),
                0.65,
                0.75,
            ],
        )
        self.assertEqual(
            station.actions,
            [
                {
                    "waist": {
                        "position": [0.2 + np.deg2rad(32.0), -0.1]
                    }
                },
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}},
                {
                    "waist": {"position": [0.4, 0.5]},
                    "leg": {"position": [0.6, 0.7]},
                },
            ],
        )
        self.assertNotIn("leg", station.actions[0])
        self.assertEqual(result["post_grasp_retreat_z_tcp_m"], 0.07)
        self.assertTrue(result["returned_to_task3_observe_pose"])
        self.assertEqual(
            saved["original_right_arm_joints_rad"],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        )
        self.assertEqual(
            saved["post_grasp_right_arm_joints_rad"],
            [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
        )
        np.testing.assert_allclose(
            saved["pull_right_arm_joint_delta_rad"],
            [np.deg2rad(-9.0), 0.0, 0.0, np.deg2rad(12.0), np.deg2rad(2.0), 0.0, 0.0],
        )
        np.testing.assert_allclose(
            saved["pull_right_arm_target_joints_rad"],
            module.joint_calls[0]["right"],
        )
        self.assertEqual(
            saved["recognition_point_images"],
            {
                "initial_cap_grasp": ("initial_cap_grasp/selected_grasp_point.png"),
                "selected_reel_slot": ("slot_recognition/selected_grasp_point.png"),
            },
        )

    def test_failed_grasp_stops_before_waist_motion(self) -> None:
        events: list[str] = []
        module = load_module(events, grasp_succeeded=False)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaises(RuntimeError):
                module.run_test_sequence(
                    FakeStation(events),
                    output_dir,
                    self.OBSERVE_POSE,
                )
            self.assertTrue(
                (output_dir / "initial_cap_grasp/selected_grasp_point.png").is_file()
            )
        self.assertEqual(events, ["grasp"])

    def test_failed_pull_arm_stops_before_tcp_retreat(self) -> None:
        events: list[str] = []
        module = load_module(events, pull_arm_succeeded=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                module.run_test_sequence(
                    FakeStation(events),
                    Path(directory),
                    self.OBSERVE_POSE,
                )
        self.assertEqual(events, ["grasp", "waist_rotate", "pull_arm"])

    def test_initial_recognition_failure_opens_and_returns(self) -> None:
        events: list[str] = []
        module = load_module(events, recognition_failure_stage="initial")
        station = FakeStation(events)
        with tempfile.TemporaryDirectory() as directory:
            result = module.run_test_sequence(
                station,
                Path(directory),
                self.OBSERVE_POSE,
            )
        self.assertEqual(
            events,
            ["grasp", "open", "observe_arms", "observe_waist_leg"],
        )
        self.assertTrue(result["recognition_failed"])
        self.assertEqual(result["recognition_stage"], "初次小圆柱帽")
        self.assertTrue(result["returned_to_task3_observe_pose"])

    def test_slot_recognition_failure_opens_and_returns(self) -> None:
        events: list[str] = []
        module = load_module(events, recognition_failure_stage="slot")
        station = FakeStation(events)
        with tempfile.TemporaryDirectory() as directory:
            result = module.run_test_sequence(
                station,
                Path(directory),
                self.OBSERVE_POSE,
            )
        self.assertEqual(
            events,
            [
                "grasp",
                "waist_rotate",
                "pull_arm",
                "retreat_z",
                "slot_xy_z",
                "open",
                "observe_arms",
                "observe_waist_leg",
            ],
        )
        self.assertTrue(result["recognition_failed"])
        self.assertEqual(result["recognition_stage"], "拉动后小圆及插槽")
        self.assertTrue(result["returned_to_task3_observe_pose"])


if __name__ == "__main__":
    unittest.main()
