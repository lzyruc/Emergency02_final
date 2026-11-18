"""task1 文件筐流程离线测试，不导入或连接真实 xrocs 硬件。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK1_PATH = ROOT / "0803/Emergency/task1.py"


def load_task1_with_fake_dependencies():
    settings = types.ModuleType("settings")
    settings.TASK = {
        "hands": {
            "left_open_position": 0.0,
            "left_close_position": 1.0,
            "right_open_position": 0.0,
            "right_close_position": 1.0,
        }
    }
    settings.LEFT_CAMERA = {"camera_matrix": np.eye(3)}
    settings.RIGHT_CAMERA = {"camera_matrix": np.eye(3)}

    arm_motion = types.ModuleType("arm_motion")
    arm_motion.moveit_cartesians = lambda *args, **kwargs: True
    arm_motion.moveit_joints = lambda *args, **kwargs: True

    blue_depth_target = types.ModuleType("blue_depth_target")
    blue_depth_target.BlueDepthTargetDetector = lambda matrix: object()

    camera_io = types.ModuleType("camera_io")
    camera_io.read_head_camera = lambda station: (None, None)
    camera_io.read_wrist_camera = lambda station, config: (None, None)

    task2_phase1 = types.ModuleType("task2_phase1")
    task2_phase1.Phase1Model = object

    deprojector = types.ModuleType("agent.perception.deprojector")
    deprojector.Deprojector = lambda station, path, camera: object()

    fake_modules = {
        "settings": settings,
        "arm_motion": arm_motion,
        "blue_depth_target": blue_depth_target,
        "camera_io": camera_io,
        "task2_phase1": task2_phase1,
        "agent.perception.deprojector": deprojector,
    }
    previous = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "task1_offline_subject",
            TASK1_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Station:
    def __init__(self) -> None:
        self.actions = []

    def execute_action(self, action) -> None:
        self.actions.append(action)


class _Navigation:
    def __init__(self) -> None:
        self.calls = []

    def moveBy(self, x, y, theta) -> None:
        self.calls.append(("moveBy", x, y, theta))

    def turnTo(self, theta) -> None:
        self.calls.append(("turnTo", theta))

    def moveTo(self, node, **kwargs) -> None:
        self.calls.append(("moveTo", node, kwargs))

    def recoverState(self, node=None, **kwargs) -> None:
        self.calls.append(("recoverState", node, kwargs))


class FileboxWorkflowTests(unittest.TestCase):
    def test_successful_horizontal_box_drag_releases_before_recovery(self) -> None:
        task1 = load_task1_with_fake_dependencies()
        station = _Station()
        navigation = _Navigation()
        executed_poses = []
        task1.time.sleep = lambda seconds: None
        task1.recognize_obstacles = lambda current_station: [
            {
                "type": "box",
                "name": "纸箱（3 kg）",
                "pose": [0.8, 0.0, 0.0],
                "distance_3d_m": 0.8,
            }
        ]
        task1._execute_object_pose = (
            lambda current_station, pose: executed_poses.append(pose) or True
        )

        succeeded = task1.remove_box(
            station,
            navigation,
            {"type": "box", "pose": [2.0, 0.0, 0.0]},
        )

        self.assertTrue(succeeded)
        self.assertEqual(len(executed_poses), 3)
        self.assertIs(executed_poses[0], executed_poses[2])
        self.assertEqual(len(station.actions), 1)
        release_index = len(navigation.calls) - 2
        self.assertEqual(
            navigation.calls[release_index:],
            [
                ("recoverState", "startpoint", {"method": "move_by"}),
                ("turnTo", 0.0),
            ],
        )
        self.assertIn(("moveBy", 0.0, -0.3, 0.0), navigation.calls)
        self.assertIn(("moveBy", 0.8, 0.0, 0.0), navigation.calls)

    def test_lost_target_recovers_without_stale_second_move(self) -> None:
        task1 = load_task1_with_fake_dependencies()
        task1.recognize_obstacles = lambda current_station: []
        navigation = _Navigation()

        succeeded = task1.remove_filebox(
            _Station(),
            navigation,
            {"type": "filebox", "pose": [2.0, 0.0, 0.0]},
        )

        self.assertFalse(succeeded)
        self.assertEqual(
            navigation.calls,
            [
                ("moveBy", 1.0, 0.0, 0.0),
                ("moveBy", 0.0, 0.0, 0.0),
                ("recoverState", None, {}),
            ],
        )

    def test_successful_right_arm_flow(self) -> None:
        task1 = load_task1_with_fake_dependencies()
        station = _Station()
        navigation = _Navigation()
        cartesian_calls = []
        task1.time.sleep = lambda seconds: None
        task1.recognize_obstacles = lambda current_station: [
            {
                "type": "filebox",
                "name": "文件筐",
                "pose": [0.8, 0.0, 0.0],
                "distance_3d_m": 0.8,
            }
        ]
        task1._execute_object_pose = lambda current_station, pose: True
        task1._filebox_wrist_candidates = lambda *args: [
            {
                "arm": "right",
                "confidence": 0.9,
                "target_pose_waist_xyz_xyzw": np.asarray(
                    [0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]
                ),
                "tcp_distance_m": 0.3,
            }
        ]

        def moveit_cartesians(current_station, **kwargs):
            cartesian_calls.append(kwargs)
            return True

        task1.moveit_cartesians = moveit_cartesians

        succeeded = task1.remove_filebox(
            station,
            navigation,
            {"type": "filebox", "pose": [2.0, 0.0, 0.0]},
        )

        self.assertTrue(succeeded)
        self.assertEqual(
            cartesian_calls,
            [{"right": [[0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]]}],
        )
        self.assertIn(("moveBy", 1.0, 0.0, 0.0), navigation.calls)
        self.assertTrue(
            any(
                call[0] == "moveBy"
                and abs(call[1] - 0.4) < 1e-12
                and call[2:] == (0.0, 0.0)
                for call in navigation.calls
            )
        )
        self.assertIn(
            ("moveTo", "startpoint", {"correct": True, "rotate": False}),
            navigation.calls,
        )
        self.assertIn(
            ("recoverState", "startpoint", {"method": "move_by"}),
            navigation.calls,
        )
        self.assertEqual(len(station.actions), 3)


if __name__ == "__main__":
    unittest.main()
