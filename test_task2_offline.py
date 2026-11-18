"""任务二一号窗视觉对齐的离线测试，不连接或控制真实机器人。"""

from __future__ import annotations

# 现场的旧识别测试曾使用：
# PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/usr/lib/python3.12/dist-packages" \
# python test_target_recognize.py

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK2_PATH = Path(__file__).resolve().with_name("task2.py")


def load_task2_with_fake_dependencies():
    calls: dict[str, list] = {
        "joints": [],
        "camera": [],
        "detect": [],
        "deprojector": [],
        "deproject": [],
        "cartesians": [],
    }
    settings = types.ModuleType("settings")
    settings.WINDOW1_POSES = [
        {},
        {"left_arm": [1.0] * 7, "right_arm": [2.0] * 7},
    ]
    settings.LEFT_CAMERA = {
        "station_camera_key": "left",
        "camera_matrix": [
            [600.0, 0.0, 640.0],
            [0.0, 601.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
    }

    arm_motion = types.ModuleType("arm_motion")

    def moveit_joints(*args, **kwargs):
        calls["joints"].append((args, kwargs))
        return arm_motion.joints_succeeded

    def moveit_cartesians(*args, **kwargs):
        calls["cartesians"].append((args, kwargs))
        return arm_motion.cartesians_succeeded

    arm_motion.joints_succeeded = True
    arm_motion.cartesians_succeeded = True
    arm_motion.moveit_joints = moveit_joints
    arm_motion.moveit_cartesians = moveit_cartesians

    color = np.zeros((4, 6, 3), dtype=np.uint8)
    depth_mm = np.full((4, 6), 325, dtype=np.uint16)
    camera_io = types.ModuleType("camera_io")

    def read_wrist_camera(station, config):
        calls["camera"].append((station, config))
        return color, depth_mm

    camera_io.read_wrist_camera = read_wrist_camera

    detection = types.SimpleNamespace(
        uv_depth_m=np.asarray([640.0, 360.0, 0.325])
    )
    handle_detection = types.ModuleType(
        "agent.perception.window_handle_detection"
    )

    def detect_handle_center(depth_m, **kwargs):
        calls["detect"].append((depth_m, kwargs))
        return detection

    handle_detection.detect_handle_center = detect_handle_center

    target_pose = np.asarray([0.4, 0.3, -0.1, 0.0, 0.0, 0.0, 1.0])
    perception = types.ModuleType("agent.perception")
    deprojector_module = types.ModuleType("agent.perception.deprojector")
    deprojector_module.__file__ = str(
        ROOT / "agent/perception/deprojector.py"
    )

    class FakeDeprojector:
        def __init__(self, *args):
            calls["deprojector"].append(args)

        def deproject(self, sample, *, face_target):
            calls["deproject"].append((sample, face_target))
            return target_pose

    deprojector_module.Deprojector = FakeDeprojector
    perception.deprojector = deprojector_module

    fake_modules = {
        "settings": settings,
        "arm_motion": arm_motion,
        "camera_io": camera_io,
        "agent.perception": perception,
        "agent.perception.deprojector": deprojector_module,
        "agent.perception.window_handle_detection": handle_detection,
    }
    previous = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "task2_offline_subject",
            TASK2_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module, calls, arm_motion, target_pose, depth_mm


class OpenWindow1Tests(unittest.TestCase):
    def test_observes_detects_deprojects_and_aligns_left_tcp(self) -> None:
        task2, calls, _, target_pose, depth_mm = (
            load_task2_with_fake_dependencies()
        )
        T_waist_tcp = np.eye(4)
        # LEFT_TCP_LINK +X 在 WAIST_YAW_LINK 中指向 +Y，验证按局部轴投影。
        T_waist_tcp[:3, :3] = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        T_waist_tcp[:3, 3] = [0.2, 0.1, -0.2]

        class Pose:
            def get_affine_matrix_m_ndarray(self):
                return T_waist_tcp

        pin_kine = types.SimpleNamespace(
            get_left_tcp_pose=lambda joints: Pose()
        )
        robot = types.SimpleNamespace(pin_kine=pin_kine)
        station = types.SimpleNamespace(
            get_robot_handle=lambda: {"robot": robot},
            get_robot_state=lambda: {"arm_joints": {"left": [0.0] * 7}},
        )

        self.assertTrue(task2.open_window1(station))

        self.assertEqual(len(calls["joints"]), 1)
        joint_args, joint_kwargs = calls["joints"][0]
        self.assertIs(joint_args[0], station)
        self.assertEqual(joint_kwargs["left"], [1.0] * 7)
        self.assertEqual(joint_kwargs["right"], [2.0] * 7)
        self.assertEqual(calls["camera"], [(station, task2.settings.LEFT_CAMERA)])

        detected_depth, detector_kwargs = calls["detect"][0]
        np.testing.assert_allclose(detected_depth, depth_mm / 1000.0)
        self.assertEqual(detector_kwargs, {"fx": 600.0, "fy": 601.0})
        self.assertEqual(calls["deproject"][0][1], False)

        cartesian_args, cartesian_kwargs = calls["cartesians"][0]
        self.assertIs(cartesian_args[0], station)
        np.testing.assert_allclose(
            cartesian_kwargs["left"],
            [0.4, 0.1, -0.1, *target_pose[3:]],
        )
        self.assertTrue(cartesian_kwargs["use_cartesian_path"])
        self.assertEqual(cartesian_kwargs["left_motion_id"], 1)

    def test_stops_before_camera_when_observation_motion_fails(self) -> None:
        task2, calls, arm_motion, _, _ = load_task2_with_fake_dependencies()
        arm_motion.joints_succeeded = False

        self.assertFalse(task2.open_window1(object()))

        self.assertEqual(len(calls["joints"]), 1)
        self.assertEqual(calls["camera"], [])
        self.assertEqual(calls["cartesians"], [])

    def test_returns_cartesian_motion_result(self) -> None:
        task2, calls, arm_motion, _, _ = load_task2_with_fake_dependencies()
        arm_motion.cartesians_succeeded = False

        class Pose:
            def get_affine_matrix_m_ndarray(self):
                return np.eye(4)

        robot = types.SimpleNamespace(
            pin_kine=types.SimpleNamespace(
                get_left_tcp_pose=lambda joints: Pose()
            )
        )
        station = types.SimpleNamespace(
            get_robot_handle=lambda: {"robot": robot},
            get_robot_state=lambda: {"arm_joints": {"left": [0.0] * 7}},
        )

        self.assertFalse(task2.open_window1(station))

        self.assertEqual(len(calls["cartesians"]), 1)


if __name__ == "__main__":
    unittest.main()
