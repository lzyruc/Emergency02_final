"""emergency_zyt任务2编排的离线顺序测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import emergency_zyt


_CHANNEL_REGION = emergency_zyt.ChannelRegion(
    ((-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)),
    outside_margin_m=0.0,
)


class _Pose:
    def get_xyz_m_xyzw_ndarray(self):
        return np.asarray([0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])


class _Kinematics:
    def get_left_tcp_pose(self, joints):
        return self

    def get_affine_matrix_m_ndarray(self):
        return np.eye(4)


class _Station:
    def __init__(self):
        self.actions = []
        robot = type("Robot", (), {"pin_kine": _Kinematics()})()
        self.robot_handle = {"robot": robot}

    def get_robot_handle(self):
        return self.robot_handle

    def get_robot_state(self):
        return {
            "arm_joints": {"left": [0.0] * 7},
            "arm_pose": {"left": _Pose()},
            "waist_joints": [0.3, 0.4],
            "leg_joints": [0.1, 0.2],
        }

    def execute_action(self, action):
        self.actions.append(action)


class _Navigation:
    def __init__(self):
        self.calls = []

    def moveTo(self, node, **kwargs):
        self.calls.append((node, kwargs))

    def recoverState(self, node, **kwargs):
        self.calls.append(("recover", node, kwargs))

    def turnTo(self, theta):
        self.calls.append(("turnTo", theta))

    def moveBy(self, dx, dy, dtheta, **kwargs):
        self.calls.append(("moveBy", dx, dy, dtheta, kwargs))

    def getPose(self):
        return {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0}, None


class _BodyController:
    def __init__(self):
        self.targets = []

    def move_to(self, target):
        self.targets.append(list(target))
        return True


class EmergencyZytTask2Tests(unittest.TestCase):
    def test_window_push_retries_absolute_targets_with_unique_motion_ids(self):
        class TcpPose:
            def get_affine_matrix_m_ndarray(self):
                return np.eye(4)

        class Robot:
            pin_kine = type(
                "Kinematics",
                (),
                {"get_left_tcp_pose": lambda self, joints: TcpPose()},
            )()

        class Station:
            def get_robot_state(self):
                return {
                    "arm_joints": {"left": [0.0] * 7},
                    "arm_pose": {
                        "left": np.asarray(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                        )
                    },
                }

        calls = []
        results = iter((False, False, True))
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "moveit_cartesians",
                side_effect=lambda station, **kwargs: calls.append(kwargs)
                or next(results),
            ),
        ):
            succeeded = emergency_zyt._push_window_with_reduced_offset(
                Station(),
                Robot(),
                first_motion_id=7,
                label="向前",
                initial_delta_tcp_m=np.asarray([0.0, 0.0, -0.30]),
            )

        self.assertTrue(succeeded)
        self.assertEqual(
            [call["left_motion_id"] for call in calls],
            [7, 8, 9],
        )
        self.assertTrue(
            np.allclose(
                [call["left"][2] for call in calls],
                [-0.30, -0.27, -0.243],
            )
        )

    def test_window_push_partial_motion_past_shorter_target_is_success(self):
        class TcpPose:
            def __init__(self, z_m):
                self.matrix = np.eye(4)
                self.matrix[2, 3] = z_m

            def get_affine_matrix_m_ndarray(self):
                return self.matrix

        class Kinematics:
            def __init__(self):
                self.call_count = 0

            def get_left_tcp_pose(self, joints):
                self.call_count += 1
                return TcpPose(0.0 if self.call_count == 1 else -0.28)

        class Robot:
            def __init__(self):
                self.pin_kine = Kinematics()

        class Station:
            def get_robot_state(self):
                return {
                    "arm_joints": {"left": [0.0] * 7},
                    "arm_pose": {
                        "left": np.asarray(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                        )
                    },
                }

        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "moveit_cartesians",
                return_value=False,
            ) as cartesian,
        ):
            succeeded = emergency_zyt._push_window_with_reduced_offset(
                Station(),
                Robot(),
                first_motion_id=7,
                label="向前",
                initial_delta_tcp_m=np.asarray([0.0, 0.0, -0.30]),
            )

        self.assertTrue(succeeded)
        self.assertEqual(cartesian.call_count, 1)

    def test_window2_detection_failure_skips_motion_and_hand_actions(self):
        station = _Station()
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "recognize_window1_handle",
                side_effect=RuntimeError("没有符合比赛把手尺寸的连通区域"),
            ),
            patch.object(emergency_zyt, "moveit_cartesians") as cartesian,
            patch.object(emergency_zyt, "moveit_joints") as joints,
        ):
            succeeded = emergency_zyt.run_window2_action(station)

        self.assertFalse(succeeded)
        cartesian.assert_not_called()
        joints.assert_not_called()
        self.assertEqual(station.actions, [])

    def test_body_action_reorders_pose_joints_and_keeps_missing_parts(self):
        station = _Station()
        body_controller = _BodyController()

        emergency_zyt.execute_body_action(
            station,
            body_controller,
            {
                "head": {"position": [0.5, 0.6, 0.7]},
                "waist": {"position": [0.8, 0.9]},
            },
        )

        self.assertEqual(body_controller.targets, [[0.2, 0.1, 0.9, 0.8]])
        self.assertEqual(
            station.actions,
            [{"head": {"position": [0.5, 0.6, 0.7]}}],
        )

    def test_integrated_box_restores_full_initial_pose_before_final_recover(self):
        station = _Station()
        events = []

        class Navigation:
            def turnTo(self, theta):
                events.append(("turnTo", theta))

            def moveBy(self, x_m, y_m, theta_deg):
                events.append(("moveBy", x_m, y_m, theta_deg))

            def recoverState(self, node, **kwargs):
                events.append(("recover", node, kwargs))

        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "_execute_object_pose",
                side_effect=lambda current_station, pose: events.append(
                    ("object_pose", pose)
                )
                or True,
            ),
            patch.object(
                emergency_zyt,
                "apply_startpoint_pose",
                side_effect=lambda current_station, pose: events.append(
                    ("initial_pose", pose)
                ),
            ),
            patch.object(emergency_zyt.time, "sleep"),
        ):
            emergency_zyt.run_box_after_approach(
                station,
                Navigation(),
                {"orientation": "vertical"},
                {"name": "initial"},
                1.2,
                "startpoint",
            )

        initial_index = next(
            index for index, event in enumerate(events) if event[0] == "initial_pose"
        )
        recover_indices = [
            index for index, event in enumerate(events) if event[0] == "recover"
        ]
        self.assertEqual(len(recover_indices), 3)
        self.assertLess(initial_index, recover_indices[2])
        self.assertEqual(
            sum(event[0] == "object_pose" for event in events),
            3,
        )

    def test_bin_and_filebox_reduce_approach_turn_by_15_degrees(self):
        class Avoid:
            def checkCollision(self, dx, dy, dtheta):
                return False

        for target_type, point, expected_turn in (
            ("bin", [1.0, 1.0, 0.0], 30.0),
            ("filebox", [1.0, -1.0, 0.0], -30.0),
            ("chair", [1.0, 1.0, 0.0], 45.0),
        ):
            target = {
                "name": target_type,
                "type": target_type,
                "point_base_m": np.asarray(point),
                "distance_xy_m": float(np.hypot(point[0], point[1])),
            }
            _, motion = emergency_zyt.select_safe_approach(Avoid(), [target])
            self.assertAlmostEqual(motion[2], expected_turn)

    def test_small_bin_and_filebox_turn_still_applies_15_degrees(self):
        class Avoid:
            def checkCollision(self, dx, dy, dtheta):
                return False

        for target_type, y_m, expected_turn in (
            ("bin", np.tan(np.deg2rad(10.0)), -5.0),
            ("filebox", -np.tan(np.deg2rad(10.0)), 5.0),
        ):
            target = {
                "name": target_type,
                "type": target_type,
                "point_base_m": np.asarray([1.0, y_m, 0.0]),
                "distance_xy_m": float(np.hypot(1.0, y_m)),
            }
            _, motion = emergency_zyt.select_safe_approach(Avoid(), [target])
            self.assertAlmostEqual(motion[2], expected_turn)

    def test_second_observation_approach_clamps_y_before_collision_check(self):
        class Avoid:
            def __init__(self):
                self.motion = None

            def checkCollision(self, dx, dy, dtheta):
                self.motion = (dx, dy, dtheta)
                return False

        target = {
            "name": "box",
            "type": "box",
            "point_base_m": np.asarray([1.0, -1.0, 0.0]),
            "distance_xy_m": float(np.hypot(1.0, -1.0)),
        }
        avoid = Avoid()
        _, motion = emergency_zyt.select_safe_approach(
            avoid,
            [target],
            max_abs_y_m=0.20,
        )
        self.assertAlmostEqual(motion[1], -0.20)
        self.assertEqual(avoid.motion, motion)

    def test_startpoint_collision_retries_from_second_observation(self):
        target = {
            "name": "垃圾桶",
            "type": "bin",
            "hits": 5,
            "point_base_m": np.asarray([1.0, 1.0, 0.0]),
            "distance_xy_m": float(np.hypot(1.0, 1.0)),
        }

        class Avoid:
            def __init__(self):
                self.motions = []

            def checkCollision(self, dx, dy, dtheta):
                self.motions.append((dx, dy, dtheta))
                return len(self.motions) == 1

        navigation = _Navigation()
        avoid = Avoid()
        with patch.object(
            emergency_zyt,
            "recognize_task1_batch",
            side_effect=[[target], [target]],
        ) as recognize:
            origin, reliable, approach = emergency_zyt.find_task1_approach(
                object(),
                navigation,
                avoid,
                object(),
                object(),
                _CHANNEL_REGION,
                {"bin"},
                1,
            )

        self.assertEqual(recognize.call_count, 2)
        self.assertEqual(
            navigation.calls,
            [
                (
                    "recover",
                    "h0_track_turn_2",
                    {"method": "move_by"},
                )
            ],
        )
        self.assertEqual(origin, "h0_track_turn_2")
        self.assertEqual(reliable, [target])
        self.assertIsNotNone(approach)
        self.assertAlmostEqual(approach[1][1], 0.20)
        self.assertAlmostEqual(avoid.motions[1][1], 0.20)

    def test_next_round_continues_at_h0_without_navigation(self):
        target = {
            "name": "纸箱",
            "type": "box",
            "hits": 5,
            "point_base_m": np.asarray([1.0, -0.5, 0.0]),
            "distance_xy_m": float(np.hypot(1.0, -0.5)),
        }

        class Avoid:
            def checkCollision(self, dx, dy, dtheta):
                return False

        navigation = _Navigation()
        with patch.object(
            emergency_zyt,
            "recognize_task1_batch",
            return_value=[target],
        ) as recognize:
            origin, _, approach = emergency_zyt.find_task1_approach(
                object(),
                navigation,
                Avoid(),
                object(),
                object(),
                _CHANNEL_REGION,
                {"box"},
                2,
                start_from_h0=True,
            )

        recognize.assert_called_once()
        self.assertEqual(origin, "h0_track_turn_2")
        self.assertIsNotNone(approach)
        self.assertEqual(navigation.calls, [])
        self.assertAlmostEqual(approach[1][1], -0.20)

    def test_startpoint_without_remaining_items_does_not_go_to_h0(self):
        navigation = _Navigation()
        with patch.object(
            emergency_zyt,
            "recognize_task1_batch",
            return_value=[],
        ) as recognize:
            origin, reliable, approach = emergency_zyt.find_task1_approach(
                object(),
                navigation,
                object(),
                object(),
                object(),
                _CHANNEL_REGION,
                {"bin"},
                1,
            )
        recognize.assert_called_once()
        self.assertEqual(navigation.calls, [])
        self.assertEqual(origin, "startpoint")
        self.assertEqual(reliable, [])
        self.assertIsNone(approach)

    def test_find_task1_approach_skips_when_no_remaining_types(self):
        navigation = _Navigation()
        with patch.object(
            emergency_zyt,
            "recognize_task1_batch",
        ) as recognize:
            origin, reliable, approach = emergency_zyt.find_task1_approach(
                object(),
                navigation,
                object(),
                object(),
                object(),
                _CHANNEL_REGION,
                set(),
                1,
            )
        recognize.assert_not_called()
        self.assertEqual(navigation.calls, [])
        self.assertEqual(origin, "startpoint")
        self.assertEqual(reliable, [])
        self.assertIsNone(approach)

    def test_task1_does_not_research_after_remaining_types_cleared(self):
        navigation = _Navigation()
        target = {
            "name": "垃圾桶",
            "type": "bin",
            "hits": 5,
            "point_base_m": np.asarray([1.0, 0.0, 0.0]),
            "distance_xy_m": 1.0,
        }
        find_calls = []

        def find_approach(*args, **kwargs):
            find_calls.append((args, kwargs))
            return "startpoint", [target], (target, (0.2, 0.0, 0.0))

        with (
            patch.object(emergency_zyt, "TARGET_NAMES", {"bin": "垃圾桶"}),
            patch.object(
                emergency_zyt, "find_task1_approach", side_effect=find_approach,
            ),
            patch.object(emergency_zyt, "run_with_retry_or_skip", return_value=True),
            patch.object(emergency_zyt, "recognize_task1_batch", return_value=[]),
            patch.object(
                emergency_zyt, "move_from_observation_to_task2_start",
            ) as move_task2,
            patch("builtins.input", return_value=""),
        ):
            origin = emergency_zyt.run_task1_debris(
                object(),
                navigation,
                object(),
                _CHANNEL_REGION,
                object(),
                object(),
                {},
                {},
            )
        self.assertEqual(len(find_calls), 1)
        move_task2.assert_not_called()
        self.assertEqual(origin, "startpoint")

    def test_task1_no_targets_returns_observation_without_handoff(self):
        navigation = _Navigation()
        with (
            patch.object(
                emergency_zyt,
                "find_task1_approach",
                return_value=("h0_track_turn_2", [], None),
            ),
            patch.object(
                emergency_zyt, "move_from_observation_to_task2_start",
            ) as move_task2,
            patch("builtins.input", return_value=""),
        ):
            origin = emergency_zyt.run_task1_debris(
                object(),
                navigation,
                object(),
                _CHANNEL_REGION,
                object(),
                object(),
                {},
                {},
            )
        self.assertEqual(origin, "h0_track_turn_2")
        move_task2.assert_not_called()

    def test_task1_handoff_timeout_retries_only_handoff(self):
        debris_calls = []
        handoff_calls = []

        def debris(*_args, **_kwargs):
            debris_calls.append(1)
            return "h0_track_turn_2"

        def handoff(_navigation, node):
            handoff_calls.append(node)
            raise RuntimeError(
                "[1001] Action 21bd3d4f-7e16-9ac4-d8b8-64fecb8f2339 "
                "timed out after 10.0 seconds"
            )

        with (
            patch.object(emergency_zyt, "run_task1_debris", side_effect=debris),
            patch.object(
                emergency_zyt,
                "move_from_observation_to_task2_start",
                side_effect=handoff,
            ),
            patch("builtins.input", return_value="s"),
        ):
            emergency_zyt.run_task1_and_go_to_task2_start(
                object(),
                object(),
                object(),
                object(),
                object(),
                object(),
                {},
                {},
            )
        self.assertEqual(debris_calls, [1])
        self.assertEqual(handoff_calls, ["h0_track_turn_2"])

    def test_filebox_detection_failure_restores_pose_before_startpoint(self):
        station = _Station()
        navigation = _Navigation()
        events = []

        def recover_state(node, **kwargs):
            events.append(("recover", node, kwargs))

        navigation.recoverState = recover_state
        with (
            patch("builtins.input", return_value=""),
            patch.object(emergency_zyt, "_execute_object_pose", return_value=True),
            patch.object(
                emergency_zyt,
                "wrist_grasp_candidates",
                return_value=[],
            ),
            patch.object(
                emergency_zyt,
                "apply_startpoint_pose",
                side_effect=lambda current_station, pose: events.append(
                    ("initial_pose", current_station, pose)
                ),
            ),
            patch.object(emergency_zyt.time, "sleep"),
        ):
            succeeded = emergency_zyt.grasp_filebox_or_bin_from_bin_pose(
                station,
                navigation,
                {"left": object(), "right": object()},
                {"left": object(), "right": object()},
                "filebox",
                1.2,
                "startpoint",
            )

        self.assertFalse(succeeded)
        self.assertEqual(events[0][0], "initial_pose")
        self.assertEqual(
            events[1],
            ("recover", "startpoint", {"method": "move_by"}),
        )

    def test_filebox_success_restores_pose_before_final_startpoint(self):
        candidate = {
            "arm": "right",
            "confidence": 0.9,
            "pixel_uv": [640, 360],
            "point_source": "grasp_line",
            "tcp_distance_m": 0.3,
            "target_pose_waist_xyz_xyzw": np.asarray(
                [0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]
            ),
        }
        station = _Station()
        events = []

        class Navigation:
            def recoverState(self, node, **kwargs):
                events.append(("recover", node, kwargs))

            def moveBy(self, x_m, y_m, theta_deg):
                events.append(("moveBy", x_m, y_m, theta_deg))

            def turnTo(self, theta):
                events.append(("turnTo", theta))

        with (
            patch("builtins.input", return_value=""),
            patch.object(emergency_zyt, "_execute_object_pose", return_value=True),
            patch.object(
                emergency_zyt,
                "wrist_grasp_candidates",
                return_value=[candidate],
            ),
            patch.object(emergency_zyt, "moveit_cartesians", return_value=True),
            patch.object(
                emergency_zyt,
                "execute_synced_waist_leg_action",
            ),
            patch.object(emergency_zyt, "moveit_joints", return_value=True),
            patch.object(
                emergency_zyt,
                "apply_startpoint_pose",
                side_effect=lambda current_station, pose: events.append(
                    ("initial_pose", pose)
                ),
            ),
            patch.object(emergency_zyt.time, "sleep"),
        ):
            succeeded = emergency_zyt.grasp_filebox_or_bin_from_bin_pose(
                station,
                Navigation(),
                {"left": object(), "right": object()},
                {"left": object(), "right": object()},
                "filebox",
                1.2,
                "startpoint",
            )

        self.assertTrue(succeeded)
        final_recover_index = max(
            index for index, event in enumerate(events) if event[0] == "recover"
        )
        initial_index = next(
            index for index, event in enumerate(events) if event[0] == "initial_pose"
        )
        self.assertLess(initial_index, final_recover_index)

    def test_task2_motion_order(self):
        station = _Station()
        navigation = _Navigation()
        joint_calls = []
        cartesian_calls = []
        synced_actions = []
        with (
            patch("builtins.input", return_value="") as prompt,
            patch.object(
                emergency_zyt,
                "recognize_window1_handle",
                return_value={"pixel_uv": [640.0, 360.0], "depth_m": 0.3},
            ),
            patch.object(
                emergency_zyt.deprojector_module,
                "Deprojector",
            ) as deprojector,
            patch.object(
                emergency_zyt,
                "moveit_joints",
                side_effect=lambda current_station, **kwargs: joint_calls.append(
                    kwargs
                )
                or True,
            ),
            patch.object(
                emergency_zyt,
                "moveit_cartesians",
                side_effect=lambda current_station, **kwargs: cartesian_calls.append(
                    kwargs
                )
                or True,
            ),
            patch.object(
                emergency_zyt,
                "execute_synced_waist_leg_action",
                side_effect=lambda current_station, action, **kwargs: synced_actions.append(
                    action
                ),
            ),
            patch.object(Path, "write_text", return_value=0),
        ):
            deprojector.return_value.deproject.return_value = np.asarray(
                [0.1, 0.2, -0.15, 0.0, 0.0, 0.0, 1.0]
            )
            emergency_zyt.run_task2_window(station, navigation)

        self.assertEqual(prompt.call_count, 31)
        self.assertEqual(
            navigation.calls,
            [
                ("h270_window1_action", {"correct": True, "align": True}),
                ("h270_task2_check", {"correct": False}),
                ("h270_window2_action", {"correct": False}),
            ],
        )
        expected_pose_indices = [0, 1, 12, 13]
        self.assertEqual(len(joint_calls), len(expected_pose_indices) + 2)
        for call, index in zip(
            joint_calls[:4], expected_pose_indices, strict=True
        ):
            self.assertEqual(
                call["left"], emergency_zyt.settings.WINDOW1_POSES[index]["left_arm"]
            )
            self.assertNotIn("right", call)
        self.assertEqual(
            joint_calls[4]["left"],
            emergency_zyt.settings.WINDOW2_PREPARE_POSE["left_arm"],
        )
        self.assertEqual(
            joint_calls[4]["right"],
            emergency_zyt.settings.WINDOW2_PREPARE_POSE["right_arm"],
        )
        self.assertEqual(
            joint_calls[5]["left"],
            emergency_zyt.settings.WINDOW2_PREPARE_POSE["left_arm"],
        )
        self.assertEqual(
            joint_calls[5]["right"],
            emergency_zyt.settings.WINDOW2_PREPARE_POSE["right_arm"],
        )
        self.assertEqual(
            [call["left_motion_id"] for call in cartesian_calls],
            list(range(1, 17)),
        )
        window2_calls = cartesian_calls[7:]
        self.assertTrue(
            np.allclose(window2_calls[2]["left"][:3], [0.4, 0.02, 0.2])
        )
        self.assertTrue(
            np.allclose(window2_calls[3]["left"][:3], [0.4, 0.1, 0.24])
        )
        self.assertTrue(
            np.allclose(window2_calls[4]["left"][:3], [0.4, 0.07, 0.2])
        )
        self.assertTrue(
            np.allclose(window2_calls[5]["left"][:3], [0.41, 0.1, 0.2])
        )
        self.assertTrue(
            np.allclose(window2_calls[7]["left"][:3], [0.4, 0.12, 0.2])
        )
        self.assertTrue(
            np.allclose(window2_calls[8]["left"][:3], [0.4, 0.1, 0.23])
        )
        self.assertEqual(
            station.actions,
            [
                {"hand": {"position": {"left": [1.0], "right": [0.0]}}},
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}},
            ],
        )
        self.assertEqual(
            synced_actions,
            [{'waist': {'position': [0.0, 0.02]}, 'leg': {'position': [-0.02, 0.02]}}],
        )

    def test_task2_window1_failure_continues_to_window2(self):
        station = _Station()
        navigation = _Navigation()
        window2_calls = []
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "run_task2_window1",
                side_effect=RuntimeError("moveTo requires AT_NODE state"),
            ),
            patch.object(
                emergency_zyt,
                "run_task2_window2",
                side_effect=lambda *_args, **_kwargs: window2_calls.append(True)
                or True,
            ),
            patch.object(
                emergency_zyt,
                "apply_startpoint_pose",
                return_value=True,
            ),
        ):
            succeeded = emergency_zyt.run_task2_windows(station, navigation)

        self.assertTrue(succeeded)
        self.assertEqual(window2_calls, [True])
        self.assertEqual(
            navigation.calls,
            [("h270_task2_check", {"correct": True, "rotate": False})],
        )

    def test_task2_window2_failure_still_goes_to_check(self):
        station = _Station()
        navigation = _Navigation()
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "run_task2_window1",
                return_value=True,
            ),
            patch.object(
                emergency_zyt,
                "run_task2_window2",
                side_effect=RuntimeError("window2 moveTo failed"),
            ),
        ):
            succeeded = emergency_zyt.run_task2_windows(station, navigation)
        self.assertFalse(succeeded)
        self.assertEqual(
            navigation.calls,
            [("h270_task2_check", {"correct": True, "rotate": False})],
        )

    def test_task3_follows_standalone_test_sequence(self):
        station = _Station()
        navigation = _Navigation()
        joint_calls = []
        synced_actions = []
        grasp_calls = []
        with (
            patch("builtins.input", return_value="") as prompt,
            patch.object(
                emergency_zyt,
                "moveit_joints",
                side_effect=lambda current_station, **kwargs: joint_calls.append(
                    kwargs
                )
                or True,
            ),
            patch.object(
                emergency_zyt,
                "execute_synced_waist_leg_action",
                side_effect=lambda current_station, action, **kwargs: (
                    synced_actions.append(action)
                ),
            ),
            patch.object(
                emergency_zyt,
                "grasp_black_single",
                side_effect=lambda current_station, arm, hands: grasp_calls.append(
                    (arm, hands)
                )
                or {"hand_executed": True},
            ),
            patch.object(emergency_zyt.time, "sleep"),
        ):
            emergency_zyt.run_task3_belts(
                station,
                navigation,
            )

        self.assertEqual(prompt.call_count, 4)
        self.assertEqual(
            navigation.calls,
            [
                ("h270_task3_check", {"correct": True}),
                ("h270_task3_right", {"correct": True}),
                ("h270_task3_left", {"correct": True}),
                ("h270_task3_back", {"correct": True}),
                ("h270_task3_return", {"correct": True}),
                ("h270_task3_place", {"correct": True}),
                ("moveBy", 0, 0.3, 0, {}),
                ("h270_task3_finish", {"correct": False}),
            ],
        )
        self.assertEqual(len(joint_calls), 3)
        self.assertEqual(len(synced_actions), 3)
        self.assertEqual(
            grasp_calls,
            [
                ("right", {"left": 0.0, "right": 0.0}),
                ("left", {"left": 0.0, "right": 1.0}),
            ],
        )
        self.assertEqual(
            station.actions,
            [
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}},
                {"hand": {"position": {"left": [0.0], "right": [1.0]}}},
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}},
            ],
        )

    def test_task3_grasp_failure_goes_to_finish(self):
        station = _Station()
        navigation = _Navigation()
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "run_task3_grasp_belts",
                return_value=False,
            ),
            patch.object(
                emergency_zyt,
                "run_task3_pull_belts",
            ) as pull,
        ):
            succeeded = emergency_zyt.run_task3_belts(station, navigation)
        self.assertFalse(succeeded)
        pull.assert_not_called()
        self.assertEqual(
            navigation.calls,
            [
                ("h270_task3_check", {"correct": True}),
                ("h270_task3_finish", {"correct": False}),
            ],
        )

    def test_task3_pull_failure_still_goes_to_finish(self):
        station = _Station()
        navigation = _Navigation()
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "run_task3_grasp_belts",
                return_value=True,
            ),
            patch.object(
                emergency_zyt,
                "run_task3_pull_belts",
                return_value=False,
            ),
        ):
            succeeded = emergency_zyt.run_task3_belts(station, navigation)
        self.assertFalse(succeeded)
        self.assertEqual(
            navigation.calls,
            [
                ("h270_task3_check", {"correct": True}),
                ("h270_task3_finish", {"correct": False}),
            ],
        )

    def test_retry_helper_does_not_skip_when_user_types_s(self):
        attempts = []

        def fail_action():
            attempts.append(1)
            return False

        with patch("builtins.input", return_value="s") as prompt:
            succeeded = emergency_zyt.run_with_retry_or_skip("任务2开窗", fail_action)
        self.assertFalse(succeeded)
        self.assertEqual(len(attempts), emergency_zyt.MAX_STAGE_ATTEMPTS)
        self.assertEqual(prompt.call_count, emergency_zyt.MAX_STAGE_ATTEMPTS)
        self.assertTrue(
            all("不允许" in call.args[0] for call in prompt.call_args_list)
        )

    def test_run_until_success_retries_same_task_until_complete(self):
        attempts = []

        def succeed_on_fourth():
            attempts.append(1)
            return len(attempts) >= 4

        with patch("builtins.input", return_value=""):
            emergency_zyt.run_until_success("任务2开窗", succeed_on_fourth)
        self.assertEqual(len(attempts), 4)

    def test_task2_success_goes_to_check_after_both_windows(self):
        station = _Station()
        navigation = _Navigation()
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                emergency_zyt,
                "run_task2_window1",
                return_value=True,
            ),
            patch.object(
                emergency_zyt,
                "run_task2_window2",
                return_value=True,
            ),
            patch.object(
                emergency_zyt,
                "apply_startpoint_pose",
                return_value=True,
            ),
        ):
            succeeded = emergency_zyt.run_task2_windows(station, navigation)
        self.assertTrue(succeeded)
        self.assertEqual(
            navigation.calls,
            [("h270_task2_check", {"correct": True, "rotate": False})],
        )


if __name__ == "__main__":
    unittest.main()
