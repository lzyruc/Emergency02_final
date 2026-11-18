"""长程识别导航的融合与接近规划离线测试。"""

from __future__ import annotations

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

import test_target_navigation as target_navigation
from test_target_navigation import (
    filter_targets_by_hits,
    fuse_targets,
    plan_standoff_move,
    select_safe_approach,
)


class _Avoid:
    def __init__(self, results):
        self.results = iter(results)

    def checkCollision(self, dx, dy, dtheta):
        return next(self.results)


class TargetNavigationTests(unittest.TestCase):
    def test_fuse_and_choose_standoff(self):
        frames = []
        for x_m, confidence in ((2.0, 0.8), (2.2, 0.9), (2.1, 1.0)):
            frames.append(
                [
                    {
                        "type": "filebox",
                        "confidence": confidence,
                        "depth_m": x_m,
                        "point_base_m": np.asarray([x_m, 0.0, 0.1]),
                        "distance_xy_m": x_m,
                    }
                ]
            )
        target = fuse_targets(frames)[0]
        self.assertEqual(target["hits"], 3)
        self.assertAlmostEqual(target["point_base_m"][0], 2.1)
        move_x, move_y, turn_deg = plan_standoff_move(target)
        self.assertAlmostEqual(move_x, 1.6)
        self.assertAlmostEqual(move_y, 0.0)
        self.assertAlmostEqual(turn_deg, 0.0)

    def test_lateral_bearing(self):
        move_x, move_y, turn_deg = plan_standoff_move(
            {"point_base_m": np.asarray([0.0, 1.0, 0.0])}
        )
        self.assertAlmostEqual(move_x, 0.0)
        self.assertAlmostEqual(move_y, 0.5)
        self.assertAlmostEqual(turn_deg, 90.0)

    def test_nearest_collision_selects_next_safe_target(self):
        targets = [
            {
                "name": "文件筐",
                "type": "filebox",
                "point_base_m": np.asarray([1.0, 0.0, 0.0]),
                "distance_xy_m": 1.0,
            },
            {
                "name": "纸箱",
                "type": "box",
                "point_base_m": np.asarray([2.0, 0.0, 0.0]),
                "distance_xy_m": 2.0,
            },
        ]
        target, motion = select_safe_approach(_Avoid([True, False]), targets)
        self.assertEqual(target["name"], "纸箱")
        self.assertAlmostEqual(motion[0], 1.65)

    def test_box_uses_35cm_standoff(self):
        target = {
            "name": "纸箱",
            "type": "box",
            "point_base_m": np.asarray([1.0, 0.0, 0.0]),
            "distance_xy_m": 1.0,
        }
        selected, motion = select_safe_approach(_Avoid([False]), [target])
        self.assertIs(selected, target)
        self.assertAlmostEqual(motion[0], 0.65)

    def test_filebox_uses_bin_pose_and_grasps_without_distance_limit(self):
        candidate = {
            "arm": "right",
            "confidence": 0.9,
            "selection_score": 0.8,
            "pixel_uv": [640, 360],
            "point_source": "grasp_line",
            "depth_m": 0.3,
            "depth_source": "抓取点5x5中值",
            "tcp_distance_m": 0.8,
            "target_pose_waist_xyz_xyzw": np.asarray(
                [0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]
            ),
        }
        executed_poses = []
        cartesian_calls = []
        synced_actions = []
        arm_calls = []

        class Station:
            def __init__(self):
                self.actions = []

            def execute_action(self, action):
                self.actions.append(action)

        class Navigation:
            def __init__(self):
                self.calls = []

            def recoverState(self, node, **kwargs):
                self.calls.append((node, kwargs))

            def moveBy(self, x_m, y_m, theta_deg):
                self.calls.append(("moveBy", x_m, y_m, theta_deg))

            def turnTo(self, theta_deg):
                self.calls.append(("turnTo", theta_deg))

        station = Station()
        navigation = Navigation()
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                target_navigation,
                "_execute_object_pose",
                side_effect=lambda station, pose: executed_poses.append(pose) or True,
            ),
            patch.object(
                target_navigation,
                "wrist_grasp_candidates",
                return_value=[candidate],
            ),
            patch.object(
                target_navigation,
                "moveit_cartesians",
                side_effect=lambda current_station, **kwargs: cartesian_calls.append(
                    kwargs
                )
                or True,
            ),
            patch.object(
                target_navigation,
                "execute_synced_waist_leg_action",
                side_effect=lambda current_station, action: synced_actions.append(
                    action
                ),
            ),
            patch.object(
                target_navigation,
                "moveit_joints",
                side_effect=lambda current_station, **kwargs: arm_calls.append(kwargs)
                or True,
            ),
            patch.object(target_navigation.time, "sleep"),
        ):
            succeeded = target_navigation.grasp_filebox_or_bin_from_bin_pose(
                station,
                navigation,
                {"left": object(), "right": object()},
                {"left": object(), "right": object()},
                "filebox",
            )
        self.assertTrue(succeeded)
        self.assertEqual(len(executed_poses), 2)
        self.assertEqual(
            cartesian_calls,
            [{"right": [candidate["target_pose_waist_xyz_xyzw"].tolist()]}],
        )
        self.assertEqual(len(station.actions), 3)
        self.assertEqual(station.actions[1]["hand"]["position"]["right"], [1.0])
        self.assertEqual(station.actions[2]["hand"]["position"]["right"], [0.0])
        self.assertEqual(len(synced_actions), 1)
        self.assertEqual(len(arm_calls), 1)
        self.assertEqual(
            navigation.calls,
            [
                ("startpoint", {"method": "move_by"}),
                ("moveBy", 0.0, -0.3, 0.0),
                ("turnTo", 90.0),
                ("moveBy", 0.8, 0.0, 0.0),
                ("startpoint", {"method": "move_by"}),
                ("turnTo", 0.0),
            ],
        )

    def test_bin_uses_bin_wrist_target_and_bin_lift_pose(self):
        candidate = {
            "arm": "left",
            "confidence": 0.9,
            "selection_score": 0.8,
            "pixel_uv": [640, 360],
            "point_source": "rim",
            "depth_m": 0.3,
            "depth_source": "抓取点5x5中值",
            "tcp_distance_m": 0.8,
            "target_pose_waist_xyz_xyzw": np.asarray(
                [0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0]
            ),
        }
        station = Mock()
        navigation = Mock()
        with (
            patch("builtins.input", return_value=""),
            patch.object(target_navigation, "_execute_object_pose", return_value=True),
            patch.object(
                target_navigation,
                "wrist_grasp_candidates",
                return_value=[candidate],
            ) as wrist_candidates,
            patch.object(target_navigation, "moveit_cartesians", return_value=True),
            patch.object(
                target_navigation,
                "execute_synced_waist_leg_action",
            ) as synced_action,
            patch.object(target_navigation, "moveit_joints", return_value=True),
            patch.object(target_navigation.time, "sleep"),
        ):
            succeeded = target_navigation.grasp_filebox_or_bin_from_bin_pose(
                station,
                navigation,
                {"left": object(), "right": object()},
                {"left": object(), "right": object()},
                "bin",
            )

        self.assertTrue(succeeded)
        self.assertEqual(wrist_candidates.call_args.args[1], "bin")
        bin_lift_pose = target_navigation.json.loads(
            target_navigation.BIN_LIFT_PATH.read_text(encoding="utf-8")
        )
        synced_action.assert_called_once_with(
            station,
            {
                "head": {"position": bin_lift_pose["head"]},
                "waist": {"position": bin_lift_pose["waist"]},
                "leg": {"position": bin_lift_pose["leg"]},
            },
        )

    def test_chair_pose_recognition_stops_without_grasp(self):
        candidate = {
            "arm": "left",
            "confidence": 0.8,
            "selection_score": 0.7,
            "pixel_uv": [640, 360],
            "point_source": "grasp_line",
            "depth_m": 0.3,
            "depth_source": "抓取点5x5中值",
            "tcp_distance_m": 0.25,
            "target_pose_waist_xyz_xyzw": np.asarray(
                [0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0]
            ),
        }
        executed_poses = []
        with (
            patch("builtins.input", return_value=""),
            patch.object(
                target_navigation,
                "_execute_object_pose",
                side_effect=lambda station, pose: executed_poses.append(pose) or True,
            ),
            patch.object(
                target_navigation,
                "wrist_grasp_candidates",
                return_value=[candidate],
            ),
        ):
            selected = target_navigation.recognize_target_from_task1_poses(
                object(),
                "chair",
                target_navigation.CHAIR_PREGRASP_PATHS,
                {"left": object(), "right": object()},
                {"left": object(), "right": object()},
            )
        self.assertIs(selected, candidate)
        self.assertEqual(len(executed_poses), 1)

    def test_all_collisions_return_none(self):
        targets = [
            {
                "name": "文件筐",
                "type": "filebox",
                "point_base_m": np.asarray([1.0, 0.0, 0.0]),
                "distance_xy_m": 1.0,
            }
        ]
        self.assertIsNone(select_safe_approach(_Avoid([True]), targets))

    def test_only_five_of_five_hits_can_move(self):
        targets = [
            {"name": "办公转椅", "hits": 4},
            {"name": "垃圾桶", "hits": 5},
        ]
        self.assertEqual(
            [target["name"] for target in filter_targets_by_hits(targets)],
            ["垃圾桶"],
        )

    def test_startpoint_pose_zeroes_body_before_initial_pose(self):
        order = []
        pose = {
            "left_arm": [1.0] * 7,
            "right_arm": [2.0] * 7,
            "head": [3.0] * 3,
            "waist": [4.0] * 2,
            "leg": [5.0] * 2,
        }

        class Station:
            pass

        def move_arms(station, **kwargs):
            order.append(("arms", kwargs))
            return True

        def move_body(station, action):
            order.append(("synced_body", action))

        def sleep(seconds):
            order.append(("sleep", seconds))

        with (
            patch.object(target_navigation, "moveit_joints", move_arms),
            patch.object(
                target_navigation,
                "execute_synced_waist_leg_action",
                move_body,
            ),
            patch.object(target_navigation.time, "sleep", sleep),
        ):
            target_navigation.apply_startpoint_pose(Station(), pose)

        self.assertEqual(
            [item[0] for item in order],
            ["synced_body", "sleep", "arms", "synced_body", "sleep"],
        )
        self.assertEqual(
            order[0][1],
            {
                "waist": {"position": [0.0, 0.0]},
                "leg": {"position": [0.0, 0.0]},
            },
        )
        self.assertEqual(order[1][1], 5.0)
        self.assertEqual(order[2][1]["left"], pose["left_arm"])
        self.assertEqual(order[3][1]["waist"]["position"], pose["waist"])
        self.assertEqual(order[4][1], 1.0)


if __name__ == "__main__":
    unittest.main()
