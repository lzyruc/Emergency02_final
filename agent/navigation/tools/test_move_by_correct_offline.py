#!/usr/bin/env python3
"""Offline tests for optional moveBy pose correction; sends no chassis commands."""

from __future__ import annotations

import math
from typing import Dict, List
import unittest
from unittest.mock import Mock

from agent.navigation.graph_models import Node, StateStatus
from agent.navigation.map_slam import MapSLAM
from agent.navigation.navigation_graph import NavigationGraph
from agent.navigation.rotation_control import rotate_by_with_feedback


class ActionFailed(Exception):
    class Code:
        value = 2001

    code = Code()


class ActionTimeout(Exception):
    class Code:
        value = 1001

    code = Code()


class TimeoutController:
    def __init__(self) -> None:
        self.cancelled: List[List[str]] = []
        self.moves: List[dict] = []

    def cancel_action(self, action_ids: List[str]) -> None:
        self.cancelled.append(action_ids)

    def move_by(self, **kwargs: float) -> dict:
        self.moves.append(dict(kwargs))
        return {"status": "completed"}


class TimeoutRotationController(TimeoutController):
    def __init__(self) -> None:
        super().__init__()
        self.poses = iter(
            [
                {"x": 0.0, "y": 0.0, "yaw": 0.0},
                {"x": 0.0, "y": 0.0, "yaw": math.pi / 2},
            ]
        )

    def move_by(self, **kwargs: float) -> dict:
        self.moves.append(dict(kwargs))
        if len(self.moves) == 1:
            raise ActionTimeout("Action rotate-1 timed out after 10 seconds")
        return {"status": "completed"}

    def get_pose(self) -> Dict[str, float]:
        return dict(next(self.poses))


class FakeMapSLAM:
    def __init__(self, poses: List[Dict[str, float]]) -> None:
        self.poses = iter(poses)

    def getCurrentPose(self) -> Dict[str, float]:
        return dict(next(self.poses))

    def mapPoseToLogical(self, pose: Dict[str, float]) -> Dict[str, float]:
        return {
            "x_m": pose["x_m"],
            "y_m": pose["y_m"],
            "theta_deg": math.degrees(pose["yaw_rad"]),
        }

    def logicalPoseToMap(self, pose: Dict[str, float]) -> Dict[str, float]:
        return {
            "x_m": pose["x_m"],
            "y_m": pose["y_m"],
            "yaw_rad": math.radians(pose["theta_deg"]),
        }


def make_graph(poses: List[Dict[str, float]]) -> NavigationGraph:
    graph = object.__new__(NavigationGraph)
    graph._state = StateStatus.AT_NODE
    graph._current_node = Node(
        name="anchor",
        pose={"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
    )
    graph._nodes = {"anchor": graph._current_node}
    graph._logical_pose = {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0}
    graph._adjustment_origin_recorded = True
    graph._last_warnings = []
    graph.linear_speed = 0.4
    graph.angular_speed = 0.3
    graph.lateral_translation_headings = {}
    graph._map_slam = FakeMapSLAM(poses)
    graph._execute_action = lambda action, **kwargs: None
    return graph


class MoveByCorrectTest(unittest.TestCase):
    def test_map_name_mismatch_is_warning_not_error(self) -> None:
        controller = Mock()
        controller.get_current_map.return_value = {"map_name": "active-map"}
        map_slam = MapSLAM(
            "configured-map",
            {
                "anchor": Node(
                    name="anchor",
                    pose={"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
                )
            },
            {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            lambda: controller,
        )

        self.assertFalse(map_slam.requireCurrentMap())

    def test_move_to_per_call_speed_override_reaches_edge_action(self) -> None:
        graph = make_graph(
            [{"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}]
        )
        graph.navigation_mode = "move_by"
        graph._route = {"scene_editor": {}}
        graph._nodes = {
            "anchor": graph._current_node,
            "target": Node(
                name="target",
                pose={"x_m": 1.0, "y_m": 0.0, "theta_deg": 0.0},
            ),
        }
        graph.shortestPath = lambda node: ["anchor", node]
        graph.executionPath = lambda node: ["anchor", node]
        graph._capture_recovery_snapshot = lambda reason: None
        calls: List[dict] = []
        graph._execute_action = lambda action, **kwargs: calls.append(kwargs)

        graph.moveTo(
            "target",
            correct=False,
            linear_speed=0.08,
            angular_speed=0.12,
        )

        self.assertEqual(calls[0]["linear_speed"], 0.08)
        self.assertEqual(calls[0]["angular_speed"], 0.12)

    def test_per_call_speed_override_reaches_action(self) -> None:
        graph = make_graph([])
        calls: List[dict] = []
        graph._execute_action = lambda action, **kwargs: calls.append(kwargs)

        graph.moveBy(
            0.1,
            0.0,
            0.0,
            correct=False,
            linear_speed=0.05,
            angular_speed=0.1,
        )

        self.assertEqual(calls[0]["linear_speed"], 0.05)
        self.assertEqual(calls[0]["angular_speed"], 0.1)

    def test_explicit_false_does_not_read_pose_or_correct(self) -> None:
        graph = make_graph([])
        graph._return_to_map_pose_with_move_by = lambda target, **kwargs: self.fail(
            "default moveBy unexpectedly corrected its pose"
        )

        pose = graph.moveBy(0.8, 0.0, 0.0, correct=False)

        self.assertEqual(pose, {"x_m": 0.8, "y_m": 0.0, "theta_deg": 0.0})
        self.assertEqual(graph._state, StateStatus.AT_NODE)

    def test_default_correct_uses_measured_body_frame_target(self) -> None:
        graph = make_graph(
            [
                {"x_m": 10.0, "y_m": 20.0, "yaw_rad": math.pi / 2},
                {"x_m": 10.12, "y_m": 21.0, "yaw_rad": math.pi / 2},
            ]
        )
        corrected_targets: List[Dict[str, float]] = []
        graph._return_to_map_pose_with_move_by = lambda target, **kwargs: (
            corrected_targets.append(dict(target)) or dict(target)
        )

        graph.moveBy(1.0, 0.0, 0.0)

        self.assertEqual(len(corrected_targets), 1)
        self.assertAlmostEqual(corrected_targets[0]["x_m"], 10.0)
        self.assertAlmostEqual(corrected_targets[0]["y_m"], 21.0)
        self.assertAlmostEqual(corrected_targets[0]["yaw_rad"], math.pi / 2)

    def test_corrects_error_outside_previous_correction_range(self) -> None:
        graph = make_graph(
            [
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                {"x_m": 0.8, "y_m": 0.0, "yaw_rad": 0.0},
            ]
        )
        corrected_targets: List[Dict[str, float]] = []
        graph._return_to_map_pose_with_move_by = lambda target, **kwargs: (
            corrected_targets.append(dict(target)) or dict(target)
        )

        graph.moveBy(1.0, 0.0, 0.0, correct=True)

        self.assertEqual(len(corrected_targets), 1)
        self.assertEqual(graph._state, StateStatus.AT_NODE)
        self.assertEqual(graph._current_node.name, "anchor")

    def test_four_round_nonconvergence_is_warning(self) -> None:
        graph = make_graph(
            [
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            ]
        )
        final_pose = graph._return_to_map_pose_with_move_by(
            {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
            raise_on_nonconvergence=False,
        )

        self.assertEqual(final_pose["x_m"], 0.0)
        self.assertEqual(
            graph._last_warnings[-1]["category"],
            "move_by_pose_correction_not_converged",
        )
        self.assertEqual(graph._last_warnings[-1]["correction_rounds"], 4)

    def test_move_by_adopts_nearest_node_after_failed_state(self) -> None:
        graph = make_graph(
            [
                {"x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0},
                {"x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0},
            ]
        )
        graph._state = StateStatus.FAILED
        graph._current_node = None

        pose = graph.moveBy(0.1, 0.0, 0.0, correct=False)

        self.assertEqual(graph._state, StateStatus.AT_NODE)
        self.assertEqual(graph._current_node.name, "anchor")
        self.assertAlmostEqual(pose["x_m"], 0.3)
        self.assertEqual(
            graph._last_warnings[-1]["category"],
            "semantic_node_adopted_without_tolerance",
        )

    def test_large_pre_correction_error_is_corrected_without_range_error(self) -> None:
        graph = make_graph(
            [{"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0}]
        )
        corrected_targets: List[Dict[str, float]] = []
        graph._return_to_map_pose_with_move_by = lambda target, **kwargs: (
            corrected_targets.append(dict(target)) or dict(target)
        )

        graph._correct_expected_state()

        self.assertEqual(len(corrected_targets), 1)

    def test_repeated_2001_continues_with_correction(self) -> None:
        graph = make_graph(
            [{"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}]
        )
        graph._execute_action = Mock(side_effect=ActionFailed("action failed"))
        corrected_targets: List[Dict[str, float]] = []
        graph._return_to_map_pose_with_move_by = lambda target, **kwargs: (
            corrected_targets.append(dict(target)) or dict(target)
        )

        graph.moveBy(1.0, 0.0, 0.0)

        self.assertEqual(len(corrected_targets), 1)
        self.assertEqual(graph._state, StateStatus.AT_NODE)
        self.assertEqual(
            graph._last_warnings[-1]["category"],
            "move_by_action_failed_continued_with_pose_correction",
        )

    def test_translation_timeout_is_cancelled_and_retried(self) -> None:
        graph = make_graph(
            [{"x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0}]
        )
        graph.linear_speed = 0.4
        graph.angular_speed = 0.3
        controller = TimeoutController()

        graph._retry_failed_translation(
            ActionTimeout("Action move-1 timed out after 10 seconds"),
            {"x_m": 1.0, "y_m": 0.0},
            controller,
            0.05,
            0.1,
        )

        self.assertEqual(controller.cancelled, [["move-1"]])
        self.assertEqual(controller.moves[0]["distance"], 800)
        self.assertEqual(controller.moves[0]["v"], 0.05)
        self.assertEqual(controller.moves[0]["w"], 0.1)
        self.assertEqual(graph._last_warnings[-1]["error_code"], 1001)

    def test_rotation_timeout_is_cancelled_and_retried(self) -> None:
        controller = TimeoutRotationController()

        report = rotate_by_with_feedback(
            controller,
            math.pi / 2,
            angular_speed=0.3,
            initial_map_pose={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            retry_action_failed=True,
        )

        self.assertEqual(controller.cancelled, [["rotate-1"]])
        self.assertEqual(len(controller.moves), 2)
        self.assertTrue(report["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
