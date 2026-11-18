"""纸箱接近后抱持、拖行和回位流程的离线顺序测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import test_box


class _Station:
    def __init__(self):
        self.actions = []

    def execute_action(self, action):
        self.actions.append(action)


class _Navigation:
    def __init__(self):
        self.calls = []

    def turnTo(self, theta):
        self.calls.append(("turnTo", theta))

    def moveTo(self, node, **kwargs):
        self.calls.append(("moveTo", node, kwargs))

    def moveBy(self, x_m, y_m, theta_deg):
        self.calls.append(("moveBy", x_m, y_m, theta_deg))

    def recoverState(self, node=None, **kwargs):
        self.calls.append(("recoverState", node, kwargs))


class BoxFlowTests(unittest.TestCase):
    def test_box_flow_command_order(self):
        station = _Station()
        navigation = _Navigation()
        pose_calls = []

        def execute_pose(current_station, pose):
            pose_calls.append((current_station, pose))
            return True

        with (
            patch("builtins.input", return_value=""),
            patch.object(test_box, "_execute_object_pose", execute_pose),
            patch.object(test_box.time, "sleep"),
        ):
            test_box.run_box_after_approach(station, navigation)

        self.assertEqual(len(station.actions), 1)
        self.assertEqual(len(pose_calls), 4)
        self.assertEqual(
            navigation.calls,
            [
                ("turnTo", 270.0),
                ("recoverState", "startpoint", {"method": "move_by"}),
                ("turnTo", 0.0),
                ("moveBy", 0.0, -0.3, 0.0),
                ("turnTo", 90.0),
                ("moveBy", 0.8, 0.0, 0.0),
                ("recoverState", "startpoint", {"method": "move_by"}),
                ("turnTo", 0.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
