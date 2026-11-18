#!/usr/bin/env python3
"""Offline tests for synchronized xrocs waist/leg station actions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from waist_leg_motion import execute_synced_waist_leg_action


class FakeMotorController:
    def __init__(self) -> None:
        self.speed_calls: list[list[float]] = []

    def set_motor_config(self, *, spd: list[float]) -> None:
        self.speed_calls.append(spd)


class FakeRobotState:
    def get_waist_pos_list(self) -> list[float]:
        return [0.1, 0.2]

    def get_leg_pos_list(self) -> list[float]:
        return [-0.01, 0.01]


class FakeStation:
    def __init__(self) -> None:
        self.robot = type(
            "FakeRobot",
            (),
            {
                "state": FakeRobotState(),
                "waist_ctrler": FakeMotorController(),
                "leg_ctrler": FakeMotorController(),
            },
        )()
        self.actions: list[dict] = []

    def get_robot_handle(self) -> dict:
        return {"robot": self.robot}

    def execute_action(self, action: dict) -> None:
        self.actions.append(action)


class WaistLegMotionTest(unittest.TestCase):
    @patch("waist_leg_motion.time.sleep", return_value=None)
    def test_waist_only_action_does_not_touch_leg(self, sleep: object) -> None:
        station = FakeStation()
        action = {"waist": {"position": [0.6, 0.2]}}
        execute_synced_waist_leg_action(station, action)
        self.assertEqual(station.actions, [action])
        self.assertEqual(station.robot.waist_ctrler.speed_calls, [[0.6, 0.3], [0.5, 0.3]])
        self.assertEqual(station.robot.leg_ctrler.speed_calls, [])

    @patch("waist_leg_motion.time.sleep", return_value=None)
    def test_combined_action_preserves_json_order(self, sleep: object) -> None:
        station = FakeStation()
        action = {
            "waist": {"position": [0.4, 0.5]},
            "leg": {"position": [-0.02, 0.02]},
        }
        execute_synced_waist_leg_action(station, action)
        self.assertEqual(station.actions, [action])
        self.assertEqual(station.robot.waist_ctrler.speed_calls[-1], [0.5, 0.3])
        self.assertEqual(station.robot.leg_ctrler.speed_calls[-1], [0.3, 0.3])


if __name__ == "__main__":
    unittest.main()
