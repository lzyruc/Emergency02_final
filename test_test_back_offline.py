"""test_back.py 的无硬件结构测试。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).with_name("test_back.py")


class TestBackScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_does_not_import_navigation(self) -> None:
        imported_modules = {
            node.module
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertNotIn("agent.navigation", imported_modules)

    def test_station_disables_force_sensors(self) -> None:
        station_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TienYi2ProRos2Station"
        ]
        self.assertEqual(len(station_calls), 1)
        force_sensor_keywords = [
            keyword
            for keyword in station_calls[0].keywords
            if keyword.arg == "force_sensor_dict"
        ]
        self.assertEqual(len(force_sensor_keywords), 1)
        self.assertIsInstance(force_sensor_keywords[0].value, ast.Dict)
        self.assertEqual(force_sensor_keywords[0].value.keys, [])

    def test_every_live_stage_has_an_enter_gate(self) -> None:
        input_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input"
        ]
        self.assertEqual(len(input_calls), 9)
        for step in range(1, 10):
            self.assertIn(f"步骤{step}/9", self.source)

    def test_uses_pose3_recognition_and_station_shutdown(self) -> None:
        self.assertIn("POSE3_PATH = CHAIR_BACK_POSE_PATHS[-1]", self.source)
        self.assertIn("recognize_chair_backtop_grasp(", self.source)
        self.assertIn("plan_chair_backtop_grasp_steps(", self.source)
        self.assertIn("station.shutdown()", self.source)


if __name__ == "__main__":
    unittest.main()
