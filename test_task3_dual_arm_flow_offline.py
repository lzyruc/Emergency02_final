"""双臂任务三主流程离线测试；不导入硬件依赖，不发送运动命令。"""

from __future__ import annotations

import ast
from datetime import datetime
import json
import math
from pathlib import Path
import types
import unittest


SUBJECT = Path(__file__).with_name("game_emergency_test.py")
FUNCTION_NAMES = (
    "apply_task3_dual_grasp_pose",
    "grasp_task3_dual_arm_with_retries",
    "run_task3_dual_arm_belts",
)


class PosePath:
    def read_text(self, *, encoding: str) -> str:
        return json.dumps(
            {
                "head": [0.0, 0.0, 0.0],
                "waist": [0.0, 0.0],
                "leg": [0.0, 0.0],
                "left_arm": [0.0] * 7,
                "right_arm": [0.0] * 7,
            }
        )


class Station:
    def __init__(self) -> None:
        self.actions: list[dict] = []

    def execute_action(self, action: dict) -> None:
        self.actions.append(action)

    def get_robot_state(self) -> dict:
        return {
            "arm_joints": {
                "left": [0.0] * 7,
                "right": [0.0] * 7,
            }
        }


class Navigation:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def moveTo(self, node: str, **kwargs: object) -> None:
        self.calls.append(("moveTo", node, kwargs))

    def moveBy(self, x: float, y: float, theta: float, **kwargs: object) -> None:
        self.calls.append(("moveBy", x, y, theta, kwargs))

    def recoverState(self, node: str, **kwargs: object) -> None:
        self.calls.append(("recoverState", node, kwargs))


def load_flow(grasp_results: dict[str, list[object]]) -> tuple[object, dict]:
    tree = ast.parse(SUBJECT.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    calls: dict[str, list] = {"grasp": [], "moveit": [], "body": []}

    def grasp_black_single(
        station: object,
        arm: str,
        hand_positions: dict[str, float],
        **kwargs: object,
    ) -> object:
        calls["grasp"].append((arm, dict(hand_positions), kwargs))
        result = grasp_results[arm].pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def moveit_joints(station: object, **kwargs: object) -> bool:
        calls["moveit"].append(kwargs)
        return True

    def execute_body(station: object, action: dict) -> None:
        calls["body"].append(action)

    namespace = {
        "Any": object,
        "NavigationGraph": object,
        "Path": Path,
        "datetime": datetime,
        "json": json,
        "math": math,
        "time": types.SimpleNamespace(sleep=lambda _: None),
        "settings": types.SimpleNamespace(
            RUNTIME_DIR=Path("/runtime"),
            TASK={"hands": {"right_close_position": 1.0}},
        ),
        "grasp_black_single": grasp_black_single,
        "moveit_joints": moveit_joints,
        "execute_synced_waist_leg_action": execute_body,
        "TASK3_DUAL_GRASP_POSE_PATH": PosePath(),
        "TASK3_DUAL_BODY_TO_ARM_DELAY_S": 0.5,
        "TASK3_DUAL_CAMERA_SETTLE_S": 3.0,
        "TASK3_DUAL_PICK_NODE": "h225_task3_pick",
        "TASK3_DUAL_RETURN_NODE": "h180_task3_dual_return",
        "TASK3_DUAL_PLACE_NODE": "h180_task3_dual_place",
        "TASK3_DUAL_RETRY_OFFSETS_Y_M": (0.0, 0.10, -0.10),
    }
    exec(compile(module, str(SUBJECT), "exec"), namespace)
    return namespace["run_task3_dual_arm_belts"], calls


def success_result() -> dict:
    return {
        "motion_succeeded": True,
        "hand_executed": True,
    }


class Task3DualArmFlowTests(unittest.TestCase):
    def test_test_entry_calls_new_flow_and_keeps_old_flow_unused(self) -> None:
        tree = ast.parse(SUBJECT.read_text(encoding="utf-8"))
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        called_names = {
            node.func.id
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("run_task3_dual_arm_belts", called_names)
        self.assertNotIn("run_task3_belts", called_names)

    def test_success_uses_verified_nodes_and_leftmost_left_target(self) -> None:
        flow, calls = load_flow(
            {
                "right": [RuntimeError("miss"), success_result()],
                "left": [success_result()],
            }
        )
        station = Station()
        navigation = Navigation()
        self.assertTrue(flow(station, navigation))

        self.assertEqual(
            navigation.calls,
            [
                ("moveTo", "h270_task3_check", {"correct": True}),
                ("moveTo", "h225_task3_pick", {"correct": True}),
                ("moveBy", 0.0, 0.10, 0.0, {"correct": False}),
                (
                    "recoverState",
                    "h225_task3_pick",
                    {"method": "move_by"},
                ),
                ("moveBy", -0.30, 0.0, -45.0, {"correct": False}),
                ("moveTo", "h180_task3_dual_return", {"correct": True}),
                ("moveTo", "h180_task3_dual_place", {"correct": False}),
                ("moveBy", -0.50, 0.0, 0.0, {"correct": False}),
                (
                    "recoverState",
                    "h180_task3_dual_return",
                    {"method": "move_by"},
                ),
            ],
        )
        selections = [call[2]["component_selection"] for call in calls["grasp"]]
        self.assertEqual(selections, ["nearest_tcp", "nearest_tcp", "leftmost"])
        spread = calls["moveit"][-2]
        self.assertAlmostEqual(spread["left"][1], math.radians(30.0))
        self.assertAlmostEqual(spread["right"][1], math.radians(-30.0))

    def test_total_grasp_failure_stays_at_pick_and_does_not_place(self) -> None:
        flow, _ = load_flow(
            {
                "right": [None, None, None],
                "left": [None, None, None],
            }
        )
        station = Station()
        navigation = Navigation()
        self.assertFalse(flow(station, navigation))
        move_to_nodes = [call[1] for call in navigation.calls if call[0] == "moveTo"]
        self.assertEqual(move_to_nodes, ["h270_task3_check", "h225_task3_pick"])
        self.assertNotIn("h180_task3_dual_return", move_to_nodes)
        self.assertNotIn("h180_task3_dual_place", move_to_nodes)
        self.assertEqual(
            navigation.calls[-1],
            ("recoverState", "h225_task3_pick", {"method": "move_by"}),
        )


if __name__ == "__main__":
    unittest.main()
