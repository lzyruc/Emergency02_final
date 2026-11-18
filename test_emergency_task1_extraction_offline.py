"""验证独立任务1脚本与比赛入口中的任务1实现保持一致。

测试只解析 Python AST，不导入 xrocs，也不创建 station 或发送运动命令。
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from unittest.mock import patch
import unittest


ROOT = Path(__file__).resolve().parent
FULL_ENTRY = ROOT / "emergency_zyt.py"
TASK1_ENTRY = ROOT / "emergency_task1.py"
TASK1_FUNCTIONS = {
    "rotation_xyzw",
    "run_with_retry_or_skip",
    "_execute_object_pose",
    "apply_startpoint_pose",
    "recognize_box_orientation_until_ready",
    "run_box_after_approach",
    "center_depth_m",
    "recognize_frame",
    "fuse_targets",
    "plan_standoff_move",
    "select_safe_approach",
    "classify_chairs_in_channel",
    "find_task1_approach",
    "return_from_approach",
    "refine_task1_approach",
    "grasp_filebox_or_bin_from_bin_pose",
    "filter_targets_by_hits",
    "filter_targets_in_channel",
    "recognize_task1_batch",
    "run_task1_bin",
    "run_task1_filebox",
    "run_task1_chair",
    "run_task1_box",
    "run_task1_debris",
    "run_task1_final_chassis_clean",
}
AUTO_ADAPTED_FUNCTIONS = {
    "run_with_retry_or_skip",
    "find_task1_approach",
    "recognize_task1_batch",
    "run_task1_chair",
    "run_task1_box",
    "run_task1_debris",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


class NormalizeTask1Copy(ast.NodeTransformer):
    """忽略独立入口必须修改的交互提示文字。"""

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "input"
        ):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<text>"), node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == "moved_to_task2":
            node.id = "stopped_early"
        return node


class EmergencyTask1ExtractionTests(unittest.TestCase):
    def test_task1_function_bodies_match_full_entry(self) -> None:
        full_functions = functions(parse(FULL_ENTRY))
        task1_functions = functions(parse(TASK1_ENTRY))
        self.assertTrue(TASK1_FUNCTIONS <= task1_functions.keys())
        normalizer = NormalizeTask1Copy()
        for name in sorted(TASK1_FUNCTIONS - AUTO_ADAPTED_FUNCTIONS):
            full_node = normalizer.visit(ast.fix_missing_locations(full_functions[name]))
            task1_node = normalizer.visit(ast.fix_missing_locations(task1_functions[name]))
            self.assertEqual(
                ast.dump(full_node, include_attributes=False),
                ast.dump(task1_node, include_attributes=False),
                name,
            )

    def test_automatic_entry_contains_no_input_calls(self) -> None:
        input_calls = [
            node
            for node in ast.walk(parse(TASK1_ENTRY))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input"
        ]
        self.assertEqual(input_calls, [])

    def test_task1_retries_until_a_full_clearance_scan(self) -> None:
        task1_functions = functions(parse(TASK1_ENTRY))
        debris = task1_functions["run_task1_debris"]
        self.assertTrue(
            any(
                isinstance(node, ast.While)
                and isinstance(node.test, ast.Constant)
                and node.test.value is True
                for node in ast.walk(debris)
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "discard"
                for node in ast.walk(debris)
            )
        )

    def test_common_yolo_waits_three_seconds(self) -> None:
        for entry in (TASK1_ENTRY, FULL_ENTRY):
            tree = parse(entry)
            settle_value = None
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    if any(
                        isinstance(target, ast.Name)
                        and target.id == "TASK1_YOLO_SETTLE_S"
                        for target in node.targets
                    ):
                        settle_value = ast.literal_eval(node.value)
            self.assertEqual(settle_value, 3.0, entry.name)
            batch = functions(tree)["recognize_task1_batch"]
            self.assertTrue(
                any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "time"
                    and node.func.attr == "sleep"
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "TASK1_YOLO_SETTLE_S"
                    for node in ast.walk(batch)
                ),
                entry.name,
            )

    def test_bin_collision_tries_every_standoff_until_safe(self) -> None:
        tree = parse(TASK1_ENTRY)
        selected = functions(tree)
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                selected["plan_standoff_move"],
                selected["select_safe_approach"],
            ],
            type_ignores=[],
        )
        namespace = {
            "math": math,
            "DEFAULT_STANDOFF_M": 0.4,
            "CHAIR_STANDOFF_M_BY_VIEW": {"front": 0.8, "back": 0.6},
            "STANDOFF_M_BY_TYPE": {"box": 0.45},
            "TURN_REDUCTION_DEG_BY_TYPE": {"bin": 0.0, "filebox": 0.0},
            "BENT_APPROACH_STANDOFFS_M_BY_TYPE": {
                "bin": (0.50, 0.60, 0.70),
                "filebox": (0.40, 0.50, 0.60, 0.70),
            },
        }
        exec(
            compile(ast.fix_missing_locations(module), str(TASK1_ENTRY), "exec"),
            namespace,
        )

        class Avoid:
            def __init__(self) -> None:
                self.motions = []

            def planApproach(self, *motion):
                self.motions.append(motion)
                return [motion] if len(self.motions) == 3 else None

            def checkCollision(self, *motion):
                raise AssertionError("bin不应走通用直线碰撞分支")

        avoid = Avoid()
        result = namespace["select_safe_approach"](
            avoid,
            [{
                "type": "bin",
                "name": "垃圾桶",
                "point_base_m": [1.0, 0.0, 0.0],
                "distance_xy_m": 1.0,
            }],
        )
        self.assertIsNotNone(result)
        self.assertEqual(
            [round(motion[0], 2) for motion in avoid.motions],
            [0.50, 0.40, 0.30],
        )
        self.assertEqual(result[0]["approach_standoff_m"], 0.70)

    def test_retry_observation_moves_directly_without_collision_check(self) -> None:
        for entry in (TASK1_ENTRY, FULL_ENTRY):
            find_approach = functions(parse(entry))["find_task1_approach"]
            move_by_calls = [
                node
                for node in ast.walk(find_approach)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "moveBy"
            ]
            retry_moves = [
                node
                for node in move_by_calls
                if node.args
                and isinstance(node.args[0], ast.Starred)
                and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id == "STARTPOINT_RETRY_OBSERVATION_MOTION"
            ]
            self.assertEqual(len(retry_moves), 1, entry.name)
            self.assertEqual(
                {keyword.arg: ast.literal_eval(keyword.value)
                 for keyword in retry_moves[0].keywords},
                {"correct": False},
                entry.name,
            )
            self.assertFalse(any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "checkBentCollision"
                for node in ast.walk(find_approach)
            ), entry.name)
            recursive_calls = [
                node
                for node in ast.walk(find_approach)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "find_task1_approach"
            ]
            self.assertEqual(len(recursive_calls), 1, entry.name)
            self.assertIn(
                "blocked_targets_from_start",
                {keyword.arg for keyword in recursive_calls[0].keywords},
                entry.name,
            )

    def test_full_entry_task1_restores_confirmation_and_gates_task2(self) -> None:
        full_functions = functions(parse(FULL_ENTRY))
        confirmation_functions = {
            "_execute_object_pose",
            "apply_startpoint_pose",
            "run_box_after_approach",
            "find_task1_approach",
            "refine_task1_approach",
            "grasp_filebox_or_bin_from_bin_pose",
            "run_task1_chair",
            "run_task1_debris",
            "run_task1_and_go_to_task2_start",
        }
        for name in confirmation_functions:
            self.assertTrue(any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"
                for node in ast.walk(full_functions[name])
            ), name)
        debris = full_functions["run_task1_debris"]
        self.assertTrue(any(
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
            for node in ast.walk(debris)
        ))
        self.assertFalse(any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "discard"
            for node in ast.walk(debris)
        ))
        gate = full_functions["run_task1_and_go_to_task2_start"]
        gate_until_success = [
            node
            for node in ast.walk(gate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_until_success"
        ]
        self.assertEqual(len(gate_until_success), 3)
        for call in gate_until_success:
            automatic = next(
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "automatic"
            )
            self.assertIs(ast.literal_eval(automatic), True)
        debris_retry_calls = [
            node
            for node in ast.walk(debris)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_with_retry_or_skip"
        ]
        self.assertTrue(debris_retry_calls)
        for call in debris_retry_calls:
            automatic = next(
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "automatic"
            )
            self.assertIs(ast.literal_eval(automatic), True)
        until_success = full_functions["run_until_success"]
        self.assertEqual(
            sum(isinstance(node, ast.While) for node in ast.walk(until_success)),
            1,
        )

    def test_full_entry_main_gates_task2_and_task3_until_success(self) -> None:
        main = functions(parse(FULL_ENTRY))["main"]
        until_calls = [
            node.args[0].value
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_until_success"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        self.assertIn("任务2开窗", until_calls)
        self.assertIn("任务3隔离带", until_calls)
        self.assertLess(
            until_calls.index("任务2开窗"),
            until_calls.index("任务3隔离带"),
        )
        source = FULL_ENTRY.read_text(encoding="utf-8")
        self.assertNotIn("输入 s 跳过该任务", source)
        self.assertNotIn("跳过并继续后续任务", source)

    def test_standalone_gates_task2_start_with_loops(self) -> None:
        gate = functions(parse(TASK1_ENTRY))["main"]
        self.assertEqual(
            sum(isinstance(node, ast.While) for node in ast.walk(gate)),
            5,
        )
        calls = {
            node.func.id
            for node in ast.walk(gate)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("move_from_observation_to_task2_start", calls)

    def test_entry_only_dispatches_task1(self) -> None:
        task1_functions = functions(parse(TASK1_ENTRY))
        calls = {
            node.func.id
            for node in ast.walk(task1_functions["main"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("run_task1_debris", calls)
        self.assertFalse(any(name.startswith("run_task2") for name in calls))
        self.assertFalse(any(name.startswith("run_task3") for name in calls))

    def test_task2_and_task3_modules_are_not_imported(self) -> None:
        imported = set()
        for node in parse(TASK1_ENTRY).body:
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertNotIn("window_handle_recognition", imported)
        self.assertNotIn("black_grasp_single", imported)

    def test_box_and_chair_preserve_approach_origin_for_return(self) -> None:
        for path in (FULL_ENTRY, TASK1_ENTRY):
            task1_functions = functions(parse(path))
            for function_name in (
                "run_box_after_approach",
                "run_task1_box",
                "run_task1_chair",
            ):
                function = task1_functions[function_name]
                self.assertIn(
                    "approach_origin_node",
                    [argument.arg for argument in function.args.args],
                    f"{path.name}:{function_name}",
                )
            for function_name in ("run_box_after_approach", "run_task1_chair"):
                calls = [
                    node
                    for node in ast.walk(task1_functions[function_name])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "return_from_approach"
                ]
                self.assertEqual(len(calls), 1, f"{path.name}:{function_name}")

    def test_empty_h0_verification_does_not_clear_types_or_finish_task1(self) -> None:
        for path in (FULL_ENTRY, TASK1_ENTRY):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                'if approach_origin_node != "startpoint" and not stable_verify_targets:',
                source,
                path.name,
            )
            self.assertNotIn("cleared_types", source, path.name)
            self.assertIn("successful_removal_count", source, path.name)
            self.assertIn(
                "不据此剔除任何类别",
                source,
                path.name,
            )
            self.assertNotIn(
                '"保持在h0并直接结束任务1，进入任务2"',
                source,
                path.name,
            )

    def test_failed_grasp_returns_to_observation_before_redetection(self) -> None:
        for path in (FULL_ENTRY, TASK1_ENTRY, ROOT / "game_emergency.py"):
            source = path.read_text(encoding="utf-8")
            self.assertIn(
                '"交由任务1外层统一恢复本轮观察点并重新识别"',
                source,
                path.name,
            )
            self.assertIn("max_attempts=1", source, path.name)
            self.assertIn(
                'navigation.recoverState(\n                    approach_origin_node,\n                    method="move_by",',
                source,
                path.name,
            )
            self.assertIn(
                'if target_ok and approach_origin_node != "startpoint":',
                source,
                path.name,
            )

    def _load_retry_helpers(self) -> dict[str, object]:
        tree = parse(FULL_ENTRY)
        namespace: dict[str, object] = {"Any": object, "MAX_STAGE_ATTEMPTS": 3}
        for name in ("run_with_retry_or_skip", "run_until_success"):
            node = functions(tree)[name]
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                    str(FULL_ENTRY),
                    "exec",
                ),
                namespace,
            )
        return namespace

    def test_retry_helper_does_not_skip_when_user_types_s(self) -> None:
        helpers = self._load_retry_helpers()
        attempts: list[int] = []

        def fail_action():
            attempts.append(1)
            return False

        with patch("builtins.input", return_value="s") as prompt:
            succeeded = helpers["run_with_retry_or_skip"]("任务2开窗", fail_action)
        self.assertFalse(succeeded)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(prompt.call_count, 3)
        self.assertTrue(
            all("不允许" in call.args[0] for call in prompt.call_args_list)
        )

    def test_run_until_success_retries_same_task_until_complete(self) -> None:
        helpers = self._load_retry_helpers()
        attempts: list[int] = []

        def succeed_on_fourth():
            attempts.append(1)
            return len(attempts) >= 4

        with patch("builtins.input", return_value=""):
            helpers["run_until_success"]("任务2开窗", succeed_on_fourth)
        self.assertEqual(len(attempts), 4)


if __name__ == "__main__":
    unittest.main()
