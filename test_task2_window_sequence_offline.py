"""任务2窗口调度离线测试；不导入硬件依赖，不发送运动命令。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
SUBJECTS = (ROOT / "game_emergency.py", ROOT / "game_emergency_test.py")


def load_window_sequence(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_task2_windows"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "Any": object,
        "NavigationGraph": object,
        "WindowHandleYolo": object,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_task2_windows"]


def function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class FakePosePath:
    def read_text(self, *, encoding: str) -> str:
        return "{}"


class WindowSequenceTests(unittest.TestCase):
    def test_candidate_runtime_errors_return_failure_without_escaping(self) -> None:
        for path in SUBJECTS:
            with self.subTest(path=path.name, window=1):
                function = function_node(path, "run_task2_window1")
                handler = next(
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.ExceptHandler)
                    and isinstance(node.type, ast.Name)
                    and node.type.id == "RuntimeError"
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "recover_window1_for_retry"
                        for node in ast.walk(handler)
                    )
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.Return)
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is False
                        for node in ast.walk(handler)
                    )
                )

            with self.subTest(path=path.name, window=2):
                function = function_node(path, "run_window2_action")
                handler = next(
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.ExceptHandler)
                    and isinstance(node.type, ast.Name)
                    and node.type.id == "RuntimeError"
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.Return)
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is False
                        for node in ast.walk(handler)
                    )
                )

    def test_push_failure_is_not_reported_as_window1_success(self) -> None:
        for path in SUBJECTS:
            with self.subTest(path=path.name):
                function = function_node(path, "run_task2_window1")
                push_failure = next(
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.op, ast.Not)
                    and isinstance(node.test.operand, ast.Name)
                    and node.test.operand.id == "push_succeeded"
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.Return)
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is False
                        for node in push_failure.body
                    )
                )

    def test_window2_base_offset_is_outside_retry_action(self) -> None:
        game_action = function_node(SUBJECTS[0], "run_window2_action")
        game_entry = function_node(SUBJECTS[0], "run_task2_window2")

        def has_minus_five_move(function: ast.FunctionDef) -> bool:
            return any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "moveBy"
                and node.args
                and isinstance(node.args[0], ast.UnaryOp)
                and isinstance(node.args[0].operand, ast.Constant)
                and node.args[0].operand.value == 0.05
                for node in ast.walk(function)
            )

        self.assertFalse(has_minus_five_move(game_action))
        self.assertTrue(has_minus_five_move(game_entry))

    def run_sequence(
        self,
        path: Path,
        *,
        window1_results: list[bool],
        window2_entry_result: bool,
        window2_retry_results: list[bool],
    ) -> list[tuple]:
        function = load_window_sequence(path)
        calls: list[tuple] = []
        window1_results = iter(window1_results)
        window2_retry_results = iter(window2_retry_results)

        def window1(*args, navigate_to_action=True):
            calls.append(("window1", navigate_to_action))
            return next(window1_results)

        def window2_entry(*args):
            calls.append(("window2_entry",))
            return window2_entry_result

        def window2_retry(*args):
            calls.append(("window2_retry",))
            return next(window2_retry_results)

        def recover_window2(*args):
            calls.append(("recover_window2",))

        def run_until_success(name, action, *, automatic):
            calls.append(("run_until_success", name, automatic))
            self.assertTrue(action())

        class Navigation:
            def moveTo(self, node, **kwargs):
                calls.append(("moveTo", node, kwargs))

        function.__globals__.update(
            {
                "run_task2_window1": window1,
                "run_task2_window2": window2_entry,
                "run_window2_action": window2_retry,
                "recover_window2_for_retry": recover_window2,
                "run_until_success": run_until_success,
                "apply_startpoint_pose": lambda *args: True,
                "INITIAL_POSE_PATH": FakePosePath(),
                "json": __import__("json"),
            }
        )
        self.assertTrue(function(object(), Navigation(), object()))
        return calls

    def test_both_windows_succeed_once(self) -> None:
        for path in SUBJECTS:
            with self.subTest(path=path.name):
                calls = self.run_sequence(
                    path,
                    window1_results=[True],
                    window2_entry_result=True,
                    window2_retry_results=[],
                )
                self.assertEqual(calls.count(("window1", True)), 1)
                self.assertEqual(calls.count(("window2_entry",)), 1)
                self.assertNotIn(("window2_retry",), calls)

    def test_window2_retries_in_place_without_reentering(self) -> None:
        for path in SUBJECTS:
            with self.subTest(path=path.name):
                calls = self.run_sequence(
                    path,
                    window1_results=[True],
                    window2_entry_result=False,
                    window2_retry_results=[False, True],
                )
                self.assertEqual(calls.count(("window2_entry",)), 1)
                self.assertEqual(calls.count(("recover_window2",)), 2)
                self.assertEqual(calls.count(("window2_retry",)), 2)

    def test_failed_window1_is_revisited_then_retried_in_place(self) -> None:
        for path in SUBJECTS:
            with self.subTest(path=path.name):
                calls = self.run_sequence(
                    path,
                    window1_results=[False, False, True],
                    window2_entry_result=True,
                    window2_retry_results=[],
                )
                window_calls = [call for call in calls if call[0] == "window1"]
                self.assertEqual(
                    window_calls,
                    [
                        ("window1", True),
                        ("window1", True),
                        ("window1", False),
                    ],
                )
                window2_index = calls.index(("window2_entry",))
                revisit_index = calls.index(("window1", True), 1)
                self.assertLess(window2_index, revisit_index)


if __name__ == "__main__":
    unittest.main()
