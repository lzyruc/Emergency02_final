"""纸箱和转椅抓取后box_point回退分流的离线测试。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
ENTRIES = (
    ROOT / "game_emergency.py",
    ROOT / "game_emergency_test.py",
)
RETURN_FUNCTIONS = (
    "run_box_after_approach",
    "drag_task1_chair_out",
    "run_task1_chair",
)


class BoxPointReturnTests(unittest.TestCase):
    def test_h0_uses_safe_return_before_boxpoint(self) -> None:
        for entry in ENTRIES:
            tree = ast.parse(
                entry.read_text(encoding="utf-8"),
                filename=str(entry),
            )
            functions = {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            for function_name in RETURN_FUNCTIONS:
                with self.subTest(entry=entry.name, function=function_name):
                    function = functions[function_name]
                    guarded_returns = [
                        node
                        for node in ast.walk(function)
                        if isinstance(node, ast.If)
                        and any(
                            isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Name)
                            and child.func.id == "return_from_approach"
                            for statement in node.body
                            for child in ast.walk(statement)
                        )
                    ]
                    self.assertEqual(len(guarded_returns), 1)
                    condition = ast.unparse(guarded_returns[0].test)
                    self.assertEqual(
                        condition,
                        "approach_origin_node != 'startpoint'",
                    )
                    boxpoint_calls = [
                        node
                        for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "recoverState"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "box_point"
                    ]
                    self.assertEqual(len(boxpoint_calls), 1)


if __name__ == "__main__":
    unittest.main()
