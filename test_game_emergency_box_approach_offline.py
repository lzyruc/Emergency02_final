"""纸箱第一段Y对中、第二段X保距的离线规划测试。"""

from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
ENTRIES = (
    ROOT / "game_emergency.py",
    ROOT / "game_emergency_test.py",
)


def load_select_safe_approach(entry: Path):
    tree = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            functions["plan_standoff_move"],
            functions["select_safe_approach"],
        ],
        type_ignores=[],
    )
    namespace = {
        "math": math,
        "Any": object,
        "AvoidCollision": object,
        "DEFAULT_STANDOFF_M": 0.4,
        "CHAIR_STANDOFF_M_BY_VIEW": {
            "front": 0.8,
            "back": 0.6,
            "side": 0.6,
        },
        "STANDOFF_M_BY_TYPE": {"box": 0.45},
        "TURN_REDUCTION_DEG_BY_TYPE": {"bin": 0.0, "filebox": 0.0},
        "BENT_APPROACH_STANDOFFS_M_BY_TYPE": {
            "bin": (0.5, 0.6, 0.7),
            "filebox": (0.45, 0.5, 0.6, 0.7),
        },
    }
    exec(
        compile(ast.fix_missing_locations(module), str(entry), "exec"),
        namespace,
    )
    return namespace["select_safe_approach"]


class BoxApproachTests(unittest.TestCase):
    def test_box_moves_full_y_then_x_to_45cm(self) -> None:
        checked = []

        class Avoid:
            def checkCollision(self, *motion):
                checked.append(motion)
                return False

        for entry in ENTRIES:
            with self.subTest(entry=entry.name):
                checked.clear()
                select_safe_approach = load_select_safe_approach(entry)
                selected, nominal_motion, motions = select_safe_approach(
                    Avoid(),
                    [{
                        "name": "纸箱",
                        "type": "box",
                        "point_base_m": [1.20, -0.30, 0.0],
                        "distance_xy_m": math.hypot(1.20, -0.30),
                    }],
                )

                self.assertEqual(selected["approach_standoff_m"], 0.45)
                self.assertEqual(nominal_motion, (0.75, -0.30, 0.0))
                self.assertEqual(
                    motions,
                    [(0.0, -0.30, 0.0), (0.75, 0.0, 0.0)],
                )
                self.assertEqual(checked, [(0.0, -0.30, 0.0)])

    def test_box_does_not_move_backward_when_already_nearer_than_45cm(self) -> None:
        class Avoid:
            def checkCollision(self, *motion):
                return False

        for entry in ENTRIES:
            with self.subTest(entry=entry.name):
                select_safe_approach = load_select_safe_approach(entry)
                _, nominal_motion, motions = select_safe_approach(
                    Avoid(),
                    [{
                        "name": "箱子",
                        "type": "box",
                        "point_base_m": [0.40, 0.20, 0.0],
                        "distance_xy_m": math.hypot(0.40, 0.20),
                    }],
                )

                self.assertEqual(nominal_motion, (0.0, 0.20, 0.0))
                self.assertEqual(motions, [(0.0, 0.20, 0.0)])


if __name__ == "__main__":
    unittest.main()
