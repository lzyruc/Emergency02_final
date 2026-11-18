#!/usr/bin/env python3
"""Run the fixed emergency navigation sequence with SLAM pose corrections."""

from __future__ import annotations

from pathlib import Path

try:
    from .auto_test_navigation_graph import main
except ImportError:  # Direct script execution.
    from agent.navigation.tools.auto_test_navigation_graph import main


DEFAULT_SCENE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "logs"
    / "emergency.navigation.scene.json"
)

TEST_SEQUENCE = (
    "startpoint",
    "h270_task2_start",
    "h270_window1_action",
    "h270_window2_action",
    "h270_task2_check",
    "h180_task3_check",
    "h180_return_origin",
)

CRITICAL_CORRECTION_NODES = frozenset(
    {
        "h270_task2_start",
        "h270_task2_check",
        "h180_task3_check",
        "h180_return_origin",
    }
)


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_scene=DEFAULT_SCENE,
            test_sequence=TEST_SEQUENCE,
            critical_correction_nodes=CRITICAL_CORRECTION_NODES,
            rotate_before_translate_nodes=frozenset(),
            lateral_translation_headings={},
            compress_path=False,
            snap_correction_yaw_to_nearest_quarter_turn=False,
            description=__doc__,
        )
    )
