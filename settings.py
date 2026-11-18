"""集中加载 Emergency 任务所需的全部配置文件。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
RUNTIME_DIR = ROOT / "runtime"
COMPETITION_NAVIGATION_ROUTE_STEM = "scene"
TEST_NAVIGATION_ROUTE_STEM = "emergency"
COMPETITION_COLLISION_SCENE_PATH = CONFIG_DIR / "scene.json"
TEST_COLLISION_SCENE_PATH = CONFIG_DIR / "scene.json"


def _load(relative: str) -> dict:
    return json.loads((CONFIG_DIR / relative).read_text(encoding="utf-8"))


TASK = _load("task.json")
PRESETS = _load("presets.json")["poses"]
HEAD_CALIBRATION_PATH = CONFIG_DIR / "head_calibration.json"
HEAD_CALIBRATION = _load("head_calibration.json")
RIGHT_CAMERA = _load("right_camera_calibration.json")
LEFT_CAMERA = _load("left_camera_calibration.json")
WINDOW1_POSES = [_load(f"window1/step_{index}.json") for index in range(14)]
WINDOW1_APPROACH_POSES = [
    _load(f"window1/new_step{index}.json") for index in range(3)
]
WINDOW2_PREPARE_POSE = _load("window2/emergency_window2_prepare.json")
TASK3_POSES = {
    "right": _load("task3/emergency_task3_right_hand.json"),
    "left": _load("task3/emergency_task3_left_hand.json"),
}
TASK3_RIGHT_PREPARE_POSE = _load("task3/emergency_task3_right_prepare.json")
TASK3_LEFT_PREPARE_POSE = _load("task3/emergency_task3_left_prepare.json")
TASK3_PLACE_RECOVER_POSE = _load("task3/emergency_task3_place_recover.json")


def prepare_route(
    route_stem: str = COMPETITION_NAVIGATION_ROUTE_STEM,
) -> Path:
    """把路线与拓扑图复制到 runtime，避免导航回写污染 config 目录。"""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    route_path = RUNTIME_DIR / f"{route_stem}.runtime.navigation.json"
    shutil.copy2(CONFIG_DIR / f"{route_stem}.navigation.json", route_path)
    shutil.copy2(
        CONFIG_DIR / f"{route_stem}.navigation.graph.json",
        route_path.with_name(f"{route_path.stem}.graph.json"),
    )
    return route_path
