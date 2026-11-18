#!/usr/bin/env python3
"""把当前机器人 ~/Documents 中的头部、左右腕标定安装到 Emergency02。

输入文件（优先读取 ``--source`` 根目录，其次读取其 camera_calibration 子目录）：

- extrinsics_head.json
- extrinsics_D405_left.json
- extrinsics_D405_right.json

输出会更新三个独立标定文件和 Deprojector 使用的合并标定文件。脚本保留
Emergency02 自己维护的头部运动学、抓取偏移等非标定参数，并在写入前自动备份。

示例：
    cd /home/nvidia/Emergency02
    python update_camera_extrinsics.py --check
    python update_camera_extrinsics.py
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIRECTORY = Path.home() / "Documents"
HEAD_SOURCE_NAME = "extrinsics_head.json"
LEFT_SOURCE_NAME = "extrinsics_D405_left.json"
RIGHT_SOURCE_NAME = "extrinsics_D405_right.json"

HEAD_TARGET_RELATIVE = Path("config/head_calibration.json")
LEFT_TARGET_RELATIVE = Path("config/left_camera_calibration.json")
RIGHT_TARGET_RELATIVE = Path("config/right_camera_calibration.json")
COMBINED_TARGET_RELATIVE = Path(
    "agent/perception/config/camera_calibration.json"
)

WRIST_CALIBRATION_KEYS = (
    "camera_name",
    "station_camera_key",
    "color_encoding",
    "depth_unit",
    "arm_name",
    "robot_type",
    "calibration_type",
    "extrinsic_direction",
    "date",
    "image_width",
    "image_height",
    "camera_matrix",
    "dist_coeffs",
    "extrinsics_cam_to_left_tcp_link_xyzrpy",
    "extrinsics_cam_to_left_tcp_link_matrix",
    "extrinsics_cam_to_right_tcp_link_xyzrpy",
    "extrinsics_cam_to_right_tcp_link_matrix",
    "num_samples",
    "calibration_accurate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "安装当前机器Documents中的头部和左右腕相机标定，并同步生成"
            "Emergency02感知模块使用的合并标定。"
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
        help="机器人标定来源目录，默认~/Documents",
    )
    parser.add_argument(
        "--emergency-root",
        type=Path,
        default=ROOT,
        help="Emergency02目录，默认脚本所在目录",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查来源、方向和待更新内容，不写文件",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"JSON根节点必须是对象：{path}")
    return document


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def find_source_file(source_directory: Path, filename: str) -> Path:
    direct_path = source_directory / filename
    calibration_path = source_directory / "camera_calibration" / filename
    if direct_path.is_file():
        return direct_path
    if calibration_path.is_file():
        return calibration_path
    raise FileNotFoundError(
        f"未找到{filename}；已检查：{direct_path}、{calibration_path}"
    )


def validate_sources(
    head: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    head_direction = head.get("extrinsic", {}).get("direction")
    if head_direction != "T_head_roll_link_camera_optical":
        raise ValueError(
            "头部外参方向必须是T_head_roll_link_camera_optical，"
            f"实际为：{head_direction!r}"
        )
    if left.get("extrinsic_direction") != "camera_to_left_tcp_link":
        raise ValueError(
            "左腕外参方向必须是camera_to_left_tcp_link，"
            f"实际为：{left.get('extrinsic_direction')!r}"
        )
    if right.get("extrinsic_direction") != "camera_to_right_tcp_link":
        raise ValueError(
            "右腕外参方向必须是camera_to_right_tcp_link，"
            f"实际为：{right.get('extrinsic_direction')!r}"
        )
    head["extrinsic"]["matrix"]
    head["intrinsics"]["fx"]
    left["camera_matrix"]
    left["extrinsics_cam_to_left_tcp_link_matrix"]
    right["camera_matrix"]
    right["extrinsics_cam_to_right_tcp_link_matrix"]


def update_wrist_profile(
    current: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(current)
    for key in WRIST_CALIBRATION_KEYS:
        if key in source:
            updated[key] = deepcopy(source[key])
    if "extrinsics_verified" in updated:
        updated["extrinsics_verified"] = bool(
            source.get("calibration_accurate", True)
        )
    return updated


def update_combined_profile(
    current: dict[str, Any],
    head: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(current)
    cameras = updated["cameras"]

    head_camera = cameras["head"]
    head_intrinsics = head["intrinsics"]
    head_camera["camera_name"] = head["camera"]["name"]
    head_camera["image_size"] = deepcopy(head["camera"]["image_size"])
    head_camera["camera_matrix"] = [
        [head_intrinsics["fx"], 0.0, head_intrinsics["cx"]],
        [0.0, head_intrinsics["fy"], head_intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ]
    head_camera["dist_coeffs"] = deepcopy(head_intrinsics["dist_coeffs"])
    head_camera["T_mount_camera"] = deepcopy(head["extrinsic"]["matrix"])
    head_camera["calibration_date"] = head["extrinsic"].get(
        "calibration_date"
    )

    for camera_key, source, arm in (
        ("left", left, "left"),
        ("right", right, "right"),
    ):
        camera = cameras[camera_key]
        camera["camera_name"] = source["camera_name"]
        camera["arm_name"] = arm
        camera["image_size"] = [
            source["image_width"],
            source["image_height"],
        ]
        camera["camera_matrix"] = deepcopy(source["camera_matrix"])
        camera["dist_coeffs"] = deepcopy(source["dist_coeffs"])
        camera["T_mount_camera"] = deepcopy(
            source[f"extrinsics_cam_to_{arm}_tcp_link_matrix"]
        )
        camera["calibration_date"] = source.get("date")
        if "num_samples" in source:
            camera["num_samples"] = source["num_samples"]
        if "calibration_accurate" in source:
            camera["calibration_accurate"] = source[
                "calibration_accurate"
            ]
    return updated


def build_updates(
    source_directory: Path,
    emergency_root: Path,
) -> tuple[dict[Path, dict[str, Any]], dict[str, Path]]:
    source_paths = {
        "head": find_source_file(source_directory, HEAD_SOURCE_NAME),
        "left": find_source_file(source_directory, LEFT_SOURCE_NAME),
        "right": find_source_file(source_directory, RIGHT_SOURCE_NAME),
    }
    sources = {name: read_json(path) for name, path in source_paths.items()}
    validate_sources(sources["head"], sources["left"], sources["right"])

    target_paths = {
        "head": emergency_root / HEAD_TARGET_RELATIVE,
        "left": emergency_root / LEFT_TARGET_RELATIVE,
        "right": emergency_root / RIGHT_TARGET_RELATIVE,
        "combined": emergency_root / COMBINED_TARGET_RELATIVE,
    }
    current = {name: read_json(path) for name, path in target_paths.items()}

    updated_head = deepcopy(current["head"])
    for key in ("schema_version", "robot", "camera", "depth", "intrinsics", "extrinsic"):
        if key in sources["head"]:
            updated_head[key] = deepcopy(sources["head"][key])
    updated_left = update_wrist_profile(current["left"], sources["left"])
    updated_right = update_wrist_profile(current["right"], sources["right"])
    updated_combined = update_combined_profile(
        current["combined"], sources["head"], sources["left"], sources["right"]
    )
    updates = {
        target_paths["head"]: updated_head,
        target_paths["left"]: updated_left,
        target_paths["right"]: updated_right,
        target_paths["combined"]: updated_combined,
    }
    return updates, source_paths


def verify_updates(updates: dict[Path, dict[str, Any]]) -> None:
    head = updates[next(path for path in updates if path.name == "head_calibration.json")]
    left = updates[next(path for path in updates if path.name == "left_camera_calibration.json")]
    right = updates[next(path for path in updates if path.name == "right_camera_calibration.json")]
    combined = updates[
        next(path for path in updates if path == path.parent / "camera_calibration.json")
    ]
    cameras = combined["cameras"]
    if cameras["head"]["T_mount_camera"] != head["extrinsic"]["matrix"]:
        raise RuntimeError("合并标定中的头部外参与独立文件不一致")
    if cameras["left"]["T_mount_camera"] != left[
        "extrinsics_cam_to_left_tcp_link_matrix"
    ]:
        raise RuntimeError("合并标定中的左腕外参与独立文件不一致")
    if cameras["right"]["T_mount_camera"] != right[
        "extrinsics_cam_to_right_tcp_link_matrix"
    ]:
        raise RuntimeError("合并标定中的右腕外参与独立文件不一致")


def install_calibration(
    source_directory: Path,
    emergency_root: Path,
    check_only: bool = False,
) -> tuple[dict[str, Path], Path | None]:
    source_directory = source_directory.expanduser().resolve()
    emergency_root = emergency_root.expanduser().resolve()
    updates, source_paths = build_updates(source_directory, emergency_root)
    verify_updates(updates)
    if check_only:
        return source_paths, None

    backup_directory = (
        emergency_root
        / "config/calibration_backups"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    backup_directory.mkdir(parents=True)
    backup_paths: dict[Path, Path] = {}
    for target_path in updates:
        relative_path = target_path.relative_to(emergency_root)
        backup_path = backup_directory / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, backup_path)
        backup_paths[target_path] = backup_path

    try:
        for target_path, document in updates.items():
            write_json_atomic(target_path, document)
        saved_updates = {path: read_json(path) for path in updates}
        verify_updates(saved_updates)
        if saved_updates != updates:
            raise RuntimeError("标定文件写入后内容校验失败")
    except BaseException:
        for target_path, backup_path in backup_paths.items():
            shutil.copy2(backup_path, target_path)
        raise
    return source_paths, backup_directory


def main() -> int:
    args = parse_args()
    source_paths, backup_directory = install_calibration(
        args.source, args.emergency_root, check_only=args.check
    )
    print("标定来源：")
    for name in ("head", "left", "right"):
        print(f"  {name}: {source_paths[name]}")
    if args.check:
        print("检查通过；未修改Emergency02文件。")
    else:
        print("已更新：")
        for relative_path in (
            HEAD_TARGET_RELATIVE,
            LEFT_TARGET_RELATIVE,
            RIGHT_TARGET_RELATIVE,
            COMBINED_TARGET_RELATIVE,
        ):
            print(f"  {args.emergency_root.expanduser().resolve() / relative_path}")
        print(f"原文件备份：{backup_directory}")
        print("三个独立标定与合并标定一致性检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
