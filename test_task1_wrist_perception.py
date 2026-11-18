#!/usr/bin/env python3
"""纯感知测试任务1左右腕抓取点，不发送任何机器人运动命令。

用法：
    python test_task1_wrist_perception.py filebox
    python test_task1_wrist_perception.py bin

机器人应由操作员提前放到待测观察姿态。每按一次 Enter 读取左右腕 RGB-D、
运行当前正式抓取点算法，并把图像和 JSON 保存到 output/task1_wrist_perception。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import agent.perception.deprojector as deprojector_module
from bin_filebox_grasp import (
    candidate_motion_info,
    select_candidate,
    wrist_yolo_grasp_candidates,
)
from camera_io import read_wrist_camera
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TARGET_NAMES = {
    "filebox": "文件框",
    "bin": "垃圾桶",
}
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "output/task1_wrist_perception"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=tuple(TARGET_NAMES))
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    """在JSON边界将NumPy数据转换为Python内置类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def print_candidate(candidate: dict[str, Any]) -> None:
    target_pose = candidate.get("target_pose_waist_xyz_xyzw")
    print(
        f"  {candidate['arm']}腕：像素={candidate['pixel_uv']}，"
        f"深度={candidate['depth_m']:.3f} m，"
        f"来源={candidate.get('point_source')}，"
        f"置信度={candidate.get('confidence', 0.0):.3f}，"
        f"TCP距离={candidate['tcp_distance_m']:.3f} m，"
        f"目标Pose(waist_yaw_link, xyz+XYZW)="
        f"{jsonable(target_pose)}",
        flush=True,
    )
    if "point_tcp_xyz_m" in candidate:
        point_tcp = candidate["point_tcp_xyz_m"]
        delta_waist = candidate["delta_waist_xyz_m"]
        print(
            "    TCP局部点(m)="
            f"({point_tcp[0]:+.4f}, {point_tcp[1]:+.4f}, {point_tcp[2]:+.4f})，"
            "waist位移(m)="
            f"({delta_waist[0]:+.4f}, {delta_waist[1]:+.4f}, "
            f"{delta_waist[2]:+.4f})",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    target_name = TARGET_NAMES[args.target]
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        deprojectors = {
            side: deprojector_module.Deprojector(
                station,
                CALIBRATION_PATH,
                side,
            )
            for side in ("left", "right")
        }
        detector = ObstacleYolo(image_size=640)
        round_index = 1
        while True:
            input(
                f"第{round_index}次{target_name}腕部感知：确认机器人保持静止，"
                "按 Enter 读取左右腕并计算；Ctrl+C结束："
            )
            output_dir = OUTPUT_ROOT / (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{args.target}"
            )
            started = time.perf_counter()
            candidates = wrist_yolo_grasp_candidates(
                station,
                args.target,
                detector,
                deprojectors,
                output_dir,
                read_wrist_camera,
            )
            for candidate in candidates:
                candidate.update(candidate_motion_info(station, candidate))
            selected = select_candidate(args.target, candidates)
            elapsed_s = time.perf_counter() - started

            print(
                f"本次{target_name}感知完成：候选={len(candidates)}，"
                f"总耗时={elapsed_s:.3f} s",
                flush=True,
            )
            for candidate in candidates:
                print_candidate(candidate)
            if selected is None:
                print("选择结果：没有可靠抓取点", flush=True)
            else:
                print(
                    f"选择结果：{selected['arm']}腕，"
                    f"TCP距离={selected['tcp_distance_m']:.3f} m",
                    flush=True,
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "target_type": args.target,
                "target_name": target_name,
                "elapsed_s": elapsed_s,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selected": selected,
            }
            (output_dir / "result.json").write_text(
                json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"本次结果已保存：{output_dir}", flush=True)
            round_index += 1
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
