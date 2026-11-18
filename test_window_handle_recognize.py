#!/usr/bin/env python3
"""循环测试左腕窗把手识别，只读相机和机器人状态，不发送运动命令。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from window_handle_recognition import recognize_window_handle
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


OUTPUT_ROOT = Path(__file__).resolve().parent / "output/window_handle_recognize"


def main() -> None:
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        round_index = 1
        while True:
            input(
                f"第{round_index}次窗把手感知：确认机器人保持静止，按 Enter "
                "读取左腕RGB-D并计算；Ctrl+C结束："
            )
            output_dir = OUTPUT_ROOT / datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            started = time.perf_counter()
            try:
                result = recognize_window_handle(station, output_dir)
            except RuntimeError as error:
                elapsed_s = time.perf_counter() - started
                failure = {
                    "success": False,
                    "elapsed_s": elapsed_s,
                    "error": str(error),
                }
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"窗把手识别失败：{error}；诊断结果已保存：{output_dir}",
                    flush=True,
                )
                round_index += 1
                continue

            elapsed_s = time.perf_counter() - started
            result["success"] = True
            result["elapsed_s"] = elapsed_s
            (output_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("窗把手识别结果：", flush=True)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            print(
                f"本次耗时={elapsed_s:.3f} s，结果已保存：{output_dir}",
                flush=True,
            )
            round_index += 1
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
