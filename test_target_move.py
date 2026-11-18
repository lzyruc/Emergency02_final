#!/usr/bin/env python3
"""实机测试两段式目标接近：1.0 m 观察位，再到 0.4 m 操作位。

本测试只控制底盘，不调用 MoveIt 或 ``station.execute_action()``。每一段移动以及
最终恢复前都需要操作员按 Enter 确认；Ctrl+C 可在确认处终止。

这里按比赛场地约定测试 3 m 以内的视觉接近；地图命名节点之间的移动仍使用
``moveTo``。
"""

from __future__ import annotations

import math
import time
from typing import Any

import settings
from agent.navigation import NavigationGraph
from task1 import recognize_obstacles
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


def plan_standoff_move(
    target: dict[str, Any],
    standoff_m: float,
) -> tuple[float, float, float]:
    """计算机体系移动量 ``(x前/米, y左/米, theta逆时针/度)``。

    平移沿当前目标方向执行，并在目标前 ``standoff_m`` 停止；随后旋转使机器人
    正面对准目标。这里只生成计划，不发送控制命令。
    """
    x_m, y_m = (float(value) for value in target["pose"][:2])
    target_distance_m = math.hypot(x_m, y_m)
    if target_distance_m == 0.0:
        raise ValueError("目标与 base_footprint 原点重合，无法计算接近方向")
    travel_m = max(0.0, target_distance_m - standoff_m)
    return (
        x_m * travel_m / target_distance_m,
        y_m * travel_m / target_distance_m,
        math.degrees(math.atan2(y_m, x_m)),
    )


def execute_approach_round(
    navigation: NavigationGraph,
    target: dict[str, Any],
    standoff_m: float,
    stage_name: str,
) -> None:
    """打印并在人工确认后执行一轮平移和转向。"""
    x_m, y_m, theta_deg = plan_standoff_move(target, standoff_m)
    print(
        f"{stage_name}：目标={target['name']}，当前3D距离="
        f"{target['distance_3d_m']:.3f} m，计划 moveBy="
        f"({x_m:.3f}, {y_m:.3f}, 0.0) 后旋转 {theta_deg:.1f}°",
        flush=True,
    )
    input(
        f"确认周围安全后按 Enter 执行{stage_name}；Ctrl+C 终止："
    )
    navigation.moveBy(x_m, y_m, 0.0)
    navigation.moveBy(0.0, 0.0, theta_deg)


def nearest_target_of_type(
    targets: list[dict[str, Any]],
    target_type: str,
) -> dict[str, Any] | None:
    """返回重新识别结果中距离最近的同类目标。"""
    matches = [target for target in targets if target["type"] == target_type]
    if not matches:
        return None
    return min(matches, key=lambda target: target["distance_3d_m"])


def main() -> None:
    """识别最近目标，执行两段接近，打印最终距离并恢复测试前位置。"""
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    navigation_config = settings.TASK["navigation"]
    try:
        time.sleep(1.0)
        with NavigationGraph(
            settings.prepare_route(),
            initial_node=navigation_config["initial_node"],
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            targets = recognize_obstacles(station)
            if not targets:
                print("未识别到可测试目标，未发送底盘命令", flush=True)
                return

            target = min(
                targets,
                key=lambda item: item["distance_3d_m"],
            )
            print(
                f"选择最近目标：{target['name']}（{target['type']}），"
                f"3D距离={target['distance_3d_m']:.3f} m",
                flush=True,
            )
            execute_approach_round(
                navigation,
                target,
                standoff_m=1.0,
                stage_name="第一段观察位接近",
            )

            target = nearest_target_of_type(
                recognize_obstacles(station),
                target["type"],
            )
            if target is None:
                print("第一段后未重新识别到同类目标，停止测试", flush=True)
                input(
                    "确认周围安全后按 Enter 恢复第一段移动前位置；"
                    "Ctrl+C 保持当前位置："
                )
                navigation.recoverState()
                return
            execute_approach_round(
                navigation,
                target,
                standoff_m=0.4,
                stage_name="第二段操作位接近",
            )

            target = nearest_target_of_type(
                recognize_obstacles(station),
                target["type"],
            )
            if target is None:
                print("第二段后未重新识别到同类目标", flush=True)
            else:
                print(
                    f"最终目标：{target['name']}，"
                    f"3D距离={target['distance_3d_m']:.3f} m",
                    flush=True,
                )

            input(
                "测试完成，确认周围安全后按 Enter 恢复第一段移动前位置；"
                "Ctrl+C 保持当前位置："
            )
            navigation.recoverState()
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
