#!/usr/bin/env python3
"""从startpoint执行两条底盘清场轨迹，不创建或控制其他机器人部件。

输入为每一步前的Enter确认；底盘相对位移使用base_footprint，x向前、y向左，
长度单位为米，角度单位为度。流程结束后机器人恢复Navigation的startpoint状态。
"""

from __future__ import annotations

import settings
from agent.navigation import NavigationGraph


def run_cleaning_motion(navigation: NavigationGraph) -> None:
    """按固定顺序执行两次前向清场并返回startpoint。"""
    input("步骤1：确认右侧0.25 m安全，按 Enter 向右移动0.25 m：")
    navigation.moveBy(0.0, -0.25, 0.0, correct=True)

    input("步骤2：确认前方2.30 m及左侧0.25 m安全，按 Enter 执行斜向清场：")
    navigation.moveBy(2.30, 0.25, 0.0, correct=False)

    input("步骤3：确认返回路径安全，按 Enter 恢复startpoint：")
    navigation.recoverState("startpoint", method="move_by")

    input("步骤4：确认左侧0.25 m安全，按 Enter 向左移动0.25 m：")
    navigation.moveBy(0.0, 0.25, 0.0, correct=True)

    input("步骤5：确认前方2.30 m安全，按 Enter 向前清场：")
    navigation.moveBy(2.30, 0.0, 0.0, correct=False)

    input("步骤6：确认返回路径安全，按 Enter 恢复startpoint：")
    navigation.recoverState("startpoint", method="move_by")
    print("两次底盘清场轨迹执行完成，已恢复startpoint", flush=True)


def main() -> None:
    navigation_config = settings.TASK["navigation"]
    with NavigationGraph(
        settings.prepare_route(),
        initial_node="startpoint",
        navigation_mode="move_by",
        linear_speed=float(navigation_config["linear_speed_m_s"]),
        angular_speed=float(navigation_config["angular_speed_rad_s"]),
        compress_path=False,
    ) as navigation:
        if not navigation.checkState():
            raise RuntimeError("机器人实测位置不在startpoint，不执行清场动作")
        input("已确认机器人位于startpoint；确认整条清场路线安全后按 Enter 开始：")
        run_cleaning_motion(navigation)


if __name__ == "__main__":
    main()
