#!/usr/bin/env python3
"""比赛场地纯底盘全流程测试，包含任务一清场硬编码位移。"""

from __future__ import annotations

import argparse
import json

import settings
from agent.navigation import NavigationGraph


# (节点名, correct, rotate, align, 说明)
TASK1_POINT_STEPS = (
    ("box_point", True, True, False, "搬运物品返回观察点"),
    ("startpoint", True, True, False, "从box_point返回起点"),
)

TASK1_TO_CHECK_STEPS = (
    ("h0_track_end", True, True, False, "通道末端"),
    ("h270_task2_start", True, True, False, "任务1判定点/任务2起点"),
)

TASK2_STEPS = (
    ("h270_window1_turn", True, True, False, "窗口1转接点"),
    ("h270_window1_action", True, True, True, "窗口1动作点"),
    ("h270_window1_turn", True, True, False, "窗口1退出转接点"),
    ("h0_track_end", False, False, False, "窗口2入场中间点，保持朝向"),
    ("h0_track_turn_2", False, False, False, "窗口2入场观察点，保持朝向"),
    ("h270_task2_turn", True, True, False, "窗口2转接点"),
    ("h270_window2_action", True, True, False, "窗口2动作点"),
    ("h270_task2_check", True, False, False, "任务2判定点，保持当前朝向"),
)

TASK3_STEPS = (
    ("h270_task3_check", True, True, False, "任务3检查点"),
    ("h225_task3_pick", True, True, False, "双臂抓柱点"),
    ("h180_task3_dual_return", True, True, False, "双柱返回点"),
    ("h180_task3_dual_place", True, True, False, "双柱放置点"),
    ("h180_task3_dual_return", True, True, False, "放置后返回点"),
    ("h270_task3_finish", True, False, False, "任务3判定点，保持约180度"),
)

INITIAL_NODE_BY_SECTION = {
    "full": "startpoint",
    "task1": "startpoint",
    "task2": "h270_task2_start",
    "task3": "h270_task2_check",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只使用比赛版scene路线测试底盘；包含任务一通道1清场和最终清场，"
            "不创建station，不运行识别、机械臂、夹爪或相机。"
        )
    )
    parser.add_argument(
        "--section",
        choices=tuple(INITIAL_NODE_BY_SECTION),
        default="full",
        help="测试范围，默认full",
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=0.30,
        help="测试最大线速度m/s，默认0.30；比赛配置为1.20",
    )
    parser.add_argument(
        "--angular-speed",
        type=float,
        default=0.50,
        help="测试最大角速度rad/s，默认0.50；比赛配置为1.57",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只打印流程，不创建NavigationGraph或控制底盘",
    )
    return parser.parse_args()


def wait_for_y(message: str) -> None:
    while input(f"{message}；输入 y 继续：").strip().lower() != "y":
        print("未输入 y，保持等待，不继续。", flush=True)


def print_pose(navigation: NavigationGraph) -> None:
    logical_pose, map_pose = navigation.getPose()
    print(
        "当前位姿："
        + json.dumps(
            {"logical": logical_pose, "map": map_pose},
            ensure_ascii=False,
        ),
        flush=True,
    )
    warnings = navigation.getLastWarnings()
    if warnings:
        print(
            "最近导航警告："
            + json.dumps(warnings, ensure_ascii=False, indent=2),
            flush=True,
        )


def move_steps(
    navigation: NavigationGraph,
    steps: tuple[tuple[str, bool, bool, bool, str], ...],
) -> None:
    for index, (node, correct, rotate, align, description) in enumerate(
        steps,
        1,
    ):
        path = navigation.shortestPath(node)
        print(
            f"导航{index}/{len(steps)}：前往{node}（{description}）\n"
            f"路径：{' → '.join(path)}\n"
            f"参数：correct={correct}, rotate={rotate}, align={align}",
            flush=True,
        )
        executed_path = navigation.moveTo(
            node,
            correct=correct,
            rotate=rotate,
            align=align,
        )
        print(
            f"已到达{node}；执行路径：{' → '.join(executed_path)}",
            flush=True,
        )
        print_pose(navigation)


def run_task1_channel1_chassis_clean(
    navigation: NavigationGraph,
) -> None:
    """与比赛版一致：startpoint达到上限后的通道1两轮清场。"""
    print("开始任务一通道1两轮清场。", flush=True)
    navigation.moveBy(0.0, -0.40, 0.0, correct=False)
    navigation.moveBy(2.10, 0.20, 0.0, correct=False)
    navigation.moveBy(-2.10, 0.0, 0.0, correct=False)
    navigation.moveBy(0.0, 0.40, 0.0, correct=False)
    navigation.moveBy(2.10, 0.0, 0.0, correct=False)
    navigation.moveBy(-0.30, -0.50, 0.0, correct=False)
    navigation.recoverState("h270_track_turn_1", method="move_by")
    navigation.recoverState("h0_track_turn_2", method="move_by")
    print("任务一通道1两轮清场完成。", flush=True)
    print_pose(navigation)


def run_task1_final_chassis_clean(
    navigation: NavigationGraph,
) -> None:
    """与比赛版一致：堆放点、通道1和通道2最终清场。"""
    print("开始任务一最终底盘清场。", flush=True)
    navigation.recoverState("startpoint", method="move_by")
    navigation.moveBy(0.40, 0.0, 0.0, correct=False)
    navigation.moveBy(0.0, 0.35, 0.0, correct=False)

    navigation.recoverState("startpoint", method="move_by")
    navigation.moveBy(0.0, -0.40, 0.0, correct=False)
    navigation.moveBy(1.95, 0.20, 0.0, correct=False)
    navigation.moveBy(-1.95, 0.0, 0.0, correct=False)
    navigation.moveBy(0.0, 0.40, 0.0, correct=False)
    navigation.moveBy(1.95, 0.0, 0.0, correct=False)
    navigation.moveBy(-0.30, -0.50, 0.0, correct=False)

    navigation.recoverState("h0_track_turn_2", method="move_by")
    navigation.moveBy(-0.40, -0.40, 0.0, correct=False)
    navigation.moveBy(0.40, 0.40, 0.0, correct=True)
    navigation.moveBy(0.40, -0.40, 0.0, correct=False)
    navigation.moveBy(-0.40, 0.15, 0.0, correct=True)
    navigation.moveBy(2.10, 0.0, 0.0, correct=False)
    navigation.recoverState("h0_track_turn_2", method="move_by")
    navigation.moveBy(2.10, 0.15, 0.0, correct=False)
    navigation.recoverState("h0_track_turn_2", method="move_by")
    print("任务一最终底盘清场完成。", flush=True)
    print_pose(navigation)


def print_plan(section: str) -> None:
    print(f"测试范围：{section}", flush=True)
    print(f"初始节点：{INITIAL_NODE_BY_SECTION[section]}", flush=True)
    if section in {"full", "task1"}:
        print(
            "任务1：box_point往返 → 通道1两轮清场 → "
            "最终底盘清场 → h0_track_end → h270_task2_start",
            flush=True,
        )
    if section in {"full", "task2"}:
        print(
            "任务2：window1_turn → window1_action → "
            "h0_track_end → h0_track_turn_2 → task2_turn → "
            "window2_action → task2_check",
            flush=True,
        )
    if section in {"full", "task3"}:
        print(
            "任务3：task3_check → task3_pick → dual_return → "
            "dual_place → dual_return → task3_finish",
            flush=True,
        )


def run_navigation_test(args: argparse.Namespace) -> None:
    initial_node = INITIAL_NODE_BY_SECTION[args.section]
    route_path = settings.prepare_route()
    print(
        f"加载比赛版路线：{route_path}\n"
        f"机器人必须实际位于：{initial_node}\n"
        "确认地图、定位、急停和清场区域安全后，按 Enter 开始。",
        flush=True,
    )
    input()

    with NavigationGraph(
        route_path,
        initial_node=initial_node,
        navigation_mode="move_by",
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
        compress_path=False,
    ) as navigation:
        if not navigation.checkState():
            raise RuntimeError(f"机器人实测位置不在初始节点{initial_node}")
        print(f"初始节点{initial_node}确认完成。", flush=True)
        print_pose(navigation)

        if args.section in {"full", "task1"}:
            move_steps(navigation, TASK1_POINT_STEPS)
            run_task1_channel1_chassis_clean(navigation)
            run_task1_final_chassis_clean(navigation)
            move_steps(navigation, TASK1_TO_CHECK_STEPS)
            wait_for_y("已到达任务1判定点h270_task2_start")

        if args.section in {"full", "task2"}:
            move_steps(navigation, TASK2_STEPS)
            wait_for_y("已到达任务2判定点h270_task2_check")

        if args.section in {"full", "task3"}:
            move_steps(navigation, TASK3_STEPS)
            wait_for_y("已到达任务3判定点h270_task3_finish")

    print("纯底盘导航测试完成。", flush=True)


def main() -> int:
    args = parse_args()
    print_plan(args.section)
    if args.list:
        return 0
    run_navigation_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
