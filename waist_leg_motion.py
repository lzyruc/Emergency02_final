"""Synchronized waist and leg station actions used by Emergency workflows."""

from __future__ import annotations

import time
from typing import Any


def joint_spd_from_delta(
    current_pos: list[float] | None,
    target_pos: list[float],
    *,
    min_spd: float = 0.2,
    max_spd: float = 0.45,
) -> list[float]:
    """Scale per-joint speed from absolute displacement for synchronized arrival."""
    if current_pos is None:
        return [min_spd] * len(target_pos)
    if len(current_pos) != len(target_pos):
        raise ValueError(
            f"current_pos 与 target_pos 长度不一致: {len(current_pos)} vs {len(target_pos)}"
        )
    delta = [abs(float(target) - float(current)) for current, target in zip(current_pos, target_pos)]
    moving = [value for value in delta if value > 0.0]
    if not moving:
        return [min_spd] * len(target_pos)
    sync_time = max(moving) / max_spd
    return [
        min_spd if value == 0.0 else min(max_spd, max(min_spd, value / sync_time))
        for value in delta
    ]


def execute_synced_waist_leg_action(
    station: Any,
    action: dict[str, Any],
    *,
    min_spd: float = 0.2,
    max_spd: float = 0.40,
) -> None:
    """Set synchronized speeds, execute waist/leg action, then restore defaults."""
    waist_target = (action.get("waist") or {}).get("position")
    leg_target = (action.get("leg") or {}).get("position")
    if waist_target is None and leg_target is None:
        raise ValueError("action 必须包含 waist.position 或 leg.position")

    robot = station.get_robot_handle().get("robot")
    waist_dof = len(waist_target) if waist_target is not None else 0
    leg_dof = len(leg_target) if leg_target is not None else 0
    current_pos: list[float] = []
    target_pos: list[float] = []
    current_missing = False
    if waist_target is not None:
        current_waist = robot.state.get_waist_pos_list()
        current_missing = current_waist is None
        if current_waist is not None:
            current_pos.extend(current_waist)
        target_pos.extend(waist_target)
    if leg_target is not None:
        current_leg = robot.state.get_leg_pos_list()
        current_missing = current_missing or current_leg is None
        if current_leg is not None:
            current_pos.extend(current_leg)
        target_pos.extend(leg_target)

    all_spd = joint_spd_from_delta(
        None if current_missing else current_pos,
        target_pos,
        min_spd=min_spd,
        max_spd=max_spd,
    )
    try:
        if waist_target is not None:
            robot.waist_ctrler.set_motor_config(spd=all_spd[:waist_dof])
        if leg_target is not None:
            robot.leg_ctrler.set_motor_config(spd=all_spd[waist_dof:])
        station.execute_action(action)
        time.sleep(0.2)
    finally:
        if waist_target is not None:
            robot.waist_ctrler.set_motor_config(spd=[0.5, 0.3])
        if leg_target is not None:
            robot.leg_ctrler.set_motor_config(spd=[0.3] * leg_dof)
