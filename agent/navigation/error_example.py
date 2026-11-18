"""Navigation 上层错误处理示例。

导入本模块不会创建 controller、连接底盘或发送命令。下面的函数仅在调用时使用调用方传入的
``NavigationGraph``。位置单位为米，角度单位为度。

返回值约定：``True`` 表示请求动作完成；``False`` 表示动作连续两次报告 2001，且示例已经
执行明确恢复或要求上层暂停当前阶段。通信、超时、取消、地图不一致、状态错误和恢复失败仍
直接向上传播。
"""

from __future__ import annotations

from .knewbot_chassis_controller_http import ChassisErrorCode, ChassisNavError
from .navigation_graph import NavigationGraph


def move_by_error_example(
    navigation: NavigationGraph,
    x_m: float,
    y_m: float,
    theta_deg: float,
) -> bool:
    """执行机体系微调；连续两次 2001 后返回微调前保存位置。"""
    # 情况：moveBy 内部已经按实时 SLAM pose 补偿一次，第二次仍返回 2001。
    # 上层建议：放弃本轮微调，用无参数 recoverState 返回第一次微调前的实测 map pose。
    try:
        navigation.moveBy(x_m, y_m, theta_deg)
    except ChassisNavError as exc:
        if exc.code != ChassisErrorCode.ACTION_FAILED:
            raise
        navigation.recoverState(method="move_by")
        return False
    return True


def move_to_error_example(
    navigation: NavigationGraph,
    target_node: str,
    *,
    correct: bool,
) -> bool:
    """执行图导航；连续两次 2001 后返回本次 moveTo 的发车快照。"""
    # 情况：普通 moveTo 的当前图边已经重试一次 move_by，第二次仍返回 2001。
    # 上层建议：停止当前业务阶段，不直接冲向最终节点；返回 moveTo 发车前保存的实测位置。
    try:
        navigation.moveTo(target_node, correct=correct)
    except ChassisNavError as exc:
        if exc.code != ChassisErrorCode.ACTION_FAILED:
            raise
        navigation.recoverState(method="move_by")
        return False
    return True


def turn_to_warning_example(
    navigation: NavigationGraph,
    theta_deg: float,
) -> bool:
    """执行纯旋转，并识别连续两次 2001 形成的非致命 warning。"""
    # 情况：turnTo 的两次旋转都返回 2001。纯旋转不会抛异常，内部采用最终实测 yaw。
    # 上层建议：后续任务依赖精确朝向时暂停当前阶段，不把 warning 当作已经到达目标角度。
    warning_count = len(navigation.getLastWarnings())
    navigation.turnTo(theta_deg)
    for warning in reversed(navigation.getLastWarnings()[warning_count:]):
        if warning.get("category") == "rotation_action_failed_retried":
            return bool(warning.get("within_tolerance"))
    return True


def named_recovery_error_example(
    navigation: NavigationGraph,
    target_node: str,
    *,
    open_area_confirmed: bool,
) -> bool:
    """默认使用 move_to 恢复；仅在人工确认空旷后回退到 move_by。"""
    # 情况：recoverState(node) 默认的底盘 move_to 返回 2001，且未进入可微调范围。
    # 上层建议：有障碍物或不能确认直线路径时继续抛出；确认空旷时才显式使用 move_by。
    try:
        navigation.recoverState(target_node)
    except ChassisNavError as exc:
        if exc.code != ChassisErrorCode.ACTION_FAILED or not open_area_confirmed:
            raise
        navigation.recoverState(target_node, method="move_by")
        return False
    return True


def strict_error_example(
    navigation: NavigationGraph,
    target_node: str,
) -> None:
    """展示不应吞掉的错误：让底层 traceback 直接终止当前流程。"""
    # 情况：通信错误、补偿后再次超时、取消、地图不一致、节点不存在或状态错误。
    # 上层建议：不要 except Exception 后继续；没有明确恢复动作时让异常直接向上传播。
    navigation.moveTo(target_node, correct=True)
