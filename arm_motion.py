"""机械臂运动封装：MoveIt（关节空间 / 笛卡尔空间）+ direct（关节空间 / QP 末端位姿）。"""

from __future__ import annotations

from typing import Any, Sequence

import settings
from xrocs.common.data_type import Coordinate, Joints, PoseArray, Rotation

WAIST_FRAME = "waist_yaw_link"
_TCP_FRAME = {"left": "left_tcp_link", "right": "right_tcp_link"}
_QP_CONTROLLER = {
    "left": "endpose_single_arm_qp_L_controller",
    "right": "endpose_single_arm_qp_R_controller",
}
_JOINTSPACE_CONTROLLER = {
    "left": "jointspace_arm_L_controller",
    "right": "jointspace_arm_R_controller",
}


def _motion_controller(station: Any):
    return station.get_robot_handle().get("robot").mc


def _waypoints(value: Any, label: str) -> list[list[float]] | None:
    """单个点或多个点统一成点列表，并校验每个点是 7 个数。"""
    if value is None:
        return None
    points = list(value)
    if not points:
        raise ValueError(f"{label}不能为空")
    if not hasattr(points[0], "__len__"):
        points = [points]
    result = []
    for point in points:
        numbers = [float(number) for number in point]
        if len(numbers) != 7:
            raise ValueError(f"每个{label}必须是 7 个数，收到 {len(numbers)} 个")
        result.append(numbers)
    return result


def _check_pair(left: Any, right: Any) -> None:
    """左右路点个数必须一致，否则底层 zip 会静默丢掉多出来的路点。"""
    if left and right and len(left) != len(right):
        raise ValueError(f"左右路点个数必须一致：左 {len(left)}、右 {len(right)}")


def _joints(points: Any) -> list[Joints] | None:
    return None if points is None else [Joints(point, num_of_dofs=7) for point in points]


def _pose_array(points: Any) -> PoseArray | None:
    if points is None:
        return None
    return PoseArray(
        pose_array_data=points,
        coordinate_system_type=Coordinate.CoordinateSystem.WAIST_YAW_LINK,
        rotation_form_type=Rotation.FormType.XYZW,
    )


def moveit_joints(
    station: Any,
    *,
    left: Sequence[Any] | None = None,
    right: Sequence[Any] | None = None,
    vel_scale: float | None = None,
    acc_scale: float | None = None,
    timeout_s: float | None = None,
) -> bool:
    """MoveIt 关节空间运动（自带避障），阻塞到动作完成。

    left/right: 单个路点（7 个关节角）或多个路点组成的列表（连续轨迹），至少给一侧。
    速度与超时缺省取 task.json 的 motion 段；多路点轨迹通常要把 timeout_s 调大。
    """
    left_points = _waypoints(left, "关节路点")
    right_points = _waypoints(right, "关节路点")
    _check_pair(left_points, right_points)
    if left_points and right_points:
        motion_mode = "dual_arm"
    elif left_points:
        motion_mode = "left_arm"
    else:
        motion_mode = "right_arm"
    motion = settings.TASK["motion"]
    return _motion_controller(station).moveit_ctrler.reach_joints_and_wait(
        motion_mode=motion_mode,
        left_target_joints=_joints(left_points),
        right_target_joints=_joints(right_points),
        vel_scale=float(motion["moveit_velocity_scale"]) if vel_scale is None else vel_scale,
        acc_scale=float(motion["moveit_acceleration_scale"]) if acc_scale is None else acc_scale,
        timeout=float(motion["moveit_timeout_s"]) if timeout_s is None else timeout_s,
    )


def moveit_cartesians(
    station: Any,
    *,
    left: Sequence[Any] | None = None,
    right: Sequence[Any] | None = None,
    use_cartesian_path: bool = False,
    left_motion_id: int = 0,
    right_motion_id: int = 0,
    vel_scale: float | None = None,
    acc_scale: float | None = None,
    timeout_s: float | None = None,
) -> bool:
    """MoveIt 笛卡尔空间运动（waist_yaw_link 坐标系），阻塞到动作完成。

    left/right: 单个位姿 [x, y, z, qx, qy, qz, qw] 或多个位姿组成的列表，至少给一侧。
    use_cartesian_path: True 走直线插补，False 由 MoveIt 自由规划。
    motion_id 是轨迹标签，底层靠它回传完成信号，不要传 None。
    """
    left_points = _waypoints(left, "位姿")
    right_points = _waypoints(right, "位姿")
    _check_pair(left_points, right_points)
    if left_points and right_points:
        group = "dual_arm"
    elif left_points:
        group = "left_arm"
    else:
        group = "right_arm"
    motion = settings.TASK["motion"]
    overall_ok, (left_result, right_result) = _motion_controller(
        station
    ).moveit_ctrler.reach_cartesian_and_wait(
        group=group,
        mode="plan_and_execute",
        use_cartesian_path=True,#use_cartesian_path,
        left_motion_id=left_motion_id,
        right_motion_id=right_motion_id,
        left_waypoints=_pose_array(left_points),
        right_waypoints=_pose_array(right_points),
        vel_scale=float(motion["moveit_velocity_scale"]) if vel_scale is None else vel_scale,
        acc_scale=float(motion["moveit_acceleration_scale"]) if acc_scale is None else acc_scale,
        timeout=float(motion["moveit_timeout_s"]) if timeout_s is None else timeout_s,
    )
    if not overall_ok:
        print("MoveIt 笛卡尔运动提交失败或等待超时", flush=True)
        return False
    # overall_ok 只表示没超时，真正的成败在各侧结果里（1 成功、0 失败）。
    for side, result in (("left", left_result), ("right", right_result)):
        if result is not None and result != 1:
            print(f"MoveIt 笛卡尔运动失败（{side}）：result={result}", flush=True)
            return False
    return True


def direct_act_joints(
    station: Any,
    arm: str,
    waypoints: Sequence[Any],
    *,
    timeout_s: float = 30.0,
) -> bool:
    """direct 关节空间运动：把单侧手臂依次移到各组关节角，任一路点失败即返回 False。

    waypoints: 单个或多个 7 关节角路点。不经 MoveIt，无避障规划。
    """
    if arm not in _JOINTSPACE_CONTROLLER:
        raise ValueError("arm 必须是 'left' 或 'right'")
    targets = _waypoints(waypoints, "关节路点")
    mc = _motion_controller(station)
    act = (
        mc.direct_ctrler.act_jointspace_arm_L_controller
        if arm == "left"
        else mc.direct_ctrler.act_jointspace_arm_R_controller
    )
    # 先停掉所有控制器再激活本次要用的，避免上一段运动（如 MoveIt）仍占着手臂。
    mc.deactivate_all_controllers(via_http=True, timeout_sec=3.0)
    mc.activate_controllers([_JOINTSPACE_CONTROLLER[arm]], via_http=True, timeout_sec=3.0)
    try:
        for index, target in enumerate(targets):
            succeeded, message = act(target=target, timeout_sec=timeout_s)
            if not succeeded:
                print(f"direct 关节运动失败（{arm} 第 {index} 个路点）：{message}", flush=True)
                return False
        return True
    finally:
        mc.deactivate_all_controllers(via_http=True, timeout_sec=3.0)


def direct_act_tcp(
    station: Any,
    arm: str,
    waypoints: Sequence[Any],
    *,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    timeout_s: float = 30.0,
) -> bool:
    """QP 末端位姿运动：把单侧 TCP 依次移到各个目标位姿，任一路点失败即返回 False。

    waypoints: 单个或多个 waist_yaw_link 下的 [x, y, z, qx, qy, qz, qw]。
    """
    if arm not in _TCP_FRAME:
        raise ValueError("arm 必须是 'left' 或 'right'")
    targets = _waypoints(waypoints, "位姿")
    mc = _motion_controller(station)
    act = (
        mc.direct_ctrler.act_endpose_single_arm_qp_L_controller
        if arm == "left"
        else mc.direct_ctrler.act_endpose_single_arm_qp_R_controller
    )
    # 先停掉所有控制器再激活本次要用的，避免上一段运动（如 MoveIt）仍占着手臂。
    mc.deactivate_all_controllers(via_http=True, timeout_sec=3.0)
    mc.activate_controllers([_QP_CONTROLLER[arm]], via_http=True, timeout_sec=3.0)
    try:
        for index, target in enumerate(targets):
            succeeded, message = act(
                from_frame=WAIST_FRAME,
                to_frame=_TCP_FRAME[arm],
                target=target,
                offset=list(offset),
                timeout_sec=timeout_s,
            )
            if not succeeded:
                print(f"QP 运动失败（{arm} 第 {index} 个路点）：{message}", flush=True)
                return False
        return True
    finally:
        mc.deactivate_all_controllers(via_http=True, timeout_sec=3.0)
