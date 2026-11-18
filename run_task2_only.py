#!/usr/bin/env python3
"""测试场地任务二独立入口：h270_task2_start到h270_task2_check。"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import agent.perception.deprojector as deprojector_module
import settings
from agent.navigation import NavigationGraph
from arm_motion import moveit_cartesians, moveit_joints
from waist_leg_motion import execute_synced_waist_leg_action
from window_handle_recognition import recognize_window_handle
from yolo import WindowHandleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
INITIAL_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_emergency/应急_初始姿态.json"
)
MAX_STAGE_ATTEMPTS = 3
CAMERA_SETTLE_S = 0.3

# 每个视角重拍5帧；所有EE视角均相对首次TCP计算，不累计位移。
WINDOW_HANDLE_VIEW_OFFSETS_WAIST_XYZ_M = (
    (0.0, 0.0, 0.0),
    (-0.05, 0.0, 0.0),
    (-0.05, 0.05, 0.0),
    (-0.05, -0.05, 0.0),
    (-0.05, 0.0, 0.05),
    (-0.05, 0.0, -0.05),
)
WINDOW_HANDLE_FRAMES_PER_VIEW = 5
WINDOW_HANDLE_FALLBACK_MIN_HITS = 2
WINDOW_HANDLE_FALLBACK_MAX_UV_DISTANCE_PX = 30.0
WINDOW_HANDLE_FALLBACK_MAX_DEPTH_DELTA_M = 0.02
WINDOW1_HANDLE_ALGORITHMS = ("v1", "v2", "v3")
WINDOW2_HANDLE_ALGORITHMS = ("v3", "v1")
WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M = 0.50
WINDOW2_HANDLE_MAX_TCP_DELTA_Z_M = 0.40
WINDOW2_HANDLE_STANDOFF_TCP_Z_M = 0.20
WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M = 0.02
WINDOW_PUSH_MAX_ATTEMPTS = 7
WINDOW_PUSH_RETRY_SCALE = 0.9
WINDOW_PUSH_MIN_RETRY_TRAVEL_M = 0.01
WINDOW_PUSH_INITIAL_DELTA_TCP_M = (0.00, 0.0, -0.30)
WINDOW2_OPEN_BASE_MOVE_XY_M = (-0.32, -0.15)
WINDOW2_RELEASE_BASE_MOVE_XY_M = (0.0, -0.10)
WINDOW2_BASE_MOVE_LINEAR_SPEED_M_S = 0.10


def run_with_retry_or_skip(
    task_name: str,
    action: Any,
    *,
    max_attempts: int = MAX_STAGE_ATTEMPTS,
    automatic: bool = False,
) -> bool:
    """执行子任务并在失败后重试。不允许跳过进入后续任务。

    action 返回 False 视为失败；返回 None/其他非 False 视为成功。
    成功返回 True；本轮次数用尽返回 False，由调用方继续重试同一任务。
    任务内步骤（如窗户1→窗户2）仍允许跳过。
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = action()
            if result is False:
                raise RuntimeError(f"{task_name}执行返回失败")
            print(f"{task_name}完成", flush=True)
            return True
        except Exception as error:
            print(f'{task_name}失败（第{attempt}/{max_attempts}次）：{error}', flush=True)
            if attempt >= max_attempts:
                if not automatic:
                    input(
                        f"{task_name}本轮重试已用尽；按 Enter 继续重试本任务"
                        "（不允许进入后续任务）："
                    )
                print(f"{task_name}未完成，继续重试", flush=True)
                return False
            if automatic:
                input(f"{task_name}失败；按 Enter 重试：")
                continue
            input(f'{task_name}：按 Enter 再次尝试（不允许跳过）：')
    return False


def run_until_success(
    task_name: str,
    action: Any,
    *,
    automatic: bool = False,
) -> None:
    """重复执行直到成功；完成前不进入后续任务。"""
    while not run_with_retry_or_skip(
        task_name,
        action,
        automatic=automatic,
    ):
        input(f"{task_name}未完成；按 Enter 继续重试，不进入后续任务：")


def apply_startpoint_pose(
    station: Any,
    pose: dict[str, Any],
    *,
    skip_safe_body_pose: bool = True,
) -> bool:
    """腰腿先回正，再恢复双臂以及初始头、腰、腿姿态。失败返回 False。"""
    # 先把腰腿回到安全位
    print("startpoint姿态恢复：移动双臂，再同步移动头、腰和腿", flush=True)
    if not skip_safe_body_pose:
        input("确认头、腰、腿回正路径安全；按 Enter 执行安全身体姿态：")
        execute_synced_waist_leg_action(station, {
            'waist': {'position': [0.0, 0.001]},
            'leg': {'position': [-0.5, 0.5]},
            'head': {'position': [0.0, 0.10, 0.0]},
        })
        time.sleep(4.0)
        input('[start]腰腿回正后按 Enter 继续恢复初始腰腿姿态：')
    else:
        input("确认腰腿恢复路径安全；按 Enter 恢复初始腰腿姿态：")
    execute_synced_waist_leg_action(station, {'waist': {'position': pose['waist']}, 'leg': {'position': pose['leg']}})
    # 腰腿为下降，这里不可以sleep
    # 再恢复双臂到应急初始姿态
    input("确认双臂恢复路径安全；按 Enter 恢复应急初始双臂姿态：")
    moveit_success = moveit_joints(station, left=pose['left_arm'], right=pose['right_arm'])
    if not moveit_success:
        print("双臂恢复应急初始姿态失败", flush=True)
        return False
    return True


def recognize_window_handle_with_tcp_z_retry(
    station: Any,
    output_dir: Path,
    max_delta_z_m: float,
    window_handle_detector: WindowHandleYolo,
    algorithms: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None:
    """主算法多视角搜索失败后，回中心用备用算法重拍。

    返回识别结果、waist_yaw_link目标Pose、
    ``waist_yaw_link <- left_tcp_link`` 齐次矩阵和目标在
    left_tcp_link下的XYZ，长度单位均为米。主算法使用全部EE偏移；备用
    算法只在中心视角重拍，并要求至少两帧像素和深度一致。所有算法失败
    时恢复首次TCP位姿并返回None。搜索位移位于waist_yaw_link，方向不变。
    """
    deprojector = deprojector_module.Deprojector(
        station, CALIBRATION_PATH, 'left',)
    robot = station.get_robot_handle().get("robot")
    initial_state = station.get_robot_state()
    initial_pose = initial_state["arm_pose"]["left"]
    initial_pose_waist = np.asarray(
        initial_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(initial_pose, "get_xyz_m_xyzw_ndarray")
        else initial_pose,
        dtype=np.float64,
    ).copy()
    if not algorithms:
        raise ValueError("窗把手识别算法序列不能为空")
    center_view = ((-0.05, 0.0, 0.0),)
    restore_motion_id = 20 + len(WINDOW_HANDLE_VIEW_OFFSETS_WAIST_XYZ_M)
    moved_from_initial = False
    for algorithm_index, algorithm in enumerate(algorithms, start=1):
        is_primary = algorithm_index == 1
        if not is_primary and moved_from_initial:
            safe_center_pose_waist = initial_pose_waist.copy()
            safe_center_pose_waist[0] -= 0.05
            input(
                f"主算法{algorithms[0]}完整搜索失败；恢复左TCP后撤5 cm的"
                f"安全中心位姿后切换{algorithm}重试；按 Enter 执行："
            )
            if not moveit_cartesians(
                station,
                left=safe_center_pose_waist,
                use_cartesian_path=True,
                left_motion_id=restore_motion_id,
            ):
                print("切换备用算法前左TCP恢复安全中心位姿失败", flush=True)
                return None
            moved_from_initial = True
            time.sleep(CAMERA_SETTLE_S)
        elif not is_primary:
            print(
                f"切换备用算法{algorithm}，保持左TCP后撤5 cm安全中心位姿重拍"
                f"{WINDOW_HANDLE_FRAMES_PER_VIEW}帧",
                flush=True,
            )

        view_offsets = (
            WINDOW_HANDLE_VIEW_OFFSETS_WAIST_XYZ_M
            if is_primary
            else center_view
        )
        fallback_candidates: list[
            tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for view_index, (offset_x_m, offset_y_m, offset_z_m) in enumerate(
            view_offsets,
            start=1,
        ):
            if offset_x_m != 0.0 or offset_y_m != 0.0 or offset_z_m != 0.0:
                view_pose_waist = initial_pose_waist.copy()
                view_pose_waist[0] += offset_x_m
                view_pose_waist[1] += offset_y_m
                view_pose_waist[2] += offset_z_m
                input(
                    f"窗把手{algorithm}当前视角识别失败；将TCP移动到"
                    f"相对首次位姿waist-X={offset_x_m * 100:+.1f} cm、"
                    f"Y={offset_y_m * 100:+.1f} cm、"
                    f"Z={offset_z_m * 100:+.1f} cm；按 Enter 执行："
                )
                if not moveit_cartesians(
                    station,
                    left=view_pose_waist,
                    use_cartesian_path=True,
                    left_motion_id=19 + view_index,
                ):
                    print(f"窗把手{algorithm}视角移动失败，切换备用算法", flush=True)
                    break
                moved_from_initial = True
                time.sleep(CAMERA_SETTLE_S)

            for frame_index in range(1, WINDOW_HANDLE_FRAMES_PER_VIEW + 1):
                attempt_dir = (
                    output_dir
                    / f"algorithm_{algorithm_index}_{algorithm}"
                    / (
                        f"view_{view_index}_x_{offset_x_m:+.3f}"
                        f"_y_{offset_y_m:+.3f}"
                        f"_z_{offset_z_m:+.3f}"
                    )
                    / f"frame_{frame_index}"
                )
                try:
                    result = recognize_window_handle(
                        station,
                        attempt_dir,
                        algorithm=algorithm,
                        window_handle_detector=window_handle_detector,
                    )
                except RuntimeError as error:
                    print(
                        f"窗把手{algorithm}视角{view_index}第{frame_index}/"
                        f"{WINDOW_HANDLE_FRAMES_PER_VIEW}帧未检出：{error}",
                        flush=True,
                    )
                    continue
                target_pose_waist = deprojector.deproject(
                    [*result['pixel_uv'], result['depth_m']],
                    face_target=False,
                )
                state = station.get_robot_state()
                T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                    state['arm_joints']['left']
                ).get_affine_matrix_m_ndarray()
                point_left_tcp_m = (
                    np.linalg.inv(T_waist_tcp)
                    @ np.append(target_pose_waist[:3], 1.0)
                )[:3]
                delta_z_m = float(point_left_tcp_m[2])
                result.update(
                    {
                        'recognition_algorithm': algorithm,
                        'recognition_algorithm_index': algorithm_index,
                        'recognition_algorithm_primary': is_primary,
                        'recognition_view': view_index,
                        'recognition_frame': frame_index,
                        'view_offset_waist_x_m': offset_x_m,
                        'view_offset_waist_y_m': offset_y_m,
                        'view_offset_waist_z_m': offset_z_m,
                        'point_left_tcp_xyz_m': point_left_tcp_m.tolist(),
                        'tcp_delta_z_accepted_range_m': [-max_delta_z_m, 0.0],
                    }
                )
                (attempt_dir / 'result.json').write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8',
                )
                print(
                    f'窗把手{algorithm}视角{view_index}第{frame_index}/'
                    f'{WINDOW_HANDLE_FRAMES_PER_VIEW}帧：'
                    f'waist-X={offset_x_m * 100:+.1f} cm，'
                    f'Y={offset_y_m * 100:+.1f} cm，'
                    f'Z={offset_z_m * 100:+.1f} cm，'
                    f'left_tcp_link delta-x={point_left_tcp_m[0] * 100:+.2f} cm，'
                    f'delta-y={point_left_tcp_m[1] * 100:+.2f} cm，'
                    f'delta-z={delta_z_m * 100:+.2f} cm，'
                    f'前向上限={max_delta_z_m * 100:.1f} cm',
                    flush=True,
                )
                if not -max_delta_z_m <= delta_z_m <= 0.0:
                    print("识别点前向距离超限，本视角继续重拍", flush=True)
                    continue
                candidate = (
                    result,
                    target_pose_waist,
                    T_waist_tcp,
                    point_left_tcp_m,
                )
                if is_primary:
                    return candidate
                consistent_count = 1 + sum(
                    math.hypot(
                        float(result['pixel_uv'][0])
                        - float(previous[0]['pixel_uv'][0]),
                        float(result['pixel_uv'][1])
                        - float(previous[0]['pixel_uv'][1]),
                    ) <= WINDOW_HANDLE_FALLBACK_MAX_UV_DISTANCE_PX
                    and abs(
                        float(result['depth_m'])
                        - float(previous[0]['depth_m'])
                    ) <= WINDOW_HANDLE_FALLBACK_MAX_DEPTH_DELTA_M
                    for previous in fallback_candidates
                )
                fallback_candidates.append(candidate)
                if consistent_count >= WINDOW_HANDLE_FALLBACK_MIN_HITS:
                    print(
                        f"备用算法{algorithm}中心视角获得"
                        f"{consistent_count}帧一致结果，接受识别点",
                        flush=True,
                    )
                    return candidate
                print(
                    f"备用算法{algorithm}已获得1帧候选，继续等待一致结果",
                    flush=True,
                )

    if moved_from_initial:
        print("所有算法均失败；左TCP保持在后撤5 cm安全搜索平面", flush=True)
    print(f"窗把手算法序列{algorithms}均未获得有效结果", flush=True)
    return None

# 窗口2：识别、对齐、扭窗、底盘开窗
def run_window2_action(
    station: Any,
    navigation: NavigationGraph,
    window_handle_detector: WindowHandleYolo,
) -> bool:
    """在h270_window2_action执行左腕识别、TCP对齐和开窗动作。"""
    output_dir = (
        settings.ROOT
        / "output/task2_recognize"
        / datetime.now().strftime("%Y%m%d_%H%M%S_window2")
    )
    input('任务2窗口2操作步骤1：机器人已到h270_window2_action；按 Enter 读取左腕RGB-D、保存图像并识别窗把手：')
    time.sleep(CAMERA_SETTLE_S)
    try:
        recognized = recognize_window_handle_with_tcp_z_retry(
            station,
            output_dir,
            WINDOW2_HANDLE_MAX_TCP_DELTA_Z_M,
            window_handle_detector,
            WINDOW2_HANDLE_ALGORITHMS,
        )
    except RuntimeError as error:
        print(f'窗口2把手识别失败，跳过后续对齐和扭窗步骤：{error}', flush=True)
        return False
    if recognized is None:
        print("窗口2开窗判定超出阈值，跳过后续操作并进入下一项目", flush=True)
        return False
    result, target_pose_waist, T_waist_tcp, point_left_tcp_m = recognized
    recognized_tcp_z_m = float(point_left_tcp_m[2])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    robot = station.get_robot_handle().get("robot")
    # 先修TCP x/y，再按识别TCP-z计算水平接近量。
    first_delta_tcp_m = np.asarray(
        [point_left_tcp_m[0], point_left_tcp_m[1], 0.0], dtype=np.float64,)
    first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(f'窗口2识别点（left_tcp_link）：x={point_left_tcp_m[0] * 100:+.2f} cm，y={point_left_tcp_m[1] * 100:+.2f} cm，z={recognized_tcp_z_m * 100:+.2f} cm', flush=True)
    print(f'窗口2第一次TCP局部偏移：x={first_delta_tcp_m[0] * 100:+.2f} cm，y={first_delta_tcp_m[1] * 100:+.2f} cm，z=+0.00 cm；方向保持不变', flush=True)
    input('任务2窗口2操作步骤2：确认路径安全；按 Enter 只修正TCP局部x、y：')
    moveit_success = moveit_cartesians(station, left=first_target_pose_waist, use_cartesian_path=True, left_motion_id=8)
    if not moveit_success:
        print("窗口2左臂第一次x/y修正失败", flush=True)
        return False
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(state['arm_joints']['left']).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    current_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64,)
    second_delta_tcp_m = np.asarray(
        [
            0.0,
            0.0,
            recognized_tcp_z_m + WINDOW2_HANDLE_STANDOFF_TCP_Z_M,
        ],
        dtype=np.float64,
    )
    second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
    second_delta_waist_m[2] = 0.0
    second_target_pose_waist = current_pose_waist.copy()
    second_target_pose_waist[:3] = T_waist_tcp[:3, 3] + second_delta_waist_m
    print(f'窗口2第二段：识别TCP-z={recognized_tcp_z_m * 100:+.2f} cm + 保留距离{WINDOW2_HANDLE_STANDOFF_TCP_Z_M * 100:.2f} cm = TCP-z计算量{second_delta_tcp_m[2] * 100:+.2f} cm；最终waist位移x={second_delta_waist_m[0] * 100:+.2f} cm，y={second_delta_waist_m[1] * 100:+.2f} cm，z=+0.00 cm', flush=True)
    result.update(
        {'point_left_tcp_xyz_m': point_left_tcp_m.tolist(), 'recognized_left_tcp_z_raw_m': recognized_tcp_z_m, 'first_delta_left_tcp_xyz_m': first_delta_tcp_m.tolist(), 'second_delta_left_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_delta_waist_xyz_m': second_delta_waist_m.tolist(), 'second_z_rule': 'recognized_left_tcp_z_m + 0.19 m, then keep waist_yaw_link z unchanged'},)
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
    input('任务2窗口2操作步骤3：确认路径安全；按 Enter 执行水平接近（waist_yaw_link Z保持不变）：')
    moveit_success = moveit_cartesians(station, left=second_target_pose_waist, use_cartesian_path=True, left_motion_id=9)
    if not moveit_success:
        print("窗口2左臂第二次z修正失败", flush=True)
        return False
    # 腰系平移靠近把手侧
    for step, motion_id, label, delta_waist_m in (
        (4, 10, "沿waist_yaw_link -Y向视觉右侧移动10 cm", [0.0, -0.10, 0.0]),
        (5, 11, "沿waist_yaw_link +Z向上移动4.5 cm", [0.0, 0.0, 0.045]),
        (6, 12, "沿waist_yaw_link -Y向视觉右侧移动3 cm", [0.0, -0.03, 0.0]),
    ):
        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
        target_pose_waist[:3] += np.asarray(delta_waist_m, dtype=np.float64)
        input(f'任务2窗口2操作步骤{step}：确认路径安全；按 Enter {label}：')
        moveit_success = moveit_cartesians(station, left=target_pose_waist, use_cartesian_path=True, left_motion_id=motion_id)
        if not moveit_success:
            print(f"窗口2左臂{label}失败", flush=True)
            return False
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    rotation_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    # 绕夹爪局部 z 顺时针 90°
    qx, qy, qz, qw = rotation_target_pose_waist[3:7]
    cos_h = math.cos(math.pi / 4.0)
    sin_h = math.sin(math.pi / 4.0)
    qw_new = qw * cos_h - qz * sin_h
    qx_new = qx * cos_h + qy * sin_h
    qy_new = qy * cos_h - qx * sin_h
    qz_new = qz * cos_h + qw * sin_h
    norm = math.sqrt(
        qw_new ** 2 + qx_new ** 2 + qy_new ** 2 + qz_new ** 2,)
    rotation_target_pose_waist[3:7] = (
        qx_new / norm,
        qy_new / norm,
        qz_new / norm,
        qw_new / norm,
    )
    input('任务2窗口2操作步骤7：确认路径安全；按 Enter 保持TCP位置不变，绕当前夹爪局部z轴顺时针旋转90°：')
    moveit_success = moveit_cartesians(station, left=rotation_target_pose_waist, use_cartesian_path=True, left_motion_id=13)
    if not moveit_success:
        print("窗口2左臂TCP绕局部z轴顺时针旋转90°失败", flush=True)
        return False
    for step, motion_id, label, delta_waist_m in (
        (8, 14, "沿waist_yaw_link +Z向上移动2 cm", [0.0, 0.0, 0.02]),
        (9, 15, "沿waist_yaw_link +Y向视觉左侧移动5 cm", [0.0, 0.08, 0.0]),
    ):
        state = station.get_robot_state()
        current_pose = state["arm_pose"]["left"]
        target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
        target_pose_waist[:3] += np.asarray(delta_waist_m, dtype=np.float64)
        input(f'任务2窗口2操作步骤{step}：确认路径安全；按 Enter {label}：')
        moveit_success = moveit_cartesians(station, left=target_pose_waist, use_cartesian_path=True, left_motion_id=motion_id)
        if not moveit_success:
            print(f"窗口2左臂{label}失败", flush=True)
            return False
    input('任务2窗口2操作步骤10：确认把手位于夹爪内；按 Enter 闭合左夹爪：')
    station.execute_action({'hand': {'position': {'left': [1.0], 'right': [0.0]}}})
    time.sleep(1.0)
    input(
        '任务2窗口2操作步骤11：确认底盘、窗户和机械臂路径安全；'
        f'按 Enter 执行底盘开窗：dx={WINDOW2_OPEN_BASE_MOVE_XY_M[0] * 100:+.0f} cm，'
        f'dy={WINDOW2_OPEN_BASE_MOVE_XY_M[1] * 100:+.0f} cm：'
    )
    # 底盘拉开窗户
    navigation.moveBy(
        *WINDOW2_OPEN_BASE_MOVE_XY_M,
        0.0,
        linear_speed=WINDOW2_BASE_MOVE_LINEAR_SPEED_M_S,
        correct=True,
    )
    time.sleep(1.0)
    input('任务2窗口2操作步骤12：检查开窗结果；按 Enter 松开左夹爪：')
    station.execute_action({'hand': {'position': {'left': [0.0], 'right': [0.0]}}})
    time.sleep(1.5)

    # 只有窗口2识别、对齐和拉窗全部成功才会进入本收尾流程。
    input("窗口2收尾：按 Enter 向右后移动30 cm、右移20 cm脱离把手：")
    navigation.moveBy(-0.30, -0.20, 0.0, correct=False)
    input("窗口2收尾：按 Enter 向左移动40 cm确保开窗角度：")
    navigation.moveBy(0.0, 0.40, 0.0, correct=False)
    input("窗口2收尾：确认腰部路径安全；按 Enter 左转腰避让：")
    execute_synced_waist_leg_action(
        station,
        {'waist': {'position': [0.1, 0.0]}},
        min_spd=0.10,
        max_spd=0.20,
    )
    time.sleep(0.5)
    input("窗口2收尾：按 Enter 向前移动20 cm：")
    navigation.moveBy(0.20, 0.0, 0.0, correct=False)
    time.sleep(1.0)
    input("窗口2收尾：确认腰部路径安全；按 Enter 右转腰开窗：")
    execute_synced_waist_leg_action(
        station,
        {'waist': {'position': [-0.30, 0.20]}},
        min_spd=0.10,
        max_spd=0.10,
    )
    time.sleep(4.0)

    input("窗口2收尾：按 Enter 向后移动30 cm：")
    navigation.moveBy(-0.30, 0.0, 0.0, correct=False)

    input("窗口2收尾：按 Enter 将腰部恢复零位：")
    execute_synced_waist_leg_action(
        station,
        {'waist': {'position': [0.0, 0.0]}},
        min_spd=0.2,
        max_spd=0.3,
    )
    # time.sleep(2.0)
    # navigation.moveBy(0.0, -0.20, 0.0, correct=False)  # 向右

    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    input("窗口2操作结束：确认双臂路径安全；按 Enter 恢复应急初始双臂姿态：")
    run_until_success(
        "窗口2操作结束后双臂恢复应急初始姿态",
        lambda: moveit_joints(
            station,
            left=initial_pose['left_arm'],
            right=initial_pose['right_arm'],
        ),
        automatic=True,
    )
    return True

# 推窗失败则缩短偏移重试
def _push_window_with_reduced_offset(
    station: Any,
    robot: Any,
    *,
    first_motion_id: int,
    label: str,
    initial_delta_tcp_m: np.ndarray,
) -> bool:
    """以首次TCP位姿为基准缩短总偏移重试，不叠加每次位移。"""
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
        state["arm_joints"]["left"]
    ).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    start_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray()
        if hasattr(current_pose, "get_xyz_m_xyzw_ndarray")
        else current_pose,
        dtype=np.float64,
    ).copy()
    initial_delta_tcp_m = np.asarray(
        initial_delta_tcp_m,
        dtype=np.float64,
    ).copy()
    initial_delta_waist_m = T_waist_tcp[:3, :3] @ initial_delta_tcp_m
    initial_distance_m = float(np.linalg.norm(initial_delta_waist_m))
    if initial_distance_m == 0.0:
        raise ValueError("推窗TCP偏移不能为零")
    push_direction_waist = initial_delta_waist_m / initial_distance_m

    for attempt in range(1, WINDOW_PUSH_MAX_ATTEMPTS + 1):
        scale = WINDOW_PUSH_RETRY_SCALE ** (attempt - 1)
        delta_tcp_m = initial_delta_tcp_m * scale
        delta_waist_m = initial_delta_waist_m * scale
        target_pose_waist = start_pose_waist.copy()
        target_pose_waist[:3] = start_pose_waist[:3] + delta_waist_m
        attempt_label = "首次" if attempt == 1 else f"第{attempt}次缩距重试"
        if attempt > 1:
            latest_state = station.get_robot_state()
            latest_T_waist_tcp = robot.pin_kine.get_left_tcp_pose(
                latest_state["arm_joints"]["left"]
            ).get_affine_matrix_m_ndarray()
            remaining_forward_m = float(
                np.dot(
                    target_pose_waist[:3] - latest_T_waist_tcp[:3, 3],
                    push_direction_waist,
                )
            )
            if remaining_forward_m <= WINDOW_PUSH_MIN_RETRY_TRAVEL_M:
                print(
                    f"当前TCP已达到或越过"
                    f"{np.linalg.norm(delta_tcp_m) * 100:.1f} cm缩距目标，"
                    "视为推窗成功，不执行可能反向的重试",
                    flush=True,
                )
                return True

        input(
            f"任务2推窗：确认机械臂已停止且路径安全；"
            f"按 Enter {attempt_label}沿左TCP{label}移动："
            f"相对首次起点的TCP总偏移="
            f"({delta_tcp_m[0] * 100:+.1f}, "
            f"{delta_tcp_m[1] * 100:+.1f}, "
            f"{delta_tcp_m[2] * 100:+.1f}) cm；Ctrl+C终止："
        )
        motion_id = first_motion_id + attempt - 1
        if moveit_cartesians(
            station,
            left=target_pose_waist,
            use_cartesian_path=True,
            left_motion_id=motion_id,
        ):
            print(
                f"左臂TCP{label}推窗成功：第{attempt}次尝试，"
                f"偏移={np.linalg.norm(delta_tcp_m) * 100:.1f} cm，"
                f"motion_id={motion_id}",
                flush=True,
            )
            return True
        if attempt < WINDOW_PUSH_MAX_ATTEMPTS:
            next_distance_cm = (
                np.linalg.norm(initial_delta_tcp_m)
                * WINDOW_PUSH_RETRY_SCALE**attempt
                * 100.0
            )
            print(
                f"左臂TCP{label}第{attempt}次推窗失败，"
                f"下一次总偏移缩短为{next_distance_cm:.1f} cm",
                flush=True,
            )
    print(
        f"左臂TCP{label}推窗连续"
        f"{WINDOW_PUSH_MAX_ATTEMPTS}次失败",
        flush=True,
    )
    return False

# ---------------------------------------------------------------------------
# 任务2失败恢复与窗口调度
def recover_window1_for_retry(
    station: Any,
    *,
    start_pose_index: int,
) -> None:
    """在窗口1原地按已验证的反向姿态收回左臂，供重试或转场。"""
    for pose_index in range(start_pose_index, 14):
        window_pose = settings.WINDOW1_POSES[pose_index]
        input(f"窗口1失败恢复：确认路径安全；按 Enter 执行 step_{pose_index}：")
        run_until_success(
            f"窗口1失败恢复 step_{pose_index}",
            lambda pose=window_pose: moveit_joints(
                station,
                left=pose["left_arm"],
            ),
            automatic=True,
        )
    input("窗口1失败恢复：按 Enter 将腰部恢复零位：")
    execute_synced_waist_leg_action(
        station,
        {'waist': {'position': [0.0, 0.0]}},
    )


def recover_window2_for_retry(station: Any) -> None:
    """窗口2安全失败后张开夹爪并恢复准备姿态，不移动底盘。"""
    input("窗口2失败恢复：按 Enter 张开双夹爪：")
    station.execute_action(
        {'hand': {'position': {'left': [0.0], 'right': [0.0]}}}
    )
    prepare_pose = settings.WINDOW2_PREPARE_POSE
    input("窗口2失败恢复：确认双臂路径安全；按 Enter 恢复窗口2准备姿态：")
    run_until_success(
        "窗口2失败恢复准备姿态",
        lambda: moveit_joints(
            station,
            left=prepare_pose['left_arm'],
            right=prepare_pose['right_arm'],
        ),
        automatic=True,
    )


def run_task2_window1(
    station: Any,
    navigation: NavigationGraph,
    window_handle_detector: WindowHandleYolo,
    *,
    navigate_to_action: bool = True,
) -> bool:
    """窗户1：可选导航一次，然后识别、推窗并恢复姿态。"""
    if navigate_to_action:
        input('任务2步骤3：确认导航路径安全；按 Enter 导航到 h270_window1_action：')
        navigation.moveTo("h270_window1_action", correct=True, align=True)
    else:
        print('任务2窗口1保持当前作业点，开始原地重试', flush=True)
    input("确认垂直抬升腰腿路径安全；按 Enter 执行：")
    execute_synced_waist_leg_action(
        station, {'waist': {'position': [0.0, 0.02]}, 'leg': {'position': [-0.02, 0.02]}},)
    time.sleep(1.0)
    for index, window_pose in enumerate(settings.WINDOW1_APPROACH_POSES):
        input(f"任务2步骤{index + 4}：确认机械臂路径安全；按 Enter 执行窗口迫近 new_step{index}：")
        moveit_success = moveit_joints(station, left=window_pose["left_arm"])
        if not moveit_success:
            print(f"窗口迫近 new_step{index} 执行失败", flush=True)
            recover_window1_for_retry(
                station,
                start_pose_index=13 - index,
            )
            return False
    output_dir = (
        settings.ROOT
        / "output/task2_recognize"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    input("任务2步骤7：new_step0～2已完成；按 Enter 读取左腕RGB-D、保存图像并识别窗把手：")
    time.sleep(CAMERA_SETTLE_S)
    recognized = recognize_window_handle_with_tcp_z_retry(
        station,
        output_dir,
        WINDOW1_HANDLE_MAX_TCP_DELTA_Z_M,
        window_handle_detector,
        WINDOW1_HANDLE_ALGORITHMS,
    )
    if recognized is None:
        print("任务2开窗判定超出阈值", flush=True)
        recover_window1_for_retry(station, start_pose_index=10)
        return False
    result, target_pose_waist, T_waist_tcp, point_left_tcp_m = recognized
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    robot = station.get_robot_handle().get("robot")
    first_delta_tcp_m = np.asarray(
        [point_left_tcp_m[0], point_left_tcp_m[1], 0.0], dtype=np.float64,)
    first_delta_waist_m = T_waist_tcp[:3, :3] @ first_delta_tcp_m
    first_delta_waist_m[1] -= WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M
    first_target_pose_waist = target_pose_waist.copy()
    first_target_pose_waist[:3] = T_waist_tcp[:3, 3] + first_delta_waist_m
    print(f'识别点（left_tcp_link）：x={point_left_tcp_m[0] * 100:+.2f} cm，y={point_left_tcp_m[1] * 100:+.2f} cm，z={point_left_tcp_m[2] * 100:+.2f} cm', flush=True)
    print(f'第一次TCP局部偏移：x={first_delta_tcp_m[0] * 100:+.2f} cm，y={first_delta_tcp_m[1] * 100:+.2f} cm，z=+0.00 cm；另沿waist_yaw_link -Y向视觉右侧移动{WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M * 100:.2f} cm，方向保持不变', flush=True)
    input("任务2步骤8：确认路径安全；按 Enter 只修正TCP局部x、y：")
    moveit_success = moveit_cartesians(
        station, left=first_target_pose_waist, use_cartesian_path=True, left_motion_id=1,)
    if not moveit_success:
        print("左臂第一次x/y修正失败", flush=True)
        recover_window1_for_retry(station, start_pose_index=10)
        return False
    state = station.get_robot_state()
    T_waist_tcp = robot.pin_kine.get_left_tcp_pose(state['arm_joints']['left']).get_affine_matrix_m_ndarray()
    current_pose = state["arm_pose"]["left"]
    current_pose_waist = np.asarray(
        current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64,)
    second_delta_tcp_m = np.asarray(
        [0.0, 0.0, point_left_tcp_m[2] + 0.2], dtype=np.float64,)
    second_delta_waist_m = T_waist_tcp[:3, :3] @ second_delta_tcp_m
    second_target_pose_waist = current_pose_waist.copy()
    second_target_pose_waist[:3] = T_waist_tcp[:3, 3] + second_delta_waist_m
    print(f'第二次TCP局部z偏移：识别z={point_left_tcp_m[2] * 100:+.2f} cm + 20.00 cm = {second_delta_tcp_m[2] * 100:+.2f} cm；方向保持不变', flush=True)
    result.update(
        {'point_left_tcp_xyz_m': point_left_tcp_m.tolist(), 'first_delta_left_tcp_xyz_m': first_delta_tcp_m.tolist(), 'first_extra_right_waist_y_m': -WINDOW1_FIRST_ALIGN_EXTRA_RIGHT_M, 'second_delta_left_tcp_xyz_m': second_delta_tcp_m.tolist(), 'second_z_rule': 'recognized_left_tcp_z_m + 0.20 m'},)
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',)
    input("任务2步骤9：确认路径安全；按 Enter 只沿TCP局部z移动：")
    moveit_success = moveit_cartesians(
        station, left=second_target_pose_waist, use_cartesian_path=True, left_motion_id=2,)
    if not moveit_success:
        print("左臂第二次z修正失败", flush=True)
        recover_window1_for_retry(station, start_pose_index=10)
        return False
    input("任务2步骤10：TCP的x/y/z对齐已完成且不闭爪；确认状态后按 Enter 继续：")
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    right_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    right_target_pose_waist[1] -= 0.10
    input("任务2步骤11：确认路径安全；按 Enter 沿waist_yaw_link -Y向视觉右侧移动10 cm：")
    moveit_success = moveit_cartesians(
        station, left=right_target_pose_waist, use_cartesian_path=True, left_motion_id=3,)
    if not moveit_success:
        print("左臂向视觉右侧移动10 cm失败", flush=True)
        recover_window1_for_retry(station, start_pose_index=10)
        return False
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    upward_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    upward_target_pose_waist[2] += 0.04
    input("任务2步骤12：确认路径安全；按 Enter 沿waist_yaw_link +Z向上移动4 cm：")
    moveit_success = moveit_cartesians(
        station, left=upward_target_pose_waist, use_cartesian_path=True, left_motion_id=4,)
    if not moveit_success:
        print("左臂向上移动4 cm失败", flush=True)
        recover_window1_for_retry(station, start_pose_index=10)
        return False
    state = station.get_robot_state()
    current_pose = state["arm_pose"]["left"]
    rotate_target_pose_waist = np.asarray(current_pose.get_xyz_m_xyzw_ndarray() if hasattr(current_pose, 'get_xyz_m_xyzw_ndarray') else current_pose, dtype=np.float64).copy()
    qx, qy, qz, qw = rotate_target_pose_waist[3:]
    half_angle_rad = math.radians(-5.0) / 2.0
    sin_half = math.sin(half_angle_rad)
    cos_half = math.cos(half_angle_rad)
    rotate_target_pose_waist[3:] = np.asarray(
        [qx * cos_half - qz * sin_half, qw * sin_half + qy * cos_half, qx * sin_half + qz * cos_half, qw * cos_half - qy * sin_half], dtype=np.float64,)
    rotate_target_pose_waist[3:] /= np.linalg.norm(rotate_target_pose_waist[3:])
    input("确认底盘右侧安全；按 Enter 右移10 cm：")
    navigation.moveBy(0.0, -0.10, 0.0, correct=False)
    time.sleep(0.5)
    input("任务2推窗：确认左臂原地旋转路径安全；按 Enter 绕TCP局部-Y轴向右旋转5°，位置保持不变：")
    moveit_success = moveit_cartesians(
        station, left=rotate_target_pose_waist, use_cartesian_path=True, left_motion_id=5,)
    if not moveit_success:
        print("左臂TCP向左旋转5°失败", flush=True)
        input("窗口1旋转失败：确认底盘左侧安全；按 Enter 左移10 cm回正：")
        navigation.moveBy(0.0, 0.10, 0.0, correct=False)
        recover_window1_for_retry(station, start_pose_index=10)
        return False

    push_succeeded = _push_window_with_reduced_offset(
        station,
        robot,
        first_motion_id=7,
        label="向前",
        initial_delta_tcp_m=np.asarray(
            WINDOW_PUSH_INITIAL_DELTA_TCP_M,
            dtype=np.float64,
        ),
    )
    input("窗口1推窗结束：确认腰部路径安全；按 Enter 执行第一段腰部恢复：")
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [-0.08, 0.1]}},
        min_spd=0.10,
        max_spd=0.20,
        )
    time.sleep(0.5)
    input("窗口1推窗结束：按 Enter 将腰部恢复零位：")
    execute_synced_waist_leg_action(station, {
        'waist': {'position': [0.0, 0.0]}})
    input("确认底盘左侧安全；按 Enter 左移10 cm：")
    navigation.moveBy(0.0, 0.10, 0.0, correct=False)
    time.sleep(0.5)
    for index in range(9, 14):
        input(f"任务2恢复：确认机械臂路径安全；按 Enter 执行窗口恢复 step_{index}：")
        window_pose = settings.WINDOW1_POSES[index]
        run_until_success(
            f"窗口恢复 step_{index}",
            lambda pose=window_pose: moveit_joints(
                station,
                left=pose["left_arm"],
            ),
            automatic=True,
        )

    if not push_succeeded:
        print(
            "左臂TCP向前推窗缩距重试仍失败；"
            "窗口恢复姿态已执行，本次窗口1判定失败",
            flush=True,
        )
        return False
    return True

def run_task2_window2(
    station: Any,
    navigation: NavigationGraph,
    window_handle_detector: WindowHandleYolo,
) -> bool:
    """窗户2：过渡导航、准备姿态，再执行现场开窗动作。"""
    input("任务2窗口2步骤1a：确认路径安全；按 Enter 导航到h270_window1_turn：")
    navigation.moveTo("h270_window1_turn", correct=True)
    input("任务2窗口2步骤1b：确认路径安全；按 Enter 导航到h0_track_end：")
    navigation.moveTo(
        "h0_track_end",
        correct=False,
        rotate=False,
    )
    input("任务2窗口2步骤1c：确认路径安全；按 Enter 导航到h0_track_turn_2：")
    navigation.moveTo(
        "h0_track_turn_2",
        correct=False,
        rotate=False,
    )

    input("任务2窗口2步骤2：确认双臂路径安全；按 Enter 切换到窗口2准备姿态：")
    prepare_pose = settings.WINDOW2_PREPARE_POSE
    run_until_success(
        "双臂切换应急_窗户2_准备姿态",
        lambda: moveit_joints(
            station,
            left=prepare_pose['left_arm'],
            right=prepare_pose['right_arm'],
        ),
        automatic=True,
    )
    input("任务2窗口2步骤3a：确认路径安全；按 Enter 恢复h270_window2_turn：")
    navigation.recoverState("h270_window2_turn", method="move_by")
    input("任务2窗口2步骤3b：确认路径安全；按 Enter 导航到h270_window2_action：")
    navigation.moveTo("h270_window2_action", correct=True)
    if run_window2_action(station, navigation, window_handle_detector):
        print("任务2窗口视觉操作完成", flush=True)
        return True
    print("任务2窗口2识别或操作失败，交由当前作业点原地重试", flush=True)
    return False

def run_task2_windows(
    station: Any,
    navigation: NavigationGraph,
    window_handle_detector: WindowHandleYolo,
) -> bool:
    """先尝试两个窗口；窗口2完成后回补失败的窗口1。"""
    window1_success = run_task2_window1(
        station,
        navigation,
        window_handle_detector,
    )
    if not window1_success:
        print(
            "任务2窗口1首次未完成，姿态已收回；先执行窗口2，"
            "窗口2完成后返回窗口1补做",
            flush=True,
        )

    input("窗口1阶段已结束；按 Enter 进入窗口2阶段：")
    window2_success = run_task2_window2(
        station,
        navigation,
        window_handle_detector,
    )
    while not window2_success:
        print(
            "任务2窗口2未完成；保持当前作业点，恢复准备姿态后原地重试",
            flush=True,
        )
        recover_window2_for_retry(station)
        input("窗口2准备姿态已恢复；按 Enter 在当前作业点重新执行窗口2：")
        window2_success = run_window2_action(
            station,
            navigation,
            window_handle_detector,
        )

    if not window1_success:
        print(
            "任务2窗口2已完成；返回窗口1补做，导航只执行一次",
            flush=True,
        )
        input("窗口2已完成；确认返回窗口1路径安全后按 Enter：")
        window1_success = run_task2_window1(
            station,
            navigation,
            window_handle_detector,
        )
        while not window1_success:
            print(
                "任务2窗口1补做未完成；保持当前作业点原地重试",
                flush=True,
            )
            window1_success = run_task2_window1(
                station,
                navigation,
                window_handle_detector,
                navigate_to_action=False,
            )

    input("两个窗口均已完成；确认路径安全后按 Enter 前往h270_task2_check：")
    run_until_success(
        "前往任务2判定点",
        lambda: (
            navigation.moveTo(
                "h270_task2_check",
                correct=True,
                rotate=False,
            ),
            True,
        )[1],
        automatic=True,
    )
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    input("已到任务2判定点；确认恢复路径安全后按 Enter 恢复应急初始姿态：")
    run_until_success(
        "任务2判定点恢复应急初始姿态",
        lambda: apply_startpoint_pose(station, initial_pose),
        automatic=True,
    )
    return window1_success and window2_success


# ---------------------------------------------------------------------------
# 任务3 隔离带：抓取（带1/带2）+ 拉隔离带


def main() -> None:
    initial_pose = json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))
    window_handle_detector = WindowHandleYolo()
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
            settings.prepare_route(settings.TEST_NAVIGATION_ROUTE_STEM),
            initial_node="h270_task2_start",
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            if not navigation.checkState():
                raise RuntimeError("机器人实测位置不在h270_task2_start")
            while input(
                "已确认位于h270_task2_start；输入 y 恢复初始姿态并开始任务二："
            ).strip().lower() != "y":
                print("未输入 y，保持等待，不执行任务二", flush=True)
            run_until_success(
                "任务二独立测试恢复应急初始姿态",
                lambda: apply_startpoint_pose(
                    station,
                    initial_pose,
                    skip_safe_body_pose=False,
                ),
            )
            if not run_task2_windows(
                station,
                navigation,
                window_handle_detector,
            ):
                raise RuntimeError("任务二两个窗口未全部完成")
            if not navigation.checkState():
                raise RuntimeError("任务二完成后实测位置不在h270_task2_check")
            print(
                "任务二独立测试完成，机器人已到达h270_task2_check",
                flush=True,
            )
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
