#!/usr/bin/env python3
"""测试场地任务三双侧抓取入口：从h270_task3_check出发并返回该点。

复用 game_emergency_test.py 的识别、抓取和重试策略；关键运动前保留
人工 Enter 确认。本脚本不测试任务三拉放路线，返回检查点后松开双夹爪。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

import settings
from agent.navigation import NavigationGraph
from arm_motion import moveit_joints
import black_grasp_single as black_grasp_module
from waist_leg_motion import execute_synced_waist_leg_action
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


MAX_STAGE_ATTEMPTS = 3
CAMERA_SETTLE_S = 1.0

# 单侧首次 MoveIt 失败后的底盘二次抓取平移。
# base_footprint：+x 前、+y 左，单位米。
TASK3_RIGHT_REGRASP_MOVE_XY_M = (0.10, 0.0)
TASK3_LEFT_REGRASP_MOVE_XY_M = (0.05, 0.05)


def grasp_black_single_for_test(
    station: Any,
    arm: str,
    hand_positions: dict[str, float],
    output_dir: Path,
    *,
    split_tcp_z_motion: bool = True,
    component_selection: str | None = None,
) -> dict[str, Any] | None:
    """执行正式感知，按需拆分TCP-Z修正，并保存漏检输入。"""
    captured: dict[str, Any] = {}
    two_stage_motion: dict[str, Any] = {}
    camera_error: RuntimeError | None = None
    detection_error: RuntimeError | None = None
    original_read_wrist_camera = black_grasp_module.read_wrist_camera
    original_detector = black_grasp_module.BlackCylinderDetector
    original_moveit_cartesians = black_grasp_module.moveit_cartesians

    def recording_read_wrist_camera(
        camera_station: Any,
        camera_config: dict[str, Any],
    ) -> tuple[Any, Any]:
        nonlocal camera_error
        try:
            bgr, depth_mm = original_read_wrist_camera(
                camera_station,
                camera_config,
            )
        except RuntimeError as error:
            camera_error = error
            raise
        captured["bgr"] = bgr
        captured["depth_mm"] = depth_mm
        return bgr, depth_mm

    class RecordingDetector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.detector = original_detector(*args, **kwargs)

        def detect(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal detection_error
            if component_selection is not None:
                kwargs["component_selection"] = component_selection
            try:
                return self.detector.detect(*args, **kwargs)
            except RuntimeError as error:
                detection_error = error
                raise

    def two_stage_moveit_cartesians(
        move_station: Any,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        target_pose_value = kwargs.get(arm)
        if not split_tcp_z_motion or args or target_pose_value is None:
            return original_moveit_cartesians(move_station, *args, **kwargs)

        target_pose = [float(value) for value in target_pose_value]
        current_pose = list(
            black_grasp_module.pose_xyz_xyzw(
                move_station.get_robot_state()["arm_pose"][arm]
            )
        )
        waist_from_tcp = black_grasp_module._waist_from_tcp_matrix(current_pose)
        rotation = [row[:3] for row in waist_from_tcp[:3]]
        delta_waist = [
            target_pose[index] - current_pose[index] for index in range(3)
        ]
        delta_tcp = [
            sum(rotation[row][column] * delta_waist[row] for row in range(3))
            for column in range(3)
        ]
        first_delta_tcp = [delta_tcp[0], delta_tcp[1], delta_tcp[2] * 0.4]
        first_delta_waist = [
            sum(rotation[row][column] * first_delta_tcp[column] for column in range(3))
            for row in range(3)
        ]
        first_target_pose = [
            *[
                current_pose[index] + first_delta_waist[index]
                for index in range(3)
            ],
            *target_pose[3:],
        ]
        first_kwargs = dict(kwargs)
        first_kwargs[arm] = first_target_pose
        two_stage_motion.update(
            {
                "full_delta_tcp_m": delta_tcp,
                "first_delta_tcp_m": first_delta_tcp,
                "first_target_pose_waist_xyz_xyzw": first_target_pose,
                "final_target_pose_waist_xyz_xyzw": target_pose,
                "first_stage_succeeded": False,
                "second_stage_succeeded": False,
            }
        )
        print(
            f"任务3单测{arm}臂第一段：TCP X/Y=100%，TCP Z=40%，"
            f"delta_tcp={first_delta_tcp} m",
            flush=True,
        )
        first_succeeded = original_moveit_cartesians(
            move_station,
            **first_kwargs,
        )
        two_stage_motion["first_stage_succeeded"] = first_succeeded
        if not first_succeeded:
            return False

        input(
            f"{arm}臂第一段已完成；确认剩余TCP-Z方向路径安全，"
            "按 Enter 执行第二段剩余60% Z："
        )
        print(
            f"任务3单测{arm}臂第二段：TCP X/Y不再增加，补完剩余60% TCP Z",
            flush=True,
        )
        second_succeeded = original_moveit_cartesians(
            move_station,
            **kwargs,
        )
        two_stage_motion["second_stage_succeeded"] = second_succeeded
        return second_succeeded

    black_grasp_module.read_wrist_camera = recording_read_wrist_camera
    black_grasp_module.BlackCylinderDetector = RecordingDetector
    black_grasp_module.moveit_cartesians = two_stage_moveit_cartesians
    try:
        try:
            result = black_grasp_module.grasp_black_single(
                station,
                arm,
                hand_positions,
                output_dir=output_dir,
                interactive_confirm=True,
            )
        except RuntimeError:
            if detection_error is None:
                raise
            result = None
    finally:
        black_grasp_module.read_wrist_camera = original_read_wrist_camera
        black_grasp_module.BlackCylinderDetector = original_detector
        black_grasp_module.moveit_cartesians = original_moveit_cartesians

    if result is not None and split_tcp_z_motion:
        result["motion_mode"] = "task3_only_two_stage_tcp_xy100_z40_then_z60"
        result["two_stage_motion"] = two_stage_motion
        (output_dir / "target.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    failure = detection_error or camera_error
    if result is not None or failure is None:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    if captured:
        bgr = captured["bgr"]
        depth_mm = captured["depth_mm"]
        cv2.imwrite(str(output_dir / "rgb_raw.jpg"), bgr)
        cv2.imwrite(str(output_dir / "depth_raw_mm.png"), depth_mm)
        cv2.imwrite(
            str(output_dir / "gripper_static_mask.png"),
            black_grasp_module.gripper_mask(depth_mm.shape).astype("uint8") * 255,
        )
    (output_dir / "failure.json").write_text(
        json.dumps(
            {
                "stage": "detection" if detection_error else "camera",
                "arm": arm,
                "error": str(failure),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{arm}腕识别失败输入已保存到{output_dir}：{failure}", flush=True)
    return None


def retry_task3_grasp_after_base_move(
    station: Any,
    navigation: NavigationGraph,
    arm: str,
    hand_positions: dict[str, float],
    first_result: dict[str, Any] | None,
    retry_move_xy_m: tuple[float, float],
    recovery_node: str,
) -> bool:
    """首次TCP MoveIt失败时，平移底盘并再抓一次。"""
    if first_result is not None and first_result["motion_succeeded"]:
        return True

    input(f"{arm}臂首次TCP MoveIt失败；按 Enter 张开当前夹爪：")
    station.execute_action(
        {
            "hand": {
                "position": {
                    "left": [hand_positions["left"]],
                    "right": [hand_positions["right"]],
                }
            }
        }
    )
    time.sleep(2.0)
    retry_x_m, retry_y_m = retry_move_xy_m
    input(
        "确认底盘微调路径安全，按 Enter 执行 "
        f"moveBy(x={retry_x_m:+.3f}, y={retry_y_m:+.3f}, theta=0)："
    )
    navigation.moveBy(retry_x_m, retry_y_m, 0.0, correct=False)
    time.sleep(CAMERA_SETTLE_S)
    input(f"确认{arm}腕视野及机械臂路径安全，按 Enter 重新识别并抓取：")
    retry_result = grasp_black_single_for_test(
        station,
        arm,
        hand_positions,
        (
            settings.RUNTIME_DIR
            / "task3_only"
            / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{arm}_retry"
        ),
    )
    retry_success = (
        retry_result is not None and retry_result["motion_succeeded"]
    )
    input(f"确认返回{recovery_node}的路径安全，按 Enter 恢复观察点：")
    navigation.recoverState(recovery_node, method="move_by")
    if not retry_success:
        print(f"任务3 {arm}臂底盘微调后重抓仍失败", flush=True)
    return retry_success


def run_task3_grasp_belt1(
    station: Any,
    navigation: NavigationGraph,
    *,
    prepared: bool = False,
) -> bool:
    """隔离带1（右侧）识别与抓取。"""
    if not prepared:
        task3_right = settings.TASK3_RIGHT_PREPARE_POSE
        input(
            "确认腰腿和双臂路径安全，按 Enter 先下发右侧隔离带腰腿姿态，"
            "0.5秒后切换双臂抓取姿态："
        )
        execute_synced_waist_leg_action(
            station,
            {
                "waist": {"position": task3_right["waist"]},
                "leg": {"position": task3_right["leg"]},
            },
        )
        time.sleep(0.5)
        if not moveit_joints(
            station,
            left=task3_right["left_arm"],
            right=task3_right["right_arm"],
        ):
            print("双臂到右侧黑色隔离带抓取姿态失败", flush=True)
            return False
        time.sleep(3.0)
        input("按 Enter 将双夹爪预置为 left=0.000、right=0.000：")
        station.execute_action(
            {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
        )
        time.sleep(2.0)
    input("确认右腕视野及右臂路径安全，按 Enter 识别并抓取右侧隔离带：")
    right_result = grasp_black_single_for_test(
        station,
        "right",
        {"left": 0.0, "right": 0.0},
        (
            settings.RUNTIME_DIR
            / "task3_only"
            / f"{datetime.now():%Y%m%d_%H%M%S_%f}_right_initial"
        ),
    )
    right_success = retry_task3_grasp_after_base_move(
        station,
        navigation,
        "right",
        {"left": 0.0, "right": 0.0},
        right_result,
        TASK3_RIGHT_REGRASP_MOVE_XY_M,
        "h270_task3_check",
    )
    if not right_success:
        print("右侧黑色隔离带抓取失败", flush=True)
    return right_success


def run_task3_grasp_belt2(
    station: Any,
    navigation: NavigationGraph,
) -> bool:
    """隔离带2（左侧）识别与抓取。"""
    input("确认右侧目标已抓稳且路径安全，按 Enter 前往h270_task3_right：")
    navigation.moveTo("h270_task3_right", correct=True)
    task3_left = settings.TASK3_LEFT_PREPARE_POSE
    input(
        "确认腰部和左臂路径安全，按 Enter 先下发左侧隔离带腰部姿态，"
        "0.5秒后只切换左臂预备姿态："
    )
    execute_synced_waist_leg_action(
        station,
        {"waist": {"position": task3_left["waist"]}},
    )
    time.sleep(0.5)
    if not moveit_joints(station, left=task3_left["left_arm"]):
        print("左臂到左侧黑色隔离带预备姿态失败", flush=True)
        return False
    time.sleep(5.0)
    input("确认路径安全，按 Enter 前往h270_task3_left：")
    navigation.moveTo("h270_task3_left", correct=True)
    input("按 Enter 将夹爪预置为 left=0.000、right=1.000：")
    station.execute_action(
        {"hand": {"position": {"left": [0.0], "right": [1.0]}}}
    )
    time.sleep(2.0)
    input("确认左腕视野及左臂路径安全，按 Enter 识别并抓取左侧隔离带：")
    left_result = grasp_black_single_for_test(
        station,
        "left",
        {"left": 0.0, "right": 1.0},
        (
            settings.RUNTIME_DIR
            / "task3_only"
            / f"{datetime.now():%Y%m%d_%H%M%S_%f}_left_initial"
        ),
    )
    left_success = retry_task3_grasp_after_base_move(
        station,
        navigation,
        "left",
        {"left": 0.0, "right": 1.0},
        left_result,
        TASK3_LEFT_REGRASP_MOVE_XY_M,
        "h270_task3_left",
    )
    if not left_success:
        print("左侧黑色隔离带抓取失败", flush=True)
    return left_success


def run_task3_grasp_belts(
    station: Any,
    navigation: NavigationGraph,
    *,
    first_round_prepared: bool = False,
) -> bool:
    """局部重抓仍失败时，回任务三检查点重做，最多三轮。"""
    for attempt_index in range(1, MAX_STAGE_ATTEMPTS + 1):
        print(
            f"任务3隔离带抓取第{attempt_index}/{MAX_STAGE_ATTEMPTS}轮",
            flush=True,
        )
        belt1_success = run_task3_grasp_belt1(
            station,
            navigation,
            prepared=first_round_prepared and attempt_index == 1,
        )
        belt2_success = (
            run_task3_grasp_belt2(station, navigation)
            if belt1_success
            else False
        )
        if belt1_success and belt2_success:
            return True

        input("本轮抓取失败；按 Enter 张开双夹爪：")
        station.execute_action(
            {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
        )
        time.sleep(2.0)
        input("确认路径安全，按 Enter 返回h270_task3_check：")
        navigation.recoverState("h270_task3_check", method="move_by")
        if attempt_index < MAX_STAGE_ATTEMPTS:
            input("已回到h270_task3_check；按 Enter 开始下一轮抓取：")
    return False


def run_task3_belts(
    station: Any,
    navigation: NavigationGraph,
    *,
    first_round_prepared: bool = False,
) -> bool:
    """只测试两侧抓取；结束时回任务三检查点并松开双夹爪。"""
    grasp_success = run_task3_grasp_belts(
        station,
        navigation,
        first_round_prepared=first_round_prepared,
    )
    if not grasp_success:
        print("任务3抓取三轮均失败，机器人已回到h270_task3_check", flush=True)
        input("按 Enter 在h270_task3_check松开双夹爪并结束测试：")
        station.execute_action(
            {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
        )
        time.sleep(2.0)
        return False

    input(
        "两个隔离柱均已抓取；确认携带目标返回路径安全，"
        "按 Enter 恢复h270_task3_check："
    )
    navigation.recoverState("h270_task3_check", method="move_by")
    input("已回到h270_task3_check；按 Enter 松开双夹爪：")
    station.execute_action(
        {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
    )
    time.sleep(2.0)
    print("任务3双侧抓取测试完成，已在h270_task3_check释放目标", flush=True)
    return True


def main() -> None:
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
            initial_node="h270_task3_check",
            navigation_mode="move_by",
            linear_speed=float(navigation_config["linear_speed_m_s"]),
            angular_speed=float(navigation_config["angular_speed_rad_s"]),
            compress_path=False,
        ) as navigation:
            if not navigation.checkState():
                raise RuntimeError("机器人实测位置不在h270_task3_check")
            input("已确认位于h270_task3_check；按 Enter 松开双夹爪：")
            station.execute_action(
                {"hand": {"position": {"left": [0.0], "right": [0.0]}}}
            )
            time.sleep(2.0)
            task3_right = settings.TASK3_RIGHT_PREPARE_POSE
            input(
                "确认腰腿和双臂路径安全，按 Enter 先下发右侧隔离带腰腿姿态，"
                "0.5秒后切换双臂抓取预备姿态："
            )
            execute_synced_waist_leg_action(
                station,
                {
                    "waist": {"position": task3_right["waist"]},
                    "leg": {"position": task3_right["leg"]},
                },
            )
            time.sleep(0.5)
            if not moveit_joints(
                station,
                left=task3_right["left_arm"],
                right=task3_right["right_arm"],
            ):
                raise RuntimeError("任务三独立测试切换右侧预备双臂姿态失败")
            time.sleep(3.0)
            input("初始化完成；按 Enter 开始任务三独立测试：")
            if not run_task3_belts(
                station,
                navigation,
                first_round_prepared=True,
            ):
                raise RuntimeError("任务三独立测试未完成")
            if not navigation.checkState():
                raise RuntimeError("任务三完成后实测位置不在h270_task3_check")
            input("任务三独立测试完成，已在h270_task3_check松爪；按 Enter 结束：")
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
