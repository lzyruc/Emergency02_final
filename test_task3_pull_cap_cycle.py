#!/usr/bin/env python3
"""Live task3 pull-cap, pull, slot insertion, release and recovery test.

Sequence: save joints/body -> cap x/y -> cap z -> close -> waist yaw +32 deg
-> configured right shoulder/elbow pull -> right TCP +z retreat -> re-detect cap/reel
-> nearest slot x/y -> TCP z -0.07 m -> open -> return the configured task3
right-wrist observation pose.

The robot and MoveIt processes from ``agent/xrocs_api.md`` must be running.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

import settings
from arm_motion import moveit_joints
from task3_pull_cap_grasp import (
    PullCapRecognitionError,
    align_task3_reel_slot,
    grasp_task3_pull_cap,
    move_right_tcp_local_z,
)
from waist_leg_motion import execute_synced_waist_leg_action
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


OUTPUT_ROOT = Path(__file__).resolve().parent / "output/task3_pull_cap_cycle"
TASK3_OBSERVE_POSE_PATH = (
    Path(__file__).resolve().parent
    / "poses_task3/隔离带_right_grip_with_right_observe.json"
)
PULL_RIGHT_ARM_JOINT_INDEX = {
    "shoulder_pitch": 0,
    "elbow_pitch": 3,
    "elbow_yaw": 4,
}


def task3_pull_right_arm_joint_delta_rad() -> np.ndarray:
    """Return the configured 7-DOF right-arm pull delta in joint order."""
    configured = settings.TASK["task3_pull_cap"]["pull_right_arm_joint_delta_rad"]
    if not isinstance(configured, dict):
        raise RuntimeError("pull_right_arm_joint_delta_rad必须是关节名到弧度的字典")
    expected_names = set(PULL_RIGHT_ARM_JOINT_INDEX)
    configured_names = set(configured)
    if configured_names != expected_names:
        missing = sorted(expected_names - configured_names)
        unexpected = sorted(configured_names - expected_names)
        raise RuntimeError(
            f"右臂拉带关节增量配置不完整：缺少={missing}，未知={unexpected}"
        )

    delta = np.zeros(7, dtype=np.float64)
    for name, index in PULL_RIGHT_ARM_JOINT_INDEX.items():
        delta[index] = float(configured[name])
    if not np.all(np.isfinite(delta)):
        raise RuntimeError("右臂拉带关节增量包含非有限数值")
    return delta


def save_recognition_point_image(
    recognition_dir: Path,
    point_uv: list[int] | tuple[int, int],
    *,
    label: str,
) -> Path:
    """Mark a returned recognition point on its source RGB frame."""
    raw_image_path = recognition_dir / "rgb_raw.png"
    image = cv2.imread(str(raw_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取本次识别原图：{raw_image_path}")

    point = np.asarray(point_uv, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise RuntimeError(f"识别点格式无效：{point_uv!r}")
    u, v = (int(round(float(value))) for value in point)
    height, width = image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        raise RuntimeError(
            f"识别点({u}, {v})超出图像范围：width={width}, height={height}"
        )

    # 黑色外轮廓确保标记在亮色、暗色背景中都清晰可见。
    cv2.circle(image, (u, v), 15, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.circle(image, (u, v), 15, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(
        image,
        (u, v),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=3,
        line_type=cv2.LINE_AA,
    )
    text = f"{label} ({u}, {v})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness,
    )
    if text_width > width - 8:
        font_scale = max(0.35, font_scale * (width - 8) / text_width)
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness,
        )
    text_x = u + 20
    if text_x + text_width > width - 4:
        text_x = u - text_width - 20
    text_x = max(4, min(text_x, width - text_width - 4))
    text_y = v - 20
    if text_y - text_height < 4:
        text_y = v + text_height + 20
    text_y = max(text_height + 4, min(text_y, height - baseline - 4))
    text_origin = (text_x, text_y)
    cv2.putText(
        image,
        text,
        text_origin,
        font,
        font_scale,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        text_origin,
        font,
        font_scale,
        (0, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    marked_image_path = recognition_dir / "selected_grasp_point.png"
    if not cv2.imwrite(str(marked_image_path), image):
        raise RuntimeError(f"抓取点标注图保存失败：{marked_image_path}")
    print(f"已保存本次识别抓取点标注图：{marked_image_path}", flush=True)
    return marked_image_path


def move_to_task3_observe_pose(
    station: Any,
    pose: dict[str, Any],
) -> None:
    """串行恢复任务3双臂和腰腿观察姿态，关节角单位为rad。"""
    if not moveit_joints(
        station,
        left=pose["left_arm"],
        right=pose["right_arm"],
    ):
        raise RuntimeError("任务3双臂观察姿态执行失败")
    execute_synced_waist_leg_action(
        station,
        {
            "waist": {"position": pose["waist"]},
            "leg": {"position": pose["leg"]},
        },
    )


def recover_after_recognition_failure(
    station: Any,
    observe_pose: dict[str, Any],
    output_dir: Path,
    stage: str,
    error: PullCapRecognitionError,
) -> dict[str, Any]:
    """Open both hands and restore task3 arm, waist and leg pose after detection failure."""
    print(f"任务3识别失败（{stage}）：{error}", flush=True)
    input(
        "识别失败；确认松开夹爪并返回初始观察姿态的路径安全，"
        "按 Enter 执行，Ctrl+C 终止："
    )
    station.execute_action(
        {
            "hand": {
                "position": {
                    "left": [float(settings.TASK["hands"]["left_open_position"])],
                    "right": [float(settings.TASK["hands"]["right_open_position"])],
                }
            }
        }
    )
    time.sleep(1.0)
    move_to_task3_observe_pose(station, observe_pose)
    result = {
        "success": False,
        "recognition_failed": True,
        "recognition_stage": stage,
        "recognition_error": str(error),
        "hands_opened_after_failure": True,
        "returned_to_task3_observe_pose": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cycle_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("识别失败恢复完成：夹爪已松开，机器人已返回初始观察姿态", flush=True)
    return result


def run_test_sequence(
    station: Any,
    output_dir: Path,
    observe_pose: dict[str, Any],
) -> dict[str, Any]:
    """Execute the complete right-cap pull and nearest-slot insertion cycle."""
    initial_state = station.get_robot_state()
    original_right_joints = np.asarray(
        initial_state["arm_joints"]["right"],
        dtype=np.float64,
    ).tolist()
    original_waist = np.asarray(
        initial_state["waist_joints"],
        dtype=np.float64,
    )
    original_waist_target = original_waist.tolist()
    yaw_delta = float(settings.TASK["task3_pull_cap"]["waist_yaw_delta_rad"])
    pull_right_arm_delta = task3_pull_right_arm_joint_delta_rad()
    rotated_waist_target = original_waist_target.copy()
    rotated_waist_target[0] += yaw_delta
    print(f"已保存右臂原始关节角（rad）：{original_right_joints}", flush=True)

    initial_recognition_dir = output_dir / "initial_cap_grasp"
    try:
        grasp = grasp_task3_pull_cap(
            station,
            {
                "left": float(settings.TASK["hands"]["left_open_position"]),
                "right": float(settings.TASK["hands"]["right_open_position"]),
            },
            execute=True,
            output_dir=initial_recognition_dir,
        )
    except PullCapRecognitionError as error:
        return recover_after_recognition_failure(
            station,
            observe_pose,
            output_dir,
            "初次小圆柱帽",
            error,
        )
    initial_point_image = save_recognition_point_image(
        initial_recognition_dir,
        grasp["cap_pixel_uv"],
        label="CAP GRASP",
    )
    if not grasp["hand_executed"]:
        raise RuntimeError("任务3小圆柱帽两段抓取未完成")

    post_grasp_right_joints = np.asarray(
        station.get_robot_state()["arm_joints"]["right"],
        dtype=np.float64,
    )
    if post_grasp_right_joints.shape != (7,) or not np.all(
        np.isfinite(post_grasp_right_joints)
    ):
        raise RuntimeError("抓取后的右臂关节状态无效")
    pull_right_arm_target = post_grasp_right_joints + pull_right_arm_delta

    print(f"腰部yaw增加{np.degrees(yaw_delta):.1f}度", flush=True)
    input(
        f"确认腰部yaw {np.degrees(yaw_delta):+.1f}度路径安全，"
        "按 Enter 执行；Ctrl+C 终止："
    )
    execute_synced_waist_leg_action(
        station,
        {"waist": {"position": rotated_waist_target}},
    )

    pull_right_arm_delta_deg = np.degrees(pull_right_arm_delta)
    print(
        "右臂拉带关节增量："
        f"肩俯仰{pull_right_arm_delta_deg[0]:+.1f}度，"
        f"肘俯仰{pull_right_arm_delta_deg[3]:+.1f}度，"
        f"肘偏航{pull_right_arm_delta_deg[4]:+.1f}度",
        flush=True,
    )
    input("确认腰转后右臂肩、肘关节拉带路径安全，按 Enter 执行MoveIt；Ctrl+C 终止：")
    if not moveit_joints(station, right=pull_right_arm_target.tolist()):
        raise RuntimeError("右臂肩、肘关节拉带动作失败")

    retreat_z = float(settings.TASK["task3_pull_cap"]["post_grasp_retreat_z_tcp_m"])
    print(f"右TCP局部z移动{retreat_z:+.3f} m", flush=True)
    input(
        f"确认抓取后右TCP局部z {retreat_z * 100:+.1f} cm路径安全，"
        "按 Enter 执行；Ctrl+C 终止："
    )
    retreat_succeeded, retreat_pose = move_right_tcp_local_z(
        station,
        retreat_z,
    )
    if not retreat_succeeded:
        raise RuntimeError("抓取后右TCP局部z移动失败")

    slot_recognition_dir = output_dir / "slot_recognition"
    try:
        slot = align_task3_reel_slot(
            station,
            execute=True,
            output_dir=slot_recognition_dir,
        )
    except PullCapRecognitionError as error:
        return recover_after_recognition_failure(
            station,
            observe_pose,
            output_dir,
            "拉动后小圆及插槽",
            error,
        )
    slot_point_image = save_recognition_point_image(
        slot_recognition_dir,
        slot["selected_slot_pixel_uv"],
        label="SELECTED SLOT",
    )
    if not slot["z_motion_succeeded"]:
        raise RuntimeError("插槽x/y对齐或z插入失败")

    time.sleep(2.0)
    print("插槽动作完成，松开右夹爪", flush=True)
    input("确认可以释放拉条，按 Enter 松开右夹爪；Ctrl+C 终止：")
    station.execute_action(
        {
            "hand": {
                "position": {
                    "left": [float(settings.TASK["hands"]["left_open_position"])],
                    "right": [float(settings.TASK["hands"]["right_open_position"])],
                }
            }
        }
    )
    time.sleep(1.0)
    print("恢复隔离带右手抓取/右腕观察姿态", flush=True)
    input("确认双臂和腰腿返回观察姿态的路径安全；按 Enter 执行，Ctrl+C 终止：")
    move_to_task3_observe_pose(station, observe_pose)

    summary = {
        "original_right_arm_joints_rad": original_right_joints,
        "original_waist_target_rad": original_waist_target,
        "rotated_waist_target_rad": rotated_waist_target,
        "post_grasp_right_arm_joints_rad": post_grasp_right_joints.tolist(),
        "pull_right_arm_joint_delta_rad": pull_right_arm_delta.tolist(),
        "pull_right_arm_target_joints_rad": pull_right_arm_target.tolist(),
        "grasp": grasp,
        "post_grasp_retreat_z_tcp_m": retreat_z,
        "post_grasp_retreat_target_pose_waist_xyz_xyzw": retreat_pose,
        "slot": slot,
        "recognition_point_images": {
            "initial_cap_grasp": str(initial_point_image.relative_to(output_dir)),
            "selected_reel_slot": str(slot_point_image.relative_to(output_dir)),
        },
        "right_hand_opened_after_slot": True,
        "returned_to_task3_observe_pose": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cycle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"任务3完整循环测试完成：{output_dir}", flush=True)
    return summary


def main() -> None:
    yaw_delta_deg = np.degrees(
        float(settings.TASK["task3_pull_cap"]["waist_yaw_delta_rad"])
    )
    pull_right_arm_delta_deg = np.degrees(task3_pull_right_arm_joint_delta_rad())
    slot_insert_z = float(settings.TASK["task3_pull_cap"]["slot_insert_z_tcp_m"])
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    right_camera_name = config.get("camera", {})["right"]["camera_name"].strip("/")
    # 当前驱动发布rect彩色图，xrocs订阅raw图；重映射仅作用于本测试进程。
    sys.argv.extend(
        [
            "--ros-args",
            "-r",
            f"/{right_camera_name}/color/image_raw:="
            f"/{right_camera_name}/color/image_rect_raw",
        ]
    )
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        observe_pose = json.loads(TASK3_OBSERVE_POSE_PATH.read_text(encoding="utf-8"))
        input(
            "确认双臂和腰腿回到隔离带右手抓取/右腕观察姿态的路径安全；"
            "按 Enter 执行，Ctrl+C 终止："
        )
        move_to_task3_observe_pose(station, observe_pose)
        input(
            "确认右臂两段抓取、"
            f"腰yaw {yaw_delta_deg:+.1f}度、"
            f"右肩俯仰{pull_right_arm_delta_deg[0]:+.1f}度、"
            f"右肘俯仰{pull_right_arm_delta_deg[3]:+.1f}度、"
            f"右肘偏航{pull_right_arm_delta_deg[4]:+.1f}度、"
            "TCP z上抬、插槽x/y、"
            f"TCP z {slot_insert_z * 100:+.1f}cm、松爪及返回路径全部安全；"
            "按 Enter 开始，Ctrl+C 终止："
        )
        run_test_sequence(station, output_dir, observe_pose)
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
