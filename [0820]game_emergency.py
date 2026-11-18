import time

import settings
from arm_motion import moveit_joints
from black_grasp import grasp_black
from blue_grasp import grasp_blue
from camera_io import read_head_camera
from head_approach import get_chassis_pose_from_rgbd
from agent.navigation import NavigationGraph
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import TienYi2ProRos2Station
from xrocs.utils.path_manager import path_manager
import math

#zyt
from body_controller import BodyController
# 双臂收回姿态（7 DoF 关节角）
ARM_HOME_LEFT = [0.0, 0.0, -0.261807, -0.523639, -1.0082, -0.261807, 0.0]
ARM_HOME_RIGHT = [0.0, 0.0, 0.261783, -0.523615, 1.00772, -0.261783, 0.0]


def warn_if_failed(succeeded: bool, step_desc: str) -> None:
    """动作未成功时打印警告，由操作员在下一次确认时决定是否继续。"""
    if not succeeded:
        input(f"警告：{step_desc}未成功完成，请检查机器人状态", flush=True)


def joint_spd_from_delta(
    current_pos,
    target_pos,
    *,
    min_spd: float = 0.1,
    max_spd: float = 0.5,
) -> list[float]:
    """按 |Δpos| 同步缩放各关节 spd（仿照 _height_sync_spd），限幅到 [min_spd, max_spd]。"""
    n = len(target_pos)
    if current_pos is None:
        return [min_spd] * n
    if len(current_pos) != n:
        raise ValueError(
            f"current_pos 与 target_pos 长度不一致: {len(current_pos)} vs {n}"
        )

    delta = [abs(float(t) - float(c)) for c, t in zip(current_pos, target_pos)]
    moving = [d for d in delta if d > 0.0]
    if not moving:
        return [min_spd] * n

    sync_time = max(moving) / max_spd
    spd = []
    for d in delta:
        if d == 0.0:
            spd.append(min_spd)
        else:
            spd.append(min(max_spd, max(min_spd, d / sync_time)))
    return spd

def publish_body_joint_command(robot_station, target) -> None:
    """向 body topic 发布 [腿1, 腿2, 腰俯仰, 腰偏航] 四关节目标（rad）。"""
    command = [float(value) for value in target]
    if len(command) != 4:
        raise ValueError(f"body 关节目标必须为 4 维，实际为 {len(command)} 维")
    if not all(math.isfinite(value) for value in command):
        raise ValueError(f"body 关节目标必须均为有限数值：{command}")
    robot = robot_station.get_robot_handle().get("robot")
    robot.mc.direct_ctrler.pub_jointspace_commands_body(command)
    time.sleep(0.2)

def publish_body_height_command(
    robot_station,
    *,
    height: float,
    yaw: float,
    pitch: float = 0.0,
) -> None:
    """按现有身高语义换算关节角，并通过 body topic 一次性下发腰腿目标。"""
    height_m = robot_station._MIN_LEG_HEIGHT_M + (
        robot_station._MAX_LEG_HEIGHT_M - robot_station._MIN_LEG_HEIGHT_M
    ) * float(height)
    waist_pitch, leg_1, leg_2 = robot_station.leg_height_to_joint_radians(height_m)
    waist_pitch += 3.0 * float(pitch)
    publish_body_joint_command(
        robot_station,
        [leg_1, leg_2, waist_pitch, yaw],
    )

def execute_synced_waist_leg_action(
    robot_station,
    action,
) -> None:
    """按位移同步腰和/或腿的速度执行动作，并恢复默认速度。"""

    #target = [action['leg']['position'][0],action['leg']['position'][1],action['waist']['position'][1],action['waist']['position'][0]]

    #publish_body_joint_command(robot_station=robot_station,target=target)
    #robot_station.execute_action(action)
    #time.sleep(3)
    #return
    
    waist_target = (action.get("waist") or {}).get("position")
    leg_target = (action.get("leg") or {}).get("position")
    if waist_target is None and leg_target is None:
        raise ValueError("action 必须包含 waist.position 或 leg.position")

    robot = robot_station.get_robot_handle().get("robot")
    waist_dof = len(waist_target) if waist_target is not None else 0
    leg_dof = len(leg_target) if leg_target is not None else 0
    current_pos = []
    target_pos = []
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
    )

    try:
        if waist_target is not None:
            robot.waist_ctrler.set_motor_config(spd=all_spd[:waist_dof])
        if leg_target is not None:
            robot.leg_ctrler.set_motor_config(spd=all_spd[waist_dof:])
        robot_station.execute_action(action)
        time.sleep(0.2)
    finally:
        if waist_target is not None:
            robot.waist_ctrler.set_motor_config(spd=[0.1] * waist_dof)
        if leg_target is not None:
            robot.leg_ctrler.set_motor_config(spd=[0.1] * leg_dof)


if __name__ == "__main__":
    # 初始化
    config_path = path_manager.config_toml_path
    cfg_loader = ConfigLoader(config_path)
    config = cfg_loader.get_config()

    robot_config = config.get("robot", {})
    robot_station = TienYi2ProRos2Station(
        robot_dict=robot_config,
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict=config.get("force_sensor", {}),
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    time.sleep(1)

    pick_blue = settings.PRESETS["pick_blue"]
    navigation = NavigationGraph(
        settings.prepare_route(),
        initial_node="startpoint",
        navigation_mode="move_by",
        linear_speed=0.2,
        angular_speed=0.3,
        compress_path=False,
    )

    try:
        """
        准备
        """
        input("即将执行：双臂/腰腿/头部回零，确认安全后按 Enter 继续；Ctrl+C 终止：")
        # 0.1、回零
        execute_synced_waist_leg_action(
            robot_station,
            {
                "hand": {"position": {"left": [0.0], "right": [0.0]}},
                "waist": {"position": [0.0, 0.0]},
                "leg": {"position": [0.0, 0.0]},
                "head": {"position": [0.0, 0.0, 0.0]},
            },
        )
        time.sleep(3)
        # 等待腰腿回零后再执行双臂回零
        succeeded = moveit_joints(
            robot_station,
            left=[0.0, 0.08, 0.0, -0.16, 0.01, 0.0, 0.0],
            right=[0.0, -0.08, 0.0, -0.16, -0.01, 0.0, 0.0],
        )
        warn_if_failed(succeeded, "双臂回零")

        """
        任务1 识别、抓取并放置蓝色文件筐
        """
        print("即将执行：头部识别蓝色文件筐并导航靠近")
        # 1、头部 RGB-D 识别蓝色文件筐
        # 没有识别到文件筐时跳过任务 1 的后续流程，直接进入任务 2。
        head_image, head_depth = read_head_camera(robot_station)
        # 获取底盘姿态
        chassis_pose = get_chassis_pose_from_rgbd(
            robot_station, head_image, head_depth, target_class="basket"
        )
        if chassis_pose is not None:
            # 2 导航到蓝色篮子，平移与转向各保留一档保守系数
            navigation.moveBy(chassis_pose[0] * 0.8, chassis_pose[1] * 0.6, 0.0)
            navigation.moveBy(0.0, 0.0, chassis_pose[2] * 0.5)

            print("即将执行：双臂到预抓取位、腰腿到抓取位")
            # 3、抓取
            # 3.1 双臂到预抓取位
            succeeded = moveit_joints(
                robot_station,
                left=pick_blue["left_arm"],
                right=pick_blue["right_arm"],
            )
            warn_if_failed(succeeded, "双臂到蓝框预抓取位")
            # 3.2 腰腿到抓取位：spd 同步缩放 spd（[0.1, 0.5]），执行后恢复默认 0.1
            execute_synced_waist_leg_action(
                robot_station,
                {
                    "leg": {"position": pick_blue["leg"]},
                    "waist": {"position": pick_blue["waist"]},
                    "hand": {"position": {"left": [0.0], "right": [0.0]}},
                },
            )
            time.sleep(5)

            print("即将执行：闭合夹爪并抓取蓝色文件筐")
            # 3.3 抓取
            # 获取右相机彩色和深度图，计算笛卡尔坐标，QP 控制到目标位置并闭合夹爪
            blue_result = grasp_blue(robot_station)
            blue_ok = (
                blue_result is not None and blue_result["motion_succeeded"]
            )
            warn_if_failed(blue_ok, "蓝色文件筐抓取")
            if not blue_ok:
                print("蓝色文件筐抓取失败，放弃任务1，进入任务2", flush=True)
            else:
                print("即将执行：拿起、导航到桌子、放下")
                # 3.4 拿起
                # 腰腿恢复零位
                execute_synced_waist_leg_action(
                    robot_station,
                    {
                        "waist": {"position": [0.0, 0.0]},
                        "leg": {"position": [0.0, 0.0]},
                    },
                )
                time.sleep(3)
                # 3.5 导航到桌子
                navigation.recoverState("h270_track_turn_1", method="move_by")
                navigation.turnTo(135.0)
                # 3.6 放下，并收回双臂，张开夹爪
                # TODO: 双臂放篮子动作
                robot_station.execute_action({"hand": {"position": {"left": [0.0], "right": [0.0]}}})
                time.sleep(1)
                navigation.turnTo(-90.0)
                succeeded = moveit_joints(
                    robot_station, left=ARM_HOME_LEFT, right=ARM_HOME_RIGHT
                )
                warn_if_failed(succeeded, "放下蓝框后收回双臂")

        """
        任务2 移动到窗口并执行开窗动作
        """
        input("即将执行：导航到任务2起点和1号窗口操作点，确认安全后按 Enter 继续；Ctrl+C 终止：")
        # 2.1、导航到任务 2 起点，并再次进行底盘方向对齐
        navigation.moveTo("h270_task2_start", correct=True, align=False)
        navigation.moveTo("h270_task2_start", correct=True, align=True)
        # 2.2、导航到 1 号窗口操作点
        navigation.moveTo("h270_window1_action", correct=False)
        # 2.3、1 号窗口四段循环：迫近 → 拧开把手 → 开窗 → 恢复
        # 2.3.1 迫近（step_0-3）
        input(f"即将执行：window1【迫近】step_{index}，确认安全后按 Enter 继续；Ctrl+C 终止：")
        for index in range(0, 4):
            if index == 3:
                # 迫近末步：闭合左夹爪抓住窗把手
                robot_station.execute_action(
                    {"hand": {"position": {"left": [1.0], "right": [0.0]}}}
                )
                time.sleep(2)
            window_pose = settings.WINDOW1_POSES[index]
            succeeded = moveit_joints(
                robot_station,
                left=window_pose["left_arm"],
                right=window_pose["right_arm"],
            )
            warn_if_failed(succeeded, f"window1【迫近】step_{index}")

        # 2.3.2 拧开把手（step_4-7）
        for index in range(4, 8):
            input(f"即将执行：window1【拧开把手】step_{index}，确认安全后按 Enter 继续；Ctrl+C 终止：")
            window_pose = settings.WINDOW1_POSES[index]
            succeeded = moveit_joints(
                robot_station,
                left=window_pose["left_arm"],
                right=window_pose["right_arm"],
            )
            warn_if_failed(succeeded, f"window1【拧开把手】step_{index}")

        # 2.3.3 开窗（step_8-10）
        for index in range(8, 11):
            input(f"即将执行：window1【开窗】step_{index}，确认安全后按 Enter 继续；Ctrl+C 终止：")
            window_pose = settings.WINDOW1_POSES[index]
            succeeded = moveit_joints(
                robot_station,
                left=window_pose["left_arm"],
                right=window_pose["right_arm"],
            )
            warn_if_failed(succeeded, f"window1【开窗】step_{index}")

        # 2.3.4 恢复（step_11-13）
        input(f"即将执行：window1【恢复】step_{index}，确认安全后按 Enter 继续；Ctrl+C 终止：")
        for index in range(11, 14):
            
            window_pose = settings.WINDOW1_POSES[index]
            succeeded = moveit_joints(
                robot_station,
                left=window_pose["left_arm"],
                right=window_pose["right_arm"],
            )
            warn_if_failed(succeeded, f"window1【恢复】step_{index}")

        input("即将执行：完成开窗、回收双臂、导航到2号窗口，确认安全后按 Enter 继续；Ctrl+C 终止：")
        # 2.4、完成开窗并回收双臂
        succeeded = moveit_joints(
            robot_station, left=ARM_HOME_LEFT, right=ARM_HOME_RIGHT
        )
        warn_if_failed(succeeded, "开窗后回收双臂")
        # 2.5、导航到 2 号窗口变换点，切换开窗姿态（待实现）
        # 2.6、导航到 2 号窗口操作点，执行 2 号开窗动作
        navigation.moveTo("h270_window2_action", correct=False)

        """
        任务3 分别抓取右侧和左侧黑色柱状体并返回放置点
        """
        input("即将执行：导航到检查点、右侧抓取姿态并抓取右侧黑色柱状体，确认安全后按 Enter 继续；Ctrl+C 终止：")
        # 3.1、导航到任务 3 检查点
        navigation.moveTo("h270_task3_check", correct=True)
        # 3.2、调整右侧抓取姿态，先到位双臂再动腰腿头
        task3_right = settings.TASK3_POSES["right"]
        succeeded = moveit_joints(
            robot_station,
            left=task3_right["left_arm"],
            right=task3_right["right_arm"],
        )
        warn_if_failed(succeeded, "双臂到右侧黑柱抓取姿态")
        execute_synced_waist_leg_action(
            robot_station,
            {
                "waist": {"position": task3_right["waist"]},
                "leg": {"position": task3_right["leg"]},
                "head": {"position": task3_right["head"]},
            },
        )
        time.sleep(3)
        # 3.3、右腕识别并抓取右侧黑色柱状体，左夹爪保持张开
        robot_station.execute_action({"hand": {"position": {"left": [0.0], "right": [0.0]}}})
        time.sleep(2)
        right_result = grasp_black(robot_station, "right", {"left": 0.0, "right": 0.0})
        warn_if_failed(
            right_result is not None and right_result["motion_succeeded"],
            "右侧黑色柱状体抓取",
        )

        input("即将执行：导航到右侧任务点、左侧抓取姿态并抓取左侧黑色柱状体，确认安全后按 Enter 继续；Ctrl+C 终止：")
        # 3.4、导航到右侧任务点
        navigation.moveTo("h270_task3_right", correct=True)
        # 3.5、调整左侧抓取姿态
        task3_left = settings.TASK3_POSES["left"]
        succeeded = moveit_joints(
            robot_station,
            left=task3_left["left_arm"],
            right=task3_left["right_arm"],
        )
        warn_if_failed(succeeded, "双臂到左侧黑柱抓取姿态")
        execute_synced_waist_leg_action(
            robot_station,
            {
                "waist": {"position": task3_left["waist"]},
                "leg": {"position": task3_left["leg"]},
                "head": {"position": task3_left["head"]},
            },
        )
        time.sleep(5)
        # 3.6、导航到左侧任务点
        navigation.moveTo("h270_task3_left", correct=True)
        # 3.7、左腕识别并抓取左侧黑色柱状体，右夹爪保持闭合
        robot_station.execute_action({"hand": {"position": {"left": [0.0], "right": [1.0]}}})
        time.sleep(2)
        left_result = grasp_black(robot_station, "left", {"left": 0.0, "right": 1.0})
        warn_if_failed(
            left_result is not None and left_result["motion_succeeded"],
            "左侧黑色柱状体抓取",
        )

        input("即将执行：携带两个目标返回放置点，确认安全后按 Enter 继续；Ctrl+C 终止：")
        # 3.8、携带两个目标返回放置点
        navigation.moveTo("h270_task3_back", correct=True)
        navigation.moveTo("h270_task3_return", correct=True)
        navigation.moveTo("h270_task3_place", correct=True)
    finally:
        navigation.close()
        robot_station.shutdown()
