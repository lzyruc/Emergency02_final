# body_controller.py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from eai_manipulator_msgs.action import JointSpace


class BodyController(Node):
    """
    Body 4 joints controller

    joints:
        0: first_leg_pitch_joint
        1: second_leg_pitch_joint
        2: waist_pitch_joint
        3: waist_yaw_joint
    """

    def __init__(self):
        super().__init__("body_4_joints")

        self.client = ActionClient(
            self,
            JointSpace,
            "/jointspace_body_controller/jointspace"
        )

        self.get_logger().info(
            "BodyController initialized"
        )


    def wait_ready(self, timeout_sec=10.0):
        self.get_logger().info(
            "Waiting for body controller..."
        )

        if not self.client.wait_for_server(timeout_sec=timeout_sec):
            raise RuntimeError("Body controller action server unavailable")

        self.get_logger().info(
            "Body controller ready"
        )


    def move_to(
        self,
        target_positions,
        wait=True
    ):
        """
        target_positions:
        [
            first_leg_pitch_joint,
            second_leg_pitch_joint,
            waist_pitch_joint,
            waist_yaw_joint
        ]
        """

        if len(target_positions) != 4:
            raise ValueError(
                "Need 4 joint positions"
            )

        goal = JointSpace.Goal()

        goal.target_positions = [
            float(x) for x in target_positions
        ]

        future = self.client.send_goal_async(goal)


        if not wait:
            return future


        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "Body goal rejected"
            )
            return False


        result_future = (
            goal_handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future
        )


        result = result_future.result().result
        if not result.success:
            self.get_logger().error(
                f"Body motion failed: {result.result_msg}"
            )
            return False

        self.get_logger().info(
            f"Body motion finished: {result.result_msg}"
        )

        return True
