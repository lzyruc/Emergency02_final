"""将头部或腕部相机像素和深度反投影到 ``WAIST_YAW_LINK``。

最小调用：

.. code-block:: python

    calibration_path = "agent/perception/config/camera_calibration.json"
    deprojector = Deprojector(station, calibration_path, "left")
    head_deprojector = Deprojector(station, calibration_path, "head")
    pose_waist = deprojector.deproject([u, v, depth_m], face_target=True)
    point_ground = deprojector.transform_to_ground(pose_waist[:3])
    poses_waist = deprojector.deproject_many(
        [[u0, v0, depth0_m], [u1, v1, depth1_m]]
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = ["Deprojector"]


class Deprojector:
    """复用相机标定和 xrocs Pinocchio 计算动态相机位姿。

    标定文件中的内参必须对应 YOLO 返回像素坐标所在的原始图像分辨率。
    ``T_mount_camera`` 是4×4齐次矩阵，方向为安装link到相机，长度单位为米。
    """

    def __init__(
        self,
        station: Any,
        calibration_path: str | Path,
        camera: Literal["head", "left", "right"],
    ) -> None:
        calibration = json.loads(
            Path(calibration_path).read_text(encoding="utf-8")
        )["cameras"][camera]
        camera_matrix = np.asarray(
            calibration["camera_matrix"], dtype=np.float64
        )
        T_mount_camera = np.asarray(
            calibration["T_mount_camera"], dtype=np.float64
        )
        distortion = calibration["dist_coeffs"]
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix 必须是 3×3 矩阵")
        if T_mount_camera.shape != (4, 4):
            raise ValueError("T_mount_camera 必须是 4×4 齐次矩阵")
        if len(distortion) != 5:
            raise ValueError("distortion 顺序必须是 k1, k2, p1, p2, k3")

        robot = station.get_robot_handle().get("robot")
        if camera == "head":
            get_mount_pose = robot.pin_kine.get_head_pose
            joint_key = "head_joints"
        elif camera == "left":
            get_mount_pose = robot.pin_kine.get_left_tcp_pose
            joint_key = "arm_joints"
        elif camera == "right":
            get_mount_pose = robot.pin_kine.get_right_tcp_pose
            joint_key = "arm_joints"
        else:
            raise ValueError("camera 必须是 'head'、'left' 或 'right'")

        self._station = station
        self._robot = robot
        self._camera = camera
        self._joint_key = joint_key
        self.image_size = tuple(calibration["image_size"])
        self._get_mount_pose = get_mount_pose
        self._fx = camera_matrix[0, 0]
        self._fy = camera_matrix[1, 1]
        self._cx = camera_matrix[0, 2]
        self._cy = camera_matrix[1, 2]
        self._distortion = tuple(float(value) for value in distortion)
        self._has_distortion = any(self._distortion)
        self._T_mount_camera = T_mount_camera.copy()

    def deproject(
        self,
        uv_depth_m: npt.ArrayLike,
        face_target: bool = True,
    ) -> npt.NDArray[np.float64]:
        """返回 ``WAIST_YAW_LINK`` 下的 ``xyz + xyzw``。

        输入为 ``[u, v, depth_m]``，像素坐标允许浮点数，深度单位为米。
        相机光学坐标系约定为 ``+x`` 向右、``+y`` 向下、``+z`` 向前。
        ``face_target=True`` 时安装link的 ``+X`` 朝向目标，``+Z`` 尽量朝腰部
        坐标系上方；为 ``False`` 时保留当前安装link旋转。
        调用时机械臂应已停止，关节状态与用于检测的相机帧对应。
        """
        sample = np.asarray(uv_depth_m, dtype=np.float64)
        if sample.shape != (3,):
            raise ValueError("uv_depth_m 必须是 [u, v, depth_m]")
        return self.deproject_many(
            sample[None, :],
            face_target=face_target,
        )[0]

    def deproject_many(
        self,
        uv_depth_m: npt.ArrayLike,
        face_target: bool = True,
    ) -> npt.NDArray[np.float64]:
        """批量返回 ``WAIST_YAW_LINK`` 下的 N×7 ``xyz + xyzw``。

        输入是 N×3 数组，每行为 ``[u, v, depth_m]``。整批只读取一次
        机器人状态并计算一次安装link位姿，因此同批点共享同一个动态变换。
        """
        samples = np.asarray(uv_depth_m, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError("uv_depth_m 必须是 N×3 数组：[u, v, depth_m]")
        if not np.all(np.isfinite(samples)) or np.any(samples[:, 2] <= 0.0):
            raise ValueError("u、v 必须有限，depth_m 必须是大于零的有限值")
        if samples.shape[0] == 0:
            return np.empty((0, 7), dtype=np.float64)

        x_distorted = (samples[:, 0] - self._cx) / self._fx
        y_distorted = (samples[:, 1] - self._cy) / self._fy
        x_ray, y_ray = x_distorted.copy(), y_distorted.copy()
        if self._has_distortion:
            k1, k2, p1, p2, k3 = self._distortion
            for _ in range(5):
                radius2 = x_ray * x_ray + y_ray * y_ray
                radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
                delta_x = 2.0 * p1 * x_ray * y_ray + p2 * (
                    radius2 + 2.0 * x_ray * x_ray
                )
                delta_y = p1 * (
                    radius2 + 2.0 * y_ray * y_ray
                ) + 2.0 * p2 * x_ray * y_ray
                x_ray = (x_distorted - delta_x) / radial
                y_ray = (y_distorted - delta_y) / radial

        state = self._station.get_robot_state()
        if self._camera == "head":
            mount_joints = state[self._joint_key]
        else:
            mount_joints = state[self._joint_key][self._camera]
        T_waist_yaw_mount = self._get_mount_pose(
            mount_joints
        ).get_affine_matrix_m_ndarray()
        depth_m = samples[:, 2]
        points_camera = np.column_stack(
            (
                x_ray * depth_m,
                y_ray * depth_m,
                depth_m,
                np.ones(samples.shape[0]),
            )
        )
        # WAIST_YAW_LINK <- camera mount link <- camera optical frame.
        T_waist_yaw_camera = T_waist_yaw_mount @ self._T_mount_camera
        points_waist = (points_camera @ T_waist_yaw_camera.T)[:, :3]

        if face_target:
            x_axes = points_waist - T_waist_yaw_mount[:3, 3]
            x_norms = np.linalg.norm(x_axes, axis=1)
            if np.any(x_norms == 0.0):
                raise ValueError("目标点不能与当前相机安装link原点重合")
            x_axes /= x_norms[:, None]
            z_axes = np.broadcast_to((0.0, 0.0, 1.0), x_axes.shape).copy()
            z_axes -= np.sum(z_axes * x_axes, axis=1)[:, None] * x_axes
            z_norms = np.linalg.norm(z_axes, axis=1)
            vertical = z_norms < 1e-12
            z_axes[vertical] = (1.0, 0.0, 0.0)
            z_axes[vertical] -= (
                np.sum(z_axes[vertical] * x_axes[vertical], axis=1)[:, None]
                * x_axes[vertical]
            )
            z_axes /= np.linalg.norm(z_axes, axis=1)[:, None]
            y_axes = np.cross(z_axes, x_axes)
            rotations = np.stack((x_axes, y_axes, z_axes), axis=2)
            quaternions = self._rotations_to_xyzw(rotations)
        else:
            quaternion = self._rotations_to_xyzw(
                T_waist_yaw_mount[None, :3, :3]
            )[0]
            quaternions = np.broadcast_to(
                quaternion, (samples.shape[0], 4)
            ).copy()

        return np.column_stack((points_waist, quaternions))

    def transform_to_ground(
        self,
        point_waist_m: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """把 ``WAIST_YAW_LINK`` 下的米制点转换到 ``BASE``。"""
        point = np.asarray(point_waist_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_waist_m 必须是有限的 [x, y, z]")
        pose = self._robot.mc.moveit_ctrler.get_transformation_between_frames(
            "base_footprint",
            "waist_yaw_link",
        )
        if pose is None:
            raise RuntimeError(
                "无法获取 base_footprint <- waist_yaw_link TF"
            )
        T_ground_waist = pose.get_affine_matrix_m_ndarray()
        return (T_ground_waist @ np.append(point, 1.0))[:3]

    def transform_many_to_ground(
        self,
        points_waist_m: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """批量转换 ``WAIST_YAW_LINK`` 点到 ``BASE_FOOTPRINT``，返回 N×3 米制坐标。

        整批只查询一次 ``base_footprint <- waist_yaw_link`` 变换，适用于扫描线、
        深度轮廓等稠密点，避免逐点读取TF。
        """
        points = np.asarray(points_waist_m, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("points_waist_m 必须是有限的 N×3 数组")
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        pose = self._robot.mc.moveit_ctrler.get_transformation_between_frames(
            "base_footprint",
            "waist_yaw_link",
        )
        if pose is None:
            raise RuntimeError(
                "无法获取 base_footprint <- waist_yaw_link TF"
            )
        T_ground_waist = pose.get_affine_matrix_m_ndarray()
        homogeneous = np.column_stack((points, np.ones(points.shape[0])))
        return (homogeneous @ T_ground_waist.T)[:, :3]

    @staticmethod
    def _rotations_to_xyzw(
        rotations: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """将 N×3×3 旋转矩阵转换为规范化 XYZW 四元数。"""
        quaternions = np.empty((rotations.shape[0], 4), dtype=np.float64)
        r00 = rotations[:, 0, 0]
        r11 = rotations[:, 1, 1]
        r22 = rotations[:, 2, 2]
        quaternions[:, 0] = np.copysign(
            0.5 * np.sqrt(np.maximum(0.0, 1.0 + r00 - r11 - r22)),
            rotations[:, 2, 1] - rotations[:, 1, 2],
        )
        quaternions[:, 1] = np.copysign(
            0.5 * np.sqrt(np.maximum(0.0, 1.0 - r00 + r11 - r22)),
            rotations[:, 0, 2] - rotations[:, 2, 0],
        )
        quaternions[:, 2] = np.copysign(
            0.5 * np.sqrt(np.maximum(0.0, 1.0 - r00 - r11 + r22)),
            rotations[:, 1, 0] - rotations[:, 0, 1],
        )
        quaternions[:, 3] = 0.5 * np.sqrt(
            np.maximum(0.0, 1.0 + r00 + r11 + r22)
        )
        quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
        return quaternions
