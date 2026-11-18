"""文件框/垃圾桶抓取姿态测试的离线计算测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from bin_filebox_grasp import (
    candidate_motion_info,
    select_bin_white_rim,
    select_candidate,
    select_filebox_directional_depth_ridge,
    select_filebox_resized_depth_ridge,
    select_vertical_depth_ridge,
    wrist_yolo_grasp_candidates,
)


class _TcpPose:
    def get_affine_matrix_m_ndarray(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [0.1, -0.1, 0.2]
        return matrix


class _Kinematics:
    def get_left_tcp_pose(self, joints):
        return _TcpPose()

    def get_right_tcp_pose(self, joints):
        return _TcpPose()


class _Station:
    def get_robot_state(self):
        return {
            "arm_joints": {"left": [0.0] * 7, "right": [0.0] * 7}
        }

    def get_robot_handle(self):
        robot = type("Robot", (), {"pin_kine": _Kinematics()})()
        return {"robot": robot}


class _CandidateStation:
    def get_robot_state(self):
        return {
            "arm_pose": {
                "left": np.zeros(7, dtype=np.float64),
                "right": np.zeros(7, dtype=np.float64),
            }
        }


class _Detector:
    def detect(self, color):
        return [
            {
                "type": "bin",
                "confidence": 0.9,
                "bbox_xyxy": [40.0, 40.0, 360.0, 340.0],
            }
        ]


class _EmptyDetector:
    def detect(self, color):
        return []


class _Deprojector:
    def deproject(self, uv_depth_m, face_target):
        return np.asarray(
            [uv_depth_m[0] / 1000.0, uv_depth_m[1] / 1000.0,
             uv_depth_m[2], 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )


class BinFileboxGraspTests(unittest.TestCase):
    def test_warm_lit_white_bin_rim_is_accepted(self):
        color = np.full((400, 400, 3), 80, dtype=np.uint8)
        depth = np.full((400, 400), 800, dtype=np.uint16)
        warm_rim_bgr = cv2.cvtColor(
            np.uint8([[[20, 120, 240]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0].tolist()
        cv2.ellipse(
            color,
            (200, 190),
            (140, 110),
            0,
            0,
            360,
            warm_rim_bgr,
            12,
        )
        cv2.ellipse(depth, (200, 190), (140, 110), 0, 0, 360, 300, 12)

        def camera_reader(station, camera_config):
            return color, depth

        with tempfile.TemporaryDirectory() as directory:
            candidates = wrist_yolo_grasp_candidates(
                _CandidateStation(),
                "bin",
                _Detector(),
                {"left": _Deprojector(), "right": _Deprojector()},
                Path(directory),
                camera_reader,
            )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(
            all(
                candidate["point_source"] == "bin_white_rim_ellipse"
                for candidate in candidates
            )
        )

    def test_ellipse_rule_rejects_light_green_rim(self):
        color = np.full((400, 400, 3), 80, dtype=np.uint8)
        depth = np.full((400, 400), 800, dtype=np.uint16)
        cv2.ellipse(
            color,
            (200, 190),
            (140, 110),
            0,
            0,
            360,
            (174, 213, 151),
            12,
        )
        cv2.ellipse(depth, (200, 190), (140, 110), 0, 0, 360, 300, 12)

        def camera_reader(station, camera_config):
            return color, depth

        with tempfile.TemporaryDirectory() as directory:
            candidates = wrist_yolo_grasp_candidates(
                _CandidateStation(),
                "bin",
                _Detector(),
                {"left": _Deprojector(), "right": _Deprojector()},
                Path(directory),
                camera_reader,
            )

        self.assertEqual(candidates, [])

    def test_yolo_bbox_is_rgbd_rule_roi_and_pixel_returns_to_full_image(self):
        color = np.full((400, 400, 3), 80, dtype=np.uint8)
        depth = np.full((400, 400), 800, dtype=np.uint16)
        cv2.ellipse(color, (200, 190), (140, 110), 0, 0, 360, (240, 240, 240), 12)
        cv2.ellipse(depth, (200, 190), (140, 110), 0, 0, 360, 300, 12)

        def camera_reader(station, camera_config):
            return color, depth

        with tempfile.TemporaryDirectory() as directory:
            candidates = wrist_yolo_grasp_candidates(
                _CandidateStation(),
                "bin",
                _Detector(),
                {"left": _Deprojector(), "right": _Deprojector()},
                Path(directory),
                camera_reader,
            )

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            u, v = candidate["pixel_uv"]
            self.assertGreaterEqual(u, 40)
            self.assertLess(u, 360)
            self.assertGreaterEqual(v, 40)
            self.assertLess(v, 340)
            self.assertEqual(candidate["bbox_xyxy"], [40.0, 40.0, 360.0, 340.0])
            self.assertEqual(candidate["roi_source"], "yolo_bbox")
            self.assertEqual(candidate["point_source"], "bin_white_rim_ellipse")

    def test_without_yolo_bbox_runs_rgbd_on_full_frame(self):
        color = np.full((400, 400, 3), 80, dtype=np.uint8)
        depth = np.full((400, 400), 800, dtype=np.uint16)
        cv2.ellipse(
            color,
            (200, 170),
            (140, 110),
            0,
            0,
            360,
            (240, 240, 240),
            12,
        )
        cv2.ellipse(depth, (200, 170), (140, 110), 0, 0, 360, 300, 12)

        def camera_reader(station, camera_config):
            return color, depth

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            candidates = wrist_yolo_grasp_candidates(
                _CandidateStation(),
                "bin",
                _EmptyDetector(),
                {"left": _Deprojector(), "right": _Deprojector()},
                output_dir,
                camera_reader,
            )
            fallback_mask = cv2.imread(
                str(output_dir / "right_fallback_mask.png"),
                cv2.IMREAD_GRAYSCALE,
            )

        self.assertEqual(len(candidates), 2)
        np.testing.assert_array_equal(fallback_mask, 0)
        for candidate in candidates:
            self.assertEqual(candidate["confidence"], 0.0)
            self.assertEqual(candidate["bbox_xyxy"], [0.0, 0.0, 400.0, 400.0])
            self.assertEqual(
                candidate["roi_source"],
                "full_frame_fallback",
            )

    def test_bin_rim_uses_near_stable_depth_cluster(self):
        color = np.full((220, 220, 3), 80, dtype=np.uint8)
        depth = np.full((220, 220), 560, dtype=np.uint16)
        cv2.ellipse(
            color,
            (110, 105),
            (72, 62),
            0,
            0,
            360,
            (240, 240, 240),
            11,
        )
        cv2.ellipse(depth, (110, 105), (72, 62), 0, 0, 360, 280, 5)

        result, _, _, _ = select_bin_white_rim(
            color,
            depth,
            [20, 20, 200, 195],
            "right",
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["depth_m"], 0.28, places=3)
        self.assertGreaterEqual(result["depth_cluster_pixels"], 8)

    def test_motion_info_uses_selected_tcp_frame(self):
        result = candidate_motion_info(
            _Station(),
            {
                "arm": "right",
                "target_pose_waist_xyz_xyzw": np.asarray(
                    [0.2, -0.3, 0.5, 0.0, 0.0, 0.0, 1.0]
                ),
            },
        )
        np.testing.assert_allclose(
            result["point_tcp_xyz_m"],
            [0.1, -0.2, 0.3],
        )
        np.testing.assert_allclose(
            result["delta_waist_xyz_m"],
            [0.1, -0.2, 0.3],
        )

    def test_both_targets_select_nearest_tcp_candidate(self):
        candidates = [
            {"point_source": "object_center", "tcp_distance_m": 0.1},
            {"point_source": "grasp_line", "tcp_distance_m": 0.3},
            {"point_source": "grasp_line", "tcp_distance_m": 0.2},
        ]
        self.assertIs(select_candidate("filebox", candidates), candidates[0])
        self.assertIs(select_candidate("bin", candidates), candidates[0])

    def test_vertical_depth_ridge_selects_center_between_two_jumps(self):
        color = np.zeros((120, 120, 3), dtype=np.uint8)
        color[15:105, 54:66] = [255, 0, 0]
        depth = np.full((120, 120), 900, dtype=np.uint16)
        depth[15:105, 54:66] = 500
        result, mask, score, material = select_vertical_depth_ridge(
            color,
            depth,
            [10, 10, 110, 110],
            "filebox",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["point_source"], "vertical_depth_ridge")
        self.assertGreaterEqual(result["pixel_uv"][0], 54)
        self.assertLess(result["pixel_uv"][0], 66)
        self.assertAlmostEqual(result["depth_m"], 0.5)
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertGreater(np.count_nonzero(score), 0)
        self.assertGreater(np.count_nonzero(material), 0)

    def test_horizontal_depth_strip_is_not_vertical_grasp_ridge(self):
        color = np.zeros((120, 120, 3), dtype=np.uint8)
        color[54:66, 15:105] = [255, 0, 0]
        depth = np.full((120, 120), 900, dtype=np.uint16)
        depth[54:66, 15:105] = 500
        result, _, _, _ = select_vertical_depth_ridge(
            color,
            depth,
            [10, 10, 110, 110],
            "filebox",
        )
        self.assertIsNone(result)

    def test_filebox_accepts_slanted_non_horizontal_depth_ridge(self):
        color = np.zeros((150, 150, 3), dtype=np.uint8)
        depth = np.full((150, 150), 900, dtype=np.uint16)
        cv2.line(color, (35, 120), (115, 30), (255, 0, 0), 13)
        cv2.line(depth, (35, 120), (115, 30), 500, 13)
        result, mask, score, _ = select_filebox_directional_depth_ridge(
            color,
            depth,
            [15, 15, 135, 135],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["point_source"], "directional_depth_ridge")
        self.assertGreater(result["line_angle_deg"], 10.0)
        self.assertLess(result["line_angle_deg"], 170.0)

    def test_filebox_half_resolution_result_returns_full_image_pixel(self):
        color = np.zeros((300, 300, 3), dtype=np.uint8)
        depth = np.full((300, 300), 900, dtype=np.uint16)
        cv2.line(color, (70, 240), (230, 60), (255, 0, 0), 26)
        cv2.line(depth, (70, 240), (230, 60), 500, 26)

        result, mask, score, material = select_filebox_resized_depth_ridge(
            color,
            depth,
            [30, 30, 270, 270],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["rule_image_size"], [150, 150])
        self.assertAlmostEqual(result["depth_m"], 0.5)
        self.assertEqual(mask.shape, depth.shape)
        self.assertEqual(score.shape, depth.shape)
        self.assertEqual(material.shape, depth.shape)
        self.assertAlmostEqual(result["depth_m"], 0.5)
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertGreater(np.count_nonzero(score), 0)

    def test_filebox_rejects_horizontal_depth_ridge(self):
        color = np.zeros((150, 150, 3), dtype=np.uint8)
        depth = np.full((150, 150), 900, dtype=np.uint16)
        cv2.line(color, (25, 75), (125, 75), (255, 0, 0), 13)
        cv2.line(depth, (25, 75), (125, 75), 500, 13)
        result, _, _, _ = select_filebox_directional_depth_ridge(
            color,
            depth,
            [15, 15, 135, 135],
        )
        self.assertIsNone(result)

    def test_bin_selects_white_ellipse_rim_without_vertical_constraint(self):
        color = np.full((140, 140, 3), 80, dtype=np.uint8)
        depth = np.full((140, 140), 800, dtype=np.uint16)
        cv2.ellipse(color, (70, 65), (48, 42), 0, 0, 360, (240, 240, 240), 9)
        cv2.ellipse(depth, (70, 65), (48, 42), 0, 0, 360, 300, 9)
        result, candidate, score, material = select_bin_white_rim(
            color,
            depth,
            [18, 18, 122, 115],
            "right",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["point_source"], "bin_white_rim_arc")
        tangent_angle = result["tangent_angle_deg"]
        horizontal_distance = min(tangent_angle, 180.0 - tangent_angle)
        self.assertGreaterEqual(horizontal_distance, 30.0)
        self.assertGreater(result["pixel_uv"][0], 70)
        self.assertAlmostEqual(result["depth_m"], 0.3)
        self.assertGreater(np.count_nonzero(candidate), 0)
        self.assertGreater(np.count_nonzero(score), 0)
        self.assertGreater(np.count_nonzero(material), 0)

    def test_bin_white_only_rule_rejects_light_green_ellipse_rim(self):
        color = np.full((140, 140, 3), 80, dtype=np.uint8)
        depth = np.full((140, 140), 800, dtype=np.uint16)
        cv2.ellipse(
            color,
            (70, 65),
            (48, 42),
            0,
            0,
            360,
            (174, 213, 151),
            9,
        )
        cv2.ellipse(depth, (70, 65), (48, 42), 0, 0, 360, 300, 9)

        result, candidate, _, material = select_bin_white_rim(
            color,
            depth,
            [18, 18, 122, 115],
            "right",
        )

        self.assertIsNone(result)
        self.assertEqual(np.count_nonzero(candidate), 0)
        self.assertEqual(np.count_nonzero(material), 0)


if __name__ == "__main__":
    unittest.main()
