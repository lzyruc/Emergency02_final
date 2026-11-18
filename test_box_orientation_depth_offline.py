"""纸箱中间区域水平物理跨度检测的离线测试。"""

from __future__ import annotations

import unittest

import numpy as np

from box_orientation_depth import detect_box_orientation_from_horizontal_span


class _Deprojector:
    def __init__(self, height_m: float = 0.30) -> None:
        self.height_m = height_m
        self.deproject_calls = 0
        self.ground_calls = 0

    def deproject_many(self, samples, face_target=False):
        self.deproject_calls += 1
        samples = np.asarray(samples, dtype=np.float64)
        result = np.zeros((samples.shape[0], 7), dtype=np.float64)
        result[:, 0] = samples[:, 0] * 0.01
        result[:, 1] = samples[:, 1] * 0.01
        result[:, 2] = np.where(samples[:, 2] < 1.5, self.height_m, 0.0)
        result[:, 6] = 1.0
        return result

    def transform_many_to_ground(self, points):
        self.ground_calls += 1
        return np.asarray(points, dtype=np.float64)


def _depth_with_object(left: int, right: int) -> np.ndarray:
    depth = np.full((100, 120), 2000, dtype=np.uint16)
    depth[20:90, left:right] = 1000
    # 模拟纸箱胶带/反光造成的小深度空洞。
    depth[20:90, 58:62] = 0
    return depth


class BoxOrientationDepthTests(unittest.TestCase):
    def test_wide_above_ground_span_is_horizontal(self):
        deprojector = _Deprojector()
        result = detect_box_orientation_from_horizontal_span(
            _depth_with_object(30, 91),
            [20.0, 10.0, 100.0, 95.0],
            deprojector,
        )

        self.assertEqual(result["orientation"], "horizontal")
        self.assertGreaterEqual(result["span_m"], 0.40)
        self.assertEqual(result["valid_row_count"], 7)
        self.assertGreater(deprojector.deproject_calls, 0)
        self.assertEqual(
            deprojector.deproject_calls,
            deprojector.ground_calls,
        )

    def test_narrow_above_ground_span_is_vertical(self):
        result = detect_box_orientation_from_horizontal_span(
            _depth_with_object(45, 76),
            [20.0, 10.0, 100.0, 95.0],
            _Deprojector(),
        )

        self.assertEqual(result["orientation"], "vertical")
        self.assertLess(result["span_m"], 0.40)

    def test_points_below_15cm_are_rejected(self):
        result = detect_box_orientation_from_horizontal_span(
            _depth_with_object(30, 91),
            [20.0, 10.0, 100.0, 95.0],
            _Deprojector(height_m=0.15),
        )

        self.assertEqual(result["orientation"], "vertical")
        self.assertEqual(result["span_m"], 0.0)
        self.assertEqual(result["valid_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
