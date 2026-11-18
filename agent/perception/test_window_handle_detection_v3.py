"""窗把手v3深度后处理离线测试。"""

from __future__ import annotations

import unittest

import numpy as np

from agent.perception.window_handle_detection_v3 import (
    detect_handle_depth_in_bbox_v3,
)


class WindowHandleDetectionV3Tests(unittest.TestCase):
    def test_selects_nearest_stable_cluster_with_depth_holes(self) -> None:
        depth_m = np.zeros((720, 1280), dtype=np.float32)
        depth_m[200:500, 500:700] = 0.20
        depth_m[230:470, 585:615] = 0.15
        depth_m[240:470:3, 585:615] = 0.0

        result = detect_handle_depth_in_bbox_v3(
            depth_m,
            [500, 200, 700, 500],
            fy=658.0,
        )

        np.testing.assert_allclose(
            result.uv_depth_m,
            [599.5, 379.4, 0.15],
            atol=1e-5,
        )
        self.assertGreater(result.cluster_pixel_count, 100)
        self.assertGreater(np.count_nonzero(result.mask), 100)

    def test_rejects_bbox_without_depth(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "有效近场深度像素不足"):
            detect_handle_depth_in_bbox_v3(
                np.zeros((100, 100), dtype=np.float32),
                [10, 10, 90, 90],
                fy=658.0,
            )


if __name__ == "__main__":
    unittest.main()
