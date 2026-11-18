"""应急通道区域判断离线测试。"""

from __future__ import annotations

import unittest

from channel_region import ChannelRegion


CHANNEL_POLYGON_XY_M = (
    (0.41, 0.525),
    (2.24, 0.525),
    (2.24, 1.425),
    (4.24, 1.425),
    (4.24, 0.405),
    (3.20, 0.405),
    (3.20, -0.545),
    (0.41, -0.525),
)


class ChannelRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.region = ChannelRegion(CHANNEL_POLYGON_XY_M, outside_margin_m=0.3)

    def test_points_inside_each_part_require_clearing(self) -> None:
        for point in ((1.0, 0.0), (2.8, 1.0), (3.0, 0.0)):
            with self.subTest(point=point):
                self.assertTrue(self.region.requires_clearing(*point))

    def test_polygon_boundary_requires_clearing(self) -> None:
        for point in ((0.41, 0.0), (2.24, 1.0), (4.24, 1.425)):
            with self.subTest(point=point):
                self.assertTrue(self.region.requires_clearing(*point))

    def test_outside_point_less_than_margin_requires_clearing(self) -> None:
        self.assertTrue(self.region.requires_clearing(3.49, 0.0))
        self.assertTrue(self.region.requires_clearing(0.12, 0.0))

    def test_outside_point_at_margin_is_ignored(self) -> None:
        self.assertFalse(self.region.requires_clearing(3.50, 0.0))

    def test_outside_point_farther_than_margin_is_ignored(self) -> None:
        self.assertFalse(self.region.requires_clearing(3.60, 0.0))
        self.assertFalse(self.region.requires_clearing(0.0, 0.0))

    def test_concave_corner_uses_euclidean_distance(self) -> None:
        self.assertTrue(self.region.requires_clearing(3.40, 0.205))
        self.assertFalse(self.region.requires_clearing(3.50, 0.105))


if __name__ == "__main__":
    unittest.main()
