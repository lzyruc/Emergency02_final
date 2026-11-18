"""静态张开夹爪mask的离线尺寸与二值性测试。"""

from __future__ import annotations

import unittest

import numpy as np

from wrist_self_mask import gripper_mask


class WristSelfMaskTests(unittest.TestCase):
    def test_native_camera_mask_is_boolean_and_covers_known_gripper_point(self) -> None:
        mask = gripper_mask((720, 1280))

        self.assertEqual(mask.shape, (720, 1280))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[509, 870])
        self.assertFalse(mask[110, 606])

    def test_resizes_with_nearest_neighbor(self) -> None:
        mask = gripper_mask((360, 640))

        self.assertEqual(mask.shape, (360, 640))
        self.assertEqual(mask.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
