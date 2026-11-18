"""align_tcp_downward.py 的纯离线姿态计算测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


TASK_DIR = Path(__file__).resolve().parent


def load_module():
    numpy = types.ModuleType("numpy")
    numpy.float64 = float
    numpy.asarray = lambda value, dtype=None: list(value)
    numpy.concatenate = lambda parts: [value for part in parts for value in part]
    data_type = types.ModuleType("xrocs.common.data_type")
    data_type.Coordinate = types.SimpleNamespace(
        CoordinateSystem=types.SimpleNamespace(WAIST_YAW_LINK=object())
    )
    data_type.PoseArray = object
    data_type.Rotation = types.SimpleNamespace(
        FormType=types.SimpleNamespace(XYZW=object())
    )
    config_loader = types.ModuleType("xrocs.core.config_loader")
    config_loader.ConfigLoader = object
    station_module = types.ModuleType(
        "xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station"
    )
    station_module.TienYi2ProRos2Station = object
    path_module = types.ModuleType("xrocs.utils.path_manager")
    path_module.path_manager = object()
    stubs = {
        name: types.ModuleType(name)
        for name in (
            "xrocs",
            "xrocs.common",
            "xrocs.core",
            "xrocs.entity",
            "xrocs.entity.station",
            "xrocs.entity.station.TienYi2Pro",
            "xrocs.utils",
        )
    }
    stubs.update(
        {
            "numpy": numpy,
            "xrocs.common.data_type": data_type,
            "xrocs.core.config_loader": config_loader,
            "xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station": station_module,
            "xrocs.utils.path_manager": path_module,
        }
    )
    sys.modules.update(stubs)
    spec = importlib.util.spec_from_file_location(
        "align_tcp_downward", TASK_DIR / "align_tcp_downward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pose:
    def get_xyz_m_xyzw_ndarray(self):
        return [0.42, -0.18, -0.31, 0.2, 0.3, 0.4, 0.5]


class AlignTcpDownwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_keeps_xyz_and_sets_identity_xyzw(self):
        target = self.module.downward_target_pose(Pose())
        self.assertEqual(
            list(target),
            [0.42, -0.18, -0.31, 0.0, 0.0, 0.0, 1.0],
        )

    def test_accepts_raw_xyz_xyzw(self):
        target = self.module.downward_target_pose(
            [1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5]
        )
        self.assertEqual(list(target), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
