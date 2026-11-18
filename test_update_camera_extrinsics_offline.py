#!/usr/bin/env python3
"""离线验证相机标定安装、参数保留、合并同步和备份。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from update_camera_extrinsics import install_calibration


ROOT = Path(__file__).resolve().parent


class UpdateCameraExtrinsicsTest(unittest.TestCase):
    def test_install_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            emergency_root = temporary_root / "Emergency02"
            source = temporary_root / "Documents"
            source.mkdir(parents=True)

            target_relatives = (
                Path("config/head_calibration.json"),
                Path("config/left_camera_calibration.json"),
                Path("config/right_camera_calibration.json"),
                Path("agent/perception/config/camera_calibration.json"),
            )
            for relative_path in target_relatives:
                destination = emergency_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative_path, destination)

            head = json.loads(
                (ROOT / target_relatives[0]).read_text(encoding="utf-8")
            )
            left = json.loads(
                (ROOT / target_relatives[1]).read_text(encoding="utf-8")
            )
            right = json.loads(
                (ROOT / target_relatives[2]).read_text(encoding="utf-8")
            )
            head["extrinsic"]["matrix"][0][3] = 0.123
            left["extrinsics_cam_to_left_tcp_link_matrix"][0][3] = 0.234
            right["extrinsics_cam_to_right_tcp_link_matrix"][0][3] = 0.345
            left["date"] = "2099-01-02"
            right["date"] = "2099-01-03"
            (source / "extrinsics_head.json").write_text(
                json.dumps(head), encoding="utf-8"
            )
            (source / "extrinsics_D405_left.json").write_text(
                json.dumps(left), encoding="utf-8"
            )
            (source / "extrinsics_D405_right.json").write_text(
                json.dumps(right), encoding="utf-8"
            )

            original_head = deepcopy(
                json.loads(
                    (emergency_root / target_relatives[0]).read_text(
                        encoding="utf-8"
                    )
                )
            )
            original_right = json.loads(
                (emergency_root / target_relatives[2]).read_text(encoding="utf-8")
            )
            _, backup_directory = install_calibration(source, emergency_root)

            saved_head = json.loads(
                (emergency_root / target_relatives[0]).read_text(encoding="utf-8")
            )
            saved_right = json.loads(
                (emergency_root / target_relatives[2]).read_text(encoding="utf-8")
            )
            combined = json.loads(
                (emergency_root / target_relatives[3]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_head["extrinsic"]["matrix"][0][3], 0.123)
            self.assertEqual(saved_head["kinematics"], original_head["kinematics"])
            self.assertEqual(
                saved_right["grasp_offset_right_tcp_m"],
                original_right["grasp_offset_right_tcp_m"],
            )
            self.assertEqual(
                combined["cameras"]["left"]["T_mount_camera"][0][3], 0.234
            )
            self.assertEqual(
                combined["cameras"]["right"]["T_mount_camera"][0][3], 0.345
            )
            self.assertIsNotNone(backup_directory)
            self.assertTrue(
                (backup_directory / target_relatives[3]).is_file()
            )


if __name__ == "__main__":
    unittest.main()
