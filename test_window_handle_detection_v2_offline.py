"""批量运行窗户把手 V2 检测器，不读取相机或发送控制命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import settings
from agent.perception.window_handle_detection_v2 import detect_handle_center_v2
from wrist_self_mask import gripper_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="包含多个时间戳子目录的 window_handle_recognize 输出目录",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="V2 标注和 JSON 输出目录",
    )
    args = parser.parse_args()

    camera_matrix = np.asarray(
        settings.LEFT_CAMERA["camera_matrix"],
        dtype=np.float64,
    )
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    sample_dirs = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_dir()
        and (path / "rgb_raw.jpg").is_file()
        and (path / "depth_raw_mm.png").is_file()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    for sample_dir in sample_dirs:
        rgb = cv2.imread(str(sample_dir / "rgb_raw.jpg"), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(
            str(sample_dir / "depth_raw_mm.png"),
            cv2.IMREAD_UNCHANGED,
        )
        static_mask = gripper_mask(depth_mm.shape)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_m[static_mask] = 0.0
        sample_output = args.output_dir / sample_dir.name
        sample_output.mkdir(parents=True, exist_ok=True)
        try:
            detection = detect_handle_center_v2(
                depth_m,
                fx=fx,
                fy=fy,
            )
        except RuntimeError as exc:
            failures += 1
            (sample_output / "failure.json").write_text(
                json.dumps(
                    {"success": False, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            continue

        successes += 1
        u, v, target_depth_m = detection.uv_depth_m
        ys, xs = np.nonzero(detection.mask)
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ]
        marked = rgb.copy()
        contours, _ = cv2.findContours(
            detection.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(marked, contours, -1, (0, 255, 0), 2)
        cv2.rectangle(
            marked,
            (bbox[0], bbox[1]),
            (bbox[2] - 1, bbox[3] - 1),
            (255, 0, 0),
            2,
        )
        cv2.circle(
            marked,
            (int(round(u)), int(round(v))),
            8,
            (0, 0, 255),
            -1,
        )
        cv2.imwrite(str(sample_output / "v2_target.jpg"), marked)
        cv2.imwrite(str(sample_output / "v2_mask.png"), detection.mask)
        (sample_output / "result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "pixel_uv": [float(u), float(v)],
                    "depth_m": float(target_depth_m),
                    "bbox_xyxy": bbox,
                    "main_length_m": detection.main_length_m,
                    "minor_length_m": detection.minor_length_m,
                    "aspect_ratio": detection.aspect_ratio,
                    "prominence_m": detection.prominence_m,
                    "score": detection.score,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "samples": len(sample_dirs),
        "successes": successes,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
