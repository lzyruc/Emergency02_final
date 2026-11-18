#!/usr/bin/env python3
"""在已录制左腕RGB-D上离线比较窗把手v2/v3，不连接机器人。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import settings
from agent.perception.window_handle_detection_v2 import detect_handle_center_v2
from agent.perception.window_handle_detection_v3 import (
    detect_handle_depth_in_bbox_v3,
)
from wrist_self_mask import gripper_mask
from yolo.window_handle_yolo import WindowHandleYolo


def detection_record(detection: Any) -> dict[str, Any]:
    u, v, depth_m = detection.uv_depth_m
    ys, xs = np.nonzero(detection.mask)
    return {
        "pixel_uv": [float(u), float(v)],
        "depth_m": float(depth_m),
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ],
        "mask_pixels": int(np.count_nonzero(detection.mask)),
        "main_length_m": float(detection.main_length_m),
    }


def draw_result(
    image: np.ndarray,
    algorithm: str,
    result: dict[str, Any],
) -> np.ndarray:
    output = image.copy()
    if result["success"]:
        x0, y0, x1, y1 = result["bbox_xyxy"]
        u, v = result["pixel_uv"]
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        cv2.circle(output, (round(u), round(v)), 10, (0, 0, 255), -1)
        label = f"{algorithm} OK z={result['depth_m']:.3f}m"
        if "yolo_confidence" in result:
            label += f" conf={result['yolo_confidence']:.3f}"
        color = (0, 255, 0)
    else:
        label = f"{algorithm} FAIL {result['error']}"
        color = (0, 0, 255)
    cv2.putText(
        output,
        label[:90],
        (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample_dirs = sorted(
        path.parent for path in args.dataset.glob("*/rgb_raw.png")
    )
    if not sample_dirs:
        raise RuntimeError(f"没有找到样本：{args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)
    detector = WindowHandleYolo()
    camera_matrix = np.asarray(
        settings.LEFT_CAMERA["camera_matrix"],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    thumbnails: list[np.ndarray] = []
    for index, sample_dir in enumerate(sample_dirs, 1):
        bgr = cv2.imread(str(sample_dir / "rgb_raw.png"), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(
            str(sample_dir / "depth_raw_mm.png"), cv2.IMREAD_UNCHANGED
        )
        if bgr is None or depth_mm is None:
            raise RuntimeError(f"样本读取失败：{sample_dir}")
        detection_depth_mm = depth_mm.copy()
        detection_depth_mm[gripper_mask(depth_mm.shape)] = 0
        depth_m = detection_depth_mm.astype(np.float64) / 1000.0
        sample: dict[str, Any] = {"sample": sample_dir.name}

        start = time.perf_counter()
        try:
            v2_detection = detect_handle_center_v2(
                depth_m,
                fx=float(camera_matrix[0, 0]),
                fy=float(camera_matrix[1, 1]),
            )
            v2 = {"success": True, **detection_record(v2_detection)}
            v2.update(
                {
                    "minor_length_m": float(v2_detection.minor_length_m),
                    "aspect_ratio": float(v2_detection.aspect_ratio),
                    "prominence_m": float(v2_detection.prominence_m),
                    "score": float(v2_detection.score),
                }
            )
        except Exception as error:
            v2 = {"success": False, "error": str(error)}
        v2["elapsed_s"] = time.perf_counter() - start

        start = time.perf_counter()
        yolo_detections = detector.detect(bgr)
        v3_detection = None
        selected_yolo = None
        v3_errors: list[str] = []
        for yolo_detection in yolo_detections:
            try:
                v3_detection = detect_handle_depth_in_bbox_v3(
                    depth_m,
                    yolo_detection["bbox_xyxy"],
                    fy=float(camera_matrix[1, 1]),
                )
            except RuntimeError as error:
                v3_errors.append(str(error))
                continue
            selected_yolo = yolo_detection
            break
        if v3_detection is not None and selected_yolo is not None:
            v3 = {"success": True, **detection_record(v3_detection)}
            v3.update(
                {
                    "bbox_xyxy": [
                        int(round(value))
                        for value in selected_yolo["bbox_xyxy"]
                    ],
                    "yolo_confidence": float(selected_yolo["confidence"]),
                    "yolo_detection_count": len(yolo_detections),
                    "depth_cluster_center_m": float(
                        v3_detection.cluster_center_m
                    ),
                    "depth_cluster_pixels": int(
                        v3_detection.cluster_pixel_count
                    ),
                }
            )
        else:
            error = (
                "YOLO未检出窗把手"
                if not yolo_detections
                else "; ".join(v3_errors)
            )
            v3 = {
                "success": False,
                "error": error,
                "yolo_detection_count": len(yolo_detections),
                "yolo_detections": yolo_detections,
            }
        v3["elapsed_s"] = time.perf_counter() - start
        sample.update({"v2": v2, "v3": v3})
        rows.append(sample)

        v2_image = draw_result(bgr, "v2", v2)
        v3_image = draw_result(bgr, "v3", v3)
        pair = np.concatenate([v2_image, v3_image], axis=1)
        cv2.putText(
            pair,
            f"{index:02d} {sample_dir.name}",
            (12, pair.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(args.output / f"{sample_dir.name}.jpg"), pair)
        thumbnails.append(cv2.resize(pair, (640, 180)))
        print(
            f"[{index:02d}/{len(sample_dirs)}] {sample_dir.name} "
            f"v2={v2['success']} v3={v3['success']}",
            flush=True,
        )

    for page_index in range(0, len(thumbnails), 20):
        page_items = thumbnails[page_index : page_index + 20]
        while len(page_items) < 20:
            page_items.append(np.zeros_like(thumbnails[0]))
        page = np.concatenate(
            [
                np.concatenate(page_items[row : row + 2], axis=1)
                for row in range(0, 20, 2)
            ],
            axis=0,
        )
        cv2.imwrite(
            str(args.output / f"contact_{page_index // 20 + 1}.jpg"),
            page,
        )

    summary = {
        "sample_count": len(rows),
        "v2_success_count": sum(row["v2"]["success"] for row in rows),
        "v3_success_count": sum(row["v3"]["success"] for row in rows),
        "v2_total_elapsed_s": sum(row["v2"]["elapsed_s"] for row in rows),
        "v3_total_elapsed_s": sum(row["v3"]["elapsed_s"] for row in rows),
    }
    summary["v2_success_rate"] = summary["v2_success_count"] / len(rows)
    summary["v3_success_rate"] = summary["v3_success_count"] / len(rows)
    summary["v2_mean_elapsed_s"] = summary["v2_total_elapsed_s"] / len(rows)
    summary["v3_mean_elapsed_s"] = summary["v3_total_elapsed_s"] / len(rows)
    (args.output / "results.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
