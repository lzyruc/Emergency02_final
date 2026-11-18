#!/usr/bin/env python3
"""只读头部相机，实时显示检测结果并打印目标相对机器人本体的坐标。

本脚本不创建Navigation，不调用MoveIt、底盘移动或station.execute_action()，
不会发送运动命令。图像仅在内存中缩放并压缩为JPEG，不保存原图或推理结果。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import agent.perception.deprojector as deprojector_module
from camera_io import read_head_camera
from yolo import ObstacleYolo
from xrocs.core.config_loader import ConfigLoader
from xrocs.entity.station.TienYi2Pro.tienyi2_pro_ros2_station import (
    TienYi2ProRos2Station,
)
from xrocs.utils.path_manager import path_manager


TARGET_COLORS = {
    "chair": (255, 180, 40),
    "filebox": (255, 90, 40),
    "bin": (60, 210, 110),
    "box": (40, 170, 255),
}
CALIBRATION_PATH = (
    Path(deprojector_module.__file__).resolve().parent
    / "config/camera_calibration.json"
)
PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Emergency Vision</title>
  <style>
    :root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #090d12; color: #e8eef5; }
    main { width: min(100%, 920px); margin: auto; padding: 18px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
    h1 { margin: 0; font-size: clamp(18px, 4vw, 26px); font-weight: 650; letter-spacing: .02em; }
    #state { color: #91a1b2; font-size: 13px; text-align: right; }
    .viewer { position: relative; overflow: hidden; border: 1px solid #26313d; border-radius: 10px; background: #111821; box-shadow: 0 18px 60px #0008; }
    img { display: block; width: 100%; min-height: 220px; object-fit: contain; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 12px; color: #aebac7; font-size: 13px; }
    .dot { display: inline-block; width: 9px; height: 9px; margin-right: 6px; border-radius: 2px; }
    footer { margin-top: 10px; color: #637282; font-size: 12px; }
  </style>
</head>
<body>
  <main>
    <header><h1>Head Camera / YOLO</h1><div id="state">connecting...</div></header>
    <div class="viewer"><img src="/stream.mjpg" alt="头部相机实时检测画面"></div>
    <div class="legend">
      <span><i class="dot" style="background:#28b4ff"></i>chair</span>
      <span><i class="dot" style="background:#285aff"></i>filebox</span>
      <span><i class="dot" style="background:#6ed23c"></i>bin</span>
      <span><i class="dot" style="background:#ffaa28"></i>box</span>
    </div>
    <footer>低带宽预览 · 原始图像不会发送或保存</footer>
  </main>
  <script>
    const state = document.querySelector('#state');
    async function refresh() {
      try {
        const r = await fetch('/status.json', {cache: 'no-store'});
        const s = await r.json();
        const base = `${s.fps.toFixed(1)} FPS · ${s.inference_ms.toFixed(0)} ms · ${s.kbps.toFixed(0)} kbit/s · ${s.objects} objects`;
        const depth = s.nearest_depth_m == null ? '--' : `${s.nearest_depth_m.toFixed(2)} m`;
        state.textContent = s.depth_enabled ? `${base} · nearest ${depth}` : base;
      } catch (_) { state.textContent = 'reconnecting...'; }
    }
    refresh(); setInterval(refresh, 1000);
  </script>
</body>
</html>""".encode("utf-8")


class StreamState:
    """在单次推理生产者与多个 MJPEG 客户端之间共享最新压缩帧。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg = b""
        self.status = (
            b'{"fps":0,"inference_ms":0,"kbps":0,"objects":0,'
            b'"depth_enabled":false,"nearest_depth_m":null}'
        )
        self.version = 0

    def publish(self, jpeg: bytes, status: dict[str, float | int | None]) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.status = json.dumps(status, separators=(",", ":")).encode()
            self.version += 1
            self.condition.notify_all()


class ViewerHandler(BaseHTTPRequestHandler):
    """提供单页、轻量状态接口和共享 MJPEG 流。"""

    server_version = "EmergencyVision/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path == "/status.json":
            state = self.server.stream_state
            with state.condition:
                payload = state.status
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/stream.mjpg":
            self._stream()
            return
        self.send_error(404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        state = self.server.stream_state
        version = -1
        try:
            while True:
                with state.condition:
                    state.condition.wait_for(lambda: state.version != version)
                    jpeg = state.jpeg
                    version = state.version
                if not jpeg:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: Any) -> None:
        return


def center_depth_m(
    depth_mm: np.ndarray,
    bbox_xyxy: list[float],
) -> float | None:
    """返回检测框中心 30% 区域的有效深度中位数，单位为米。"""
    x_min, y_min, x_max, y_max = bbox_xyxy
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    half_width = 0.15 * (x_max - x_min)
    half_height = 0.15 * (y_max - y_min)
    left = max(0, int(center_x - half_width))
    right = min(depth_mm.shape[1], int(center_x + half_width) + 1)
    top = max(0, int(center_y - half_height))
    bottom = min(depth_mm.shape[0], int(center_y + half_height) + 1)
    values = depth_mm[top:bottom, left:right]
    valid = values[(values > 0) & np.isfinite(values)]
    return None if valid.size == 0 else float(np.median(valid)) / 1000.0


def print_base_coordinates(
    detections: list[dict[str, Any]],
    depth_mm: np.ndarray,
    deprojector: deprojector_module.Deprojector,
) -> list[float | None]:
    """打印每个检测的base_footprint XYZ，并返回对应中心深度（米）。

    检测像素使用原图坐标，深度取框中心30%区域中位数。Deprojector先把
    ``camera optical`` 点转换到 ``waist_yaw_link``，再转换到
    ``base_footprint``。坐标单位为米，+x向前、+y向左、+z向上。
    """
    depths_m = [
        center_depth_m(depth_mm, detection["bbox_xyxy"])
        for detection in detections
    ]
    valid_indices = [
        index for index, depth_m in enumerate(depths_m)
        if depth_m is not None
    ]
    if not valid_indices:
        for detection in detections:
            print(
                f"识别到 {detection['type']}：confidence="
                f"{detection['confidence']:.3f}，中心区域无有效深度，"
                "base_footprint XYZ无法计算",
                flush=True,
            )
        return depths_m

    samples = [
        [
            detections[index]["center_uv"][0],
            detections[index]["center_uv"][1],
            depths_m[index],
        ]
        for index in valid_indices
    ]
    poses_waist = deprojector.deproject_many(samples, face_target=False)
    points_base = deprojector.transform_many_to_ground(poses_waist[:, :3])
    base_by_index = dict(zip(valid_indices, points_base, strict=True))
    for index, detection in enumerate(detections):
        if index not in base_by_index:
            print(
                f"识别到 {detection['type']}：confidence="
                f"{detection['confidence']:.3f}，中心区域无有效深度，"
                "base_footprint XYZ无法计算",
                flush=True,
            )
            continue
        point_base = base_by_index[index]
        print(
            f"识别到 {detection['type']}：confidence="
            f"{detection['confidence']:.3f}，depth={depths_m[index]:.3f} m，"
            f"base_footprint XYZ=({point_base[0]:.3f}, "
            f"{point_base[1]:.3f}, {point_base[2]:.3f}) m",
            flush=True,
        )
    return depths_m


def annotate_and_compress(
    image: Any,
    detections: list[dict[str, Any]],
    *,
    width: int,
    quality: int,
) -> bytes:
    """缩放 BGR 图并绘制检测框，返回用于网络传输的 JPEG。"""
    scale = min(1.0, width / image.shape[1])
    frame = cv2.resize(
        image,
        (int(image.shape[1] * scale), int(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    for detection in detections:
        x_min, y_min, x_max, y_max = (
            int(round(value * scale)) for value in detection["bbox_xyxy"]
        )
        obstacle_type = detection["type"]
        color = TARGET_COLORS[obstacle_type]
        label = f"{obstacle_type} {detection['confidence']:.2f}"
        if "depth_m" in detection:
            depth_m = detection["depth_m"]
            depth_text = "--m" if depth_m is None else f"{depth_m:.2f}m"
            label = f"{label} {depth_text}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.rectangle(
            frame,
            (x_min, max(0, y_min - 22)),
            (x_min + 9 * len(label) + 8, y_min),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x_min + 4, max(15, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (10, 14, 18),
            1,
            cv2.LINE_AA,
        )
    succeeded, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not succeeded:
        raise RuntimeError("JPEG 编码失败")
    return encoded.tobytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="头部相机 YOLO 低带宽实时页面")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--quality", type=int, default=55)
    parser.add_argument(
        "--depth",
        action="store_true",
        help="在前端页面显示检测框中心深度；终端坐标计算始终使用深度",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = ObstacleYolo()
    config = ConfigLoader(path_manager.config_toml_path).get_config()
    station = TienYi2ProRos2Station(
        robot_dict=config.get("robot", {}),
        camera_dict=config.get("camera", {}),
        hand_dict=config.get("hand", {}),
        force_sensor_dict={},
        control_rate_hz=config.get("basic", {}).get("control_rate_hz", 100),
    )
    try:
        deprojector = deprojector_module.Deprojector(
            station,
            CALIBRATION_PATH,
            "head",
        )
        state = StreamState()
        server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
        server.stream_state = state
        server_thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
        )
        server_thread.start()
        print(f"实时页面：http://<机器人IP>:{args.port}", flush=True)
        print(
            "终端坐标系=base_footprint：+x向前、+y向左、+z向上，单位米",
            flush=True,
        )

        interval_s = 1.0 / args.fps
        last_publish = time.monotonic()
        try:
            while True:
                loop_started = time.monotonic()
                color, depth_mm = read_head_camera(station)
                if color is None or depth_mm is None:
                    continue
                inference_started = time.monotonic()
                detections = detector.detect(color)
                inference_ms = (
                    time.monotonic() - inference_started
                ) * 1000.0
                depths_m = print_base_coordinates(
                    detections,
                    depth_mm,
                    deprojector,
                )
                valid_depths = [
                    depth_m for depth_m in depths_m
                    if depth_m is not None
                ]
                display_detections = (
                    [
                        {**detection, "depth_m": depth_m}
                        for detection, depth_m in zip(
                            detections,
                            depths_m,
                            strict=True,
                        )
                    ]
                    if args.depth
                    else detections
                )
                jpeg = annotate_and_compress(
                    color,
                    display_detections,
                    width=args.width,
                    quality=args.quality,
                )
                now = time.monotonic()
                elapsed = now - last_publish
                state.publish(
                    jpeg,
                    {
                        "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
                        "inference_ms": inference_ms,
                        "kbps": len(jpeg) * 8.0 / 1000.0
                        / max(elapsed, interval_s),
                        "objects": len(detections),
                        "depth_enabled": args.depth,
                        "nearest_depth_m": (
                            min(valid_depths)
                            if args.depth and valid_depths
                            else None
                        ),
                    },
                )
                last_publish = now
                time.sleep(
                    max(
                        0.0,
                        interval_s - (time.monotonic() - loop_started),
                    )
                )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()
    finally:
        station.shutdown()


if __name__ == "__main__":
    main()
