"""使用 RealSense 深度相机实时测试商品识别模块。"""

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from product_yolo import detect_products


WINDOW_NAME = "Product YOLO + Depth"
FRAME_TIMEOUT_MS = 5000
MAX_STARTUP_TIMEOUTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时显示商品识别中心点和深度")
    parser.add_argument("--serial", default="", help="RealSense 相机序列号")
    parser.add_argument(
        "--products",
        nargs="*",
        default=None,
        help="需要识别的商品 ID；不传入时在程序启动后输入",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def read_product_ids(values: list[str] | None) -> list[str]:
    """读取并检查商品 ID，多个 ID 可以使用空格或逗号分隔。"""
    if values is None:
        text = input("请输入商品 ID（多个 ID 用空格或逗号分隔）：").strip()
        values = [value for value in re.split(r"[\s,，]+", text) if value]

    if not values:
        raise ValueError("至少需要输入一个商品 ID")

    product_ids = []
    for value in values:
        if not value.isdigit() or not 1 <= int(value) <= 107:
            raise ValueError(f"商品 ID 必须是 1 到 107 的整数：{value}")
        product_ids.append(str(int(value)))
    return list(dict.fromkeys(product_ids))


def draw_results(
    image: np.ndarray,
    results: dict[str, list[dict[str, object]]],
    fps: float,
) -> np.ndarray:
    """在 BGR 画面上标记识别中心点、商品 ID 和深度。"""
    output = image.copy()
    detected_count = 0

    for product_id, items in results.items():
        for item in items:
            detected_count += 1
            center_x, center_y = item["center"]
            depth = item["average_depth"]

            cv2.drawMarker(
                output,
                (center_x, center_y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                24,
                2,
                cv2.LINE_AA,
            )
            text = f"ID {product_id}  {depth:.3f} m"
            text_y = max(22, center_y - 14)
            cv2.putText(
                output,
                text,
                (max(0, center_x - 55), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                text,
                (max(0, center_x - 55), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    status = f"FPS {fps:.1f}  Detected {detected_count}  Q/ESC quit  S save"
    cv2.putText(
        output,
        status,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        status,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def wait_for_frames(pipeline: rs.pipeline) -> rs.composite_frame:
    """等待一组相机帧，超时时给出更容易排查的中文错误。"""
    for attempt in range(1, MAX_STARTUP_TIMEOUTS + 1):
        try:
            return pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        except RuntimeError as exc:
            if attempt == MAX_STARTUP_TIMEOUTS:
                raise RuntimeError(
                    "RealSense 已连接但没有输出画面，请重新插拔相机 USB 后再运行"
                ) from exc
            print(f"等待相机画面超时，正在重试（{attempt}/{MAX_STARTUP_TIMEOUTS}）...")

    raise RuntimeError("无法读取 RealSense 画面")


def main() -> int:
    args = parse_args()
    product_ids = read_product_ids(args.products)
    print(f"准备识别商品 ID：{', '.join(product_ids)}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    device = profile.get_device()
    camera_name = device.get_info(rs.camera_info.name)
    camera_serial = device.get_info(rs.camera_info.serial_number)
    depth_scale = device.first_depth_sensor().get_depth_scale()
    align_to_color = rs.align(rs.stream.color)

    print(f"Camera: {camera_name} serial={camera_serial}")
    print(f"Depth scale: {depth_scale} meter/unit")
    print("Press Q or ESC to quit. Press S to save the current preview.")

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

    frame_count = 0
    fps = 0.0
    previous_time = time.perf_counter()

    try:
        # 丢弃启动阶段曝光尚未稳定的帧。
        for _ in range(5):
            wait_for_frames(pipeline)

        while True:
            frames = align_to_color.process(wait_for_frames(pipeline))
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr_image = np.asanyarray(color_frame.get_data()).copy()
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            depth_image_m = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32)
                * depth_scale
            )

            results = detect_products(
                rgb_image,
                depth_image_m,
                product_ids,
            )

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else fps * 0.9 + current_fps * 0.1

            preview = draw_results(bgr_image, results, fps)
            frame_count += 1

            if args.no_display:
                detected = {
                    product_id: items
                    for product_id, items in results.items()
                    if items
                }
                print(f"Frame {frame_count}: {detected}")
            else:
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    output_path = Path(__file__).with_name(
                        f"product_preview_{int(time.time())}.jpg"
                    )
                    cv2.imwrite(str(output_path), preview)
                    print(f"Saved: {output_path}")

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n测试已停止")
    except Exception as exc:
        print(f"\n启动失败：{exc}")
        if sys.stdin.isatty():
            input("按回车键关闭窗口...")
