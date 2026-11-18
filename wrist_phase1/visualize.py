from __future__ import annotations

import cv2
import numpy as np

from .utils import CLASS_COLORS


def draw_result(rgb, detailed):
    image = rgb.copy()
    primary = detailed.get("primary")
    if primary is None:
        cv2.putText(
            image,
            "NO TARGET",
            (18, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return image
    cls = primary["class"]
    color = CLASS_COLORS.get(cls, (255, 255, 255))
    x1, y1, x2, y2 = primary["bbox"]
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
    label = (
        f"{cls} conf={primary['confidence']:.3f} "
        f"sel={primary['selection_score']:.3f}"
    )
    cv2.putText(
        image,
        label,
        (max(4, x1), max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    geometry = primary.get("grasp_geometry")
    if geometry:
        if geometry["type"] == "line":
            p0, p1 = [tuple(map(int, point)) for point in geometry["endpoints_uv"]]
            cv2.line(image, p0, p1, (0, 0, 255), 5, cv2.LINE_AA)
        elif geometry["type"] == "ellipse":
            ellipse = geometry["ellipse"]
            center = tuple(int(round(v)) for v in ellipse["center"])
            axes = tuple(max(1, int(round(v / 2))) for v in ellipse["axes"])
            cv2.ellipse(
                image,
                center,
                axes,
                float(ellipse["angle_deg"]),
                0,
                360,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
    result = detailed.get("result")
    if result and result.get("x") is not None:
        point = (int(result["x"]), int(result["y"]))
        cv2.drawMarker(
            image,
            point,
            (0, 255, 0),
            cv2.MARKER_CROSS,
            24,
            4,
            cv2.LINE_AA,
        )
        cv2.circle(image, point, 7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def contact_sheet(images, labels, columns=4, tile_width=360):
    if not images:
        return np.zeros((180, tile_width, 3), np.uint8)
    h0, w0 = images[0].shape[:2]
    tile_height = int(round(tile_width * h0 / w0))
    caption = 34
    rows = int(np.ceil(len(images) / columns))
    canvas = np.full(
        (rows * (tile_height + caption), columns * tile_width, 3),
        242,
        np.uint8,
    )
    for index, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, columns)
        x, y = col * tile_width, row * (tile_height + caption)
        tile = cv2.resize(image, (tile_width, tile_height), cv2.INTER_AREA)
        canvas[y : y + tile_height, x : x + tile_width] = tile
        cv2.putText(
            canvas,
            str(label),
            (x + 5, y + tile_height + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas

