from __future__ import annotations

import copy

import cv2
import numpy as np

from .fast_pipeline import run_fast_pipeline
from .utils import depth_meters, load_camera, safe_point


class WristRGBDModel:
    """Independent single-frame wrist RGB-D detector."""

    def __init__(self, camera):
        self.camera = load_camera(camera)
        self.last_result = None

    def _validate(self, img):
        if not isinstance(img, dict):
            raise TypeError("img must be {'rgb': BGR uint8, 'depth': uint16}")
        if "rgb" not in img:
            raise ValueError("img missing rgb")
        rgb = np.asarray(img["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be uint8 HxWx3 BGR")
        depth = img.get("depth")
        if depth is None:
            depth = np.zeros(rgb.shape[:2], np.uint16)
        depth = np.asarray(depth)
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("depth must be HxW and aligned with rgb")
        if rgb.shape[1] != self.camera["width"] or rgb.shape[0] != self.camera["height"]:
            raise ValueError(
                f"input resolution {rgb.shape[1]}x{rgb.shape[0]} != "
                f"camera {self.camera['width']}x{self.camera['height']}"
            )
        return rgb, depth

    def detect_detailed(self, img, target_class=None):
        """Run routed inference, or force one class supplied by head YOLO."""
        rgb, depth = self._validate(img)
        z, _ = depth_meters(depth, self.camera)
        detections, ignore, pipeline_debug = run_fast_pipeline(
            rgb,
            z,
            self.camera,
            img.get("ignore_mask"),
            target_class=target_class,
        )
        detections.sort(
            key=lambda item: float(item.get("selection_score", 0.0)),
            reverse=True,
        )
        primary = detections[0] if detections else None
        result = self._simple_result(primary, rgb.shape)
        public = {
            "result": result,
            "primary": self._public_detection(primary),
            "detections": [
                self._public_detection(item) for item in detections
            ],
            "debug": {
                "ignore_fraction": round(
                    float(np.mean(ignore > 0)), 6
                ),
                "depth_scale_m": self.camera["depth_scale_m"],
                "scene_geometry": None,
                "fast_pipeline": pipeline_debug,
            },
        }
        self.last_result = public
        return public

    def detect(self, img):
        result = self.detect_detailed(img)["result"]
        if result is None:
            return {"class": None, "x": None, "y": None}
        # Keep the team-facing contract exact.  point_source and geometry stay
        # available through detect_detailed(), not in the production payload.
        return {
            "class": result["class"],
            "x": result["x"],
            "y": result["y"],
        }

    @staticmethod
    def _simple_result(primary, image_shape):
        if primary is None:
            return None
        geometry = primary.get("grasp_geometry")
        point = None
        source = None
        if geometry:
            if geometry["type"] == "line":
                endpoints = np.asarray(geometry.get("endpoints_uv"), np.float32)
                if endpoints.shape == (2, 2):
                    point = np.mean(endpoints, axis=0)
                    source = "grasp_line"
            elif geometry["type"] == "ellipse":
                point = geometry.get("grasp_point_uv")
                source = "rim"
        if point is None and primary["class"] != "trash_can":
            x1, y1, x2, y2 = primary["bbox"]
            point = [(x1 + x2 - 1) / 2, (y1 + y2 - 1) / 2]
            source = "object_center"
        if point is None:
            return {"class": primary["class"], "x": None, "y": None}
        point = safe_point(point, image_shape)
        return {
            "class": primary["class"],
            "x": int(point[0]),
            "y": int(point[1]),
            "point_source": source,
        }

    @staticmethod
    def _public_detection(item):
        if item is None:
            return None
        out = copy.deepcopy(item)
        out.pop("mask", None)
        return out


def load_pair(rgb_path, depth_path=None):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if depth_path is None:
        depth = np.zeros(rgb.shape[:2], np.uint16)
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(depth_path)
    return {"rgb": rgb, "depth": depth}
