"""面向团队联调的单目标、单像素接口。"""

from pathlib import Path

import numpy as np

from .basket_detector import detect_basket
from .io_utils import load_intrinsics
from .pipeline import process_frame


_CLASS_ORDER = {
    "box": 0,
    "chair": 1,
    "basket": 2,
    "trash_can": 3,
}


class Phase1Model:
    """将完整 RGB-D 识别结果压缩为一个类别和一个候选像素。

    调用合同::

        result = model.detect({
            "rgb": rgb_bgr,
            "depth": depth_mm,
        })

    检测成功返回 ``{"class": str, "x": int, "y": int}``。垃圾桶没有
    可靠桶上沿时 x/y 为 ``None``；完全没有识别到目标时返回 ``None``。
    """

    def __init__(self, intrinsics):
        if isinstance(intrinsics, (str, Path)):
            intrinsics = load_intrinsics(intrinsics)
        if not isinstance(intrinsics, dict):
            raise TypeError("intrinsics 必须是内参 dict 或 intrinsics.json 路径")

        required = ("fx", "fy", "cx", "cy", "width", "height")
        missing = [key for key in required if key not in intrinsics]
        if missing:
            raise ValueError(f"intrinsics 缺少字段: {', '.join(missing)}")

        self.intrinsics = {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
        }
        self.last_frame_result = None

    def detect(self, img, *, target_class=None, mode="grasp"):
        """识别一帧，并返回全帧最高置信目标的类别与候选像素。

        Args:
            img: ``{"rgb": rgb_bgr, "depth": depth_mm}``。RGB 为
                ``uint8 (H,W,3)`` OpenCV BGR；depth 为与其对齐的
                ``uint16 (H,W)`` 毫米深度。
            target_class: 可选的目标类别。传入 ``"basket"`` 时只在
                文件筐结果中选择最高置信目标，不会被画面中的椅子、
                纸箱或垃圾桶替代。
            mode: ``"grasp"`` 返回安全抓取候选点；``"navigation"``
                只用于文件筐底盘接近，跳过抓取线计算并返回筐体区域中心。

        Returns:
            目标类别与候选像素 dict，或无目标时 ``None``。
        """
        rgb_bgr, depth_mm = self._validate_img(img)
        if target_class is not None and target_class not in _CLASS_ORDER:
            raise ValueError(f"不支持的目标类别: {target_class}")
        if mode not in ("grasp", "navigation"):
            raise ValueError(f"不支持的检测模式: {mode}")
        if mode == "navigation" and target_class != "basket":
            raise ValueError("navigation 模式当前只支持 basket")
        if target_class == "basket":
            frame = detect_basket(
                depth_mm,
                self.intrinsics,
                rgb_bgr,
                navigation_mode=mode == "navigation",
            )
        else:
            frame = process_frame(
                depth_mm,
                self.intrinsics,
                rgb_bgr=rgb_bgr,
            )
        self.last_frame_result = frame
        return self._select_output(
            frame,
            rgb_bgr.shape[:2],
            target_class=target_class,
            prefer_object_center=mode == "navigation",
        )

    def _validate_img(self, img):
        if isinstance(img, np.ndarray):
            raise ValueError(
                "本版精确识别需要对齐深度；请传 "
                "{'rgb': rgb_bgr, 'depth': depth_mm}"
            )
        if not isinstance(img, dict):
            raise TypeError("img 必须是包含 rgb 和 depth 的 dict")
        if "rgb" not in img or "depth" not in img:
            raise ValueError("img 必须同时包含 rgb 和 depth")

        rgb_bgr = np.asarray(img["rgb"])
        depth_mm = np.asarray(img["depth"])
        if rgb_bgr.dtype != np.uint8 or rgb_bgr.ndim != 3:
            raise ValueError("img['rgb'] 必须是 uint8 (H,W,3) BGR 图")
        if rgb_bgr.shape[2] != 3:
            raise ValueError("img['rgb'] 必须恰好有 3 个 BGR 通道")
        if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            raise ValueError("img['depth'] 必须是 uint16 (H,W) 毫米深度图")
        if rgb_bgr.shape[:2] != depth_mm.shape:
            raise ValueError("RGB 与 depth 必须同分辨率并逐像素对齐")

        height, width = depth_mm.shape
        if (
            width != self.intrinsics["width"]
            or height != self.intrinsics["height"]
        ):
            raise ValueError(
                "输入分辨率与 intrinsics 不一致："
                f"输入={width}x{height}, "
                f"内参={self.intrinsics['width']}x"
                f"{self.intrinsics['height']}"
            )
        return rgb_bgr, depth_mm

    def _select_output(
        self,
        frame,
        image_shape,
        *,
        target_class=None,
        prefer_object_center=False,
    ):
        objects = list(frame.objects or [])
        if target_class is not None:
            objects = [obj for obj in objects if obj.cls == target_class]
        if not objects:
            return None

        # 先确定最高置信物品，再在该物品上选点；避免类别与点来自不同物体。
        obj = max(
            objects,
            key=lambda item: (
                float(item.confidence),
                -_CLASS_ORDER.get(item.cls, 99),
            ),
        )
        point = (
            self._object_center_point(obj, image_shape)
            if prefer_object_center
            else self._safe_grasp_point(obj, image_shape)
        )
        if point is None and obj.cls == "trash_can":
            return {
                "class": "trash_can",
                "x": None,
                "y": None,
            }
        if point is None:
            point = self._object_center_point(obj, image_shape)
        if point is None:
            return None

        x, y = self._round_in_image(point, image_shape)
        return {
            "class": str(obj.cls),
            "x": x,
            "y": y,
        }

    def _safe_grasp_point(self, obj, image_shape):
        annotations = [
            item
            for item in (obj.grasp_annotations or [])
            if isinstance(item, dict)
        ]
        if obj.cls == "chair":
            lines = [
                item
                for item in annotations
                if item.get("type") == "line"
                and item.get("name") == "backrest_top_edge"
            ]
            return self._best_line_midpoint(lines, image_shape)

        if obj.cls == "basket":
            lines = [
                item
                for item in annotations
                if item.get("type") == "line"
                and item.get("name") == "basket_divider_free_edge"
                and item.get("grasp_safe") is True
                and not item.get("inferred", False)
            ]
            return self._best_line_midpoint(lines, image_shape)

        if obj.cls == "trash_can":
            circles = [
                item
                for item in annotations
                if item.get("type") == "circle"
                and item.get("name") == "top_rim"
                and item.get("grasp_safe") is True
                and not item.get("inferred", False)
            ]
            circles.sort(
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )
            for circle in circles:
                point = self._trash_rim_point(circle, image_shape)
                if point is not None:
                    return point
            return None

        if obj.cls == "box":
            lines = [
                item
                for item in annotations
                if item.get("type") == "line"
                and str(item.get("name", "")).startswith(
                    "visible_face_edge_"
                )
            ]
            return self._box_line_midpoint(lines, image_shape)
        return None

    def _best_line_midpoint(self, lines, image_shape):
        ranked = sorted(
            lines,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        for item in ranked:
            uv = self._line_uv(item)
            if uv is None:
                continue
            point = np.mean(uv, axis=0)
            if self._point_in_image(point, image_shape):
                return point
        return None

    def _box_line_midpoint(self, lines, image_shape):
        candidates = []
        height, width = image_shape
        for index, item in enumerate(lines):
            uv = self._line_uv(item)
            if uv is None:
                continue
            point = np.mean(uv, axis=0)
            if not self._point_in_image(point, image_shape):
                continue
            border_distance = min(
                float(point[0]),
                float(point[1]),
                float(width - 1 - point[0]),
                float(height - 1 - point[1]),
            )
            candidates.append((border_distance, -index, point))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    def _line_uv(self, annotation):
        endpoints_uv = annotation.get("endpoints_uv")
        if endpoints_uv is not None:
            uv = np.asarray(endpoints_uv, dtype=np.float64)
            if uv.shape == (2, 2) and np.all(np.isfinite(uv)):
                return uv

        endpoints = annotation.get("endpoints")
        if endpoints is None:
            return None
        points = np.asarray(endpoints, dtype=np.float64)
        if (
            points.shape != (2, 3)
            or not np.all(np.isfinite(points))
            or np.any(points[:, 2] <= 0.05)
        ):
            return None
        return self._project(points)

    def _trash_rim_point(self, circle, image_shape):
        try:
            center = np.asarray(circle["center"], dtype=np.float64)
            axis_u = np.asarray(circle["axis_u"], dtype=np.float64)
            axis_v = np.asarray(circle["axis_v"], dtype=np.float64)
            radius = float(circle["radius"])
            contour = np.asarray(circle["contour_uv"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None

        if (
            center.shape != (3,)
            or axis_u.shape != (3,)
            or axis_v.shape != (3,)
            or contour.ndim != 2
            or contour.shape[1] != 2
            or contour.shape[0] < 12
            or radius <= 0.0
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(axis_u))
            or not np.all(np.isfinite(axis_v))
            or not np.all(np.isfinite(contour))
        ):
            return None

        theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        points = (
            center[None, :]
            + radius * np.cos(theta)[:, None] * axis_u[None, :]
            + radius * np.sin(theta)[:, None] * axis_v[None, :]
        )
        valid_depth = points[:, 2] > 0.05
        if not np.any(valid_depth):
            return None
        uv = self._project(points)

        height, width = image_shape
        margin = 3.0
        valid = (
            valid_depth
            & np.all(np.isfinite(uv), axis=1)
            & (uv[:, 0] >= margin)
            & (uv[:, 0] <= width - 1 - margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 1] <= height - 1 - margin)
        )
        if not np.any(valid):
            return None

        valid_indices = np.flatnonzero(valid)
        nearest_index = valid_indices[
            int(np.argmin(points[valid_indices, 2]))
        ]
        projected = uv[nearest_index]

        contour_in_image = np.array(
            [
                self._point_in_image(point, image_shape)
                for point in contour
            ],
            dtype=bool,
        )
        if not np.any(contour_in_image):
            return None
        contour_indices = np.flatnonzero(contour_in_image)
        distances = np.linalg.norm(
            contour[contour_indices] - projected[None, :],
            axis=1,
        )
        nearest_contour = contour_indices[int(np.argmin(distances))]
        if float(np.min(distances)) > 3.0:
            return None
        return contour[nearest_contour]

    def _object_center_point(self, obj, image_shape):
        bbox = (obj.debug or {}).get("pixel_bbox")
        if bbox is not None:
            values = np.asarray(bbox, dtype=np.float64)
            if (
                values.shape == (4,)
                and np.all(np.isfinite(values))
                and values[2] > 0
                and values[3] > 0
            ):
                point = np.array(
                    [
                        values[0] + 0.5 * values[2],
                        values[1] + 0.5 * values[3],
                    ],
                    dtype=np.float64,
                )
                if self._point_in_image(point, image_shape):
                    return point

        position = np.asarray(obj.position_cam, dtype=np.float64)
        if (
            position.shape == (3,)
            and np.all(np.isfinite(position))
            and position[2] > 0.05
        ):
            point = self._project(position[None, :])[0]
            if self._point_in_image(point, image_shape):
                return point
        return None

    def _project(self, points):
        points = np.asarray(points, dtype=np.float64)
        z = points[:, 2]
        u = (
            points[:, 0] * self.intrinsics["fx"] / z
            + self.intrinsics["cx"]
        )
        v = (
            points[:, 1] * self.intrinsics["fy"] / z
            + self.intrinsics["cy"]
        )
        return np.stack([u, v], axis=1)

    @staticmethod
    def _point_in_image(point, image_shape):
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            return False
        height, width = image_shape
        return bool(
            0.0 <= point[0] <= width - 1
            and 0.0 <= point[1] <= height - 1
        )

    @staticmethod
    def _round_in_image(point, image_shape):
        height, width = image_shape
        x = int(np.clip(np.rint(point[0]), 0, width - 1))
        y = int(np.clip(np.rint(point[1]), 0, height - 1))
        return x, y
