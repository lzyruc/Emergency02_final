"""应急通道地面区域判断。

输入点和多边形顶点均位于 Navigation logical 坐标系，单位为米：x 向前、
y 向左。多边形按顶点顺序隐式闭合。障碍物中心投影位于通道内、边界上，或
位于通道外但距边界小于外侧缓冲距离时需要清除；位于通道外至少该距离时忽略。

最小调用：

    region = ChannelRegion(polygon_xy, outside_margin_m=0.3)
    if region.requires_clearing(center_x_m, center_y_m):
        ...
"""

from __future__ import annotations

from collections.abc import Sequence


class ChannelRegion:
    """预计算通道边线，并判断障碍物中心投影是否需要清除。"""

    def __init__(
        self,
        polygon_xy: Sequence[Sequence[float]],
        *,
        outside_margin_m: float = 0.3,
    ) -> None:
        vertices = tuple((float(point[0]), float(point[1])) for point in polygon_xy)
        self.outside_margin_squared_m2 = float(outside_margin_m) ** 2
        self.edges = tuple(
            (
                start_x,
                start_y,
                end_x - start_x,
                end_y - start_y,
                (end_x - start_x) ** 2 + (end_y - start_y) ** 2,
            )
            for (start_x, start_y), (end_x, end_y) in zip(
                vertices,
                vertices[1:] + vertices[:1],
            )
        )

    def requires_clearing(self, x_m: float, y_m: float) -> bool:
        """障碍物中心投影在通道内或外侧缓冲带内时返回 ``True``。

        多边形边界属于通道。通道外到最近边界的距离恰好等于外侧缓冲距离时
        返回 ``False``，即无需清除。
        """
        inside = False
        minimum_distance_squared = float("inf")
        for start_x, start_y, delta_x, delta_y, length_squared in self.edges:
            offset_x, offset_y = x_m - start_x, y_m - start_y
            projection = max(
                0.0,
                min(1.0, (offset_x * delta_x + offset_y * delta_y) / length_squared),
            )
            nearest_x = start_x + projection * delta_x
            nearest_y = start_y + projection * delta_y
            distance_squared = (x_m - nearest_x) ** 2 + (y_m - nearest_y) ** 2
            if distance_squared < minimum_distance_squared:
                minimum_distance_squared = distance_squared
            if (start_y > y_m) != (start_y + delta_y > y_m) and x_m < (
                start_x + (y_m - start_y) * delta_x / delta_y
            ):
                inside = not inside

        # 留出平方距离的浮点舍入余量，确保理论上恰好位于缓冲边界的点被忽略。
        return inside or minimum_distance_squared < (
            self.outside_margin_squared_m2 - 1e-12
        )
