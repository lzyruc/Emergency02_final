"""应急场景固定矩形障碍物的底盘扫掠碰撞检查与弯腰接近规划。

最小调用：

    avoid = AvoidCollision(navigation, "emergency.json")
    collision = avoid.checkCollision(1.0, -0.2, 30.0)
    motions = avoid.planOperationApproach(1.2, -0.1)

``dx/dy/dtheta`` 与 ``NavigationGraph.moveBy`` 一致：dx 前、dy 左，单位米；
dtheta 逆时针，单位度。旧 ``checkCollision`` 保留圆形底盘直线检查；
``checkBentCollision`` 和 ``planApproach`` 使用前0.65 m、后0.30 m、左右各0.35 m的
弯腰矩形。规划只返回动作，不下发底盘。
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

__all__ = ["AvoidCollision"]


class AvoidCollision:
    """使用 NavigationGraph 逻辑位姿检查机器人坐标系相对运动。"""

    def __init__(
        self,
        navigation: Any,
        scene_path: str | Path,
        *,
        collision_radius_m: float = 0.40,
        bent_front_m: float = 0.65,
        bent_rear_m: float = 0.30,
        bent_half_width_m: float = 0.35,
        path_resolution_m: float = 0.05,
        sweep_translation_step_m: float = 0.02,
        sweep_rotation_step_deg: float = 2.0,
    ) -> None:
        path = Path(scene_path)
        if not path.is_absolute() and not path.is_file():
            path = Path(__file__).resolve().parent / "data/logs" / path
        document = json.loads(path.read_text(encoding="utf-8"))
        obstacles = document.get("obstacles")
        if obstacles is None:
            obstacles = document.get("scene_editor", {}).get("obstacles")
        if obstacles is None:
            companion = path.with_name(f"{path.stem}.navigation.scene.json")
            document = json.loads(companion.read_text(encoding="utf-8"))
            obstacles = document["obstacles"]

        self.navigation = navigation
        self.collision_radius_m = float(collision_radius_m)
        self.bent_front_m = float(bent_front_m)
        self.bent_rear_m = float(bent_rear_m)
        self.bent_half_width_m = float(bent_half_width_m)
        self.path_resolution_m = float(path_resolution_m)
        self.sweep_translation_step_m = float(sweep_translation_step_m)
        self.sweep_rotation_step_deg = float(sweep_rotation_step_deg)
        self.obstacles = tuple(
            {
                "name": obstacle["name"],
                "min_x": min(float(obstacle["x1"]), float(obstacle["x2"])),
                "max_x": max(float(obstacle["x1"]), float(obstacle["x2"])),
                "min_y": min(float(obstacle["y1"]), float(obstacle["y2"])),
                "max_y": max(float(obstacle["y1"]), float(obstacle["y2"])),
            }
            for obstacle in obstacles
        )

    def checkCollision(self, dx: float, dy: float, dtheta: float) -> bool:
        """检查相对动作的直线平移扫掠范围，碰撞返回 ``True``。

        当前位姿来自 ``NavigationGraph.getPose()`` 的 logical pose。内部使用
        ``logical <- base_footprint`` 4×4齐次矩阵转换终点。场景运动顺序为先平移
        后旋转；底盘按半径 ``collision_radius_m`` 的圆处理，所以旋转不改变占用。
        """
        motion = (float(dx), float(dy), float(dtheta))
        if not all(math.isfinite(value) for value in motion):
            raise ValueError("dx、dy、dtheta 必须是有限数值")
        logical_pose, _ = self.navigation.getPose()
        if logical_pose is None:
            raise RuntimeError("NavigationGraph 当前没有可用的逻辑位姿")

        theta_rad = math.radians(float(logical_pose["theta_deg"]))
        cosine, sine = math.cos(theta_rad), math.sin(theta_rad)
        # logical frame <- base_footprint，长度单位为米。
        T_logical_base = (
            (cosine, -sine, 0.0, float(logical_pose["x_m"])),
            (sine, cosine, 0.0, float(logical_pose["y_m"])),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        endpoint_base = (motion[0], motion[1], 0.0, 1.0)
        end_x = sum(
            T_logical_base[0][index] * endpoint_base[index]
            for index in range(4)
        )
        end_y = sum(
            T_logical_base[1][index] * endpoint_base[index]
            for index in range(4)
        )
        start_x = float(logical_pose["x_m"])
        start_y = float(logical_pose["y_m"])
        radius = self.collision_radius_m

        return any(
            self._segment_intersects_rectangle(
                start_x,
                start_y,
                end_x,
                end_y,
                obstacle["min_x"] - radius,
                obstacle["max_x"] + radius,
                obstacle["min_y"] - radius,
                obstacle["max_y"] + radius,
            )
            for obstacle in self.obstacles
        )

    def checkBentCollision(self, dx: float, dy: float, dtheta: float) -> bool:
        """检查弯腰矩形执行“先平移、后旋转”的完整扫掠区。

        输入为当前 ``base_footprint`` 坐标系下的米/度，不下发运动。
        """
        motion = (float(dx), float(dy), float(dtheta))
        if not all(math.isfinite(value) for value in motion):
            raise ValueError("dx、dy、dtheta 必须是有限数值")
        logical_pose, _ = self.navigation.getPose()
        if logical_pose is None:
            raise RuntimeError("NavigationGraph 当前没有可用的逻辑位姿")
        start_x = float(logical_pose["x_m"])
        start_y = float(logical_pose["y_m"])
        start_yaw = math.radians(float(logical_pose["theta_deg"]))
        end_x = start_x + math.cos(start_yaw) * motion[0] - math.sin(start_yaw) * motion[1]
        end_y = start_y + math.sin(start_yaw) * motion[0] + math.cos(start_yaw) * motion[1]
        return self._bent_sweep_collides(
            start_x,
            start_y,
            start_yaw,
            end_x,
            end_y,
            start_yaw + math.radians(motion[2]),
        )

    def planApproach(
        self,
        dx: float,
        dy: float,
        dtheta: float,
        *,
        max_expansions: int = 50000,
    ) -> list[tuple[float, float, float]] | None:
        """规划到目标底盘位姿的弯腰安全动作列表。

        目标 ``dx/dy/dtheta`` 在调用时的 ``base_footprint`` 下表示。
        由于底盘可全向平移，规划期间保持当前朝向，用八邻域 A* 绕过
        固定障碍物，最后再原地旋转。返回的每项可依次传给
        ``NavigationGraph.moveBy(*motion)``；无路返回 ``None``。
        """
        requested = (float(dx), float(dy), float(dtheta))
        if not all(math.isfinite(value) for value in requested):
            raise ValueError("dx、dy、dtheta 必须是有限数值")
        logical_pose, _ = self.navigation.getPose()
        if logical_pose is None:
            raise RuntimeError("NavigationGraph 当前没有可用的逻辑位姿")
        start_x = float(logical_pose["x_m"])
        start_y = float(logical_pose["y_m"])
        start_yaw = math.radians(float(logical_pose["theta_deg"]))
        goal_x = start_x + math.cos(start_yaw) * requested[0] - math.sin(start_yaw) * requested[1]
        goal_y = start_y + math.sin(start_yaw) * requested[0] + math.cos(start_yaw) * requested[1]
        goal_yaw = start_yaw + math.radians(requested[2])

        if self._bent_pose_collides(start_x, start_y, start_yaw):
            return None
        if self._bent_pose_collides(goal_x, goal_y, start_yaw):
            return None
        if self._bent_sweep_collides(
            goal_x, goal_y, start_yaw, goal_x, goal_y, goal_yaw
        ):
            return None
        if not self._bent_sweep_collides(
            start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw
        ):
            return [requested] if any(requested) else []

        resolution = self.path_resolution_m
        goal_cell = (
            round((goal_x - start_x) / resolution),
            round((goal_y - start_y) / resolution),
        )
        start_cell = (0, 0)
        obstacle_min_x = min((item["min_x"] for item in self.obstacles), default=start_x)
        obstacle_max_x = max((item["max_x"] for item in self.obstacles), default=goal_x)
        obstacle_min_y = min((item["min_y"] for item in self.obstacles), default=start_y)
        obstacle_max_y = max((item["max_y"] for item in self.obstacles), default=goal_y)
        footprint_span = max(
            self.bent_front_m,
            self.bent_rear_m,
            self.bent_half_width_m,
        )
        margin = footprint_span + 2.0 * resolution
        min_x = min(start_x, goal_x, obstacle_min_x) - margin
        max_x = max(start_x, goal_x, obstacle_max_x) + margin
        min_y = min(start_y, goal_y, obstacle_min_y) - margin
        max_y = max(start_y, goal_y, obstacle_max_y) + margin

        def cell_pose(cell: tuple[int, int]) -> tuple[float, float]:
            return (
                start_x + cell[0] * resolution,
                start_y + cell[1] * resolution,
            )

        def in_bounds(cell: tuple[int, int]) -> bool:
            x_m, y_m = cell_pose(cell)
            return min_x <= x_m <= max_x and min_y <= y_m <= max_y

        def heuristic(cell: tuple[int, int]) -> float:
            return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

        frontier: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start_cell), 0, start_cell)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start_cell: 0.0}
        sequence = 0
        expansions = 0
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
        reached: tuple[int, int] | None = None
        while frontier and expansions < int(max_expansions):
            _, _, current = heapq.heappop(frontier)
            expansions += 1
            current_x, current_y = cell_pose(current)
            if current == goal_cell:
                if not self._bent_sweep_collides(
                    current_x,
                    current_y,
                    start_yaw,
                    goal_x,
                    goal_y,
                    start_yaw,
                ):
                    reached = current
                    break
            for step_x, step_y in neighbors:
                neighbor = (current[0] + step_x, current[1] + step_y)
                if not in_bounds(neighbor):
                    continue
                next_x, next_y = cell_pose(neighbor)
                if self._bent_sweep_collides(
                    current_x,
                    current_y,
                    start_yaw,
                    next_x,
                    next_y,
                    start_yaw,
                ):
                    continue
                new_cost = cost[current] + math.hypot(step_x, step_y)
                if new_cost >= cost.get(neighbor, math.inf):
                    continue
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                sequence += 1
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic(neighbor), sequence, neighbor),
                )
        if reached is None:
            return None

        cells = [reached]
        while cells[-1] != start_cell:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        points = [cell_pose(cell) for cell in cells]
        points[-1] = (goal_x, goal_y)
        simplified = [points[0]]
        source_index = 0
        while source_index < len(points) - 1:
            target_index = len(points) - 1
            while target_index > source_index + 1:
                if not self._bent_sweep_collides(
                    points[source_index][0],
                    points[source_index][1],
                    start_yaw,
                    points[target_index][0],
                    points[target_index][1],
                    start_yaw,
                ):
                    break
                target_index -= 1
            simplified.append(points[target_index])
            source_index = target_index

        cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
        motions: list[tuple[float, float, float]] = []
        for first, second in zip(simplified, simplified[1:]):
            world_dx = second[0] - first[0]
            world_dy = second[1] - first[1]
            motions.append(
                (
                    cosine * world_dx + sine * world_dy,
                    -sine * world_dx + cosine * world_dy,
                    0.0,
                )
            )
        if abs(requested[2]) > 1e-9:
            motions.append((0.0, 0.0, requested[2]))
        return motions

    def planOperationApproach(
        self,
        target_x: float,
        target_y: float,
        *,
        operation_distance_m: float = 0.40,
        max_operation_distance_m: float = 0.70,
        operation_distance_step_m: float = 0.10,
        approach_yaw_deg: float | None = None,
        max_expansions: int = 50000,
    ) -> list[tuple[float, float, float]] | None:
        """让目标操作点落在机器人最终正前方，返回弯腰安全动作。

        ``target_x/target_y`` 是目标在当前 ``base_footprint`` 下的坐标（米）。
        默认面向目标；``operation_distance_m`` 是首选停靠距离，默认0.40 m，
        与当前 bin/filebox 接近参数一致。若最终占用或路径碰撞，按
        ``operation_distance_step_m`` 逐步退让，直到 ``max_operation_distance_m``；
        默认依次尝试0.40/0.50/0.60/0.70 m。停靠距离与弯腰矩形向前
        占用的0.65 m是两个独立概念。
        """
        target = (float(target_x), float(target_y))
        if not all(math.isfinite(value) for value in target):
            raise ValueError("target_x、target_y 必须是有限数值")
        distance = float(operation_distance_m)
        maximum_distance = float(max_operation_distance_m)
        distance_step = float(operation_distance_step_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("operation_distance_m 必须是非负有限数值")
        if not math.isfinite(maximum_distance) or maximum_distance < distance:
            raise ValueError(
                "max_operation_distance_m 必须是不小于首选距离的有限数值"
            )
        if not math.isfinite(distance_step) or distance_step <= 0.0:
            raise ValueError("operation_distance_step_m 必须是正有限数值")
        yaw_deg = (
            math.degrees(math.atan2(target[1], target[0]))
            if approach_yaw_deg is None
            else float(approach_yaw_deg)
        )
        if not math.isfinite(yaw_deg):
            raise ValueError("approach_yaw_deg 必须是有限数值")
        yaw_rad = math.radians(yaw_deg)
        candidate_count = int(
            math.floor((maximum_distance - distance) / distance_step + 1e-9)
        )
        distances = [
            distance + index * distance_step
            for index in range(candidate_count + 1)
        ]
        if distances[-1] < maximum_distance - 1e-9:
            distances.append(maximum_distance)
        for candidate_distance in distances:
            motions = self.planApproach(
                target[0] - candidate_distance * math.cos(yaw_rad),
                target[1] - candidate_distance * math.sin(yaw_rad),
                yaw_deg,
                max_expansions=max_expansions,
            )
            if motions is not None:
                return motions
        return None

    def _bent_sweep_collides(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        end_x: float,
        end_y: float,
        end_yaw: float,
    ) -> bool:
        """按Navigation的先平移后旋转顺序离散检查矩形扫掠。"""
        distance = math.hypot(end_x - start_x, end_y - start_y)
        translation_steps = max(
            1,
            math.ceil(distance / self.sweep_translation_step_m),
        )
        for index in range(translation_steps + 1):
            ratio = index / translation_steps
            x_m = start_x + (end_x - start_x) * ratio
            y_m = start_y + (end_y - start_y) * ratio
            if self._bent_pose_collides(x_m, y_m, start_yaw):
                return True
        yaw_delta = math.atan2(
            math.sin(end_yaw - start_yaw),
            math.cos(end_yaw - start_yaw),
        )
        rotation_steps = max(
            1,
            math.ceil(
                abs(math.degrees(yaw_delta)) / self.sweep_rotation_step_deg
            ),
        )
        for index in range(1, rotation_steps + 1):
            yaw = start_yaw + yaw_delta * index / rotation_steps
            if self._bent_pose_collides(end_x, end_y, yaw):
                return True
        return False

    def _bent_pose_collides(self, x_m: float, y_m: float, yaw_rad: float) -> bool:
        """用SAT检查一个有向弯腰矩形与所有固定AABB。"""
        cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
        local_corners = (
            (self.bent_front_m, self.bent_half_width_m),
            (self.bent_front_m, -self.bent_half_width_m),
            (-self.bent_rear_m, -self.bent_half_width_m),
            (-self.bent_rear_m, self.bent_half_width_m),
        )
        robot = tuple(
            (
                x_m + cosine * local_x - sine * local_y,
                y_m + sine * local_x + cosine * local_y,
            )
            for local_x, local_y in local_corners
        )
        axes = ((cosine, sine), (-sine, cosine), (1.0, 0.0), (0.0, 1.0))
        for obstacle in self.obstacles:
            rectangle = (
                (obstacle["min_x"], obstacle["min_y"]),
                (obstacle["min_x"], obstacle["max_y"]),
                (obstacle["max_x"], obstacle["max_y"]),
                (obstacle["max_x"], obstacle["min_y"]),
            )
            separated = False
            for axis_x, axis_y in axes:
                robot_projection = tuple(
                    point_x * axis_x + point_y * axis_y
                    for point_x, point_y in robot
                )
                obstacle_projection = tuple(
                    point_x * axis_x + point_y * axis_y
                    for point_x, point_y in rectangle
                )
                if (
                    max(robot_projection) < min(obstacle_projection)
                    or max(obstacle_projection) < min(robot_projection)
                ):
                    separated = True
                    break
            if not separated:
                return True
        return False

    @staticmethod
    def _segment_intersects_rectangle(
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> bool:
        """用线段参数区间裁剪判断线段是否接触闭合轴对齐矩形。"""
        delta_x, delta_y = end_x - start_x, end_y - start_y
        lower, upper = 0.0, 1.0
        for start, delta, minimum, maximum in (
            (start_x, delta_x, min_x, max_x),
            (start_y, delta_y, min_y, max_y),
        ):
            if delta == 0.0:
                if start < minimum or start > maximum:
                    return False
                continue
            first = (minimum - start) / delta
            second = (maximum - start) / delta
            lower = max(lower, min(first, second))
            upper = min(upper, max(first, second))
            if lower > upper:
                return False
        return True
