# Shop 与 emergency 多步导航规则

## 共同机制

1. graphGUI scene 提供逻辑节点和人工确认的无向安全连接；转换器将每条连接生成两个有向
   traversal。最短路只能使用已有边，不能因为几何距离近而跨越无边区域。
2. 实机启动时必须确认机器人所在的初始语义节点，并读取当前 SLAM map 与实时 pose。该对应
   关系建立逻辑坐标（`+x` 前、`+y` 左、角度逆时针）到 map 坐标的二维刚体变换。
3. 当前实测流程使用 `move_by`。`shortestPath()` 决定安全走廊，`executionPath()` 决定是否
   允许合并同向共线中间点；单段平移最大 3 m，超长平移按实时 pose 分段反馈。
4. 每段失败后进入 `FAILED`，不假装到达。`2001 ACTION_FAILED` 只在实测几何误差满足明确
   容差时允许记为软成功；取消、超时、通信失败不适用。
5. `moveBy(x, y, theta)` 是节点外的小范围机体系调整：x 前、y 左、theta 逆时针度。第一次
   调整前保存当前实测 pose；无参数 `recoverState()` 回到该 pose，指定节点的
   `recoverState(name)` 则先走 SLAM `move_to` 回到节点附近，再用受限 `move_by` 校正。

## Shop

- 固定验收序列定义在 `auto_test_navigation_graph.py`，每个目标先按图求最短路。
- 默认允许安全的同向共线合并，但仍以原图路径选择走廊，且不超过 3 m。
- `recognize_left`、`recognize_right` 到达前先转向再平移，避免靠近桌面后原地旋转。
- 同排 `leftN` 沿 x 移动时保持逻辑 270° 后横移；同排 `rightN` 保持逻辑 90° 后横移。
  `centerN`、跨排和沿 y 的边不套用。
- `start_point`、`hub_left`、`hub_center`、`hub_right` 是脚本选择的校正点。每次到达后做两轮
  实时 `get_pose + move_by`：先修正 map 平移，再重新采样并修正最短 yaw 误差。
- 平移命令阈值 20 mm，旋转命令阈值 1°；底盘报告 2001 后仅在位置误差不超过 5 cm或 yaw
  误差不超过 1°时软成功。hub 的 yaw 目标可取当前最近的逻辑正交方向。

## Emergency

- 固定业务顺序为：

  `startpoint -> h270_task2_start -> h270_window1_action ->`
  `h270_window2_action -> h270_task2_check -> h180_task3_check ->`
  `h180_return_origin`。
- 禁止路径压缩，每个安全中间节点和边都保留；例如窗口和桌面附近不能用几何直线替代图边。
- 校正点是 `h270_task2_start`、`h270_task2_check`、`h180_task3_check`、
  `h180_return_origin`，目标是节点记录的精确 map 位置和 yaw，不使用 shop 的正交朝向吸附。
- 应急脚本复用 shop 自动测试的执行框架，但传入独立序列、校正点、空的货架策略和
  `compress_path=False`。

## 校正机制的当前边界与规范化结论

`NavigationGraph` 已提供通用校正和微调机制，但不保存任何场景节点名单：

1. `moveTo(node, *, correct, rotate=True, align=False, rotate_before_translate=False)` 由调用方
   显式选择到点校正、图边旋转、发车前正交对齐和最终 action 的先转后移。
2. `correctState()` 对当前内部逻辑位姿做有范围限制的平移和 yaw 校正。
3. `turnTo(theta)` 根据实时 SLAM yaw 转到逻辑绝对角度，调用方不再自行
   `getPose() + moveBy(theta=...)`。
4. `moveBy(x, y, theta)` 执行机体系相对微调；`recoverState()` 恢复第一次微调前的实测
   pose 或指定节点。
5. shop 的最近正交 yaw、emergency 的精确节点 yaw，以及哪些点调用 `correct=True`，仍由
   外部任务规则选择，不能把节点名称硬编码进类。
