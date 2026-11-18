# NavigationGraph 接口文档

## 1. 定位与适用范围

`NavigationGraph` 是上层 Agent（状态机、任务规划器等）调用导航能力的统一入口。上层只需要使用
语义节点名称，不需要了解节点之间具体包含多少次轨道或 `move_by` 底盘动作。

典型调用关系：

```text
状态机 / Agent
      |
      | moveTo("emergency.point2", correct=True)
      v
NavigationGraph（状态、最短路、动作编排、失败返回）
      |
      v
TienYi2Pro 底盘控制器
```

当前版本适合已人工记录、确认节点间可达且现场环境稳定的路线。需要注意：

- SLAM 对齐来自 record JSON 的单个初始逻辑/map 位姿对应关系。
- `get_pose()` 当前只读取一次，尚未实现多秒稳定采样和离群值过滤。
- `checkState()` 的角度容差当前固定为 5°。
- 运行时状态只保存在当前 Python 对象中，进程重启后不会恢复。
- 同一个 `NavigationGraph` 对象不应被多个线程同时调用。

## 2. 导入

从包含 `agent/` 的工程或交付目录运行，并设置 `PYTHONPATH=.`：

```python
from pathlib import Path

from agent.navigation import (
    NavigationGraph,
    NavigationGraphError,
    Node,
    StateStatus,
)
```

## 3. 初始化与关闭

### `NavigationGraph(..., navigation_mode="move_by", compress_path=True)`

参数：

- `route_path: Path`：GUI scene 或 route JSON；参数名为兼容旧调用保留。
- `controller`：可选的底盘控制器实例。传入后由调用方管理其生命周期。
- `initial_node: str | None`：仅用于覆盖初始逻辑状态；正常使用时省略，默认读取 JSON 第一个节点。
- `navigation_mode`：全图执行模式，允许 `move_by` 或实验性的 `track`，默认 `move_by`。
- `linear_speed`：`move_by` 最大线速度，单位 `m/s`；`0` 表示不额外限制。
- `angular_speed`：`move_by` 最大角速度，单位 `rad/s`；`0` 表示不额外限制。
- `compress_path`：是否合并同方向共线图边，默认 `True`。合并后每个执行段最长 `3 m`；超过时保留
  适当的中间节点。`track` 模式始终不合并。
- `lateral_translation_headings`：可选的“节点名前缀 → 逻辑朝向角”映射。同一前缀的数字节点在
  同一 y 行沿 x 移动时，先校正到指定绝对逻辑朝向，再用机体横向平移。例如商超场景传入
  `{"left": 270.0, "right": 90.0}`。跨排、沿 y 移动和其他名称不受影响。
- `rebuild_graph`：默认 `False`。直接修改 route 后传 `True`；scene 输入每次自动转换，不需要
  设置。

GUI scene 是推荐的唯一编辑源：

```text
scene_shop.json                    -> GUI scene，传给 NavigationGraph
scene_shop.navigation.json         -> 每次自动生成的 route
scene_shop.navigation.graph.json   -> 每次自动生成并加载的 graph
```

构造函数读取内容后自动判断 `robots/connections` scene 或 `points/actions` route。scene 每次
初始化都会重新转换，不需要调用方执行 convert。直接传 route 时，如果 `.graph.json` 存在则
读取已有图，否则根据 route 创建；修改 route 后传 `rebuild_graph=True`，或运行
`python3 -m agent.navigation.tools.rebuild_navigation_graph <route.json>`，否则仍会读取旧
graph。

坐标对齐不读取 `initial_node`。它固定使用 record JSON 中
`coordinate_systems.logical.initial_pose` 与 `map_context.initial_map_pose` 计算逻辑到 map 的
二维刚体变换。`track` 模式缺少地图名称或初始 map pose 时，初始化直接失败。

推荐使用上下文管理器，让内部创建的底盘控制器得到关闭：

```python
scene = Path("config/scene_shop.json")

with NavigationGraph(scene, initial_node="start_point") as navigation:
    navigation.moveTo("hub_center", correct=True)
```

如果没有传入 `controller`，第一次执行真实移动时会使用默认参数创建
`KnewbotChassisController`。如果传入 controller，退出上下文时不会替调用方关闭它。

### `close() -> None`

关闭由 `NavigationGraph` 自己创建的 controller。重复调用是安全的。

## 4. 核心导航接口

### `moveTo(node: str, *, correct: bool, rotate: bool = True, align: bool = False, rotate_before_translate: bool = False) -> list[str]`

从当前节点规划到目标节点的最短路径，并按全图 `navigation_mode` 执行。

```python
path = navigation.moveTo("emergency.end", correct=True)
print(path)
# ["emergency.start", "emergency.point2", "emergency.end"]
```

行为：

1. 要求当前状态为 `AT_NODE`。
2. `track` 模式确认当前底盘地图名称。
3. 使用 Dijkstra 计算路径，权重为 actions 中平移距离绝对值之和。
4. 距离相同时，优先选择边数更少的路径。
5. 执行过程中状态为 `IN_TRANSIT`。
6. `track` 模式严格按最短路中的每条边逐段建轨和执行，不合并中间节点。
7. `correct=True` 时，最终误差在 `0.15 m / 5°` 范围内使用最多两轮 `move_by` 精校正到
   `0.05 m / 1°`；超出范围或校正失败时进入 `FAILED`。
8. `rotate_before_translate=True` 时，仅在到达最终目标的最后一个 action 先旋转，通过实时
   SLAM yaw 反馈校正后重新计算机体平移方向，再执行平移；默认 `False`。
9. `rotate=False` 时跳过图边旋转，但根据图中原计划朝向换算后续平移方向；默认 `True`。
10. `align=True` 时，在发车前根据实时 SLAM yaw 转到逻辑坐标系中最近的
    `0°/90°/180°/270°`；它不限制随后平移方向，默认 `False`。
11. 成功后返回实际规划的节点名称列表，状态为 `AT_NODE`。

如果目标就是当前节点，不执行底盘动作，返回只包含当前节点的列表。

失败语义：

- `move_by` 返回 `1001 NAV_TIMEOUT` 或 `2001 ACTION_FAILED` 时，会读取实时 SLAM pose，按动作
  开始前确定的 map 目标重新计算剩余平移或旋转并补偿一次；`1001` 会先取消超时 action，普通
  `moveTo()` 不会隐式切换到底盘 `move_to`。
- `moveBy(..., correct=False)` 的动作及补偿连续返回 `1001/2001` 时，读取实时SLAM pose更新
  内部逻辑位姿，保留动作前语义节点和 `AT_NODE`，使后续 `moveTo()` 可以继续；同时写入
  `move_by_action_failed_continued_from_measured_pose` warning。是否完整到位交给后续节点校正。
- 其他底盘动作失败时，状态变为 `FAILED`，节点为 `None`，底层异常原样向上传递。
- 到位几何检查失败时，状态变为 `FAILED` 并抛出 `NavigationGraphError`。
- 不自动返回原节点、不自动切换地图，也不自动重定位。

`track` 模式在每条边开始前读取实时 SLAM pose，将该边的逻辑位移和角度差叠加到实时
map pose。轨道按 `起点名-终点名`（例如 `p1-p2`）命名：不存在时创建，已存在时更新两个
waypoint；调用 `move_to_locations` 后保留在底盘地图中。反向边使用独立名称，例如 `p2-p1`。

`move_by` 模式保留最短路中的节点顺序。每条边的既定中间 actions 仍按顺序执行，最后一个
action 会根据实时 SLAM pose 重新计算到下一节点绝对 map pose 的平移方向和距离，允许小角度
斜向补偿并避免横向误差逐边累计。它不会跳过图节点；actions 在两种模式下都继续用于最短路
边权。

Navigation 在下发前按最终整数毫米检查平移：`<20 mm` 跳过、`20~3000 mm` 直接执行、
`>3000 mm` 分段执行；绝对值 `<1°` 的旋转跳过。过小动作不会作为底盘错误向上抛出。

初始化和主要运动接口会向标准输出打印带 `[NavigationGraph]` 前缀的实时进度；包括路径逐边
执行、到达、恢复、过小动作跳过和 `1001/2001` 补偿。状态读取接口保持静默。

商超工具的推荐配置如下：

```python
navigation = NavigationGraph(
    route,
    navigation_mode="move_by",
    lateral_translation_headings={"left": 270.0, "right": 90.0},
)

navigation.moveTo(
    "recognize_left",
    correct=False,
    rotate_before_translate=True,
)
```

此配置使识别点采用“先转后移”，使 `leftN` 同排移动始终面向逻辑 `-y`，使 `rightN` 同排
移动始终面向逻辑 `+y`。`centerN` 保持普通导航行为。

窄通道发车前需要先消除底盘偏航时显式开启：

```python
navigation.moveTo(
    "aisle_end",
    correct=False,
    rotate=False,
    align=True,
)
```

该选项会查询并确认当前 SLAM 地图，只校正发车朝向。开启前必须由业务调用方确认当前位置有
足够原地旋转空间。

### `moveBy(x: float, y: float, theta: float, *, correct: bool = True) -> dict[str, float]`

供上层进行小范围机体系微调。

- `x`：机器人前方为正，单位米。
- `y`：机器人左方为正，单位米。
- `theta`：逆时针为正，单位度。
- `correct`：默认 `True`，按本次动作开始前的实测 map pose 校正最终位姿；宽敞区域可传
  `False` 跳过动作成功后的补偿。
- 返回值：微调后的内部逻辑位姿。

```python
# 向前微调 0.2 m，向右微调 0.1 m，不旋转
pose = navigation.moveBy(0.2, -0.1, 0.0)

# 宽敞区域快速移动，不执行成功后的最终位姿补偿
pose = navigation.moveBy(0.8, 0.0, 0.0, correct=False)
```

名义平移由一次 `move_by` 完成；`correct=True` 时不区分初始误差大小，根据最终实测误差最多
追加两轮校正。两轮后仍未收敛时记录 warning、保留 `AT_NODE` 并采用最终实测 pose，不因到位
误差抛出异常。旋转时先读取当前 map yaw，调用一次
`move_by(move_mode=1)`，再读取 map yaw 计算角度误差，并用第二次 `move_by` 修正；误差已经小于
内部阈值时跳过第二次命令。如果平移和旋转都非零，则先平移、后旋转。

微调成功后仍保留微调前的语义节点，即 `getState()` 仍返回该节点，但 `getPose()` 的内部逻辑位姿
会包含微调量。因此这个接口只适合短距离操作对位，不应用来替代节点间导航。

同一节点第一次调用 `moveBy()` 前，系统会保存当时的节点、内部逻辑位姿和实时 SLAM map
位姿。该节点后续多次 `moveBy()` 不会覆盖这个恢复点。抓取或操作完成后调用无参数
`recoverState()`，即可回到第一次微调前的位置和朝向。

`correct=False` 时，如果名义动作及内部补偿连续返回 `1001/2001`，不会因该动作进入 `FAILED`；
系统读取实时SLAM pose、保留原语义节点并返回，使下一条 `moveTo()` 继续执行。其他失败仍将
状态置为 `FAILED`，底层异常原样向上传递，不自动返回原节点。

### `turnTo(theta: float) -> dict[str, float]`

转到逻辑坐标系中的绝对朝向。

- `theta`：目标绝对角度，单位度，逆时针为正。
- 返回值：更新朝向后的内部逻辑位姿。

```python
pose = navigation.turnTo(90.0)
```

该接口读取实时 SLAM yaw，转换到逻辑坐标系后计算最小有符号转角，再使用带 yaw 反馈的
`move_by` 旋转。小于 `1°` 的转角不下发命令。成功后保留当前语义节点，只更新
`theta_deg`；失败时进入 `FAILED`。它与 `moveBy(..., theta)` 的区别是前者使用绝对逻辑
朝向，后者使用相对旋转角。

`turnTo()` 和 `moveBy()` 共享同一轮微调恢复快照，后续可以调用无参数 `recoverState()` 回到
第一次微调前的实测位置和朝向。

## 5. 状态与位姿接口

### `getState() -> tuple[StateStatus, Node | None]`

返回两个字段：运行状态和当前节点。

```python
status, node = navigation.getState()
if status is StateStatus.AT_NODE and node is not None:
    print(node.name)
    print(node.pose)
```

`Node` 是不可变的公开数据类型，包含语义名称 `name` 和逻辑坐标 `pose`。

状态定义：

| 状态 | 节点 | 含义 |
| --- | --- | --- |
| `StateStatus.AT_NODE` | 当前不可变 `Node` | 逻辑上位于某个节点 |
| `StateStatus.IN_TRANSIT` | `None` | 正在执行边或微调动作 |
| `StateStatus.FAILED` | `None` | 移动命令失败或到位检查失败 |

### `getPose() -> tuple[dict | None, dict | None]`

返回 `(logical_pose, map_pose)`：

```python
logical_pose, map_pose = navigation.getPose()
```

逻辑位姿字段为：

```json
{"x_m": 1.5, "y_m": 0.0, "theta_deg": 0.0}
```

仅在 `AT_NODE` 时返回位姿；`IN_TRANSIT` 或 `FAILED` 时返回 `(None, None)`。

map 位姿字段为：

```json
{"x_m": 1.5, "y_m": 0.0, "yaw_rad": 0.0}
```

这里返回的是由当前预期逻辑位姿换算出的目标 map pose，不会额外调用底盘 `get_pose()`。

### `checkState(epi: float = 0.15) -> bool`

检查当前地图名称以及实时 SLAM pose。位置误差不超过 `epi` 米且角度误差不超过5°时返回
`True`；状态不是 `AT_NODE` 时返回 `False`。地图不匹配时抛出 `NavigationGraphError`，底盘查询
异常原样向上传递。

### `recoverState(node: str | None = None) -> None`

尝试恢复最近保存的实测位姿，或移动到指定逻辑节点：

```python
# 返回第一次 moveBy 前的实测位置；若不存在保存点，兼容原来的最近节点恢复。
navigation.recoverState()

# 恢复到指定图节点的理论 map 位姿。
navigation.recoverState("emergency.start")
```

- 指定节点不存在时抛出 `NavigationGraphError`。
- 恢复移动失败时状态变为 `FAILED`，并继续抛出底层异常。
- 每次真正执行 `moveTo()` 前都会保存发车前的节点、逻辑位姿和实测 map 位姿，用于导航失败
  后的无参数恢复。
- 某节点第一次 `moveBy()` 前会用微调前的实测位姿覆盖上述恢复点；该节点后续多次微调不覆盖。
- 不传 `node` 且存在保存点时，使用最多两轮 `get_pose + move_by` 返回该实测 map 位姿，并
  恢复保存的节点和逻辑位姿。位置进入 `5 cm`、角度进入 `1°` 即认为恢复成功。
- 不传 `node` 且没有保存点时，兼容原行为：读取实时 map pose，选择平面距离最近的图节点。
- `method` 只接受 `"move_to"` 和 `"move_by"`，默认 `"move_to"`。无参数且存在保存点时始终
  使用反馈式 `move_by` 返回该局部快照。
- 传入节点名称且使用默认 `method="move_to"` 时忽略保存点，先使用一次 SLAM `move_to` 完成长距离恢复；实测位姿进入
  `30 cm / 15°` 范围后，再使用最多两轮 `get_pose + move_by` 收敛到 `5 cm / 1°`。
  `move_to` 报告 `2001 ACTION_FAILED` 时立即原样抛出，不执行后续 `move_by` 校正，由业务层
  决定恢复路线；其他底层异常同样直接抛出。
- `method="move_by"` 时不调用 `move_to`，根据实时 SLAM pose 与目标节点 map pose 进行最多两轮
  反馈移动；距离超过 `3 m` 时分段且每段重新测量。该模式没有全局避障规划，只能用于人工确认
  空旷的区域。
- 保存点恢复成功或指定节点恢复成功后，会清除本轮恢复记录；下次 `moveBy()` 会建立新记录。
- 该接口只在上层显式调用时执行，不会被普通导航失败自动触发。

### 旋转 `ACTION_FAILED` 恢复

`turnTo()` 和纯旋转 `moveBy(0, 0, theta)` 遇到底盘 `1001 NAV_TIMEOUT` 或 `2001 ACTION_FAILED`
时，会先取消超时 action，再读取实时 SLAM yaw；未到达目标则按剩余角度重试一次。第二次仍为
`1001/2001` 时不会抛出该异常或进入 `FAILED`，
而是保留当前语义节点、采用实测逻辑朝向，并在 `getLastWarnings()` 中记录
`rotation_action_failed_retried`。通信错误和外部取消仍直接向上传播。

带平移的 `moveBy()` 第一次返回 `1001 NAV_TIMEOUT` 或 `2001 ACTION_FAILED` 时，会读取实时
SLAM pose，并按该动作
开始前确定的 map 目标重新计算剩余方向和距离，只补偿一次。补偿成功时在
`getLastWarnings()` 中记录 `move_by_translation_action_interruption_retried`；补偿再次失败则保留
底层异常并进入 `FAILED`。超过 `3 m` 的平移仍分段执行，每个失败分段遵循相同的一次补偿规则。

## 6. 路径查询接口

### `shortestPath(target: str) -> list[str]`

只计算路径，不执行底盘动作：

```python
path = navigation.shortestPath("emergency.end")
```

要求当前状态为 `AT_NODE`。目标不存在、当前状态不允许规划或图上不可达时，抛出
`NavigationGraphError`。

## 7. 图数据接口

以下接口主要用于离线建图和编辑工具。任务状态机通常只读，不应在机器人执行任务时修改图。

### 只读属性

```python
navigation.nodes       # Mapping[str, Node]，节点名到不可变节点
navigation.node_order  # Sequence[str]
navigation.edges       # Mapping[from, Mapping[to, list[action]]]
navigation.graph_path  # 同名 .graph.json 路径
navigation.route_path  # record JSON 路径
```

每个 action 的格式：

```json
{
  "direction_deg_relative": 0.0,
  "distance_m": 0.5,
  "rotation_deg_ccw": 0.0
}
```

### `createEdge(node_a, node_b, actions=None, save=True) -> None`

创建或覆盖一条双向边。

- `actions` 是 `node_a -> node_b` 的动作列表。
- 未传 `actions` 时，根据两个节点逻辑位姿生成一次直达动作。
- 反向边会自动生成，且动作列表顺序反转。
- `save=True` 时立即原子写入 `.graph.json`。

```python
navigation.createEdge(
    "emergency.start",
    "emergency.point2",
    actions=[
        {
            "direction_deg_relative": 0.0,
            "distance_m": 1.5,
            "rotation_deg_ccw": 0.0,
        }
    ],
)
```

### `deleteEdge(node_a, node_b, save=True) -> None`

同时删除两个方向的边。边不存在时抛出 `NavigationGraphError`。

### `save() -> Path`

把当前图原子写入 `.graph.json`，返回文件路径。不会修改 record JSON。

## 8. 弯腰矩形碰撞检查与接近规划

`AvoidCollision` 保留原有圆形底盘 `checkCollision()`，另提供弯腰矩形模型：
机器人pose位于 `base_footprint` 原点，矩形相对pose前0.65 m、后0.30 m、左右各0.35 m。
因此pose不在矩形几何中心，矩形总长0.95 m、总宽0.70 m。`checkBentCollision()` 以2 cm/2°
采样完整检查“先平移、后旋转”的扫掠区。

`planApproach(dx, dy, dtheta)` 使用5 cm八邻域A*，保持当前朝向全向平移绕障，
最后原地旋转。`planOperationApproach(target_x, target_y)` 依次尝试使目标操作点
落在机器人最终正前方0.40/0.50/0.60/0.70 m，返回第一条完整无碰撞的
路线；该值与弯腰向前占用0.65 m相互独立。两个规划接口都返
`(dx, dy, dtheta)` 列表，无路
返回 `None`；只规划，不下发底盘。

```python
from agent.navigation import AvoidCollision

avoid = AvoidCollision(navigation, "config/scene.json")
motions = avoid.planOperationApproach(1.30, -0.20)
if motions is not None:
    for motion in motions:
        navigation.moveBy(*motion)
```

## 9. 上层状态机推荐用法

```python
from pathlib import Path

from agent.navigation import (
    NavigationGraph,
    NavigationGraphError,
    Node,
    StateStatus,
)


route = Path("agent/navigation/data/logs/scene_route.json")

with NavigationGraph(route, navigation_mode="move_by") as navigation:
    try:
        status, current = navigation.getState()
        if status is not StateStatus.AT_NODE or current is None:
            raise RuntimeError("navigation is not ready")

        navigation.moveTo("emergency.point2", correct=True)

        # 操作模块完成任务前的小范围对位。
        navigation.moveBy(0.10, 0.0, 0.0)

        status, current = navigation.getState()
        if status is StateStatus.AT_NODE:
            print(f"logical node: {current.name}")
            print(f"logical pose: {current.pose}")

    except Exception as exc:
        status, current = navigation.getState()
        print(f"navigation error: {exc}")
        print(f"state={status.value}, node={current.name if current else None}")
        # 状态机在这里转入恢复、人工介入或任务失败分支。
```

捕获移动异常后，`getState()` 返回 `FAILED + None`。机器人位置不再由逻辑图声明，上层应决定
是否调用 `recoverState()`、执行其他恢复方式或交由人工处理。

## 10. 异常约定

`NavigationGraphError` 用于以下情况：

- record 或 graph 数据不合法。
- 节点不存在或路径不可达。
- 当前状态不允许规划或移动。

底层 controller 异常不包装、不吞没，保持原异常类型向上传递。文件与图数据错误使用
`NavigationGraphError` 或原始文件异常。上层任务边界建议同时记录异常、`getState()` 和当前
目标节点。

## 11. 后续增强项

在把 `NavigationGraph` 作为长期运行的正式导航服务前，建议按顺序完成：

1. 为初始对齐和关键到位检查增加稳定时间窗采样、圆周 yaw 平均和漂移检测。
2. 依据实机数据配置位置、角度和定位置信度阈值。
3. 为状态变化和每条边执行增加可复现的运行日志。
4. 如果多个 Agent 会并发调用，增加单执行者队列或互斥机制。
