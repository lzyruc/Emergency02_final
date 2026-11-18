# 路线记录、图导航与重放工具

## 第一部分：使用方法

同学实际运行时只需要阅读本部分。

### 固定顺序自动导航测试

先进行 dry-run，检查所有点位和每段规划路径，不连接底盘：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.auto_test_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json
```

确认机器人实际位于 `start_point` 且场地安全后，使用 `move_by` 模式自动执行固定验收序列：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.auto_test_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --linear-speed 0.2 --nav-timeout 30 --execute
```

运行期间可随时输入：

- `stop`：调用底盘 `cancel_action([])` 取消当前导航，并停止发送后续目标。
- `q`：如果仍在移动则先取消当前导航，然后退出程序。

`stop` 是软件层的导航 action 取消，不能替代机器人硬件急停按钮。
自动测试会合并同方向共线节点，但合并后的单段长度不超过 `3 m`。
每当序列到达 `start_point`、`hub_left`、`hub_center` 或 `hub_right` 后，脚本执行两轮
`get_pose + move_by` 微调。每轮先按实时 map 坐标差计算机体平移方向和距离，平移后再取一次
pose，用最短有符号角度误差执行旋转 `move_by`。`start_point` 恢复到程序启动时记录的 map
位置和朝向；三个 hub 的位置恢复到逻辑图目标，但朝向选择当前方向最近的逻辑正交方向
（`0°/90°/180°/270°`），避免为匹配节点记录角度而额外偏航。校正不调用 `move_to` 或
`move_to_poi`；位置误差小于 `20 mm`、角度误差小于 `1°` 时不发送对应命令。

hub/start 校正命令若被底盘返回 `2001 ACTION_FAILED`，脚本会立即读取实时 pose。平移后的
目标误差不超过 `5 cm`，或旋转后的目标误差不超过 `1°` 时，记录“校正软成功”并继续；仍超出
对应容差时原样抛出异常并停止测试。该规则不接受网络、超时、取消或其他错误码。

固定测试还包含以下场景动作约束：

- 到达 `recognize_left`、`recognize_right` 前先旋转、再平移，避免靠近桌面后原地旋转。
- 同一排 `leftN` 点之间沿 x 方向移动时，先朝逻辑 `-y`（`270°`），再横移。
- 同一排 `rightN` 点之间沿 x 方向移动时，先朝逻辑 `+y`（`90°`），再横移。
- `centerN` 不使用货架横移策略；跨排连接（例如 `left12 -> left6`）也不套用该规则。

### 应急场景固定自动测试

应急场景使用独立的固定序列和校正点。先执行 dry-run，只检查图路径：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.auto_test_emergency
```

确认机器人位于 `startpoint`、SLAM 地图及周边安全后再实机执行：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.auto_test_emergency \
  --linear-speed 0.2 --angular-speed 0.3 --nav-timeout 30 --execute
```

固定序列为：

```text
startpoint -> h270_task2_start -> h270_window1_action
-> h270_window2_action -> h270_task2_check
-> h180_task3_check -> h180_return_origin
```

路径经过 `h270_task2_start`、`h270_task2_check`、`h180_task3_check` 或
`h180_return_origin` 时，执行两轮 `get_pose + move_by`，校正到该节点记录的精确 SLAM
期望位置和朝向。应急图禁止合并跨节点路径。

### GUI 点位图转换与实机演示

网页逻辑地图编辑器位于：

```text
agent/navigation/tools/graphGUI/
```

商超编辑器可直接用浏览器打开 `shop/navigation_visualization_system.html`，导入和导出场景
JSON；完整操作、命名规则和酒店/应急兼容限制见
[`tools/graphGUI/README.md`](tools/graphGUI/README.md)。

GUI 文件中的 `robots` 直接作为逻辑节点，`connections` 直接作为双向拓扑边。`x/y` 的
单位是米，坐标约定仍为前方 `+x`、左方 `+y`、逆时针角度为正；`originXOffset` 只作为
编辑器显示信息保留，不参与坐标计算。

构造 `NavigationGraph` 时可以直接传 GUI scene；每次都会自动转换并加载最新 graph：

```python
navigation = NavigationGraph(
    "agent/navigation/tools/graphGUI/shop/scene_shop.json",
    initial_node="start_point",
)
```

需要单独检查转换产物时，仍可使用离线工具：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.convert_scene_navigation \
  agent/navigation/tools/graphGUI/shop/scene_shop.json
```

输出为：

```text
scene_shop.navigation.json
scene_shop.navigation.graph.json
```

实机演示前，先在底盘软件中加载正确的 SLAM 地图并完成定位，再把机器人准确放在
`start_point` 的物理位置和朝向。启动交互式轨道导航：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.demo_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --execute
```

启动后需要手动输入机器人实际所在的初始点位，并再次完整输入同一个点位名确认。只有两次输入
一致后，程序才会连接底盘、读取实时 pose 并建立 SLAM 锚点。

默认使用 `move_by` 模式。需要试验尚未完成实机验收的 `track` 模式时显式增加
`--navigation-mode track`。最短路搜索得到多个节点时，`track` 严格按图边逐段执行，不合并
共线节点。每一段都会：

1. 调用 `get_pose()` 获取该段起点的实时 SLAM map 位姿。
2. 根据该图边的逻辑位移和角度差，计算 map 坐标系下的目标位姿。
3. 以 `起点名-终点名`（例如 `p1-p2`）查询底盘轨道；不存在则创建，已存在则用实时起终点更新。
4. 调用 `move_to_locations(..., uptrack_enabled=False)` 沿轨道移动。
5. 执行后保留轨道；下一段仍重新取位姿，并创建或更新它自己的轨道。

这样会保留人工图中的每个避障关键点，而不是从起点直接跨越多条边前往终点。

`--linear-speed` 和 `--angular-speed` 只用于 `move_by` 模式，不写入轨道。调速时：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.demo_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --navigation-mode move_by --linear-speed 0.8 --execute
```

`--linear-speed` 单位为 `m/s`；`0` 表示底盘不限制。可用 `--angular-speed` 设置旋转速度上限，
单位为 `rad/s`，默认 `0` 表示不限制。速度参数作用于 `move_by`；显式使用
`recoverState(node, method="move_to")` 时，底盘 `move_to` API 没有单动作速度参数。

demo 的单个底盘 action 等待超时默认为 `30 s`（底盘控制器原始默认值为 `300 s`）。需要更快
失败或为长距离动作放宽时可显式设置，例如：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.demo_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --nav-timeout 15 --execute
```

输入点位名称后，程序会显示最短拓扑路径并依次执行这些边。输入 `list` 查看全部点位，
输入 `q` 退出。

如果机器人启动时实际位于其他已知节点，必须在启动后的两次提示中完整输入该节点名称；不能把
当前位置错误地声明为 `start_point`，否则后续固定动作会从错误的位置或朝向开始执行。两次
输入不一致时程序会在连接底盘前中止。

不带 `--execute` 时只规划路径，不连接或移动底盘：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.demo_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json
```

GUI 量尺建图需要满足：节点坐标与实际场地一致；节点角度是期望到达朝向；每条 connection
确实表示两个无障碍、可直线通行的可达节点。GUI 障碍物目前仅作为元数据，不参与
`NavigationGraph` 的最短路代价计算。

跨节点路径合并默认开启，但单个合并段最长为 `3 m`。例如总长 `4.8 m` 的中轴线会保留
`hub_center`，拆为两个不超过 `3 m` 的执行段。如需完全按原始图边执行，可使用
`--no-compress-path`。`track` 模式始终不合并。
路径优化按几何关系合并同方向共线节点，不以节点度数或中间节点朝向作为限制；几何转弯节点
仍然保留。GUI 导航经过所有中间关键点时只做平移，不执行任何旋转，后续平移方向按机器人
尚未改变的当前朝向重新计算。只有最终目标才旋转到 GUI 定义的 angle。GUI 的 `move_by` 模式
执行合并后的关键点；跨越多个原始边时，根据合并段两端逻辑坐标生成一次等价平移。录制路线
生成的非 GUI 图不进行这种跨边合并。

商超 demo 对 `leftN/rightN` 的同排货架移动有显式例外：它先用实时 SLAM yaw 校正到货架
朝向，再按逻辑坐标差执行机体横移。正反向共用同一朝向，区别仅是横移方向相反。该策略通过
`NavigationGraph(lateral_translation_headings={"left": 270, "right": 90})` 配置，不影响
`centerN` 或其他命名节点。

晓悟底盘单次 `move_by` 平移范围为 `0.02～3.0 m`。合并后的平移超过 `3 m` 时会自动等分，
例如 `4.8 m` 首段执行约 `2.4 m`；首段完成后调用 `get_pose`，根据总目标 SLAM map 坐标与
实时 pose 重新计算第二段的机体相对方向和剩余距离。所有分段完成后才执行最终目标旋转。

每条 `move_by` 执行前都会保存当时的 SLAM map pose，并据相对方向、距离和旋转计算该动作
应到达的 map pose。若底盘返回 `1001 NAV_TIMEOUT` 或 `2001 ACTION_FAILED`，系统先取消超时
action，再读取实时 pose，按同一 map 目标重新计算剩余平移或旋转并补偿一次；补偿再次失败才
进入上层错误或软校正逻辑。普通 `moveTo()` 不会因此隐式切换到底盘 `move_to`。

上层抓取微调使用 `NavigationGraph.moveBy()` 时，同一节点第一次微调前会保存实测 SLAM
位姿，后续多次微调不覆盖。抓取结束调用无参数 `recoverState()`，系统用两轮以内的
`get_pose + move_by` 返回第一次微调前的位置和朝向；传入节点名称时先用一次 SLAM
`move_to` 回到节点附近，再用受限 `move_by` 校正最终位姿。人工确认恢复方向为空旷区域时，
可以调用 `recoverState(node, method="move_by")`，直接使用反馈式 `move_by` 返回目标 map pose；
该模式不提供全局避障规划。

自动恢复仍失败时，demo 不退出进程，而是保留 `FAILED` 状态。确认机器人实际位置后输入：

```text
recover 节点名称
```

系统会用 SLAM `move_to` 恢复到该逻辑节点附近，再用受限 `move_by` 校正；恢复成功后可以
继续输入导航目标。

`turnTo()` 和纯旋转 `moveBy(0, 0, theta)` 对底盘的 `1001 NAV_TIMEOUT` 或 `2001 ACTION_FAILED`
采用一次实测 yaw 反馈重试。两次均报告 `1001/2001` 时不退出流程，内部朝向采用最终实测值，并通过
`getLastWarnings()` 返回 `rotation_action_failed_retried`。带平移的 `moveBy()` 第一次返回
`1001/2001` 时按实时 pose 到原 map 目标补偿一次；通信错误和外部取消不重试。

失败诊断中的 `category` 含义：

- `chassis_rejected_before_motion`：底盘未产生明显位移就拒绝动作，常见于目标姿态不可达、
  全局规划失败或障碍物/膨胀区阻挡；底盘 API 未必提供更细原因。
- `chassis_failed_during_motion`：机器人已经运动，随后由底盘报告失败，常见于途中避障、路径
  失效或控制无法继续。
- `chassis_action_cancelled`：底盘 action 被取消。
- `chassis_action_timeout`：等待底盘 action 超时。
- `chassis_communication_error`：连接或 HTTP 请求失败。
- `initial_state_mismatch`：发车前实测 SLAM pose 与 NavigationGraph 当前节点不符。
- `preflight_state_check_error`：发车前读取地图或定位状态时出错，例如当前地图不匹配。
- `arrival_state_mismatch`：底盘报告动作完成，但最终位置或角度超出图导航容差。
- `execution_exception`：不属于以上已知类别的执行异常。

### 1. 进入工程目录

```bash
cd /home/nvidia/X-humanoid
```

运行真实动作前，确认机器人定位正常、急停可用，并清空移动区域内的人员和障碍物。

### 2. 记录路线

先进行 dry-run，熟悉交互但不移动机器人：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.record_navigation
```

**【执行记录】**确认无误后真实记录：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.record_navigation --execute
```

默认最大线速度为 `0.2 m/s`。需要临时调整时使用：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.record_navigation \
  --linear-speed 0.15 --execute
```

不指定文件名时，记录自动保存在 `agent/navigation/data/logs`，文件名包含当前时间戳。
也可以指定文件：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.record_navigation scene_route.json --execute
```

程序会依次询问：

```text
当前点位是否需要记录（Y/y/N/n）：
当前点位名称(按照规则)：
请用空格分割，输入移动方向（0～360°）、距离（米）、旋转角度（逆时针为正，q结束）：
```

动作使用三个参数：

```text
direction_deg distance_m rotation_deg
```

移动方向相对于机器人当前朝向：

- `0°`：前
- `90°`：左
- `180°`：后
- `270°`：右
- `360°`：等同于 `0°`

常用示例：

```text
0 1.0 0
```

向前移动 1 米，不旋转。

```text
90 1.0 0
```

向左移动 1 米，不旋转。

```text
0 1.0 90
```

先向前移动 1 米，再逆时针旋转 90°。

点位名称使用小写树状命名，例如：

```text
emergency.window.approach
```

输入 `q` 结束并保存记录。

record 会在每次执行前检查下一逻辑位姿，安全范围为：

```text
0.0 <= x <= 3.7 m
-2.7 <= y <= 2.7 m
```

边界值允许到达；目标位姿越界时，该动作不会写入 actions，也不会发送到底盘，程序会重新提示输入。

### 3. 重放路线

先预览路线，不移动机器人：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.replay_navigation scene_route.json
```

**【执行重放】**将机器人放回记录时的同一物理起点和初始朝向，确认现场安全后执行：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.replay_navigation scene_route.json --execute
```

replay 默认采用路线文件记录的速度。重放旧路线时可以显式限制最大线速度：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.replay_navigation scene_route.json \
  --linear-speed 0.2 --execute
```

重放完成后，路线文件旁会生成独立的 `*.replay_时间.json` 执行日志。

### 4. 查看和编辑 GUI 场景连接

graphGUI 的 `scene.json` 是唯一地图编辑源。查看并校验节点、边：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.edit_navigation \
  agent/navigation/tools/graphGUI/shop/scene_shop.json list
```

创建或删除一条双向连接时，按节点名称操作。默认输出新文件，不覆盖源 scene：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.edit_navigation \
  scene_shop.json connect hub_left recognize_left

PYTHONPATH=. python3 -m agent.navigation.tools.edit_navigation \
  scene_shop.json disconnect hub_left recognize_left
```

显式传入 `--in-place` 才原子覆盖源文件。输出会通过与 `convert_scene` 相同的完整校验；删除桥
导致图不连通时拒绝保存。派生的 `*.navigation.json` 和 `*.graph.json` 不作为编辑入口。

## 第二部分：定义与实现说明

### 1. 动作定义

每行输入固定按以下顺序执行：

1. 按 `direction_deg` 指定的机体相对方向平移。
2. 平移完成后，按 `rotation_deg` 原地旋转。

平移使用：

```python
move_by(direction=direction_rad, distance=distance_mm, move_mode=0)
```

旋转使用两阶段 `move_by` 反馈修正：

1. 调用 `get_pose()` 获取旋转前的 map yaw，并计算目标 yaw。
2. 调用 `move_by(direction=rotation_rad, distance=0, move_mode=1)` 执行首次旋转。
3. 再次调用 `get_pose()`，按 SLAM yaw 计算与目标 yaw 的最短角误差。
4. 如果误差不小于 `1°`，再次调用 `move_by` 修正；小于底盘最小旋转阈值时跳过。
5. 修正后调用 `get_pose()`，记录最终 yaw 误差。

工具不使用固定角度补偿；第二次旋转量完全来自本次动作后的实时测量误差。

### 2. 逻辑坐标系

`logical_pose` 是路线自身的局部坐标：

- 启动位置固定为 `(x=0, y=0, theta=0°)`。
- `+x` 为初始前方。
- `+y` 为初始左方。
- 逆时针为正。
- 绝对角度范围为 `[0, 360)`。

命名点位只保存 `logical_pose`。逻辑平移的全局方向为：

```text
当前逻辑 theta + direction_deg
```

第一参数只决定平移方向，不改变逻辑 theta；第三参数才改变逻辑 theta。

record 的逻辑工作区域是闭合矩形 `[0.0, 3.7] × [-2.7, 2.7] m`。因为每次平移都是直线且该
区域为凸集，只要动作前的当前位姿和动作后的目标位姿都在矩形内，整段直线路径也在边界内。

### 3. 地图坐标系

`map_pose_before/after` 来自底盘 `get_pose()`，属于当前 SLAM 地图坐标系：

```json
{
  "x_m": 1.23,
  "y_m": 4.56,
  "yaw_rad": 0.78
}
```

map 位姿只用于记录真实执行位置、计算旋转反馈误差和失败恢复，不能写入逻辑点位。

### 4. JSON 动作与落盘

如果一行同时包含平移和旋转，JSON 会拆成两个 command：

- 平移 command 的 `method` 为 `move_by`。
- 新记录的旋转 command 的 `method` 为 `move_by_with_pose_feedback`。
- replay 仍兼容旧记录中的 `move_to_at_current_map_xy` 方法名称，但会按新的反馈旋转方式执行。

每条真实命令执行前都会：

1. 调用 `get_pose()`。
2. 保存 `map_pose_before`。
3. 将状态设置为 `pending` 并立即落盘。
4. 执行底盘命令。
5. 保存 `map_pose_after` 和控制器返回值。
6. 将状态更新为 `completed` 并再次落盘。

旋转 command 还会在 `rotation_feedback` 中逐步落盘首次旋转、修正旋转和最终 yaw 误差。

写文件采用临时文件加原子替换，程序异常中断时不会留下半个 JSON 文件。

### 5. 失败恢复

如果记录过程中某条底盘命令因障碍物或其他原因失败：

1. 将该命令更新为 `failed` 并落盘。
2. 读取该命令执行前保存的 `map_pose_before`。
3. 调用 `move_to` 尝试返回当步原 map 位姿。
4. 保存恢复目标、结果、恢复后 map 位姿或失败原因。

如果一行输入包含“平移 + 旋转”，两者是独立命令。例如旋转失败时，恢复目标是旋转开始前的
位置，也就是已经完成平移后的位置，而不是整行输入之前的位置。

### 6. 重放范围

可以使用以下参数只重放部分动作：

```bash
--start-action N --stop-action M
```

动作范围采用 `[N, M)`。局部重放前必须自行保证机器人的实际起始位姿与所选动作的
`logical_pose_before` 一致，否则后续逻辑坐标没有意义。

### 7. 图文件

GUI scene 是推荐的唯一编辑源，构造函数每次自动生成 route 和 graph。直接传 route 时，
`NavigationGraph` 第一次加载会根据 points/actions 构建同名 `.graph.json`，之后优先读取该
派生图。修改 route 后使用
`python3 -m agent.navigation.tools.rebuild_navigation_graph <route.json>` 显式刷新，不直接
编辑 graph。

上层 Agent 和状态机的完整调用约定见
[`docs/NavigationGraph_API.md`](docs/NavigationGraph_API.md)。

职责划分如下：

- `navigation_graph.py`：定义图结构，并负责从 record JSON 首次构图、读写 graph JSON、规划和执行导航。
- `map_slam.py`：负责逻辑/map坐标转换及底盘地图、位姿薄封装。
- `tools/edit_navigation.py`：只对 graphGUI 源 scene 创建、删除或列出双向连接。
- `tools/test_navigation_graph.py`：针对指定路线验证路径规划或执行实机导航。

每条逻辑边以两个有向 traversal 保存，但 edit 将它作为一条双向边创建或删除。一个 traversal
的 `actions` 始终是列表，可以包含多次平移和旋转。

反向 action 按以下规则生成，并反转 action 列表顺序：

```text
reverse_direction = direction + 180° - rotation
reverse_distance = distance
reverse_rotation = -rotation
```

### 8. NavigationGraph 状态

`StateStatus` 只有三种值：

- `AT_NODE`：位于一个语义节点，`getState()` 同时返回不可变 `Node`。
- `IN_TRANSIT`：正在边上移动，`getState()` 的 Node 为 `None`。
- `FAILED`：移动命令失败或到位检查失败，Node 为 `None`。

`NavigationGraph` 的 `navigation_mode` 是全图统一的执行模式：

- `move_by`：默认值，重放每条图边记录的相对 actions。
- `track`：实验模式；每条图边使用实时 SLAM pose 创建或更新同名两点直线轨道并逐段执行。

`checkState()` 会检查当前地图名称、实时位置误差和角度误差。默认位置容差为 0.15 m，当前固定
角度容差为 5°。底盘查询异常原样向上传递；`move_by` 模式执行图边时不强制调用该检查，从而
保留无 SLAM 的旧流程。

### 9. MapSLAM

`MapSLAM` 从 record JSON 读取逻辑初始位姿、`map_context.initial_map_pose` 和地图名称，由此建立
固定的二维刚体变换。每个节点仍只保存逻辑位姿；需要点导航时才动态计算目标 map 位姿。

`initial_node` 不参与坐标变换。正常初始化默认把 JSON 中第一个命名点作为当前逻辑节点；保留的
可选 `initial_node` 只用于明确覆盖运行状态或离线测试。

`MapSLAM` 已实现当前地图查询、实时位姿、逻辑/map 双向坐标转换和动态 `getPoint(name)`。
`loadMap(name)` 只是显式调用底盘 `start_localization()`，请求底盘激活其内部已有的同名地图；
它不会读取本地地图文件，也不会自动等待、重定位或重试。当前每次只读取一次实时 pose，尚未
加入多秒稳定采样。

图导航命令失败时，状态直接进入 `FAILED`，底盘异常原样向上传递。系统不会自动返回边的起始
节点、切换地图、重定位或重试；这些策略由上层 Agent 或人工决定。

`recoverState()` 是供上层显式调用的恢复工具。不传节点且没有微调快照时，它会依据当前实时
map pose 选择距离最近的图节点；命名节点恢复先执行一次 SLAM `move_to`，进入
`30 cm / 15°` 范围后再用最多两轮 `move_by` 收敛到 `5 cm / 1°`。

### 10. 最短路和微调

`moveTo(node, *, correct, rotate=True, align=False, rotate_before_translate=False)`
使用 Dijkstra，首先按 actions 的总移动距离选择路径；距离相同时选择边数更少的路径。调用方
必须显式传入 `correct`。`rotate` 默认保留图边旋转，传入 `False` 时只执行平移。
`move_by` 模式保留每个图节点，并在每条边最后一个 action 前读取实时 SLAM pose，重新计算到
下一节点绝对 map pose 的平移方向和距离，以小角度斜向移动吸收累计的位置偏差。
平移按最终整数毫米执行底盘 `[20, 3000] mm` 边界，小于 `20 mm` 时跳过，超过 `3000 mm`
时分段；绝对值小于 `1°` 的旋转同样跳过，不向底盘发送非法动作。
初始化、主要运动接口、动作跳过及 `1001/2001` 补偿会使用 `[NavigationGraph]` 前缀向标准输出
实时打印进度；状态读取接口不打印。
`align=True` 会在发车前根据实时 SLAM yaw 转到最近的逻辑正交朝向，但不限制平移
方向；调用方必须确认当前位置允许原地旋转。需要在最终目标前先转向再抵近时，调用方显式传入
`rotate_before_translate=True`，NavigationGraph 不保存场景节点名单。

`moveBy(x, y, theta, correct=True)` 将 `(x, y)` 解释为机器人机体系下的米制微调，`theta`
为逆时针角度；默认按本次动作开始前的实测 map pose 补偿最终误差，宽敞区域传
`correct=False` 跳过。微调完成后仍保留原语义节点，但内部逻辑 pose 会按微调量更新。

`turnTo(theta)` 将 `theta` 解释为逻辑坐标系中的绝对角度，读取实时 SLAM yaw 后按最小转角
执行反馈旋转。成功后保留原语义节点，只更新内部逻辑朝向。
