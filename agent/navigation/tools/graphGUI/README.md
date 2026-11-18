# 逻辑地图网页 GUI

## 1. 用途

本目录是商超、酒店、应急赛项的逻辑地图编辑工具。它使用实地卷尺测量值维护语义点位、障碍物
示意和点位之间的双向连接，再通过导航转换脚本生成 `NavigationGraph` 使用的逻辑拓扑图。

推荐工作流：

```text
现场测量
  -> 网页 GUI 编辑 robots / connections
  -> 导出场景 JSON
  -> convert_scene_navigation.py 校验并生成 graph
  -> graphGUI 人工复核
  -> demo/auto_test 实机测试
```

GUI 只负责逻辑地图，不读取或修改底盘 SLAM 地图。运行 demo 时，用户需要连续两次输入
机器人实际所在点位；确认一致后，导航系统才使用该点位的实时 SLAM pose 建立逻辑坐标系到
map 坐标系的对齐。

## 2. 文件与兼容性

| 场景 | 编辑器或数据 | 当前能否直接用于 NavigationGraph |
| --- | --- | --- |
| 商超 | `shop/navigation_visualization_system.html`、`shop/scene_shop.json` | 可以，推荐使用 |
| 酒店 | `hotel/navigation4hotel.html`、`hotel/scene_hotel.json` | 暂不可以直接使用 |
| 应急 | `emergency/scene_emergency.json` | 仅为参考数据，目录内暂无对应 HTML |

商超编辑器采用导航系统统一坐标：

- 原点为 `start_point` 的实测位置。
- `+x` 指向机器人初始正前方，画面中向上。
- `+y` 指向机器人初始左方，画面中向左。
- `0°` 为 `+x`，逆时针为正，角度归一化到 `[0, 360)`。
- 坐标单位为米，角度单位为度。

酒店编辑器目前采用“x 向右、y 向上、原点在左下角”，与当前导航逻辑坐标定义不同。酒店和
应急示例中还存在 `机器人1` 等中文默认名、应急示例重复名称；当前转换器会拒绝这些名称。
在完成坐标转换、唯一英文命名和现场复核前，不得直接用于实机导航。

## 3. 打开商超编辑器

HTML 不依赖网络、npm 或 Python Web 框架，可以直接用浏览器打开：

```bash
open agent/navigation/tools/graphGUI/shop/navigation_visualization_system.html
```

Linux 桌面可使用：

```bash
xdg-open agent/navigation/tools/graphGUI/shop/navigation_visualization_system.html
```

也可以在解压包根目录启动一个本地静态服务器：

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://127.0.0.1:8000/agent/navigation/tools/graphGUI/shop/navigation_visualization_system.html
```

网页完全在本机运行，不会自动上传地图。

## 4. 导入已有逻辑地图

1. 打开商超编辑器。
2. 点击右上角“导入”。
3. 选择 `shop/scene_shop.json` 或其他符合相同格式的场景 JSON。
4. 检查场地尺寸、障碍物、节点坐标、朝向和连接。

不要点击“初始化”来打开已有地图；初始化会清空当前页面中的障碍物、节点和连接。浏览器页面
关闭前未导出的修改无法恢复，建议每次编辑前备份原 JSON。

## 5. 编辑操作

### 点位

- 切换到“机器人”模式，在画面中点击可创建点位并输入朝向。
- 也可在“精确添加”区域输入 `x/y/角度` 后添加。
- 在右侧点位列表展开一项，可精确修改名称、坐标和角度。
- 拖动点位可快速调整坐标；实测地图最终应在右侧列表中填写精确值。
- 右键点击点位或使用右侧“删除”按钮可删除；删除点位会同时删除关联边。

GUI 中的“机器人”实际表示 NavigationGraph 的逻辑节点，不代表多台机器人。

### 连接

1. 切换到“连接”模式。
2. 点击第一个节点。
3. 点击第二个节点。

生成的是无向连接，转换后成为两个方向的可通行边。每条连接必须满足：

- 两点之间经过现场确认可以直线通行。
- 不能为了缩短路径跨过货架、桌面、围挡或狭窄禁行区。
- 需要强制经过的安全拐点必须显式保留为节点和连接。
- 整张图必须连通，否则转换器会拒绝生成图。

### 障碍物与视图

- “障碍物”模式可添加、拖动、修改和删除矩形障碍物。
- 障碍物目前只用于 GUI 示意，不参与最短路、碰撞检测或底盘避障。
- 鼠标滚轮缩放。
- 中键拖动或 `Ctrl + 左键` 拖动平移视图。
- “只读”模式下左键只平移视图，不修改对象。
- “重置视图”恢复自动适配比例。

## 6. 点位命名规则

转换器要求名称：

```text
^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$
```

即：

- 必须以小写英文字母开头。
- 只能使用小写字母、数字、下划线和分层用的点。
- 每个节点名称必须唯一。
- 不允许空格、中文、连字符或重复名称。

有效示例：

```text
start_point
hub_left
left6
emergency.window.approach
```

点位名称、打点路线和命名含义以现场负责人提供的表格为准。

## 7. 导出和保存

点击“导出”后，浏览器下载名为 `scene.json` 的文件。GUI 不会直接覆盖原文件。

建议：

1. 将下载文件重命名为场景名，例如 `shop.json`。
2. 保留带时间戳的备份。
3. 重新构造 `NavigationGraph`，由它自动校验、转换并加载最新 graph。
4. 不要把 `*.navigation.json` 或 `*.graph.json` 再导入 GUI；GUI 的入口始终是原始场景 JSON。

## 8. 交给 NavigationGraph

业务代码可以直接传 GUI scene，不需要手工执行 convert：

```python
navigation = NavigationGraph(
    "agent/navigation/tools/graphGUI/shop/scene_shop.json",
    initial_node="start_point",
)
```

每次构造都会重新生成 `scene_shop.navigation.json` 和
`scene_shop.navigation.graph.json`。需要在命令行单独检查转换结果时执行：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.convert_scene_navigation \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --initial-node start_point
```

生成：

```text
shop.navigation.json
shop.navigation.graph.json
```

转换器会检查：

- 节点 ID 和名称是否唯一。
- 坐标和角度是否为有限数字。
- 边是否引用有效节点。
- 是否存在自环或重复边。
- 整张图是否连通。

`originXOffset` 仅作为编辑器显示元数据保留，不会加到逻辑坐标。

转换前必须在 graphGUI 中检查节点、连接和空间位置；转换后可用
`edit_navigation.py ... list` 对源 scene 做同规则校验和文本复核。

命令行只需要增删连接时，也可以使用：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.edit_navigation \
  agent/navigation/tools/graphGUI/shop/scene_shop.json list

PYTHONPATH=. python3 -m agent.navigation.tools.edit_navigation \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  connect node_a node_b --output /tmp/scene_shop.edited.json
```

该工具只编辑 graphGUI 的源 scene，不编辑派生的 `*.graph.json`。新连接与网页 GUI 一样写成
`{"fromId": ..., "toId": ...}`；`connect` 可以修复尚未连通的 scene，删除导致结果图不连通
时拒绝保存。默认写入
`<scene>.edited.json`，只有显式传入 `--in-place` 才覆盖源文件。

## 9. dry-run 与实机

先只规划，不移动机器人：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.demo_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --navigation-mode move_by
```

确认机器人真实位于 `start_point`、底盘已加载正确的 SLAM 地图且定位稳定后执行：

```bash
PYTHONPATH=. python3 -m agent.navigation.tools.demo_navigation_graph \
  agent/navigation/tools/graphGUI/shop/scene_shop.json \
  --navigation-mode move_by \
  --linear-speed 0.2 \
  --nav-timeout 30 \
  --execute
```

demo 启动后会要求手动输入机器人实际所在点位，并再次完整输入同一个点位名确认。只有两次
输入一致后，实机模式才会连接底盘，重新读取当前 map 和该点位的实时 pose，建立逻辑地图与
SLAM 地图的刚体对齐。因此 GUI JSON 不应写死 SLAM map 坐标。

## 10. 商超特殊规则

当前商超 demo/19 点自动测试还配置了以下策略：

- 工具在调用 `moveTo("recognize_left/right", ...)` 时显式传入
  `rotate_before_translate=True`，到达前先旋转、重新读取 yaw，再平移。
- 同排 `leftN` 移动时先朝逻辑 `-y`（`270°`）再横移。
- 同排 `rightN` 移动时先朝逻辑 `+y`（`90°`）再横移。
- `centerN` 和跨排连接不使用货架横移。
- `start_point` 和三个 hub 到位后做两轮 SLAM pose 校正。
- hub 朝向取当前最近的逻辑正交方向，减少不必要偏航。

这些是导航工具传入的场景策略，不保存在 GUI JSON 中。修改节点名称时必须同步检查 demo 中的
策略配置，否则相应特殊行为不会触发。
