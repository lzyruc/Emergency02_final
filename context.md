# Emergency02 交接上下文

更新时间：2026-08-16（Asia/Shanghai）

## 1. 当前工作目录与基准

- 当前主要开发目录：`/Users/mac/X-humanoid/Emergency02`
- 比赛入口：`Emergency02/emergency_zyt.py`
- 任务1独立入口：`Emergency02/emergency_task1.py`；无`input()`，全自动清障，确认清场后前往`h270_task2_start`并退出，不执行任务2或任务3动作。
- 比赛机目录：`nvidia@192.168.41.2:/home/nvidia/Emergency02`
- AI 修改记录：`Emergency02/修改记录.md`，最新记录在最上方；当前最新为第120次。
- `Emergency02` 是当前业务逻辑基准，不要为了适配 `0803/Emergency` 反向修改它。
- Navigation 权威源码位于 `/Users/mac/X-humanoid/agent/navigation`。Navigation 修改后需同步到：
  - `Emergency02/agent/navigation`
  - `0803/Emergency/agent/navigation`
- 当前上述三份 Navigation 的 `map_slam.py`、`navigation_graph.py` 内容一致。

## 2. 当前导航地图与起点

Emergency02 使用：

- 路线：`config/scene.navigation.json`
- 图：`config/scene.navigation.graph.json`
- 固定障碍物场景：`config/scene.json`
- SLAM 地图：`/opt/config/custom/bslam/map/scan_map/v2-game-emergency.pcd`
- 逻辑起点：`startpoint = (0, 0, 0°)`
- startpoint 对应 SLAM 位姿：
  - `x = 0.34097319312584295 m`
  - `y = -1.3039173100960066 m`
  - `yaw = 8° = 0.13962634015954636 rad`

该 SLAM 起点由旧位姿 `(0.44, -1.29, 8°)` 沿逻辑 X 负方向平移 10 cm 得到。逻辑节点和连接没有因此改变。

## 3. 远端同步状态

- 2026-08-16 13:36 已再次将本地`Emergency02/`成功同步到`nvidia@192.168.41.2:/home/nvidia/Emergency02/`，包含第115次垃圾桶椭圆桶缘算法修改。
- 本次同步未使用`--delete`，排除了`output/`、`runtime/`、`analysis/`、Python缓存、zip和`.DS_Store`。
- `bin_filebox_grasp.py`、对应离线测试、`config/task.json`、`test_target_live_clean.py`和`emergency_zyt.py`的本地/远端SHA-256逐项一致。
- 后续同步建议：

```bash
rsync -avz --partial --timeout=30 \
  --exclude 'output/' \
  --exclude 'runtime/' \
  --exclude 'analysis/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.zip' \
  --exclude '.DS_Store' \
  /Users/mac/X-humanoid/Emergency02/ \
  nvidia@192.168.41.2:/home/nvidia/Emergency02/
```

- 不同步 `output/` 和 `runtime/`。程序重启时 `settings.prepare_route()` 会把 `config` 中的路线复制到 `runtime`。

## 4. 任务1当前关键参数与策略

任务1可从比赛总入口执行，也可独立运行：

```bash
cd /home/nvidia/Emergency02
source /home/nvidia/xos/setup.bash
source /home/nvidia/XARM/install/setup.bash
source /home/nvidia/eai/env/bin/activate
python emergency_task1.py
```

独立入口与总入口共用任务配置、路线、固定障碍物场景、相机标定、YOLO模型和抓取姿态。独立入口任务1不等待人工输入；总入口任务1恢复了观察、接近、抓取、回位、复核和进入任务2起点前的`input()`安全确认。两者都保留无限清障和任务2准入门控，不允许因失败人工跳过任务1。总入口进入`h270_task2_start`后继续任务2、任务3；任务之间不允许跳过，完成前不进入下一任务，但任务内步骤允许跳过（例如窗户1未完成仍可进入窗户2）。

任务1每轮仍扫描全部四类目标，不再因单类达到3次尝试或一次复核未检出而永久忽略该类。垃圾桶和文件框遇到碰撞时分别按`(0.50, 0.60, 0.70) m`和`(0.40, 0.50, 0.60, 0.70) m`从近到远尝试弯腰矩形接近。

在startpoint识别到稳定目标、但所有接近候选均碰撞时，碰撞检查只做路径规划，没有发送运动，因此机器人仍在startpoint，无需先调用`recoverState()`。备用观察位及其路径已由维护者确认为空，代码不再调用`checkBentCollision(1.5, -1.5, 45.0)`，而是直接执行`moveBy(1.5, -1.5, 45.0, correct=False)`并在新位置重识别。换位重识别仍无安全目标时才恢复startpoint，再进入h0路线，避免从偏移位置直接执行图路线。h0观察位的接近路线碰撞不套用“仍在startpoint”的假设。

全部候选均碰撞时不执行危险路径、不放行任务2。只有从startpoint开始并完成startpoint、h0_track_turn_2两处观察后均无4/5稳定命中的待处理目标，才判定清场并前往`h270_task2_start`。独立入口随后退出，总入口随后继续任务2。

相机读取异常不作为“无障碍物”，会进入自动重试。现有类别语义保持不变：椅子仅处理`front`，`back`和`lean`仍按当前策略忽略，因此这里的“清场”是指当前任务1算法认定需要处理的稳定目标全部消失。

两个入口的任务1均保留以下等待：每批通用任务1五帧YOLO前等待`3.0 s`，用于等待腰腿到位；启动稳定等待`1.0 s`；预设姿态腰腿动作后`1.0 s`；初始姿态安全回正后`4.0 s`；纸箱方向重拍间隔`0.20 s`；纸箱张爪后`1.0 s`；椅子朝向与任务1五帧识别间隔`0.10 s`；文件框/垃圾桶张爪后`1.0 s`、闭爪后`1.0 s`、腰腿提起后`3.0 s`、释放后`1.0 s`；转椅每只成功闭合的夹爪等待`1.0 s`。

只读实时检查脚本`test_target_live_clean.py`复用头部RGB-D、Obstacle YOLO、Navigation logical位姿和`ChannelRegion`规则，页面只给需要清理的目标绘制矩形框。`--node`默认使用`startpoint`作为地图—logical对齐参考；每帧均重新读取实测SLAM并换算当前logical位姿，因此机器人在脚本运行期间原地旋转或平移后不会继续使用旧位姿。通道外侧缓冲距离由`config/task.json`按类别统一配置：转椅和纸箱为`0.30 m`，垃圾桶和文件框为`0.15 m`，实时脚本与正式任务1共用。脚本不发送运动命令，默认页面端口为`8112`。

### 4.1 第一段接近

- `plan_standoff_move()` 对目标 `(x, y)` 做径向同比例缩短：

```text
scale = (distance - standoff) / distance
move_x = x * scale
move_y = y * scale
```

- 纸箱第一段：`STANDOFF_M_BY_TYPE["box"] = 0.45 m`。
- 正面椅子：`0.80 m`；背面和 lean 当前直接忽略，不执行抓取。
- 垃圾桶与文件框已经拆成两组可独立调节的参数：

```python
BENT_APPROACH_STANDOFFS_M_BY_TYPE = {
    "bin": (0.50, 0.60, 0.70),
    "filebox": (0.50, 0.60, 0.70),
}
```

- bin/filebox 每个候选距离均使用弯腰矩形碰撞规划，首个安全候选生效，抓取前不旋转。
- 从 `h0_track_turn_2` 接近时，横向位移 `|y|` 最大截断为 `0.25 m`；此时不再严格保持 XY 比例。
- `DEFAULT_STANDOFF_M` 仍在代码中作为未知类别/缺失字段兜底，但当前 chair/box/bin/filebox 均有专用参数，实际不由它控制。

### 4.2 纸箱二次微调

- 纸箱第一次接近后做近距离五帧复识别。
- 复识别成功：使用新识别结果，最终保持 `base x = 0.45 m`，Y 直接对中，不旋转。
- 复识别不稳定：不再直接回 startpoint。使用：

```text
当前估计坐标 = 第一次识别的 base 坐标 - 第一段累计实际平移
```

  然后按同一 `x=0.45 m`、Y 对中规则执行第二段移动。
- 第二段仍经过 2 cm 死区和固定障碍碰撞检查。
- 相机 RGB-D 读取异常仍视为真实故障，不使用旧识别掩盖。

### 4.3 纸箱横竖判断

- 二次微调后最多采集三帧头部 RGB-D，用 YOLO 纸箱框和水平物理跨度判断方向。
- `span >= 0.40 m`：横箱；否则竖箱。
- 连续三帧均未获得有效方向结果时，当前代码按纸箱外层尝试次数回退：
  - 第1、2次默认`horizontal`，优先横抱宽边。
  - 第3次起默认`vertical`，改为竖抱窄边。
- YOLO检出但`valid_row_count == 0`同样视为方向识别失败，会继续采集下一帧。
- 得到有效深度跨度时始终服从实际横竖判断，不受失败回退策略影响。随后抱取流程仍会：
  1. 切纸箱预备姿态。
  2. 无条件切横抱姿态。
  3. 仅当结果为 `vertical` 时再切竖抱姿态。
- “YOLO 未检出时改用图像中心 80% ROI 做深度判断”目前只讨论过，尚未实现，是待办事项。
- 没有有效深度跨度时底层检测器当前会判为 `vertical`；若实现中心 ROI 回退，应由上层在多帧都无有效跨度时统一回退 `horizontal`。

### 4.4 纸箱姿态执行顺序

- 当前纸箱预备、横抱、竖抱和释放前预备均调用 `_execute_object_pose()`：
  1. MoveIt 先移动双臂。
  2. 用户按 Enter。
  3. 再控制腰腿。
- 之前“腰腿先动并固定 sleep 3 秒”的纸箱专用逻辑已删除。
- 竖纸箱姿态 `poses_box/应急_纸箱_2竖箱子.json` 的右臂第二关节当前为 `-3° = -0.05235987755982989 rad`。

### 4.5 垃圾桶桶沿识别

- 文件：`bin_filebox_grasp.py`
- 原逻辑依赖白色桶沿；现在保留白色 mask，并新增浅绿色 mask：

```text
浅绿色 HSV：H=55～95，S=30～150，V=90～255
```

- 材质 mask 为“白色 ∪ 浅绿色”，之后仍使用原有：
  - 圆弧几何位置
  - 非水平切线方向
  - 深度范围与近深度簇
  - 深度梯度和局部密度评分
- `green_bin` 四张头部相机图像中人工采样的 21 个浅绿色像素全部落入该范围。
- 这些样本来自头部相机，仍需使用腕部相机现场验证色偏。
- 离线单测：`python -m unittest test_bin_filebox_grasp_offline`，当前 12 项通过。

### 4.6 椅子

- 专用模型：`yolo/models/chair_yolo26l_200e_best.pt`，类别为 front/back/lean。
- 当前只处理 front；back 与 lean 均直接忽略。
- front 两段接近均使用 `0.80 m`。
- 正面椅子抓取前腕部 TCP 预移动：前方 5 cm，左右分别 5 cm；背面参数保留为 0，但当前背面不会进入抓取。
- 一只手抓取成功也继续拖行；两只手都失败才返回失败。

## 5. Navigation 最新比赛容错修改

权威文件：

- `/Users/mac/X-humanoid/agent/navigation/map_slam.py`
- `/Users/mac/X-humanoid/agent/navigation/navigation_graph.py`
- 使用文档：`/Users/mac/X-humanoid/agent/navigation_api.md`

当前行为：

- `POSE_RECOVERY_ROUNDS = 4`，由原 2 轮增加为 4 轮。
- `moveBy()` 或 `turnTo()` 调用前不是 `AT_NODE`：
  - 读取当前 SLAM 实测位姿。
  - 无节点容差限制地采用最近逻辑节点作为语义锚点。
  - 记录 `semantic_node_adopted_without_tolerance` warning 后继续。
- 当前 SLAM 地图名称与配置不同：只打印 warning，继续使用配置中的逻辑↔SLAM标定，不再抛地图名不匹配错误。
- 修正前误差超过旧范围：不再拒绝，直接尝试反馈修正。
- 四轮后仍未收敛：记录 `move_by_pose_correction_not_converged` warning，不因位姿误差抛错。
- `recoverState()` 最终到位检查不通过：记录 warning，保留语义节点并继续。
- 仍会上抛的真实异常：底盘通信失败、SLAM地图/位姿查询失败、缺少节点、非法参数及其他非容错范围的底层异常。

验证：

- Navigation 针对性离线测试 12 项全部通过。
- 全部 Navigation 工具测试 16 项中 15 项通过。
- 唯一既存失败：根目录权威 scene 与 route 的任务3点位不一致；与上述 Navigation 修改无关，未顺手修改。

## 6. 当前本地还包含的现场参数

- 窗口2完成开窗并松开夹爪后，双臂恢复到`应急_初始姿态.json`中的关节姿态，腰腿保持当前位置不变。
- 窗口2识别前底盘动作当前为：

```python
navigation.moveBy(0.08, 0.0, 0.0, correct=True)
```

- 任务3右侧抓取前的：

```python
navigation.moveBy(-0.10, 0.10, 0, correct=True)
```

  当前已被注释，不执行。
- 窗口2 TCP Z 保留距离：`WINDOW2_HANDLE_STANDOFF_TCP_Z_M = 0.19 m`。
- Window1 接近姿态使用：
  - `config/window1/new_step0.json`
  - `config/window1/new_step1.json`
  - `config/window1/new_step2.json`

## 7. 当前待办与风险

1. 将纸箱方向判断增加“YOLO 未检出时使用中心 80% ROI”的深度回退；多帧仍无有效跨度时默认横抱。
2. 用左右腕部 RGB-D 实拍验证浅绿色垃圾桶 mask；必要时只调整 HSV，不先放宽几何和深度约束。
3. 重新连接 41.2 后先做 rsync dry-run，再同步本地 Emergency02；不要覆盖远端 `output/`。
4. 同步后必须重启 `emergency_zyt.py`，使 `settings.prepare_route()` 重新生成 runtime 路线。
5. Navigation 现在对地图名、语义节点和未收敛误差更宽容，比赛部署更快，但必须由现场人员确认实际地图与路线没有装错；代码不会再通过地图名称阻止运动。
6. 未经用户明确授权，不运行会向真实机器人发送运动命令的脚本；Codex只做静态、离线和fake controller测试。

## 8. 比赛机运行方式

```bash
cd /home/nvidia/Emergency02

source /home/nvidia/xos/setup.bash
source /home/nvidia/XARM/install/setup.bash
source /home/nvidia/eai/env/bin/activate

python emergency_zyt.py
```

## 9. 最近完成但尚未确认远端已包含的修改

- 第94次：纸箱近距离五帧复识别失败时，按第一次识别与第一段实际平移估算坐标继续第二段移动。
- 第95次：垃圾桶和文件框停靠距离拆为两组独立参数。
- 第96次：垃圾桶材质 mask 增加浅绿色范围及对应离线测试。
- 第97次：Navigation 去除比赛不利的地图名/`AT_NODE`硬阻塞，修正改为四轮且未收敛软继续。
- 第98次：纸箱前两次方向识别失败横抱宽边，第三次起识别失败改为竖抱窄边。
