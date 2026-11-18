# Navigation tools 清单

本文按当前开发目录 `agent/navigation/tools` 整理。所有带 `--execute` 的工具默认只能
dry-run；没有明确实机授权时不得添加该参数。

## 正式保留

| 文件 | 功能与输入输出 | 建议 |
| --- | --- | --- |
| `convert_scene_navigation.py` | 校验 graphGUI `scene.json`，生成同目录或指定位置的 `*.navigation.json` 与 `*.navigation.graph.json`。不连接底盘。 | 保留，scene 到运行图的唯一转换入口。 |
| `demo_navigation_graph.py` | 交互选择初始节点和目标节点；dry-run 只规划，`--execute` 时读取实时 map/pose 建立锚点并调用 `NavigationGraph`。 | 保留，通用人工验收入口。 |
| `edit_navigation.py` | 校验并列出 graphGUI 源 scene，或按节点名增删一条无向连接。默认输出新 scene，`--in-place` 才覆盖。 | 保留，和 graphGUI 共用 `robots/connections` 契约。 |
| `rebuild_navigation_graph.py` | 从旧式顺序录制 route JSON 原子重建同名 `*.graph.json`，不连接底盘；拒绝 scene 派生 route。 | 仅保留给旧式顺序 route；scene 修改后必须重新运行 convert。 |
| `auto_test_navigation_graph.py` | 商超固定多点验收；包含场景序列、关键点校正选择、识别点先转后移和货架横移配置。 | 保留但标记为 shop 验收脚本，不是通用 API。 |
| `auto_test_emergency.py` | 向通用自动验收入口传入应急固定序列、校正点和禁止路径压缩配置。 | 保留，emergency 验收入口。 |

## 记录与回放工具

| 文件 | 功能与约束 | 建议 |
| --- | --- | --- |
| `record_navigation.py` | 交互发送/记录 `move_by` 路线，记录逻辑位姿、实时 map pose、动作结果和失败恢复；包含商超尺寸边界。 | 仅用于旧式现场打点。边界是场景特定常量，发布前应移到部署配置，否则不进入通用工具集。 |
| `replay_navigation.py` | 校验、预览或逐条重放 record JSON；支持动作区间、速度覆盖、失败诊断和独立 replay 日志。 | 与 record 成对保留；如果 v1.0 只发布 graphGUI 流程，可整体移入 `legacy_tools` 或不发布。 |

## 专项测试与便捷入口

| 文件 | 功能与问题 | 建议 |
| --- | --- | --- |
| `test_move_by_recovery.py` | 在 shop 的 `hub_center` 做三轮人工微调，再测试无参数 `recoverState()` 返回保存 pose。 | 专项实机回归，不是单元测试；改名为 `manual_test_move_by_recovery.py` 后放验收目录。 |
| `test_navigation_graph.py` | 对一份旧 record 强制创建 `emergency.start <-> emergency.point2`，再规划/执行到 `emergency.end`。默认路径写死为 `/home/nvidia/...`，且会修改 graph 文件。 | 功能和数据均过时，建议删除，不进入发布。 |
| `run_emergency_navigation_test.sh` | 用固定参数启动 `demo_navigation_graph.py`，目标是 logs 下的 emergency scene。 | 与 `auto_test_emergency.py` 职责重叠且路径布局陈旧，建议删除或改成唯一、目录无关的发布入口，不能两者并存。 |
| `__init__.py` | 标记工具包。 | 保留。 |

## graphGUI

| 文件 | 功能与状态 | 建议 |
| --- | --- | --- |
| `graphGUI/shop/navigation_visualization_system.html` | 商超逻辑地图的浏览器编辑器；导入/导出 `robots`、`connections`、障碍物与场景显示参数。 | 当前主编辑器，保留。 |
| `graphGUI/shop/scene_shop.json` | 商超源 scene，是派生导航文件的数据源。 | 保留并纳入版本/哈希管理。 |
| `graphGUI/shop/scene_shop.navigation.json` | 从源 scene 生成的运行路线。 | 派生文件；开发期可留作回归样本，发布时由确定流程生成或作为锁定资源二选一。 |
| `graphGUI/shop/scene_shop.navigation.graph.json` | 从源 scene 生成的运行拓扑。 | 同上，不应手工编辑。 |
| `graphGUI/hotel/navigation4hotel.html` | 酒店旧坐标系编辑器。 | 与统一 `+x 前、+y 左` 不一致，未转换前标记 unsupported；不进入 v1.0。 |
| `graphGUI/hotel/scene_hotel.json` | 酒店旧 scene。 | 仅参考，不进入 v1.0。 |
| `graphGUI/emergency/scene_emergency.json` | 旧应急 GUI scene，仍有默认中文名，当前不能通过正式转换。 | 与已实测 emergency 数据不一致，建议移到 `archive` 或删除，避免误用。 |
| `graphGUI/README.md` | GUI 坐标、命名、编辑、转换和实测前检查规则。 | 保留并作为地图制作说明。 |
| `.DS_Store`（两处） | macOS Finder 元数据。 | 删除并加入 `.gitignore`。 |

旧 `visualize_navigation.py` 和生成的 `*.graph.svg` 已移除。原因不是单纯减少文件，而是它直接
修改派生 graph，和 graphGUI 源 scene 形成两个真相源；下一次 scene 转换会覆盖其编辑结果。
