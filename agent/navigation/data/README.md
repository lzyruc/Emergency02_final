# 导航数据目录

本目录只保存路线/图数据格式及运行产物：

- `motion_log.py`：record JSON 的格式、校验和原子读写。
- `logs/`：路线 JSON、同名 graph JSON 和 replay 日志。

运行时导航核心位于上一级 `navigation_graph.py` 和 `map_slam.py`；命令行入口位于
`../tools/`。完整使用说明见 [`../README.md`](../README.md)。
