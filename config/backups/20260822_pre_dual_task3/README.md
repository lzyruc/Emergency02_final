# 任务三双臂导航修改前备份

此目录保存新增 `h225_task3_pick` 双臂任务三分支前的测试场地导航文件。

- `emergency.navigation.json` SHA-256：`7932c2df3a280291aeafa7bacb8e62cceb8b26e9c1f79835b596efccda2c7699`
- `emergency.navigation.graph.json` SHA-256：`3bc7ba1ca72d18ca0ac15c39c910d2eb221736c494d63028d1dd76e69a7f0545`

在 `Emergency02` 目录下回退：

```bash
cp -p config/backups/20260822_pre_dual_task3/emergency.navigation.json config/emergency.navigation.json
cp -p config/backups/20260822_pre_dual_task3/emergency.navigation.graph.json config/emergency.navigation.graph.json
```

该备份只对应测试版 `emergency` 路线；比赛版 `scene` 路线未参与本次修改。
