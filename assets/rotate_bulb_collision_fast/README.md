# 独立灯泡碰撞资产

20260927 目录由 `scripts/diagnostics/build_bulb_collision_fast.py` 从当前 threaded
灯泡 USD 生成。原 USD 保留，候选不替换共享任务默认资产。最终选择和训练结果见
`docs/rotate_bulb_collision_fast_2026-09-27.md`（实验完成后写入）。

| 副本 | 外螺纹 | 内螺纹 | 两片 Shell |
| --- | --- | --- | --- |
| baseline.usdc | 21814 面，SDF 512 | 5199 面，trimesh | 各 3199 面，SDF 256 |
| threads_low.usdc | 6037 面，SDF 128 | 4613 面，trimesh | 保持基准 |
| shell_hulls.usdc | 保持基准 | 保持基准 | 每片一个 convexHull |
| combined.usdc | 同 threads_low | 同 threads_low | 同 shell_hulls |

螺纹网格使用 1 mm 空间聚类、去退化三角形及重复面，是明确允许误差的碰撞近似。
内螺纹保留凹形开口。Shell 的可视及触觉顶点保留，只改变 PhysX 碰撞近似；
每片单凸包会填平其凹处，接触位置可以改变。移除 Shell 的 SDF API 后，文件中残留的
SDF resolution 数值不再参与其凸包碰撞计算。

原 slide、turn/mimic 关节和运行时生成的 attachment 不变；质量、惯量、材质、
摩擦、控制、观测、奖励和成功条件均不在本次资产修改范围。
Robot 自碰撞开关由独立 `collision_fast` 配置控制，不写入这些 Lamp 文件。

`robot_palm_hull_v3.usdc` 是独立 Robot 副本：掌部 28 个凸形状替换为
67 顶点/130 三角面的单凸包；其余碰撞、可视、质量、惯量和关节属性保留。
被替换的 28 个 CollisionAPI/MeshCollisionAPI 已移除，避免通用 startup
属性回调重新启用。对应 JSON 保存源层哈希、替换路径和代理几何信息。

`manifest.json` 保存源资产 SHA256、各副本 SHA256 和面数。可运行
`scripts/diagnostics/validate_bulb_collision_assets.py` 检查允许的属性差异及网格有效索引。
需要带 USD/PhysX 插件的 Isaac Python；生成器会启动 Isaac Sim，验证器只读取 USD。
