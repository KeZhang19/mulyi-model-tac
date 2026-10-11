# 共享运行库索引

这些入口是相对符号链接，指向仓库内原实现，不复制、不改变Python包结构：

| 入口 | 共享职责 |
| --- | --- |
| `force_map/` | 任务压力传感器、校准参数加载、reference fitting与trace契约 |
| `tactile_representation/` | collection数据契约、direct_calibration曲面坐标、网络输入及训练加载器 |
| `tactile_runtime/` | integrate中的深度／HydroShear／RGB运行适配与第三方依赖 |
| `tacmap/` | 几何深度、传感器映射与压头等共享资源 |
| `visualize_rl_tactile_obs.py` | 被三模态采集器复用的RL触觉可视化与压头控制辅助 |

共享手模型另见 `../assets/revo21_right_touch`。真机SDK在 `../brainco-revo3-sdk/`。
这些库还服务训练／回放，保持原导入路径；calibration不是可以脱离仓库独立运行的完整软件包。
