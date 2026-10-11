# 仿真数据采集与缓存

| 入口 | 用途 | 运行边界 |
| --- | --- | --- |
| `collect_index_cross_modal.py` | 食指RGB／Depth／Marker同步采集、NPZ分片与多worker合并 | 依赖Isaac及共享可视化器；正常采集会启动仿真并写数据 |
| `prepare_mmap_cache.py` | 将已完成manifest的NPZ数据转换为mmap缓存 | 会写缓存；需匹配训练schema |
| `run_rl_ball_probe_pressure_calib.py` | RL场景球探针压力片扫描 | 依赖Isaac／GPU；不是机械台或真机控制入口，不把其help当无副作用检查 |

前两者默认路径统一为 `calibration/datasets/revo3_index_cross_modal_v1`（采集）与
`calibration/datasets/revo3_index_sweep_parallel_v1`（缓存历史示例）；两者名称不同，
缓存时应显式传入本次真实数据集路径，例如：

```bash
python calibration/collection/prepare_mmap_cache.py --dataset calibration/datasets/MY_COMPLETE_DATASET
```

该示例需要对应数据集实际存在；不是整理过程中已经生成的数据。
旧脚本路径保留符号链接；训练器本身保持在 `scripts/tactile_representation/`。
几何标定仿真输出与这里的训练NPZ不直接兼容，转换条件见[主说明第13节](../机械标定台仿真搭建_README.md#13-数据保存转换与使用)。
