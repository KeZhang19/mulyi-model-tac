# D 型插销修复版：双卡 8192 环境从头重启

2026-09-13，北京时间 02:09 验收。已按用户要求同步本地修复，停止旧插销任务，并在 `dsw-8prvxfz53nbu67qzsu` 的 A100 GPU 2、3 上启动新训练。GPU 0、1 上的 Repose 训练进程保留并持续运行。

## 实际保存配置

| 项目 | 值 |
| --- | --- |
| 任务 ID | `BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0` |
| 修复后的任务契约 | v3；旧任务称 v2，注册 ID 保持不变 |
| 分布式 | 2 个进程，NCCL，每卡 4096 环境，总计 8192 |
| γ / GAE λ | 0.999 / 0.95 |
| 回合 | 15 秒，30 Hz，最多 450 个控制步 |
| 每轮采样 | 每环境 32 步，双卡合计 262,144 步 |
| 策略加载 | `agent.resume=false`，从头训练 |
| 模态还原 | `env.tactile_policy_enabled=false`；不加载还原模型，不生成 RGB/Depth/marker 或执行编码器推理 |
| 观测 | 669 维，保留压力、接触和状态观测 |
| 动作尺度 | 机械臂 0.1，手指 0.02 |
| 初始学习率 / 探索标准差 | 3e-5 / 0.15 |
| 训练长度 / 保存间隔 | 15,000 轮 / 25 轮 |

## 验收结果

- 服务器插销回归检查 **127 项通过**。
- 21 个同步文件哈希一致，11 个已有运行依赖与本地一致；共享训练入口和其他任务代码未覆盖。
- 前三次 PPO 更新均验证跨卡梯度均值正确，每次 20 次梯度归约，更新后参数最大差异 **0**。
- 验收到日志第 3 轮（从 0 开始，共 4 次更新），成功保存 `model_0.pt`，12,257,983 字节，权重及归一化张量均为有限数。
- 首轮包含首次物理场景更新，耗时 200.39 秒；随后三轮分别 **26.30、26.47、26.95 秒**，平均 **26.57 秒/轮**。这是启动后的短时实测，长期性能仍需观察。
- GPU 2、3 分别占用 18,149、18,125 MiB，约 **17.7 GiB/卡**，未出现 OOM 或训练异常。
- 原插销进程 `111081 / 111086 / 111091 / 111092` 已停止，旧 `model_200.pt` 及全部历史日志保留。Repose 的 `112330 / 112342 / 112343` 仍运行。

这次大规模冷启动较慢：每个进程的场景创建约 70 秒，物理/传感器初始化约 550–556 秒，随后还逐环境处理机器人实例和接触材质。运行栈采样证实初始化持续推进。01:45:16 启动，约 02:04 进入首轮采样；这些启动开销不代表之后每轮 26–27 秒的耗时。训练刚从头开始，本次验收不证明插入成功率已经提高。

## 运行与恢复位置

服务器项目根目录：`/home/admin/workspace/xinyu/mulyi-model-tac`。

新运行相对目录：

```text
logs/rsl_rl/dexsuite_revo3_insert_d_peg_v3/20260912T174516Z_from_scratch_a100_gpu23_8192_tactile_off_gamma0999_ep15
```

该目录保存 `console.log`、`launch.json`、`launch_command.txt`、`params/agent.yaml`、`params/env.yaml`、`tactile_policy_contract.json`、`distributed_validation.json`、`d_peg_training.jsonl` 和 `deployment_verification.json`。时间戳目录使用 UTC。启动命令已显式写入每卡 4096 环境和关闭模型加载的覆盖项，不依赖通用双卡脚本的环境数默认值。

新启动器 PID 为 `123699`，torchrun 为 `123704`，两卡 worker 为 `123711 / 123712`。原始远端文件备份在 `/home/admin/workspace/xinyu/.codex-sync-backups/d_peg_v3_20260912T174321Z`，其中包含同步前哈希清单与旧任务停止记录。

本地证据：[部署与配置验收](../logs/diagnostics/d_peg_v3_deploy_2026-09-13/server_snapshot/deployment_verification.json)、[双卡同步验收](../logs/diagnostics/d_peg_v3_deploy_2026-09-13/server_snapshot/distributed_validation.json)、[服务器回归输出](../logs/diagnostics/d_peg_v3_deploy_2026-09-13/remote_cpu_tests.log)、[原任务停止记录](../logs/diagnostics/d_peg_v3_deploy_2026-09-13/old_training_stopped.json)。修复依据及本地实验见 [本地修复报告](d_peg_v3_local_fix_2026-09-13.md)。

同步前刷新并复核本地 GitNexus 图。配置函数、还原观察类和奖励状态类的已解析调用风险为 LOW，影响 D 型插销配置及 PLAY 入口；动态启动契约的 UNKNOWN 通过实际 `EventTerm` 注册和运行验证补充核实。没有提交代码。
