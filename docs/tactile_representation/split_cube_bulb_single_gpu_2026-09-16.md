# GPU 0 基础方块 / GPU 1 Custom 灯泡独立训练

服务器：`dsw-715zhrv7l6bux2ldbl`，hostname：`dsw-949505-64bd8d4dc8-wpvz4`。
项目目录：`/home/admin/workspace/xinyu/mulyi-model-tac`。

按用户要求，于北京时间 2026-09-16 16:16 停止原双卡灯泡任务。
原 launcher、torchrun 和两个 worker（590594、590600、590606、590607）均已退出；
`model_11600.pt` 及旧运行日志保留。随后按顺序启动两个独立单卡 PPO 任务，均从零训练。

| 配置 | GPU 0 | GPU 1 |
| --- | --- | --- |
| 任务 | `BrainCo-Direct-Revo3-Repose-Cube-v0` | `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0` |
| 版本 | 基础方块任务 | 已修复碰撞栈的 Custom 灯泡 |
| 环境数 | 8192（基础默认值） | 4096 |
| 最大迭代 | 10000 | 15000 |
| gamma / lambda | 0.99 / 0.95 | 0.999 / 0.95 |
| 回合 / 控制频率 | 10 秒 / 30 Hz | 20 秒 / 60 Hz |
| 每环境每轮步数 | 16 | 32 |
| seed / 恢复模型 | 42 / 不恢复 | 42 / 不恢复 |
| 保存间隔 | 250 轮（基础默认值） | 25 轮 |
| 模态还原 | 基础状态观测 | 关闭，保留压力、接触与状态观测 |
| 碰撞栈 | 基础默认值 | 2147483648 字节（2 GiB） |
| 启动时间（北京时间） | 16:20:18 | 16:21:01 |
| 训练 Python PID | 2271483 | 2271750 |
| 监督进程 PID | 2271477 | 2271743 |

各进程分别设置 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，
进程内部均使用逻辑 `cuda:0`。没有传入 `--distributed`，两个任务不共享策略或梯度。
灯泡保留逐指接触奖励、动作惩罚以及 10 项手指网格共享设置。

两项运行使用服务器原有 RSL-RL checkout。启动前核对了 19 个源文件；其中训练入口
与本地的唯一差异是插销任务识别范围，已审查后保留服务器版本，其余全部哈希一致。
没有修改基础方块、灯泡奖励或共享训练入口源码。

## 模型和日志位置

模型路径相对于服务器项目根目录：

```text
logs/rsl_rl/brainco_hand/20260916T081801Z_cube_base_gpu0_scratch_seed42
logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/20260916T081801Z_bulb_custom_gpu1_scratch_seed42
```

运行控制目录：

```text
remote_train_logs/split_cube_bulb_20260916/20260916T081801Z_cube_base_gpu0_scratch_seed42
remote_train_logs/split_cube_bulb_20260916/20260916T081801Z_bulb_custom_gpu1_scratch_seed42
```

每项独立保存 `train.log`、`state.json`、`launch_manifest.json` 和 `launch_receipt.json`。
监督进程记录进度，并在 PhysX/CUDA 错误、非有限损失/奖励或训练停止推进时终止
自己启动的进程树。启动期允许 3600 秒初始化，进入训练后无更新超时为 600 秒。
旧训练停止记录保存在控制根目录的 `previous_training_stop.json`。

本地验证与启动记录：`logs/diagnostics/split_cube_bulb_20260916/`。

## 启动验收

北京时间 16:31:49 验收通过。GPU 0 已完成 25 次更新（日志迭代 24），GPU 1
已完成 8 次更新（日志迭代 7），两个监督状态均为 running。

- 两项都从零训练，实际 `agent.resume=false`，未加载旧 RL 检查点。
- 各自已保存 `model_0.pt`，方块 92 个、灯泡 76 个检查点张量全部有限，涵盖模型、
  观测归一化及优化器状态。
- 保存配置确认环境数、gamma、回合时长与上表一致；灯泡碰撞栈实际为 2 GiB。
- CUDA 进程和环境变量均确认方块绑定物理 GPU 0，灯泡绑定物理 GPU 1。
- 本次两项运行的完整启动日志均未发现 PhysX/CUDA 错误、碰撞栈溢出或接触丢弃。
- 验收快照的单轮采样加学习耗时分别为约 23.9 秒、11.1 秒；GPU 显存约
  13.9 GiB、21.2 GiB。这是启动阶段测量，不代表长期速度或策略收敛情况。

灯泡从启动到首轮采样经历约 9 分钟初始化，期间堆栈确认正在创建传感器刚体视图，
之后正常进入训练。监督进程将继续监测各自任务，不会在会话结束时退出。

服务器总体验证记录：`remote_train_logs/split_cube_bulb_20260916/validation.json`。
