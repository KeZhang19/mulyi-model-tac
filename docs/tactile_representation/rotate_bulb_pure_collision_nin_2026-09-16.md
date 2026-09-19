# 灯泡纯碰撞训练对照（2026-09-16）

目标服务器：`dsw-nin473ap6o7410zyra`，实际主机名 `dsw-912204-674b5c47c9-25k4h`。
代码目录：`/home/admin/workspace/xinyu/mulyi-model-tac`。该目录在两台 DSW 上共享；正在运行的旧训练已经加载旧模块，保留原物理模式。

## 同步与启动

- 已同步用户新增的 `env.bulb_pure_collision` 及依赖代码、说明和验证文件，共 10 个文件，逐一核对 SHA-256。
- 旧文件备份：`/home/admin/workspace/xinyu/.codex-sync-backups/pure_collision_nin_20260916T090415Z`。
- 目标服务器实际只有一张 RTX 4090，启动前显存占用 1 MiB，没有正在运行的训练。
- 正式启动时间：北京时间 17:06:09。任务：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`。
- 显式设置 `env.bulb_pure_collision=true`，GPU 0、4096 环境、seed 42、从零训练、15000 次 PPO 更新，每 25 次保存。每次采样 32 步，episode 20 秒，gamma 0.999。
- 与原灯泡训练一致：关闭策略触觉编码、保留接触相关奖励、动作惩罚 -0.0001 / -0.0002、接触奖励 0.1、force_scale 0.5、碰撞栈 2147483648 字节，以及 10 项手指网格共享设置。
- 启动前验证 27 个源码/资产哈希，保存本次源码快照。完整命令在 `launch_manifest.json`，由独立 supervisor 托管。

运行名：`20260916T090502Z_bulb_pure_collision_gpu0_scratch_seed42`。

相对代码根目录的输出路径：

```text
logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/20260916T090502Z_bulb_pure_collision_gpu0_scratch_seed42
remote_train_logs/pure_collision_nin_20260916/20260916T090502Z_bulb_pure_collision_gpu0_scratch_seed42/train.log
remote_train_logs/pure_collision_nin_20260916/20260916T090502Z_bulb_pure_collision_gpu0_scratch_seed42/state.json
```

查看状态：

```bash
ssh dsw-nin473ap6o7410zyra 'cd /home/admin/workspace/xinyu/mulyi-model-tac && python3 remote_train_logs/pure_collision_nin_20260916/status.py'
```

## 对照设置

旧模式对照继续使用 `dsw-715zhrv7l6bux2ldbl` GPU 1 的 `20260916T081801Z_bulb_custom_gpu1_scratch_seed42`。它在本参数加入前启动，具有螺旋关节和程序控制的 attachment 解锁。

两组正式训练的环境数、随机种子、PPO、奖励、初始三圈深度和资源设置保持一致。比较使用相同迭代编号。新开关同时移除轴线/螺距约束、关节摩擦和程序解锁，并改用位姿测量实际进度；因此这是两种物理模式的对照，不能单独归因到某一条解锁语句。

## 服务器真实物理验证

部署后在目标 4090 上分别运行 2 环境的 `true` 与 `false`，测试代码为已有的 `validate_modes.py`，没有修改物理实现。纯碰撞通过 19 项检查，旧模式通过 17 项检查；两组日志均无 PhysX/CUDA 错误、碰撞栈溢出或接触丢弃。

- 纯碰撞模式活动关节和 articulation 数量为 0，内外螺纹碰撞仍启用；释放检测只记录阶段，不改写刚体状态。
- 旧模式仍有 4 个活动关节，包括灯泡 attachment。
- 两组完整重置、部分重置、正常 step 和超时重置均通过。

移开机械手，向第二个环境的灯泡施加相同 0.02 N·m 轴向扭矩，结果如下。这是力学探针，不是策略训练成功率：

| 指标 | 旧关节模式 | 纯碰撞模式 |
|---|---:|---:|
| 首次记录 released | 0.85 秒 | 1.30 秒 |
| 有效旋转圈数 | 3.000 | 0.619 |
| 轴向上升 | 18.005 mm | 3.713 mm |
| 侧向位移 | 0 mm | 约 255.5 mm |

纯碰撞的有效圈数由旋转和轴向位移共同限制；此探针实际累计转角约 2.579 圈，但位移不足三圈。它发生了侧向偏移，不能把 released 解释为稳定地完整拧出三圈。

轨迹还显示一个统计边界：1.00 秒时灯泡轴线侧向偏移约 61.3 mm、下降约 30.4 mm，`released` 仍为 false；1.067 秒时侧向偏移超过 104 mm，仍未记录释放。当前判据只看整个螺纹包围盒是否越过灯座口的轴向平面，不能覆盖所有侧向滑脱。`bulb_dropped` 又要求 released 后接触桌面，因此纯碰撞组的掉落日志也不能单独视为所有滑脱的统计。本轮保留用户实现，记录问题以供后续判断，未改变判据。

旧模式在该探针中能正常完成三圈并解锁，说明解锁机制并非在所有情况下都失效；仍不能据此排除策略接触状态下的问题。正式训练需要结合有效圈数、释放后掉落、持握与最终成功率判断。

## 检查记录

本地本轮相关测试：30 通过、1 跳过、1 既有失败。失败为 `test_asset_swap_retains_original_actions_observation_definitions_and_rewards` 对 `RotateBulbStableSuccess` 的身份一致性要求；用户上一轮已在修改前代码中复现。新纯碰撞测量与碰撞栈测试通过，本轮没有修改奖励、成功条件或测试断言。

GitNexus 仓库 `mulyi-model-tac`，本地工作区 `/home/liuxinyu/workspace/mulyi-model-tac`。连接模式/碰撞重置直接调用链风险 LOW；Gym 注册、spawn 和 EventTerm 的 UNKNOWN 已用文本注册点确认。已有代码变更涉及 22 个已跟踪符号、1 条验证执行流程，整体 MEDIUM。未提交代码。

本地证据目录：`logs/diagnostics/pure_collision_nin_20260916/`。包含同步清单、备份记录、启动命令、源码快照哈希、两种模式的服务器物理验证、训练检查脚本。

## 正式训练验收与初期比较

北京时间 17:17:04 验收通过。已完成 13 次 PPO 更新（日志 iteration 12），GPU 0 上的 Python PID 为 1700，launcher PID 1695，supervisor PID 1694。显存约 21107 MiB，GPU 利用率约 91%，每轮约 10 秒。

- 实际保存的配置确认 `bulb_pure_collision=true`，灯泡使用独立刚体结构，无 lamp actuators。
- 与旧组保存配置逐项比较，差异仅为物理模式带来的 lamp 结构、模式开关和运行目录/名称；PPO、奖励、重置深度等其他配置一致。
- `model_0.pt` 已生成，模型和优化器共 76 个张量通过有限值检查，从零启动且没有加载旧 checkpoint。
- 全日志没有 PhysX/CUDA 错误、接触丢弃或碰撞栈溢出。日志存在与旧场景相同的视觉引用/路径警告，未阻止训练。

按相同日志 iteration 3–12 对齐，以下为每轮日志值的算术平均，尚非收敛结果，也不是按 episode 数加权的总体统计：

| 指标 | 旧关节模式 | 纯碰撞模式 |
|---|---:|---:|
| 任务成功率 | 0% | 0% |
| episode 峰值有效圈数 | 0.13526 | 0.00495 |
| episode 最终有效圈数 | 0.08968 | 0.00160 |
| 达到一圈的比例 | 0.712% | 0% |
| released 日志值 | 0 | 0.01096 |
| bulb_dropped 终止日志值 | 0 | 0.01332 |
| Mean reward | 0.062 | -0.118 |

初期纯碰撞组出现释放/掉落记录，但有效拧动进度低于旧组，不能将释放记录增多视为改进。结合力学探针，当前没有证据认定旧解锁设置就是 0% 成功率的原因；新模式的侧向滑脱与轴向释放判据不一致，是已观察到的具体问题。两组仍在训练，需要更长的同轮次比较再判断学习效果。

验收 JSON：服务器 `remote_train_logs/pure_collision_nin_20260916/validation.json`；完整逐轮指标：同目录 `iteration_metrics.json`。本地证据目录也保存了副本。
