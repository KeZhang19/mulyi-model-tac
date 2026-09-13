# Rotate-Bulb 双卡续训记录（2026-09-12）

服务器：`ssh dsw-nin473ap6o7410zyra`。
项目：`/home/admin/workspace/xinyu/mulyi-model-tac`。
任务：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0`。

北京时间 10:51:33 启动；10:57:14 验证时已完成迭代 652，训练继续运行。

## 恢复与配置

- 从指定旧运行的最新检查点恢复模型和优化器：
  `logs/rsl_rl/dexsuite_revo3_rotate_bulb/2026-09-11_14-56-37_rotate_bulb_20260911T145626Z_normalizer_fixed_env512/model_650.pt`。
- 两个 torchrun worker，GPU 0 和 GPU 1，NCCL，world_size=2；每卡 512 环境，共 1024。
- 每进程碰撞栈为 `2**28` 字节（256 MiB）。初始旋入范围仍为 0.25～0.5 圈。
- 沿用归一化计数修复、有限动作检查和原触觉编码器权重。
- 按 RSL-RL 的恢复语义从保存的迭代编号 650 开始，追加 14350 轮，最后迭代编号 14999；每 25 轮保存。
- `log_all_ranks=False`，仅 rank 0 写 TensorBoard 和训练进度；两卡仍参与全部梯度更新。
- 显式指定两卡共同的输出目录；配置 YAML 仅由 rank 0 保存。

## 验证结果

- 本地和服务器各 20 项测试通过，包括双进程正常梯度平均及禁用、空实现、漏调用同步的拒绝测试。
- 实际前 3 轮 PPO 更新各执行 20 次梯度同步，更新后两卡参数最大绝对差异均为 0。
- 首次梯度同步独立计算了跨卡平均值，并与生产 PPO 同步后的全部梯度逐项比较通过。
- 新运行保存的 `model_650.pt` 已有 17 个参数张量发生变化；模型与优化器所有浮点张量均有限。
- 首次更新的 actor/critic 归一化计数均从 10698752 增至 10715136，增加 512×32，保持线性增长。
- 验证时只有一份 TensorBoard 事件文件，其最新标量迭代为 652。
- 本次验证覆盖启动后的前三轮更新；长时间运行由独立监督进程继续监控。

## 日志与运行控制

下列路径均相对于服务器项目目录。

新模型与唯一 TensorBoard 事件目录：

`logs/rsl_rl/dexsuite_revo3_rotate_bulb/2026-09-12_02-49-30_rotate_bulb_20260912T024930Z_resume650_stack256_dual_gpu/`

合并终端日志：

`remote_train_logs/rotate_bulb_resume_20260912T024930Z/train.log`

监控状态入口、启动参数及验证报告：

- `remote_train_logs/rotate_bulb_resume_20260912T024930Z/latest.json`
- `remote_train_logs/rotate_bulb_resume_20260912T024930Z/launch_manifest.json`
- `remote_train_logs/rotate_bulb_resume_20260912T024930Z/validation.json`
- 新模型目录中的 `distributed_validation.json`

监督 PID 为 4175；训练 worker PID 为 4187、4188（仅代表启动时现场，操作前应核对）。监督进程检测到 PhysX/CUDA 错误、同步失败或进度停滞时，会停止它启动的训练进程树并记录原因。

同步前的服务器文件备份位于：

`remote_train_logs/rotate_bulb_resume_20260912T024930Z/before/`

本次更新了任务环境配置、任务 PPO 日志配置和训练入口，并新增 `scripts/rsl_rl/distributed_training.py`。同步前校验旧文件 SHA-256，同步后校验新文件；原检查点保留在旧运行目录。
