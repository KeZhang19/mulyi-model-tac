# Repose 观测重置污染修复：2026-09-12

任务：`BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0`。

## 原因与修复

`TactileObservationHistory.append()` 和环境的观测缓存分支返回历史张量的视图。
RSL-RL 在 `act()` 时保留观测引用，等 `env.step()` 返回后才复制进 rollout storage。
如果这一步发生回合重置，原来的 `reset()` 会原地清零历史，连同 PPO 尚未保存的上一帧观测一起清零。
于是训练数据中的观测与动作、旧策略均值、价值估计不再对应。

现在 `reset()` 先复制历史缓冲区，再清零指定环境。已经返回的观测保持原值，后续回合仍按原逻辑重新填充历史。
这同时覆盖正常返回和观测缓存分支；奖励、输入顺序、维度、编码器和动作映射均未修改。

本次代码改动：

- `source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/policy.py`：重置前复制历史缓冲区，增加解释注释。
- `tests/test_tactile_policy_encoder.py`：新增 6 个回归用例，覆盖单帧/四帧历史、空/部分/全部重置、连续重置，以及 PPO 延后复制旧观测的行为。

## 验证

- 远端先添加回归测试：旧实现 6 项全部失败。
- 修复后远端和本地各 38 项相关测试通过。
- 服务器独立 Isaac 仿真：64 个环境、8 步、`model_1525.pt`，固定权重及归一化，主动触发一次真实回合超时。没有执行优化器更新。

| 检查 | 修复前 | 修复后 |
|---|---:|---:|
| 重置导致旧观测被改写的环境数 | 1 | 0 |
| PPO storage 保存错误观测的环境数 | 1 | 0 |
| 重置样本的 KL（相同权重及归一化） | 96.9895 | 0 |
| 该步 64 个样本的平均 KL | 1.51546 | 0 |

本地验证记录：`outputs/repose_observation_fix_20260912/simulation_after.json`。
修复前记录：`outputs/repose_model_1450_20260912/observation_audit.json`。

测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_tactile_policy_encoder.py \
  tests/test_direct_repose_visuotactile_task.py \
  tests/test_rotate_bulb_tactile_encoder.py \
  tests/test_tactile_runtime_performance.py
```

## 续训

等待旧训练的 `model_1550.pt` 完整保存并检查张量有限性后，停止旧进程并于北京时间 11:49:07 启动修复后的进程。

- 新运行：`repose_obs_fixed_2gpu_env1024_total2048_20260912T034907Z`。
- 启动 PID：`71499`。
- 两卡各 1024 个环境，总计 2048 个环境。
- 从旧运行的 `model_1550.pt` 恢复策略和优化器；追加 8450 轮，目标迭代上限保持 10000。
- 保持每 25 轮保存；启用已有双卡梯度及参数同步检查。
- 最新运行入口：远端 `remote_train_logs/repose_unaligned_2gpu_latest.json`。

北京时间 11:54:57 核验：已完成迭代 1550–1552，进程仍在运行。前三轮每轮 20 次梯度平均均通过检查，两卡更新后参数最大差异为 0。
新 checkpoint 张量全部有限，TensorBoard HTTP 状态为 200。
实际学习率从首轮的 `1e-5` 升至后两轮的 `1.7085937724914402e-4`，已离开此前长期停留的下限；这只是早期更新恢复的证据，尚不能据此判断旋转任务收敛。
正式运行验证记录：`outputs/repose_observation_fix_20260912/restart_validation.json`。

## 影响范围与备份

GitNexus 绑定本地 `mulyi-model-tac`，索引提交 `ee6e431`、更新时间 2026-09-12 11:12。
目标源文件在编辑前经 SHA256 确认远端和本地相同。
`TactileObservationHistory.reset` 上游图分析为 LOW，解析到 Direct 任务的 `_reset_idx`。
图包含动态调用边界，因此同时检查了历史缓存使用处及直接读取 `frames` 的路径。
全工作区 `detect-changes` 报告还包含原有无关修改，且目标文件尚未被 Git 跟踪，不能将该命令作为此次未跟踪文件的完整差异检查；另按修改前备份核对了这两个文件的精确差异。
本次未提交代码。

远端备份、回归失败日志及仿真记录：
`/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/repose_observation_fix_20260912T034300Z`。

本地修改前备份：`/tmp/repose_observation_fix_20260912_local_before`。
如需回退，可按备份撤销本次缓冲区复制和新增回归测试；运行中的 Python 进程需重启才能加载代码变更。
修复证明观测污染消失，旋转能力的恢复仍需观察续训与独立评估。
