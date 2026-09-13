# Custom 旋转圈数奖励从零训练（2026-09-12）

服务器：`dsw-715zhrv7l6bux2ldbl`，hostname：`dsw-949505-64bd8d4dc8-wpvz4`。
项目：`/home/admin/workspace/xinyu/mulyi-model-tac`。
任务：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`。

北京时间 2026-09-12 23:48:58 启动新训练。旧 Custom 训练及其 supervisor/torchrun/worker
均已停止，旧模型和日志保留。停止记录见本次控制目录的 `previous_training_stop.json`。

当前训练从第 0 轮开始，`resume=false`，不载入旧 RL 检查点；触觉编码器继续使用原来的冻结预训练权重。
双卡每卡 512 环境，共 1024；seed=42，3 圈、20 秒、60 Hz、gamma=0.999；目标 15000 轮，每 25 轮保存。
保留旧运行的动作惩罚覆盖：`action_l2=-0.0001`、`action_rate_l2=-0.0002`。
新增基础圈数 `unscrew_turns=0.5`，抓持圈数 `unscrew_progress=1.5`，充分抓持时三圈仍最多 6 分。

TensorBoard/模型目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/2026-09-12_15-45-34_custom_turns_scratch_seed42_dual_gpu
```

控制与终端日志目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/rotate_bulb_custom_turns_restart_20260912T154534Z
```

其中 `state.json` 为持续更新的训练状态，`train.log` 为终端日志，`launch_manifest.json` 为启动参数，
`validation.json` 为首次三轮验收。`before/` 保存同步前的文件；只同步了 6 个实际变化文件，
Custom 包、训练入口、相关依赖及冻结编码器的其余核对文件原本已与本地一致。

服务器 152 项相关测试通过。前三次 PPO 更新每次执行 20 次梯度同步，两卡参数最大差异均为 0；
首次跨卡梯度均值独立核验通过。新 `model_0.pt` 的迭代编号为 0，优化器步数均为 20，
模型与优化器张量均有限；日志未加载旧模型，实际保存配置确认新权重和 `resume=false`。

新增的五个 `Metrics/object_pose/unscrew_*` 指标和两项旋出奖励均已写入唯一的 TensorBoard 文件，
读取值全部有限。第 2 轮日志：平均回合最大有效旋出约 0.2102 圈、结束时约 0.1577 圈，
达到 1/2/3 圈比例均为 0。这些是启动阶段记录，且遵循现有 rank 0 日志范围，不能据此判断长期成功率。

最终检查时训练状态为 `running`，已完成 4 次更新；
检查时间为 2026-09-12T15:55:01.107722+00:00。监督进程会继续监控它启动的进程树、数值错误和训练进度。

本地验收记录：`logs/diagnostics/rotate_bulb_custom_turns_restart_20260912T154534Z`。
GitNexus 在本地最新索引中复核了奖励和命令类的上游影响，风险 LOW，图中流程 0；
动态事件和管理器调用由配置引用、精确服务器源文件差异以及运行测试补充确认。
服务器未安装独立 GitNexus 索引；没有修改或提交 Git 索引。
