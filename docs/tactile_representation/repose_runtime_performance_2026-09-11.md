# Repose 触觉训练性能修复：2026-09-11

远端：`dsw-8prvxfz53nbu67qzsu`，项目：`/home/admin/workspace/xinyu/mulyi-model-tac`。

已按“远端修复与测试 → 重启训练 → 同步本地”的顺序执行。训练仍使用两张 A100、每卡 1024 个环境、总计 2048 个环境，观测维度 1432，预训练 unaligned simulation 编码器不变。

## 修改

- Marker 像素投影改为整批张量计算，去掉每步 5120 次 Python 循环；原有无效 Marker 掩码和投影公式保留。
- 编码器默认模态掩码由输入可用性确定时，减少不必要的 GPU 到 CPU 同步；显式掩码、缺失模态和全部无效输入继续校验。
- Repose 的编码批大小从 32 调整为 256。FP16/BF16 实测更慢，因此继续使用 FP32。
- 项目内的 RSL-RL runner 包装器保留上游指标写入，仅修正终端 elapsed/ETA 的天数显示，以及完成后的剩余轮数。没有修改第三方安装包。

## 验证

- 远端 Python 3.11：60 项测试通过。
- 本地 `revolab` Python 3.11：相同 60 项测试通过。使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 避免无关的用户级 Dash 插件干扰。
- 真实仿真运行 120 步，覆盖接触、无接触、自然重置和主动重置。RGB、Depth、Marker、有效标志与旧实现逐元素一致；归一化编码特征最大绝对差异为 3.0778348e-05。
- 双卡完整配置验证 5 轮，正常退出（退出码 0）；保存的 checkpoint 共检查 92 个张量，均为有限值。
- 当前正式任务已完成到迭代号 4，生成新的 `model_0.pt`，权重均为有限值；TensorBoard HTTP 状态为 200。
- 本地 GitNexus 索引已更新，并运行 `detect-changes --scope all --repo . --limit 1000`。工作区原有未提交修改保留；同步前后均按文件 SHA256 核对。
- 验证运行关闭 Isaac 时存在 headless A100 图形设备枚举提示；训练、保存和 torchrun 退出均成功。

| 相同 2048 环境配置 | 修复前 | 修复后 |
|---|---:|---:|
| 稳定轮次平均耗时 | 59.179 秒 | 38.422 秒 |
| 相对速度 | 1.00× | 1.54× |

轮次耗时减少 35.1%。对比采用修复前第 3–29 轮及验证运行第 1–4 轮，排除首次预热轮次；实际耗时仍会随接触状态变化。

独立微基准：1024 份触觉输入编码从 494.4 ms 降至 320.7 ms；5120 份 Marker 的纯投影从 391.520 ms 降至 0.179 ms，投影输出完全相同。纯投影数字不包含上游传感器和 HydroShear 计算。

## 正式运行

- 运行名：`repose_perf_fixed_2gpu_env1024_total2048_20260911T114821Z`
- 启动时间：`2026-09-11T11:48:21.564204+00:00`
- 启动 PID：`29292`
- GPU：`0,1`
- 目标：10000 轮；保存间隔：25 轮，当前速度约 16 分钟。
- 恢复来源：`/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/brainco_repose_cube_visuotactile/2026-09-11_10-52-24_repose_unaligned_2gpu_env1024_total2048_20260911T105214Z/model_0.pt`
- 旧任务停止前日志到第 51 轮，但只保存了 `model_0.pt`。尚未落盘的进度未恢复；验证短跑的权重也未混入正式恢复来源。
- 新 checkpoint：`/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/brainco_repose_cube_visuotactile/2026-09-11_11-48-32_repose_perf_fixed_2gpu_env1024_total2048_20260911T114821Z/model_0.pt`
- 训练日志：`/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/repose_perf_fixed_2gpu_env1024_total2048_20260911T114821Z.log`
- 指标 JSON：`/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/repose_perf_fixed_2gpu_env1024_total2048_20260911T114821Z.metrics.json`
- TensorBoard 继续使用原来的 6006 端口。

## 复现与备份

远端完整记录、原文件、差异补丁、微基准脚本和真实观测对比脚本在：

`/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/repose_perf_fix_20260911T113100Z`

本地同步前文件备份在 `/tmp/repose_perf_sync.8hF6rc/local_before`。

测试命令（使用 Python 3.11 且包含项目依赖）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_tactile_runtime_performance.py \
  tests/test_tactile_policy_encoder.py \
  tests/test_rotate_bulb_tactile_encoder.py \
  tests/test_direct_repose_visuotactile_task.py \
  tests/test_tactile_representation_network.py
```
