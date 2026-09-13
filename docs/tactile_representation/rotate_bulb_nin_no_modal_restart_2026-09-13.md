# Rotate-Bulb Custom：`dsw-nin473ap6o7410zyra` 无模态重启与在线自监督诊断

检查时间：2026-09-13（北京时间）。

## 旧训练判断

旧运行 `custom_scratch_reward_v2_episode20_seed42_dual_gpu` 使用 2 张 GPU、每卡 512 个环境、20 秒回合、3 圈起点、`gamma=0.999`，实际观测契约为 **1949 维**，其中包含冻结的 1280 维预训练触觉恢复特征。约 1300 轮时成功率仍为 0%；最近奖励约 0.67，`unscrew_progress` 约 0.112，位置误差约 0.115 m，姿态误差约 1.79 rad，超时约 99.8%。

奖励随训练轮数上升主要来自旋出进度增加和动作惩罚变小；`fingers_to_object`、`good_finger_contact` 和最终成功长期为 0。冻结策略回放显示确定性策略平均约 2.924 圈后停止，对向指尖接触时间占比约 0.504%，手指接近软限位约 51.1%，动作目标越过软限位约 42.4%。这说明策略能获得早期旋出进度，却没有学会持续对向抓持、换抓和释放后的稳定搬运。

## 本次调整

新运行从头初始化 PPO，并显式设置 `env.tactile_policy_enabled=false`，使策略输入回到 669 维；不加载旧 1949 维 checkpoint。显式启用已验证的逐指 `any_finger_contact` 辅助项（权重 0.1、每指力尺度 0.5 N），并保持 gamma、动作惩罚、3 圈起点、20 秒回合和 PPO 结构不变，以隔离“去除冻结模态计算 + 持续接触引导”的影响。旧运行目录、检查点和日志保留。

## 自监督诊断边车

仓库当前没有训练期间在线联合更新 encoder 的入口，`RotateBulbPretrainedTactile` 仍是冻结、无梯度推理；服务器也没有可直接供 `train_tri_modal_cross_autoencoder.py` 使用的采集数据集。因此同时启动轻量自监督诊断边车：周期性读取训练产生的 `state.json` 和 TensorBoard/终端指标，记录伪标签式的接触、旋出、释放、掉落、超时与成功关系，检查奖励是否真的改善后段行为。它不热切换策略权重，也不冒充 encoder 自监督训练；如后续准备好 RGB/Depth/marker 数据集，再单独启动离线自监督表示学习并用于下一轮策略。


## 已启动验收

新运行目录：

```text
logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/20260913T054600Z_nomodal_contact_seed42_dual_gpu
```

控制与监控目录：

```text
remote_train_logs/rotate_bulb_nin_nomodal_20260913T054600Z
```

启动保存的契约确认 `feature=disabled`、`tactile_policy_enabled=false`、`observation_dim=669`；
`any_finger_contact` 的实际保存权重为 0.1、力尺度为 0.5 N，`resume=false`，起点为 3 圈、
回合为 20 秒，gamma/lambda 为 0.999/0.95。前三次跨卡 PPO 更新均完成，每次 20 次梯度归约，
两卡参数最大差异为 0，首次梯度均值核验通过。约第 40 轮时日志已写入非零接触辅助项，仍未出现
最终成功；训练进程和自监督诊断边车均在运行，未发现 CUDA/PhysX/非有限数值错误。
