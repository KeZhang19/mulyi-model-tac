根据用户要求，已停止此前的 Repose v2 从零训练，将本地与服务器 `dsw-8prvxfz53nbu67qzsu` 的 RSL-RL PPO `gamma` 从 0.99 改为 **0.999**，单回合时长从 10 秒改为 **15 秒**，并启动新的从零训练。

| 参数 | 新运行配置 |
| --- | --- |
| gamma / GAE lambda | 0.999 / 0.95 |
| 物理步长 / 控制降频 | 1/120 秒 / 4 |
| 策略控制频率 | 30 Hz |
| 回合时长 / 最大控制步数 | 15 秒 / 450 步 |
| 并行环境 | GPU 0、1，各 1024 环境，共 2048 |
| 每轮采样 / PPO 更新 | 每环境 16 步；5 epochs × 4 minibatches |
| 计划训练 / 保存间隔 | 10000 轮 / 25 轮 |
| 初始学习率 | 5e-4，KL 自适应 |
| 恢复旧 PPO | false；策略、价值网络、优化器、归一化统计均重新初始化 |
| 触觉编码器 | 保留冻结的 sim_policy_encoder_unaligned.pt |

本轮仅修改 [PPO 配置](../../source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/agents/visuotactile_rsl_rl_ppo_cfg.py) 的 gamma 和 [环境配置](../../source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/brainco_hand_visuotactile_env_cfg.py) 的 episode_length_s；奖励、动作和课程沿用上一轮 v2 实现。按 30 Hz，近似折扣跨度为 1000 个控制步，即约 33.3 秒；15 秒后的奖励权重约为 0.999^450 ≈ 0.637。

新运行名：`repose_v2_g0999_ep15s_from_scratch_2gpu_env1024_total2048_20260912T142144Z`。北京时间 2026-09-12 22:21:44 发起，服务器运行目录：`/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/brainco_repose_cube_visuotactile/repose_v2_g0999_ep15s_from_scratch_2gpu_env1024_total2048_20260912T142144Z`。启动明确传入 `agent.resume=false agent.algorithm.gamma=0.999 env.episode_length_s=15.0`，未指定 `--resume`、`--load_run` 或 `--checkpoint`。完整命令见 [formal_launch.json](../../outputs/repose_gamma0999_15s_20260912T1419Z/formal_launch.json)。

本地与服务器各 13 项现有任务集成/运行时测试通过，见 [本地结果](../../outputs/repose_gamma0999_15s_20260912T1419Z/local_tests.log) 和 [服务器结果](../../outputs/repose_gamma0999_15s_20260912T1419Z/server_tests.log)。部署前比对原文件哈希，备份后再写入；部署后两端哈希一致，见 [文件清单](../../outputs/repose_gamma0999_15s_20260912T1419Z/file_manifest.json) 与 [本次两行差异](../../outputs/repose_gamma0999_15s_20260912T1419Z/configuration.diff)。

北京时间 22:27:25，已完成编号 0–2 的前三轮更新，[24 项启动核验](../../outputs/repose_gamma0999_15s_20260912T1419Z/formal_start_validation.json)全部通过。运行保存的 `agent.yaml` 确认 gamma=0.999、resume=false，`env.yaml` 确认 episode_length_s=15.0、dt=1/120、decimation=4，即每回合 450 控制步。首个 `model_0.pt` 已保存，包含模型及优化器的 93 个张量均为有限值；优化器所有参数 step=20，actor/critic 归一化计数均为 16384（每卡 1024 环境 × 16 个新采样步），噪声标准差约 0.35038，课程状态 revision 2/stage 0。命令、配置和日志均确认未载入历史 PPO 检查点。前三次 PPO 更新每次 20 次梯度同步，两卡参数最大差异均为 0；TensorBoard HTTP 200，后台监控 running/ok。训练继续运行，这些结果验证启动与配置，不代表策略已收敛。

旧运行已停止，已保存的 `model_50.pt` 保留于 `/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/brainco_repose_cube_visuotactile/repose_v2_from_scratch_2gpu_env1024_total2048_20260912T133944Z/model_50.pt`，见 [停止记录](../../outputs/repose_gamma0999_15s_20260912T1419Z/stop_request.json)。修改前的两份服务器配置备份在 `/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/repose_gamma0999_15s_20260912T1419Z/backup/`。最新任务入口为 `remote_train_logs/repose_unaligned_2gpu_latest.json`，监控每 60 秒更新，TensorBoard 端口 6006。

影响分析针对仓库 mulyi-model-tac（本地路径 /home/liuxinyu/workspace/mulyi-model-tac，索引基线 ee6e431）。环境配置有 1 个直接导入依赖，图风险 LOW；PPO 配置图风险 UNKNOWN，已文本确认其通过该任务的 Gym 注册入口动态加载。未提交 Git。
