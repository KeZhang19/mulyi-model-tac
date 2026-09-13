# Custom 关闭多模态观测的从零训练（2026-09-13）

目标服务器：`dsw-lal9ty3huw0odwjjz5`，hostname：`dsw-962368-77588f9ddc-zmphf`。
项目：`/home/admin/workspace/xinyu/mulyi-model-tac`。
任务：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`。

北京时间 2026-09-13 00:45:19 启动。目标机原来的标准版 Rotate-Bulb 训练及其
supervisor、torchrun 和 worker 共 5 个相关进程已退出，旧日志和检查点保留。
新训练 `agent.resume=false`，不加载旧 RL 检查点。

对照为 `dsw-715zhrv7l6bux2ldbl` 上的 `custom_turns_scratch_seed42_dual_gpu`，
它继续使用预训练多模态观测。两台机器共享项目目录，源码同步可被两台看到；
已经初始化的对照进程仍使用原来的开启模式。

| 配置 | 本次关闭观测训练 / 开启观测的对照 |
| --- | --- |
| GPU / 环境数 | 2 卡 × 每卡 512，共 1024 |
| seed / 从零训练 | 42 / resume=false |
| 最大迭代 / 保存间隔 | 15000 / 25 |
| gamma / lambda / rollout | 0.999 / 0.95 / 32 |
| 初始旋入 / 回合 / 控制频率 | 3 圈 / 20 秒 / 60 Hz |
| 动作幅度 / 变化惩罚 | -0.0001 / -0.0002 |
| 基础圈数 / 抓持圈数奖励 | 0.5 / 1.5 |
| 预训练触觉观测接入 | **关闭 / 开启** |
| actor、critic 观测维度 | **669 / 1949** |

开关为 `env.tactile_policy_enabled=false`，Custom 的训练和 PLAY 默认均关闭。
开启时使用 `env.tactile_policy_enabled=true`，恢复原来的五指 1280 维 latent。
关闭时保留 285 维压力观测及其他状态、点云观测；不加载模态还原模型，不执行
该路径的 RGB / Depth / marker 生成、编码推理及重建诊断。
原有 `tactile_reconstruction_diagnostics` 默认仍为 false，仅在总开关开启时有效。
开关在环境初始化时读取，支持 Hydra 覆盖；运行中切换不受支持。
旧多模态策略回放和续训必须显式开启开关，观测契约会拒绝跨模式加载。

TensorBoard / 检查点目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/2026-09-12_16-41-52_custom_turns_tactile_off_scratch_seed42_dual_gpu
```

控制目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/rotate_bulb_custom_tactile_switch_20260912T163441Z
```

`train.log` 为终端日志，`state.json` 持续记录训练状态，`launch_manifest.json` 为完整命令，
`previous_training_stop.json` 记录旧进程身份和退出结果；`before/` 保存同步前的 4 个文件。
本次只同步 Custom 的 `env_cfg.py`、`tactile.py`、README、一个原有测试及一个新测试。
167 个相关文件的哈希与本地一致，包含 140 个传感器依赖和 Custom 资产文件；
两台服务器的 42 个 RSL-RL Python 文件一致。

本地 173 项测试通过、1 项缺少本地 USD 依赖而跳过，目标服务器 174 项全部通过。
本地 GPU 分别验证开启与关闭模式的真实环境初始化、物理状态、观测、步进及超时重置，
没有 PhysX/CUDA 错误；关闭模式即使指定不存在的 encoder checkpoint 也可正常运行，
并确认未初始化 HydroShear adapter、RGB 背景及稠密 Depth 张量。
GPU 验证仅为适应本地显存而降低 PhysX 缓冲区，正式服务器训练使用原有缓冲区。

服务器启动验收通过，详见控制目录的 `validation.json`。逐字段比较两次训练实际保存的
`params/env.yaml` 和 `params/agent.yaml`，差异仅为新增关闭开关、`log_dir` 和 `run_name`；
对照配置没有该字段，因为当时尚未加入开关，其观测契约确认实际使用开启模式。
前三次 PPO 更新每次完成 20 次梯度同步，两卡参数最大差异均为 0，首次跨卡梯度均值核验通过。
`model_0.pt` 的迭代编号为 0、优化器步数均为 20，模型及优化器张量有限；
actor 和 critic 第一层输入均为 669 维。五项旋转圈数指标和两项旋出奖励均已写入 TensorBoard。

北京时间 00:49:44 检查时，新训练处于 `running`，已完成 27 次更新；
开启观测的对照仍处于 `running`，已完成 72 次更新。
此时两组各自最近 10 次更新的吞吐中位数为 **7299.5 / 690 steps/s**，约 **10.6 倍**；
每次采样耗时中位数为 **4.399 / 47.338 秒**。这是启动阶段的速度记录，见 `final_status.json`。

本地记录位于 `logs/diagnostics/rotate_bulb_custom_tactile_switch_20260912T163441Z`。
GitNexus 上游分析覆盖 Custom 环境类、触觉项和启动契约，已解析依赖风险 LOW；
启动契约的 UNKNOWN 结果由事件注册引用和运行测试补充验证。
全工作区图变更检查包含本次与已有未提交改动，不能将其整体风险视为仅本次开关的风险。
审查使用临时 Git 索引，实际 Git 索引未改变，未执行提交。
