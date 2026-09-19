# Custom 灯泡控制与失败约束修订（2026-09-18）

本轮保持目标位置与 669 维观测，保留“已释放、目标 3 cm 内连续 1 秒”的成功条件，
没有新增成功持握门槛。修改同时作用于旧关节导向和纯碰撞模式，只修改 Custom。
本轮不启动正式重训，原训练及旧 checkpoint 保留。

## 实现

| 文件 | 变化 |
| --- | --- |
| `RotateBulbCustom/actions.py` | 名称识别 7 臂/21 手关节；动作 ±1；臂/手每控制步最多 0.025/0.05 rad；当前角度加增量并夹限位；四子步固定目标；非有限动作报错 |
| `scripts/rsl_rl/bulb_training.py` | 复用 Repose tanh 高斯数学，独立 Bulb v2 标记；均值平滑 ±2，std 0.05–0.7、初始 0.35；保存和严格校验动作契约；旧模型只允许显式诊断 |
| `RotateBulbCustom/agents/rsl_rl_ppo_cfg.py` | 独立策略类、初始噪声 0.35、熵系数 0.001 |
| `RotateBulbCustom/self_collision.py`、`environment.py` | 源实例修复后再克隆；恢复 192 项过滤，保留相邻、固定腕件及手内排除；要求开启 articulation 自碰撞 |
| `RotateBulbCustom/env_cfg.py` | 失握计时、远离代价、非有限检查、失败优先级、动作代价和持握拧动进度 |
| `scripts/rsl_rl/train.py`、`play.py` | 仅加入 Custom 路由；旧 PPO 拒绝续训；显式旧诊断严格加载原网络/normalizer，输出裁剪并隔离日志/视频/导出 |
| `scripts/rsl_rl/bulb_diagnostics.py`、`validate_bulb_revision.py` | 逐步诊断和两环境原生验收入口 |
| `assets/rotate_bulb_custom/probe_self_collision.py` | 两环境非邻接臂段碰撞 A/B 探针 |

动作和变化代价均为臂/手各占一半的均方，权重 −0.05/−0.02，按秒积分，无 1000 封顶。
失握有初始 0.5 秒宽限；所有手指灯泡接触力低于 0.1 N 连续 0.75 秒，或
`max(拇指距离, 其他四指最小距离)>0.20m` 连续 0.25 秒即失败。距离参考为灯泡局部
`(+0.04,0,0)` 壳体点。远离代价率为 `−clamp((d−0.08)/0.12,0,1)^2`。
进入原成功位置窗口且已释放时，清零新失握计时并关闭新远离代价；离开后重新累计。

落桌仍为灯泡与桌面接触力 >0.5 N，删除先释放要求。落桌、越界、失握、非有限状态
统一每回合一次 −6，超时不额外罚；失败帧所有正奖励为零，失败优先于成功。
关闭无持握的独立转动奖励，有接触的拧动进度改为每圈最多 2 分，保留 0.2 秒换抓记忆。
原连续目标跟踪、其他持握预算保持，因此本轮不宣称已经消除所有等待收益。

## 验收证据

本地 CPU/USD 回归 **346 passed，0 skipped**，包含实际 PPO 更新、tanh 概率/熵、
原生 JIT/ONNX 输出比对、动作四子步缓存/限位/部分 reset、30/60/120 Hz 计时、
目标保持窗口、失败重叠与同帧成功、非有限状态及原灯泡/Repose/D-Peg 相关回归。
日志：`logs/diagnostics/bulb_revision_20260918/cpu_validation_final.log`。
测试中修正了几条旧的共享断言：标准灯泡仍按原持握成功和历史最佳搬运进度，
Custom 按其原有位置成功和连续跟踪；标准任务源码未改。

服务器候选目录：`/home/admin/workspace/xinyu/mulyi-model-tac_candidate_bulb_20260918`。
715 GPU1 验证旧关节模式，nin GPU0 验证纯碰撞模式；各 2 环境、120 步物理探针、
2 次 PPO 更新、120 步旧模型诊断。均通过：

- 实际请求增量最大 0.025/0.05 rad，四个物理子步目标逐元素相同。
- 669 维观测及所有观测组/奖励有限，非有限动作在进入物理前被拒绝。
- 两个克隆均已恢复 192 对，保持其余 91 项过滤。
- checkpoint 实际重新加载，26 个 state_dict 张量（含 normalizer）严格一致，推理误差 0。
- 旧 guided 诊断两次失握，各扣 −6；纯碰撞短诊断未触发失握。这不是收敛或成功率评估。

对应报告：`logs/diagnostics/bulb_revision_20260918/server/logs/{guided,pure}_acceptance_v1/report.json`。
逐步日志的 `returned_state_*` 字段为 `env.step()` 返回状态，终止后可能已 reset；
同帧终止标志和 `step_reward` 仍记录该次失败事件。

| 两环境受控 link2–link6 物理探针 | 恢复过滤 | 原过滤对照 |
| --- | ---: | ---: |
| 峰值接触力 | 76.87 / 76.21 N | 0 / 0 N |
| 终点交叠内切半径 | 0.0954 / 0.0953 mm | 7.727 / 7.708 mm |

CPU 参考初态检查了 180 对有 collider 的关系，均有间隙。物理报告在
`logs/diagnostics/bulb_arm_reversal_20260918/collision_acceptance/`。
首次恢复过滤探针计算通过后在 Isaac 清理阶段超时，保留原证据，改用快速关闭后在
nin 复验通过且退出码为 0，报告为 `restored_clean_exit.json`。
715 视频在渲染器启动阶段崩溃，发生于加载任务前；移除 CUDA 重映射并指定物理 GPU
仍不能启动，因此不能将原因确定为重映射。两种模式的视频/导出入口改在 nin 验证。
715 的无渲染物理与 PPO 验收已通过，其渲染器启动问题仍保留为机器环境限制。
所有失败均发生在候选目录，正式源码未被该次试验修改。

GitNexus 修改前入口影响 LOW，失败聚合 MEDIUM；动态注册边界经实际配置引用确认。
最终增量索引出现空/错误符号 ID，完整重建后已修复。重建后的工作区检查为 HIGH、
9 条流程：本轮公共入口关联 5 条（其中 1 条为未改的既有 Repose 分支），其余 4 条
来自已有抓姿验证/插销改动。此前损坏索引给出的 CRITICAL/335 流程不能用于判断。
新动作/碰撞实现为 LOW，策略与加载配置为 MEDIUM，工厂/回调的动态边界为 UNKNOWN，
已核对实际注册引用。索引仍有流程预算和动态 Python 覆盖限制，不视为 clean 全覆盖证明。
完整 JSON 和审计说明保存在同一诊断目录，
本轮部署清单只包含 Custom 相关文件，原 USD 和共享 spawn 不变。

## 启动方式

本地新训练示例（留待下一步执行；不使用旧 checkpoint resume）：

```bash
cd /home/liuxinyu/workspace/mulyi-model-tac
export PYTHONPATH="$PWD/source/BrainCo_DexHand:$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0 \
  --num_envs 4096 --device cuda:0 --seed 42 --headless \
  --run_name bulb_v2_guided \
  env.bulb_pure_collision=false env.tactile_policy_enabled=false
```

纯碰撞模式把 `env.bulb_pure_collision` 改为 `true`，并使用独立 run_name。
服务器运行时改项目目录为 `/home/admin/workspace/xinyu/mulyi-model-tac`，Python 改为
`/home/admin/workspace/xinyu/IsaacLab/_isaac_sim/python.sh`，按空闲 GPU 选择设备。

原 `model_6100.pt` 的兼容诊断：

```bash
cd /home/liuxinyu/workspace/mulyi-model-tac
export PYTHONPATH="$PWD/source/BrainCo_DexHand:$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/rsl_rl/play.py \
  --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0 \
  --checkpoint checkpoints/bulb_review_20260917/guided/model_6100.pt \
  --bulb-legacy-diagnostic --num_envs 2 --device cuda:0 --seed 42 --real-time \
  env.bulb_pure_collision=false env.tactile_policy_enabled=false
```

诊断会采用新动作/碰撞/失败规则，因此不能用它复现原训练动力学。添加
`--video --video_length 600` 可保存固定长度视频；日志、JIT 和 ONNX 会写入独立的
`logs/diagnostics/bulb_legacy/<时间>/`。原网络配置、观测契约必须与 checkpoint 一起保留。
新版模型回放不加诊断参数，需同时保留 `bulb_action_contract.json`。

## 同步与回滚

只同步验收过且哈希匹配的候选文件；远端 `train.py` 保留服务器已有的 D-Peg 分支差异。
逐文件备份和部署清单位于正式项目的 `remote_train_logs/bulb_revision_20260918/deploy_backup/`。
部署脚本在任何文件写入前检查全部原文件 SHA-256；替换失败会恢复本轮备份。
两台服务器部署后分别验证 SHA-256 和 Python 加载路径。没有修改旧 checkpoint，
没有停止或重启原训练与监督进程。715 的旧 guided 训练在本轮开始前已自行完成 15000 次。
