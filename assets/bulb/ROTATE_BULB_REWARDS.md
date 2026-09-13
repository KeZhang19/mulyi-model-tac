# Revo3 旋灯泡任务奖励

任务：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0`。
配置：`dexsuite_revo3_env_cfg_rotate_bulb.py` 中的 `DexsuiteRevo3RotateBulbRewardCfg`。

## 场景与物理脱离

奖励按接近外壳、抓持并旋出、搬运、稳定保持的顺序设计。
任务在生成场景时扩展原始 USD：原生 `screw_turn/screw_slide` 仍通过 Mimic 联动，
驱动一个无碰撞的 `ScrewFollower`；真正的灯泡为独立刚体，通过
`excludeFromArticulation=True` 的普通固定关节连接到它。
到达螺口出口后关闭这个连接，灯泡由接触力和重力运动；不会在运行中逐帧写灯泡位置。
重置时才恢复关节、灯泡姿态、速度、阶段标志和奖励历史，支持仅重置部分环境。

原 USD 的最大伸出位置仍有约 `5.7 mm` 外螺纹处于螺口内。
任务把出口位置设为灯座原点沿螺旋轴 `30.3 mm`，恰好比原出口多一个导程，
保留螺纹相位，并使最低外螺纹碰撞点越过螺口约 `0.28 mm`。
原 24 mm / 四圈的关节行程保留，因此四圈起点对应出口往内 24 mm，
不是原 USD 的最深 CAD 装配零位。原始 USD 文件及独立演示不受影响；其原有边界见
[THREADED.md](hiveboard_lamp/THREADED.md)。

默认从灯泡已明显旋入支架的状态开始，训练和播放一致：

| 参数 | 默认设置 |
| --- | --- |
| 灯座位置，桌面坐标系 | X `[0.43, 0.47] m`，Y `[0.08, 0.12] m` |
| 剩余旋出圈数 | 每回合固定 `3` 圈，18 mm 旋出行程，尖端深入支架口约 17.69 mm |
| 目标位置，相对灯座原点的世界坐标偏移 | X/Y 各 `±4 cm`，Z `12～16 cm` |
| 回合 | `30 s`，目标只在重置时采样 |
| 重力 | 从第一回合起 `−9.81 m/s²`；灯泡启用重力 |
| 外壳摩擦 | 静摩擦 `0.9`，动摩擦 `0.8`；螺纹仍用原材质 |
| 被动螺旋阻力 | 转动关节静摩擦 `0.004 Nm`、动摩擦 `0.003 Nm`、黏性摩擦 `0.0005 Nm·s/rad` |
| 掉落 | 已脱离的灯泡与桌面接触力超过 `0.5 N` 时终止 |

目标发送给策略时仍转换到机器人基座坐标系。目标范围处于新的越界范围内部，
不会再要求灯泡移动到越界终止区域。外壳摩擦和支撑关节的 10 g 跟随体是仿真设置，
未作实物标定；脱离后灯泡本体质量仍为原来的 0.2 kg。
重力经导程产生约 `0.0019 Nm` 的旋入力矩，因此需要上述被动摩擦阻止灯泡自行旋到底。
它不提供正向旋出驱动，驱动刚度和阻尼仍为零。摩擦参数按本项目的 Isaac Sim 5.1 努力单位设置。

## 奖励与初始权重

| 项 | 权重 | 条件及含义 |
| --- | ---: | --- |
| 接近外壳 | 1 | 掌心/指尖到灯泡局部 `(0.04, 0, 0)` 外壳中心的平均距离，`std=0.2 m`；旋出完成后关闭 |
| 接近进步 | 5 | 只奖励本回合达到的新最佳接近程度 |
| 有效抓持 | 0.5 | 拇指加至少一个其他指尖与灯泡接触，力均超过 `1 N` |
| 旋出进步 | 40 | 有效抓持时才支付新增旋出进度；回转后再次到达旧进度不重复支付 |
| 旋出完成 | 含在上一项 | 每回合一次，额外支付 `40 × 0.5 = 20` |
| 搬运位置 | 8 | 物理连接已解除且有效抓持后启用，`std=0.2 m` |
| 搬运精定位 | 4 | 同上，`std=0.05 m` |
| 稳定到位 | 20 | 距目标小于 `3 cm`，灯泡原点高于灯座原点 `5 cm`，线速度小于 `0.05 m/s`，角速度小于 `0.5 rad/s`，有效抓持并连续保持 `0.3 s` |
| 动作幅度 / 动作变化 | 各 −0.005 | 保留原有动作正则项 |

旋出进度取角度进度与轴向进度的较小值，按每个环境实际采样的起点计算：

```text
p_turn  = clamp((screw_turn - start_turn) / (-start_turn), 0, 1)
p_slide = clamp((screw_slide - start_slide) / (-start_slide), 0, 1)
p       = min(p_turn, p_slide)
complete = p_turn >= 0.999 and p_slide >= 0.999
```

这样用累计关节角区分四圈旋转，而不是用会在每圈重复的四元数误差。
完成容差为起始行程的最后 `0.1%`。脱离后阶段锁存，不会因空载跟随体运动而回退。
无抓持时产生的进度也会更新历史最大值，但不支付奖励，避免事后重新接触补领旧进度。
稳定到位奖励在满足保持时间后持续支付；丢失抓持、偏离目标或速度超标立即清零保持计时。

Isaac Lab 会将奖励项乘以 `step_dt`。两种进度奖励和一次性完成奖励在函数内除以
`step_dt`，使完成同一段运动的总奖励不随控制频率改变。表中其他项按每秒权重积分。

## 配套设置与调参

- `policy` 观测包含两维 `screw_progress`，以及两维 `bulb_phase`（是否脱离、回合剩余时间比例）。
  旧检查点的输入维度不同，不能直接按原结构恢复。
- `Metrics/object_pose/success` 与稳定到位奖励使用同一判定，另记录 `released` 比例。
  已关闭原先仅凭位置误差升级、并会调整全局重力的抓取 ADR。
- 当前默认固定三圈、30 秒，没有自动提高圈数的课程。需要改变深度时显式覆盖
  `initial_screw_turns_range`，以旋出和稳定搬运成功率、完成耗时决定是否调整。
- 这些权重是起始配置，尚未通过长时间 PPO 训练验证收敛。

RSL-RL 训练例子（从仓库根目录，在 `brainco` 环境运行）：

```bash
python scripts/rsl_rl/train.py --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0 --num_envs 64 --headless
# 默认三圈、30 秒；需要更深起点时，在新训练或结构兼容的恢复训练中切换为四圈：
python scripts/rsl_rl/train.py --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0 --num_envs 64 --headless 'env.initial_screw_turns_range=[4.0,4.0]' env.episode_length_s=30.0
```

圈数参数在运行时重置中读取，因此 Hydra 在配置构造后的覆盖也生效。
时间可直接覆盖 `episode_length_s`；目标不会因回合延长而在中途改变。
PPO 网络及折扣参数未随场景更改，完整四圈阶段可另做折扣因子对比实验。

先检查 `fingers_to_object` 与 `good_finger_contact` 是否提高，再看
`unscrew_progress`。若靠近后一直不旋转，可降低持续接近/接触奖励，或提高
旋出进度权重。当前默认需要旋出三圈；若显式调整初始圈数，重置与进度基准会按该配置同步更新，
不应只把成功阈值改成小角度就称为完全旋出。
稳定到位较差时，优先检查抓持、目标误差和速度，
避免只放宽成功容差。

## 验证

在 `brainco` 环境中运行：

```bash
python -m pytest -q tests/test_revo3_rotate_bulb_task.py tests/test_rotate_bulb_rewards.py
python assets/bulb/tools/validate_rotate_bulb_task.py --headless --skip-taxim-rgb --num-envs 2
python assets/bulb/tools/validate_rotate_bulb_task.py --headless --skip-taxim-rgb --num-envs 2 --play
python assets/bulb/tools/validate_rotate_bulb_task.py --headless --skip-taxim-rgb --num-envs 2 --turns 4 4
```

CPU 测试覆盖累计转数、角度/行程共同门控、反转、无接触空转、部分重置、
控制频率、阶段门控、外壳接近和稳定保持。
仿真脚本覆盖目标范围、5 秒无接触保持初始深度、连接状态、向上旋出、出口处几何间隙、脱离后的独立运动和重力、
启动/整批/部分重置、桌面掉落检测，以及有限值奖励和观测。
机械探测期间移开机器人，并施加受控力矩、临时重力补偿和速度；自由运动测试前移除补偿。
这些操作只在验证脚本内使用，用来验证物理能力，不代表策略已经学会手抓搬运。
`--skip-taxim-rgb` 只在验证进程中关闭可选 RGB 观测。
