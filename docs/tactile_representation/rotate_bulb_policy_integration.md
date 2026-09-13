# Rotate-Bulb：默认使用预训练多模态恢复 encoder

当前任务方向为从支架中**旋出并搬运**。训练和 PLAY 默认均为固定 **3 圈、20 秒回合上限**，
灯泡尖端初始进入支架口约 **17.69 mm**；无需额外传入圈数或回合时长参数。
启动和每次重置都会恢复该深度。旧命令若显式覆盖这些参数，应删除覆盖或改为
`'env.initial_screw_turns_range=[3.0,3.0]' env.episode_length_s=20.0`。
修改对新建环境生效，已经运行的训练进程需要重新启动才能使用；旧浅起点策略仍需适应三圈任务。
几何依据和旧浅起点问题见[初始旋入深度排查](rotate_bulb_depth_diagnosis_2026-09-12.md)。

## 桌面安装与支架随机化

Rotate-Bulb 的训练和 PLAY 使用独立的桌面安装布局。以下坐标相对于每个并行环境的原点：

| 项目 | 默认值 |
| --- | --- |
| 桌子中心 / 尺寸 | `(0.55, 0, 0.38)` / `(1.2, 1.6, 0.76)` 米 |
| 机械臂根位置 / 朝向 | `(1.0, 0, 0.766)` 米 / 绕 Z 轴 −90° |
| 支架中心随机区域 | X `[0.43, 0.47]`，Y `[0.08, 0.12]` 米 |
| 支架原点高度 | `0.804732485` 米，网格底面贴合 0.76 米高的桌面 |

桌子向 X 正方向平移 55 厘米，覆盖机械臂底座。实际 USD 的 `Base_R` 网格最低点在根坐标下方
6 毫米，因此根高度为 0.766 米，底座底面与桌面齐平，距最近桌边约 9.35 厘米。
机械臂仍采用固定底座与原有重力设置，代表安装在桌面上的机器人。

为补偿底座抬高 18.6 厘米，`Joint1_R` 到 `Joint7_R` 的初始角度改为
`[-0.506634, -0.649688, 1.570792, -0.171813, -0.514927, -0.450123, 0.949530]` 弧度。
这是离线求出的固定配置，保持原有手掌和腕部位姿；肩肘中间位置变化约 24 厘米，手指初始角度保持不变。
运行时不增加逆运动学求解。

支架在启动和每次重置时独立采样，回合内保持固定。重置函数使用桌子局部坐标：
X `[-0.12, -0.08]`、Y `[0.08, 0.12]`，转换后保留原来的环境内随机区域。
部分重置仅改变选中的环境。Lift、共享机器人资产、动作/观测维度和奖励定义均保持原状。
旧 checkpoint 仍需在新布局中验证表现；重新创建环境后才会载入此配置。

桌面几何与物理验收可运行：

```bash
OMP_NUM_THREADS=2 python assets/bulb/tools/validate_rotate_bulb_task.py \
  --headless --num-envs 2 --device cuda:0 \
  --report /tmp/rotate_bulb_tabletop_train.json --scene-export /tmp/rotate_bulb_tabletop.glb

OMP_NUM_THREADS=2 python assets/bulb/tools/validate_rotate_bulb_task.py \
  --headless --num-envs 2 --device cuda:0 --play \
  --report /tmp/rotate_bulb_tabletop_play.json
```

验收使用实际 USD 网格和 PhysX 刚体位姿，覆盖启动、20 次完整重置、随机区域四角、部分重置、
手掌位姿保持及原有旋出/释放/重力/掉落流程。无界面验收会关闭需要远程 USD 的目标坐标标记，
保留全部策略触觉观测。GLB 导出保存实际网格与仿真位姿，可用于全景检查。

2026-09-12 本地验收：训练与 PLAY 各两个环境，上述检查全部通过；底座/支架贴桌误差小于
0.001 毫米，手掌/腕部位移误差小于 0.001 毫米、朝向误差小于 0.001°。
FCL 检查实际网格未发现非相邻机械臂连杆相交。原有任务和奖励测试 20 项通过，
另确认 Lift 配置及共享机器人默认值未被修改。
旧 `model_650.pt` 在新布局中通过两个环境、300 步（5 秒）回放，动作、观测和奖励均有限；
期间没有完成旋出，此结果仅验证加载兼容性与短时运行稳定性。

[初始布局全景](../../logs/diagnostics/rotate_bulb_tabletop_20260912/scene.png) ·
[旧策略回放全景](../../logs/diagnostics/rotate_bulb_tabletop_20260912/replay_300.png) ·
[完整验证记录](../../logs/diagnostics/rotate_bulb_tabletop_20260912/summary.json)。
这些诊断产物位于本地 `logs/diagnostics/rotate_bulb_tabletop_20260912/`，不随 Git 提交。

## 控制频率、折扣与奖励预算（2026-09-12 晚间更新）

保留 **240 Hz 物理、60 Hz 控制**，回合上限调整为 **20 秒**。RSL-RL 默认 `gamma` 从 `0.99` 改为 `0.999`，
保持 `lambda=0.95`、每环境 32 步 rollout。折扣发生在每个控制步，10 秒后奖励的相对权重从
`0.99^600≈0.002405` 变为 `0.999^600≈0.548647`；PPO 迭代的墙钟速度不会改变这个折扣。
32 步 rollout 有末端 value bootstrap，不代表只能学习半秒任务。
D-Peg 继承该 PPO 类，现显式固定自己的 `gamma=0.99`。其他训练器的折扣未在本次修改。

奖励按有效分数与阶段预算核算，而非直接比较 `weight`。事件和进度函数返回 `credit/step_dt`，
预算型连续奖励先按秒积分并封顶，再由 RewardManager 的乘 `step_dt` 得到该步分数。

| 项目 | 当前有效奖励 | 每回合上限 / 触发条件 |
| --- | --- | --- |
| 接近外壳的新最佳进度 | 1 × 接近分数增量 | ≤1 分，回退不能重复领 |
| 旋出前接触引导 | 0.25 分/秒 × 连续接触质量 | ≤1 分；旋出完成后停止 |
| 旋出进度 | 2 分/实际新增圈 | 默认三圈 ≤6 分 |
| 释放后连续抓稳 | 一次性 2 分 | 物理释放后严格双侧接触连续 0.3 秒 |
| 释放后稳定抓持 | 0.5 分/秒 × 连续稳定质量 | ≤2 分，允许合理搬运速度 |
| 搬运位置跟踪 | 目标接近分数 × 双侧接触质量，逐控制步积分 | 使用 `std=0.2`、权重0.5的连续塑形，20 秒理论上限约10分 |
| 最终位置成功 | 一次性 10 分并结束回合 | 进入目标位置 3 cm 内并连续保持 1 秒 |
| 掉落 / 释放后越界 | 一次性 −6 分 | 同一步重叠仅扣一次，不惩罚单纯超时 |
| 动作 / 动作变化正则 | 每项 `−0.001 × clamp(sum(square), max=1000)` 分/秒 | 每项最大扣 1 分/秒 |

原 40 分/圈、旋出瞬间额外 20 分已由上述预算替代。静态接近和重复的严格接触常驻项权重设为0；
保留接近项配置对象，供共享 Revo3 mixin 设置指尖实体。搬运项改为仿照 v2-test 的
`1-tanh(distance/std)` 连续位置跟踪，`std=0.2`、权重0.5，理论累计上限约10分，只在释放后并保持双侧接触时计分；
RewardManager 按控制步积分，因此接近目标的每一步都能提供梯度。

旋出进度仍取角度增量 `/2π` 和轴向增量 `/6mm` 的较小值，按每个环境历史最高值计分。
最多 0.2 秒的线性衰减接触记忆、无接触进度不补领、释放后空载螺杆不计分等约束保留。
接触质量仍由拇指与最强其他指的接触力计算，1N 饱和；成功和释放后抓稳里程碑仍要求双侧 `>1N`。
搬运只奖励释放后的有效跟踪；失败或失去双侧接触时该项为零。

稳定塑形要求释放且灯泡原点高于灯座原点15mm，平移速度不超过0.2m/s时不扣稳定质量，
超过后连续衰减；角速度以2rad/s为尺度连续衰减。释放后抓稳里程碑进一步要求线速度
小于0.2m/s、角速度小于1rad/s。最终成功改为要求灯泡距目标位置在3cm内并连续保持
1秒；物理释放仍作为阶段门控，抓持、离座和速度不再作为成功终止的硬门槛，失败标志仍会抑制成功奖励。
稳定预算和成功奖励独立记录，一次性奖励不会因失去后重新抓住而重复支付。
最终成功后当步发奖并结束回合，避免奖励发完后剩余动作成本诱导松手。
终止判断实时读取成功奖励的最终参数，Hydra覆盖不会使两边阈值分叉；回合成功日志使用
重置前的终止标志，避免场景重置清除成功状态后把成功误记为0。

改动不增加观测或动作维度；旧 PPO 权重结构兼容，但 critic 需要适应新的回报尺度与折扣。
本次只修改本地，服务器正在运行的训练尚未使用这些设置。正式采用时需要同步后新建运行或明确恢复运行，
并核对实际 `params/env.yaml` 和 `params/agent.yaml`。新旧奖励总数不能直接比较，
应评估真实最终成功率、释放后持有时间与掉落率；这些预算是待训练验证的初始选择。
细节及验证见[奖励平衡说明](rotate_bulb_reward_balance_2026-09-12.md)。

## 预训练触觉编码器

`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0` 的训练和播放配置默认直接读取
`runs/revo3_cross_modal_restoration_v1/best.pt`。无需先导出 encoder，也无需先完成
CLIP 对齐。当前权重来自第 112 epoch，`d_model=256`，SHA-256 为
`c8e30bd4005aefba60d18aae46f992b722088d05a2cb9cd2ec66b316181c632a`。

每指输入为 RGB `[3,240,320]`（0–255）、Depth `[1,240,320]`（米）和
100 个二维 marker `[x0_px,y0_px,dx_px,dy_px,valid]`。输入使用采集链路的
标定、RGB 量化和 checkpoint 中的归一化参数；HydroShear 的三维位移先投影为
像素位移。传感器顺序 middle/index/ring/pinky/thumb 显式转换成编码器约定的
little/ring/middle/index/thumb。

预训练权重严格加载，encoder 保持 `eval()`、`requires_grad=False`，PPO 只更新
Actor/Critic。默认输出 L2 归一化的 `h`，每指 256 维；没有随机 projection，
没有 sim–real 对齐。缺失权重、Git LFS 指针或不兼容权重会报错，不会回退到旧 ResNet。

| PPO 观测组 | 默认维度 | 内容 |
| --- | ---: | --- |
| policy | 43 | 物体姿态、目标、上一动作、螺纹进度和阶段 |
| proprio | 1714 | 接触、285 维压力、关节和指尖状态、1280 维五指 encoder 特征 |
| perception | 192 | 物体点云 |
| 合计 | 1949 | 当前帧，Actor/Critic 使用相同三组 |

旧的两个 ResNet 特征和 1500 维原始 HydroShear 不再作为策略输入。
压力阵列、任务状态、动作和奖励保持原有定义。默认并行环境数为 32；8GB 显卡
建议先用 4 个环境测显存和速度。Lift 和 Direct Repose-Cube 的观测配置不受此次切换影响。

## 训练与恢复

从当前仓库根目录，在安装了 Isaac Lab 的环境中运行。显式设置 `PYTHONPATH`
可避免本机 editable install 指向另一个 BrainCo checkout。

```bash
conda activate brainco
export PYTHONPATH="$PWD/source/BrainCo_DexHand${PYTHONPATH:+:$PYTHONPATH}"

# 如果本地 best.pt 仍然是 Git LFS 指针，先运行：
# git lfs pull --include='runs/revo3_cross_modal_restoration_v1/best.pt'

OMP_NUM_THREADS=2 python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0 \
  --num_envs 4 --max_iterations 3 --headless \
  --logger tensorboard --run_name bulb_pretrained_smoke

# 替换成上一条命令输出的实际运行目录名。
OMP_NUM_THREADS=2 python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0 \
  --num_envs 4 --max_iterations 2 --headless \
  --logger tensorboard --run_name bulb_pretrained_resume \
  --resume --load_run <run_directory_name> --checkpoint model_2.pt
```

不再需要关闭 RGB 或手动安装 `torch_scatter` 来绕过导入：该任务与 Direct 路径
共享基于 PyTorch `scatter_reduce_` 的 Taxim 兼容实现。

可覆盖 `env.tactile_policy_checkpoint=/absolute/path/to/best.pt`。
`env.tactile_encoder_chunk_size=8` 和 `env.tactile_taxim_chunk_size=8` 控制分块大小；
增减分块不改变观测定义。`env.tactile_reconstruction_diagnostics=True` 可另外计算
恢复结果，保存在 `env.latest_tactile_reconstruction`，不会将恢复图再次输入 encoder。

RSL-RL 的 train/play 会保存并校验 `tactile_policy_contract.json`，记录实际观测布局、
特征模式、权重 SHA-256 和传感器标定。旧 ResNet 策略与新输入不兼容，需要新建运行；
改变 encoder 权重或特征定义后也不能直接恢复原 PPO。原始或导出的 encoder 文件
需与 PPO checkpoint 一起保留；PPO 文件本身只保存 Actor/Critic。

未来可以将 `tactile_policy_checkpoint` 指向完成对齐后导出的 sim policy bundle，
输入维度会自动解析，但须重新训练对应的策略。当前 Actor 仍包含仿真物体状态和
点云，这些输入的真机获取不由 CLIP 对齐解决。

## 验证

```bash
OMP_NUM_THREADS=2 python -m pytest -q \
  tests/test_rotate_bulb_tactile_encoder.py \
  tests/test_tactile_policy_encoder.py \
  tests/test_revo3_rotate_bulb_task.py \
  tests/test_rotate_bulb_rewards.py \
  tests/test_direct_repose_visuotactile_task.py
```

测试覆盖原始权重与导出权重特征一致性、严格加载和冻结、传感器单位和手指顺序、
每步缓存、局部 reset、运行信息序列化和恢复兼容校验。短训练只能证明工作流可执行，
不能证明任务收敛或真机迁移效果。

2026-09-12 较早的三圈默认配置变更已分别通过训练和 PLAY 的 GPU 物理验证：各使用两个环境，
不传圈数或时长覆盖参数，确认当时的默认值为 `[3.0,3.0]`、`30.0 s`，并通过启动/部分重置、
5 秒静置、约束保持、受控旋出释放、重力和掉落检测。任务配置与奖励的 20 项现有测试通过。
此次验证检查机械能力，不代表旧浅起点策略已学会三圈任务。原始报告和日志保存在
`logs/diagnostics/rotate_bulb_three_turn_20260912/`。

同日随后将训练和PLAY的默认回合上限改为20秒。77项相关测试通过；两种模式分别使用两个
GPU环境、不传圈数或时长覆盖参数完成物理验证，实际配置均为三圈、20秒，保留完整触觉。
结果见[20秒配置验证汇总](../../logs/diagnostics/rotate_bulb_episode20_20260912/validation_summary.json)。
此次仅修改本地默认值，尚未部署到服务器；20秒与15秒的策略成功率仍需训练对照。
