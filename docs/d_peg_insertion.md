# D 形柱插入任务

任务 ID：`BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0`。

另有完全隔离的 Flexiv 场景变体：
`BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0`。它保留本页的
`peg.usd`、`socket.usd` 和插入状态机，但使用 `RotateBulbCustom` 的桌面/灯光/传感器
场景以及 `rizon4_training.usda`（Flexiv Rizon4 + Revo3 右手）资产。该变体有独立的
28 关节动作映射、观测契约和 PPO 配置；其 Flexiv 预抓文件是
`assets/d_peg_insertion/pregrasp_flexiv.json`。当前保留编辑器导出
`20260913_142053_9886dc87` 的实测关节姿态，并加入小幅、仅手指的驱动预载；最大接触标注误差
仍为 18.43 mm、穿透 3.80 mm，状态为未通过物理验收的 `preload_candidate`，必须先做 Flexiv
物理预抓验收再开始长时间训练。

当前本地实现采用 v3 动作与奖励语义，任务 ID 保持不变。v1/v2 训练记录保留用于对照，v3 从独立目录重新训练。2026-09-13 本地诊断与验证见 [`d_peg_v3_local_fix_2026-09-13.md`](d_peg_v3_local_fix_2026-09-13.md)。本次修改尚未同步到服务器。

每个回合从已抓住柱体的姿态开始，控制机械臂和灵巧手对准 D 形孔、插入并稳定保持。柱体是受重力作用的自由刚体；回合内没有固定关节、位姿传送或自动插入控制。PPO 从头训练，仅复用原多模态恢复 encoder。

## 模型与坐标

资产位于 [`assets/d_peg_insertion`](../assets/d_peg_insertion)：`peg.usd`、`socket.usd` 用于仿真；`cad/*.step` 和 `meshes/*.stl` 用于加工或打印。USD 单位为米，STEP/STL 单位为毫米。几何尺寸、坐标与资产指纹记录在 `metadata.json`，图片见 [`preview.png`](../assets/d_peg_insertion/preview.png)。完整场景见 [`scene_preview.png`](../assets/d_peg_insertion/scene_preview.png) 和可旋转查看的 [`scene_preview.glb`](../assets/d_peg_insertion/scene_preview.glb)。

| 项目 | 定义 |
| --- | --- |
| D 柱轴段 | 圆弧半径 12 mm、平面 x=6 mm、轴段长 35 mm、尖端倒角 1 mm |
| 抓持段 | 直径 50 mm、高 50 mm，位于柱体局部 z=35–85 mm |
| 孔座 | 80×80×50 mm，孔深 30 mm，单边几何间隙 0.75 mm，孔口倒角 1 mm |
| 柱体坐标 | 原点为尖端，轴向 +Z，D 形平面法向 +X |
| 孔座坐标 | 原点在底面，孔口 z=50 mm、孔底 z=20 mm |
| 目标插入 | 深度 28 mm，尖端距盲孔底面 2 mm |
| 桌子 | 中心 `(0.55,0,0.38)` m，尺寸 `(1.2,1.6,0.76)` m |
| 机械臂根 | `(1,0,0.766)` m，绕 Z 轴 −90°，固定底座 |
| 孔座重置 | 名义原点 `(0.45,0.10,0.76)` m，XY 各 ±5 mm，yaw ±10° |

每个并行环境独立随机孔座。柱体与关节初始状态来自 `pregrasp.json`，其位置使用环境局部坐标、四元数使用 `wxyz`。该文件包含全部 28 个关节的实际位置，以及可选的独立 `robot_joint_targets` 夹持目标。缺失文件、缺少关节或无效位姿会报错，不会回退到未抓持状态。

重置只写选中环境的物体/孔座/关节状态并清理历史；离线校准中的物理稳定过程不会放进训练重置。v2 的动作以固定的 `robot_joint_targets` 为参考，零动作持续保持经校准的重力补偿和夹持预载。

孔座采用底面和显式凸分块侧壁碰撞体，保留实际凹孔。柱体提供独立闭合 `TactileSurface`，供指尖压力 SDF 和 TacMap 使用；它与视觉表面一致，没有额外碰撞。`scene["object"]` 指向柱体，`scene["socket"]` 指向固定孔座。

## 动作、观测与网络

动作保持 28 维：7 个机械臂关节、21 个手部关节。v3 使用预抓驱动目标上的位置残差：`q_target = q_pregrasp_drive_target + scale × action`，机械臂 scale=0.1、手指 scale=0.02；v2 的全部关节 scale=0.1。参考值不随实际关节偏移改变，连续残差不积分；策略仍可调整机械臂和手指，也可主动松开柱体。动作没有硬裁剪，不改变 PPO 高斯输出或对数概率。物理频率 240 Hz，策略频率 30 Hz（`decimation=8`），回合长 15 秒，最多 450 个控制步。环境类默认 32 个并行环境；原服务器 v2 已通过每卡 512、共 1024 个环境的双卡容量验证。

| 观测组 | 维度 | 内容 |
| --- | ---: | --- |
| policy | 43 | 柱体姿态 4、最终目标 pose 7、上一动作 28、插入进度 2、插入阶段 2 |
| proprio | 434，开启还原通路时 1714 | 指尖接触力 15、压力阵列 285、关节位置/速度各 28、指尖状态 78；可选五指 encoder 特征 1280 |
| perception | 192 | 柱体表面点云 64×3，表达在机器人根坐标系 |
| 合计 | **669**，开启还原通路时 **1949** | 当前帧，Actor/Critic 使用相同三组 |

插入进度为归一化深度和横向偏差；阶段为满足对准条件的标志及剩余时间比例。最终目标 pose 由实际孔座位姿计算，包含完整方向，回合内不重采样。柱体姿态和点云为仿真完整状态。

`tactile_policy_enabled=False` 为训练和 PLAY 的共同默认值。关闭时该观察项返回零列，不加载还原 checkpoint，也不生成 RGB/Depth/marker 或执行 encoder/重建诊断；接触力、285 维压力阵列和其他状态观察继续保留。开关在观察管理器创建时读取，因此支持配置构造后的 Hydra 覆盖。

在 `train.py` 或 `play.py` 命令末尾添加 `env.tactile_policy_enabled=true` 可打开通路。打开时默认读取 `runs/revo3_cross_modal_restoration_v1/best.pt`。每指输入 RGB `[3,240,320]`、以米为单位的 Depth `[1,240,320]` 和 100 个二维 marker。encoder 保持 `eval()`、无梯度，每指输出 256 维归一化 `h`；不需要 CLIP 对齐权重，也不加载灯泡 PPO。已训练策略的开关必须与其保存的契约一致。

Actor/Critic 均为 `[512,256,128]` MLP、ELU，带观测归一化。初始动作标准差为 `0.15`；从头训练时将 Actor 最后一层权重与偏置置零，使初始确定性动作精确保留预抓目标，再叠加探索噪声。恢复训练和 PLAY 不重新初始化 Actor。D 柱专属 PPO 初始学习率设为 `3e-5`，继续使用 adaptive 调度，折扣因子保持 `gamma=0.999`，GAE lambda 保持 `0.95`。折扣按控制步计算，在 30 Hz 下延迟 15 秒的奖励保留约 `0.999^450 ≈ 63.7%` 的权重。策略和算法配置均独立复制，灯泡配置不受影响。独立运行目录为 `dexsuite_revo3_insert_d_peg_v3`。

检查点每 25 次迭代保存一次。按 1024 环境容量测试中约 76 秒/迭代估算，保存间隔约 32 分钟；这项配置只改变文件保存频率，不改变 PPO 更新。

`tactile_policy_contract.json` 校验任务 ID、669/1949 维实际布局，以及柱体、孔座、几何元数据和预抓文件；开启还原通路时还校验 encoder 和传感器标定指纹。`task_schema_version=3` 固定描述本地新奖励状态机，`action_schema` 记录实际关节顺序、预抓驱动目标、动作公式与各关节组 scale。v1/v2 D 柱、灯泡及开关不同的 checkpoint 均不能直接恢复；需要 v3 从头训练。几何元数据在配置和启动阶段均校验；本版本不接受仅修改目标深度而保留原成功阈值的配置。

## 奖励与成功

任务在孔座坐标系内计算横向偏差、轴倾角、D 形绕轴角及真实轴段与孔壁的几何关系。只有经历正确入孔过程，才能进入插入完成阶段。

| 奖励项 | 计分方式 |
| --- | --- |
| 靠近孔口 | 目标为实际孔口 z=0，距离尺度 30 mm；新增接近进度 ×5 |
| 对准 D 形孔 | 横向、倾角、D 形方向及孔口以上高度的联合评分；新增对准进度 ×8 |
| 精细对准 | 横向尺度 1.5 mm、倾角/绕轴尺度 5°、孔口以上高度尺度 10 mm；新增最好进度 ×8，入孔 2 mm 后仍有效 |
| 有效抓持丢失 | 未满足原有抓持条件时 −0.2/秒，不添加原地抓持的正奖励 |
| 入孔里程碑 | 正确入孔至少 2 mm 后一次 +5 |
| 插入深度 | 新增归一化深度 ×40 |
| 稳定成功 | 一次 +80 |
| 掉落、出界、非有限状态 | 一次 −20，并终止回合 |
| 过大孔座接触力 | 超过 30 N 后连续惩罚，最大 −2/秒 |
| 时间 | −0.1/秒 |
| 动作及动作变化 | 沿用 `action_l2_clamped`、`action_rate_l2_clamped`，权重各 −0.005 |

接近、对准、入孔和深度奖励要求实际抓持：拇指和至少另一指的柱体接触力均超过 1 N。历史最好进度按环境独立保存，回退或重新抓取不会重复领取旧进度；无抓持通过孔口也会消耗已发生的里程碑。进度奖励按控制步长换算，避免改变控制频率时改变总计分。

v3 将“实际从孔口进入的历史”和“当前满足几何包含”分开保存。短暂不满足孔壁包含条件时，当前帧不给插入或稳定成功信用；重新对准后可继续领取新增深度。退出孔口、离开孔口区域、穿底或回合结束会清除入孔历史，初始/侧向置入仍不被承认。精细对准同样使用一次性的历史最好进度预算，孔外下沉、无抓持移动或重复进退不能重复领分。

v2 将接近目标从孔口上方 20 mm 移到实际孔口，并将接近距离尺度设为 30 mm；对准评分的高度项也使用距孔口的正高度、尺度 30 mm，使已对准柱体继续下降到入口时仍能得到有界进度奖励。孔口下方仍需真实轮廓包含和正确入孔过程，无抓持或孔外下沉不能获得插入分。

成功要求正确入孔、继续抓持，插入深度 27–30 mm，横向误差小于 0.375 mm，轴倾角和 D 形绕轴误差均小于 3°，柱体线速度小于 0.02 m/s、角速度小于 0.2 rad/s，并连续满足 0.5 秒（30 Hz 下为 15 个控制步）。桌面接触力超过 0.5 N 判定为掉落；越界范围为环境局部 X `[0.1,0.8]`、Y `[-0.35,0.55]`、Z `[0.60,1.35]` m。

## 本地短训练与回放

在仓库根目录、已安装 Isaac Lab 的 `brainco` 环境中运行。预抓文件必须已完成实际重力和接触校准。

```bash
conda activate brainco
export PYTHONPATH="$PWD/source/BrainCo_DexHand${PYTHONPATH:+:$PYTHONPATH}"

OMP_NUM_THREADS=2 python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0 \
  --num_envs 4 --max_iterations 3 --headless --device cuda:0 \
  --logger tensorboard \
  --log_dir logs/rsl_rl/dexsuite_revo3_insert_d_peg_v3/local_smoke

OMP_NUM_THREADS=2 python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0 \
  --num_envs 4 --max_iterations 2 --headless --device cuda:0 \
  --logger tensorboard --resume --load_run local_smoke --checkpoint model_2.pt \
  --log_dir logs/rsl_rl/dexsuite_revo3_insert_d_peg_v3/local_resume

python scripts/rsl_rl/play.py \
  --task BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0 \
  --num_envs 4 --device cuda:0 \
  --checkpoint logs/rsl_rl/dexsuite_revo3_insert_d_peg_v3/local_resume/model_3.pt
```

最后一条命令打开交互回放，需主动关闭窗口。当前训练和 PLAY 任务配置相同；下面的验证工具显式分别创建训练和 PLAY 类，并提供有限步数的无界面回放。

## 完整短验证

```bash
OMP_NUM_THREADS=2 python -m pytest -q \
  tests/test_d_peg_actions.py tests/test_d_peg_task_integration.py \
  tests/test_d_peg_ppo_cfg.py tests/test_d_peg_insertion_rewards.py

python assets/d_peg_insertion/tools/validate_assets.py \
  --report logs/diagnostics/d_peg_assets.json

OMP_NUM_THREADS=2 python assets/d_peg_insertion/tools/validate_pregrasp.py \
  --headless --num-envs 4 --verify-restored --settle-s 0 \
  --seed assets/d_peg_insertion/pregrasp.json \
  --report logs/diagnostics/d_peg_pregrasp.json

OMP_NUM_THREADS=2 python assets/d_peg_insertion/tools/validate_insertion_physics.py \
  --headless --device cuda:0 \
  --report logs/diagnostics/d_peg_insertion_physics.json

OMP_NUM_THREADS=2 python assets/d_peg_insertion/tools/validate_training.py \
  --headless --device cuda:0 \
  --output logs/diagnostics/d_peg_full_validation
```

`validate_training.py` 默认串行启动独立仿真进程：训练/PLAY 各两个环境、各 20 次完整重置和局部重置检查；四环境 PPO 训练 3 次迭代；新进程精确加载权重和优化器后续训 2 次迭代；最后由新 PLAY 进程回放 300 步。默认关闭模态还原通路，并断言没有创建相关运行时；传 `--tactile-policy-enabled` 可验证开启模式，包含 encoder 冻结和输入检查。两种模式均保留压力阵列与接触力。

验证检查动作/观测维度、每步数值有限、权重和优化器保存加载一致、PPO 权重确有更新、冻结 encoder 的状态哈希不变，以及局部重置后重新计算触觉观测仍不改变未重置环境的历史。结果写入输出目录的 `report.json`，各阶段另有 `report.json` 和 `console.log`。重复验证请使用新目录，避免覆盖检查点。

支持 `--mode reset_train`、`reset_play`、`train`、`resume`、`play` 单独执行；`resume/play` 必须提供 `--checkpoint`。例如有界回放：

```bash
python assets/d_peg_insertion/tools/validate_training.py \
  --mode play --num-envs 4 --play-steps 300 --headless --device cuda:0 \
  --checkpoint logs/diagnostics/d_peg_full_validation/resume/checkpoint.pt \
  --output logs/diagnostics/d_peg_play_recheck
```

短验证证明流程和数值稳定性，不代表策略已经学会插入。更换资产、预抓或观测契约后，应重新运行完整验证。

## v2 已完成的本地验证

使用正式 `env.step()` 动作路径、4 个环境、完整触觉、零残差连续运行 2 秒，4 个环境均通过。最大位置漂移 1.8931 mm、姿态漂移 2.8074°，每个环境的有效抓持帧比例均为 100%，拇指最小接触力 1.1709 N，零动作驱动目标与预抓目标的最大差值为 0。报告见 [`zero_control/report.json`](../logs/diagnostics/d_peg_v2_local/zero_control/report.json)。这项检查证明新动作控制能够实际保留夹持预载。

标准差为 0.15 的随机残差诊断中，4 个环境连续运行 2 秒，均没有掉落或重置，有效抓持帧比例为 85.83%–93.33%。动作带来的位置变化为 5.65–10.62 mm，未满足静态 3 mm 和连续接触验收标准，原始报告的 `passed` 为 false；该结果仅作为探索行为诊断，不计作静态预抓验收通过。

动作、契约、奖励、PPO 初始化与诊断及灯泡相关 CPU 回归在本地和服务器均为 **105 项通过**。其中真实 Isaac `configclass` 检查覆盖父配置实例复制和策略/算法字段隔离。反作弊检查仍覆盖错误 D 形方向、孔外下沉、盲孔底穿透、无握持入孔与重新抓握补领、重复进退、短暂稳定、高速经过及控制周期变化。

修复 PPO 配置导入后，v2 本地完整五阶段流程全部通过：训练和 PLAY 各 2 个环境的 20 次完整重置及部分重置检查、4 个环境训练 3 次 PPO 更新、恢复后更新 2 次、300 步播放。报告见 [`full_pipeline_fixed/report.json`](../logs/diagnostics/d_peg_v2_local/full_pipeline_fixed/report.json)。

初次 3 次本地 PPO 更新使用学习率 `1e-3`：第 0 次更新的首 minibatch KL 为 0.000280、裁剪比例为 0，而整次更新 KL 达到 0.176、裁剪比例为 0.7125，学习率降至 `1e-5`。随后两次更新的首 minibatch KL 分别为 0.001439、0.000685，裁剪比例均为 0。这些记录更支持初始优化更新过大，因此将初始学习率改为 `3e-5`。调整后补充的 3 次更新也通过数值与训练检查，第 0 次平均 KL 降至 0.02747；但 4 环境的小批量测试中学习率仍触及 `1e-5` 下限，尚不能认为该问题已完全解决，也不据此宣称插入成功率提升。

v2 本地验证汇总及各次 PPO 诊断见 [`v2_local.json`](../assets/d_peg_insertion/validation/v2_local.json)。

## v1 本地验收历史

以下为修改动作与奖励前完成的 v1 验收。资产和预抓保持相同；训练检查点的动作语义与 v2 不兼容。

2026-09-12 已固定最终 `pregrasp.json`，柱体尖端初始位置为 `(0.45,0.10,0.84)` m，初始方向为单位四元数。最终姿态在 4 个环境中按真实重力保持 2 秒，最大位置漂移 1.901 mm、最大角度漂移 2.808°，拇指最小接触力 1.164 N；详见 [`pregrasp_final.json`](../assets/d_peg_insertion/validation/pregrasp_final.json)。沿 58 mm 插入行程的几何间隙检查中，手部到桌面最小间隙 3.112 mm、到孔座最小间隙 2.725 mm，详见 [`clearance_final.json`](../assets/d_peg_insertion/validation/clearance_final.json)。

独立物理验证中，对准柱体能够达到 27.8966 mm 插入深度；横向偏移 5 mm 和 D 形方向反转 180° 的反例均被实际孔壁阻挡。记录见 [`insertion_physics_validation.json`](../assets/d_peg_insertion/insertion_physics_validation.json)。这些检查验证资产和预抓的物理可行性。

完整触觉链路的五阶段验证全部通过，记录见 [`logs/diagnostics/d_peg_full_validation/report.json`](../logs/diagnostics/d_peg_full_validation/report.json)：

| 阶段 | 实际配置与检查 | 耗时 |
| --- | --- | ---: |
| 训练环境重置 | 2 环境、20 次完整重置，部分重置及随后重算观测均保持未重置环境历史不变 | 13.1 秒 |
| PLAY 环境重置 | 2 环境、同样 20 次完整重置和部分重置隔离 | 12.9 秒 |
| 从头短训练 | 4 环境、3 次 PPO 迭代，策略参数更新、checkpoint 保存一致 | 32.5 秒 |
| 新进程续训 | 4 环境、2 次迭代，模型与优化器精确恢复、contract 匹配、参数继续更新 | 24.7 秒 |
| 新进程回放 | 4 环境、300 步，加载相同 contract，策略参数保持不变 | 74.8 秒 |

各阶段均确认动作 28 维、观测分组 `43/1714/192`，传感器和 PPO 数值有限，冻结 encoder 的状态哈希前后相同。加上部分触觉缓存与实际输入诊断，相关 CPU 回归共 84 项通过。实际触觉信号见 [`local_tactile_signals.json`](../assets/d_peg_insertion/validation/local_tactile_signals.json)：接触指具有非零深度和 marker 位移，五指 latent 的 L2 范数为 1。此时策略仅经历流程短训，尚无学会插入的成功率结论。

预抓验收从直接恢复状态开始计算漂移；接触传感器清零后用第一个正常控制帧（16.7 ms）填充报告，再连续检查 2 秒接触力。这个采样等待只发生在离线验收工具中，任务 reset 不执行物理步进。

## v2 服务器双 GPU 容量验证与训练

服务器仓库 `/home/admin/workspace/xinyu/mulyi-model-tac` 已通过每卡 512、双卡共 **1024 个环境**的 3 次 PPO 更新容量验证，共采样 98,304 条状态转移。完整五指触觉保留原模态、240×320 分辨率和推理 chunk 配置，没有为扩大环境数降低观测内容。

| 容量测试迭代 | 耗时 | 平均 KL | 更新后 adaptive 学习率 |
| --- | ---: | ---: | ---: |
| 0 | 87.04 秒 | 0.01039 | 3.417×10⁻⁴ |
| 1 | 77.10 秒 | 0.02148 | 1×10⁻⁵ |
| 2 | 75.84 秒 | 0.01771 | 1×10⁻⁵ |

第 0 到第 2 次保存模型的 Actor 参数最大变化为 0.00848484，确认策略确有更新。NCCL 首次梯度均值检查通过，全部 3 次更新后的两卡参数最大差异均为 0。观测仍为 1949 维、动作 28 维、契约 v2。报告见 [`v2_server_capacity.json`](../assets/d_peg_insertion/validation/v2_server_capacity.json)。

容量测试证明该规模下仿真、完整触觉、PPO 更新和跨卡同步能够执行；尚无插入成功率结论。adaptive 学习率在第 1、2 次迭代仍回到 `1e-5` 下限，需结合后续训练的 KL 和阶段统计继续观察。

以下命令可在新目录复现同样容量的从头训练：

```bash
cd /home/admin/workspace/xinyu/mulyi-model-tac
export PYTHONPATH="$PWD/source/BrainCo_DexHand${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/admin/workspace/xinyu/IsaacLab/_isaac_sim/python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0 \
  --distributed --expected_world_size 2 --num_envs 512 --headless \
  --logger tensorboard --max_iterations 15000 \
  --log_dir logs/rsl_rl/dexsuite_revo3_insert_d_peg_v2/dual_gpu_1024_from_scratch
```

`train.py` 在每个进程直接设置 `env_cfg.scene.num_envs`，没有除以 world size；因此此命令使用 `--num_envs 512`，表示**每卡 512、双卡合计 1024 个环境**。省略此参数则采用任务默认值每进程 32，双卡合计 64。每个进程根据 local rank 选择 GPU，并对 seed 加 rank。`--expected_world_size 2` 会检查实际进程数及 PPO 梯度同步。两卡使用共同 `--log_dir`，v2 从头训练不要传 `--resume`、v1 或灯泡 checkpoint。

`bash scripts/rsl_rl/launch_d_peg_dual_gpu.sh <新的日志目录>` 提供同样的双卡入口。`D_PEG_ENVS_PER_GPU=512` 指定每卡环境数，`D_PEG_MAX_ITERATIONS=3` 可用于短程容量验收。保持全部五指 RGB、Depth、marker、冻结特征与压力输入；encoder 和 Taxim 的推理 chunk 可按显存调整，chunk 大小不改变观测契约。

v2 另记录 `d_peg_training.jsonl`，包含两卡汇总的 PPO KL、裁剪比例、探索动作、观测归一化变化及回合统计，用于区分抓持、对齐、下降、入孔各阶段的变化。

通过容量验证后，最终正式训练已于 **2026-09-12 16:02:30（北京时间）**启动，PID 为 `13014`，使用 GPU 0/1，每卡 512 环境，总计 1024 环境，计划 15000 次迭代，每 25 次迭代保存检查点。该运行从头初始化 PPO，没有恢复 v1 或容量测试检查点。正式目录为：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/dexsuite_revo3_insert_d_peg_v2/20260912T080230Z_dual_gpu_1024
```

启动参数归档见 [`v2_server_training_launch.json`](../assets/d_peg_insertion/validation/v2_server_training_launch.json)。正式训练进度查看目录内的 `console.log`，优化与回合统计查看 `d_peg_training.jsonl`，两卡同步证据查看 `distributed_validation.json`；容量验证的 3 次更新与此正式运行分别记录。

北京时间 16:08 的检查确认正式训练已完成第 0–3 次更新，进程仍在运行，`model_0.pt` 已保存，日志无运行错误。正式运行前 3 次更新的两卡参数最大差异均为 0；审计快照见 [`v2_formal_training_audit.json`](../assets/d_peg_insertion/validation/v2_formal_training_audit.json)。当前尚无有效入孔或成功事件，早期结果只证明训练可执行。

本次服务器同步的 16 个文件及保存频率调整后的补充 7 个文件均通过 SHA256 校验，补充同步的覆盖前备份标识为 `20260912T080230Z`；既有日志及检查点均保留。

2026-09-12 只读核查确认，上述服务器 Python 入口解析到 `/isaac-sim/python.sh`，提供 Python 3.11.13、PyTorch 2.7.0+cu128、Isaac Lab 0.54.2 和 RSL-RL 3.1.2。服务器短验证也用这个入口替换本地命令中的 `python`。已有 Pixi 环境的 `activate_brainco.sh` 指向另一个旧项目，不用于本任务。

## v1 服务器验证与训练历史

以下报告来自 v1 的 32 环境双卡运行，不能作为 v2 或 1024 环境容量已通过的证据。

v1 服务器已完成 84 项 CPU 回归、4 环境重力预抓、三工位力控插入和完整五阶段训练/恢复/播放验证，结果均通过。报告见 [`server_pregrasp.json`](../assets/d_peg_insertion/validation/server_pregrasp.json)、[`server_insertion_physics.json`](../assets/d_peg_insertion/validation/server_insertion_physics.json) 和 [`server_training_summary.json`](../assets/d_peg_insertion/validation/server_training_summary.json)。服务器预抓最大漂移 1.907 mm、2.818°，拇指最低接触力 1.154 N。

另以正式 `train.py` 入口进行了双卡 3 次 PPO 更新检查：NCCL 后端、每卡 16 环境，首次梯度等于两卡均值，每次更新后的两卡参数差异均为零。完整日志位于服务器 `logs/diagnostics/d_peg_server_validation/`。

v1 正式训练于 2026-09-12 13:50:13（北京时间）从头启动，原计划 15000 次迭代。历史启动进程 PID 为 `5106`，使用 GPU 0/1，总计 32 环境。历史目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/dexsuite_revo3_insert_d_peg/20260912T055013Z_dual_gpu
```

历史 `console.log` 记录训练进展，`distributed_validation.json` 保存前 3 次更新的跨卡同步证据，`tactile_policy_contract.json` 固定 v1 运行契约，`model_*.pt` 为旧检查点。该目录下的 `launch.json` 保留当时的实际启动命令。

v1 训练已于 2026-09-12 15:41:19（北京时间）停止，当时运行到第 992 次迭代；保留第 0、250、500、750 次的检查点及全部日志。停止记录为旧运行目录内的 `stop.json`，v2 使用新目录从头训练。

同步使用 SHA256 对比和覆盖前备份，没有删除服务器已有文件。初次与工具补充备份分别保存在 `/home/admin/workspace/xinyu/.codex-sync-backups/d_peg_20260912T054002Z` 和 `d_peg_20260912T054850Z`。短程结果证明环境与训练流程可执行，长期成功率需要继续评估。
