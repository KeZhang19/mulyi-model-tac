# Rotate-Bulb-Custom

任务 ID：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`。

这是 Rotate-Bulb 的独立配置副本，使用 **Flexiv Rizon4 + Revo3 右手**。场景和初始关节姿态现在对齐抓取编辑器导出的 `20260912_102202_ed803448`；控制频率、回合长度及网络隐藏层参数沿用原配置。Custom 的触觉观测开关、旋出奖励和圈数日志按下文独立配置。

2026-09-13 新增 `tactile_policy_enabled`，训练和 PLAY **默认均为 `False`**。
关闭时保留压力、关节、物体状态等其他观测；预训练模态还原模型不加载，
专用 RGB / Depth / marker 生成、编码推理及重建诊断均跳过，不向策略拼入触觉 latent。
观测管理器中的 `pretrained_tactile` 项此时宽度为 0；使用当前每指 256 维的 checkpoint
打开开关时，该项恢复为 5 × 256 = 1280 维。开关在创建环境时生效，不支持训练中途切换。

在现有训练或 PLAY 命令后追加即可恢复原来的多模态观测：

```bash
env.tactile_policy_enabled=true
```

显式关闭使用 `env.tactile_policy_enabled=false`。原来的
`env.tactile_reconstruction_diagnostics=false` 仅控制额外的解码重建诊断，默认仍关闭；
它不会启用上述总开关，只有两个开关都为 `true` 才执行重建诊断。
由于开关改变策略输入维度，续训或回放必须与训练时一致；已有多模态策略需要显式设为
`true`。RSL-RL 的观测契约会拒绝不匹配的设置，开启模式保留原契约以兼容已有策略。
修改默认值不会改变已运行的训练进程；新启动的训练才会使用关闭模式。

2026-09-12 已同步当前 Custom Rotate-Bulb 的奖励与 RSL-RL 参数：

| 参数 | Custom 任务 |
| --- | --- |
| 物理步进 / 控制频率 | 240 Hz / 60 Hz，decimation=4 |
| 训练及 PLAY 回合上限 | 20 秒，最多 1200 个控制步；成功、掉落提前结束 |
| 默认旋入深度 | 3 圈 |
| RSL-RL gamma / lambda / rollout | 0.999 / 0.95 / 32 步 |
| 接近 / 初始抓持 | 累计最多 1 分 / 1 分 |
| 三圈旋出 | 累计最多 6 分，无额外旋出完成奖；Custom 分为基础圈数与抓持圈数两项 |
| 释放后稳定 / 持续抓稳里程碑 | 累计最多 2 分 / 一次性 2 分 |
| 搬运引导 | 连续位置跟踪，权重 0.5，20 秒理论累计最多约 10 分 |
| 最终成功 | 进入目标 3 cm 内并持稳 1 秒后一次性 10 分，当步终止 |
| 释放后掉落或越界 | 一次性扣 6 分；重叠不重复，单纯超时不扣 |
| 动作幅度 / 动作变化正则 | Custom 默认 -0.0001 / -0.0002；标准任务仍各为 -0.001 |
| 指尖接触辅助项 | Custom 每指最高 0.02 分/秒，五指合计最高 0.1 分/秒；20 秒最多 2 分 |

成功统计同步读取终止标志，避免场景 reset 清空状态后漏记。奖励实现仍保存在 Custom
自己的模块中，可独立调整。Custom 的机械臂、抓姿、重置、场景、触觉与摩擦设置保持原值。
RL-Games 的两套配置原本彼此一致，仍保留 gamma=0.99；上表的 gamma=0.999 对应当前使用的
RSL-RL 入口。两种后端应分别与同后端任务比较。

2026-09-13 已将当前训练使用的低动作惩罚写入 Custom 的默认奖励配置：
`action_l2=-0.0001`、`action_rate_l2=-0.0002`，训练和 PLAY 均生效。
新启动无需再通过命令行覆盖这两项；已有训练的同值覆盖可以继续保留。

2026-09-13 新增持续辅助项 `any_finger_contact`，默认 `weight=0.1`、`force_scale=0.5` N。
随后按五根指尖分别计分：每指奖励率为 `0.02 × clip(该指接触力 / 0.5 N, 0, 1)`，再相加。
每根达到 0.5 N 即取得该指最高 0.02 分/秒；1 / 2 / 3 / 4 / 5 根分别最高
0.02 / 0.04 / 0.06 / 0.08 / 0.10 分/秒。低于 0.5 N 保留线性引导。
使用只过滤灯泡的接触力矩阵，碰桌面或灯座不计分；单指超过 0.5 N 不能代领其他指尖的份额。
与原有每回合最多 1 分的 `grasp_shaping` 独立，持续接触持续给分，旋出后仍可获得；
释放后掉落或越界当步停止给分。该项仅复用接触传感器，不依赖多模态观测开关。

这是每秒奖励，由 RewardManager 乘 `step_dt`；60 Hz 下每指最高每步约 `0.000333` 分，
五指合计最高每步约 `0.001667` 分。
关闭观测组在 2026-09-13 01:37 左右的动作惩罚日志为 `-0.0074 - 0.0200 = -0.0274`，
回合接近完整 20 秒；五指均满接触约为该平均强度的 3.6 倍，单指 0.02 尚不足抵消该均值，
两指 0.04 才高于它。实际贡献随各指接触强度和时长变化。
动作惩罚算子对平方和截断到 1000，两项理论最大合计仍为 0.3 分/秒；这里按当前典型强度
选取辅助权重，不按动作大小动态补偿。旋转圈数、成功奖励和动作惩罚权重保持原值。
TensorBoard 会新增 `Episode_Reward/any_finger_contact`，便于直接和两项动作惩罚比较。
可用 `env.rewards.any_finger_contact.weight=0.0` 关闭辅助项；源码同步后需重新创建环境才生效。

Custom 的 `unscrew_turns` 为每新增历史最佳有效圈数 0.5 分，`unscrew_progress`
为每新增圈数乘抓持质量再乘 1.5 分；充分抓持的三圈总预算仍为 6 分。
基础圈数不依赖接触，抓持项保留 0.2 秒换抓记忆；两项都受角度和轴向进度共同约束，
不奖励原地等待、回拧再转或释放后的空载螺杆运动。进度函数除以 `step_dt`，
由奖励管理器积分，因此权重表示每圈得分，不随控制频率改变。

RSL-RL TensorBoard 的 `Metrics/object_pose/` 下新增：

| 指标后缀 | 含义 |
| --- | --- |
| `unscrew_turns_max` | 已结束回合内最大有效旋出圈数的均值 |
| `unscrew_turns_final` | 已结束回合终止时有效旋出圈数的均值 |
| `unscrew_reached_1_turn` | 已结束回合曾达到 1 圈的比例 |
| `unscrew_reached_2_turns` | 已结束回合曾达到 2 圈的比例 |
| `unscrew_reached_3_turns` | 已结束回合曾达到 3 圈的比例 |

圈数日志独立于奖励权重和接触评分，包含成功、失败和超时回合，记录终止当步，
释放后冻结到释放前最后一次有效测量。每个结束回合贡献一个样本，RSL-RL 拼接后求均值，
避免不同大小的重置批次被等权平均；初始化不计入样本，没有结束回合的步不重复计数。
没有新结束回合的日志窗口不产生新的圈数点。多卡时沿用现有 RSL-RL 日志进程的采样范围，
这些指标不是额外做过跨卡聚合的全局统计。

比较旧 30 秒训练日志时，`Episode_Reward/*` 除以最大回合秒数，相同累计贡献在 20 秒配置
下会显示为原来的 1.5 倍。新旧对照应同时使用原始回合回报、成功率及掉落率。

摩擦此次保持不变。Custom 本地 GPU 启动后直接读取 PhysX 确认：灯泡外壳为静摩擦
0.9 / 动摩擦 0.8，螺纹为 0.15 / 0.10，灯泡质量为 0.20 kg。指尖实际仍采用
`robot_physics_material` 的 0.5–1.0 随机值；USD 柔顺材质中显示的 0.5 / 0.5 不代表
所有指尖最终都使用同一个系数。按材质声明的 `average` 合并模式估算，手指与外壳的接触系数约为
静摩擦 0.70–0.95、动摩擦 0.65–0.90，计算方式见
[PhysX 材质合并说明](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html)。
这些值不足以支持“外壳摩擦太低导致掉落”的结论；是否仍有夹持下滑移，需要与释放后的
对向接触和夹持力一起判断，暂不同时改变摩擦这一实验因素。

| 文件 | 修改内容 |
| --- | --- |
| `robot_asset_cfg.py` | `ROBOT_USD_PATH` 是本地机械臂与手部组合 USD 的入口；同文件包含源资产的驱动器、刚度、阻尼和物理属性 |
| `robot_contract.py` | 机器人根位置/朝向、28 个关节的初始值、掌心/指尖名称及触觉 link 映射 |
| `robot_setup.py` | 机械臂与手部关节名称、动作配置、末端 link、接触传感器、压力与视觉触觉参数和标定资产路径 |
| `env_cfg.py` | 训练/Play 环境、灯泡资产、奖励、重置、回合长度、触觉 encoder checkpoint 与推理批大小 |
| `base_env_cfg.py` | 基础场景、观测、事件、命令、终止条件和仿真配置 |
| `tactile.py` | 触觉编码器接入、手指顺序、观测缓存和新任务的观测契约 |
| `agents/rsl_rl_ppo_cfg.py` | PPO actor/critic 网络层、激活函数、观测分组、学习率与训练参数 |
| `agents/rsl_rl_distillation_cfg.py` | student/teacher 网络层和蒸馏参数 |
| `agents/rl_games_ppo_cfg.yaml` | RL-Games 网络结构和训练参数 |

这些文件在新任务命名空间内定义自己的配置；内部沿用原类名以保留复制对应关系，不导入原任务或 Lift 的环境、机器人和 agent 配置。Isaac Lab、通用 MDP 算子及底层传感器/编码器实现仍作为公共库使用。

机械臂与手部来自 `assets/rotate_bulb_custom/robot/usd/rizon4_training.usda`，USD 依赖层均已复制。抓取参考保存在 `assets/rotate_bulb_custom/grasp_reference/simulation_state.json`，运行不依赖编辑器或资产来源项目目录。`robot_contract.py` 按关节名读取全部 7 个机械臂和 21 个手部关节角，避免导出数组与 PhysX 原生顺序不同造成错位；重置位置/速度偏移为零。

参考坐标系为机器人底座坐标系，单位米/弧度，四元数顺序 wxyz。机器人根位置为 `(0, 0, 0)`，朝向为 `(1, 0, 0, 0)`；支架原点 `(-0.72, 0.04, 0.035042166926915974)`；灯泡原点 `(-0.72, 0.04, 0.047342166926915966)`。桌面高度为 `-0.009690000000000087`，尺寸和旋转也直接读取快照。地面移到桌底，观察视角和越界范围随场景调整。

启动、完整 reset 和单个环境 reset 均恢复该姿态，每个并行环境仅增加自身 `env_origin`，不会再随机移动支架。默认旋入 3 圈，螺纹转角 `-6π`、滑移 `-0.018 m`，还原相同的灯泡相对支架位姿。显式覆盖 `initial_screw_turns_range` 仍可改变深度，这时灯泡姿态会相应偏离该参考。

这里只移植场景与初始状态，仍使用 Isaac Lab 原生的螺纹约束、重力、接触和释放逻辑。快照自身的 `pose_candidate_pass=false`，接触最大距离约 19.07 mm、穿透约 3.86 mm；对齐后也不会自动变成已通过动态抓取验证的结果。开始施加动作/推进物理后允许正常运动。

自碰撞开关为 `True`，源资产的 `rizon4_collision_filters.usda` 完整保留，因此过滤表指定的机械臂/手部碰撞对仍被屏蔽。`robot/usd/config.yaml` 只是源 URDF 转换记录；运行使用 training USD 及 `robot_asset_cfg.py` 的属性。

生成机器人时先展开碰撞网格实例并应用原配置的接触偏移，再创建物理视图；这样触觉材质的启动回调不会删除已被视图引用的形状。关节启动/reset 同时恢复位置目标并清除旧速度/力矩目标。

动作仍为原来的 `RelativeJointPositionActionCfg(joint_names=[".*"], scale=0.1)`，关节位置/速度观测与动作均跟随新资产的同一原生关节顺序，共 28 维。未移植 v5 动作限幅、30 Hz 控制或蒸馏观测定义；当前仍为 240 Hz 物理步进、60 Hz 控制。接触传感器保留原逻辑名称，物理绑定转到新 link，掌心和指尖状态也绑定新资产。

保留现有触觉编码器及手指顺序 `middle/index/ring/pinky/thumb`。285 个压力 taxel、每指 100 个 marker 的数量与排列不变；新资产的局部 link 坐标不同，因此 `assets/rotate_bulb_custom/tactile/` 内保存转换后的压力 pad 原点、TacMap 点/法线与 marker 光线坐标，图像坐标及光学标定不变。转换记录和源 USD 校验值见同目录上一级的 `manifest.json`；`prepare_assets.py` 是离线再生成工具，不参与任务运行。触觉表面效果仍需在 Isaac Sim 中验收。

`tactile_policy_checkpoint` 是新任务独立的配置字段，仍指向 `runs/revo3_cross_modal_restoration_v1/best.pt`。预训练权重本体仍共享；要使用不同的编码器权重，修改新任务的该字段即可，编码器结构由所选 checkpoint 决定。

在已配置 Isaac Lab / RSL-RL 的环境中，从仓库根目录启动：

```bash
python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0 \
  --num_envs 32 \
  --headless
```

RSL-RL 的训练日志及策略 checkpoint 写入 `logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/`；RL-Games 的实验名为 `tianji_revo3_rotate_bulb_custom`。

对齐验证命令（检查启动、三次完整 reset、扰动后的部分 reset，不执行动态抓取）：

```bash
python assets/rotate_bulb_custom/validate_grasp_scene.py \
  --headless --num_envs 2 --full-observations
```

已通过 57 项离线回归检查；真实 Isaac Lab 的 2 个环境、完整观测模式通过 440 项姿态/关节检查，连杆位置最大误差约 1 微米。详细报告见 `assets/rotate_bulb_custom/grasp_reference/runtime_validation.json`。该报告验证初始状态与重置一致性，不表示灯泡抓取或动态轨迹已通过验证。

本次奖励对齐后的验证：标准与 Custom 各运行 60 项奖励行为测试，连同配置及资产检查共
130 项通过、1 项因 CPU 环境无 `pxr` 跳过。Custom 另在 GPU 的两个环境中重新通过
440 项抓姿/完整与部分 reset 检查，实际 RGB 输入为 `[2,5,3,240,320]`；5 个控制步的
完整观测和奖励均有限，成功终止项正常。第 5 步后的指尖 PhysX 材质仍保留启动随机值。
有效报告为仓库根目录下的
`logs/diagnostics/rotate_bulb_custom_alignment_20260912/runtime_custom_final.json`。
这些检查不代表训练收敛或策略抓取成功率；此次仅更新本地配置，没有部署或重启服务器训练。
