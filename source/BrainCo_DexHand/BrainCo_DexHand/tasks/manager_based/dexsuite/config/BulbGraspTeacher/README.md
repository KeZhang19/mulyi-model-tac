# 灯泡抓取教师（V2-test 移植，2026-10-04）

独立任务 `BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`，
回放任务 `BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-Play-v0`。
来源是 dextac-Depth 的 Stable v2-test 灯泡教师；不继承装配任务的369维观测或tanh动作。

## 2026-10-12：v12 完整帧观测

当前默认在原每帧380维后增加15维：奖励阶段one-hot(3)、root抬升误差(3)、
真实螺纹端点浅插误差(3)、灯座系螺纹轴方向(3)、旋入已开始/净圈数/剩余时间(3)。
两套位置误差在三个阶段均保留固定含义，不把同一新通道在阶段间改成不同材料点。
螺纹轴允许自由自转；三阶段累计转数来自原真实旋入状态。

**先拼同一帧395维，再按从旧到新的5帧展平为1975维**，Actor/Critic顺序一致。
新task_state组同样保留5帧，整体晋级reset清空全部旧阶段历史。
Isaac内部按term保存历史；`policy.py:stack_bulb_observation_frames`解包并按帧重排，
输入为`[frame(t−4),...,frame(t)]`，每帧为39+149+192+15。
可选LSTM输入一帧395维；隐藏层和动作不变。

`TaskStateNormalization`只对每帧原380维做运行归一化，新增15维使用固定物理尺度，
避免阶段/计圈常量首次变化时被接近零的运行方差放大。两误差单位米、轴为单位向量、
圈数除以目标3圈、剩余秒数除以固定8秒；各项完整定义见实验记录。
契约升级12：旧1900维检查点及旧统计排列不能直接加载，本次不自动迁移旧模型。
奖励、成功判据、原初态与下述v11全局课程保持；观测实现当时只做隔离测试。
19:01按用户新授权已停止c3g旧训练，以c3g已验v12变体从零启动双卡各4096；
[当前正式运行记录](../../../../../../../../experiments/bulb_v12_c3g_scratch_2x4096_20261012_1855/README.md)。
[完整观测布局、验证与兼容边界](../../../../../../../../experiments/bulb_task_observations_20261012/README.md)。

## 2026-10-12：v11 全局奖励课程

默认配置开启 `commands.object_pose.global_stage_curriculum`。所有环境使用相同训练阶段：

1. 阶段一中心到灯座上方15cm目标的距离<5cm；达到过即记一次回合成功。
2. 阶段二从同样桌面初态追踪真实螺纹端点的浅插目标；深度0–10mm、底端径向≤12mm、倾角<30°时记成功。
3. 阶段三仍从同样初态开始，先实际浅插再按原螺距计三圈；原对中、速度、稳定保持和6秒旋入时限不变。

`stage_two` / `stage_three` 在新模式代表全局奖励阶段，不代表回合完成；回合完成看
`stage_one_reached` / `stage_two_reached` 和 `metrics["stage*_success"]`。
阶段三只有 `screw_active` 为真才累计旋入和启用螺旋前导目标；浅插前仍追踪入口，不会凭切换课程获得虚假圈数。
阶段一/二回合始终用各自目标与奖励，不再因单环境成功立即换奖励或延长回合；统一8秒。
阶段三浅插前也有8秒上限，实际进入后按原有6秒计时。

晋级使用所有rank完整回合的整数成功数/总数，最近100次PPO更新滚动窗口，至少8192个完整回合；
成功率**≥80%**时，在本轮PPO更新结束后只晋一级，不回退。失败终止计入分母且不算成功；
阶段一/二用本回合达到过判据的记录，阶段三仍用原终态完整成功。
各rank即使没有回合结束也参与一次all-reduce，不能平均各rank的百分比。
晋级时所有环境从原初态重置，部分未结束回合不当作成功/失败，清空旧窗口和观测/动作历史；
动作配置、可训练网络参数、物理约束、reset事件与分布保持。ADR不会因这次主动reset升降；
ADR和重力仍只看原完整任务成功，不能把阶段一80%误接到重力升级。
阈值/窗口/最小回合量分别为 `stage_success_threshold` / `stage_success_window_updates` /
`stage_success_min_episodes`。

三项成本 `carry_relative_lin_vel_l2`、`carry_relative_ang_vel_l2`、`drop_after_lift` 权重全部0，
所有阶段均被RewardManager跳过。最终成功判据中的速度上限仍保留。
其余奖励保持各自基线；本地反比例版与c3g近目标方向增强版不是相同奖励，测试副本不混用两者。

专用 `teacher_runner.py` 保存 `bulb_global_stage_curriculum`，完整恢复会保留阶段和窗口；
显式 `model_only=True` 保留网络/输入归一化但将全局阶段重置到1。专用play也恢复保存的阶段。
契约升级到11；旧v10检查点不会被标准入口当作同任务自动续训/回放。
日志入口为 `Curriculum/global_reward_stage/{stage,evaluated_stage,success_rate,successes,episodes,promoted}`；
晋级那轮success_rate对应evaluated_stage，stage是下一轮阶段。无完整回合时不写虚构的0%成功率。

本次仅本地修改和c3g隔离短测试，不替换或重启现有训练。
[实现、测试证据与边界](../../../../../../../../experiments/bulb_global_reward_curriculum_20261012/README.md)。

## 2026-10-11：恢复位置引导／二阶段反比例方向奖励（代码定义）

按用户要求，两项 `carry_relative_lin_vel_l2` / `carry_relative_ang_vel_l2` 权重暂设0，保留配置便于恢复。
二阶段success保留原完整 `10pF` 位置部分，再加方向 `5qg`；原tracking 5保持、thread_down 5改用同一q。
位置相关最高15/秒、直接方向最高10/秒，二阶段这三项合计最高25/秒（原混合版20/秒）。
`p=(1−tanh(d/.1))²`、`F=.2+.8(1−θ/π)²` 与原位置引导保持；接触门控g保持。
方向q采用有界反比例，θ为倾角：`q=(1−θ/π)/(1+θ/(π/4))`，即以45°为尺度。
θ=55/30/15/0°时，两项直接方向合计3.125/5/6.875/10每秒；越接近朝下，斜率越大，零误差无奇点。
新参数 `orientation_scale_deg=45` 写入success/thread_down；0保留线性曲线。
`orientation_mix=.5` 现在表示额外方向份额，不再削减位置份额；同名参数不代表与旧实现奖励相同，旧协议回放须用冻结源码。
反比例仅二阶段，第三阶段thread_down仍线性、success仍原核；一阶段水平灯泡姿态惩罚−.1→−.3/秒，参数stage_one_scale=3。
gamma=.995、原5帧MLP、资产/物理/三阶段成功与时间预算保持。
CPU66/66通过，覆盖曲线斜率与有限性、位置牵引保持、阶段/接触隔离和全部原几何/晋级/终态约束；没有本次Kit/GPU学习效果验证。
源码更新须随新进程启动生效；c3g23979仍按下方21:10启动的原线性混合版本训练。最新同步状态见本次记录。
[本次备份、差异和检查](../../../../../../../../experiments/bulb_reciprocal_orientation_20261011/README.md)。

## 2026-10-11：c3g当前运行21:10部署的方向混合／gamma0.995

原目录内旧8058作业已停止，最新完整保存model2700用于恢复；日志2738的未保存38轮不恢复，旧模型及源码备份保留。
21:10:07启动23979，双卡各8192／总16384、原5帧MLP无LSTM、ADR初值0、原容量／时间／成功／物理保持。
run `logs/rsl_rl/bulb_grasp_teacher_v10_posemix_gamma0995_mlp/2026-10-11_2100_resume_model2700_2x8192`；追加27300，目标总上限30000。
两rank生效mix=.5、gamma=.995、lambda=.95；29项原生恢复／同步／日志检查通过，2700–2714共15次实际PPO。
优化器、归一化和重力状态恢复并更新，完整env仅新增mix，agent仅改gamma／剩余预算。没有独立任务成功评测。
[命令、备份及原生验收](../../../../../../../../experiments/bulb_v10_posemix_gamma_resume_20261011_2100/README.md)。当前训练已启动，勿重复；下方本地说明保留变更定义和部署前验证历史。

## 2026-10-11：长程折扣 gamma=0.995

`agent_cfg.py`重建源快照后显式覆盖 `algorithm.gamma=.995`，其余PPO参数保持，lambda=.95、rollout32步、原5帧MLP保持。
冻结 `agent_snapshot.json`仍为历史源值.99；训练保存的实际 `params/agent.yaml` 才是run折扣口径。
按30Hz，延迟6/8秒的单项奖励回传系数由16.38%／8.96%变为40.57%／30.03%；折扣在控制步计算。
更高gamma同时重视持续的稠密奖励，不保证旋入改善，也不补足缺失的奖励信号。
该次修改时一阶段只含−.2权重的灯泡倾角惩罚（水平−.1/秒），thread_down正奖励及方向混合未开启，一→二只看位置；当时未改一阶段引导。
CPU实际配置与已安装RSL PPO实例确认gamma=.995，其他参数／源快照保持；未做PPO更新或Kit／学习效果验证。
此节记录部署前的本地修改与CPU验证；随后8058已停止并部署续训，当前作业见上方。新配置不会热更新已有进程。
[本次证据](../../../../../../../../experiments/bulb_gamma_0995_20261011/README.md)。

## 历史：2026-10-11 21:10部署版本的位置／方向奖励各占主权重

该次在v10入口契约上设置 `rewards.success.params.orientation_mix=0.5`。
二阶段原10权重的倾角耦合位置项改为：5×原位置核×倾角软系数 + 5×朝下对齐度×夹持接触。
加上原 `position_tracking` 5及 `thread_down` 5，位置／方向各最高10/秒，合计最高20/秒；
不增加二阶段基础上限。朝下对齐度为 `clamp(1−θ/π,0,1)`，夹持门控沿用拇指与至少另一指均>1 N。
方向份额不再随位置误差衰减，因此允许先调整姿态再对中。保留20%的位置软系数下限，允许绕螺纹轴自转。
手心朝下项仍关闭；只奖励灯泡方向。第一／第三阶段、奖励权重、搬运稳定惩罚、晋级及最终判据、资产／动作／5帧观测保持。

`orientation_mix=0.0`恢复本次修改前的奖励；目标／阶段契约仍为10，但不同mix的训练属于不同奖励协议，不能公平比较总回报。
新run保存的env参数须包含实际mix；回放旧协议使用修改前源码或显式0。当前已按用户授权部署续训，见上方记录。
CPU62/62通过，含姿态与端点偏移取舍、阶段隔离、接触、轴向自由和全部原课程／终态回归；未做原生仿真或学习效果评测。
[分析、备份及验证](../../../../../../../../experiments/bulb_stage2_orientation_main_20261011/README.md)。

## 2026-10-11：v10 宽松底端入口基线

二→三改为真实螺纹端点横向偏差 **≤12 mm**、相对孔口深度 **0–10 mm（含端点）**、倾角 **<30°**，单控制步满足即晋级，无二阶段速度/保持门槛。
目标仍为孔轴线下5.5 mm；一阶段、直接底端坐标、奖励权重、一次晋级+10、资产/物理/5帧MLP保持。
新增 `shallow_insertion_radial_tolerance_m=.012`，仅用于二阶段；原 `.0005` 径向阈值仅用于最终装配验收，避免同时放宽终态。

此范围是进入第三阶段的课程区域，不证明真实螺纹啮合。进入后旋入奖励项已启用，但正奖励仍需端点/口沿轴线最大偏差≤2 mm、倾角≤10°、夹持接触与符合螺距的实际进展；偏轴空转不计奖励。
三阶段原位置/朝下奖励可用于继续调整；6 s预算从晋级开始，旋入深度/相位基准也取晋级状态，最终目标深度随入口深度改变（标称尺度18–28 mm）。这可能使较偏入口调整时间不足，学习改善需后续评测。
完整三圈、最终≤0.5 mm/≤5°、速度/保持及物理失败判据保持；这些三阶段稳定门槛未用于二→三。

MDP版本10，显式 `--migrate-v9-entry-window` 允许 v9→v10 全状态续训，默认回放/加载严格匹配版本，不接受v8跨级迁移。
本地CPU60/60通过，覆盖0/10/12 mm边界、范围外拒绝、宽松晋级后偏轴旋转无奖励、恢复对中后有效旋入奖励及原最终验收。
[本次证据与部署状态](../../../../../../../../experiments/bulb_v10_entry_region_20261011/README.md)。

## 历史：2026-10-11 v9 直接底端位置命令

二、三阶段的7D命令前三维直接保存螺纹底端目标（机器人基座系），不再按当前姿态反推中心目标。
`insertion.py:insertion_tip_target`直接给孔轴线上深度目标；`bulb_tip_position_w`计算真实端点世界位置。
`curriculum.py:tracked_position_w`在阶段一返回刚体根位置，在阶段二/三返回真实端点。
两个位置奖励通过局部适配函数直接比较此参考点与目标，原核、权重、接触门控和阶段二朝向系数保持。
阶段一仍为中心到孔口上方15 cm、距离<5 cm晋级；阶段二仍为深度5–6 mm、端点径向≤0.5 mm、倾角<30°即时晋级。
三阶段继续原测量相位/螺距/小领先目标，原计圈、物理约束、速度、保持及最终验收不变；位置命令/位置奖励改用底端。

无需重建资产，不改变碰撞、质量或刚体原点。底端沿用螺纹最前端截面轴心，不是倾斜后的网格世界最低顶点。
`insertion_target`仅保留给几何探针/CPU构造状态；生产命令不调用该中心换算。
观测布局/5帧1900维、动作28维保持，但目标数值含义已变，旧v8检查点不能凭维度相同自动恢复或当作v9评测；用户明确授权的续训须用显式迁移参数。
train/play在载入检查点前读取相邻`params/env.yaml`版本，缺失或不匹配即报错；旧模型继续用冻结v8检出回放。

目标标记画直接位置命令、实际标记画本阶段参考点；二阶段目标不会随灯泡姿态平移。
标记按真实阶段条件着色，不再按原5 cm近距变绿；球半径目标5 mm/实际4 mm仅用于显示，与成功门槛无关。
`progress.py`提供阶段完成状态及浅插/最终旋入逐项条件；回放HUD显示XYZ，trace保存同一结构。
本地CPU58/58通过；[示意图/证据](../../../../../../../../experiments/bulb_direct_tip_target_20261011/README.md)。
用户随后授权同目录同步及早期v8检查点续训，见[运行证据](../../../../../../../../experiments/bulb_v9_resume_20261011/README.md)。
训练入口可显式`--migrate-v8-tip-command`迁移，回放继续严格检查；未做新版策略录像/独立成功率评测。

## 历史：2026-10-07 v8 第二阶段改为螺纹端点浅插

按本次用户要求，第二阶段的位置目标按灯泡**当前姿态**换算中心位置，使原
`position_tracking`和`success`位置核的距离等于实际螺纹端点到孔内浅插目标的距离。
端点沿用`insertion_geometry.json:bulb_tip_offset_m`，即螺纹最前端截面的轴心，
不是倾斜后整颗灯泡网格的世界最低顶点；轴向目标仍为口沿下5.5 mm。
目标朝向沿用当前轴向相位，只引导朝下，不再要求源CAD的预设旋入相位。

阶段二→三现在只要求：端点深度5–6 mm、端点径向偏差≤0.5 mm、螺纹轴倾角**严格<30°**。
满足的第一个控制步即晋级，不检查线速度/角速度，不要求连续0.2 s。
口沿处的整根轴线对中不再作为第二阶段门槛；否则其0.5 mm容差会隐含更紧的倾角约束。
实际螺纹碰撞保持，穿过侧壁/倾斜插入是否物理可达仍由仿真决定。
这里只将浅插定义为无需旋转的晋级条件，没有新增“零接触/尚未啮合”的物理检测或保证。

两项位置奖励的核、权重、接触门控、朝向软系数及晋级一次+10均保持；第一阶段保持。
第三阶段从实际浅插深度/相位重新计圈，原三圈、螺距、≤5°、低速、0.2 s保持和一次+20均保持。
`shallow_insertion_tilt_tolerance_deg=30`只供二阶段；旧`insertion_tilt_tolerance_deg`、
`insertion_speed_limit_m_s`、`insertion_angular_speed_limit_rad_s`、`insertion_hold_s`现仅供三阶段最终验收。
观测宽度、动作、物理、资产、随机化、8+6 s时间预算不变；目标含义和晋级协议已变化，MDP版本升为8。
旧checkpoint可作为继续学习的候选，但用本代码评测不再属于原v7协议。

`insertion_tip_radial_error_m`新增记录二阶段端点偏差；旧`insertion_radial_error_m`
仍为端点/口沿轴线最大偏差，供三阶段使用。旧`insertion_hold_s`指标恒0，回放HUD按v8显示新条件。
CPU课程检查52/52通过，包含真实数学/命令生命周期和USD资产回归；场景张量为替身，未运行物理轨迹。
未同步/重启远端，10月7日已有v7 LSTM部署仍是原版本。下方v5/v6/v7描述保留为历史。
[本次差异、检查和边界](../../../../../../../../experiments/bulb_stage2_tip_entry_20261007/README.md)。

## 2026-10-07：可选单帧 LSTM

专用训练和回放入口新增 `--lstm`，双卡启动器原样透传。省略时仍使用原来的5帧1900维MLP。
启用时各观测组历史长度为1，教师输入为 policy/proprio/perception=39/149/192，共380维。
拼接、原有归一化后，Actor/Critic各用独立的单层256维LSTM，再接原512→256→128 MLP；
分别输出28维动作均值和1维价值。保留原探索标准差0.05–0.5、动作、奖励、物理和完整三阶段判据。
LSTM状态跨控制步和rollout保留，回合终止/超时时只清空对应环境；阶段晋级不清空。
PPO按原32步rollout执行循环序列更新，局部适配末步价值bootstrap，避免重复推进critic记忆。
`policy.py`复用RSL的ActorCriticRecurrent；共享PPO/runtime文件未修改。
启用时入口选择仓库的 `runtime/bulb_grasp_teacher`，部署须携带该runtime及任务局部policy.py。

```bash
# 从零训练示例；会启动Isaac/GPU并写入新run，本次未执行。
bash scripts/rsl_rl/train_bulb_grasp_teacher.sh --headless --lstm --num_envs 8192 --max_iterations 30000
# 双卡入口在原有参数后追加同一开关：
bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh --lstm
# 回放需明确指定使用--lstm训练的checkpoint：
bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --lstm --checkpoint /absolute/path/to/lstm/model_N.pt
```

旧1900维MLP权重不能直接加载到LSTM；首次启用从零训练，后续续训/回放同样传 `--lstm`。
`params/env.yaml`、`params/agent.yaml`及`launch_contract.json`记录实际历史长度与策略类型。
CPU检查7/7通过：包含真实runner策略构造、两轮循环PPO共40次优化、异步回合reset、
bootstrap记忆不变、噪声边界及严格checkpoint往返/旧MLP拒绝。
使用合成观测/奖励，无Isaac场景；未验证原生物理、GPU/DDP或学习效果，未同步/重启远端。
[检查日志](../../../../../../../../experiments/bulb_teacher_lstm_20261007/cpu_checks.log)。

## 2026-10-07：v7 二阶段浅插成功一次+10（当前本地默认，未切换远端）

原浅插判据连续保持达标、首次进入第三阶段时，`stage2_success_bonus`一次+10；每回合不重复领取，reset重新允许。
按dt抵消后实际支付10分，同帧物理失败不给；第三阶段最终一次+20及其余奖励/判据/物理保持。
48项CPU检查通过，未原生运行v7。当前指定DSW仍运行v6，代码修改不会热更新活动进程。
[实现与检查](../../../../../../../../experiments/bulb_stage2_bonus_20261007/README.md)。

## 2026-10-07：v6 阶段二位置奖励耦合朝向（当前远端运行）

本地只修改一项奖励：阶段二 `success` 由纯位置核改为
`10 × (1 − tanh(d / 0.1))² × [0.2 + 0.8 × (1 − θ / π)²]`。
`d` 为灯泡中心到当前目标的距离（m），`θ` 为螺纹朝下倾角（rad），仍允许绕轴自转。
这里只在 `stage_two & ~stage_three` 生效；阶段一、三沿用原位置奖励，原函数/snapshot不改。
保留20%位置收益，不增加接触门控；`rewards.success.params.orientation_floor=1.0`
可恢复v5奖励数值（代码/版本仍为v6，精确旧版回放用原冻结源码）。

其余19项奖励、两阶段8 s/第三阶段6 s、动作/观测/物理/随机化、浅插和三圈验收均不改。
`thread_down`仍为 `5 × (1 − θ/π) × 双侧指尖>1 N`，阶段二和三启用；未同时加权或放宽接触。
避免直接设置原 `success.rot_std`：原函数使用完整四元数误差，会额外约束螺纹轴向相位。

依据为model8800旧录像：728个阶段二帧中642帧中心误差≤1 cm，但倾角最小39.11°，无≤30°帧；
最接近中心的帧误差2.72 mm、倾角77.66°。这是一条回放，不是独立成功率。
双侧接触不变、仅比较四项位置/朝向奖励的静态例子中，
从中心到位/75°转为中心偏2 cm/0°，原收益17.84→15.94，新收益12.56→15.94（分/s）。
这解释了最小调整的动机，不是动力学轨迹或训练效果预测。

CPU44项通过，涵盖阶段隔离、局部reset、单调性、轴向自转、接触独立、关闭门控和原完整课程回归。
初次实施时仅完成CPU检查；随后已于10月7日部署至`dsw-wpmt7kzoh4b53jpmov`独立v6副本，
双卡各8192从v5/model9500续训。原生观察12次PPO更新、双rank同步、容量日志及配置核验通过。
未做独立成功率评测，不能据此声称重定向改善。见
[启动记录](../../../../../../../../experiments/bulb_v6_dsw_resume_20261007/README.md)。
[提取设置/数值分析/差异与检查](../../../../../../../../experiments/bulb_stage2_reward_20261007/)。
后续优先看阶段二专属倾角分布和接触占比；若仍停滞，再分别试朝向奖励软接触或增加其权重。

## 2026-10-06：三阶段课程 v5（远端历史运行；本地v6继承下列规则）

用户要求新增第三阶段：灯泡旋入3圈，第三阶段单独6 s，最终大成功奖励一次+20。

| 项目 | 阶段一 | 阶段二 | 阶段三 |
| --- | --- | --- | --- |
| 目标 | 随机灯座口世界竖直上方15 cm，跟踪灯泡刚体中心 | 螺纹尖端深入口沿5.5 mm，由几何和倍率换算中心 | 从实际浅插位姿继续顺时针旋入3圈，目标沿测得螺旋进度超前1/8圈 |
| 晋级/成功 | 中心误差严格<5 cm | 插深5–6 mm；尖端及口沿轴线误差≤0.5 mm；倾角≤5°；线速≤0.02 m/s、角速≤0.5 rad/s连续0.2 s | 有效净旋入≥3圈；实际深度距浅插起点+3个螺距的误差≤1 mm×倍率；对中、倾角、低速与0.2 s保持要求同阶段二 |
| 时间 | 与阶段二共8 s | 与阶段一共8 s | 从晋级时刻单独6 s |
| 相对速度惩罚 | 原两项保持 | 原两项保持 | carry_relative_lin_vel_l2和carry_relative_ang_vel_l2均返回0 |

顺时针指从灯座开口上方朝下看；对应灯泡相对灯座局部+X的负向旋转。
源螺距6 mm，每环境实际螺距为6 mm×倍率。名义尺寸从浅插5.5 mm再旋入3圈，
最终尖端深入约23.5 mm，中心约高于口沿30.26 mm；实际基线取阶段二晋级时的深度与轴向角。
阶段一、二沿用v4的随机灯座/真实螺纹SDF/配对缩放/初始净空；没有增加引导关节、外力或瞬移。

### 第三阶段奖励

| 项 | 配置权重 | 实际支付方式 |
| --- | ---: | --- |
| screw_in_force | +9 | 每新增1圈有效物理进度×受力质量；三圈至多27分 |
| screw_in_progress | +3 | 每首次跨越1/4圈支付0.75分×受力质量；三圈至多9分 |
| screw_backtrack | −14 | 新增历史最大回退圈数×14，0.01圈死区；恢复不退还，不重复处罚同一回退幅度 |
| final_screw_success | +20 | 达到三圈、深度/姿态/稳定验收后每回合仅一次，物理失败同帧不给 |

受力质量为灯泡过滤后的最强指尖力/0.5 N，裁剪到[0,1]；支持0.2 s线性衰减的换指记忆。
参考RotateBulbCustom的RotateBulbGuidedRotation与RotateBulbRandomized回退成本思路，
只支付历史新高；无接触进度也会被消费，不能停下重新接触后补领。奖励按每秒1圈上限限制单步信用。
新进度/里程碑/回退/最终奖励均先除以step_dt，再由RewardManager乘回，20分不是20×dt。
三圈两项旋入奖励最高36分；旧四项接触奖励总上限4.7分/s，6秒约28.2分，旋入约高28%。
最终+20额外支付。原接触项及其他奖励保持；没有引入第三阶段接触预算包装器或驻留惩罚。

连续位置`success`奖励仍是原tanh位置核，和二值成功指标/新增最终奖励不同。
两个阶段沿用的灯泡朝向惩罚−0.2、阶段二及之后的thread_down+5保留。
最终奖励只发一次，但终态成功不锁存：后续退出、过深、偏斜或掉落仍会使终态失败。
仍不因成功提前终止；ADR/重力课程读取完整三阶段终态成功。

### 计圈、防误报与时间顺序

`screwing.py`从实际刚体相对四元数解包轴向角，跨±π/四元数换号保持连续；
有效进度取净旋转圈数与实际轴向推进/螺距的较小值，并要求螺旋误差≤1.5 mm×倍率。
计圈期间径向≤2 mm、倾角≤10°，最终验收收紧回≤0.5 mm/5°。
离孔/失去对中或单步超过3圈/s的跳变会清零当前角度累计，历史已付新高不清零。
超出螺距对应深度的正向空转被丢弃，不能先空转三圈再直线压入补领；反向运动不退款。
第三阶段目标只领先实际有效进度1/8圈（名义0.75 mm），上限为三圈终点，避免直接命令压入18 mm。
角度完成比较仅含1e−5圈的浮点容差。

阶段1/2共8 s；阶段3在进入时计数归零，180个后续控制步后超时。最长回合14 s。
`curriculum_update`位于termination列表第一位，先读取本物理步并晋级，再判time_out；
阶段二最后一帧达标仍能获得第三阶段6 s。其他失败终止保持。
命令重采样/普通metrics更新不累计保持或奖励，局部reset清理对应环境的圈数、计时和一次性奖励状态。

### 资产、接口与验证

灯座每回合继承原目标基座系XY范围X∈[−0.85,−0.55]m、Y∈[−0.25,0.25]m，底面贴桌、开口向上。
灯泡保留0.8–1.2倍尺寸随机化，灯座在prestartup匹配倍率，回合内不移动。
`bulb_insert.usd`保留原外螺纹SDF512、壳SDF256及灯座三角碰撞；质量、点云、源snapshot不变。
28维动作、policy/proprio/perception=195/745/960、5帧历史、30Hz/120Hz保持。
宽度相同不代表MDP与旧checkpoint语义相同，旧模型仅为初始化候选。

metrics新增screw_turns、screw_angular_turns、screw_best_turns、screw_depth_target_m、
screw_helix_error_m、screw_hold_s、stage3_elapsed_s；stage1_success/2_success表示已晋级，
success/stage3_success/terminal_success采用完整终态。录像HUD/trace显示第三阶段圈数和剩余时间。

CPU40项通过，覆盖三圈、力门控、重复奖励、反向、空转/压入、阶段独立超时及+20一次性发放。
原生完整任务2环境×245步通过（含+20项注册）；三倍率×13个四分之一圈位姿，
原生零重力释放39/39未明显挤出。该探针使用制造位姿，无机器人策略，不是学习成功率。
没有训练v5或验证任意初始螺纹相位下的自主旋入；8192环境SDF容量未评测。
[完整证据](../../../../../../../../experiments/bulb_grasp_screw_stage3_20261005/README.md)。
远端没有同步/重启。旧v1成功率与视频不能作为三阶段成功率；v4浅插实现/实验历史保留在
[前一版证据](../../../../../../../../experiments/bulb_grasp_insertion_20261005/README.md)。

## 直接验证

从本仓库根目录运行，默认加载已复制的原始 `model_22200.pt`：

```bash
bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --num_envs 16 --real-time
```

关闭窗口结束。少量环境、有限步数的无界面检查：

```bash
bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --headless --num_envs 4 --steps 180   --report logs/diagnostics/bulb_grasp_teacher_manual/report.json
```

`--checkpoint /absolute/path/model_N.pt` 可指定同一教师契约的模型。
`BULB_TEACHER_PYTHON`、`ISAACLAB_ROOT` 可替换脚本中的本机环境默认值。
当前环境：brainco Python 3.11、Isaac Sim 5.1、Isaac Lab d94504bc、RSL-RL 3.1.1 fork。

## 资产与接口

- 机器人使用本仓库 `assets/rotate_bulb_custom/robot/usd/rizon4_training.usda`。
- 灯泡来自本仓库 `assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd`。
  `build_asset.py` 生成 `assets/bulb_grasp_teacher/`：只取灯泡三个组件，居中为自由刚体，
  原版采用教师的三个convex hull、0.2kg质量和64点点云；当前v4再由build_insertion_asset.py生成SDF派生资产。原资产不改写。
- 灯泡源文件SHA256和生成点云SHA256与源教师一致；机器人入口USD也一致。
  这不表示已经验证整个机器人引用树的动力学完全等价。
- 观测按policy/proprio/perception拼接为195/745/960维，总1900；5帧历史。
- 动作为28维相对关节增量：先乘0.1，再裁剪机械臂±0.25rad、手±0.6rad；
  每个物理子步按当前q重新计算目标。策略原始动作不再额外裁剪到±1。
- 控制30Hz、物理120Hz。基础MDP来自教师快照；当前课程差异见上节。
- 回放恢复checkpoint中的重力课程，当前22200对应−9.8m/s²。
  原checkpoint不保存ADR全部逐环境状态，回放保留源运行配置的初始difficulty=0.7317；
  不宣称和服务器同一时刻具有完全相同的随机分布或成功率。

## 训练入口

必须使用专用入口，包含源教师的重力触发、iteration事件和checkpoint恢复逻辑。
通用 `train.py` / `play.py` 不支持此任务的 `BulbGraspTeacherRunner`，会明确拒绝，避免静默遗漏课程。

```bash
bash scripts/rsl_rl/train_bulb_grasp_teacher.sh --headless --num_envs 8192   --max_iterations 1000 --checkpoint checkpoints/bulb_grasp_teacher_20261004/model_22200.pt
```

这是可选手动续训命令，本次未启动长期训练。`--max_iterations` 为额外迭代数；省略checkpoint从头训练。
单卡入口已验证；双卡入口为 `scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh`，每卡4096环境。
无checkpoint时ADR从0开始，重力触发进度归零；沿用教师快照的初始重力−1m/s²，满足成功阈值后逐步到−9.8。
双卡脚本使用任务专属 `runtime/bulb_grasp_teacher/rsl_rl`，种子42/43，每个rank独立Kit目录。
第一次优化后检查梯度all-reduce执行且所有策略参数SHA256在两rank一致；证据写入各rank的distributed_verified.json。
配置使用源agent快照，避免本机新版Isaac配置额外字段改变算法设置。
本机PPO无share_cnn_encoders参数时，仅允许移除其False值（该任务为MLP）。

## 实现与验证证据

`teacher_snapshot.json` / `agent_snapshot.json` 保存实际运行配置，`env_cfg.py`重建Isaac配置类型。
`mdp/`和`flexiv_rewards.py`是源代码的任务局部依赖闭包，移植时39个函数/类的AST与源一致；
当前新增的`curriculum.py`通过子类与配置覆盖扩展它们，原移植依赖文件保持；
`teacher_runner.py`是原runner的完整副本，保留其许可证。来源哈希见`port_manifest.json`。
不依赖源dextac-Depth仓库的Python导入或模型资产路径。

本次证据位于 `logs/diagnostics/bulb_grasp_teacher_port_20261004/`：

- `smoke2.json`：4环境×180步，strict checkpoint加载、1900/28维和有限状态通过，完成7个回合。
- `semantic_audit.json`：仅资产/点云路径、log_dir及未使用的recorder枚举表示发生配置差异。
- `training_audit.json`、`train_smoke4/`：4环境×1次PPO更新，优化器状态存在、loss有限、
  重力trigger=11502、gravity=−9.8，原始checkpoint哈希不变。
- 前三次训练接线失败及第一次回放失败保留，不用作验收。小批量PPO试验模型只用于接线检查，
  默认回放始终使用`checkpoints/bulb_grasp_teacher_20261004/model_22200.pt`。

上述检查不构成抓取成功率评测，也不构成装配成功验证。
