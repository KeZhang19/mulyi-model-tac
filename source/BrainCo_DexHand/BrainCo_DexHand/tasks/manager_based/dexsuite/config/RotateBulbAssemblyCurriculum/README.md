# 桌面横躺灯泡装配课程 v2

本任务是抓取、抬起、搬运、对准、旋入；不是 Randomized 灯泡旋出任务。
入口仍为 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`，课程设置与检查点物理契约版本为 2。
原 v1 检查点需在原冻结源码中使用，不能在 v2 直接恢复。独立 Custom 装配和 Randomized 旋出不改。

## 旧版的问题与本次改动

- v1 先固定竖直灯泡训练抓持，再训练自由竖直抓持。固定夹具可承担物体重量，迁移不能保证会抓起自由物体。
  v2 从自由横躺灯泡开始，重力、桌面接触、手部和机械臂自碰撞始终存在。
- v1 接近信用只奖历史最好值，退远不收回；动作代价也不能表达第一关节走了多长路径。
  v2 用实际壳体位置的拇指／最近对指距离与掌距，提供连续接近反馈及有正负号的距离进度，
  并对第一关节实际速度积分和超过初态 60° 的偏离给予软代价。没有修改关节限位或指定单侧转向。
- 原 `tactile_policy_enabled=false` 只关闭图像编码器，285-taxel 压力和 15 维接触观测仍存在。
  v2 无触觉模式移除压力、接触观测及 WarpSDF/Tacmap/HydroShear 传感器，输入为 369 维。
  抓持、掉落、桌面净空等物理判据仍使用七个接触传感器。真值位姿／点云观测仍是仿真状态，非纯视觉部署方案。
- v1 240 Hz 物理、30 Hz 控制，每动作 8 步；v2 候选为 120 Hz、30 Hz、4 步，
  `env.bulb_curriculum.physics_hz=240` 可显式选择 8 步。求解器迭代、螺纹和自碰撞保持。
  120 Hz 的桌面稳定性验证不等于完整旋入接触质量验证，后者仍须用成功策略复核。
- reset 的阶段采样、回合结束读取改为批量处理，移除每环境每字段 GPU 标量同步；
  每个完整回合的成功／失败、阶段事件和第一关节轨迹指标均保留。
  训练不再重复输出奖励诊断 JSONL；TensorBoard 和课程 JSONL 保留，冻结评测保留详细奖励记录。

## 课程

| 阶段 | 起点／难度 | 通过条件 | 时限 |
| --- | --- | --- | --- |
| 0 抓持 | 自由横躺灯泡，固定位置，开手近距接近姿态 | 拇指＋至少一指 >1 N，低速且相对滑移 ≤5 mm／5°，连续 >1 s | 5 s |
| 1 抬起 | 同样横躺；近、中、远臂姿，再扩大 XY 到 ±5 cm、yaw 到 ±90°，共九档 | 碰撞表面离桌和相对初态均 ≥3 cm、桌力 <0.1 N、双指抓持、低速，保持 0.3 s | 10 s |
| 2 搬运对准 | 来自真实 Stage1 成功状态库；混合前阶段回合 | 灯座上方 10 cm±1 cm，径向 <6 mm、轴误差 <0.2 rad，保持 0.5 s | 8 s |
| 3 下降旋入 | 来自真实 Stage2 成功状态库；混合前阶段回合 | 原物理螺纹啮合、旋转与推进关系、18 mm 插入和 1 s 稳定成功 | 15 s |
| 4 全流程 | 80% 从横躺完整随机初态出发，20% 前技能复习；冻结评测全部从桌面开始 | 原完整装配成功判据 | 20 s |

阶段 2／3 使用联合物理状态库训练子技能，不能作为“从桌面成功”的证据。
reset 不运行隐藏物理步；恢复后的前三个正常策略控制步检查抓持与相对位姿，失败计入评测。
跨阶段仍需两组训练窗口、3 个独立评测 seed；门槛保留为 90%／80%／85%／70%。
Stage1 需逐档通过，禁止仅凭固定灯泡旧模型跳过自由抓取。

双卡用 `train_bulb_curriculum.py --num-processes 2 --num_envs 4096`：每卡4096环境，
NCCL共同训练一个策略。`distributed.py`只在PPO更新边界合并回合结果和真实成功状态库，
两卡共用晋级窗口；不同步每一次reset，避免不同回合长度导致死锁。各rank日志分开，
主卡保存全局课程、各rank计数／随机状态和共享策略检查点。恢复要求相同worker数量及seed映射；
单卡冻结评测可读取双卡检查点。只保存课程采样器状态，不声称逐物理步精确续接。

容量按每卡环境数计算。横躺1024环境曾使512MiB碰撞栈溢出；1GiB复测无该错误。
4096环境候选为3.75GiB碰撞栈、三种配对容量各2²⁴、接触点2²⁴、接触片2²⁰。
碰撞栈／接触流是主机锁页内存，配对表含设备分配；不能全算作显存。
栈字段是uint32，4GiB会越界；3.75GiB是容量候选，不能保证任意策略／更大场景不溢出。
原生探针会检查Kit日志；丢接触、缓冲溢出或CUDA错误一律否决该次结果。

`floor_pregrasp_v2.json` 保存横躺碰撞顶点、命名关节与源资产 SHA256。
初始净空 2 mm；42 个离线离散路径样本通过碰撞筛查，joint1 为 −8.35° 至 +10.32°。
这是有限几何证据；不证明随机分布全域可达或学得策略不会绕转。

## 奖励与验证口径

接近奖励（每秒）为 `1-tanh(d/0.1)`，另给 `4*(d_previous-d)` 的单步信用；
抬起后停止接近项。这个 shaping 不声称具有严格的最优策略不变性。
机械臂路程项每秒为 `-0.08*(abs(qdot1)+0.1*mean(abs(qdot2..7)))`；
第一关节偏离项每秒为 `-0.5*(relu(abs(q1-q1_default)-pi/3)/(pi/3))^2`。
距离和路径来自实际状态；坐标不按 2π 折返。其余抓持／抬起／对准／旋入信用来自原物理奖励账本。
实际奖励看 `Episode_Reward/*`；旧账本 reach 的诊断值不代表新接近项。

2026-10-04新增 **仅Stage0** 的 `reach_best_progress`，移植本机
`dextac-Depth-main/dextac-Depth/source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/rewards.py:object_ee_distance_delta`。
它奖励历史最小距离的新纪录，使用五个指尖到灯泡壳体中心（灯泡局部+X 4cm）的**最大距离**D，
与原接近项的加权距离d不同。最终单步信用为 `100 * max(D_best - D, 0)`；
然后更新D_best。不需要先接触，回退/重走旧距离不重复给分；每个reset后的首个有效奖励样本只初始化。
掉落/失握/越界/非有限状态/恢复失败的终止步以及已标记抬起后不给该项，保持原成功和reset规则。

原参考项为`weight=200, std=0.4`，返回`delta/std`再被RewardManager乘dt；
这里返回`delta/(std*dt)`，配置`weight=40, std=0.4`，即每新增靠近1cm给+1分，
改善3–5cm给+3～5分，不随控制频率改变。原项若在30Hz下，每cm约+0.1667分，新项为其6倍。
Stage0初态最远指尖约7.39cm；该项总分受首次有效距离约束，并非每步持续发大额奖励。
保留原持续接近/有符号进度、动作幅度−0.02、动作变化−0.01、臂路程−0.08、偏离−0.5项。
作为量级参考，第一关节单向走30°产生约−0.0419路程分；不是实际下降所需运动或所有负奖励的估计。
严格抓持完成仍为+8分；当前系数是有单位依据的初始候选，尚无训练收益证据。

新项在 `configure_assembly_rewards` 清理旧奖励之后注册；Stage1–4、独立Custom装配/旋出不启用。
修改了课程源码哈希，旧课程checkpoint的严格契约检查会拒绝直接恢复；旧录像需匹配旧冻结源码。
实际日志看`Episode_Reward/reach_best_progress`（按原RewardManager口径归一化，不等于逐回合原始积分）。

同日按用户要求，Stage0另启用dextac的 `any_finger_contact`：任意一指对灯泡的过滤接触力
严格>1N即奖励0.7分/秒（30Hz每步约0.02333分，5秒最多3.5分）。多指同时接触也只计一次，
不要求拇指参与、低速、位姿稳定或保持时间；与源实现一样，不另设阶段内失败/抬起门控。
复用已有`assembly_any_finger_contact`，只读取灯泡过滤力，不用碰桌/灯座的总接触力；
原父配置虽有此项，但此前被课程账本替换清掉，现在在替换之后仅为Stage0重新注册。
原账本双侧接触质量可给>0.1N的弱双侧接触部分分数，单指接触不触发该项；严格双侧保持/稳定成功仍要求>1N。
当前保留原双侧稳定奖励、历史接近+1分/cm及全部动作惩罚；没有另加参考中的good_contact +2分/秒。
这使单指探索先有反馈，不能据奖励已启用认定学会稳定抓持。检查入口现为11项CPU检查，无新训练结论。

验证入口：`scripts/diagnostics/check_bulb_floor_contract.py`（CPU）、
`scripts/diagnostics/probe_bulb_floor.py`（原生、有限步数）、
`scripts/rsl_rl/train_bulb_curriculum.py`（训练＋独立晋级评测）。
阶段 4 直接用 `evaluate_bulb_curriculum.py` 做端到端评测。
旧 `export_bulb_curriculum.py` 的 669 维 Custom 导出不适用于 v2 的 369 维横躺任务。

## 公开实现参考

- [Isaac Lab Lift 配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py)：
  参考接近→抬起→目标跟踪分解和较弱的早期动作代价；没有直接套用其平行夹爪参数。
- [Isaac Lab Lift 奖励](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py)：
  参考 tanh 距离反馈及抬起后再做目标跟踪的门控。
- [UniDexGrasp](https://arxiv.org/abs/2303.00938)：参考目标抓姿与课程的分工，未声称复现该方法或使用其训练结果。
- [NVIDIA 性能手册](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html)：
  按传感器、物理步数和实际测量分开定位；时间步增大需验证接触精度，不能只比较吞吐。

本次运行证据和已知限制以 `docs/STATUS.md`、`EXPERIMENTS.md` 为准。
