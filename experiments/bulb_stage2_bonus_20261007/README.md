# 二阶段浅插成功奖金 v7（2026-10-07）

用户要求二阶段浅插成功、进入第三阶段时增加独立的一次性奖金，并追问此前是否从未成功。

## 已实现

只改curriculum.py与env_cfg.py：新增stage_two_bonus_credit，在现有浅插连续保持达标、首次进入stage_three时置1，下一控制步清零。
stage_three本身持续锁存到reset，因此不需要额外paid标志；局部reset清除对应信用。
新增rewards.stage2_success_bonus，权重+10；reward函数除以step_dt，抵消RewardManager的dt乘法。
同帧physical terminated不给奖；时间上限不是物理失败，不抑制有效晋级。
第三阶段final_screw_success继续一次+20；原20项奖励、动作、观测、阶段门槛和物理不改。版本6→7。
+10是本次选择的起始值，不是已经验证最优的权重。

## 验证

48项CPU回归通过，包括稳定保持前不发奖、只付一次、失效后再次到位不重复、局部reset重新领取、同帧物理失败不给、三种dt金额一致。
补充测试证实：成功进入stage_three后，即使物体离开浅插位姿，stage2_success仍锁存；reset返回日志保留1。
第一次未带RUNBOOK的OpenUSD路径，5项资产检查报pxr缺失；按既有运行手册设置路径后48项全过，保留两次日志。
未运行新版本原生仿真、训练或独立成功率评测；正在运行的远端v6没有被修改或重启，新奖励尚未在该进程生效。

## 对“是否一次都没成功”的修正

15:33:55原始TensorBoard事件记录9500–10264共765个连续点，rank0 stage2_success、stage3_success、stage3_elapsed_s、旋入及最终奖励均严格0。
不是终端四位小数显示或TensorBoard抽样造成的零。rank0的二阶段成功标志锁存到回合结束，后来掉落不会擦除已晋级事实。
但rank1禁用日志，阶段二指标没有跨rank归约，不能断言两卡历史上从无浅插成功。
GravityCurriculum/success_rate跨两rank归约，765点全零，说明已记录的结束回合没有完整任务终态成功；不能排除中途达标后又失败。
瞬时insertion_candidate只在重置统计中记录，不能据其零值断言每一个物理帧都没有候选姿态。
上述结论不覆盖尚未结束或尚未写盘回合，也不是独立评测成功数/总数。

## 影响分析

绑定本工作区main/167fc3c，索引2026-10-07 11:59。GitNexus精确生产类UID上游分析均LOW，各3项，直接涉及env_cfg/Play/agent和benchmark导入，无索引进程。
测试类未索引UNKNOWN，补查fixture、unittest入口、注册及train/play调用。没有修改共享runner/记录格式，也没有安装依赖。

证据：source_delta.patch、verification.json、cpu_checks_usd.log、success_audit.json、before/。
