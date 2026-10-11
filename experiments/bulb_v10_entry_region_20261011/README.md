# BULB-V10-ENTRY-20261011

用户要求二→三真实底端径向≤12 mm、孔口轴向深度0–10 mm。倾角<30°、无二阶段速度/持续保持维持。
只改变入口门槛，新增独立浅插径向配置；三阶段有效旋入2 mm/10°/螺距一致/接触、原最终0.5 mm/5°/低速/0.2 s及三圈验收保持。
二阶段固定5.5 mm直接底端目标、三阶段计圈/小领先深度目标、奖励权重、一次+10及5帧1900维MLP保持。无需改资产。
这是宽松的课程入口，不能视为物理啮合，也不保证已得到旋转正奖励。三阶段6 s从晋级开始，深度/相位也在晋级时记录；较宽入口可能带来对中时间不足与不同最终深度（标称18–28 mm）。入场后若先做超过1.5 mm量级的轴向调整而没有对应计入的旋转，仍可能被原螺距一致性门槛挡住；若出现此现象，应单独评估延后到有效啮合时建立计圈基准，本次未实施。学习效果未验证。

## 本地验证

`local_v9_before.tar.gz`及`before_sha256.json`保存修改前任务/训练入口；`changes.patch`仅包含本次代码差异。
CPU60/60通过：0/10 mm和12 mm含边界，负深度/过深/超半径拒绝，<30°且无速度/保持，偏轴旋转无奖励与调整后有效旋入，以及全部原阶段/资产/最终成功回归。
初轮失败日志保留：新测试先在偏轴时制造了1圈对应的深度，随后误期望这段无效进展能记账；测试恢复到原入口深度再开始有效旋入，生产旋入判据未改变。
严格契约10，显式9→10迁移可续训；缺失/旧v8/未授权版本默认拒绝。调用图存在CRITICAL与UNKNOWN/错配节点，已告知并补源码调用核查，不称图分析干净。

复现（CPU、不启动Kit/GPU）：

```bash
PYTHONPATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
LD_LIBRARY_PATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:/home/liuxinyu/miniconda3/envs/brainco/lib \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_grasp_curriculum.py
```

## 服务器更新

已验证并停止自己的v9 launcher4515/rank4531/4532；停止前日志1156，最近保存model1100。全部v9模型保留；未保存尾段不恢复。
7个相关文件SHA同步通过；`v9_replaced_files_backup.tar.gz`服务器备份及本地下载SHA一致；服务器CPU60/60通过。
14:44:07启动launcher8058，run `logs/rsl_rl/bulb_grasp_teacher_v10_mlp/2026-10-11_resume_model1100_2x8192`；双卡各8192/总16384，5帧MLP无LSTM，ADR0、原容量，目标总迭代上限30000（追加28900）。
全状态恢复包括优化器/归一化/重力；ADR/环境/RNG轨迹不作精确恢复承诺。显式9→10变更MDP记录于两rank迁移JSON，原v9及v8模型回放使用对应旧源码。
27项原生启动验收通过（`startup_verified.json`）：1100–1110共11次PPO更新，两rank实际版本10/阈值/5帧/8192、真实all-reduce及同参数SHA、optimizer22040→22060、归一化计数恢复后增长、模型/优化器有限、重力状态恢复。
外层及两份Kit日志无Error/CUDA/PhysX容量异常；原视觉引用Warning存在。两卡显存34575/35677 MiB。
原v9三份检查点SHA与尺寸/mtime保留，原v8七十份清单及model900/model6900哈希保留；launcher8058仍存活。学习效果/独立成功率尚未验证。`launch.sh`已经执行，请勿重复启动。
