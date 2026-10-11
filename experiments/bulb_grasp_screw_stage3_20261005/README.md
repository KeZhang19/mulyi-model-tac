# 灯泡课程 v5：三圈旋入与最终成功奖励

2026-10-05 开始，2026-10-06 完成本地验证。任务为
`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0` / `...-Play-v0`。
Git 基线 `167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e`（main）；工作区已有其他未提交改动，本次未提交。

## 实现契约

- 阶段一仍跟踪灯泡刚体中心至随机灯座口上方 15 cm；阶段二仍浅插 5–6 mm、对中/姿态/低速连续保持 0.2 s 后晋级。前两阶段共用 8 s。
- 阶段三从晋级时单独计时 6 s，以实际浅插深度/相位为基线，沿实际螺距继续顺时针旋入 3 圈。只改变命令目标与奖励，不施加引导力、不增加约束关节、不改变原碰撞。
- 阶段三关闭两项 carry 相对线/角速度惩罚，阶段一、二保持。
- 带力新进度 +9/圈，首次每 1/4 圈里程碑 +0.75；受力质量门控和每秒 1 圈信用上限作用于二者，三圈最高 36 分。原接触四项最高 4.7 分/s，6 s 约 28.2 分。
- 新增历史最大回退按 −14/圈处罚，死区 0.01 圈，恢复不退款、同一幅度不重复处罚。有效计圈同时检查实际轴向推进、倾角、对中和螺旋误差，不能先空转再压入补领。
- 最终大成功 `final_screw_success` 权重 **20**：三圈、终点深度误差≤1 mm×倍率、对中≤0.5 mm、倾角≤5°、线速≤0.02 m/s、角速≤0.5 rad/s 连续保持 0.2 s 后，每回合一次性 +20。与物理失败同帧不发放；函数除以 step_dt 后由 RewardManager 乘回，实际奖金为 20。后续退出成功区不能重复领取，终态成功也不会被锁存。
- 沿用 28 维动作、195/745/960 观测组和 30 Hz 控制。MDP 版本变为 5，相同张量宽度不表示旧策略已适应三阶段。

详细配置见[任务说明](../../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。
设计参考 [RotateBulbGuidedRotation](../../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/env_cfg.py)
和 [RotateBulbRandomized 回退成本](../../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbRandomized/progress_penalty.py)，未修改这两个参考任务。

## 本次验证

| 检查 | 证据 | 结论与范围 |
| --- | --- | --- |
| CPU 逻辑 | [cpu_success_bonus2.log](cpu_success_bonus2.log) | 40/40 通过，包括完整三阶段转换、三圈耦合计数、空转/反向/重领防护、独立时限、局部重置及一次性 +20 |
| 原生完整任务 | [native_success_bonus.json](native_success_bonus.json)、[日志](native_success_bonus.log) | 2 环境 × 245 步、2 个完成回合；strict 加载原 model_22200、有限状态、动作/观测接口正常；确认 +20 注册及晋级先于超时检查 |
| 原生几何探针 | [physics_path.json](physics_path.json)、[日志](physics_path.log) | 0.8/1/1.2 倍，每 1/4 圈至 3 圈共 39 个制造位姿，各零重力释放 0.2 s；39/39 通过预设公差 |
| 源码/资产审计 | [validation.json](validation.json) | 修改范围 Python AST、快照哈希与保护文件哈希；最终相关源码保存在 source_snapshot/ |

几何探针最大中心位移约 0.00000782 mm；它只检查 CAD 对应螺旋位姿无明显碰撞挤出，
没有机器人或策略，不能当作自主旋入成功率。原生任务运行检查也不是三阶段策略成功率评测。
未训练 v5，未验证任意初始螺纹相位的自主旋入或 8192 环境 SDF 容量，未生成新录像；未同步或重启远端作业。

保留早期 CPU/原生检查日志。`cpu_success_bonus.log` 的唯一失败为测试使用精确浮点相等，
实际 19.999998 被误判；改用 Torch 标准浮点比较后 40 项通过，未为此修改奖励/成功阈值。
`before/` 和 `before_manifest.json` 保存此次修改前的 v4 相关代码及受保护资产哈希。

GitNexus 修改前影响分析：课程命令类和环境类均 LOW（3 个关联），新增符号/动态配置入口为 UNKNOWN；
已补查任务注册、Play 继承、配置调用、诊断和启动脚本引用。索引时间早于 v4/v5，UNKNOWN 没有被当作无影响。
未提交，因此没有以陈旧图的零结果充当提交验收。

## 实际执行命令

以下命令在仓库根目录执行。需要重跑时另选报告/日志路径，保留原始证据。

```bash
USD_RUNTIME=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_RUNTIME" LD_LIBRARY_PATH="$USD_RUNTIME/bin:/home/liuxinyu/miniconda3/envs/brainco/lib" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_grasp_curriculum.py

PYTHONPATH="$PWD/runtime/bulb_grasp_teacher" OMP_NUM_THREADS=2 bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --headless --device cuda:0 --num_envs 2 --steps 245 --report experiments/bulb_grasp_screw_stage3_20261005/native_success_bonus.json

ISAACLAB_ROOT=/home/liuxinyu/docker_workplace/IsaacLab PYTHONPATH=/home/liuxinyu/docker_workplace/IsaacLab/source/isaaclab OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/probe_bulb_screw_path.py --headless --device cuda:0 --report experiments/bulb_grasp_screw_stage3_20261005/physics_path.json
```
