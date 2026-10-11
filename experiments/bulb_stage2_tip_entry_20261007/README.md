# 第二阶段螺纹端点浅插：本地MDP v8

2026-10-07（Asia/Shanghai），用户要求最小修改第二阶段引导及二→三晋级条件。
仓库main/167fc3c，已有未提交/未跟踪修改保留。未连接远端或操作训练作业。
本机进程核查未发现train_bulb/play_bulb/torchrun/通用RSL训练进程使用本检出。

## 改动

- `curriculum.py`：二阶段目标为螺纹端点口沿下5.5 mm；按当前姿态换算中心命令，因此原`position_tracking`与`success`位置奖励都变为实际端点误差。目标相位沿用当前相位，不要求先转到CAD参考相位。
- 晋级要求：深度5–6 mm、端点径向≤0.5 mm、倾角严格<30°。不要求速度阈值或连续保持。口沿整根轴线对中仍供三阶段计圈/最终验收使用。
- 阶段一、第三阶段、动作、观测宽度、物理、随机化、奖励权重和一次奖金保持。MDP版本8；同维度旧模型不代表同一评测协议。
- `env_cfg.py`：新增二阶段专用角度参数；旧角度/速度/保持参数仅用于第三阶段最终成功。
- `play_bulb_grasp_teacher.py`：v8 HUD显示端点偏差、<30°、无速度/保持门槛；trace增加端点径向。旧`insertion_hold_s`指标为0，`insertion_radial_error_m`仍为端点/口沿轴线最大偏差。

端点定义沿用资产元数据`bulb_tip_offset_m`，是螺纹最前端截面轴心，不是倾斜后网格世界最低顶点。
保留原浅插深度；二阶段不要求或奖励旋转，不代表新增了“尚未物理啮合/无接触”的检测。
本次没有修改碰撞以允许斜插；新分布的物理可达性和策略表现待原生轨迹验证。
第三阶段从晋级时真实深度/相位开始计三圈，原≤5°、低速和0.2 s稳定判据保持。

## 验证

CPU课程检查52/52通过，使用真实Isaac数学、CommandTerm生命周期及USD读取，场景为张量替身。
新增覆盖高速单步晋级、精确30°比较、物理四元数29.9°/30.1°、实际尖端奖励/尺寸倍率、轴向相位与计圈隔离、三阶段稳定门槛。
保留原未达第一阶段、悬停/过深/侧偏、局部reset、+10/+20单次支付与失败门控、螺距/回退/空转、8+6 s超时和资产回归。
修改前文件在`before/`，哈希在`before_sha256.json`；首次旧断言失败保留`cpu_first.log`，通过日志为`cpu_second.log`。

复现（仓库根目录，仅CPU，不启动Kit/GPU）：

```bash
PYTHONPATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
LD_LIBRARY_PATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:/home/liuxinyu/miniconda3/envs/brainco/lib \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_grasp_curriculum.py
```

未启动Isaac物理、PPO、录像、远端同步/重启或独立成功率评测；CPU检查不证明物理可达或学习改善。

## 图分析与范围

修改前使用GitNexus upstream impact。索引commit匹配但内容过期且部分节点/路径错配：
课程类和配置类用精确UID后各3个上游、LOW，含配置导入及Play继承，但另出现无关benchmark引用。
按名奖励路径曾报CRITICAL、配置HIGH；测试/几何精确UID为UNKNOWN；回放文件候选HIGH/UNKNOWN。
已向用户报告风险，补查source/scripts引用：课程/配置用于教师train/play，奖励仅在此任务注册，检查函数在独立CPU入口调用。
返回流程混入历史快照和无关模块，不能把空流程/错误边当作无影响；本次未提交、未宣称图检查干净。
