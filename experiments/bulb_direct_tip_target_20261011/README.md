# v9 直接底端目标与完成标记

2026-10-11；本地 main/5d9d4885，保留所有既有改动。修改前核查本机无本项目训练/回放。

二/三阶段7D位置命令直接保存螺纹端点目标；实际端点、两项位置奖励、指标和可视化共用该参考点。
二阶段孔内5.5 mm目标不随灯泡姿态平移；三阶段仍使用真实计圈及螺距+小领先的原深度曲线。
阶段一中心到孔口上15 cm/<5 cm晋级不变，浅插5–6 mm/径向≤0.5 mm/倾角<30°和最终旋入判据不变。
无需修改资产/刚体原点/质量/碰撞；底端定义仍为螺纹最前端截面轴心。
5帧/1900维输入及28维动作不变，但命令语义不同，不能据维度认为检查点兼容。

标记点缩小，仅按实际阶段成功谓词变绿；HUD/trace显示各阶段完成状态、真实参考点与目标XYZ及逐项条件。
浅插条件直接读取生产张量谓词，避免显示层浮点换算造成边界判定不一致。
[坐标/完成状态示意](direct_tip_completion.png)、[生产HUD预览](hud_preview.png)是构造CPU状态，不是训练或物理成功证据。

本地及c3g服务器CPU58/58通过，使用Isaac数学/CommandTerm、USD资产读取和场景替身；覆盖旋转/尺寸固定目标、真实端点奖励、
第三阶段底端曲线、标记成功颜色、完成状态/局部reset及检查点契约。初次检查缺USD路径及一个旧中心断言失败，日志保留。
`cpu_checks_final.log`为通过日志；`before/`保留修改前文件（train入口新增检查前态由撤销本次三行检查准确重建）。
新文件及与前态的差异见`changes.patch`，验证摘要见`validation.json`。

复现（不启动Kit/GPU）：

```bash
PYTHONPATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
LD_LIBRARY_PATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:/home/liuxinyu/miniconda3/envs/brainco/lib \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_grasp_curriculum.py
```

相同环境运行`experiments/bulb_direct_tip_target_20261011/render_preview.py`重绘图与HUD。
尚未运行新版策略录像/独立评测；原生双卡续训的结果另见[续训证据](../bulb_v9_resume_20261011/README.md)。

GitNexus索引落后1提交且部分节点损坏：精确课程UID报CRITICAL/801影响、配置HIGH/3影响，
目标方法/奖励/几何/检查脚本UID未找到（UNKNOWN）；按文件查询目标方法错配到历史快照。
已向用户报告风险；补查真实引用，生产链为独立教师配置→命令advance/compute→奖励/回放，
旧中心helper只在几何探针/构造状态使用，未改其他任务。未提交，未宣称图分析干净。

默认train/play严格检查相邻params/env.yaml。用户随后明确授权从早期v8检查点续训，
专用训练入口增加`--migrate-v8-tip-command`，仅显式允许8→9并保存迁移JSON；回放保持严格检查。
