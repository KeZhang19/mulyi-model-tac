# 用户插销抓取候选及导入验证

源导出：`revo3_template_editor_standalone/Data_generation/output/d_peg_custom/editing/simulation_exports/20260913_142053_9886dc87/`。
`simulation_state.json` 为原始导出的逐字节副本；校验值和原路径见 `provenance.json`。
84 MB 的 `scene.mjb` 和其加载脚本保留在原导出目录，未重复放入训练仓库。
`EDITOR_README_CN.md` 是原导出说明副本。原任务候选见 `previous_pregrasp_flexiv.json`。

本次已将 28 个关节及插销/插孔支架位姿原样接入
`assets/d_peg_insertion/pregrasp_flexiv.json`，供
`BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0` 启动与重置。
编辑器未导出关节执行器或预紧目标，因此初始驱动参考等于导出关节位置，
不能称为已校准的物理夹持预载。`validated=false`、`physics_validated=false`。

编辑器优化收敛但候选验收失败：最大接触锚点误差 18.4267 mm、最大穿透 3.7982 mm。
重载 MJB 后，食指/无名指锚点误差分别为 18.43/13.15 mm；
最深穿透发生在小指、食指、无名指和拇指指尖与插销握持部分之间。
详细接触对和碰撞记录见 `editor_diagnostics.json`。

本机 Isaac 检查使用 2 个环境、零位姿随机化；启动、3 次完整 reset 和部分 reset
共 33 项检查全部通过，实际姿态与原导出一致。
新任务同时修复了旧场景越界范围，以及 reset 覆盖支架原始旋转的问题。
默认完整观测下策略输入为 43+434+192=669 维，动作 28 维，观测均有限。

2 秒零动作（60 个控制步）抓取保持验收失败：

| 指标 | 环境 0 | 环境 1 | 验收阈值 |
| --- | ---: | ---: | ---: |
| 最大插销位移 | 6.59 mm | 11.56 mm | <3 mm |
| 最大插销转角 | 14.23° | 7.26° | <5° |

部分指尖接触低于验收阈值；期间没有终止或超时，支架位移约 5.6e-9 m。
完整观测与精简观测得到相同的物理结果，见 `full_runtime.json`、`geometry_runtime.json`。
这些结果验证导入/重置链路，尚不表示稳定抓取或成功插入。此次没有启动 PPO 训练。

复现完整观测检查：

```bash
/home/liuxinyu/miniconda3/envs/brainco/bin/python \
  assets/d_peg_insertion/tools/validate_flexiv_pregrasp.py --headless --num-envs 2 \
  --reference assets/d_peg_insertion/flexiv_grasp_reference/20260913_142053_9886dc87/simulation_state.json \
  --report /tmp/flexiv_d_peg_pregrasp_report.json
```
