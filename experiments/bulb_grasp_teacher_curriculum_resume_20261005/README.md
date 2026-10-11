# 灯泡两阶段教师同步续训：2026-10-05 17:48（Asia/Shanghai）

已将本地两阶段课程同步至 `dsw-wpmt7kzoh4b53jpmov:/home/admin/workspace/xinyu/mulyi-model-tac`，停止原单阶段进程21435/21451/21452；旧日志、模型均保留。

- 恢复父模型：`logs/rsl_rl/bulb_grasp_teacher_v2test/2026-10-04_scratch_2x4096/rank0/model_12000.pt`；SHA256 `0a4af76a184d476c1d6509e884aa814b9437a507734d2bf99fd4e323b77f7995`。
- 新run：`logs/rsl_rl/bulb_grasp_teacher_v2test/2026-10-05_curriculum_resume12000_2x4096`；torchrun PID62901；训练输出见同名experiments目录`train.log`。
- 两张卡各4096环境，总8192；保留seed42/43，追加18000轮，到原总预算30000（最后循环索引29999）。
- 同步7个新增/变更文件，93份相关文件SHA256全部一致；覆盖前备份到远端`.sync-backups/bulb_grasp_curriculum_20261005_1744`。源码未额外修改。
- 本地CPU16项通过；Isaac原生4环境×300步、严格加载model10300通过，1900观察/28动作、30Hz契约正常。该检查不是成功率评测。
- 两rank首个优化器更新发生梯度all-reduce，更新后参数哈希一致；新model12000优化器步数240020→240040，证明保留旧优化器状态。新旧同名model位于不同run，旧文件未覆盖。
- 新配置确认课程version1、回合8s、重定向奖励权重5、真实灯座路径。重力trigger5730保留，TensorBoard progress1、gravity_z−9.8；launch_contract中initial_gravity_z−1是课程初值，不是恢复后的有效重力；env.yaml sim.gravity也不是运行时重力课程值。
- 17:48:44核验到12012轮：阶段一60.9218%，完整两阶段0.53267%；训练标量而非独立评测，短期数值有波动。两GPU利用率77%/80%，显存8941/9155MiB。未发现Traceback/CUDA/PhysX致命错误；历史机器人visuals引用警告仍存在。
- ADR逐环境状态不在旧checkpoint，按当前配置初始难度0.7317创建环境；不声称完全恢复原MDP/环境随机状态。新成功判据不同，不能与旧抓取成功率直接比较。

证据：`launch.json`、`resume_source.json`、`sync_verified.json`、`startup_verified.json`、`local_smoke.json`、`local_validation.json`。旧重力课程已经完成；新任务继续全重力学习。

复现命令（已有作业运行，勿重复执行）：

```bash
RUN_DIR=/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/bulb_grasp_teacher_v2test/2026-10-05_curriculum_resume12000_2x4096 \
ITERATIONS=18000 bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh \
  --checkpoint /home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/bulb_grasp_teacher_v2test/2026-10-04_scratch_2x4096/rank0/model_12000.pt
```
