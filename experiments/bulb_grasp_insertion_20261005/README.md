# 灯泡课程v4：浅插入口5–6 mm

用户选择“插入螺纹入口约5–6 mm，先不旋紧”。本次仅本地实现和有限验证，无新版训练、远程同步或作业重启。

## 实现与口径

阶段一保持随机灯座口竖直上方15 cm、跟踪刚体中心，误差<5 cm晋级。
阶段二让尖端深入5.5 mm，由实际几何和倍率换算中心；名义尺寸中心高口沿48.2602 mm。
终态成功需阶段一已完成，插深5–6 mm、尖端和口沿处轴线误差均≤0.5 mm、轴向倾角≤5°、
相对线速度≤0.02 m/s和角速度≤0.5 rad/s，连续0.2 s（6控制步）。条件失效即清零。
两阶段共8 s，原奖励/权重、动作28、观测195/745/960和5帧历史保持，阶段二位置奖励改跟新目标。
这是手持浅插，不要求松手或旋紧。`success_reward`仍为原连续位置核，与二值成功指标不同。

新增派生bulb_insert.usd保留源外螺纹SDF512/两壳SDF256实际碰撞；灯座原三角碰撞不改。
原灯泡0.8–1.2倍随机化保留，灯座采用同环境同倍率，不单独扩大内孔、不删除难例。
灯泡初始Z提高1 cm保留与最大灯座的初始净空。源资产、旧抓取资产、点云和snapshot的保护哈希通过。

## 本次验证

- `cpu_checks.log`：首轮29项通过；`cpu_checks2.log`：最终30项通过（含真实Isaac事件参数验证器）。
- `native_smoke.json`：首次失败。旧v2的reset_socket可选env_ids被Isaac EventManager拒绝，已修为必需参数（调用时可传None）；没有隐藏失败。
- `native_smoke2.json`：本地RTX5070 Laptop，完整任务2环境×245步，通过strict原model_22200加载、有限观测/奖励、28维动作和回合重置，2个完成回合。只验证接线，不记录为插入成功数。
- `physics_probe3.json`（最终，进程正常退出）：无机器人/策略/关节引导的隔离碰撞诊断。倍率0.8/1/1.2，在制造的5.5 mm目标位姿零重力释放1 s，位移均小于0.000016 mm；从口上1 mm以参考相位正常重力自由下落2 s，末态深度12.19/9.06/15.62 mm，全部不满足浅插成功。零重力是明确标记的诊断条件，生产任务重力课程保持。
- `validation.json`：最终源码、派生资产和保护文件SHA；`source_snapshot/`：相关最终源码快照，`before/`：v3备份。

日志保留已有机器人visual引用和marker警告；本次未做渲染验收。
前两轮隔离探针均完成物理步并保存JSON，但收尾持续占用CPU/GPU；分别终止本次探针PID97567/107058，原报告physics_probe.json/physics_probe2.json保留。探针最终按ManagerBasedEnv的流程清理callbacks和SimulationContext单例，第三轮正常退出，未留下进程。完整任务native_smoke2本身正常退出。
全仓git diff --check仅报告用户原有根README.md末尾空行，本次未修改该文件。
Kit关闭可能掩盖失败退出码，必须检查JSON中的passed/completed，不能只看shell exit 0。

## 实际命令

CPU（USD路径按本机安装）：

```bash
USD_RUNTIME=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_RUNTIME" LD_LIBRARY_PATH="$USD_RUNTIME/bin:/home/liuxinyu/miniconda3/envs/brainco/lib" \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_grasp_curriculum.py
```

原生完整任务（使用原单阶段模型，仅验证接口）：

```bash
PYTHONPATH="$PWD/runtime/bulb_grasp_teacher" OMP_NUM_THREADS=2 \
bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --headless --device cuda:0 --num_envs 2 --steps 245 \
  --report experiments/bulb_grasp_insertion_20261005/native_smoke2.json
```

隔离碰撞探针：

```bash
ISAACLAB_ROOT=/home/liuxinyu/docker_workplace/IsaacLab \
PYTHONPATH=/home/liuxinyu/docker_workplace/IsaacLab/source/isaaclab \
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/probe_bulb_shallow_insertion.py \
  --headless --device cuda:0 --report experiments/bulb_grasp_insertion_20261005/physics_probe3.json
```

重跑请使用新报告路径，保护以上证据。首次派生资产由build_insertion_asset.py在CPU USD环境生成，输出见asset_build.json。

## 边界与后续

没有v4策略训练或独立成功率评测，不能用旧v1的训练成功率表示实际插入。
没有验证8192环境下的SDF显存/性能；仅名义参考相位的三个尺寸碰撞诊断，不能保证任意姿态自然落座。
远端运行代码未改变。后续训练需明确采用v4契约及新资产，不能只按网络维度宣称完全恢复旧任务。
GitNexus命令/配置类影响LOW；事件、新脚本、文档UNKNOWN，经配置、脚本入口、测试和导航引用补核。
仓库原有大范围改动保留，本次未提交。
