# 灯泡 v5：DSW 双卡每卡 8192 环境

2026-10-06（Asia/Shanghai）。用户授权同步项目到指定服务器并启动双卡训练，
随后将服务器从四卡重启为两卡，明确通知重启完成后继续。

- 服务器：`dsw-c3g3e0ieowszisvolv`。
- 独立项目副本：`/home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006`。
- 任务：`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`，课程版本 5。
- GPU 0/1，每卡 8192 环境，总 16384；seed 42/43；从零训练，沿用启动入口 30000 轮。
- run：项目副本下 `logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192`。
- 日志：项目副本下 `experiments/bulb_v5_dsw_2x8192_20261006/train.log`。
- 启动 PID：948；启动参数见 [launch.sh](launch.sh)，身份见 [launch.json](launch.json)。

保留旧项目、旧模型和已有日志。同步源码、资产、任务数据、任务独立 RSL-RL runtime
及相关说明，1729 项约 549.6 MiB，全部通过远端 SHA256/相对链接核验。
不复制历史训练日志、检查点、采集数据和环境安装；使用服务器现有 Isaac/PyTorch 环境。
[同步清单](sync_manifest.json)包含本次具体范围；保留相对符号链接及所选目标资源。

## 碰撞容量

只在专用训练入口增加可选参数，原默认行为不变。本次每个 rank 使用：

| PhysX 字段 | 原教师快照 | 本次 |
| --- | ---: | ---: |
| gpu_collision_stack_size | 67108864（64 MiB） | 4026531840（3.75 GiB） |
| gpu_max_rigid_contact_count | 8388608 | 16777216 |
| gpu_max_rigid_patch_count | 1500000 | 3000000 |
| gpu_found_lost_pairs_capacity | 2097152 | 4194304 |
| gpu_found_lost_aggregate_pairs_capacity | 33554432 | 67108864 |
| gpu_total_aggregate_pairs_capacity | 2097152 | 4194304 |
| gpu_heap_capacity | 67108864 | 134217728 |
| gpu_temp_buffer_capacity | 16777216 | 33554432 |

碰撞栈及部分接触缓冲为 pinned host memory，不能直接加总为显存用量。
碰撞栈保持低于 uint32 的 4 GiB 上限；实际峰值和后期密集接触仍须由运行日志判断。
没有更改 SDF、自由刚体、碰撞开关、求解器、奖励、完整阶段或成功判据。
默认与覆盖逻辑的实际 AST 离线检查通过；本地课程 CPU 40 项全部通过，见 [日志](cpu_checks.log)。

GitNexus 使用当前仓库 main / 167fc3c 的索引（2026-10-05 22:39）；环境类 LOW，
训练文件 UNKNOWN。已补核单卡/双卡 shell 入口与 split_gpu 的 runpy 动态调用链。
本次未提交。工作区原有未提交改动保留。

## 运行验证

00:40–00:41（Asia/Shanghai）本次核验至日志迭代9/30000，已完成10次PPO更新。
[startup_verified.json](startup_verified.json)确认：

- 两rank实际各8192，总16384，种子42/43；课程version5、最大回合14s、120Hz物理/30Hz控制、最终成功奖励20。
- 每rank真实执行gradient all-reduce；首次优化器更新后两rank策略参数SHA256相同。
- `rank0/model_0.pt`已保存，25个模型状态张量全部有限；优化器step均为20，证明发生实际学习更新。
- 外层训练日志及两个Kit原生日志完整扫描，未发现Traceback、CUDA OOM、PhysX Error、碰撞容量溢出或丢接触信息。
- GPU0/1显存35427/35131 MiB（约34.6/34.3 GiB），利用率99%/100%；约10GiB/卡显存余量。
- 首次冷启动约数分钟；保留原有机器人visual引用等警告，没有以关闭碰撞解决警告。

这是启动和初期容量验证，没有测量最小所需容量；后期密集插入/旋入接触峰值仍未知。
从零训练使用ADR初值0及原重力课程初值−1m/s²；不是旧模型的全重力续训。
不把启动或初期更新当作三阶段学习成功。独立完整任务成功数/总数尚未评测。
