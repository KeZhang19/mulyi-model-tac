# 灯泡v6：双卡8192环境，从v5 model9500续训

2026-10-07（Asia/Shanghai）。用户授权同步到指定DSW、双卡各8192环境启动完整任务，
并由本次核查决定碰撞容量和续训方案。

- 服务器：`dsw-wpmt7kzoh4b53jpmov`；启动时两GPU空闲，各46068 MiB。
- 独立项目：`/home/admin/workspace/xinyu/mulyi-model-tac-v6-20261007`。
- 任务：`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`，课程版本6。
- run：项目下`logs/rsl_rl/bulb_grasp_teacher_v6/2026-10-07_resume9500_2x8192`。
- 外层日志：项目下`experiments/bulb_v6_dsw_resume_20261007/train.log`。
- 2026-10-07 12:38:40启动，torchrun PID1446，rank PID1468/1469；后续操作前重新核实身份。
- GPU0/1各8192、总16384、seed42/43。追加20500轮，原runner从9500计数至29999。
- 完整抓取抬起→浅插→三圈旋入，每回合仍从第一阶段开始。

## 续训选择与状态边界

使用v5项目下`logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192/rank0/model_9500.pt`。
SHA256为`a7c1862d74bf39a3fe9303a0b0085adf8c2806091e3d69b83a64e32fd693ad2b`。
模型全部张量有限、原优化器步数190020，保存学习率约2.1825e-5。
最近保存日志9510轮stage1_success约88.96%，stage2/3成功为0；这不是独立评测。

选择该模型是因为其SDF自由刚体、三阶段目标、120Hz物理/30Hz控制、28动作/1900观测、
归一化和策略配置均与新任务相近。旧v1的model29999使用不同的两阶段目标、凸包和重力状态，未选作父模型。
本次保留策略/critic、归一化、优化器及重力课程，属于修改奖励后的继续学习，
不宣称完整恢复逐环境物理状态、ADR状态或随机数序列。critic将随新奖励重新适应。

原checkpoint不存逐环境ADR；专用train入口增加可选`--initial_adr_difficulty`，本次显式传0，
匹配v5原分布，避免恢复时回落到源教师默认0.7317。未指定时保留原行为。
重力trigger为null，恢复原课程时有效重力仍为−1m/s²；是否晋级由完整三阶段终态成功决定。
参数验证执行实际AST分支：5个正常情况、4个非法值通过，见`adr_override_checks.json`。

## 同步和资源

从服务器已保存的v5源码/资产清单构造新副本，再覆盖本地v6修改、训练入口及相关文档。
1735/1735项文件或相对链接匹配预期SHA/链接文本。旧源仅有4份状态文档相对10月6日清单变化，
新副本采用本地当前文档；旧目录和检查点不覆盖。必要运行依赖沿用服务器已有环境，没有安装新软件。

每rank采用已运行过的8192容量配置：

| 字段 | 本次值 |
| --- | ---: |
| gpu_collision_stack_size | 4026531840（3.75 GiB） |
| gpu_max_rigid_contact_count | 16777216 |
| gpu_max_rigid_patch_count | 3000000 |
| gpu_found_lost_pairs_capacity | 4194304 |
| gpu_found_lost_aggregate_pairs_capacity | 67108864 |
| gpu_total_aggregate_pairs_capacity | 4194304 |
| gpu_heap_capacity | 134217728 |
| gpu_temp_buffer_capacity | 33554432 |

对应原64MiB碰撞栈提高到3.75GiB、其余七项原默认×2。保持uint32碰撞栈上限以下；
不把这些容量简单等同于显存分配，部分使用pinned host memory。未额外翻倍、未改SDF/碰撞/求解器。
后期密集插入接触的最大容量需求仍待训练验证。

## 验证与证据

- `launch.sh` / `launch.json`：可复查命令和进程身份；已有作业时勿重复执行。
- `sync_manifest.json` / `sync_verified.json`：同步前后验证。
- `resume_source.json`：父模型与选择依据。
- `cpu_checks.log`：服务器CPU44/44课程检查通过。
- `training_entry_delta.patch`：本次唯一新增训练入口改动；默认不改变。
- `verify_startup.py`：只读核对配置、真实all-reduce、模型/优化器更新、父模型哈希及两rank原生日志。
  它不启动或干预训练。成功报告为同目录`startup_verified.json`；尚未生成时不能声称启动验收通过。

原有visual引用警告保留。本任务没有创建自动重启或长期监督循环，也没有独立成功率评测。

## 12:48启动验收通过

`startup_verified.json`全部18项检查通过，观察到9500–9511共12次PPO更新。
两rank真实执行gradient all-reduce，首次更新后的参数SHA相同。
新model9500优化器步数190020→190040，actor权重实际改变且全部模型张量有限；
归一化计数未重置、重力trigger恢复，父模型SHA保持。
实际配置确认两卡8192、v6、floor0.2、ADR0、14s/120Hz/30Hz；
动作、观测（仅项目路径重定位）、命令、PhysX和其余19项奖励与父run一致。
显存34403/35687MiB，两卡利用率95%/97%；外层及两份Kit原生日志没有Error或致命/容量溢出匹配。
这些检查证明任务已按新奖励继续学习；后期峰值及独立完整任务成功率仍未知。
