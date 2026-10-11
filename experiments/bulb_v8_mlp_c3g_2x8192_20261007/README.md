# v8 五帧 MLP：c3g双卡各8192环境

## 2026-10-11只读核查

该目录已无活动训练进程，最后日志第6971轮（0起编号、共6972次已记录更新），没有完成计划30000轮。
最新已保存`rank0/model_6900.pt`，70份检查点；最后事件时间10月9日00:31:53。未重启作业，停止原因未由日志确认。
原始rank0事件全部6972点：最近100轮进入第二阶段87.54885%，进入第三阶段0.0199116%；最近500轮分别88.11774%/0.0292633%。
第三阶段晋级首次出现在1593轮，593个训练轮次有非零记录；该数字不是晋级回合数。
完整三圈成功指标始终0，重力全程−1m/s²。没有稳定完成旋入的证据。
这些率是rank0重置批次均值的逐轮平均，rank1无事件文件；无法精确重建两卡累计晋级回合数，也不是独立评测成功率。
[完整统计](progress_20261011.json)。下方保留原启动记录。

2026-10-07（Asia/Shanghai）。用户授权同步本项目到指定DSW并启动完整抓取抬起→浅插→旋入三圈。
最初要求LSTM，随后明确停止旧训练、改回原5帧历史；本次实际启动为MLP，未启动LSTM。

- 指定实例：`dsw-c3g3e0ieowszisvolv`，实际hostname `dsw-1097512-7bf444cb8b-pgwdn`。
- 独立项目：`/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007`。
- 任务：`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`，MDP v8。
- GPU0/1各8192环境，共16384；seed42/43；从零训练，不加载旧模型。
- 使用原5帧历史，policy/proprio/perception=195/745/960，共1900维；ActorCritic MLP512/256/128，28动作。
- 沿用启动器默认30000次PPO更新、rollout32、5 epochs/4 minibatches、每100轮保存。
- run：项目下`logs/rsl_rl/bulb_grasp_teacher_v8_mlp/2026-10-07_scratch_2x8192`。
- 外层日志：项目下`experiments/bulb_v8_mlp_c3g_2x8192_20261007/train.log`。
- 22:35:08启动，launcher PID98487；身份及参数见`launch.json`。操作前须重新核对PID/cwd。

## 任务与容量

本地v8第二阶段按当前姿态下的螺纹端点引导到口沿下5.5 mm；深度5–6 mm、端点径向≤0.5 mm、倾角严格<30°单步晋级。
第二阶段不检查速度/保持时间，不要求先旋转对相；第三阶段实际三圈、螺距、≤5°/低速/0.2 s稳定及一次+20保持。
二阶段一次+10、120Hz物理/30Hz控制、前两阶段共8s/第三阶段6s、资产及碰撞保持。详情见
[本地修改记录](../bulb_stage2_tip_entry_20261007/README.md)。
从零ADR0、重力−1m/s²，随后按完整任务成功触发原课程；不代表全重力下训练。

每rank七项PhysX buffer×2，collision stack4026531840字节（3.75GiB），不改变碰撞质量或判据。
部署过程未修改生产源码，直接使用当前本地任务和原训练入口；`launch.sh`未传`--lstm`，拒绝复用已存在run。

## 同步与旧训练保留

本地来源main/167fc3c及当前未提交/未跟踪源码；这是带SHA清单的工作区快照，不是干净commit。
1735项约549.8MiB源码、任务独立runtime、资产、任务数据和相关说明；不含旧训练日志/模型、原始采集数据、环境安装或凭据。
完整选择见`sync_manifest.json`。1724项先与远端现有v7副本按SHA/链接比较后复制，11项从本地上传；旧副本未改。
最终1735/1735项通过：`sync_verified.json`。77项教师/运行库关键文件另与当前本地SHA核对一致。
两个状态文档在同步中被并行工作更新，最终清单固定为已上传tar的实际字节；初始清单和问题说明保留。

用户明确授权停止旧c3g v5；核对launcher948/rank1009/1010的cwd、cmdline、start ticks后向launcher发SIGTERM。
退出超过首轮30秒等待，后续确认三个进程均退出且GPU无计算进程，没有额外强杀。
122份已保存检查点保留，最新`model_12100.pt`停止前后SHA256均为：
`d4cb434848dc55130169063de874942c7b0f418599ba0dfa95ed2b0ad18277d1`。
证据`old_stop_requested.json`、`old_stopped.json`；旧TensorBoard和其他实例任务未操作。

## 验证状态

远端CPU课程52/52通过，见`check_bulb_grasp_curriculum.log`和`cpu_verified.json`。
第一次准备CPU路径时glob误匹配python3.1/3.11双路径，尚未启动检查；改为查询既有解释器实际版本，未安装依赖。
初始同步文档不一致与上述路径问题见`preparation_issues.json`；运行代码未因此修改。

原生启动验收通过，观察迭代0–9共10次PPO更新，17项检查全部通过，见`startup_verified.json`。

- 两rank实际各8192、v8、5帧历史、ActorCritic MLP；actor/critic首层权重均为512×1900，未含LSTM memory参数。
- 首次更新真实执行gradient all-reduce，两rank参数SHA一致；`rank0/`、`rank1/`保存实际launch/DDP/env/agent证据。
- model0模型/优化器均有限，优化器step20、归一化已采样，证明实际执行了训练更新。
- 二阶段<30°及v8目标、第三阶段原稳定配置、奖励、随机化、容量均符合本次启动计划。
- 完整外层日志和两份原生Kit日志无Error、CUDA/PhysX致命错误或接触/碰撞容量溢出；保留原有visual引用警告。
- 验收时显存35429/35131MiB（每卡总46068MiB），利用率95%/95%。后期密集接触峰值仍未知。
- 旧model12100哈希再次一致。训练继续，未改变其他实例或旧TensorBoard。

`startup_train_tail.log`保存启动阶段日志尾部，`active_processes.json`保存本次训练身份。
这是启动/初期容量与数值验证，不等于学会任务，尚无独立三阶段成功数/总数。
