# TACSL-TRAIN4096-20261007

用户本次授权启动至少 4096 并行环境训练。已在 RTX 5090 主机 `brainco1-ThinkStation-P3-Tower-Gen-2` 启动 **4096 环境**作业，当前有效运行是 **v3**。没有占用其他主机或 GPU，没有恢复此前 8 环境的短训练权重。

## 当前运行

- 工程：`/home/brainco1/workspace/xinyu/tacsl_insertion_lab`
- 运行目录：`reports/train4096_20261007_v3/train`
- systemd 用户服务：`tacsl-train4096-20261007-v3.service`
- 启动 shell PID 48279，Python 训练 PID 48296；PID 为启动时记录，后续先核实服务状态。
- 4096 环境，seed42，从零初始化；PPO MLP [256,128,64]，14 维原默认状态观测，6 维动作。
- 计划 1500 次更新；每次 128×4096=524288 transitions，合计上限 786432000；4 epochs、32 minibatches，每 minibatch 16384。
- lr1e-4、gamma .99、GAE .95、clip .2、value coef2、entropy0；每25次更新保存，迭代0已保存 `model_0.pt`。
- 原奖励、严格成功阈值、完整失败分布和 reset/控制步长保留。关闭不参与状态策略的触觉采集计算；GelSight 机器人资产、碰撞与 compliant contact 保留。此次不是触觉图像策略训练。
- PhysX 相关容量 scale4，collision stack **4293918720 bytes**（4 GiB−1 MiB）；仅扩充容量，不降低碰撞质量、删接触或修改求解器16/0。
- 服务 MemoryMax=52G、MemorySwapMax=0、RuntimeMaxSec=24h。按启动吞吐粗估约23小时；实际时间随接触负载变化，24小时是后台服务上限。

任务完成训练后，脚本自动载入 `final.pt`，以 seed1042、256环境×4回合评测 **1024 回合**，结果写到同级 `eval/report.json`。异常退出/限时终止不会运行这个后续步骤；届时保留已有检查点，不能将未完成评测写成成功。

## 启动验收

[startup_verified.json](startup_verified.json) 的时间为 2026-10-07 16:57:22 +08:00：

- 已真实完成迭代0–4，共5次更新、2621440 transitions；完成8192回合，训练成功 **0/8192**。是训练初期指标，未作收敛或学会任务结论。
- `model_0.pt` 的 actor/critic 网络权重相对初始化的 L1 变化分别29.6532 / 66.9937，optimizer128步。
- 检查点、optimizer张量和TensorBoard标量全部有限。
- v3运行日志未发现 Error/Fatal/Traceback/接触丢弃；显存28274/32607 MiB，GPU利用率82%，服务活动。
- 日志仍保留原基线已记录的非致命 plug 子路径视图查询/headless/TGS warning；没有把 warning 包装为完全后端等价。
- 8环境的日志扩展回归完成2次更新、一次全体重置；原任务10方法AST及奖励/成功/动作解析检查再次通过。

## 失败记录

1. `reports/train4096_20261007_v1`：collision stack为scale4后的256MiB；PhysX报告接触被丢弃，所需峰值4062841280 bytes。主动停止本次作业，**不使用其检查点作为有效训练结果**。
2. `reports/train4096_20261007_v2`：3.75GiB仍小于所需峰值；新增guard检测第一条错误后退出，未进入有效PPO训练。
3. `reports/train4096_20261007_v3`：按上述接近4GiB容量从零重启，前5次更新和2次全体回合结束已验收。初期通过不证明后续任何接触峰值都不会超容量；guard保留实时失效保护。

`scripts/guard_run.py` 记录完整日志并在仿真 Error/Fatal、CUDA/OOM、异常 Traceback 或明确接触丢弃时终止其自己启动的进程组。合成错误注入检查确认退出码2，v2真实溢出也触发退出。RSL的NaN检查保持开启。不会自动重启坏配置继续积累训练。

## 观察与控制

本机 TensorBoard：[http://127.0.0.1:6007/#scalars](http://127.0.0.1:6007/#scalars)。只读查看，不影响训练；与已有6006端口的其他任务分开。
服务器 TensorBoard 服务 `tacsl-tensorboard-20261007.service` 仅监听127.0.0.1:6007，本机SSH仅转发loopback。
关注 `Metrics/success_rate`、`Metrics/keypoint_error_m`、`Train/mean_reward`，训练成功不能替代独立评测。

```bash
ssh brainco1@192.168.10.252 'systemctl --user status tacsl-train4096-20261007-v3'
ssh brainco1@192.168.10.252 'tail -40 /home/brainco1/workspace/xinyu/tacsl_insertion_lab/reports/train4096_20261007_v3/train.log'
# 仅在用户要求停止时执行；会停止训练及排队的评测
ssh brainco1@192.168.10.252 'systemctl --user stop tacsl-train4096-20261007-v3'
```

已有作业时不重复启动。[launch.sh](launch.sh) 是本次冻结启动脚本；`--output` 必须选新目录，服务名也必须唯一。
验收脚本 [verify_startup.py](verify_startup.py) 只读检查模型/标量/服务和日志，输出当前启动验收快照；[code_sha256.json](code_sha256.json) 固定本次代码。
运行目录 `code/` 保留对应脚本和任务源码，原始资产来源沿用[移植实验](../tacsl_insertion_20261007/README.md)。

## 本次代码范围

`run.py` 增加horizon/minibatches/保存间隔/容量/compact logging参数；默认小规模入口保留。
`env.py` 仅将末步结果向量化、去掉跨步残留的日志值，并在compact模式将详细reset样本限为前8环境、按批次记录**全部**环境的成功分子/分母和误差；没有从仿真或训练中剔除失败环境。
`env_cfg.py` 增加记录开关；`train4096.sh`/`guard_run.py` 管理独立后台运行和失败保护。
修改前核实无活动仿真作业；GitNexus env LOW（直接调用入口run.py）、cfg MEDIUM（图9直接引用/1过程），脚本UNKNOWN并补查CLI/动态注册引用；仅改独立工程。根文档UNKNOWN（未索引），补查README/AGENTS引用后仅增加索引。
已有本仓库工作、远端Lab检出和其他任务保持原样；未创建Git提交。
