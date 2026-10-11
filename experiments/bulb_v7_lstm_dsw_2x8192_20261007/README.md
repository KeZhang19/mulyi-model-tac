# v7 LSTM双卡：每卡8192环境

2026-10-07（Asia/Shanghai）。用户先要求单卡，随后明确改为双卡每卡8192；最终配置总16384环境。

- DSW：`dsw-wpmt7kzoh4b53jpmov`。
- 项目：`/home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007`。
- 任务：`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`，最新v7完整抓取抬起→对齐浅插→旋入三圈。
- run：项目下`logs/rsl_rl/bulb_grasp_teacher_v7_lstm/2026-10-07_scratch_2x8192`。
- 外层日志：项目下`experiments/bulb_v7_lstm_dsw_2x8192_20261007/train.log`。
- 17:22:17启动，launcher14243、rank14259/14260。后续操作须重新核实PID/cwd，勿重复启动。
- GPU0/1各8192环境、seed42/43，从零30000更新；无旧MLP权重加载。

## 配置及同步

LSTM观测历史1帧，policy/proprio/perception为39/149/192，共380维；actor/critic各单层LSTM256，
后接原MLP512/256/128，28动作。PPO rollout32、5 epochs/4 minibatches、每100更新保存。
完整v7阶段、物理和判据保持：120Hz物理/30Hz控制、前两阶段共8秒/第三阶段6秒，
浅插达标一次+10、最终三圈稳定达标一次+20。ADR从0，重力按原从零课程。

每rank七项PhysX buffer×2、collision stack4026531840字节（3.75GiB），沿用已运行的8192容量配置。
未降低SDF、碰撞或求解器质量。`launch.sh`使用既有双卡启动器与`--lstm`；每rank独立Kit缓存和GPU映射。

本地main/167fc3c及当前未提交内容同步，1733/1733项SHA/链接通过；无生产代码改动或新依赖安装。
DSW CPU LSTM7/7、课程48/48通过。完整清单、失败测试重跑日志和说明见
[前置同步记录](../bulb_v7_lstm_dsw_1x8192_20261007/README.md)。

## 旧作业保留及单卡取消

用户确认后核实旧v6 launcher1446/rank1468/1469身份，再向launcher发送SIGTERM，全部退出。
13份已保存检查点保持，最新rank0/model10700停止前后SHA均为
`c524166c6390aab46ad5cd1116c7eee851f2b6a42573ee48ad3d3725f5921828`。
旧run仍在v6目录，TensorBoard服务未改。没有将未保存的迭代宣称为已保存检查点。
单卡PID13929于17:21启动，用户改双卡后在初始化期间停止；保留启动/停止记录，不计为训练结果。

## 启动验证

`verify_startup.py`只读核对两rank实际配置、8192环境、LSTM380→256、v7/奖励/容量、
真实梯度all-reduce及参数SHA一致、model0/optimizer有限性和更新数、旧checkpoint保持，
以及外层和两份Kit原生日志。至少10次PPO更新后输出`startup_verified.json`。
17:29启动验收完成，`startup_verified.json`的17项检查全部通过，观察0–10共11次PPO更新（5767168个环境步）。
两rank首次更新真实all-reduce，参数SHA一致；model0包含actor/critic的380→256 LSTM，
优化器step20、模型/优化器全部有限，观测归一化计数为正。实际两rank配置为v7/8192/单帧LSTM，
120Hz/30Hz/14s、浅插+10/最终+20、ADR0和初始重力−1，容量与计划一致。
外层及两份Kit原生日志无Error或致命/容量溢出匹配；既有visual引用警告保持。
验收时显存GPU0/1为39705/43499MiB（每卡总46068MiB）；这是采样值，不保证后期密集接触/循环序列峰值。
训练继续，不自动重启；没有独立三阶段成功数/总数，初期日志完整任务成功率0不能作为独立评测。

## 操作入口

远端查看训练日志：

```bash
tail -n 150 /home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007/experiments/bulb_v7_lstm_dsw_2x8192_20261007/train.log
```

检查点每100更新写入run下`rank0/model_N.pt`，rank1不独立保存模型。检查进程先核实PID、cwd和命令；
如需停止本作业，应向已核实的torchrun launcher发送SIGTERM并确认两rank退出。
旧TensorBoard6006仍指向旧v6 run；本次未更换该服务或其隧道。

## 19:53进度与TensorBoard

事件至540轮，最新model500，近期16.5866秒/轮；rank0近100轮stage1_success0.106419%，
stage2/3及旋入圈数全部历史0。平均奖励首100轮6.56278→近期8.16275，但尚未稳定完成第一阶段。
异常机器人终止27.4303%，non_finite全0；无独立评测。详见progress_20261007_1952.json。

本机 http://127.0.0.1:6006/#scalars 现已指向本次LSTM run（139个标量标签），浏览器已打开。
远端新TensorBoard PID20216、loopback6008；本机SSH隧道PID381517、6006→6008。
原远端v6 TensorBoard6006保持；本机6007的其他任务保持。隧道依赖本机和DSW连接。
具体命令见tensorboard_launch.json与tensorboard_tunnel.json。
