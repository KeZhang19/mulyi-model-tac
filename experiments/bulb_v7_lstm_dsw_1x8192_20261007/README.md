# v7 单卡8192环境LSTM：同步及启动准备

2026-10-07 17:16（Asia/Shanghai）。用户要求同步项目到指定DSW并启动单卡8192环境LSTM，
完整抓取抬起→对齐浅插→三圈旋入。

- 主机：`dsw-wpmt7kzoh4b53jpmov`。
- 已同步独立目录：`/home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007`。
- 本地来源：main / `167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e`及当前未提交文件；不等于干净commit快照。
- 范围：1733项源码、运行依赖、资产、任务数据和文档，含完整source/scripts/runtime。
  不复制Git历史、旧训练日志/模型、原始采集数据或凭据。非训练标定资产沿用上次部署清单。
  `sync_manifest.json`为本地当前SHA/链接；未变文件从远端v6逐项校验后复制，20项变更由本地上传。
  `sync_verified.json`：1733/1733通过，旧目录未覆盖。已有断链`assets/calibration_bench_urdf`保留原链接文本，与本任务无关。

## 已验证

- DSW现有Python/Isaac Lab可用，无新软件安装。
- LSTM CPU检查7/7通过，包括真实循环PPO两轮/40次优化、记忆重置、bootstrap和checkpoint往返。
- 课程CPU检查48/48通过，包括v7二阶段一次+10和最终+20、完整阶段/旋入判据、USD几何。
- 首次课程检查漏设ISAACLAB_ROOT失败；第二次48项中5项因缺少pxr搜索路径失败。
  加入现有Isaac omni.usd.libs路径后48/48通过。三份日志均保留，没有修改测试或生产代码。
- GitNexus query输出含异常映射；impact未找到configure_bulb_teacher_lstm，risk UNKNOWN。
  已补核train/play引用及teacher_runner注册；未修改现有函数/类，不宣称图分析干净。

## 准备的启动配置（尚未执行）

`launch.sh`使用现有split_gpu包装器的单进程模式，隔离Kit缓存及渲染GPU；无torchrun/DDP。
GPU0、8192环境、seed42、从零、30000次更新、每次32步，预定输出：
`logs/rsl_rl/bulb_grasp_teacher_v7_lstm/2026-10-07_scratch_1x8192`。

`--lstm`：观测历史1帧，39/149/192共380维，actor/critic各单层LSTM256及原MLP512/256/128，28动作。
旧MLP不加载；ADR从0、重力按原从零课程。完整v7奖励/物理/成功判据保持。
容量沿用已运行的8192配置：七项buffer×2、碰撞栈4026531840字节，不改变碰撞质量。
脚本拒绝覆盖已有run；执行前再次确认GPU及作业身份。

## 阻塞与交接

17:14远端旧v6双卡作业仍活动，launcher PID1446、rank PID1468/1469，cwd均为旧v6目录。
两卡分别约34405/35689 MiB、利用率96/97%，不足以追加新8192环境任务。
最新已保存rank0检查点为model10700；PID和checkpoint只代表记录时刻，操作前重查。
`preflight_old_job.json`保存身份及路径。旧训练/TensorBoard均未停止或重启。

已向用户询问是否停止旧双卡训练、保留检查点后在GPU0启动新训练；当前未收到确认。
需先取得停止旧作业授权，再重新核实PID/cwd/最新检查点，停止旧launcher及其rank，
确认释放GPU后后台执行launch.sh。随后核验launch_contract、env/agent配置、LSTM权重、
至少若干实际PPO更新及外层/原生PhysX日志；原生8192环境容量和学习效果尚未验证。

## 17:22后续：用户改为双卡

用户确认停止旧训练后，17:20向核实过的v6 launcher1446发送SIGTERM，两个rank均退出。
旧13份检查点保留，最新model10700停止前后SHA均为
`c524166c6390aab46ad5cd1116c7eee851f2b6a42573ee48ad3d3725f5921828`。
17:21启动单卡PID13929后，用户要求改为双卡每卡8192；单卡在初始化期间停止，无训练结果声明。
17:22:17启动双卡launcher14243/rank14259、14260，见[双卡记录](../bulb_v7_lstm_dsw_2x8192_20261007/README.md)。
本节取代上方“未收到确认/旧作业活动”的历史状态。
