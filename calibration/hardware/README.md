# 真机采集入口

2026-10-12 当前主标定采用视触觉与台架标准六维力。
`py_force_calibration` 的旧压阻 ADC 采集属[其他标定](../../../ros_ws/src/py_force_ws/doc/other_calibration/README.md)，暂不参与主流程。

当前工作顺序见[本地标定流程](../docs/local_workflow.md)，坐标与预定位点见[config/](../config/README.md)。
本项目SDK副本位于 `calibration/vendor/brainco-revo3-sdk/runtime/python`；
原 `calibration/brainco-revo3-sdk`保留兼容链接，解释器选择保持。

面向仿真—真机对齐训练的详细字段、参考关联、米制深度派生与训练转换条件见
[真机采集数据接口](../docs/real_data_interface.md)。

## 标定台 ROS 2 接口

2026-10-07 已按 `/home/liuxinyu/workspace/ros_ws/src/py_force_ws/` 的接口定义和节点源码核对。
下表替代旧机械说明中“台架SDK待提供”的占位描述；本次仅静态核对，尚未联合采集或真机验收。
`py_force_interfaces/{action,srv,msg}` 是字段来源，`py_force_control` 负责运动、限位和测力。
复用其 ROS 2 接口，不在本仓库另开电机串口或复制底层驱动。

| 用途 | ROS 名称／类型 | 字段与语义 |
| --- | --- | --- |
| 单轴定位 | `/axis_move` · `AxisMove` action | `axis="X"/"Y"/"Z"`、`target_mm`、`timeout_s`；结果含 `success/final_position_mm/status` |
| 三轴定位 | `/move_to_3d_pose` · `MoveTo3DPose` action | `x_mm/y_mm/z_mm/safe_z_mm/timeout_s`；依次 Z安全位→X→Y→Z目标，不是五轴姿态接口 |
| 接触／按压／抬起 | `/z_touch`、`/z_press`、`/z_lift` actions | `ZTouch` 无请求字段，结果看 `terminated_by_force`；`ZPress` 用 `target_force_n/hold_time_s/small_force_mode`；`ZLift` 用 `max_position_mm` |
| 位置查询／停止 | `/axis_query` · `AxisQuery`、`/axis_stop` · `AxisStop` services | 请求均为单轴 `axis`；检查 `success/status`，失败时返回的0不能当位置；停止不接受 `ALL`，需逐轴请求 |
| 轴状态 | `/axis_state` · `AxisState`、`/multi_axis_state` · `MultiAxisState` topics | 位置为 mm／脉冲，含 `moving/fault/status`；组合状态含 `x/y/z/force_locked/active_action` |
| 标准力 | `/force_data` · `ForceData` topic | `timestamp_ns`、`fx/fy/fz`（N）、`mx/my/mz`（N·m）；当前控制器压缩力为 `max(0,-fz)`，原六维值仍须保存 |
| 清零／解锁 | `/force_sensor/tare` · `std_srvs/Trigger`、`/clear_force_lock` · `ClearForceLock` services | 清零会改变力基准；解锁请求 `axis`，支持 `ALL`，不是停止接口，不自动调用 |
| 双转台 | `/turntable_move` · `TurntableMove` action；`/turntable_query`、`/turntable_stop` services；`/turntable_state` topic | `table="1"/"2"`；运动用 `target_deg/use_degrees=true` 或 `target_pulses/use_degrees=false`，另有 `timeout_s`；轴系及TCP外参仍待标定 |
| 台架相机采集 | `/z_camera_repeat_press` · `ZCameraRepeatPress` action | 结果含 `success/completed_count/failed_count/run_id/output_dir/status`；字段语义随 `mode` 变化，见下文 |

接入时保留以下区别：

- `ZCameraRepeatPress.mode=6/7` 中，`target_force_n` 实为压入深度 mm，`force_step_n` 实为步长 mm；
  模式8的 `force_step_n` 则是力增量 N。模式6用 ViTai，模式7另用 BC V3 marker，不能仅凭字段名或模态名混用数据。
- `/axis_stop` 不会取消仍在执行的 action；停止流程还需取消对应 goal 并核对终态，停止成功不代表已卸载。
- 轴／转台状态没有时间戳；`MultiAxisState` 使用各轴最后缓存位置，不是同刻三轴测量。
  `ForceData.timestamp_ns` 是节点解析报文时的 ROS 时钟，不是传感器硬件时钟。
  采集端另记接收单调时钟、查询起止时间和数据新鲜度；未校时前不与 Revo3／相机时间直接相减。
- 接口没有独立的“稳定完成”消息或 `command_id/episode_id` 字段；使用 action UUID 关联本地事件，
  保存反馈 `phase` 和终态。接受 goal 不代表完成，稳定窗口需按力／位姿记录复核。
- 台架 mm／deg 转仿真 m／rad 只解决单位；Tag世界系、轴方向、TCP和转台外参确认后才能下发坐标。

源码入口：`py_force_control/py_force_control/nodes/{multi_axis_control_node,multi_axis_camera_control_node,force_sensor_node,turntable_control_node}.py`。
三轴主控只运行一个；带相机的入口为 `multi_axis_camera.launch.py`，双转台使用独立的 `turntable_control.launch.py`。
启动节点会连接设备；台架相机采集与下述 Revo3 相机脚本不能同时占用同一相机。

`ros_ws` 中另有 `py_force_calibration/py_force_calibration/ros_control.py` 可参考 ROS 客户端接法。
其 Revo3 后端使用旧 SDK 1.5.1 的 `revo3_get_touch_summary` 等接口及42区域汇总；
本仓库使用 SDK 2.1.0 的 `hand.touch.snapshot()` 和模块点阵，不能直接替换。
当前更新仅补齐台架接口约定；联合记录、外参和同步验收仍待实现。

## Revo3 读取与离线推理

- `probe_local.sh`：选择已有SDK解释器，转入只读探测器；支持 `REVO3_PYTHON`。
- `probe_revo3_readonly.py`：有限次数读取关节／阵列并逐路短采相机，输出到指定的新目录。
- `probe_revo3_trimodal.py`：对指定采集目录离线推理；不打开硬件，不代表物理标定。

命令见[总入口](../README.md)，设备身份与历史读取结果见[机械说明第7节](../docs/机械标定台仿真搭建_README.md#7-revo3-sdk与真机只读采集2026-10-03)。

## 三模态空载与后续加载采集（2026-10-05）

`collect_real_trimodal.py`协调现有相机只读worker和离线SDK推理，不打开手部串口、不控制运动。
SDK解释器：`/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python`。
以下实际命令会打开相机并写新目录；`--help`仅解析参数。

```bash
# 用户确认当前空载后：五路各16帧，首帧为参考，其余15帧作为样本。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/hardware/collect_real_trimodal.py --phase baseline --confirm-unloaded \
  --frames 16 --output calibration/captures/NEW_BASELINE

# 后续加载（需当次授权及实际加载状态）：明确选择SN，复用已完成的空载会话。
# CAMERA_SN是实际相机序列号；不根据编号猜测手指。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/hardware/collect_real_trimodal.py --phase loaded \
  --reference-session calibration/captures/NEW_BASELINE --serial CAMERA_SN \
  --frames 16 --output calibration/captures/NEW_LOADED
```

可用`--finger-map confirmed_map.json`提供已确认的`{"相机SN":"index"}`映射；允许
`little/ring/middle/index/thumb`，未确认保持null。加载前检查参考会话状态、SN、原始哈希及已知手指冲突。
相机逐路采集，三模态源自同一RGB；不表示多指同步或与机械台／力传感器同步。

输出：
- `session.json`：原始采集契约、会话UUID、参考会话、手指、阶段、逐路schema与SHA256。
- `raw/<SN>/frames.npz`：原RGB与主机读取时间；`last_frame.png`为预览。
- `derived/<SN>/reference.npz`和`reference_rgb.png`：实际使用的空载参考。
- `derived/<SN>/trimodal.npz`：SDK原始输出、warp、depth、origin/current/offset、源帧索引。
- `derived/<SN>/samples.npz`：CHW RGB、CHW warped RGB、`depth_sdk` (N,1,H,W)、
  `marker_2d` (N,400,5)=[x0,y0,dx,dy,finite]、`marker_finite`、独立source_episode_id、
  episode_step、时间和来源帧号。finite只表示有限数值，不代表跟踪置信度。

这是**供后续标定转换的原生三模态数据**，不是可直接喂现有训练器的complete manifest。
`training_ready=false`、`pair_valid=false`；缺少绝对深度尺度时不生成`depth_m`，
不猜测目标力／接触力，不将400光流点截成100点，不拉伸180×320到240×320。
现有训练加载器要求`rgb/depth_m/marker_2d/marker_valid`及回合/接触/目标力等元数据；
两塔按`episode_id/episode_step`配对。这里的source_episode_id仅作源数据标识，
必须在几何、单位和对应仿真状态核实后显式生成两域的规范配对键。
仅空载样本不能构成有辨别力的CLIP对齐训练集。

现有`probe_revo3_trimodal.py`新增`--reference-capture RAW_DIRECTORY`：
按同SN读取旧raw的第0帧，处理当前所有帧；不传时保留第0帧参考、第1帧起推理的旧行为。
此低层离线工具只检查SN/格式/尺寸；完整空载会话与哈希检查由新采集入口负责。
