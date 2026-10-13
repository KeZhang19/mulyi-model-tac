# Tag 系可视化仿真采集框架

入口为 [`visual_collect.py`](../simulation/visual_collect.py)，默认配置为
[`visual_config.json`](../config/simulation/visual_config.json)。在 Isaac Sim 中显示当前
视触觉手、M6×16 支架和选定的压头／法兰，右侧面板用于定位、采集、暂停、停止保存及回放。
不导入 ROS 节点，不打开串口、相机 SDK 或测力设备。

## 来源和坐标

- [`坐标转换.yaml`](../config/坐标转换.yaml)：设备 XYZ 的单位为 mm，Tag 原点为
  `(−267.5, −14.5, −308.2)` mm；`T.x=−(M.y+14.5)`、`T.y=M.x+267.5`、
  `T.z=M.z+308.2`（公式双方均用 mm）。支座的 180° 已包含在轴向关系中，不再次旋转。
- [`机械说明`](机械标定台仿真搭建_README.md)：当前 Tag 槽底中心为唯一支架根，
  平台在 Tag Z=0.5 mm；采用当前 M6×16 派生网格。旧平台读数 −308.5605 mm不参与换算。
- `ros_ws/src/py_force_ws/py_force_control/config/params.yaml`：采用**配置文件中的软件行程**，
  X `[−340, 0]`、Y `[0, 170]`、Z `[−310, 0]` mm。不是 `axis_config.py` 的工厂默认
  X `[−350,0]`、Y `[0,188]`、Z `[−300,0]`。`multi_axis_control_node.py` 按参数构建 AxisConfig。
  本入口没有查询活动 ROS 节点，不能声称这些值就是当前在线节点参数或独立实测机械极限。
- 当前完整资产：
  [`revo3_calibration_scene_xyz.urdf`](../assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf)。
  直接加载其手／支架网格和关节位姿，不从旧 V4 源包重建支架；过滤原静态工具以免与活动工具重叠。
  球／三角／锥压头采用 `assets/head/` 对应源 URDF 的法兰到尖端变换。

设备显示 XYZ 暂对应**名义工作尖端**，实际 TCP、全外参、码纸厚度与接触零点仍未验收；
该假设显式保存在 `coordinate_contract.json`，数据不能直接用于真机下发。
Tag 原点本身位于设备 Y=−14.5，超出软件 Y 行程；支架上的参考点不必都是压头可达目标。
Tag 下的可达包络为 X `[−184.5,−14.5]`、Y `[−72.5,267.5]`、Z `[−1.8,308.2]` mm。
XYZ 坐标变换也适用于非可达的参考点，但运动请求单独检查行程。

双转台的限位参数只作来源登记：1号启用−10～90°、2号限位未启用。
未确认转轴、旋转中心和手／压头外参，禁止将其直接映射为虚拟 twist/roll。
相机区域 XY 范围是另一层历史采集工作区，只登记，不代替整轴行程。

## 启动和操作

从仓库根执行；使用已有解释器，不安装依赖。输出目录必须不存在，示例目录重复使用会被拒绝。

```bash
export CALIB_PY=/home/liuxinyu/miniconda3/envs/brainco/bin/python

# 坐标、配置来源、行程与回合规模；CPU，不初始化 Isaac、不写数据
"$CALIB_PY" calibration/simulation/visual_collect.py --plan-only

# 交互界面；打开本地 Isaac/GPU，写入新会话目录，直到用户关闭
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CALIB_PY" \
  calibration/simulation/visual_collect.py \
  --output calibration/outputs/simulation/my_visual_session

# 有限 CPU 采集六协议，无 Isaac/GPU 或硬件
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CALIB_PY" \
  calibration/simulation/visual_collect.py --offline --episodes 6 \
  --output calibration/outputs/simulation/my_visual_cpu
"$CALIB_PY" calibration/simulation/verify_visual.py \
  calibration/outputs/simulation/my_visual_cpu

# 有限原生场景、面板、位姿及 RTX 渲染核对；不是动力学验证
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CALIB_PY" \
  calibration/simulation/visual_collect.py --headless --autostart \
  --episodes 1 --viewer-frames 160 --capture \
  --output calibration/outputs/simulation/my_visual_native

# CPU 回归：坐标/边界/NaN拒绝、暂停、回放隔离、故障帧与中断保存
"$CALIB_PY" calibration/simulation/check_visual.py
```

`--coordinates`、`--ros-workspace` 可切换输入，仍只读取本地文件。
`--head spherical|triangle|conical`、`--finger index|middle|ring|little|thumb`、
`--segment DIP|MCP|PIP` 可选择工具／位置；只有当前记录的食指 DIP 配置经过本轮采集检查，
其他组合的朝向、可达性和非目标净空需分别验证。

面板操作：

1. 填入设备 XYZ，按 **Move to XYZ**；同 XY 时只走 Z，改变 XY 时先升至几何上方，
   串行移动 X、Y，再下降。蓝色线框表示软件行程包络，三色轴表示 Tag，黄色点表示当前传感面。
2. **Sensor clearance / retract** 返回当前选定传感位置的 3 mm名义净空。
3. 选 normal、hold、shear_x/y 或 slip_x/y 与压入深度，按 **Start episode**。
   保留 baseline→approach→load→hold→扫描（若有）→unload→retract→release 全部阶段。
   曲面跟随造成的扫描／Z 增量也串行处理，与原离线采集器的同时 XYZ 路径并不等价。
4. **Pause / resume** 冻结仿真时间；**Stop + save interrupted** 保留部分数据并标为失败。
   超限或非目标抽样穿入同样保留触发帧并停止。停止不会自行启动真机或额外运动。
5. 已保存回合可拖动回放滑条或 **Replay peak load**；回放不新增采样。

RGB、深度、压力及 marker 同步显示；深度和压力采用固定色标，模型合力曲线显示最近1800个样本。
交互手选实验的 `split=unassigned`；离线计划沿用组合分组，不把相邻帧随机拆分。

## 数据与模块

| 模块 | 职责 |
| --- | --- |
| `stage_config.py` | YAML 锚点／轴向验证、mm↔Tag m、软件行程、仅相关参数的来源快照 |
| `visual_session.py` | 串行运动、逐帧采样、限位／深度／力／抽样几何检查、保存与回放 |
| `visual_scene.py` | USDA 场景、软件包络／Tag 轴／传感面标记、通道图与力曲线 |
| `visual_ui.py` | Isaac 原生操作面板和实时读数 |
| `visual_collect.py` | 配置与有限 CPU／原生交互入口 |
| `verify_visual.py` / `check_visual.py` | 保存数据契约核验／状态机回归 |

每个会话保存配置、坐标／限位契约及源文件 SHA256、`layout.npz`、`manifest.json`；
每回合保存计划、报告及 `episode_XXXXX.npz`。自动完成、失败、中断分开标记，文件不会覆盖旧会话。

关键数组为 `machine_position_mm`、`command_machine_position_mm`、`T_tag_tip`、`T_tag_flange`、
`time_s`、`phase`、`rgb`、`depth_m`、`marker`、`pressure_pa`、`taxel_force_n`、
`total_normal_load_n`、`force_on_sensor_w_n`、接触／有效／饱和标记。
兼容模型内部名称 `T_world_flange`、`T_world_tip` 在**本 schema 中 world=Tag**；
不能与旧 `mechanical_calibration_v1` 的 world=lower_mount 混用。
`time_s` 是固定步长仿真时钟，不是 ROS 时间／墙钟／设备时间戳；UI 刷新慢时仿真也变慢。

CPU模式额外输出动画 `scene.usda`、`channels.png`、`force_trace.png`；原生 `--capture` 输出
`isaac_scene.png` 及通道图；`isaac_verification.json` 核对网格、活动压头变换和面板更新。
`verify_visual.py` 拒绝覆盖已有验证报告。

## 验证边界

这是几何射线＋现有 HydroShear marker＋临时材料／光照模型的受控运动采集。
没有 PhysX/FEM 接触求解、独立六维力真值或经过标定的相机投影／光学响应；
压力×面积与合力的核对只是模型内部一致性。没有凭空增加机械台实体、惯量或旋转轴。
约每15帧检查工具采样点是否包含于非目标闭合网格，开口网格保留 uncertified 标记；
未完成连续碰撞／全行程净空或真实触觉精度验收，不能将“完成回合”称为物理标定成功。

本 schema 需先转成下述 CLIP 输入分片。正式跨域配对仍需完成真实接口中的物理外参、
真实相机 ROI／投影、时间配对、确认空载参考及独立验证。
旧 `run.py` 默认资产、配置和 schema 保持；新增 Tag/M6×16 资产选择是显式选项。

## 完整原始会话转 CLIP 仿真输入

从仓库根目录执行；`--source` 指向含 `manifest.json` 和 `episode_*.npz` 的原始会话，
输出目录必须全新且位于源会话之外。只读取原数据／checkpoint，不启动 Isaac 或硬件。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/tactile_representation/convert_visual_session.py \
  --source calibration/outputs/simulation/MY_SESSION \
  --output calibration/datasets/MY_CLIP_SIM \
  --checkpoint runs/revo3_cross_modal_restoration_v1/best.pt
```

`--checkpoint` 读取原编码器 normalization／尺寸，不加载网络运行；省略时使用
240×320 图像、100 个 Marker、3 mm 深度归一化尺度。NPZ 保存原始单位，不预先归一化。
RGB 从 HWC 双线性缩放为 CHW uint8，深度保持米制 float32，Marker 的起点及位移
按图像横纵轴比例缩放；不使用着色深度 PNG 或放大位移箭头图作为训练值。

输出 `sim/manifest.json` 的 schema 为 `tactile_alignment_domain_npz_v1`，
`sim/shards/*.npz` 包含 `rgb/depth_m/marker_2d/marker_valid/pair_id/pair_valid/training_ready`。
默认每分片最多128帧；转换全部已保存帧，保留完整阶段、失败／中断回合及零帧失败记录。
`provenance/` 保存源报告、原阶段字符串、坐标、模型力、接触记录；`conversion.json`
记录源SHA、帧数／阶段计数、变换与局限，`verification.json` 是格式验证结果。

加载器 `training/alignment_domain.py:AlignmentDomain` 校验各分片后按需读取，最多缓存两个分片。
原 `AlignmentData` 同时支持旧单分片和新多分片；旧数据指纹与恢复契约保持。
本输出只有仿真域，`pair_id` 只是源会话内回合／帧索引；不会生成真机域、`pairs.json`
或跨域划分。`pair_valid/training_ready` 保持false；完成格式转换不等于完成物理配对。
后续只有在对应真机记录及配对验收通过后，才组装完整 CLIP 配对数据集。
