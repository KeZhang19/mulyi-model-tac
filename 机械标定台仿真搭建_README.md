# 机械标定台仿真搭建

## 2026-10-03：当前资产切换为任务视触觉手

当前入口为[视触觉手＋V4支架＋球压头XYZ](assets/calibration_bench_visuotactile_urdf/README.md)，
[URDF](assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf)、
[Isaac截图](outputs/calibration_bench/visuotactile_v1/isaac_scene.png)、
[场景](outputs/calibration_bench/visuotactile_v1/scene.usda)。默认标定配置已更新。
使用 `assets/revo21_right_touch` 原手，保留21关节、5相机frame、橡胶面、压力描述及碰撞；
按任务既有rubber→DIP变换对齐，压头位置与轨迹重新生成。原支架、原资产与旧数据保留。
8/8短回合／99项数据检查、URDF／FK与Isaac回放通过；拇指零位与支架抽样穿入约2.383mm，
机械配合仍待复核，未改变安装固定结构或关闭碰撞。下文旧Revo3手路径及数据为历史版本。

## 2026-10-03：可运行的标定仿真入口

已新增 [标定仿真系统](scripts/calibration_bench/README.md)，入口为
`scripts/calibration_bench/run.py`。使用下文原始 V4 手／支架与三种压头，默认球压头；
按用户要求，场景现只显示压头／法兰和手／V4支架装配；已移除虚拟龙门架、导轨、台面及滑块。
XYZ压头轨迹保留，暂不展示尚未确认的机械台本体。
仿真采集为受控运动的几何／触觉模型，场景在 Isaac Sim 回放，不修改支架和螺钉结构。
真机 SDK 已另行完成只读连接与短采样，见[第7节](#7-revo3-sdk与真机只读采集2026-10-03)；尚未与仿真联动或对齐。

当前任务反查确认，已有 encoder 三模态是 **RGB＋Depth＋Marker**，压力单独保留。
HydroShear 是核心参考与 marker 模型来源，仓库公式稿不是完整目标论文；
本系统同时服务曲面深度、marker、压力及后续三模态研究，不把所有任务认定为单篇论文复现。

- [采集字段、协议、论文依据与注意事项](scripts/calibration_bench/README.md)
- [当前精简版 Isaac 场景截图](outputs/calibration_bench/head_fixture_only/isaac_scene.png)
- [当前可回放场景](outputs/calibration_bench/head_fixture_only/scene.usda)
- [含压头XYZ运动的URDF](assets/calibration_bench_urdf/urdf/revo3_calibration_scene_xyz.urdf)；
  [坐标系调整与轨迹说明](assets/calibration_bench_urdf/README.md)
- [数据验证报告](outputs/calibration_bench/validated_v1/verification.json)

已完成球压头8/8短回合、1,574帧、99项一致性检查；三角／锥头各1/1短回合、各115帧。
原生场景54个网格拓扑可读，动画位姿与数据最大元素误差3.71e-10；这是导入／回放检查，
不是动力学、真实触觉精度或全行程碰撞验收。压力和 RGB 仍为未标定模型，真机对齐后置。
输出在 Git 忽略的本地 `outputs/` 中，跨机器需单独携带。下文保留原始资产测量和未处理的机械需求。

## 1. 当前范围与状态

本文记录机械标定场景中已经拿到的 **Revo3 右手＋标定下支架 V4**，以及三种压头和连接法兰的
资产、组装入口与坐标关系。当前工作区仍没有机械标定台本体或运动轴的可用资产，因此这里给出的
压头 URDF 是独立工具装配，不包含标定台运动轴或场景级安装位姿。

当前工作区根目录为：

```text
/home/liuxinyu/workspace/mulyi-model-tac
```

下文未特别说明的路径均相对于该目录。当前可用的支架包为：

```text
assets/revo3_breadboard_mount/generated_calibration_v4/
```

注意：

- 这是当前唯一保留的支架版本；V1/V2/V3 和 V4 生成脚本已经不在当前仓库中，不能在仓库内重新生成 V4。
- 该目录目前在本地工作区中存在，但 `git status` 显示它尚未被 Git 跟踪；换分支、重新克隆或换机器前需要单独确认资产是否已带上。
- 原资产交付时只有 URDF；本轮新增的受控运动场景和采集入口见本文顶部，完整动力学台架仍待验证。
- 历史文档中提到过 `calibration_bench_sim_package/`，但该目录当前不存在，不能把历史路径当成当前可用资产。

## 2. 当前资产路径

### 2.1 手和支架的完整装配入口

仿真中需要同时加载右手和支架时，优先使用下面这一份组合 URDF：

```text
assets/revo3_breadboard_mount/generated_calibration_v4/urdf/
revo3_right_calibration_lower_mount_v4.urdf
```

完整单行路径：

```text
assets/revo3_breadboard_mount/generated_calibration_v4/urdf/revo3_right_calibration_lower_mount_v4.urdf
```

该 URDF 已经包含：

- Revo3 右手运动链；
- 标定下支架 `lower_mount`；
- 支架到右手的固定装配关系；
- 球槽底点、标定平面和 AprilTag 槽底三个辅助 frame。

装配拓扑为 61 个 link、60 个 joint，其中 21 个为可动关节。不要再在场景中重复加载一份独立支架，
否则会出现重叠几何。

### 2.2 支架独立入口

只加载支架、不加载手时使用：

```text
assets/revo3_breadboard_mount/generated_calibration_v4/urdf/lower_calibration_v4.urdf
```

支架的几何文件为：

| 用途 | 当前路径 |
| --- | --- |
| URDF 中的视觉和碰撞网格 | `assets/revo3_breadboard_mount/generated_calibration_v4/meshes/mount/lower_calibration_v4_print.stl` |
| 3D 打印文件 | `assets/revo3_breadboard_mount/generated_calibration_v4/meshes/mount/lower_calibration_v4_print.3mf` |
| STEP CAD | `assets/revo3_breadboard_mount/generated_calibration_v4/cad/lower_calibration_v4.step` |
| BREP CAD | `assets/revo3_breadboard_mount/generated_calibration_v4/cad/lower_calibration_v4.brep` |

支架 STL 的源单位是毫米，URDF 中通过 `scale="0.001 0.001 0.001"` 转成米。

### 2.3 Revo3 右手资产

当前 V4 包里没有单独的右手 URDF；右手 URDF 内容已经嵌入完整装配入口中。对应网格目录为：

| 用途 | 当前路径 |
| --- | --- |
| 右手视觉网格 | `assets/revo3_breadboard_mount/generated_calibration_v4/meshes/hands/visual/right/` |
| 右手碰撞网格 | `assets/revo3_breadboard_mount/generated_calibration_v4/meshes/hands/collision/right/` |

手部 mesh 在组合 URDF 中没有额外的 `scale`，导入器按 URDF 默认单位米解释。不要把历史记录中的外部
`brainco-description` 路径当成当前运行依赖；当前组合包引用的是上述包内相对路径。

### 2.4 预览、设计参数与验证记录

| 内容 | 当前路径 |
| --- | --- |
| 手＋支架装配预览 | `assets/revo3_breadboard_mount/generated_calibration_v4/preview.png` |
| 支架预览 | `assets/revo3_breadboard_mount/generated_calibration_v4/fixture_preview/preview.png` |
| 支架设计参数 | `assets/revo3_breadboard_mount/generated_calibration_v4/design.json` |
| 装配、URDF 和 mesh 验证记录 | `assets/revo3_breadboard_mount/generated_calibration_v4/validation.json` |
| CAD 验证记录 | `assets/revo3_breadboard_mount/generated_calibration_v4/cad_validation.json` |

URDF 使用 `../meshes/...` 相对引用。移动资产时应整体保留
`generated_calibration_v4/urdf/` 与 `generated_calibration_v4/meshes/` 的相对目录结构。

### 2.5 三种压头与连接法兰的装配入口

压头和法兰源资产位于：

```text
assets/head/
```

仿真中按压头形状选择以下一份 URDF，不要再额外加载同一法兰或同一压头 STL：

| 压头 | 完整装配 URDF |
| --- | --- |
| 三角压头＋连接法兰 | `assets/head/triangle_press_head_with_flange.urdf` |
| 球压头＋连接法兰 | `assets/head/spherical_press_head_with_flange.urdf` |
| 锥压头＋连接法兰 | `assets/head/conical_press_head_with_flange.urdf` |

每份 URDF 都包含 `connection_flange`、`press_head` 和 `press_head_tip` 三个 link，以及两个 fixed
joint。`connection_flange` 是根 link；`press_head_tip` 是无几何的工作面参考 frame。视觉和碰撞均
直接引用原始 STL，STL 源单位为毫米，并通过 `scale="0.001 0.001 0.001"` 转为米。三份 URDF
与 4 份 STL 应保持在同一目录；只复制 URDF 会造成相对 mesh 引用缺失。

## 3. 当前坐标系约定

当前完整装配使用支架自身坐标系，根 link 为：

```text
lower_mount
```

URDF 没有规定 `lower_mount` 在仿真世界中的位姿。搭建场景时，应把标定台世界坐标到
`lower_mount` 的变换作为场景级配置单独设置；在该变换确定前，不要修改现有手和支架的内部固定关系。

URDF 中的长度单位为米、角度单位为弧度。以下变换均相对于父 link：

| joint / frame | 父 link → 子 link | `xyz`（m） | `rpy`（rad） | 含义 |
| --- | --- | --- | --- | --- |
| `lower_mount_to_right_hand` | `lower_mount` → `right_hand_base_link` | `(0.060, -0.0002, 0.040)` | `(-π/2, -π/2, 0)` | 右手在支架上的固定装配位姿 |
| `calibration_plane_fixed` | `lower_mount` → `calibration_plane` | `(0.170, 0.1625, 0.036)` | `(0, 0, 0)` | 支架标定平台参考 frame |
| `calibration_sphere_bottom_fixed` | `lower_mount` → `calibration_sphere_bottom` | `(0.170, 0.125, 0.033)` | `(0, 0, 0)` | 球槽局部最低点 |
| `apriltag_pocket_floor_fixed` | `lower_mount` → `apriltag_pocket_floor` | `(0.170, 0.200, 0.0355)` | `(0, 0, 0)` | AprilTag 槽底，不是贴纸表面 |

其中 `calibration_sphere_bottom` 只是球槽局部基准，不是整个支架的最低点；
`apriltag_pocket_floor` 也不包含未知的贴纸厚度。

## 4. 支架现有几何尺寸

本节只记录支架和支架上的标定结构，**不是压头尺寸**。

| 项目 | 当前值 |
| --- | --- |
| 支架 mesh 边界 | X `[0, 220]` mm，Y `[-3, 249]` mm，Z `[0, 45]` mm |
| 支架总体边界尺寸 | 220 × 252 × 45 mm |
| 标定平台高度 | Z = 36 mm |
| 球面槽 | 口径 40 mm，深 3 mm，球半径约 68.1667 mm |
| 球面槽底点 | `(170, 125, 33)` mm，位于 `lower_mount` 坐标系 |
| AprilTag 方槽 | 55 × 55 mm，深 0.5 mm |
| AprilTag 槽底中心 | `(170, 200, 35.5)` mm，位于 `lower_mount` 坐标系 |

AprilTag 的 family、ID、实际打印外形和贴纸表面高度目前没有确定。

## 5. 标定台与压头资产

### 5.1 源文件和 mesh 测量结果

`assets/head/` 中每个零件同时提供 SolidWorks `SLDPRT` 和二进制 STL。以下尺寸由 STL 几何测得，
用于仿真装配和量纲核查，不代替制造图纸、公差或实物测量：

| 零件 | CAD / mesh | STL 总体边界尺寸（mm） | mesh 得到的工作端特征 |
| --- | --- | --- | --- |
| 连接法兰 | `连接法兰.SLDPRT` / `连接法兰.STL` | 63.000 × 62.943 × 37.700 | 下端为压头安装孔，法兰自身不作为接触工作面 |
| 三角压头 | `三角.SLDPRT` / `三角.STL` | 5.991 × 12.598 × 6.000 | 端面为边长约 2.2 mm 的三角形，端面面积约 2.096 mm² |
| 球压头 | `球压头.SLDPRT` / `球压头.STL` | 7.079 × 13.418 × 7.059 | 球端外半径约 3.54 mm |
| 锥压头 | `锥.SLDPRT` / `锥.STL` | 9.070 × 14.567 × 9.064 | 锥尖保留约 0.30 mm 直径的平端，不是数学尖点 |

三种压头具有相同的名义直径 6 mm、长度 8 mm 尾柄。法兰下端 mesh 中的对应孔名义直径约
6.4 mm、深 8 mm，因此装配后的径向间隙约 0.2 mm、直径间隙约 0.4 mm。压头与法兰的横向销孔
中心均位于 4 mm 插入深度处；URDF 中已把压头绕装配轴旋转，使两侧销孔同轴。上述接口尺寸同样
来自 mesh，尚未由图纸或实物公差确认。

### 5.2 URDF 装配坐标

三份 URDF 使用相同坐标约定：根 frame 位于法兰中心轴和法兰盘安装面交点，即
`连接法兰.STL` 原始坐标 `(31.5, 31.5, 30.0)` mm。`+Z` 从压头指向机械本体，压入方向为 `-Z`。
原始压头 STL 的轴向是 `+Y`；URDF 的 mesh 变换使用
`rpy=(π/2, 0, π/2)`，把尾柄轴向转到 `+Z`，同时把压头销孔转到法兰销孔方向。

| URDF | `connection_flange` → `press_head`（m） | `press_head` → `press_head_tip`（m） | 根 frame 中 tip 的 Z（m） |
| --- | --- | --- | --- |
| `triangle_press_head_with_flange.urdf` | `(0, 0, -0.030)` | `(0, 0, -0.004598076912)` | `-0.034598076912` |
| `spherical_press_head_with_flange.urdf` | `(0, 0, -0.030)` | `(0, 0, -0.005417815208)` | `-0.035417815208` |
| `conical_press_head_with_flange.urdf` | `(0, 0, -0.030)` | `(0, 0, -0.006567255020)` | `-0.036567255020` |

`press_head` frame 位于法兰安装孔入口和压头尾柄肩部中心；尾柄沿 `+Z` 完整插入 8 mm，工作端沿
`-Z` 伸出。`press_head_tip` 位于三角端面中心、球面轴向最低点或锥形平端中心，且保持与根 frame
相同的轴方向。法兰盘 `Z=30 mm` 被选为仿真安装基准是基于当前 mesh 结构的坐标约定，尚未确认它
是否等于机械图纸或设备控制软件中的法兰零位。

### 5.3 动力学和场景级待确认项

当前工作区仍未提供机械标定台本体或运动轴资产。场景搭建时还需要明确至少两个场景级变换：

```text
world / calibration_bench   -> lower_mount
press_axis / press_carriage -> connection_flange
```

三份 URDF 没有填写质量、质心和惯量，因为零件材料和实物质量尚未提供；导入动态场景时需要按
实物补齐，或明确使用导入器的密度/惯量生成设置。碰撞几何目前与闭合原始 STL 相同，未做凸分解；
若压头作为动态刚体使用，需要在 Isaac Sim 导入后核对实际碰撞近似和接触面精度。最终接触行程
应由 `press_head_tip` 相对于目标触觉面的几何关系计算，不能直接把 `calibration_sphere_bottom`、
支架最低点或法兰安装面当作压头接触零位。

## 6. 当前验证边界

2026-10-03 本次复核支架两份 URDF 均通过 XML 解析；两份 URDF 共 143 个 mesh 引用，当前缺失数
为 0。三种压头和法兰的 4 份 STL 均为闭合、法向一致的单实体网格；三份新增装配 URDF 均通过 XML
和拓扑解析，mesh 引用无缺失。装配几何核查确认尾柄插入 8 mm、销孔中心对齐，并确认三种
`press_head_tip` 与对应 mesh 工作端重合。已有记录还包含支架 CAD/STL 的离线几何检查。
ROS Humble `check_urdf` 本次因本机缺少 `liburdfdom_model.so.3.0` 未执行。

源 URDF 的动力学导入、质量与惯量标定、全行程碰撞检查、压头接触参数、实物装配或真机标定
尚未验收。本轮新增系统只完成从这些资产生成的 USD 原生查看／回放与采集模型一致性检查，
不能将此升级为动力学或真实标定验收。

## 7. Revo3 SDK与真机只读采集（2026-10-03）

### 7.1 SDK副本和运行入口

已将 `/home/liuxinyu/workspace/brainco-revo3-sdk` 复制到
[`calibration/brainco-revo3-sdk/`](calibration/brainco-revo3-sdk/README.md)。
保留原目录；不复制 `.git`、虚拟环境、缓存、日志和编辑器配置。
原环境中的 **bc-revo3-sdk 2.1.0**、**pyvitaisdk 1.0.17** 及其随包模型、原生库、接口存根和许可证
另存于副本的 `runtime/python/`，因此复制的不只是示例代码。
共676个文件、276,582,053字节，来源版本 `52e3fcb35bcbe91e17ab66bd8a0cbccc19d8e69f`；
逐文件SHA256见[复制清单](calibration/sdk_copy_manifest.json)。
源目录 `artifacts/vision_touch_capture/` 是历史图片，不能作为本次连接证据。

新增[只读探测脚本](calibration/probe_revo3_readonly.py)，从仓库根目录运行：

```bash
/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/probe_revo3_readonly.py \
  --output calibration/captures/my_new_readonly_probe
```

输出目录必须不存在。脚本优先导入复制的 `runtime/python/`，其余依赖使用上述已存在环境：
Python3.11.15、NumPy1.26.4、OpenCV4.11.0.86。本次未安装包或改动系统驱动；
这个源环境仍是运行依赖，SDK副本不是可直接迁移的完整Python环境。
`--help` 不打开设备。默认只读3次State/Touch、枚举ViTai，并依次读取5个视触觉相机，各预热5帧、保存8帧。
手部／枚举／每路相机分别有40／25／20秒子进程超时；异常和失败日志保留，不覆盖旧记录。
相机枚举只匹配 `ViTai-GFBC` 的 `video-index0`，不采集笔记本摄像头或metadata节点。
流程没有手部运动、使能修改、清零、归零或固件写入；仅相机打开流并请求视频格式。

### 7.2 本次实际读取结果

最终短测时间：2026-10-03 16:05:42–16:05:47（北京时间）。
证据在 [`calibration/captures/revo3_readonly_20261003_final/`](calibration/captures/revo3_readonly_20261003_final/report.json)，
附[5路图像预览](calibration/captures/revo3_readonly_20261003_final/cameras_montage.png)和
[独立数据核查](calibration/captures/revo3_readonly_20261003_final/validation.json)。
`report.json` 的 `passed` 仅表示读取接口及图像数量检查通过；不表示全部模块启用、标定准确或健康验收。

| 项目 | 实测结果 |
| --- | --- |
| 设备 | UltraVisionTouch，右手；SN `BCUVR1205J2600002`，硬件 `0.0.1`，控制器固件 `9.9.9.ML` |
| 主链路 | Modbus RTU，`/dev/ttyUSB0`，5,000,000 bps，从站127（`0x7F`） |
| 稳定串口路径 | `/dev/serial/by-id/usb-BrainCo_USB__-__RS485_Converter-if00-port0` |
| 关节反馈 | 3次完整读取，21个位置／速度／电流／运行状态；逻辑关节布局随报告保存 |
| 触觉布局 | 指腹5模块＋掌部1模块；声明总点数301；不是仿真旧285-taxel布局 |
| 实际触觉数据 | 每次244点有效；拇指指腹57点被禁用，保留 `Disabled` 和 `points=null` |
| 模式 | `PointArray`；各阵列 `TouchValueMode.Force`，SDK输出单位mN；使能掩码 `0x0551` |
| 指尖相机 | 5/5路各8帧，共40帧；每帧640×480×3、uint8；PNG可读，NPZ存RGB |
| 健康读取 | error_code=0，faulted_motor_count=0；但 `SafetyState.Unknown`，电压回报约240.84 V，需核实协议／固件量纲，不能据此确认设备健康 |

主链路阵列顺序以 `module_id` 为准，不能把数组下标当物理模块ID：

| module_id | 部位 | layout声明点数 | 本次有效点数／状态 |
| --- | --- | ---: | --- |
| 2 | 拇指指腹 | 57 | 0，Disabled |
| 4 | 食指指腹 | 52 | 52，Valid |
| 6 | 中指指腹 | 52 | 52，Valid |
| 8 | 无名指指腹 | 52 | 52，Valid |
| 10 | 小指指腹 | 52 | 52，Valid |
| 0 | 掌部 | 36 | 36，Valid |

指尖图像是独立通道；相机SN与物理手指的对应关系**尚未确认**，不从SN前缀或video编号猜测。
以下设备号只适用于本次插接，应使用 `/dev/v4l/by-id/*<SN>-video-index0` 定位：

| 相机SN | 本次节点 | 接收结果 |
| --- | --- | --- |
| WTUVL2198X260001A | `/dev/video2` | 8/8帧 |
| WTUVL3194X26000EB | `/dev/video10` | 8/8帧 |
| WTUVL3195X26000EE | `/dev/video6` | 8/8帧 |
| WTUVL3197X2600106 | `/dev/video4` | 8/8帧 |
| WTUVL3197X260010B | `/dev/video8` | 8/8帧 |

请求MJPG、640×480、30 fps，驱动实际回报120 fps；8帧短窗口接收约58–62 fps。
这是逐路短测，未验证5路同时采集带宽或持续帧率，也没有硬件曝光时间戳和丢帧计数。

### 7.3 已确认接口与尚待标定接口

主链路依据副本的[Revo3 API](calibration/brainco-revo3-sdk/docs/api/REVO3_API.zh-CN.md)、
[Touch协议](calibration/brainco-revo3-sdk/docs/device/REVO3_TOUCH_API.md)及
[Python接口存根](calibration/brainco-revo3-sdk/runtime/python/bc_revo3_sdk/main_mod.pyi)：

| 接口 | 输出／用途 | 本次验证 |
| --- | --- | --- |
| `sdk.Manager().discover(...)`、`manager.connect(detected)` | 指定端口、Modbus、5M、从站127进行发现与连接 | 已执行 |
| `hand.device_info`、`hand.firmware_info`、`hand.joint_layout` | SN、左右手、型号、固件、逻辑关节布局 | 已读取；子模块SN／固件列表为空，不能补造 |
| `await hand.state.snapshot()` | `positions_deg/rad`、`velocities_rpm/rad_s`、`currents_ma/a`、`operating_states`、timestamp | 21关节×3次 |
| `hand.touch.layout`、`await hand.touch.enabled_mask()` | 模块ID、部位、布局ID、点数、信号及使能状态 | 已读取 |
| `await hand.touch.read_mode()`、`await hand.touch.value_mode(module_id)` | 判定points是ADC、mN，还是兼容汇总数据 | 已读取，PointArray／Force |
| `await hand.touch.snapshot()` | `sequence/timestamp/modules`；每模块 `sample_state/points` 等 | 已读取3次，保留Disabled模块 |
| `await hand.touch.snapshot(module_indices=[4])`、`await hand.touch.module_snapshot(4)` | 按**物理模块ID**选择食指指腹，降低读取量 | SDK接口存在；本次未另测选择读取 |
| `await hand.health.snapshot()` | 电流、电压、温度、故障和安全状态 | 已读取；异常解释待确认 |
| `await hand.close()`、`await manager.close()` | 释放串口 | 每次正常退出执行；外部超时会终止子进程并关闭其FD |

`hand.touch` **不提供指尖视触觉图像、深度或marker**。本次相机发现验证了
`pyvitaisdk.VTSDeviceFinder().get_devices()`（`name/SN/index`），原始图像通过
`cv2.VideoCapture(stable_by_id, cv2.CAP_V4L2).read()` 获取；OpenCV输出BGR，脚本转RGB后写入NPZ。

ViTai高层接口见副本
[VTSensor存根](calibration/brainco-revo3-sdk/runtime/python/pyvitaisdk/vts/core/vt_sensor.pyi)、
[GFSensorBase存根](calibration/brainco-revo3-sdk/runtime/python/pyvitaisdk/vts/core/base/gf_sensor_base.pyi)与
[VTSDataType定义](calibration/brainco-revo3-sdk/runtime/python/pyvitaisdk/vts/core/base/profile.pyi)。
以下是已核查的接口定义，**本次没有生成或验收这些推理结果**：

| 接口 | 后续用途及前提 |
| --- | --- |
| `VTSensor(config=..., model_path=..., force_model_path=...)` | 根据已确认传感器型号、对应模型创建运行对象；不能以通用GFBC名称代替具体型号确认 |
| `get_raw_frame()`、`get_warped_frame()`、`set_warp_params(...)` | 原始／透视变换图像；需要记录裁剪、变换参数与输出尺寸 |
| `set_background(...)`、`init_marker(...)`、`tracking(...)` | 确认无接触后建立图像和marker基准，再跟踪；本次没有擅自把当前姿态认定为空载 |
| `get_origin_markers()`、`get_markers()`、`get_markers_displacement()` | marker参考／当前位置／位移；像素顺序、ID稳定性、丢点掩码和物理尺度待核对 |
| `recon3d(frame)`、`get_depth_map()` | 模型估计深度；单位、零面和几何精度需用标定压头验证 |
| `collect_sensor_data(*data_types)` | `RAW_IMG/DEPTH_MAP/MARKER_*_VECTOR/FORCE6D_VECTOR`等；可请求不等于本机已验证 |
| `release()` | 释放ViTai高层对象资源 |

副本含GFBCI/GFBCT深度和力模型，但没有验证具体SN对应哪一模型。
模型深度和力是估计结果，不能充当压头位移或参考测力计真值；mN点力也不能直接当成Pa压力，
转换还需要taxel面积及曲面法向标定。

### 7.4 论文标定采集字段和联调约定

| 数据 | 最低保存内容 | 当前阶段 |
| --- | --- | --- |
| 指尖图像 | SN、原始RGB、分辨率、颜色顺序、相机格式、主机接收时间、采集序号、读取失败 | 已短采样；相机内参／曝光设置／曝光时间戳未标定 |
| 手部状态 | 手SN、关节布局、21关节位置／速度／电流、SDK与主机时间 | 已短采样；到URDF关节的符号／零位映射待核对 |
| 指腹／掌部阵列 | module_id、layout_id、value/read mode、sample_state、points、使能掩码 | 已短采样；禁用／故障不得补零后当有效标签 |
| marker | reference/current、位移、ID和有效掩码、坐标单位、基准图像、算法版本 | 尚未采集 |
| 深度 | 原始估计、零面、有效掩码、单位、模型与标定哈希 | 尚未采集 |
| 机械台／参考力 | XYZ命令与反馈、压头tip位姿、接触零点、参考力／力矩、阶段、方向、加载速率 | 台架SDK尚未提供／连接；仿真轨迹不是真机反馈 |
| 样本元数据 | episode/frame ID、工具几何、实际手指映射、材料状态、空载／加载／保持／卸载、重复次数、失败原因 | 后续与仿真schema建立显式转换 |

本次输出：`hand.json`保留完整SDK快照；各SN目录中的`frames.npz`含
`rgb`、`host_read_start_monotonic_ns`、`host_read_end_monotonic_ns`、`host_read_end_unix_ns`，
另有`last_frame.png`、`camera.json`和每个子进程日志。`vitai_finder.json`记录相机发现结果。
每路相机先后采样，序号仅是应用保存索引，不能用相同数组下标拼成同步训练样本。

同步时保留SDK的 `Timestamp.clock/sec/nsec`：本机返回 `ProcessMonotonic`，其起点不能直接与
Python `time.monotonic_ns()` 或其他SDK时钟相减。使用每次读取前后主机时间估计时间偏差与区间，
记录配对误差；后续联合采集须另做并发、延迟和丢帧测量。
先人工确认相机SN→手指和阵列ID→曲面映射，再做相机／marker／深度标定、压头XYZ与接触Z零点对齐。
机械台到法兰、支架到世界的变换单独保存，不因这次SDK读取去修改当前URDF。
仿真训练所用RGB＋Depth＋Marker和额外压力通道仍按各自单位保存，不能直接把301点布局或244点有效子集塞进旧285-taxel接口。

## 待解决的问题（原始需求保留）
支架高度问题已按 M6×20 规格处理：V5 从底面截短 10 mm，原 M6 孔保持，
无垫片名义伸出 6 mm、加 1 mm 垫片名义伸出 5 mm。交付见
`assets/revo3_breadboard_mount/generated_calibration_v5_m6x20/`；实际有效牙深仍待实装确认。
还需要做的方案设计
目前还需要搭建仿真和真机联调的数字孪生
1.根据论文和目前的仿真选择用什么平台来搭建仿真的数字孪生,isaaclab,gazebo或者其它可视化,目前压头默认用球压头
2.目前的手部sdk和标定的sdk如何连接在一起
3.连接在一起后的对齐(z坐标需要接触标定,定位孔x,y坐标有具体的点坐标)
目前手部可用的sdk:/home/liuxinyu/workspace/brainco-revo3-sdk里面可以获取视触觉传感器信息
