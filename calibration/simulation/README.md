# 机械标定仿真系统 v1

2026-10-03更新：默认配置已切换至 `hand_asset=revo21_touch_v4`，采用任务引用的
`assets/revo21_right_touch` 视触觉手。当前[完整URDF](../assets/calibration_bench_visuotactile_urdf/README.md)、
[场景](../outputs/simulation/visuotactile_v1/scene.usda)、
[截图](../outputs/simulation/visuotactile_v1/isaac_scene.png)。
`build_touch_fixture.py` 先生成独立手／V4支架包；`export_urdf.py` 默认读新场景，
`verify_touch_asset.py` 核查几何、任务坐标及XYZ，并输出任务参考marker映射。
五路相机frame保留，但实际采集仍为虚拟正交模型。拇指零位与支架穿入抽样约2.383mm，机械配合待复核。
旧配置缺少 `hand_asset` 时继续使用旧手，保证历史回放不会被静默换模型。
下文 `head_fixture_only`／`validated_v1` 为旧手历史证据；新手验证不能引用它们代替。

本目录提供**可运行的受控运动触觉标定仿真**：V4 支架＋Revo3 右手＋真实压头网格，
CPU 采集几何深度、压力模型、HydroShear marker 和诊断 RGB；USD 场景在 Isaac Sim 中回放。
默认球压头，原始资产只读。按用户要求，场景只显示压头／法兰与 Revo3／V4支架装配，
不再生成虚拟龙门架、导轨、台面、Z轴柱和滑块；XYZ轨迹仍直接作用于压头。

旧手精简场景：[scene.usda](../outputs/simulation/head_fixture_only/scene.usda)，
[截图](../outputs/simulation/head_fixture_only/isaac_scene.png)。
该目录复用 `validated_v1` 的首回合数据，仅重新导出场景，不是重新采集的数据集。

另有[自包含 URDF 资产包](../assets/calibration_bench_urdf/README.md)：保留当前手／支架／球压头，
新增无外观XYZ移动关节及121帧CSV；`world_to_press_stage`集中定义后续可调整的压头坐标系。
导出脚本为 `export_urdf.py`，不会修改USD场景或原始资产。

**它是规定压头轨迹的接触／触觉模型台架，不是已经验收的 PhysX/FEM 动力学标定台。**
手部在 URDF 指定姿态固定，工具位姿精确跟随命令；不宣称电机跟踪误差、负载动力学或真机精度。
不连接手 SDK／运动台 SDK，不改支架螺钉结构。真机坐标和时钟对齐按本轮要求后置。

## 1. 论文主线与实际任务的对应

不能把整个项目认定为某一篇论文的复现。当前仓库证据为：

| 证据 | 对本系统的要求 |
| --- | --- |
| [`hydroshear_uv_formula_corrected.tex`](../../hydroshear_uv_formula_corrected.tex) | 原 HydroShear 与曲面 UV 度量实现的公式对照；不是完整投稿稿件，也不是完整测地线模型 |
| [`integrate/curved_hydroshear_adapter.py`](../../integrate/curved_hydroshear_adapter.py) | 本系统直接调用现有曲面 HydroShear；跨帧保持物体材料采样点身份，每回合 reset |
| [`policy.py`](../../source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/policy.py) | 已有 encoder 三模态为 **RGB、Depth、Marker**，不是压力、深度、marker；压力单独保留 |
| [`collect_index_cross_modal.py`](../collection/collect_index_cross_modal.py) | 现有采集契约及预处理需单独核对；本系统使用新 schema，不能直接混入原训练缓存 |
| Repose-Cube、D-Peg、灯泡旋出／桌面装配 | 是本项目不同下游任务，标定数据不等于这些任务的成功率，检查点也不能互换 |

参考文献及采用范围（2026-10-03 查阅原文）：

- [HydroShear, arXiv:2603.00446，§III-C/IV-A](https://arxiv.org/html/2603.00446v1)：
  采用分离法向压入、切向运动和滑移的标定思路；参数按 λd→λs→K→μ 分阶段辨识。
  原论文使用 GelSight Mini 与机器人球压头；本项目 Revo3 曲面、约 3.54 mm 球端半径和相机不同，
  不能照搬其参数、像素换算、实验样本数或结果。
- [TacMap, arXiv:2602.21625](https://arxiv.org/html/2602.21625v1)：参考几何深度表示和受控压入采集思路。
  本实现以实际 mesh 的局部法向射线得到几何参考；没有声称复现其整个算法或真实标签流程。
- [TacSL, arXiv:2408.06506](https://arxiv.org/abs/2408.06506)：作为后续传感器模拟比较项；本轮没有完成基线比较。
- [`docs/pressure_sdf_tactile.md`](../docs/pressure_sdf_tactile.md)：遵循 L0 解析／L1 模型参考／
  L2 离线物理 oracle／L3 真实标定的证据区分。模型输出不能作为自身精度的独立真值。

选择 Isaac Sim/Isaac Lab 作为场景查看入口，是为了沿用项目环境和 USD 资产。
数据模型与场景播放器分离，计划和采集均可在 CPU 上执行；本轮无需迁移到 Gazebo。

## 2. 运行入口

在仓库根目录运行。当前本机已准备解释器：

```bash
export CALIB_PY=/home/liuxinyu/miniconda3/envs/brainco/bin/python

# 只输出规模；不会启动 Isaac、不初始化 GPU、不写数据
"$CALIB_PY" calibration/simulation/run.py --plan-only

# 一个完整回合，输出目录必须不存在；CPU 采集并生成 USD、图和数据
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$CALIB_PY" calibration/simulation/run.py \
  --episodes 1 --output calibration/outputs/simulation/my_first_run

# 检查完整性、时钟、单位、接触、复位及分组划分
"$CALIB_PY" calibration/simulation/verify.py calibration/outputs/simulation/my_first_run

# Isaac 原生查看：默认有限 120 帧，需可用本地 GPU
"$CALIB_PY" calibration/simulation/preview_isaac.py \
  calibration/outputs/simulation/my_first_run --frames 120

# 无窗口原生渲染与网格／动画变换验证
"$CALIB_PY" calibration/simulation/preview_isaac.py \
  calibration/outputs/simulation/my_first_run --headless --capture --frames 30
```

原生验证会写 `isaac_verification.json`，**必须看 `passed`，不能只看进程退出码或“窗口能打开”**。
同一目录拒绝覆盖原生验证结果；要重测则用新采集目录。`scene.usda` 可以直接在 Isaac Sim 中打开，
选择 `/World/Overview` 相机、拖动时间轴。仅回放首个回合；其余回合的逐帧位姿保存在各自 NPZ。

切换真实工具与传感位置：

```bash
"$CALIB_PY" calibration/simulation/run.py --head triangle --episodes 1 \
  --output calibration/outputs/simulation/triangle_new
"$CALIB_PY" calibration/simulation/run.py --head conical --finger middle --segment PIP \
  --episodes 1 --output calibration/outputs/simulation/middle_pip_new
"$CALIB_PY" calibration/simulation/run.py --protocols shear_x shear_y --episodes 1 \
  --output calibration/outputs/simulation/shear_new
```

`--finger` 支持 index/middle/ring/little/thumb，`--segment` 支持 DIP/MCP/PIP。
DIP 使用 rubber 表面，MCP/PIP 使用 touch 表面；某处朝向或可达性不满足条件时应调整独立配置中的
命名关节姿态，不能旋转原始手／支架内部装配来伪装成相同标定姿态。
这些是位置选择入口，不代表已经逐一完成五指、所有压力片的可达性验收。掌部压力片暂未接入。

CPU 依赖 NumPy、SciPy、trimesh、rtree、PyTorch、matplotlib；查看另需既有 Isaac Lab/Sim。
本轮没有安装或升级环境。轻量回归：`"$CALIB_PY" calibration/simulation/checks.py`。

八协议回归配置为 `smoke_config.json`：
`"$CALIB_PY" calibration/simulation/run.py --config calibration/simulation/smoke_config.json --output calibration/outputs/simulation/new_smoke`。
它只运行一个位置、一个深度各一次，含两种虚拟旋转，不是默认270回合正式矩阵。

## 3. 坐标、模型与分辨率

- `world = lower_mount`，支架底面约 Z=0；沿用源 URDF 的 61 link / 60 joint / 21 活动关节。
  关节在指定位置固定，不启动 RL。全部长度 m、角度 rad、时间 s、压力 Pa、力 N；矩阵为列向量
  `p_world = T_world_local @ p_local`，HydroShear 接口四元数为 wxyz。
- 各工具的 `T_flange_tip` 从对应 URDF 读取，法兰安装面不是接触零点。
  球槽底点和贴码槽底原 frame 保存在 `frames.json`，不参与手指接触零位定义。
- `contact_zero()` 用压头下表面与目标触觉表面求首个**离散网格接触**。
  默认滑动时 Z 随曲面接触高度变化；这仍是 XYZ 轨迹，不是增加旋转轴。
  网格有限分辨率会影响接触零点，正式评测须做分辨率收敛试验。
- 法向深度为从触觉面内侧沿外法向到压头下表面的交点距离，正值为侵入；无命中单独保存
  `ray_hit_valid=false`，不能把 `signed_gap_m=0` 的无命中格当成精确接触。
- 压力模型暂为 `p = min(E * (depth / shell_thickness)^gamma, saturation)`，
  `taxel_force_n = p * surface_area_m2`。面积含曲面投影修正；合力按各格法向求和。
  它不是求解柔性体后的压力，不包含粘弹性松弛、独立切向力或真实传感器串扰。
- Marker 调用现有曲面 HydroShear；位移 m 经虚拟正交相机像素间距换算为 px。
  这些位移不是 N，不能把内部 tracker 向量当成六维力传感器读数。
- RGB 为三光源 Lambertian **未标定诊断渲染**，非 Revo3 实测 RGB，也不是 TAXIM 原生输出。
  正式三模态学习应先选择并验证与下游一致的光学渲染器／相机契约；本轮不训练 encoder。
- 默认 `grid_hw=[32,40]`、100 marker 是快速检查配置；射线 UV 来自当前 mesh 的虚拟正交投影，
  与旧 320×240 相机标定不同。即使改为 `[240,320]`，也不自动兼容旧 encoder，禁止拉伸凑维度。
  本轮不复用另一版本手的 marker 三维位置；真机对应关系稍后单独标定。

## 4. 需要采集的数据

每个回合 `episode_XXXXX.npz` + 同名 JSON。时间维为 T，表面 H×W，marker 数 K=100，
静态几何另存 `layout.npz`，无 Python pickle。所有阶段包括空载、卸载和失败帧均保留。

| 数据 | 当前输出／形状 | 用途及注意事项 |
| --- | --- | --- |
| 回合与时钟 | `episode_id/frame_id/time_s/phase`，T | 使用仿真采样时钟；不是现实 SDK 接收时间 |
| 轨迹命令 | `command_xyzrpy`，T×6 | dx/dy/压入量/roll/pitch/yaw；前3项 m，后3项 rad |
| 实际模型位姿 | `T_world_flange/T_world_tip`，T×4×4；`flange_velocity_w_mps` | 记录本模型执行位姿；理想执行不等于编码器实测 |
| 接触零点 | `contact_zero_flange_w_m`，3 | 回合初始偏移处的 flange 零位；随曲面滑动的实际 Z 见逐帧变换 |
| 手状态 | `joint_pos_rad/joint_vel_rad_s`，T×21 | 顺序写入 manifest；未指定关节按 URDF 零位固定 |
| 静态表面 | surface_points/normals/valid、`taxel_area_m2`，H×W | 世界系 m／单位法向／m²；不是硬件 285-taxel 布局 |
| 几何深度 | `depth_m/depth_velocity_mps/signed_gap_m`，T×H×W | L1 几何模型参考，不能宣称测得真实胶体变形 |
| 接触和有效性 | `contact_mask/ray_hit_valid`，T×H×W；`surface_valid` | 保留无接触和缺测的区别 |
| 压力及力 | `pressure_pa/taxel_force_n`，T×H×W；`total_normal_load_n`，T | 明确 Pa 与 N；`total_normal_load_n` 是各格法向负载之和，不等于曲面合力模长 |
| 向量合力、CoP | `force_on_sensor_w_n`，T×3；`center_of_pressure_px`，T×2；`cop_valid` | 力作用于传感器，通常朝内；无负载 CoP 数值为0但 validity=false |
| 饱和 | `pressure_saturated`，T×H×W | 保留模型饱和样本，不能静默删除 |
| Marker | `marker`，T×K×5：[u0,v0,du,dv,valid]；`marker_displacement_w_m`，T×K×3 | 虚拟相机像素／世界米；保持 ID 与跨帧历史，回合之间清空 |
| RGB | `rgb`，T×H×W×3，uint8 | 当前是诊断合成 RGB；与深度／marker 同一采样时刻 |
| 复现信息 | `manifest/config/plan/frames.json` | 有效参数、seed、源代码与资产 SHA256、Git HEAD、模型来源、失败原因 |

**论文后续必须另外获得的独立数据**（当前没有，以 manifest 的 unavailable_channels 明示）：
实测原始 RGB／空载参考图、相机内外参和 marker 追踪置信度、压头编码器实测位姿、独立六维力／
标量力参考、材质和胶层厚度、环境温度、各 SDK 的采样与接收时钟、丢帧和曝光信息。
如需声称稠密压力精度，另需匹配几何和边界条件的 FEM／hydroelastic oracle 或真实压力阵列；
总力计、PhysX 稀疏触点、RGB 重建图都不能独自证明逐 taxel 压力正确。

## 5. 采集矩阵与顺序

以下数值是**本项目仿真起步配置**，不是实物安全限值或论文原文要求。
默认法向压入 0.1/0.3/0.5 mm，中心和 ±X/±Y 1.5 mm 五个位置，每组合重复3次。
六类 XYZ 协议共270回合、55,890帧，30 Hz 仿真时间约31分钟；实际 CPU 耗时另计。
只执行 `--episodes 1` 不代表完成全矩阵。完整默认采集需要去掉该参数，输出到新目录。

| 协议 | 默认动作 | 主要用途 |
| --- | --- | --- |
| 每回合 baseline/release | 接触前后各0.2 s，3 mm离面 | 空载零点、泄漏、复位、marker残留 |
| normal | 接近→分级深度的单次加载→保持0.3 s→卸载→退出 | 深度／接触面积／λd、压力曲线；不同深度分回合 |
| hold | 加载后保持1.5 s，再完整退出 | 时间漂移／松弛辨识的采集入口；当前模型不含松弛，不能据零漂移证明真机稳定 |
| shear_x/y | 接触后 +1 mm→−1 mm→中心 | λs/K、方向性、反向历史 |
| slip_x/y | 接触后 +3 mm→−3 mm→中心 | 更长滑动扫描，供 μ 辨识；协议名称不构成滑移真值，当前未输出已验证 slip label |
| twist/roll | 工具 tip 处虚拟转动±0.15/±0.08 rad | SE(3) 泛化检查；需显式选择，标记 `virtual_rotation`，不是三轴真台能力 |

默认移动速度2 mm/s；滑动曲面跟随会改变实际 Z 速度，需读取保存的 `flange_velocity_w_mps`，
不能把参数中的切向速度当成每个轴的真实速度。最大允许**模型法向深度**0.8 mm、模型负载20 N。
超限、非有限值或抽样发现非目标几何穿入会记录故障并结束该回合，保留触发帧。

建议的正式论文覆盖，**尚未全量执行**：

1. 先完成食指 DIP 球压头的深度和方向扫描、分辨率收敛，再扩展五个 DIP；MCP/PIP 压力片另组采集。
2. 球压头做拟合，三角／锥头做未见形状留出；中心拟合、边缘／更大偏移留出；增加多个速度和预载。
3. 独立保存建模参数、标定参数、验证参数。λd、λs、K、μ 按控制变量逐项辨识；
   压力的 E／胶厚／增益存在耦合，没有独立力／材料参考时不能声称全部辨识成功。
4. `plan.json` 以 protocol×depth×offset 为组，固定 seed 分 train/val/test；相同组合的重复回合不跨组。
   这是组合留出，不自动等于形状／位置完全留出。正式跨形状或跨位置试验需另建配置，锁定名单，
   同一连续轨迹或相邻帧绝不随机拆到训练和测试两侧。

## 6. 验证门槛和注意事项

- `checks.py`：解析球射线距离、源 tip 装配、运动速度／完整阶段、非法深度拒绝、分组确定性。
- `verify.py`：同步尺寸、时钟、有限值、空载归零、压力×面积=力、合力坐标符号、正常加载单调性。
  这些检查是实现一致性，不能证明真实材料准确性，也不评价下游 RL 成功率。
- 原生 USD 验证：网格拓扑可读、逐帧 tool 位姿与 NPZ 一致、RTX 图像可获取。仅场景导入和回放。
- 几何诊断目前每15帧做工具采样点对非目标闭合 mesh 的包含检查。现有食指内部视觉 mesh 有开口，
  写入 `uncertified_open_mesh:right_index_DIP_link`；**全行程净空和连续碰撞尚未验收**。
  本模型没有 PhysX 碰撞求解，不能用受控侵入位移证明刚体动力学接触稳定，也未关闭原训练碰撞配置。
- 三角／锥端很小，粗网格容易漏掉接触或放大离散零位偏差；正式分辨率必须看收敛结果。
- 保留未接触、超限、饱和、丢失 marker、边缘接触和失败回合。不能为了“更好结果”删除困难样本。
- 本系统没有质量／惯量估计、闭环恒力控制、机械台真行程限位、螺钉预紧或打印验收。
  README 中 M8×20 螺钉问题及历史 M6 描述仍待机械核查，本轮未改 CAD。
- 冻结 dataset manifest 后再做论文评测。建议报告：深度 RMSE/接触 IoU/接触起止偏差，marker RMSE
  与非零向量 cosine，独立总力 MAE、CoP 误差，以及逐回合有效数/总数和失败原因。
  对阈值尚未获得实验依据的项目标“待确定”；同一公式产生的压力和总力不能互相充当独立真值。

## 7. 输出边界

本次实测入口：[最终球压头短测](../outputs/simulation/validated_v1/verification.json)、
[Isaac原生检查](../outputs/simulation/validated_v1/isaac_verification.json)、
[原始场景截图（含已移除的虚拟机构）](../outputs/simulation/validated_v1/isaac_scene.png)。八回合1,574帧、99项数据检查通过；
三角／锥头各一回合各115帧。人为降低模型力上限的故障试验在第52帧终止并保留触发帧，退出码2。
完整记录见 [CALIB-SIM-20261003-01](../../EXPERIMENTS.md#calib-sim-20261003-01)。

`outputs/` 是本地忽略目录，原始 V4 与压头资产也尚未 Git 跟踪。换机器需携带完整资产包、
本目录、现有 HydroShear adapter、输出 manifest 和数据，不应仅复制 `scene.usda` 当作完整证据包。
场景 USD 嵌入当前视觉网格便于打开，但原始 URDF、配置与 SHA 清单才说明来源和模型契约。
不修改 `.gitignore`，不自动上传、不提交、不启动全量采集或训练。
